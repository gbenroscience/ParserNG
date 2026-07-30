package com.github.gbenroscience.simdext.turbo.tools.utils;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Locale;

/**
 * Mechanics: How This Works Under the Hood The Linux Path (sched_setaffinity):
 * Passing a thread ID (pid) of 0 tells the Linux kernel to apply the affinity
 * change directly to the calling thread. Because Linux treats threads as
 * Light-Weight Processes (LWPs), this targets your specific worker pipeline
 * thread without shifting any neighboring processing streams. The
 * Arena.ofConfined() block sets up a thread-local allocation to construct the
 * cpu_set_t bitmask layout on the native stack, avoiding any persistent memory
 * footprints or GC overhead.
 *
 * The Windows Path (SetThreadAffinityMask): Calling GetCurrentThread() returns
 * a pseudo-handle representing the active thread execution context. Passing
 * this handle alongside a bitmask (where bit 0 represents Core 0, bit 1
 * represents Core 1, etc.) forces the Windows thread scheduler to immutably
 * lock that thread down to the selected hardware thread execution block.
 *
 * Threading Safety: The downcall method handles are fully thread-safe. Multiple
 * thread workers can concurrently call ThreadAffinity.pinCurrentThread(id)
 * during their initial run loops without crashing or cross-contending.
 *
 * Integrating with Your Workers When spinning up your execution pipeline, pass
 * a unique sequence identifier to each worker thread so it knows exactly which
 * logical core it owns:
 *
 * @author GBEMIRO
 */
public final class CPUPinner {

    private static final MethodHandle SetThreadAffinityMaskHandle;
    private static final MethodHandle GetCurrentThreadHandle;
    private static final MethodHandle SchedSetAffinityHandle;

    static {
        MethodHandle setMask = null;
        MethodHandle getThread = null;
        MethodHandle schedSet = null;

        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        Linker linker = Linker.nativeLinker();

        try {
            if (os.contains("win")) {
                // Windows Kernel32 Binding
                SymbolLookup kernel32 = SymbolLookup.libraryLookup("kernel32", Arena.global());

                getThread = linker.downcallHandle(
                        kernel32.find("GetCurrentThread").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS)
                );

                setMask = linker.downcallHandle(
                        kernel32.find("SetThreadAffinityMask").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
                );
            } else if (os.contains("nix") || os.contains("nux")) {
                // Linux glibc Binding
                SymbolLookup libc = linker.defaultLookup();

                schedSet = linker.downcallHandle(
                        libc.find("sched_setaffinity").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
                );
            }
        } catch (Throwable t) {
            // Fall back gracefully if native access is blocked or symbols missing
            System.err.println("[Affinity] Failed to initialize native hooks: " + t.getMessage());
        }

        GetCurrentThreadHandle = getThread;
        SetThreadAffinityMaskHandle = setMask;
        SchedSetAffinityHandle = schedSet;
    }

    /**
     * Binds the current calling thread tightly to the specified core index.
     *
     * @param coreIndex The logical core index (0-indexed)
     * @return true if the OS accepted the affinity adjustment, false otherwise.
     */
    public static boolean pinCurrentThread(int coreIndex) {
        if (coreIndex < 0) {
            return false;
        }

        // --- Windows Execution Path ---
        if (SetThreadAffinityMaskHandle != null && GetCurrentThreadHandle != null) {
            try {
                MemorySegment threadHandle = (MemorySegment) GetCurrentThreadHandle.invokeExact();
                long mask = 1L << coreIndex; // Supports up to 64 logical cores in the current group
                long previousMask = (long) SetThreadAffinityMaskHandle.invokeExact(threadHandle, mask);
                return previousMask != 0;
            } catch (Throwable t) {
                return false;
            }
        }

        // --- Linux Execution Path ---
        if (SchedSetAffinityHandle != null) {
            // glibc cpu_set_t is typically a 128-byte bitmask (1024 bits)
            final long CPU_SET_SIZE_BYTES = 128;

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment maskSegment = arena.allocate(CPU_SET_SIZE_BYTES); // Allocates zeroed native memory

                // Locate the target byte and target bit within that byte
                int byteOffset = coreIndex / 8;
                int bitOffset = coreIndex % 8;

                if (byteOffset < CPU_SET_SIZE_BYTES) {
                    byte currentByte = maskSegment.get(ValueLayout.JAVA_BYTE, byteOffset);
                    maskSegment.set(ValueLayout.JAVA_BYTE, byteOffset, (byte) (currentByte | (1 << bitOffset)));

                    // pid = 0 instructs Linux to configure the current calling thread specifically
                    int result = (int) SchedSetAffinityHandle.invokeExact(0, CPU_SET_SIZE_BYTES, maskSegment);
                    return result == 0;
                }
            } catch (Throwable t) {
                return false;
            }
        }

        return false;
    }
}
