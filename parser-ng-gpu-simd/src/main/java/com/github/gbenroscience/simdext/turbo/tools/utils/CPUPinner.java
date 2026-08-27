package com.github.gbenroscience.simdext.turbo.tools.utils;

import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

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
 * Topology Awareness: {@code pinCurrentThread(coreIndex)} still takes a raw
 * LOGICAL processor index — it has no concept of physical cores or SMT
 * siblings. On hyperthreaded/SMT CPUs, naively pinning worker N to logical
 * index N can put several workers on the same physical core (its SMT
 * siblings) while other physical cores sit idle. {@link #detectPhysicalCoreGroups()}
 * solves that: it queries the OS for the actual physical-core -> logical-CPU
 * grouping so callers can pick ONE logical index per distinct physical core.
 *
 * @author GBEMIRO
 */
public final class CPUPinner {

    private static final MethodHandle SetThreadAffinityMaskHandle;
    private static final MethodHandle GetCurrentThreadHandle;
    private static final MethodHandle SchedSetAffinityHandle;
    private static final MethodHandle GetLogicalProcessorInformationExHandle;

    static {
        MethodHandle setMask = null;
        MethodHandle getThread = null;
        MethodHandle schedSet = null;
        MethodHandle getLpiEx = null;

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

                // BOOL GetLogicalProcessorInformationEx(
                //     LOGICAL_PROCESSOR_RELATIONSHIP RelationshipType,
                //     PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX Buffer,
                //     PDWORD ReturnedLength);
                getLpiEx = linker.downcallHandle(
                        kernel32.find("GetLogicalProcessorInformationEx").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
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
        GetLogicalProcessorInformationExHandle = getLpiEx;
    }

    /**
     * Binds the current calling thread tightly to the specified LOGICAL core
     * index. This has no awareness of SMT/physical-core topology by itself —
     * see {@link #detectPhysicalCoreGroups()} to pick logical indices that
     * land on distinct physical cores.
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

    /**
     * Returns, for each distinct PHYSICAL core (in a stable order), the set
     * of logical processor indices that are SMT/hyperthread siblings of that
     * core. E.g. on a 2-physical-core / 4-logical-thread CPU this typically
     * returns {@code [[0,1], [2,3]]} — but the exact logical-index grouping
     * is read from the OS rather than assumed, since it is NOT guaranteed to
     * always be simple interleaving.
     * <p>
     * Callers that want to pin N worker threads to N distinct physical cores
     * should use {@code group[i % group.length][0]} as the pin target for
     * worker {@code i}, rather than the raw logical index {@code i}.
     * <p>
     * Falls back to one logical processor per "core" (i.e. no grouping, same
     * behavior as before this method existed) if the real topology can't be
     * determined — this keeps behavior no worse than the naive approach, never
     * worse.
     */
    public static int[][] detectPhysicalCoreGroups() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        try {
            if (os.contains("win") && GetLogicalProcessorInformationExHandle != null) {
                int[][] groups = detectWindowsCoreGroups();
                if (groups != null && groups.length > 0) {
                    return groups;
                }
            } else if (os.contains("nix") || os.contains("nux")) {
                int[][] groups = detectLinuxCoreGroups();
                if (groups != null && groups.length > 0) {
                    return groups;
                }
            }
        } catch (Throwable t) {
            System.err.println("[Affinity] Physical core topology detection failed, "
                    + "falling back to 1 logical CPU per group: " + t.getMessage());
        }
        return naiveFallbackGroups();
    }

    private static int[][] naiveFallbackGroups() {
        int n = Runtime.getRuntime().availableProcessors();
        int[][] fallback = new int[n][];
        for (int i = 0; i < n; i++) {
            fallback[i] = new int[]{i};
        }
        return fallback;
    }

    /**
     * Parses the result of GetLogicalProcessorInformationEx(RelationProcessorCore, ...).
     * Layout reference (x64, default struct packing):
     * <pre>
     * SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX {
     *     DWORD Relationship;      // offset 0
     *     DWORD Size;              // offset 4
     *     // union starts at offset 8; for RelationProcessorCore (0) it's PROCESSOR_RELATIONSHIP:
     *     BYTE  Flags;             // offset 8
     *     BYTE  EfficiencyClass;   // offset 9
     *     BYTE  Reserved[20];      // offset 10..29
     *     WORD  GroupCount;        // offset 30..31
     *     GROUP_AFFINITY GroupMask[GroupCount]; // offset 32.. (8-byte aligned)
     * }
     * GROUP_AFFINITY {
     *     KAFFINITY Mask; // ULONG_PTR, offset 0, 8 bytes
     *     WORD Group;     // offset 8
     *     WORD Reserved[3]; // offset 10..15
     * } // 16 bytes total
     * </pre>
     */
    private static int[][] detectWindowsCoreGroups() throws Throwable {
        final int RELATION_PROCESSOR_CORE = 0;
        final long BUFFER_SIZE = 65536; // generous; RelationProcessorCore records are small

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(BUFFER_SIZE);
            MemorySegment lengthSeg = arena.allocate(ValueLayout.JAVA_INT);
            lengthSeg.set(ValueLayout.JAVA_INT, 0, (int) BUFFER_SIZE);

            int ok = (int) GetLogicalProcessorInformationExHandle.invokeExact(
                    RELATION_PROCESSOR_CORE, buffer, lengthSeg);

            int returnedLength = lengthSeg.get(ValueLayout.JAVA_INT, 0);
            if (ok == 0 || returnedLength <= 0 || returnedLength > BUFFER_SIZE) {
                return null; // let caller fall back
            }

            List<int[]> cores = new ArrayList<>();
            long offset = 0;
            while (offset + 8 <= returnedLength) {
                int relationship = buffer.get(ValueLayout.JAVA_INT, offset);
                int size = buffer.get(ValueLayout.JAVA_INT, offset + 4);
                if (size <= 0) {
                    break; // malformed / safety guard against infinite loop
                }

                if (relationship == RELATION_PROCESSOR_CORE) {
                    int groupCount = Short.toUnsignedInt(buffer.get(ValueLayout.JAVA_SHORT, offset + 8 + 22));
                    long groupMaskBase = offset + 8 + 24;
                    List<Integer> siblings = new ArrayList<>();

                    for (int g = 0; g < groupCount; g++) {
                        long entryOffset = groupMaskBase + (long) g * 16;
                        if (entryOffset + 16 > offset + size) {
                            break; // guard against reading past this record
                        }
                        long mask = buffer.get(ValueLayout.JAVA_LONG, entryOffset);
                        short group = buffer.get(ValueLayout.JAVA_SHORT, entryOffset + 8);
                        if (group == 0) { // single-group system is overwhelmingly the common case
                            for (int bit = 0; bit < 64; bit++) {
                                if ((mask & (1L << bit)) != 0) {
                                    siblings.add(bit);
                                }
                            }
                        }
                    }

                    if (!siblings.isEmpty()) {
                        int[] arr = new int[siblings.size()];
                        for (int k = 0; k < arr.length; k++) {
                            arr[k] = siblings.get(k);
                        }
                        cores.add(arr);
                    }
                }

                offset += size;
            }

            return cores.isEmpty() ? null : cores.toArray(new int[0][]);
        }
    }

    /**
     * Groups logical CPUs by (physical_package_id, core_id) read from sysfs,
     * so cores are distinguished correctly across multiple sockets too.
     */
    private static int[][] detectLinuxCoreGroups() throws IOException {
        int n = Runtime.getRuntime().availableProcessors();
        Map<String, List<Integer>> byCoreKey = new LinkedHashMap<>();

        for (int cpu = 0; cpu < n; cpu++) {
            Path coreIdPath = Path.of("/sys/devices/system/cpu/cpu" + cpu + "/topology/core_id");
            Path pkgIdPath = Path.of("/sys/devices/system/cpu/cpu" + cpu + "/topology/physical_package_id");

            if (!Files.isReadable(coreIdPath)) {
                return null; // topology not exposed; let caller fall back
            }

            String coreId = Files.readString(coreIdPath).trim();
            String pkgId = Files.isReadable(pkgIdPath) ? Files.readString(pkgIdPath).trim() : "0";
            String key = pkgId + ":" + coreId;

            byCoreKey.computeIfAbsent(key, k -> new ArrayList<>()).add(cpu);
        }

        int[][] result = new int[byCoreKey.size()][];
        int i = 0;
        for (List<Integer> siblings : byCoreKey.values()) {
            int[] arr = new int[siblings.size()];
            for (int k = 0; k < arr.length; k++) {
                arr[k] = siblings.get(k);
            }
            result[i++] = arr;
        }
        return result;
    }
}