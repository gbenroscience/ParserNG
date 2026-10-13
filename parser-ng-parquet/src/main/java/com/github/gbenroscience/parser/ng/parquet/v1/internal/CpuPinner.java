package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Best-effort thread-to-CPU pinning through the foreign-function API (Linux {@code sched_setaffinity},
 * Windows {@code SetThreadAffinityMask}), plus discovery of the machine's physical-core topology so that
 * workers can be spread over distinct physical cores rather than SMT siblings.
 *
 * <p>Nothing here is required for correctness and every failure is soft: if native access is blocked, a symbol
 * is missing, the OS refuses the request, or the platform is unsupported (macOS), {@link #pinCurrentThread}
 * returns {@code false} and the thread simply stays unpinned. The class is only initialised when pinning is
 * actually requested ({@code ParquetScan.pinWorkerThreads(true)}), so the default scan never touches it. Why
 * initialisation failed, if it did, is available from {@link #initFailure()}; nothing is printed.
 *
 * <h2>Behaviour worth knowing before relying on it</h2>
 * <ul>
 *   <li>Pinning lasts for the life of the thread and is per thread (Linux pid 0 = the calling thread; Windows
 *       current-thread pseudo-handle). Only pin threads you own and that end with the work.</li>
 *   <li>{@link #pinCurrentThread} takes a LOGICAL processor index. On SMT machines use {@link #pinOrder()},
 *       which lists one logical CPU per physical core first and only then the siblings.</li>
 *   <li>Linux: only CPUs the process is currently allowed to use (its cpuset/affinity mask) are considered, so
 *       a container limited to CPUs 4-7 is not offered CPUs 0-3. Windows: only processor group 0 (at most 64
 *       logical CPUs) is handled; a CPU the process may not use is refused by the OS and reported as
 *       {@code false}.</li>
 *   <li>Pinning removes the scheduler's freedom to move a thread off a busy core. It can therefore make things
 *       slower when other work (the consumer thread, other scans, GC, other processes) shares the chosen
 *       cores. It is an experiment to measure, not a default.</li>
 *   <li>Needs {@code --enable-native-access=ALL-UNNAMED} (or the module name) on the JVM command line;
 *       without it the JDK warns on first use and, in future releases, refuses.</li>
 * </ul>
 *
 * @author GBEMIRO
 */
public final class CpuPinner {

    private CpuPinner() {}

    private static final MethodHandle SET_THREAD_AFFINITY_MASK;
    private static final MethodHandle GET_CURRENT_THREAD;
    private static final MethodHandle GET_LOGICAL_PROCESSOR_INFORMATION_EX;
    private static final MethodHandle SCHED_SET_AFFINITY;
    private static final MethodHandle SCHED_GET_AFFINITY;
    private static final String INIT_FAILURE;

    /** glibc cpu_set_t: 1024 bits. */
    private static final int CPU_SET_BYTES = 128;

    static {
        MethodHandle setMask = null, getThread = null, getLpiEx = null, schedSet = null, schedGet = null;
        String failure = null;
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        try {
            Linker linker = Linker.nativeLinker();
            if (os.contains("win")) {
                SymbolLookup kernel32 = SymbolLookup.libraryLookup("kernel32", Arena.global());
                getThread = linker.downcallHandle(kernel32.find("GetCurrentThread").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.ADDRESS));
                setMask = linker.downcallHandle(kernel32.find("SetThreadAffinityMask").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
                // BOOL GetLogicalProcessorInformationEx(LOGICAL_PROCESSOR_RELATIONSHIP, PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD)
                getLpiEx = linker.downcallHandle(kernel32.find("GetLogicalProcessorInformationEx").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            } else if (os.contains("nix") || os.contains("nux")) {
                SymbolLookup libc = linker.defaultLookup();
                // int sched_setaffinity / sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask)
                FunctionDescriptor fd = FunctionDescriptor.of(ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS);
                schedSet = linker.downcallHandle(libc.find("sched_setaffinity").orElseThrow(), fd);
                schedGet = linker.downcallHandle(libc.find("sched_getaffinity").orElseThrow(), fd);
            } else {
                failure = "unsupported platform: " + os;
            }
        } catch (Throwable t) {
            failure = t.toString();
            setMask = null; getThread = null; getLpiEx = null; schedSet = null; schedGet = null;
        }
        SET_THREAD_AFFINITY_MASK = setMask;
        GET_CURRENT_THREAD = getThread;
        GET_LOGICAL_PROCESSOR_INFORMATION_EX = getLpiEx;
        SCHED_SET_AFFINITY = schedSet;
        SCHED_GET_AFFINITY = schedGet;
        INIT_FAILURE = failure;
    }

    /** True if this platform's pinning entry point could be bound (it may still be refused per call). */
    public static boolean isSupported() {
        return (SET_THREAD_AFFINITY_MASK != null && GET_CURRENT_THREAD != null) || SCHED_SET_AFFINITY != null;
    }

    /** Why native hooks could not be initialised, or {@code null} if they were (or the failure was only "no hook for this OS"). */
    public static String initFailure() { return INIT_FAILURE; }

    // ------------------------------------------------------------------ pinning

    /**
     * Binds the calling thread to one LOGICAL CPU.
     *
     * @return true if the OS accepted the request; false for any refusal, unsupported platform, or an index
     *         outside what the platform call can express (Windows: 0-63 in group 0; Linux: 0-1023)
     */
    public static boolean pinCurrentThread(int coreIndex) {
        if (coreIndex < 0) return false;

        if (SET_THREAD_AFFINITY_MASK != null && GET_CURRENT_THREAD != null) {
            if (coreIndex >= 64) return false; // a shift by >= 64 would silently wrap onto a different CPU
            try {
                MemorySegment thread = (MemorySegment) GET_CURRENT_THREAD.invokeExact();
                long previous = (long) SET_THREAD_AFFINITY_MASK.invokeExact(thread, 1L << coreIndex);
                return previous != 0;
            } catch (Throwable t) {
                return false;
            }
        }

        if (SCHED_SET_AFFINITY != null) {
            if (coreIndex >= CPU_SET_BYTES * 8) return false;
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment mask = arena.allocate(CPU_SET_BYTES); // zeroed
                int byteOffset = coreIndex / 8;
                mask.set(ValueLayout.JAVA_BYTE, byteOffset, (byte) (1 << (coreIndex % 8)));
                // pid 0 = the calling thread
                int rc = (int) SCHED_SET_AFFINITY.invokeExact(0, (long) CPU_SET_BYTES, mask);
                return rc == 0;
            } catch (Throwable t) {
                return false;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ topology

    /**
     * For each distinct PHYSICAL core, in a stable order, the logical CPU indices that are SMT siblings of it,
     * read from the OS (never assumed to be simple interleaving). Falls back to one logical CPU per "core"
     * when the real topology cannot be determined, which is exactly the naive behaviour and never worse.
     */
    public static int[][] detectPhysicalCoreGroups() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        try {
            int[][] groups = null;
            if (os.contains("win") && GET_LOGICAL_PROCESSOR_INFORMATION_EX != null) {
                groups = detectWindowsCoreGroups();
            } else if (os.contains("nix") || os.contains("nux")) {
                groups = detectLinuxCoreGroups();
            }
            if (groups != null && groups.length > 0) return groups;
        } catch (Throwable t) {
            // fall through to the naive grouping
        }
        return naiveGroups();
    }

    /**
     * The order in which to hand out CPUs to workers: the first logical CPU of every physical core, then every
     * core's second sibling, and so on, so N workers land on N distinct physical cores whenever N does not
     * exceed the core count, and only then share cores. Never empty.
     */
    public static int[] pinOrder() {
        return interleave(detectPhysicalCoreGroups());
    }

    /** Pure ordering rule behind {@link #pinOrder()}: round-robin across groups by sibling depth. */
    static int[] interleave(int[][] groups) {
        int total = 0, deepest = 0;
        for (int[] g : groups) { total += g.length; deepest = Math.max(deepest, g.length); }
        int[] order = new int[total];
        int n = 0;
        for (int depth = 0; depth < deepest; depth++) {
            for (int[] g : groups) if (depth < g.length) order[n++] = g[depth];
        }
        return order;
    }

    private static int[][] naiveGroups() {
        int[] allowed = linuxAllowedCpus();
        if (allowed == null) {
            int n = Runtime.getRuntime().availableProcessors();
            allowed = new int[n];
            for (int i = 0; i < n; i++) allowed[i] = i;
        }
        int[][] out = new int[allowed.length][];
        for (int i = 0; i < allowed.length; i++) out[i] = new int[] {allowed[i]};
        return out;
    }

    /**
     * Logical CPUs this process may run on, from {@code sched_getaffinity}; null if unavailable (not Linux, or
     * the call failed). This is what makes container cpusets work: their CPU numbers need not start at 0.
     */
    private static int[] linuxAllowedCpus() {
        if (SCHED_GET_AFFINITY == null) return null;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment mask = arena.allocate(CPU_SET_BYTES);
            int rc = (int) SCHED_GET_AFFINITY.invokeExact(0, (long) CPU_SET_BYTES, mask);
            if (rc != 0) return null;
            List<Integer> cpus = new ArrayList<>();
            for (int b = 0; b < CPU_SET_BYTES; b++) {
                int v = mask.get(ValueLayout.JAVA_BYTE, b) & 0xFF;
                for (int bit = 0; bit < 8; bit++) if ((v & (1 << bit)) != 0) cpus.add(b * 8 + bit);
            }
            if (cpus.isEmpty()) return null;
            int[] out = new int[cpus.size()];
            for (int i = 0; i < out.length; i++) out[i] = cpus.get(i);
            return out;
        } catch (Throwable t) {
            return null;
        }
    }

    /** Groups the allowed logical CPUs by (physical_package_id, core_id) from sysfs, so sockets are told apart. */
    private static int[][] detectLinuxCoreGroups() throws IOException {
        int[] allowed = linuxAllowedCpus();
        if (allowed == null) {
            int n = Runtime.getRuntime().availableProcessors();
            allowed = new int[n];
            for (int i = 0; i < n; i++) allowed[i] = i;
        }
        Map<String, List<Integer>> byCore = new LinkedHashMap<>();
        for (int cpu : allowed) {
            Path coreId = Path.of("/sys/devices/system/cpu/cpu" + cpu + "/topology/core_id");
            Path pkgId = Path.of("/sys/devices/system/cpu/cpu" + cpu + "/topology/physical_package_id");
            if (!Files.isReadable(coreId)) return null; // topology not exposed here
            String pkg = Files.isReadable(pkgId) ? Files.readString(pkgId).trim() : "0";
            byCore.computeIfAbsent(pkg + ":" + Files.readString(coreId).trim(), k -> new ArrayList<>()).add(cpu);
        }
        int[][] result = new int[byCore.size()][];
        int i = 0;
        for (List<Integer> siblings : byCore.values()) {
            int[] arr = new int[siblings.size()];
            for (int k = 0; k < arr.length; k++) arr[k] = siblings.get(k);
            result[i++] = arr;
        }
        return result;
    }

    /**
     * Parses GetLogicalProcessorInformationEx(RelationProcessorCore). Layout (x64):
     * <pre>
     * SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX { DWORD Relationship @0; DWORD Size @4;
     *   PROCESSOR_RELATIONSHIP @8 { BYTE Flags @8; BYTE EfficiencyClass @9; BYTE Reserved[20] @10; WORD GroupCount @30;
     *                               GROUP_AFFINITY GroupMask[] @32 }  }
     * GROUP_AFFINITY (16 bytes) { KAFFINITY Mask @0 (8); WORD Group @8; WORD Reserved[3] }
     * </pre>
     */
    private static int[][] detectWindowsCoreGroups() throws Throwable {
        final int relationProcessorCore = 0;
        final long bufferSize = 65536;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buffer = arena.allocate(bufferSize, 8);
            MemorySegment lengthSeg = arena.allocate(ValueLayout.JAVA_INT);
            lengthSeg.set(ValueLayout.JAVA_INT, 0, (int) bufferSize);
            int ok = (int) GET_LOGICAL_PROCESSOR_INFORMATION_EX.invokeExact(relationProcessorCore, buffer, lengthSeg);
            int returned = lengthSeg.get(ValueLayout.JAVA_INT, 0);
            if (ok == 0 || returned <= 0 || returned > bufferSize) return null;

            List<int[]> cores = new ArrayList<>();
            long offset = 0;
            while (offset + 8 <= returned) {
                int relationship = buffer.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
                int size = buffer.get(ValueLayout.JAVA_INT_UNALIGNED, offset + 4);
                if (size <= 0) break; // malformed: never loop forever
                if (relationship == relationProcessorCore) {
                    int groupCount = Short.toUnsignedInt(buffer.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset + 30));
                    List<Integer> siblings = new ArrayList<>();
                    for (int g = 0; g < groupCount; g++) {
                        long entry = offset + 32 + (long) g * 16;
                        if (entry + 16 > offset + size) break;
                        long mask = buffer.get(ValueLayout.JAVA_LONG_UNALIGNED, entry);
                        short group = buffer.get(ValueLayout.JAVA_SHORT_UNALIGNED, entry + 8);
                        if (group == 0) { // pinCurrentThread only addresses group 0
                            for (int bit = 0; bit < 64; bit++) if ((mask & (1L << bit)) != 0) siblings.add(bit);
                        }
                    }
                    if (!siblings.isEmpty()) {
                        int[] arr = new int[siblings.size()];
                        for (int k = 0; k < arr.length; k++) arr[k] = siblings.get(k);
                        cores.add(arr);
                    }
                }
                offset += size;
            }
            return cores.isEmpty() ? null : cores.toArray(new int[0][]);
        }
    }
    public static void main(String[] args) {
        System.out.println(java.util.Arrays.deepToString(CpuPinner.detectPhysicalCoreGroups()));
System.out.println(java.util.Arrays.toString(CpuPinner.pinOrder()));
    }
}
