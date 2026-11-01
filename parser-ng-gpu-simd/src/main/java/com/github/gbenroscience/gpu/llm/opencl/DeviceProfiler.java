package com.github.gbenroscience.gpu.llm.opencl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * OpenCL event profiler (enable with {@code -Dllm.profile=true}).
 *
 * <p>It is installed by wrapping the OpenCL entry-point MethodHandles in {@link OpenCLBindings}, so every
 * existing call site is covered without being edited. For each enqueued kernel / transfer it captures a
 * cl_event and, after the queue has been synchronised ({@link #flush()}), reads
 * CL_PROFILING_COMMAND_START/END -- the time the DEVICE actually spent executing. Separately it measures the
 * time the HOST spent inside each call (submission cost, or blocked time for blocking reads / clFinish), and
 * counts launches, synchronisations, clSetKernelArg calls, allocations and bytes moved.
 *
 * <p>Per flush (one per token) it also records the span between the first event's start and the last event's
 * end and the sum of the individual durations; their difference is time the device sat idle waiting for the
 * host -- the signature of a submission-bound pipeline.
 *
 * <p>Profiling enables CL_QUEUE_PROFILING_ENABLE on the queue and captures an event per command, so it adds a
 * little overhead; it does NOT insert extra synchronisation inside the transformer.
 */
final class DeviceProfiler {

    static final boolean ON = Boolean.getBoolean("llm.profile");

    private static final int CL_PROFILING_COMMAND_START = 0x1282;
    private static final int CL_PROFILING_COMMAND_END = 0x1283;

    // counter indices (per phase)
    static final int LAUNCHES = 0, SYNCS = 1, BLOCKING_READS = 2, BLOCKING_WRITES = 3, ALLOCS = 4, ALLOC_BYTES = 5,
            SET_ARGS = 6, H2D_BYTES = 7, D2H_BYTES = 8, D2D_BYTES = 9, LAUNCH_HOST_NS = 10, SET_ARG_HOST_NS = 11,
            FINISH_HOST_NS = 12, TRANSFERS = 13, N_COUNTERS = 14;

    static final class Stat {

        long count, deviceNs, hostNs, bytes, minNs = Long.MAX_VALUE, maxNs;
    }

    private static final class Pending {

        final String phase;
        final int layer;
        final String op;
        final long bytes;
        final long hostNs;
        final MemorySegment event;

        Pending(String phase, int layer, String op, long bytes, long hostNs, MemorySegment event) {
            this.phase = phase;
            this.layer = layer;
            this.op = op;
            this.bytes = bytes;
            this.hostNs = hostNs;
            this.event = event;
        }
    }

    private final MemorySegment slot = Arena.global().allocate(ValueLayout.ADDRESS);
    private final MemorySegment out8 = Arena.global().allocate(ValueLayout.JAVA_LONG);
    private MethodHandle getInfo;
    private MethodHandle releaseEvent;
    private final Map<Long, String> kernelNames = new HashMap<>();
    private final List<Pending> pending = new ArrayList<>();
    private final Map<String, Map<String, Stat>> ops = new TreeMap<>();
    private final Map<String, Map<Integer, long[]>> layers = new TreeMap<>(); // phase -> layer -> {deviceNs, commands}
    private final Map<String, long[]> windows = new TreeMap<>();               // phase -> {spanNs, busyNs, count}
    private final Map<String, long[]> counters = new TreeMap<>();
    private long unprofiled;

    /** Current labels, set by the engine; events are tagged with them at enqueue time. */
    private volatile long kernelBytesHint;

    /** Host code states how many bytes the NEXT kernel launch reads (for an effective-bandwidth column). */
    void hintBytes(long bytes) {
        kernelBytesHint = bytes;
    }

    volatile String phase = "load";
    volatile int layer = -1;

    void setEventApi(MethodHandle getInfo, MethodHandle releaseEvent) {
        this.getInfo = getInfo;
        this.releaseEvent = releaseEvent;
    }

    synchronized void nameKernel(MemorySegment kernel, String name) {
        kernelNames.put(kernel.address(), name);
    }

    private synchronized long[] c(String ph) {
        return counters.computeIfAbsent(ph, k -> new long[N_COUNTERS]);
    }

    synchronized long counter(String ph, int idx) {
        return c(ph)[idx];
    }

    synchronized void count(int idx, long add) {
        c(phase)[idx] += add;
    }

    private synchronized void add(String op, long bytes, long hostNs, MemorySegment event) {
        pending.add(new Pending(phase, layer, op, bytes, hostNs, event));
    }

    // ---------------------------------------------------------------- wrapped entry points

    private static MemorySegment takeEvent(DeviceProfiler p) {
        MemorySegment ev = p.slot.get(ValueLayout.ADDRESS, 0);
        p.slot.set(ValueLayout.ADDRESS, 0, MemorySegment.NULL);
        return ev;
    }

    static int ndRange(MethodHandle t, DeviceProfiler p, MemorySegment q, MemorySegment k, int dim, MemorySegment off,
            MemorySegment g, MemorySegment l, int nw, MemorySegment wl, MemorySegment ev) throws Throwable {
        if (!ev.equals(MemorySegment.NULL)) {
            return (int) t.invokeExact(q, k, dim, off, g, l, nw, wl, ev);
        }
        long t0 = System.nanoTime();
        int rc = (int) t.invokeExact(q, k, dim, off, g, l, nw, wl, p.slot);
        long host = System.nanoTime() - t0;
        p.count(LAUNCHES, 1);
        p.count(LAUNCH_HOST_NS, host);
        if (rc == 0) {
            String name;
            synchronized (p) {
                name = p.kernelNames.getOrDefault(k.address(), "kernel@" + Long.toHexString(k.address()));
            }
            long hint = p.kernelBytesHint;
            p.kernelBytesHint = 0;
            p.add(name, hint, host, takeEvent(p));
        }
        return rc;
    }

    static int write(MethodHandle t, DeviceProfiler p, MemorySegment q, MemorySegment buf, int blocking, long off, long size,
            MemorySegment ptr, int nw, MemorySegment wl, MemorySegment ev) throws Throwable {
        if (!ev.equals(MemorySegment.NULL)) {
            return (int) t.invokeExact(q, buf, blocking, off, size, ptr, nw, wl, ev);
        }
        long t0 = System.nanoTime();
        int rc = (int) t.invokeExact(q, buf, blocking, off, size, ptr, nw, wl, p.slot);
        long host = System.nanoTime() - t0;
        p.count(TRANSFERS, 1);
        p.count(H2D_BYTES, size);
        if (blocking != 0) {
            p.count(BLOCKING_WRITES, 1);
        }
        if (rc == 0) {
            p.add("h2d write", size, host, takeEvent(p));
        }
        return rc;
    }

    static int read(MethodHandle t, DeviceProfiler p, MemorySegment q, MemorySegment buf, int blocking, long off, long size,
            MemorySegment ptr, int nw, MemorySegment wl, MemorySegment ev) throws Throwable {
        if (!ev.equals(MemorySegment.NULL)) {
            return (int) t.invokeExact(q, buf, blocking, off, size, ptr, nw, wl, ev);
        }
        long t0 = System.nanoTime();
        int rc = (int) t.invokeExact(q, buf, blocking, off, size, ptr, nw, wl, p.slot);
        long host = System.nanoTime() - t0; // for a blocking read this is mostly time spent WAITING for earlier work
        p.count(TRANSFERS, 1);
        p.count(D2H_BYTES, size);
        if (blocking != 0) {
            p.count(BLOCKING_READS, 1);
        }
        if (rc == 0) {
            p.add("d2h read", size, host, takeEvent(p));
        }
        return rc;
    }

    static int copy(MethodHandle t, DeviceProfiler p, MemorySegment q, MemorySegment src, MemorySegment dst, long so, long dof,
            long size, int nw, MemorySegment wl, MemorySegment ev) throws Throwable {
        if (!ev.equals(MemorySegment.NULL)) {
            return (int) t.invokeExact(q, src, dst, so, dof, size, nw, wl, ev);
        }
        long t0 = System.nanoTime();
        int rc = (int) t.invokeExact(q, src, dst, so, dof, size, nw, wl, p.slot);
        long host = System.nanoTime() - t0;
        p.count(TRANSFERS, 1);
        p.count(D2D_BYTES, size);
        if (rc == 0) {
            p.add("d2d copy", size, host, takeEvent(p));
        }
        return rc;
    }

    static int finish(MethodHandle t, DeviceProfiler p, MemorySegment q) throws Throwable {
        long t0 = System.nanoTime();
        int rc = (int) t.invokeExact(q);
        p.count(SYNCS, 1);
        p.count(FINISH_HOST_NS, System.nanoTime() - t0);
        return rc;
    }

    static MemorySegment createBuffer(MethodHandle t, DeviceProfiler p, MemorySegment ctx, long flags, long size, MemorySegment host,
            MemorySegment err) throws Throwable {
        MemorySegment r = (MemorySegment) t.invokeExact(ctx, flags, size, host, err);
        p.count(ALLOCS, 1);
        p.count(ALLOC_BYTES, size);
        return r;
    }

    static int setKernelArg(MethodHandle t, DeviceProfiler p, MemorySegment k, int idx, long size, MemorySegment val) throws Throwable {
        long t0 = System.nanoTime();
        int rc = (int) t.invokeExact(k, idx, size, val);
        p.count(SET_ARGS, 1);
        p.count(SET_ARG_HOST_NS, System.nanoTime() - t0);
        return rc;
    }

    private static final MethodHandles.Lookup L = MethodHandles.lookup();

    /** Binds (target, profiler) as the first two arguments of the named static wrapper. */
    static MethodHandle wrap(String method, MethodType type, MethodHandle target, DeviceProfiler p) {
        try {
            MethodHandle h = L.findStatic(DeviceProfiler.class, method, type);
            return MethodHandles.insertArguments(h, 0, target, p);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot wrap OpenCL entry point " + method, e);
        }
    }

    // ---------------------------------------------------------------- event resolution

    /** Reads the device timestamps of every captured event. Call only after the queue has been synchronised. */
    synchronized void flush() throws Throwable {
        Map<String, long[]> win = new HashMap<>(); // phase -> {first, last, busy}
        for (Pending e : pending) {
            long start = query(e.event, CL_PROFILING_COMMAND_START);
            long end = query(e.event, CL_PROFILING_COMMAND_END);
            releaseEvent.invoke(e.event);
            if (start < 0 || end < 0 || end < start) {
                unprofiled++;
                continue;
            }
            long dur = end - start;
            Stat s = ops.computeIfAbsent(e.phase, k -> new TreeMap<>()).computeIfAbsent(e.op, k -> new Stat());
            s.count++;
            s.deviceNs += dur;
            s.hostNs += e.hostNs;
            s.bytes += e.bytes;
            s.minNs = Math.min(s.minNs, dur);
            s.maxNs = Math.max(s.maxNs, dur);
            long[] l = layers.computeIfAbsent(e.phase, k -> new TreeMap<>()).computeIfAbsent(e.layer, k -> new long[2]);
            l[0] += dur;
            l[1]++;
            long[] w = win.computeIfAbsent(e.phase, k -> new long[]{Long.MAX_VALUE, 0, 0});
            w[0] = Math.min(w[0], start);
            w[1] = Math.max(w[1], end);
            w[2] += dur;
        }
        pending.clear();
        for (Map.Entry<String, long[]> e : win.entrySet()) {
            long[] w = windows.computeIfAbsent(e.getKey(), k -> new long[3]);
            w[0] += e.getValue()[1] - e.getValue()[0];
            w[1] += e.getValue()[2];
            w[2]++;
        }
    }

    private long query(MemorySegment event, int param) throws Throwable {
        int rc = (int) getInfo.invoke(event, param, 8L, out8, MemorySegment.NULL);
        return rc == 0 ? out8.get(ValueLayout.JAVA_LONG, 0) : -1;
    }

    /** Forgets everything measured so far (use after a warm-up run). Pending events must already have been flushed. */
    synchronized void resetStats() {
        ops.clear();
        layers.clear();
        windows.clear();
        counters.clear();
        pending.clear();
        unprofiled = 0;
    }

    // ---------------------------------------------------------------- report

    synchronized String report(Map<String, Integer> tokensPerPhase) {
        StringBuilder sb = new StringBuilder("\n==================== OpenCL event profile ====================\n");
        sb.append("device times come from CL_PROFILING_COMMAND_START/END; host times are time spent inside the API call\n");
        sb.append("(submission cost, or blocked-waiting for blocking reads / clFinish).\n");
        for (String ph : ops.keySet()) {
            long busy = 0;
            for (Stat s : ops.get(ph).values()) {
                busy += s.deviceNs;
            }
            long[] w = windows.getOrDefault(ph, new long[3]);
            long[] ct = c(ph);
            int tokens = tokensPerPhase.getOrDefault(ph, 0);
            sb.append(String.format("%n---- phase: %s ----%n", ph));
            sb.append(String.format("device busy %.1f ms; first-event-to-last-event span %.1f ms over %d flush windows -> device idle inside windows %.1f ms (%.1f%%)%n",
                    busy / 1e6, w[0] / 1e6, w[2], (w[0] - w[1]) / 1e6, w[0] == 0 ? 0.0 : 100.0 * (w[0] - w[1]) / w[0]));
            sb.append(String.format("%-34s %8s %12s %10s %12s %12s %9s%n", "operation", "count", "device ms", "avg us", "host ms", "MB touched", "GB/s"));
            List<Map.Entry<String, Stat>> rows = new ArrayList<>(ops.get(ph).entrySet());
            rows.sort((a, b) -> Long.compare(b.getValue().deviceNs, a.getValue().deviceNs));
            for (Map.Entry<String, Stat> r : rows) {
                Stat s = r.getValue();
                sb.append(String.format("%-34s %8d %12.2f %10.1f %12.2f %12.2f %9s%n", r.getKey(), s.count, s.deviceNs / 1e6,
                        s.deviceNs / 1e3 / Math.max(s.count, 1), s.hostNs / 1e6, s.bytes / 1e6,
                        s.bytes > 0 && s.deviceNs > 0 ? String.format("%.1f", s.bytes / (double) s.deviceNs) : "-"));
            }
            sb.append("device ms per transformer layer (-1 = embedding / final norm / lm head):\n  ");
            for (Map.Entry<Integer, long[]> l : layers.getOrDefault(ph, new TreeMap<>()).entrySet()) {
                sb.append(String.format("L%d=%.1f ", l.getKey(), l.getValue()[0] / 1e6));
            }
            sb.append(String.format("%nkernel launches %d, clFinish %d (host blocked %.1f ms), blocking reads %d, blocking writes %d, transfers %d%n",
                    ct[LAUNCHES], ct[SYNCS], ct[FINISH_HOST_NS] / 1e6, ct[BLOCKING_READS], ct[BLOCKING_WRITES], ct[TRANSFERS]));
            sb.append(String.format("clSetKernelArg calls %d (host %.1f ms), launch host time %.1f ms, buffer allocations %d (%.1f MB)%n",
                    ct[SET_ARGS], ct[SET_ARG_HOST_NS] / 1e6, ct[LAUNCH_HOST_NS] / 1e6, ct[ALLOCS], ct[ALLOC_BYTES] / 1e6));
            sb.append(String.format("bytes moved: host->device %.2f MB, device->host %.2f MB, device->device %.2f MB%n",
                    ct[H2D_BYTES] / 1e6, ct[D2H_BYTES] / 1e6, ct[D2D_BYTES] / 1e6));
            if (tokens > 0) {
                sb.append(String.format("per forward pass (%d): %.1f kernel launches, %.1f clFinish, %.1f blocking reads, %.1f blocking writes, %.1f clSetKernelArg, device busy %.1f ms%n",
                        tokens, ct[LAUNCHES] / (double) tokens, ct[SYNCS] / (double) tokens, ct[BLOCKING_READS] / (double) tokens,
                        ct[BLOCKING_WRITES] / (double) tokens, ct[SET_ARGS] / (double) tokens, busy / 1e6 / tokens));
            }
        }
        if (unprofiled > 0) {
            sb.append(String.format("%n%d commands had no usable profiling timestamps%n", unprofiled));
        }
        return sb.toString();
    }
}
