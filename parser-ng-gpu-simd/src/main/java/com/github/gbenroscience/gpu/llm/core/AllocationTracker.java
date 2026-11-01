package com.github.gbenroscience.gpu.llm.core;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Registry of live device allocations owned by one GPU context. Every backend
 * registers each buffer it creates and only releases a buffer if
 * {@link #remove} succeeds, which gives three guarantees the raw APIs do not:
 * a double free is a harmless no-op, a buffer whose owning object failed to
 * construct (and therefore can never be closed) is still released when the
 * context is closed, and tests can assert that nothing leaked. It also tracks
 * live and peak bytes, which is what a device-memory report is built from.
 * The key is the backend's native handle value (cl_mem address, CUdeviceptr,
 * MTLBuffer id).
 */
public final class AllocationTracker {

    private static final class Rec {

        final long bytes;
        final String category;

        Rec(long bytes, String category) {
            this.bytes = bytes;
            this.category = category;
        }
    }

    private final Map<Long, Rec> live = new ConcurrentHashMap<>();
    private final AtomicLong liveBytes = new AtomicLong();
    private final AtomicLong peakBytes = new AtomicLong();

    public void add(long key) {
        add(key, 0L);
    }

    public void add(long key, long bytes) {
        add(key, bytes, "uncategorised");
    }

    public void add(long key, long bytes, String category) {
        if (key == 0L) {
            return;
        }
        if (live.putIfAbsent(key, new Rec(bytes, category)) != null) {
            throw new IllegalStateException("device handle 0x" + Long.toHexString(key) + " registered twice");
        }
        long now = liveBytes.addAndGet(bytes);
        peakBytes.accumulateAndGet(now, Math::max);
    }

    /** @return true if the handle was live (the caller must now release it). */
    public boolean remove(long key) {
        if (key == 0L) {
            return false;
        }
        Rec r = live.remove(key);
        if (r == null) {
            return false;
        }
        liveBytes.addAndGet(-r.bytes);
        return true;
    }

    public int size() {
        return live.size();
    }

    public long liveBytes() {
        return liveBytes.get();
    }

    public long peakBytes() {
        return peakBytes.get();
    }

    /** Live bytes and allocation count per category, sorted by name: {bytes, count}. */
    public java.util.SortedMap<String, long[]> byCategory() {
        java.util.TreeMap<String, long[]> m = new java.util.TreeMap<>();
        for (Rec r : live.values()) {
            long[] a = m.computeIfAbsent(r.category, k -> new long[2]);
            a[0] += r.bytes;
            a[1]++;
        }
        return m;
    }

    /** Removes and returns every live handle; the caller releases them. */
    public long[] drain() {
        long[] out = live.keySet().stream().mapToLong(Long::longValue).toArray();
        for (long k : out) {
            remove(k);
        }
        return out;
    }
}
