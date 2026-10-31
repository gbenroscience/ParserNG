package com.github.gbenroscience.gpu.llm.core;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of live device allocations owned by one GPU context. Every backend
 * registers each buffer it creates and only releases a buffer if
 * {@link #remove} succeeds, which gives three guarantees the raw APIs do not:
 * a double free is a harmless no-op, a buffer whose owning object failed to
 * construct (and therefore can never be closed) is still released when the
 * context is closed, and tests can assert that nothing leaked.
 * The key is the backend's native handle value (cl_mem address, CUdeviceptr,
 * MTLBuffer id).
 */
public final class AllocationTracker {

    private final Set<Long> live = ConcurrentHashMap.newKeySet();

    public void add(long key) {
        if (key != 0L && !live.add(key)) {
            throw new IllegalStateException("device handle 0x" + Long.toHexString(key) + " registered twice");
        }
    }

    /** @return true if the handle was live (the caller must now release it). */
    public boolean remove(long key) {
        return key != 0L && live.remove(key);
    }

    public int size() {
        return live.size();
    }

    /** Removes and returns every live handle; the caller releases them. */
    public long[] drain() {
        long[] out = live.stream().mapToLong(Long::longValue).toArray();
        for (long k : out) {
            live.remove(k);
        }
        return out;
    }
}
