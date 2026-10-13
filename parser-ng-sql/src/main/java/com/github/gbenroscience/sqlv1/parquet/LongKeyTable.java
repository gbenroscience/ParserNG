package com.github.gbenroscience.sqlv1.parquet;

import java.util.Arrays;

/**
 * An id for every distinct {@code long}, with no boxing: group keys and {@code COUNT(DISTINCT)} over integer, date,
 * timestamp, boolean and (as raw bits) floating-point columns. Null is a key of its own. Open addressing, doubles
 * at 50% load. Not thread safe.
 */
final class LongKeyTable {

    private long[] keys = new long[16];
    private int[] slots = new int[64]; // id + 1, 0 = empty
    private int size;
    private int nullId = -1;

    int size() {
        return size;
    }

    int nullId() {
        if (nullId < 0) nullId = append(0);
        return nullId;
    }

    /** The value of key {@code id} (meaningless for the null key). */
    long key(int id) {
        return keys[id];
    }

    private static int hash(long v) {
        long h = v * 0x9E3779B97F4A7C15L;
        return (int) (h ^ (h >>> 32));
    }

    int idOf(long v) {
        int mask = slots.length - 1;
        int s = hash(v) & mask;
        while (true) {
            int e = slots[s];
            if (e == 0) break;
            int id = e - 1;
            if (id != nullId && keys[id] == v) return id;
            s = (s + 1) & mask;
        }
        int id = append(v);
        slots[s] = id + 1;
        if (size * 2 > slots.length) rehash();
        return id;
    }

    private int append(long v) {
        if (size == keys.length) keys = Arrays.copyOf(keys, size * 2);
        keys[size] = v;
        return size++;
    }

    private void rehash() {
        int[] next = new int[slots.length * 2];
        int mask = next.length - 1;
        for (int id = 0; id < size; id++) {
            if (id == nullId) continue;
            int s = hash(keys[id]) & mask;
            while (next[s] != 0) s = (s + 1) & mask;
            next[s] = id + 1;
        }
        slots = next;
    }

    /** A double as a key: {@code -0.0} and {@code 0.0} are the same group, as in SQL. */
    static long ofDouble(double d) {
        return d == 0.0 ? 0L : Double.doubleToLongBits(d);
    }
}
