package com.github.gbenroscience.sqlv1.parquet;

import java.util.Arrays;

/**
 * An id for every distinct byte string, found without allocating per lookup: the caller copies a value's bytes
 * into a reusable buffer and asks for the id; a key array is allocated only when a value is seen for the first
 * time. Open addressing with linear probing, the table doubles at 50% load. Null is a key of its own
 * ({@link #nullId()}). Not thread safe. Used for string group keys and {@code COUNT(DISTINCT string)}, where
 * boxing a {@code Text} and hashing a list per row dominated the cost.
 */
final class ByteKeyTable {

    private byte[][] keys = new byte[16][];
    private int[] hashes = new int[16];
    private int[] slots = new int[64]; // id + 1, 0 = empty
    private int size;
    private int nullId = -1;

    /** Number of distinct keys so far, null included. */
    int size() {
        return size;
    }

    /** The id of the null key, creating it on first use. */
    int nullId() {
        if (nullId < 0) nullId = append(null, 0);
        return nullId;
    }

    /** The bytes of key {@code id}; null for the null key. */
    byte[] key(int id) {
        return keys[id];
    }

    private static int hash(byte[] b, int len) {
        int h = 0x9E3779B9;
        for (int i = 0; i < len; i++) h = (h ^ b[i]) * 0x01000193;
        h ^= h >>> 16;
        return h * 0x85EBCA6B;
    }

    /** The id of {@code b[0..len)}, assigning a new one if it was not seen before. */
    int idOf(byte[] b, int len) {
        int h = hash(b, len);
        int mask = slots.length - 1;
        int s = h & mask;
        while (true) {
            int e = slots[s];
            if (e == 0) break;
            int id = e - 1;
            if (hashes[id] == h && keys[id] != null && keys[id].length == len && regionEquals(keys[id], b, len)) return id;
            s = (s + 1) & mask;
        }
        int id = append(Arrays.copyOf(b, len), h);
        slots[s] = id + 1;
        if (size * 2 > slots.length) rehash();
        return id;
    }

    private static boolean regionEquals(byte[] k, byte[] b, int len) {
        for (int i = 0; i < len; i++) if (k[i] != b[i]) return false;
        return true;
    }

    private int append(byte[] key, int h) {
        if (size == keys.length) {
            keys = Arrays.copyOf(keys, size * 2);
            hashes = Arrays.copyOf(hashes, size * 2);
        }
        keys[size] = key;
        hashes[size] = h;
        return size++;
    }

    private void rehash() {
        int[] next = new int[slots.length * 2];
        int mask = next.length - 1;
        for (int id = 0; id < size; id++) {
            if (keys[id] == null) continue; // the null key is not in the table
            int s = hashes[id] & mask;
            while (next[s] != 0) s = (s + 1) & mask;
            next[s] = id + 1;
        }
        slots = next;
    }
}
