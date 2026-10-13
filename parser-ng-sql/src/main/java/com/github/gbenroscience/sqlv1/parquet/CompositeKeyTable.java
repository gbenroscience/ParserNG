package com.github.gbenroscience.sqlv1.parquet;

import java.util.Arrays;

/**
 * Open-addressing hash table of composite group keys for the Java {@code GROUP BY} fold. A key has a fixed number
 * of parts, each a {@link #LONG} (integers, dates, timestamps, booleans and floating point already folded to a
 * {@code long} by {@link VecKeys#longOf}) or a {@link #STRING} (UTF-8 bytes), and each part may be null. Nulls are
 * equal to each other here, as in SQL grouping.
 *
 * <p>Per row: fill the current key with {@link #setNull}, {@link #setLong} and {@link #setString}, then call
 * {@link #find()}, which returns the dense group id (0, 1, 2 ... in order of first appearance) and inserts the key
 * when it is new. Looking up an existing key allocates nothing; inserting copies the key once (strings into a
 * shared byte arena). Not thread safe.
 */
final class CompositeKeyTable {

    static final byte LONG = 0, STRING = 1;

    private static final long NULL_HASH = 0x5851F42D4C957F2DL;

    private final byte[] kinds;
    private final int n;

    // stored keys, group-major: part p of group g is at [g * n + p]
    private long[] vals;
    private boolean[] nulls;
    private int[] hashes = new int[16];
    private byte[] arena = new byte[256];
    private int arenaLen;
    private int[] slots = new int[64]; // group id + 1, 0 = empty
    private int mask = 63;
    private int size;
    /** Test hook: ANDed into every hash so tests can force collisions and prove equality, not the hash, separates keys. */
    private int hashMask = -1;

    // the current key
    private final long[] cv;
    private final boolean[] cn;
    private final byte[][] cs;
    private final int[] cl;

    CompositeKeyTable(byte[] kinds) {
        this.kinds = kinds.clone();
        this.n = kinds.length;
        this.vals = new long[Math.max(1, n) * 16];
        this.nulls = new boolean[Math.max(1, n) * 16];
        this.cv = new long[n];
        this.cn = new boolean[n];
        this.cs = new byte[n][];
        this.cl = new int[n];
    }

    /** Restricts hashes to the bits of {@code m}; for tests only. Must be called before the first key is inserted. */
    void limitHashBitsForTest(int m) {
        if (size != 0) throw new IllegalStateException("call before inserting");
        this.hashMask = m;
    }

    int size() {
        return size;
    }

    void setNull(int p) {
        cn[p] = true;
    }

    void setLong(int p, long v) {
        cn[p] = false;
        cv[p] = v;
    }

    /** Sets part {@code p} to {@code b[0..len)}; the bytes are read at once, so the caller may reuse {@code b}. */
    void setString(int p, byte[] b, int len) {
        cn[p] = false;
        if (cs[p] == null || cs[p].length < len) cs[p] = new byte[Math.max(len, 32)];
        System.arraycopy(b, 0, cs[p], 0, len);
        cl[p] = len;
    }

    private static long hashBytes(byte[] b, int len) {
        long h = 0xCBF29CE484222325L;
        for (int i = 0; i < len; i++) h = (h ^ (b[i] & 0xFF)) * 0x100000001B3L;
        return h;
    }

    private int hashCurrent() {
        long h = 0x9E3779B97F4A7C15L;
        for (int p = 0; p < n; p++) {
            long x = cn[p] ? NULL_HASH + p : kinds[p] == STRING ? hashBytes(cs[p], cl[p]) : cv[p];
            h = (h ^ x) * 0xFF51AFD7ED558CCDL;
            h ^= h >>> 32;
        }
        return (int) (h ^ (h >>> 32)) & hashMask;
    }

    private boolean matches(int g) {
        int base = g * n;
        for (int p = 0; p < n; p++) {
            boolean sn = nulls[base + p];
            if (sn != cn[p]) return false;
            if (sn) continue;
            long sv = vals[base + p];
            if (kinds[p] == STRING) {
                int off = (int) (sv >>> 32), len = (int) sv;
                if (len != cl[p]) return false;
                byte[] c = cs[p];
                for (int i = 0; i < len; i++) if (arena[off + i] != c[i]) return false;
            } else if (sv != cv[p]) {
                return false;
            }
        }
        return true;
    }

    private int slot(int h) {
        int x = h * 0x9E3779B1;
        return (x ^ (x >>> 15)) & mask;
    }

    /** @return the dense group id of the current key, inserting it when new; a new key gets {@code size()} before the call */
    int find() {
        int h = hashCurrent();
        int i = slot(h);
        while (true) {
            int s = slots[i];
            if (s == 0) break;
            int g = s - 1;
            if (hashes[g] == h && matches(g)) return g;
            i = (i + 1) & mask;
        }
        int g = size++;
        if (size > hashes.length) {
            int cap = Math.max(size, hashes.length * 2);
            hashes = Arrays.copyOf(hashes, cap);
            vals = Arrays.copyOf(vals, cap * Math.max(1, n));
            nulls = Arrays.copyOf(nulls, cap * Math.max(1, n));
        }
        int base = g * n;
        for (int p = 0; p < n; p++) {
            nulls[base + p] = cn[p];
            if (cn[p]) continue;
            if (kinds[p] == STRING) {
                int len = cl[p];
                if (arenaLen + len > arena.length) arena = Arrays.copyOf(arena, Math.max(arenaLen + len, arena.length * 2));
                System.arraycopy(cs[p], 0, arena, arenaLen, len);
                vals[base + p] = ((long) arenaLen << 32) | (len & 0xFFFFFFFFL);
                arenaLen += len;
            } else {
                vals[base + p] = cv[p];
            }
        }
        hashes[g] = h;
        slots[i] = g + 1;
        if (size * 2 > slots.length) rehash();
        return g;
    }

    private void rehash() {
        slots = new int[slots.length * 2];
        mask = slots.length - 1;
        for (int g = 0; g < size; g++) {
            int i = slot(hashes[g]);
            while (slots[i] != 0) i = (i + 1) & mask;
            slots[i] = g + 1;
        }
    }
}
