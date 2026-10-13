package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** The allocation-free key tables, against {@code HashMap}. */
class KeyTablesTest {

    @Test
    void byteKeysGetStableIdsAndNullIsItsOwnKey() {
        Random r = new Random(5);
        for (int trial = 0; trial < 50; trial++) {
            ByteKeyTable t = new ByteKeyTable();
            Map<String, Integer> ref = new HashMap<>();
            int distinct = 1 + r.nextInt(3000);
            for (int i = 0; i < 10_000; i++) {
                if (r.nextInt(30) == 0) {
                    int id = t.nullId();
                    Integer prev = ref.putIfAbsent(null, id);
                    assertTrue(prev == null || prev == id);
                    continue;
                }
                String s = r.nextInt(distinct) + "x".repeat(r.nextInt(4));
                byte[] junk = new byte[s.length() + 8];
                byte[] sb = s.getBytes();
                System.arraycopy(sb, 0, junk, 3, sb.length);
                byte[] val = Arrays.copyOfRange(junk, 3, 3 + sb.length);
                int id = t.idOf(val, val.length);
                Integer prev = ref.putIfAbsent(s, id);
                assertTrue(prev == null || prev == id, s);
                assertArrayEquals(val, t.key(id));
            }
            assertEquals(ref.size(), t.size());
        }
    }

    @Test
    void anEmptyKeyIsNotTheSameAsNullOrAnotherKey() {
        ByteKeyTable t = new ByteKeyTable();
        byte[] b = new byte[100];
        Arrays.fill(b, (byte) 7);
        int empty = t.idOf(b, 0);
        assertEquals(empty, t.idOf(b, 0));
        assertNotEquals(empty, t.idOf(b, 1));
        assertNotEquals(empty, t.nullId());
    }

    @Test
    void longKeysGetStableIdsAndNullIsItsOwnKey() {
        Random r = new Random(6);
        for (int trial = 0; trial < 50; trial++) {
            LongKeyTable t = new LongKeyTable();
            Map<Long, Integer> ref = new HashMap<>();
            int distinct = 1 + r.nextInt(5000);
            for (int i = 0; i < 10_000; i++) {
                if (r.nextInt(40) == 0) {
                    int id = t.nullId();
                    Integer prev = ref.putIfAbsent(null, id);
                    assertTrue(prev == null || prev == id);
                    continue;
                }
                long v = (r.nextInt(distinct) - distinct / 2) * (r.nextBoolean() ? 1L : 1_000_000_007L);
                int id = t.idOf(v);
                Integer prev = ref.putIfAbsent(v, id);
                assertTrue(prev == null || prev == id, "" + v);
                assertEquals(v, t.key(id));
            }
            assertEquals(ref.size(), t.size());
        }
    }

    @Test
    void zeroAndNullAreDifferentKeysAndNegativeZeroIsZero() {
        LongKeyTable t = new LongKeyTable();
        int zero = t.idOf(0);
        int nul = t.nullId();
        assertNotEquals(zero, nul);
        assertEquals(zero, t.idOf(0));
        assertEquals(nul, t.nullId());
        assertEquals(LongKeyTable.ofDouble(-0.0), LongKeyTable.ofDouble(0.0));
        assertNotEquals(LongKeyTable.ofDouble(1.5), LongKeyTable.ofDouble(2.5));
    }
}
