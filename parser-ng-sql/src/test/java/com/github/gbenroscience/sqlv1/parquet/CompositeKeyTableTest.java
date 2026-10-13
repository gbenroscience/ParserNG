package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** The composite group-key table, against {@code HashMap<List<Object>>}. */
class CompositeKeyTableTest {

    private static final byte L = CompositeKeyTable.LONG, S = CompositeKeyTable.STRING;

    /** Random keys of (long, string, long) with nulls; ids must equal the reference's first-seen numbering. */
    private static void check(int hashBits, long seed) {
        Random r = new Random(seed);
        CompositeKeyTable t = new CompositeKeyTable(new byte[] {L, S, L});
        if (hashBits != -1) t.limitHashBitsForTest(hashBits);
        Map<List<Object>, Integer> ref = new HashMap<>();
        for (int i = 0; i < 40_000; i++) {
            Long a = r.nextInt(8) == 0 ? null : (long) r.nextInt(40);
            String b = r.nextInt(8) == 0 ? null : r.nextInt(30) + "x".repeat(r.nextInt(3));
            Long c = r.nextInt(8) == 0 ? null : (long) r.nextInt(5) - 2;
            if (a == null) t.setNull(0); else t.setLong(0, a);
            if (b == null) {
                t.setNull(1);
            } else {
                byte[] raw = b.getBytes(StandardCharsets.UTF_8);
                byte[] padded = new byte[raw.length + 5]; // the buffer is longer than the value, as a reused scratch is
                Arrays.fill(padded, (byte) 'z');
                System.arraycopy(raw, 0, padded, 0, raw.length);
                t.setString(1, padded, raw.length);
            }
            if (c == null) t.setNull(2); else t.setLong(2, c);
            int id = t.find();
            List<Object> key = Arrays.asList(a, b, c);
            Integer expected = ref.get(key);
            if (expected == null) {
                expected = ref.size();
                ref.put(key, expected);
            }
            assertEquals(expected.intValue(), id, "row " + i + " key " + key);
        }
        assertEquals(ref.size(), t.size());
    }

    @Test
    void idsFollowFirstSeenOrderAndNullsAreTheirOwnValue() {
        for (long seed = 1; seed <= 5; seed++) check(-1, seed);
    }

    @Test
    void equalityNotTheHashSeparatesKeys() {
        check(0, 7);   // every hash is 0: everything collides
        check(3, 8);   // eight distinct hashes
    }

    @Test
    void emptyStringAndNullAreDifferentKeys() {
        CompositeKeyTable t = new CompositeKeyTable(new byte[] {S, L});
        t.setString(0, new byte[4], 0);
        t.setLong(1, 1);
        int empty = t.find();
        t.setNull(0);
        t.setLong(1, 1);
        int nul = t.find();
        assertNotEquals(empty, nul);
        t.setString(0, new byte[0], 0);
        t.setLong(1, 1);
        assertEquals(empty, t.find());
        assertEquals(2, t.size());
    }

    @Test
    void arenaAndTablesGrowWithManyDistinctKeys() {
        CompositeKeyTable t = new CompositeKeyTable(new byte[] {S, L});
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < 200_000; i++) {
            byte[] b = ("key-" + i).getBytes(StandardCharsets.UTF_8);
            t.setString(0, b, b.length);
            t.setLong(1, i % 3);
            ids.add(t.find());
        }
        assertEquals(200_000, t.size());
        for (int i = 0; i < 200_000; i += 997) {
            byte[] b = ("key-" + i).getBytes(StandardCharsets.UTF_8);
            t.setString(0, b, b.length);
            t.setLong(1, i % 3);
            assertEquals(ids.get(i).intValue(), t.find());
        }
        assertEquals(200_000, t.size());
    }
}
