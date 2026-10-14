package com.github.gbenroscience.sqlv1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** {@code SELECT DISTINCT} in {@link ArrowQuery}: numeric, per batch, first-seen order. */
class ArrowQueryDistinctTest {

    private static Float8Vector col(BufferAllocator a, String name, Double... values) {
        Float8Vector v = new Float8Vector(name, a);
        v.allocateNew(Math.max(values.length, 1));
        for (int i = 0; i < values.length; i++) {
            if (values[i] == null) v.setNull(i);
            else v.set(i, values[i]);
        }
        v.setValueCount(values.length);
        return v;
    }

    private static Double[] read(VectorSchemaRoot r, String name) {
        Float8Vector v = (Float8Vector) r.getVector(name);
        Double[] out = new Double[r.getRowCount()];
        for (int i = 0; i < out.length; i++) out[i] = v.isNull(i) ? null : v.get(i);
        return out;
    }

    private static Double[] run(String sql, Double[] x, Double[] y) {
        try (BufferAllocator a = new RootAllocator();
             Float8Vector vx = col(a, "x", x);
             Float8Vector vy = col(a, "y", y);
             VectorSchemaRoot in = new VectorSchemaRoot(List.<FieldVector>of(vx, vy));
             ArrowQuery q = ArrowQuery.compile(sql);
             VectorSchemaRoot out = q.execute(in)) {
            return read(out, out.getFieldVectors().get(0).getName());
        }
    }

    private static final Double[] X = {3.0, 1.0, 3.0, 2.0, 1.0};
    private static final Double[] Y = {10.0, 20.0, 10.0, 30.0, 20.0};

    @Test
    void duplicateRowsAreRemovedInFirstSeenOrder() {
        assertArrayEquals(new Double[] {3.0, 1.0, 2.0}, run("SELECT DISTINCT x FROM t", X, Y));
    }

    @Test
    void rowsAreDistinctAsTuples() {
        try (BufferAllocator a = new RootAllocator();
             Float8Vector vx = col(a, "x", 1.0, 1.0, 2.0, 1.0);
             Float8Vector vy = col(a, "y", 5.0, 6.0, 5.0, 5.0);
             VectorSchemaRoot in = new VectorSchemaRoot(List.<FieldVector>of(vx, vy));
             ArrowQuery q = ArrowQuery.compile("SELECT DISTINCT x, y FROM t");
             VectorSchemaRoot out = q.execute(in)) {
            assertEquals(3, out.getRowCount());
            assertArrayEquals(new Double[] {1.0, 1.0, 2.0}, read(out, "x"));
            assertArrayEquals(new Double[] {5.0, 6.0, 5.0}, read(out, "y"));
        }
    }

    @Test
    void nullsAreEqualToEachOtherAndDistinctFromNaN_negativeZeroEqualsZero() {
        Double[] x = {null, Double.NaN, null, 0.0, -0.0, Double.NaN};
        Double[] y = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        Double[] got = run("SELECT DISTINCT x FROM t", x, y);
        assertEquals(3, got.length);
        assertNull(got[0]);
        assertTrue(Double.isNaN(got[1]));
        assertEquals(Double.valueOf(0.0), got[2]);
    }

    @Test
    void orderByAndLimitApplyAfterDeDuplication() {
        assertArrayEquals(new Double[] {3.0, 2.0}, run("SELECT DISTINCT x FROM t ORDER BY x DESC LIMIT 2", X, Y));
        assertArrayEquals(new Double[] {3.0, 1.0}, run("SELECT DISTINCT x FROM t LIMIT 2", X, Y));
    }

    @Test
    void whereAndComputedColumnsWork() {
        // x > 1 keeps 3, 3, 2 -> d = 6, 6, 4 -> distinct 6, 4 -> ordered 4, 6
        assertArrayEquals(new Double[] {4.0, 6.0}, run("SELECT DISTINCT x * 2 AS d FROM t WHERE x > 1 ORDER BY d", X, Y));
    }

    @Test
    void anEmptyInputGivesAnEmptyResult() {
        assertEquals(0, run("SELECT DISTINCT x FROM t", new Double[0], new Double[0]).length);
    }

    @Test
    void distinctAppliesToAGroupedResult() {
        // groups x=1 -> 2, x=2 -> 2, x=3 -> 2: three groups, one distinct value
        Double[] x = {1.0, 1.0, 2.0, 3.0};
        Double[] y = {1.0, 1.0, 2.0, 2.0};
        assertArrayEquals(new Double[] {2.0}, run("SELECT DISTINCT SUM(y) AS s FROM t GROUP BY x", x, y));
    }

    @Test
    void distinctIsPerBatchNotAcrossExecuteCalls() {
        try (BufferAllocator a = new RootAllocator();
             ArrowQuery q = ArrowQuery.compile("SELECT DISTINCT x FROM t")) {
            for (int batch = 0; batch < 2; batch++) {
                try (Float8Vector vx = col(a, "x", 1.0, 1.0);
                     VectorSchemaRoot in = new VectorSchemaRoot(List.<FieldVector>of(vx));
                     VectorSchemaRoot out = q.execute(in)) {
                    assertEquals(1, out.getRowCount(), "each batch is de-duplicated on its own");
                }
            }
        }
    }

    @Test
    void orderByMustNameASelectListColumnOrAlias() {
        ArrowSqlException e = assertThrows(ArrowSqlException.class, () -> run("SELECT DISTINCT x FROM t ORDER BY y", X, Y));
        assertTrue(e.getMessage().contains("select-list"), e.getMessage());
    }

    @Test
    void aNonNumericOutputColumnIsRejectedWithAClearError() {
        try (BufferAllocator a = new RootAllocator();
             VarCharVector name = new VarCharVector("name", a);
             Float8Vector vx = col(a, "x", 1.0, 2.0)) {
            name.allocateNew(2);
            name.setSafe(0, "a".getBytes(StandardCharsets.UTF_8));
            name.setSafe(1, "b".getBytes(StandardCharsets.UTF_8));
            name.setValueCount(2);
            try (VectorSchemaRoot in = new VectorSchemaRoot(List.<FieldVector>of(name, vx));
                 ArrowQuery q = ArrowQuery.compile("SELECT DISTINCT name FROM t")) {
                ArrowSqlException e = assertThrows(ArrowSqlException.class, () -> q.execute(in));
                assertTrue(e.getMessage().contains("numeric columns only") && e.getMessage().contains("name"), e.getMessage());
            }
        }
    }

    @Test
    void theBridgeEntryPointAndTheReusableOutputPathStillRejectDistinct() {
        assertThrows(UnsupportedOperationException.class,
                () -> ArrowQuery.compile("label", SqlParser.parse("SELECT DISTINCT x FROM t")));
        try (BufferAllocator a = new RootAllocator();
             Float8Vector vx = col(a, "x", 1.0, 1.0);
             VectorSchemaRoot in = new VectorSchemaRoot(List.<FieldVector>of(vx));
             ArrowQuery q = ArrowQuery.compile("SELECT DISTINCT x FROM t")) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, () -> q.allocateReusableOutput(in, 8));
            assertTrue(e.getMessage().contains("DISTINCT"), e.getMessage());
        }
    }

    @Test
    void countDistinctAndOffsetAreStillRejected() {
        assertThrows(UnsupportedOperationException.class, () -> ArrowQuery.compile("SELECT COUNT(DISTINCT x) AS c FROM t"));
        assertThrows(UnsupportedOperationException.class, () -> ArrowQuery.compile("SELECT DISTINCT x FROM t OFFSET 1"));
    }

    @Test
    void integerColumnsAreComparedExactly() {
        // 2^53 and 2^53 + 1 are the same double, so only an exact long comparison keeps them apart.
        try (BufferAllocator a = new RootAllocator();
             BigIntVector v = new BigIntVector("x", a)) {
            v.allocateNew(4);
            v.set(0, 9007199254740992L);
            v.set(1, 9007199254740993L);
            v.set(2, 9007199254740992L);
            v.setNull(3);
            v.setValueCount(4);
            try (VectorSchemaRoot in = new VectorSchemaRoot(List.<FieldVector>of(v));
                 ArrowQuery q = ArrowQuery.compile("SELECT DISTINCT x FROM t");
                 VectorSchemaRoot out = q.execute(in)) {
                assertEquals(3, out.getRowCount(), "2^53, 2^53 + 1 and null");
            }
        }
    }

    @Test
    void distinctOverAGlobalAggregateOfNoRowsStillSeesItsOneRow() {
        // A global aggregate always yields one row, even when no row survives the WHERE.
        assertArrayEquals(new Double[] {0.0}, run("SELECT DISTINCT COUNT(*) AS c FROM t WHERE x > 100", X, Y));
    }
}