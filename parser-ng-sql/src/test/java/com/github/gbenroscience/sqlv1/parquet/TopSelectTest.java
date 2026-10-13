package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** The heap selection behind bounded top-N, against a full stable sort. */
class TopSelectTest {

    private static TopSelect.Key key(double[] v, boolean[] nulls) {
        return new TopSelect.Key() {
            @Override public boolean isNull(int row) { return nulls[row]; }

            @Override public double get(int row) { return v[row]; }
        };
    }

    @Test
    void picksTheSameRowsAsAFullStableSort() {
        Random r = new Random(5);
        for (int trial = 0; trial < 500; trial++) {
            int rows = r.nextInt(400), limit = 1 + r.nextInt(40), nk = 1 + r.nextInt(2);
            boolean[] desc = new boolean[nk];
            for (int k = 0; k < nk; k++) desc[k] = r.nextBoolean();
            double[][] v = new double[nk][rows];
            boolean[][] nl = new boolean[nk][rows];
            int range = 1 + r.nextInt(12);
            for (int k = 0; k < nk; k++) {
                for (int i = 0; i < rows; i++) {
                    nl[k][i] = r.nextInt(6) == 0;
                    v[k][i] = r.nextInt(range);
                }
            }
            TopSelect.Key[] keys = new TopSelect.Key[nk];
            for (int k = 0; k < nk; k++) keys[k] = key(v[k], nl[k]);
            Integer[] ord = new Integer[rows];
            for (int i = 0; i < rows; i++) ord[i] = i;
            Arrays.sort(ord, (x, y) -> {
                for (int k = 0; k < nk; k++) {
                    int c;
                    if (nl[k][x] || nl[k][y]) c = nl[k][x] == nl[k][y] ? 0 : (nl[k][x] ? 1 : -1);
                    else {
                        c = Double.compare(v[k][x], v[k][y]);
                        if (desc[k]) c = -c;
                    }
                    if (c != 0) return c;
                }
                return 0;
            });
            int[] got = new TopSelect(limit, desc).select(rows, keys, null);
            int want = Math.min(limit, rows);
            assertEquals(want, got.length, "rows=" + rows + " limit=" + limit);
            for (int i = 0; i < want; i++) assertEquals((int) ord[i], got[i], "rows=" + rows + " limit=" + limit + " at " + i);
        }
    }

    @Test
    void aThresholdKeepsOutRowsThatCannotEnter() {
        int rows = 500;
        double[] v = new double[rows];
        Random r = new Random(1);
        for (int i = 0; i < rows; i++) v[i] = r.nextInt(50);
        TopSelect.Key[] keys = {key(v, new boolean[rows])};
        double thr = 5;
        TopSelect.Threshold th = (ks, row) -> ks[0].get(row) < thr;
        for (int row : new TopSelect(10, new boolean[] {false}).select(rows, keys, th)) assertTrue(v[row] < thr);
    }
}
