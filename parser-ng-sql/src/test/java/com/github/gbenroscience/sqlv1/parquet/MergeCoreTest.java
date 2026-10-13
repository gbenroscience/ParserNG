package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** The merge of per-batch partial aggregates, against a direct aggregation of all the rows. */
class MergeCoreTest {

    // aggregates: COUNT(*) [slot 0], COUNT(x) [1], SUM(x) [value 2, count 1], AVG(x) [2, 1], MIN(x) [3, 1], MAX(x) [4, 1]
    private static final MergeCore.Kind[] KINDS = {MergeCore.Kind.COUNT, MergeCore.Kind.COUNT, MergeCore.Kind.SUM,
            MergeCore.Kind.AVG, MergeCore.Kind.MIN, MergeCore.Kind.MAX};
    private static final int[] VAL = {0, 1, 2, 2, 3, 4};
    private static final int[] CNT = {-1, -1, 1, 1, 1, 1};

    @Test
    void mergingPartialsEqualsAggregatingEverything() {
        Random r = new Random(9);
        for (int trial = 0; trial < 500; trial++) {
            int rows = r.nextInt(300), groups = 1 + r.nextInt(5), batch = 1 + r.nextInt(60);
            double[] x = new double[rows];
            boolean[] xn = new boolean[rows];
            int[] gk = new int[rows];
            for (int i = 0; i < rows; i++) {
                xn[i] = r.nextInt(4) == 0;
                x[i] = r.nextInt(200) - 100;
                gk[i] = r.nextInt(groups);
            }
            double[] cnt = new double[groups], vcnt = new double[groups], sum = new double[groups];
            double[] mn = new double[groups], mx = new double[groups];
            Arrays.fill(mn, Double.POSITIVE_INFINITY);
            Arrays.fill(mx, Double.NEGATIVE_INFINITY);
            for (int i = 0; i < rows; i++) {
                cnt[gk[i]]++;
                if (!xn[i]) {
                    vcnt[gk[i]]++;
                    sum[gk[i]] += x[i];
                    mn[gk[i]] = Math.min(mn[gk[i]], x[i]);
                    mx[gk[i]] = Math.max(mx[gk[i]], x[i]);
                }
            }
            MergeCore m = new MergeCore(KINDS, VAL, CNT);
            Map<Integer, Integer> gid = new HashMap<>();
            for (int s = 0; s < rows; s += batch) {
                int e = Math.min(rows, s + batch);
                for (int g = 0; g < groups; g++) {
                    double c = 0, vc = 0, sm = 0, lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
                    boolean present = false;
                    for (int i = s; i < e; i++) {
                        if (gk[i] != g) continue;
                        present = true;
                        c++;
                        if (!xn[i]) {
                            vc++;
                            sm += x[i];
                            lo = Math.min(lo, x[i]);
                            hi = Math.max(hi, x[i]);
                        }
                    }
                    if (!present) continue;
                    // when a batch has no values the engine may return anything for sum, min and max: use junk on purpose
                    double[] vals = {c, vc, vc > 0 ? sm : Double.NaN, vc > 0 ? lo : Double.POSITIVE_INFINITY, vc > 0 ? hi : Double.NaN};
                    boolean[] nl = {false, false, vc == 0 && r.nextBoolean(), vc == 0, false};
                    Integer id = gid.get(g);
                    if (id == null) {
                        id = m.addGroup();
                        gid.put(g, id);
                    }
                    m.mergeRow(id, vals, nl);
                }
            }
            for (Map.Entry<Integer, Integer> en : gid.entrySet()) {
                int g = en.getKey(), id = en.getValue();
                assertEquals(cnt[g], m.number(0, id));
                assertEquals(vcnt[g], m.number(1, id));
                if (vcnt[g] == 0) {
                    for (int a = 2; a < 6; a++) assertNull(m.number(a, id), "a group with no values gives null");
                } else {
                    assertEquals(sum[g], m.number(2, id));
                    assertEquals(sum[g] / vcnt[g], m.number(3, id), 1e-9);
                    assertEquals(mn[g], m.number(4, id));
                    assertEquals(mx[g], m.number(5, id));
                }
            }
            assertEquals(gid.size(), m.groups());
        }
    }

    @Test
    void aGroupThatSawNoRowsHasCountZeroAndNullSum() {
        MergeCore m = new MergeCore(new MergeCore.Kind[] {MergeCore.Kind.COUNT, MergeCore.Kind.SUM}, new int[] {0, 1}, new int[] {-1, 2});
        int g = m.addGroup();
        assertEquals(0.0, m.number(0, g));
        assertNull(m.number(1, g));
    }
}
