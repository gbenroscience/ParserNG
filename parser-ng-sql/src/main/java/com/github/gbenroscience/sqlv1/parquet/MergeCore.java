package com.github.gbenroscience.sqlv1.parquet;

import java.util.Arrays;

/**
 * The arithmetic of merging partial aggregates, with no Arrow dependency. Each batch is aggregated by the engine
 * into <em>partial</em> values (a count, a sum with its count of values, a minimum or maximum with its count of
 * values); this class folds them into the final value of each aggregate, per group.
 *
 * <p>A partial sum, minimum or maximum is used only when its partial count is positive, so what the engine returns
 * for a batch with no values (zero, {@code NaN}, an infinity or a null) never leaks into the result. The final
 * value is null when the group has no values at all, as SQL defines for {@code SUM}, {@code AVG}, {@code MIN}
 * and {@code MAX}; counts are never null.
 */
final class MergeCore {

    /** The aggregates that can be merged from partials. */
    enum Kind { COUNT, SUM, AVG, MIN, MAX }

    private final Kind[] kinds;
    /** Partial slot that carries each aggregate's value (the count itself for COUNT). */
    private final int[] valSlot;
    /** Partial slot with the count of values behind a SUM/AVG/MIN/MAX, or -1 for COUNT. */
    private final int[] cntSlot;
    private double[][] value;
    private double[][] count;
    private int groups;

    MergeCore(Kind[] kinds, int[] valSlot, int[] cntSlot) {
        this.kinds = kinds.clone();
        this.valSlot = valSlot.clone();
        this.cntSlot = cntSlot.clone();
        this.value = new double[kinds.length][16];
        this.count = new double[kinds.length][16];
    }

    int groups() {
        return groups;
    }

    /** Creates group {@code groups()} and returns its index. */
    int addGroup() {
        int g = groups++;
        if (kinds.length > 0 && g >= value[0].length) {
            for (int a = 0; a < kinds.length; a++) {
                value[a] = Arrays.copyOf(value[a], Math.max(g + 1, value[a].length * 2));
                count[a] = Arrays.copyOf(count[a], value[a].length);
            }
        }
        for (int a = 0; a < kinds.length; a++) {
            value[a][g] = kinds[a] == Kind.MIN ? Double.POSITIVE_INFINITY : kinds[a] == Kind.MAX ? Double.NEGATIVE_INFINITY : 0.0;
            count[a][g] = 0.0;
        }
        return g;
    }

    /**
     * Folds one partial row into group {@code g}.
     *
     * @param vals  the partial value of each slot
     * @param nulls whether each slot is null
     */
    void mergeRow(int g, double[] vals, boolean[] nulls) {
        for (int a = 0; a < kinds.length; a++) {
            int vs = valSlot[a];
            if (kinds[a] == Kind.COUNT) {
                if (!nulls[vs]) value[a][g] += vals[vs];
                continue;
            }
            int cs = cntSlot[a];
            double c = nulls[cs] ? 0.0 : vals[cs];
            if (!(c > 0) || nulls[vs]) continue;
            double v = vals[vs];
            switch (kinds[a]) {
                case SUM, AVG -> value[a][g] += v;
                case MIN -> { if (v < value[a][g]) value[a][g] = v; }
                default -> { if (v > value[a][g]) value[a][g] = v; }
            }
            count[a][g] += c;
        }
    }

    /** The final value of aggregate {@code a} for group {@code g}, or {@code null} for no value. */
    Double number(int a, int g) {
        if (kinds[a] == Kind.COUNT) return value[a][g];
        if (!(count[a][g] > 0)) return null;
        return kinds[a] == Kind.AVG ? value[a][g] / count[a][g] : value[a][g];
    }
}
