package com.github.gbenroscience.gpu.llm.core;

/**
 * Error metrics between an actual (device) result and an expected (reference)
 * result. Every number the validation framework reports comes from here so all
 * backends are judged by exactly the same arithmetic (accumulation in double).
 */
public final class ErrorStats {

    public final int count;
    public final double maxAbsError;
    public final double meanAbsError;
    /** max over i of |a-e| / max(|e|, relFloor). */
    public final double maxRelError;
    public final double rmsError;
    public final int nanCount;
    public final int infCount;
    /** Elements outside the tolerance passed to {@link #compare}. */
    public final int mismatchCount;
    public final int worstIndex;
    public final float worstActual;
    public final float worstExpected;

    private ErrorStats(int count, double maxAbs, double meanAbs, double maxRel, double rms,
            int nan, int inf, int mismatches, int worstIndex, float worstActual, float worstExpected) {
        this.count = count;
        this.maxAbsError = maxAbs;
        this.meanAbsError = meanAbs;
        this.maxRelError = maxRel;
        this.rmsError = rms;
        this.nanCount = nan;
        this.infCount = inf;
        this.mismatchCount = mismatches;
        this.worstIndex = worstIndex;
        this.worstActual = worstActual;
        this.worstExpected = worstExpected;
    }

    /**
     * An element matches when |a-e| &lt;= absTol + relTol*|e|. Non-finite actual
     * values never match (unless the expected value is the identical
     * infinity). The "worst" element is the one with the largest |a-e|
     * (a non-finite mismatch always outranks a finite one).
     */
    public static ErrorStats compare(float[] actual, float[] expected, double absTol, double relTol) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException("length mismatch: " + actual.length + " vs " + expected.length);
        }
        final double relFloor = 1e-6;
        int n = actual.length;
        double maxAbs = 0, sumAbs = 0, sumSq = 0, maxRel = 0;
        int nan = 0, inf = 0, bad = 0, worst = -1;
        double worstErr = -1;
        int finite = 0;
        for (int i = 0; i < n; i++) {
            float a = actual[i];
            float e = expected[i];
            boolean aNan = Float.isNaN(a);
            boolean aInf = Float.isInfinite(a);
            if (aNan) {
                nan++;
            }
            if (aInf) {
                inf++;
            }
            if (aNan || aInf || Float.isNaN(e) || Float.isInfinite(e)) {
                boolean same = !aNan && !Float.isNaN(e) && a == e;
                if (!same) {
                    bad++;
                    if (worstErr < Double.MAX_VALUE) {
                        worstErr = Double.MAX_VALUE;
                        worst = i;
                    }
                }
                continue;
            }
            double err = Math.abs((double) a - (double) e);
            finite++;
            sumAbs += err;
            sumSq += err * err;
            if (err > maxAbs) {
                maxAbs = err;
            }
            double rel = err / Math.max(Math.abs((double) e), relFloor);
            if (rel > maxRel) {
                maxRel = rel;
            }
            if (err > absTol + relTol * Math.abs((double) e)) {
                bad++;
            }
            if (err > worstErr) {
                worstErr = err;
                worst = i;
            }
        }
        double denom = Math.max(finite, 1);
        return new ErrorStats(n, maxAbs, sumAbs / denom, maxRel, Math.sqrt(sumSq / denom), nan, inf, bad, worst,
                worst >= 0 ? actual[worst] : 0f, worst >= 0 ? expected[worst] : 0f);
    }

    public boolean passed() {
        return mismatchCount == 0 && nanCount == 0 && infCount == 0;
    }

    @Override
    public String toString() {
        return String.format("n=%d maxAbs=%.3e meanAbs=%.3e maxRel=%.3e rms=%.3e nan=%d inf=%d mismatches=%d worst[%d] actual=%s expected=%s",
                count, maxAbsError, meanAbsError, maxRelError, rmsError, nanCount, infCount, mismatchCount,
                worstIndex, worstActual, worstExpected);
    }
}
