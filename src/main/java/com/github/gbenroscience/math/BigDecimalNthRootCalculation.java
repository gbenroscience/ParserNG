package math;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * Based on https://stackoverflow.com/questions/22695654/computing-the-nth-root-of-p-using-bigdecimals
 */
public final class BigDecimalNthRootCalculation {

    private static MathContext default_mode = new MathContext(10, RoundingMode.HALF_DOWN);

    private BigDecimalNthRootCalculation() {
    }

    public static BigDecimal nthRoot(final int n, final BigDecimal a) {
        return nthRoot(n, a, default_mode);
    }

    public static BigDecimal nthRoot(final int n, final BigDecimal a, MathContext mc) {
        return nthRoot(n, a, BigDecimal.valueOf(0.1).movePointLeft(mc.getPrecision()), mc);
    }

    private static BigDecimal nthRoot(final int n, final BigDecimal a, final BigDecimal p, MathContext mc) {
        if (a.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("nth root can only be calculated for positive numbers");
        }
        if (a.equals(BigDecimal.ZERO)) {
            return BigDecimal.ZERO;
        }
        BigDecimal xPrev = a;
        BigDecimal x = a.divide(new BigDecimal(n), mc);
        while (x.subtract(xPrev).abs().compareTo(p) > 0) {
            xPrev = x;
            x = BigDecimal.valueOf(n - 1.0)
                    .multiply(x)
                    .add(a.divide(x.pow(n - 1), mc))
                    .divide(new BigDecimal(n), mc);
        }
        return x;
    }

}