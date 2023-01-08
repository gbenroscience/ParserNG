package parser.methods.ext;


import parser.MathExpression;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class Utils {

    /**
     * Converts list of strings to list of big decimals
     */
    public static List<BigDecimal> stringsToBigDecimals(final List<String> tokens) {
        return stringsToBigDecimals(tokens, 0);
    }


    /**
     * Converts list of strings to list of big decimals, and custs first
     *
     * @param headCut members to cut of frombthebegining
     */
    public static List<BigDecimal> stringsToBigDecimals(final List<String> tokens, final int headCut) {
        List<BigDecimal> l = new ArrayList<>(Math.max(0, tokens.size() - headCut));
        for (int x = headCut; x < tokens.size(); x++) {
            l.add(new BigDecimal(tokens.get(x)));
        }
        return l;
    }

    /**
     * Will remove toRemove elements from BOTH sides of tokens, leaving at least minLength of elements in place.
     * It always take from both sides, or nto at all
     */
    public static List<?> trimList(final List<?> tokens, final int toRemove, final int minLength) {
        for (int x = 0; x < toRemove; x++) {
            if (minLength == 0) {
                if (tokens.isEmpty()) {
                    break;
                }
            } else {
                if (tokens.size() - 2 < minLength) {
                    break;
                }
            }
            tokens.remove(0);
            if (tokens.size() > 0) {
                tokens.remove(tokens.size() - 1);
            }
        }
        return tokens;
    }


    /**
     * @return the sum of all elements in the data set with BigDecimal precission
     */
    public static BigDecimal sum(final List<BigDecimal> l) {
        BigDecimal summ = new BigDecimal(0);
        for (BigDecimal token : l) {
            summ = summ.add(token);
        }
        return summ;
    }

    /**
     * @return the geomethrical (multiplied) sum of all elements in the data set
     */
    public static BigDecimal gsum(final List<BigDecimal> l) {
        BigDecimal gsumm = new BigDecimal(1);
        for (BigDecimal token : l) {
            gsumm = gsumm.multiply(token);
        }
        return gsumm;
    }

    public static int getFirstTokenAsInt(List<String> tokens) {
        //if an output of function is used as first parameter (eg count(1,2,4)/3) , it is eg 1.0, thus float
        BigDecimal toRemoveOrig = new BigDecimal(tokens.get(0), new MathContext(10, RoundingMode.HALF_DOWN));
        BigDecimal toRemoveD = toRemoveOrig.setScale(0, RoundingMode.HALF_DOWN);
        int toRemove = toRemoveD.intValue();
        return toRemove;
    }

    public static boolean checkOnlyNumbers(List<String> tokens) {
        try {
            for (String token : tokens) {
                new BigDecimal(token);
            }
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    public static List<BigDecimal> evaluateSingleToken(List<String> tokens) {
        try {
            return tokensToNumbers(tokens);
        } catch (NumberFormatException ex) {
            return Arrays.asList(new BigDecimal(new MathExpression(connectTokens(tokens)).solve()));
        }
    }

    private static List<BigDecimal> tokensToNumbers(List<String> tokens) {
        List<BigDecimal> r = new ArrayList(tokens.size());
        for (String token : tokens) {
            r.add(new BigDecimal(token));
        }
        return r;
    }

    public static String connectTokens(List<String> tokens) {
        StringBuilder expression = new StringBuilder();
        for (String token : tokens) {
            expression.append(token);
        }
        return expression.toString();
    }

}
