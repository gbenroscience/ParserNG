package parser.methods.ext;


import math.BigDecimalNthRootCalculation;
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

    public static int getFirstBigDeciamalTokenAsInt(List<BigDecimal> tokens) {
        return getFirstStringTokenAsInt(Arrays.asList(tokens.get(0).toString()));
    }

    public static int getFirstStringTokenAsInt(List<String> tokens) {
        //if an output of function is used as first parameter (eg count(1,2,4)/3) , it is eg 1.0, thus float
        BigDecimal toRemoveOrig = new BigDecimal(tokens.get(0), new MathContext(10, RoundingMode.HALF_DOWN));
        BigDecimal toRemoveD = toRemoveOrig.setScale(0, RoundingMode.HALF_DOWN);
        int toRemove = toRemoveD.intValue();
        return toRemove;
    }

    /**
     * This is method, which allows the client functions to workaround the https://github.com/gbenroscience/ParserNG/issues/25
     * Once the issue is fixed, this method will simply change to convert list of strings to list of big decimals withot any evaluations
     *
     * @param tokens list of numbers or parts of mathematical expresion
     * @return the converted numbers or evaluated expression as number.
     */
    public static List<BigDecimal> evaluateSingleToken(List<String> tokens) {
        if (tokens == null) {
            return null;
        }
        try {
            List r = new ArrayList(tokens.size());
            for (String token : tokens) {
                r.add(new BigDecimal(token));
            }
            return r;
        } catch (NumberFormatException ex) {
            String expression = "";
            for (String token : tokens) {
                expression = expression + token;
            }
            return Arrays.asList(new BigDecimal(new MathExpression(expression).solve()));
        }
    }

    /**
     * helper method to unify argument check on standart functions
     *
     * @param name
     * @param count
     * @param tokens
     * @throws RuntimeException
     */
    public static void checkAtLeastArgs(String name, int count, List tokens) throws RuntimeException {
        if (tokens.size() < count) {
            throw new RuntimeException(name + " requires at least" + count + " argument(s). Was " + tokens.size() + "(" + tokens.toString() + ")");
        }
    }
}
