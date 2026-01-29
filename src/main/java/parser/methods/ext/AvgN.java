package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.List;

public class AvgN implements BasicNumericalMethod {

    public static final int DEFAULT_AVGN_MISIZE = 2;

    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> convertedTokens = Utils.tokensToNumbers(tokens);
        Utils.checkAtLeastArgs(getName(), 1, convertedTokens);
        if (convertedTokens.size() == 1) {
            return "0";
        }
        int toRemove = Utils.getFirstBigDeciamalTokenAsInt(convertedTokens);
        List<BigDecimal> l = convertedTokens.subList(1, convertedTokens.size());
        Collections.sort(l);
        Utils.trimList(l, toRemove, DEFAULT_AVGN_MISIZE);
        BigDecimal summ = Utils.sum(l);
        return summ.divide(new BigDecimal(l.size()), new MathContext(10, RoundingMode.HALF_DOWN)).toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), " this is counting limited avg value. First argument is not part of the sum, but is pointing, how much tokens have to be remove from sorted arguments list from both ends, before avg is calculated from the rest. avgN(1,2,4,2) is actually not 9/4 but 2, as first 2 and last 4 from  2,2,4 (as 1 is argument) were removed. Always at least two items are left list");
    }

    @Override
    public String getName() {
        return "avgN";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
