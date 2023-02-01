package parser.methods.ext;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

import math.BigDecimalNthRootCalculation;
import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

public class GeomN implements BasicNumericalMethod {

    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> convertedTokens = Utils.evaluateSingleToken(tokens);
        Utils.checkAtLeastArgs(getName(), 1, convertedTokens);
        if (convertedTokens.size() == 1) {
            return "0";
        }
        int toRemove = Utils.getFirstBigDeciamalTokenAsInt(convertedTokens);
        List<BigDecimal> l = convertedTokens.subList(1, convertedTokens.size());
        Collections.sort(l);
        Utils.trimList(l, toRemove, AvgN.DEFAULT_AVGN_MISIZE);
        BigDecimal gsumm = Utils.gsum(l);
        return BigDecimalNthRootCalculation.nthRoot(l.size(), gsumm).toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(),
                " this is counting limited geom-avg value. First argument is not part of the sum, but is pointing, how much tokens have to be remove from sorted arguments list from both ends, "
                        + "before geom-avg is calculated from the rest. geomN(1,2,4,2) is actually not 16^1/4 but 2, as first 2 and last 4 from  2,2,4 (as 1 is argument) were removed.");
    }

    @Override
    public String getName() {
        return "geomN";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
