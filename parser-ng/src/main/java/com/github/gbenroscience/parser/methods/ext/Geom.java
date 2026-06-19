package com.github.gbenroscience.parser.methods.ext;

import com.github.gbenroscience.math.BigDecimalNthRootCalculation;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Help;

import java.math.BigDecimal;
import java.util.List;

public class Geom implements BasicNumericalMethod {

    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> convertedTokens = Utils.tokensToNumbers(tokens);
        if (convertedTokens.size() == 0) {
            return "0";
        }
        BigDecimal gsumm = Utils.gsum(convertedTokens);
        return BigDecimalNthRootCalculation.nthRoot(convertedTokens.size(), gsumm).toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), "returns geometrical average of elements. s. Eg: " + getName() + "(1,2,4) will evaluate (1*2*4)^1/3 thus na 8^1/3 thus to 2");
    }

    @Override
    public String getName() {
        return "geom";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
