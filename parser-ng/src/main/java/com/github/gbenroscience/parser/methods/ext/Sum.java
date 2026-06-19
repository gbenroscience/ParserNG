package com.github.gbenroscience.parser.methods.ext;

import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Help;

import java.math.BigDecimal;
import java.util.List;

public class Sum implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> numericalTokens = Utils.tokensToNumbers(tokens);
        BigDecimal result = new BigDecimal("0");
        for(BigDecimal numericalToken: numericalTokens){
            result = result.add(numericalToken);
        }
        return result.toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), " sum of arguments sum(1,2+2) is 1+2+2=5");
    }

    @Override
    public String getName() {
        return "sum";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
