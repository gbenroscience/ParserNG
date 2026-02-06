package com.github.gbenroscience.parser.methods.ext;

import java.math.BigDecimal;
import java.util.List;

import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Help;

public class Abs implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        Utils.checkTokensCount("Abs", 1, tokens);
        return new BigDecimal(tokens.get(0)).abs().toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), "absolute value abs(1.3)-> 1.3 abs(-2.5) -> 2.5");
    }

    @Override
    public String getName() {
        return "abs";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
