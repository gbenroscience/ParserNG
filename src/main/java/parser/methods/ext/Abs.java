package parser.methods.ext;

import java.math.BigDecimal;
import java.util.List;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

public class Abs implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        if (tokens.size() != 1) {
            throw new RuntimeException("abs function takes exactly one argument");
        }
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
