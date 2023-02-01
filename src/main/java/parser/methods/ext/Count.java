package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.util.List;

public class Count implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> convertedTokens = Utils.evaluateSingleToken(tokens);
        return "" + convertedTokens.size();
    }


    @Override
    public String getHelp() {
        return Help.toLine(getName(), "number of elements in the function. Eg: " + getName() + "(1,1.1) will evaluate 3");
    }

    @Override
    public String getName() {
        return "count";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
