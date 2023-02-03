package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.util.List;

public class Gsum implements BasicNumericalMethod {

    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> convertedTokens = Utils.tokensToNumbers(tokens);
        if (convertedTokens.size() == 0) {
            return "0";
        }
        return Utils.gsum(convertedTokens).toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), "product of all elements. s. Eg: " + getName() + "(1,2) will evaluate 1*2 to 2");
    }

    @Override
    public String getName() {
        return "gsum";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
