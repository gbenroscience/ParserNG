package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.List;

public class Avg implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> convertedTokens = Utils.evaluateSingleToken(tokens);
        if (convertedTokens.size() == 0) {
            return "0";
        }
        BigDecimal summ = Utils.sum(convertedTokens);
        return summ.divide(new BigDecimal(convertedTokens.size()), new MathContext(10, RoundingMode.HALF_DOWN)).toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), "avarage of all elements. s. Eg: " + getName() + "(1,2) will evaluate (1*2)/2 to 0.5");
    }

    @Override
    public String getName() {
        return "avg";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
