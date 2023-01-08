package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.util.List;

public class Sum implements BasicNumericalMethod {


    @Override
    public String solve(List<String> tokens) {
        List<BigDecimal> numericalTokens = Utils.evaluateSingleToken(tokens);
        BigDecimal result = new BigDecimal("0");
        for(BigDecimal numericalToken: numericalTokens){
            result = result.add(numericalToken);
        }
        return result.toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), " sum of arguments suim(1,2+2) is 1+2+2=5");
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
