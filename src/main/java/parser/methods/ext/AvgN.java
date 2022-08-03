package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AvgN implements BasicNumericalMethod {

    @Override
    public String solve(List<String> tokens) {
        int toRemove = new BigDecimal(tokens.get(0)).round(new MathContext(0, RoundingMode.HALF_DOWN)).intValue();
        List<BigDecimal> l = new ArrayList<>(tokens.size() - 1);
        for (int x = 1; x < tokens.size(); x++) {
            l.add(new BigDecimal(tokens.get(x)));
        }
        Collections.sort(l);
        for (int x = 0; x < toRemove; x++) {
            if (l.size() < 3) {
                break;
            }
            l.remove(0);
            l.remove(l.size() - 1);
        }
        BigDecimal summ = new BigDecimal(0);
        for (BigDecimal token : l) {
            summ = summ.add(token);
        }
        return summ.divide(new BigDecimal(l.size()), new MathContext(10, RoundingMode.HALF_DOWN)).toString();
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), " this is counting limited avg value. First argument is not part of the sum, but is pointing, how much tokens have to be remove from sorted arguments list from both ends, before avg is calculated from the rest. avgN(1,2,4,2) is actually not 8/3 but 2, as first 2 and last 4 were removed.");
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
