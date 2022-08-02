package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.util.List;

public class WeirdFunction implements BasicNumericalMethod {
    @Override
    public String solve(List<String> tokens) {
        return "0." + tokens.get(0);
    }

    @Override
    public String getHelp() {
        return Help.toLine(getName(), "this is help to " + getName());
    }

    @Override
    public String getName() {
        return "weirdFce";
    }

    @Override
    public String getType() {
        return TYPE.NUMBER.toString();
    }
}
