package com.github.gbenroscience.parser.methods.ext;

import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.BasicNumericalMethod;
import com.github.gbenroscience.parser.methods.Help;

import java.util.List;

public class Lengths {

    public static class Length implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            Utils.checkTokensCount("Length", 1, tokens);
            int[] l = Utils.decimalAndFractionalParts(tokens.get(0));
            return "" + (l[0] + l[1]);
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "number of both fractional and decimal digits");
        }

        @Override
        public String getName() {
            return "length";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class LengthDecimal implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            Utils.checkTokensCount("LengthDecimal", 1, tokens);
            int[] l = Utils.decimalAndFractionalParts(tokens.get(0));
            return "" + l[0];
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "number of decimal digits");
        }

        @Override
        public String getName() {
            return "lengthDecimal";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class LengthFractional implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            Utils.checkTokensCount("LengthFractional", 1, tokens);
            int[] l = Utils.decimalAndFractionalParts(tokens.get(0));
            return "" + l[1];
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "number of fractional digits ");
        }

        @Override
        public String getName() {
            return "lengthFractional";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }
}
