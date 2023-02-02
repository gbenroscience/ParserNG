package parser.methods.ext;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

public class Rounding {

    /* Implemented is only round, rundN, roundDigitsN and roundX ( 5 up)
     * Whole set is to add also  5 down):
     * round, roundDown
     * roundN, roundDownN
     * roundDigitsN, roundDigitsDown
     * roundX, roundDownX
     */

    public static class RoundN implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.evaluateSingleToken(tokens);
            Utils.checkTokensCount("RoundN", 2, tt);
            int fractionalDigits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return naturalRound(fractionalDigits, tt.get(1), RoundingMode.HALF_UP).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "natural round of decimal number to another decimal by rounding to N fractional digits. eg roudN(4, 4.2585) -> 4.259");
        }

        @Override
        public String getName() {
            return "roundN";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class Round implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.evaluateSingleToken(tokens);
            Utils.checkTokensCount("Round", 1, tt);
            return naturalRound(0, tt.get(0), RoundingMode.HALF_UP).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "natural round of decimal number to integer: 1.1->1, 1,4->1, 1.44->1, 1.45->1,  1.5->2, 1.9->2. It is same as calling roundN(0, ..)");
        }

        @Override
        public String getName() {
            return "round";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class RoundX implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.evaluateSingleToken(tokens);
            Utils.checkTokensCount("RoundX", 2, tt);
            int fractionalDigits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return unnaturalRound(fractionalDigits, tt.get(1)).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "unnatural round of decimal number to another decimal by rounding to N digits. eg roundX(1, 4.445)-> 4.5 unlike natural 4.4. 12.444 would remain natural.");
        }

        @Override
        public String getName() {
            return "roundX";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }

    }

    public static class RoundDigitsN implements BasicNumericalMethod {

        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.evaluateSingleToken(tokens);
            Utils.checkTokensCount("roundDigitsN", 2, tt);
            int digits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return roundDigits(digits, tt.get(1), RoundingMode.HALF_UP);
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "round the input to number of N digits roundDigitsN(2,125.134) -> 130");
        }

        @Override
        public String getName() {
            return "roundDigitsN";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }


    public static String roundDigits(int digits, BigDecimal bd, RoundingMode m) {
        return bd.round(new MathContext(digits, m)).toString();
    }


    static BigDecimal naturalRound(int origScale, BigDecimal orig, RoundingMode m) {
        if (origScale < 0) {
            throw new RuntimeException("Scale must be 0 or positive");
        }
        BigDecimal rounded = orig.setScale(origScale, m);
        return rounded;
    }

    static BigDecimal unnaturalRound(int origScale, String orig) {
        return unnaturalRound(origScale, new BigDecimal(orig));
    }

    static BigDecimal unnaturalRound(int origScale, BigDecimal orig) {
        if (origScale < 0) {
            throw new RuntimeException("Scale must be 0 or positive");
        }
        BigDecimal rounded = orig;
        while (Utils.decimalAndFractionalParts(rounded)[1] > origScale) {
            int[] l = Utils.decimalAndFractionalParts(rounded);
            rounded = rounded.setScale(l[1] - 1, RoundingMode.HALF_UP);
        }
        return rounded;
    }
}
