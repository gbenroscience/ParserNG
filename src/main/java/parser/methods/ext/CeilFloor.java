package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

public class CeilFloor {

    public static class Ceil implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.tokensToNumbers(tokens);
            Utils.checkTokensCount("Ceil", 1, tt);
            return Rounding.naturalRound(0, tt.get(0), RoundingMode.CEILING).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "ceiling fractional digits towards positive infinity. Same as ceilN(0,...)");
        }

        @Override
        public String getName() {
            return "ceil";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class Floor implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.tokensToNumbers(tokens);
            Utils.checkTokensCount("Floor", 1, tt);
            return Rounding.naturalRound(0, tt.get(0), RoundingMode.FLOOR).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "flooring fractional digits towards negative infinity. Same as floorN(0,...)");
        }

        @Override
        public String getName() {
            return "floor";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }
    public static class CeilN implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.tokensToNumbers(tokens);
            Utils.checkTokensCount("CeilN", 2, tt);
            int fractionalDigits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return Rounding.naturalRound(fractionalDigits, tt.get(1), RoundingMode.CEILING).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "ceiling fractional digits towards positive infinity to N valid fractional digits");
        }

        @Override
        public String getName() {
            return "ceilN";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class FloorN implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.tokensToNumbers(tokens);
            Utils.checkTokensCount("FloorN", 2, tt);
            int fractionalDigits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return Rounding.naturalRound(fractionalDigits, tt.get(1), RoundingMode.FLOOR).toString();
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "flooring fractional digits towards negative infinity to N valid fractional digits");
        }

        @Override
        public String getName() {
            return "floorN";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }


    public static class CeilDigitsN implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.tokensToNumbers(tokens);
            Utils.checkTokensCount("CeilDigitsN", 2, tt);
            int digits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return Rounding.roundDigits(digits, tt.get(1), RoundingMode.CEILING);
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "ceiling digits towards positive infinity to N valid digits");
        }

        @Override
        public String getName() {
            return "ceilDigitsN";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }

    public static class FloorDigitsN implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            List<BigDecimal> tt = Utils.tokensToNumbers(tokens);
            Utils.checkTokensCount("FloorDigitsN", 2, tt);
            int digits = Utils.getFirstBigDeciamalTokenAsInt(tt);
            return Rounding.roundDigits(digits, tt.get(1), RoundingMode.FLOOR);
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "flooring digits towards negative infinity to N valid digits");
        }

        @Override
        public String getName() {
            return "floorDigitsN";
        }

        @Override
        public String getType() {
            return TYPE.NUMBER.toString();
        }
    }
}
