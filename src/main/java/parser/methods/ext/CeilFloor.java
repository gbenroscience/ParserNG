package parser.methods.ext;

import parser.TYPE;
import parser.methods.BasicNumericalMethod;
import parser.methods.Help;

import java.math.RoundingMode;
import java.util.List;

public class CeilFloor {

    public static class Ceil implements BasicNumericalMethod {


        @Override
        public String solve(List<String> tokens) {
            Utils.checkTokensCount("Ceil", 1, tokens);
            return Rounding.naturalRound(0, tokens.get(0), RoundingMode.CEILING).toString();
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
            Utils.checkTokensCount("Floor", 1, tokens);
            return Rounding.naturalRound(0, tokens.get(0), RoundingMode.FLOOR).toString();
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
            Utils.checkTokensCount("CeilN", 2, tokens);
            int fractionalDigits = Utils.getFirstTokenAsInt(tokens);
            return Rounding.naturalRound(fractionalDigits, tokens.get(1), RoundingMode.CEILING).toString();
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
            Utils.checkTokensCount("FloorN", 2, tokens);
            int fractionalDigits = Utils.getFirstTokenAsInt(tokens);
            return Rounding.naturalRound(fractionalDigits, tokens.get(1), RoundingMode.FLOOR).toString();
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
            Utils.checkTokensCount("CeilDigitsN", 2, tokens);
            int digits = Utils.getFirstTokenAsInt(tokens);
            return Rounding.roundDigits(digits, tokens.get(1), RoundingMode.CEILING);
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
            Utils.checkTokensCount("FloorDigitsN", 2, tokens);
            int digits = Utils.getFirstTokenAsInt(tokens);
            return Rounding.roundDigits(digits, tokens.get(1), RoundingMode.FLOOR);
        }

        @Override
        public String getHelp() {
            return Help.toLine(getName(), "flooring digits towards negative infinity to N valid  digits");
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
