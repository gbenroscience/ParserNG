package parser.logical;

import java.math.BigDecimal;
import java.util.Arrays;

public class ComparingExpressionParser extends AbstractSplittingParser implements LogicalExpressionMemberFactory.LogicalExpressionMember {

    private static final String[] primaryChars1 = new String[]{"!=", "==", ">=", "<="};
    private static final String[] primaryChars2 = new String[]{"le", "ge", "lt", "gt"};
    private static final String[] secondaryChars1 = new String[]{"<", ">"};
    private static final String[] secondaryChars2 = new String[]{};

    public ComparingExpressionParser(String expression, ExpressionLogger log) {
        super(expression, log);
    }

    public ComparingExpressionParser() {
        super();
    }


    @Override
    public boolean isLogicalExpressionMember(String originalExpression) {
        for (String s : primaryChars1) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        for (String s : primaryChars2) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        for (String s : secondaryChars1) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        for (String s : secondaryChars2) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String[] getPrimaryChars1() {
        return Arrays.copyOf(primaryChars1, primaryChars1.length);
    }

    @Override
    public String[] getPrimaryChars2() {
        return Arrays.copyOf(primaryChars2, primaryChars2.length);
    }

    @Override
    public String[] getSecondaryChars1() {
        return Arrays.copyOf(secondaryChars1, secondaryChars1.length);
    }

    @Override
    public String[] getSecondaryChars2() {
        return Arrays.copyOf(secondaryChars2, secondaryChars2.length);
    }

    @Override
    public boolean evaluate() {
        log.log("evaluating comparison: " + getOriginal());
        if (split.size() == 1) {
            boolean r = parseBooleanStrict(split.get(0).trim(), log);
            log.log("is: " + r);
            return r;
        } else if (split.size() == 3) {
            BigDecimal result1 = new AlgebraExpressionParser(split.get(0), new ExpressionLogger.InheritingExpressionLogger(log)).evaluate();
            BigDecimal result2 = new AlgebraExpressionParser(split.get(2), new ExpressionLogger.InheritingExpressionLogger(log)).evaluate();
            String op = split.get(1);
            log.log("... " + result1.toString() + " " + op + " " + result2.toString());
            if (">".equals(op) || "gt".equals(op)) {
                boolean r = result1.compareTo(result2) > 0;
                log.log("is: " + r);
                return r;
            } else if ("<".equals(op) || "lt".equals(op)) {
                boolean r = result1.compareTo(result2) < 0;
                log.log("is: " + r);
                return r;
            } else if (">=".equals(op) || "ge".equals(op)) {
                boolean r = result1.compareTo(result2) >= 0;
                log.log("is: " + r);
                return r;
            } else if ("<=".equals(op) || "le".equals(op)) {
                boolean r = result1.compareTo(result2) <= 0;
                log.log("is: " + r);
                return r;
            } else if ("!=".equals(op)) {
                boolean r = result1.compareTo(result2) != 0;
                log.log("is: " + r);
                return r;
            } else if ("==".equals(op)) {
                boolean r = result1.compareTo(result2) == 0;
                log.log("is: " + r);
                return r;
            } else {
                throw new ArithmeticException("unknown comparison operator" + op);
            }
        } else {
            throw new ArithmeticException("The comparison operator needs to be operand between operators or true/false. Is " + getOriginal());
        }
    }

    public static boolean parseBooleanStrict(String trim, ExpressionLogger log) {
        if ("true".equalsIgnoreCase(trim)) {
            return true;
        } else if ("false".equalsIgnoreCase(trim)) {
            return false;
        } else {
            throw new ArithmeticException(trim + " is not true/false");
        }
    }

    @Override
    public String getName() {
        return "Comparing operators";
    }

    public static class ComparingExpressionParserFactory implements  LogicalExpressionMemberFactory {

        @Override
        public LogicalExpressionMember createLogicalExpressionMember(String expression, ExpressionLogger log) {
            return new ComparingExpressionParser(expression, log);
        }
    }
}
