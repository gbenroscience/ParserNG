package parser.logical;


import java.util.Arrays;

public class LogicalExpressionParser extends AbstractSplittingParser {

    private static final String[] chars = new String[]{"imp", "eq", "or", "and", "|", "&"};

    public LogicalExpressionParser(String expression, ExpressionLogger log) {
        super(expression, log);
    }


    public String[] getPrimaryChars() {
        return new String[]{"impl", "xor"};
    }

    public String[] getSecondaryChars() {
        return Arrays.copyOf(chars, chars.length);
    }

    public boolean evaluate() {
        log.log("evaluating: " + getOriginal());
        boolean result = new ComparingExpressionParser(split.get(0), new ExpressionLogger.InheritingExpressionLogger(log)).evaluate();
        for (int i = 1; i <= split.size() - 2; i = i + 2) {
            String op = split.get(i);
            ComparingExpressionParser comp2 = new ComparingExpressionParser(split.get(i + 1),new ExpressionLogger.InheritingExpressionLogger(log));
            boolean r2 = comp2.evaluate();
            log.log("... " + result + " " + op + " " + r2);
            if ("&".equals(op) || "and".equals(op)) {
                result = result && r2;
            } else if ("|".equals(op) || "or".equals(op)) {
                result = result || r2;
            } else if ("impl".equals(op) || "imp".equals(op)) {
                if (result && !r2) {
                    return false;
                } else {
                    return true;
                }
            } else if ("eq".equals(op)) {
                result = result == r2;
            } else if ("xor".equals(op)) {
                result = (result != r2);
            } else {
                throw new ArithmeticException("invalid operator " + op);
            }
        }
        log.log("is: " + result);
        return result;
    }

    @Override
    public String getName() {
        return "Logical operators";
    }
}
