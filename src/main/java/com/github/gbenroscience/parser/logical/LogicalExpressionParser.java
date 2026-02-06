package com.github.gbenroscience.parser.logical;


import java.util.Arrays;

import com.github.gbenroscience.parser.methods.Declarations;

public class LogicalExpressionParser extends AbstractSplittingParser implements LogicalExpressionMemberFactory.LogicalExpressionMember {

    private static final String[] chars1 = new String[]{};
    private static final String[] chars2 = new String[]{"impl", "xor"};
    private static final String[] secchars1 = new String[]{"|", "&"};
    private static final String[] secchars2 = new String[]{"imp", "eq", "or", "and"};
    private final LogicalExpressionMemberFactory subexpressionFactory;

    public LogicalExpressionParser(String expression, ExpressionLogger log) {
        this(expression, log, new ComparingExpressionParser.ComparingExpressionParserFactory());
    }

    public LogicalExpressionParser(String expression, ExpressionLogger log, LogicalExpressionMemberFactory subexpressionFactory) {
        super(expression, log);
        this.subexpressionFactory = subexpressionFactory;
    }

    @Override
    public boolean isLogicalExpressionMember(String originalExpression) {
        String[] methods = Declarations.getInbuiltMethods();
        for (String method : methods) {
            //otherwise eg floor would match or, and thus delegate to Logical parser
            originalExpression = originalExpression.replace(method, "kNoWnMeThOd");
        }
        for (String s : new String[]{"false", "true"}) {
            if (originalExpression.toLowerCase().contains(s)) {
                return true;
            }
        }
        for (String s : chars1) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        for (String s : chars2) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        for (String s : secchars1) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        for (String s : secchars2) {
            if (originalExpression.contains(s)) {
                return true;
            }
        }
        if (subexpressionFactory.createLogicalExpressionMember("", log).isLogicalExpressionMember(originalExpression)) {
            return true;
        }
        return false;
    }


    public String[] getPrimaryChars1() {
        return Arrays.copyOf(chars1, chars1.length);
    }

    public String[] getPrimaryChars2() {
        return Arrays.copyOf(chars2, chars2.length);
    }

    public String[] getSecondaryChars1() {
        return Arrays.copyOf(secchars1, secchars1.length);
    }

    public String[] getSecondaryChars2() {
        return Arrays.copyOf(secchars2, secchars2.length);
    }

    public boolean evaluate() {
        log.log("evaluating logical: " + getOriginal());
        boolean result = subexpressionFactory.createLogicalExpressionMember(split.get(0), new ExpressionLogger.InheritingExpressionLogger(log)).evaluate();
        for (int i = 1; i <= split.size() - 2; i = i + 2) {
            String op = split.get(i);
            LogicalExpressionMemberFactory.LogicalExpressionMember comp2 = subexpressionFactory.createLogicalExpressionMember(split.get(i + 1), new ExpressionLogger.InheritingExpressionLogger(log));
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

    public LogicalExpressionMemberFactory getSubexpressionFactory() {
        return subexpressionFactory;
    }
}
