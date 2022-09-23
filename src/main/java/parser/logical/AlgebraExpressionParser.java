package parser.logical;

import parser.MathExpression;

import java.math.BigDecimal;

public class AlgebraExpressionParser {

    private final MathExpression mathExpression;
    private final String original;
    private final ExpressionLogger log;

    public AlgebraExpressionParser(String expr, ExpressionLogger log) {
        original = expr;
        this.log = log;
        mathExpression = new MathExpression(original);
    }

    public BigDecimal evaluate() {
        log.log("evaluating: " + original);
        BigDecimal r =  new BigDecimal(mathExpression.solve());
        log.log("is: " + r.toString());
        return r;
    }
}
