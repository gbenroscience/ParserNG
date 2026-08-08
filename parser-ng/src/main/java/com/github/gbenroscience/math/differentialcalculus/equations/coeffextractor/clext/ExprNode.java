package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;
import com.github.gbenroscience.parser.MathExpression;
import java.util.List;

/**
 * Minimal generic expression-tree node used by LinearFormExtractor. This is NOT
 * meant to replace ParserNG's real AST/Token representation — it is a small
 * adapter shape so the extraction algorithm below is concrete and testable
 * without depending on exactly how ParserNG's parser builds its own tree.
 * Wiring this into ParserNG for real means either (a) building one of these
 * from ParserNG's existing postfix Token array (a standard one-pass stack
 * conversion, the same shape as every RPN evaluator in this codebase already
 * uses), or (b) adapting LinearFormExtractor directly onto ParserNG's real node
 * class if one already exists with equivalent information.
 *
 * The one piece of ParserNG-specific knowledge this needs and does not assume:
 * how y[k] — an indexed reference to the k-th state derivative — is
 * represented. That is captured by the stateIndex field below: null for every
 * node that is not a state-variable reference, an integer k for a leaf that
 * denotes y[k]. How that gets populated when building the tree from ParserNG's
 * real token stream depends on how indexed access is tokenized — flagged
 * explicitly in LinearFormExtractor's class javadoc as the one open dependency.
 */
public final class ExprNode {

    public enum Kind {
        NUMBER, VARIABLE, OP
    }

    public final Kind kind;

    /**
     * Valid when kind == NUMBER.
     */
    public final double numberValue;

    /**
     * Valid when kind == VARIABLE. The variable's name, e.g. "t".
     */
    public final String variableName;

    /**
     * Valid when kind == VARIABLE and this leaf denotes a state reference
     * y[stateIndex]. Null for every other variable (e.g. "t" itself, or a plain
     * scalar parameter).
     */
    public final Integer stateIndex;

    /**
     * Valid when kind == OP. '+', '-', '*', '/', '^', or a unary negation
     * marker.
     */
    public final char opChar;

    /**
     * Valid when kind == OP and this is a named function call, e.g. "sin". Null
     * for plain operators.
     */
    public final String funcName;

    /**
     * Valid when kind == OP.
     */
    public final List<ExprNode> children;

    public static ExprNode number(double v) {
        return new ExprNode(Kind.NUMBER, v, null, null, '\0', null, null);
    }

    public static ExprNode variable(String name) {
        return new ExprNode(Kind.VARIABLE, 0, name, null, '\0', null, null);
    }

    public static ExprNode stateVariable(String name, int index) {
        return new ExprNode(Kind.VARIABLE, 0, name, index, '\0', null, null);
    }

    public static ExprNode op(char opChar, List<ExprNode> children) {
        return new ExprNode(Kind.OP, 0, null, null, opChar, null, children);
    }

    public static ExprNode func(String funcName, List<ExprNode> children) {
        return new ExprNode(Kind.OP, 0, null, null, '\0', funcName, children);
    }

    private ExprNode(Kind kind, double numberValue, String variableName, Integer stateIndex,
            char opChar, String funcName, List<ExprNode> children) {
        this.kind = kind;
        this.numberValue = numberValue;
        this.variableName = variableName;
        this.stateIndex = stateIndex;
        this.opChar = opChar;
        this.funcName = funcName;
        this.children = children;
    }

    public boolean isStateVariable() {
        return kind == Kind.VARIABLE && stateIndex != null;
    }

    public boolean isOp(char c) {
        return kind == Kind.OP && funcName == null && opChar == c;
    }

    public boolean isMultiplicative() {
        return isOp('*') || isOp('/');
    }

    public String toExpressionString() {
        return print(this, 0);
    }

    private static String print(ExprNode node, int parentPrec) {
        switch (node.kind) {
            case NUMBER:
                return formatNumber(node.numberValue);

            case VARIABLE:
                return node.isStateVariable()
                        ? "y[" + node.stateIndex + "]"
                        : node.variableName;

            case OP:
                if (node.funcName != null) {
                    StringBuilder sb = new StringBuilder(node.funcName).append('(');
                    for (int i = 0; i < node.children.size(); i++) {
                        if (i > 0) {
                            sb.append(", ");
                        }
                        sb.append(print(node.children.get(i), 0));
                    }
                    return sb.append(')').toString();
                }

                // Unary negation: single child, no funcName.
                if (node.children.size() == 1) {
                    int unaryPrec = 4;
                    String s = "-" + print(node.children.get(0), unaryPrec);
                    return unaryPrec < parentPrec ? "(" + s + ")" : s;
                }

                // Binary op.
                int prec = precedence(node.opChar);
                String left = print(node.children.get(0), prec);
                // prec + 1 on the right forces parens for non-associative cases
                // like a - (b - c) or a / (b / c); harmless extra parens on +/*.
                String right = print(node.children.get(1), prec + 1);
                String s = left + " " + node.opChar + " " + right;
                return prec < parentPrec ? "(" + s + ")" : s;

            default:
                throw new IllegalStateException("Unhandled kind: " + node.kind);
        }
    }
    
      public static ODEFunction compileStandard(ExprNode expression, String independentVariableName,
                                               int tSlot, int ySlotStart) {
       // validate(expression, independentVariableName);
          MathExpression me = new MathExpression(expression.toExpressionString());
        return (vars, out) -> out[0] = me.solveGeneric().scalar;//expression, vars, independentVariableName, tSlot, ySlotStart);
    }

    private static int precedence(char op) {
        switch (op) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
                return 2;
            case '^':
                return 3;
            default:
                throw new IllegalArgumentException("Unknown op: " + op);
        }
    }

    private static String formatNumber(double v) {
        if (v == Math.rint(v) && !Double.isInfinite(v)) {
            return Long.toString((long) v);
        }
        return Double.toString(v);
    }

    @Override
    public String toString() {
        return toExpressionString();
    }
}
