package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.STRING;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
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
     * Valid when kind == OP and funcName != null: the {@code int} opcode
     * {@link FunctionOpcodes} resolved funcName to, given this node's actual
     * arity ({@code children.size()}) — resolved exactly once, here in the
     * {@link #func} factory, via a single {@code Map} lookup. -1
     * ({@link FunctionOpcodes#UNRESOLVED}) if funcName names a function this
     * compiler doesn't support at this arity; {@link ExprNodeCompiler} is what
     * turns that into a compile-time error, and is also the only place this
     * field is ever read — its evaluator switches on this int exclusively and
     * never compares funcName as a string. Meaningless (always -1) for plain
     * operator nodes (funcName == null) and for NUMBER/VARIABLE nodes.
     */
    public final int funcOpcode;

    /**
     * Valid when kind == OP.
     */
    public final List<ExprNode> children;

    /**
     * ParserNG's real VariableRegistry-assigned frame slot for this leaf, when
     * this tree was built from actual scanned Tokens (see TokenTreeBuilder) —
     * -1 for hand-built or test trees with no real registry behind them. Purely
     * additive: nothing that already reads variableName/stateIndex/
     * isStateVariable() needs to change or even knows this field exists.
     * ExprNodeCompiler prefers this when it's set (>= 0), reading
     * vars[frameIndex] directly instead of reconstructing a position from
     * tSlot/ySlotStart and a name — that's what actually fixes the
     * frame-binding bug from the MathExpression-round-trip approach: no
     * reconstruction, no guessing, just the slot ParserNG itself already
     * assigned.
     */
    public final int frameIndex;

    public static ExprNode number(double v) {
        return new ExprNode(Kind.NUMBER, v, null, null, '\0', null, null, -1, FunctionOpcodes.UNRESOLVED);
    }

    public static ExprNode variable(String name) {
        return new ExprNode(Kind.VARIABLE, 0, name, null, '\0', null, null, -1, FunctionOpcodes.UNRESOLVED);
    }

    public static ExprNode stateVariable(String name, int index) {
        return new ExprNode(Kind.VARIABLE, 0, name, index, '\0', null, null, -1, FunctionOpcodes.UNRESOLVED);
    }

    public static ExprNode op(char opChar, List<ExprNode> children) {
        return new ExprNode(Kind.OP, 0, null, null, opChar, null, children, -1, FunctionOpcodes.UNRESOLVED);
    }

    /**
     * Builds a function-call node, resolving funcName to its opcode right here
     * — once, via a single Map lookup keyed on both name and this call's actual
     * arity ({@code children.size()}). Everything downstream (ExprNodeCompiler)
     * reads the resolved funcOpcode instead of ever re-inspecting funcName. An
     * arity other than 1 or 2, or a name with no matching entry, resolves to
     * FunctionOpcodes.UNRESOLVED (-1); this factory does not throw for that —
     * ExprNodeCompiler's validation is what turns an unresolved opcode into a
     * reported compile-time error.
     *
     * @param funcName
     * @param children
     * @return
     */
    public static ExprNode func(String funcName, List<ExprNode> children) {
        int opcode = children.size() == 1 ? FunctionOpcodes.resolveOneArg(funcName)
                : children.size() == 2 ? FunctionOpcodes.resolveTwoArg(funcName)
                : FunctionOpcodes.UNRESOLVED;
        return new ExprNode(Kind.OP, 0, null, null, '\0', funcName, children, -1, opcode);
    }

    /**
     * Like variable(), but carrying the real frame slot ParserNG's
     * VariableRegistry assigned.
     *
     * @param name
     * @param frameIndex
     * @return
     */
    public static ExprNode variableWithFrame(String name, int frameIndex) {
        return new ExprNode(Kind.VARIABLE, 0, name, null, '\0', null, null, frameIndex, FunctionOpcodes.UNRESOLVED);
    }

    /**
     * Like stateVariable(), but carrying the real frame slot ParserNG's
     * VariableRegistry assigned.
     *
     * @param name
     * @param index
     * @param frameIndex
     * @return
     */
    public static ExprNode stateVariableWithFrame(String name, int index, int frameIndex) {
        return new ExprNode(Kind.VARIABLE, 0, name, index, '\0', null, null, frameIndex, FunctionOpcodes.UNRESOLVED);
    }

    private ExprNode(Kind kind, double numberValue, String variableName, Integer stateIndex,
            char opChar, String funcName, List<ExprNode> children, int frameIndex, int funcOpcode) {
        this.kind = kind;
        this.numberValue = numberValue;
        this.variableName = variableName;
        this.stateIndex = stateIndex;
        this.opChar = opChar;
        this.funcName = funcName;
        this.children = children;
        this.frameIndex = frameIndex;
        this.funcOpcode = funcOpcode;
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

    /**
     * True for a named-function-call node (e.g. sin(...)), as opposed to a
     * plain operator node.
     *
     * @return
     */
    public boolean isFunctionCall() {
        return kind == Kind.OP && funcName != null;
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
                        ? node.variableName + "[" + node.stateIndex + "]"
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

    private static int precedence(char op) {
        switch (op) {
            case '+':
            case '-':
                return 1;
            case '*':
            case '/':
            case '%':
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

    public static ODEFunction compileStandard(MathExpression me) {  
        MathExpression.Slot[] slots = me.getSlotItems();
        return (vars, out) -> {
            for (MathExpression.Slot s : slots) {
                me.updateSlot(s.getSlot(), vars[s.getSlot()]);
            }
            out[0] = me.solveGeneric().scalar;
        };

        /**
         * return (vars, out) -> out[0] =
         * me.solveGeneric(vars).scalar;//expression, vars,
         * independentVariableName, tSlot, ySlotStart);
         */
    }

    public static ODEFunction compileTurbo(MathExpression me) throws Throwable {  
        MathExpression.Slot[] slots = me.getSlotItems();
        FastCompositeExpression fce = new ScalarTurboEvaluator1(me).compile();
        return (vars, out) -> {
            for (MathExpression.Slot s : slots) {
                me.updateSlot(s.getSlot(), vars[s.getSlot()]);
            }
            out[0] = fce.applyScalar(me.getExecutionFrame());
        };
    }
}
