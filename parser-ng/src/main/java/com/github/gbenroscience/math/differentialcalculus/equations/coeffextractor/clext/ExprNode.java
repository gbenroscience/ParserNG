package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

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
}
