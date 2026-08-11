package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;

import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Builds an {@link ExprNode} tree directly from ParserNG's real postfix
 * {@code Token[]} array — the standard one-pass RPN-to-tree stack
 * conversion, the same shape as every other postfix evaluator in this
 * codebase, just building tree nodes instead of evaluating numbers.
 *
 * <h2>Why this exists instead of the string round-trip</h2>
 * An earlier version of this pipeline reconstructed an expression string
 * from ExprNode and re-parsed it through a fresh {@code MathExpression},
 * which meant a fresh, independent VariableRegistry with no required
 * relationship to the real global execution frame — silently wrong once the
 * two diverged. This adapter avoids that class of bug entirely: it never
 * loses ParserNG's own frame assignment in the first place. Every VARIABLE
 * token already carries the real slot {@code VariableRegistry.getSlot(name)}
 * assigned it; this class copies that straight into
 * {@link ExprNode#frameIndex}, and {@link ExprNodeCompiler} reads it
 * directly — no reconstruction, no naming-convention guesswork.
 *
 * <h2>Indexed state variables — the one assumption this makes</h2>
 * With {@code y[3]}-style names now valid, vetted ParserNG variable names,
 * this class assumes the <em>entire</em> literal {@code "y[3]"}, brackets
 * included, is what ends up in {@code Token.name} — i.e. ParserNG's
 * registry has no structural notion of "y at index 3", it is simply an
 * opaque variable name like any other. This class splits that name back
 * into a base ({@code "y"}) and an integer index ({@code 3}) via a small
 * regex, purely to reconstruct the {@code (variableName, stateIndex)} shape
 * {@link ExprNode} and everything downstream (CoefficientExtractor,
 * LinearFormExtractor, TopDerivativeExtractor) already expect. If the real
 * grammar represents indexed variables some other way (a MATRIX-kind token,
 * a separate index field, etc.), only {@link #tryBuildIndexedLeaf} needs to
 * change — nothing else in this class or downstream depends on how the
 * split happens.
 *
 * <h2>Scope: which token kinds and operators are handled</h2>
 * Only NUMBER, VARIABLE, OPERATOR, FUNCTION, and METHOD tokens are
 * supported — LPAREN/RPAREN/COMMA should never appear in a postfix stream
 * and are rejected if seen. Among operators, only the arithmetic set this
 * whole pipeline already understands is supported: unary minus, and binary
 * +, -, *, /, ^. Relational/logical operators (&lt;, &gt;, ==, &amp;&amp;,
 * ...), postfix operators other than unary minus (!, superscript-2,
 * superscript-3, sqrt, ...), and modulo (%) have no meaning in a
 * differential-equation coefficient and are rejected with a specific message
 * rather than silently mishandled.
 *
 * <h2>Scope: what this does NOT do</h2>
 * This does not isolate a single argument out of a larger scanned call the
 * way {@link ArgumentIsolator} does for the string/Lexer path — it assumes
 * the caller hands it the postfix for one already-isolated expression (e.g.
 * one argument of a diffeqn(...) call, compiled on its own). ParserNG's real
 * compiler almost certainly already splits call arguments into separate
 * sub-expressions upstream of this point, making a Token-level
 * re-implementation of that splitting redundant.
 */
public final class TokenTreeBuilder {

    private static final Pattern INDEXED_NAME = Pattern.compile("^([A-Za-z_][A-Za-z0-9_]*)\\[(\\d+)]$");

    private TokenTreeBuilder() {
    }

    public static ExprNode fromPostfix(Token[] postfix) {
        if (postfix == null || postfix.length == 0) {
            throw new IllegalArgumentException("postfix must not be null or empty");
        }

        Deque<ExprNode> stack = new ArrayDeque<>();

        for (Token t : postfix) {
            switch (t.kind) {
                case Token.NUMBER: {
                    stack.push(ExprNode.number(t.value));
                    break;
                }
                case Token.VARIABLE: {
                    stack.push(buildVariableLeaf(t));
                    break;
                }
                case Token.OPERATOR: {
                    stack.push(buildOperator(t, stack));
                    break;
                }
                case Token.FUNCTION:
                case Token.METHOD: {
                    stack.push(buildFunctionCall(t, stack));
                    break;
                }
                default:
                    throw new IllegalArgumentException(
                            "Unsupported token kind in postfix stream for equation compilation: "
                            + Token.getKind(t.kind) + " (name=" + t.name + ") — only NUMBER, VARIABLE, "
                            + "OPERATOR, FUNCTION, and METHOD tokens are expected in a postfix expression.");
            }
        }

        if (stack.size() != 1) {
            throw new IllegalStateException(
                    "Malformed postfix stream: expected exactly 1 result on the stack, got " + stack.size());
        }
        return stack.pop();
    }

    // ------------------------------------------------------------------

    private static ExprNode buildVariableLeaf(Token t) {
        ExprNode indexed = tryBuildIndexedLeaf(t);
        return indexed != null ? indexed : ExprNode.variableWithFrame(t.name, t.frameIndex);
    }

    /** Returns a state-variable leaf if t.name matches the "base[digits]" shape, otherwise null. */
    private static ExprNode tryBuildIndexedLeaf(Token t) {
        if (t.name == null) {
            return null;
        }
        Matcher m = INDEXED_NAME.matcher(t.name);
        if (!m.matches()) {
            return null;
        }
        String base = m.group(1);
        int index = Integer.parseInt(m.group(2));
        return ExprNode.stateVariableWithFrame(base, index, t.frameIndex);
    }

    private static ExprNode buildOperator(Token t, Deque<ExprNode> stack) {
        if (t.arity == 1) {
            if (t.opChar != '-') {
                throw new IllegalArgumentException(
                        "Unsupported unary operator '" + t.opChar + "' for equation compilation — only unary "
                        + "minus is supported (other postfix operators have no meaning in a "
                        + "differential-equation coefficient).");
            }
            requireStackSize(stack, 1, "unary '-'");
            ExprNode operand = stack.pop();
            return ExprNode.op('-', List.of(operand));
        }

        if (t.opChar != '+' && t.opChar != '-' && t.opChar != '*' && t.opChar != '/' && t.opChar != '^') {
            throw new IllegalArgumentException(
                    "Unsupported binary operator '" + t.opChar + "' for equation compilation — only +, -, *, "
                    + "/, ^ are supported (relational, logical, and modulo operators have no meaning in a "
                    + "differential-equation coefficient).");
        }
        requireStackSize(stack, 2, "binary '" + t.opChar + "'");
        ExprNode b = stack.pop();
        ExprNode a = stack.pop();
        return ExprNode.op(t.opChar, List.of(a, b));
    }

    private static ExprNode buildFunctionCall(Token t, Deque<ExprNode> stack) {
        requireStackSize(stack, t.arity, "function '" + t.name + "'");
        ExprNode[] args = new ExprNode[t.arity];
        for (int i = t.arity - 1; i >= 0; i--) {
            args[i] = stack.pop();
        }
        return ExprNode.func(t.name, List.of(args));
    }

    private static void requireStackSize(Deque<ExprNode> stack, int needed, String label) {
        if (stack.size() < needed) {
            throw new IllegalStateException(
                    "Malformed postfix stream: " + label + " needs " + needed + " operand(s) but only "
                    + stack.size() + " available on the stack.");
        }
    }
}