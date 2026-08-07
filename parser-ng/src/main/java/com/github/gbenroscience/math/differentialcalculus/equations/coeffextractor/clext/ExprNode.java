/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext;
 
 
import java.util.List;

/**
 * @author GBEMIRO
 * Minimal generic expression-tree node used by LinearFormExtractor. This is
 * NOT meant to replace ParserNG's real AST/Token representation — it is a
 * small adapter shape so the extraction algorithm below is concrete and
 * testable without depending on exactly how ParserNG's parser builds its own
 * tree. Wiring this into ParserNG for real means either (a) building one of
 * these from ParserNG's existing postfix Token array (a standard one-pass
 * stack conversion, the same shape as every RPN evaluator in this codebase
 * already uses), or (b) adapting LinearFormExtractor directly onto
 * ParserNG's real node class if one already exists with equivalent
 * information.
 *
 * The one piece of ParserNG-specific knowledge this needs and does not
 * assume: how y[k] — an indexed reference to the k-th state derivative — is
 * represented. That is captured by the stateIndex field below: null for
 * every node that is not a state-variable reference, an integer k for a leaf
 * that denotes y[k]. How that gets populated when building the tree from
 * ParserNG's real token stream depends on how indexed access is tokenized —
 * flagged explicitly in LinearFormExtractor's class javadoc as the one open
 * dependency.
 */
public final class ExprNode {

    public enum Kind { NUMBER, VARIABLE, OP }

    public final Kind kind;

    /** Valid when kind == NUMBER. */
    public final double numberValue;

    /** Valid when kind == VARIABLE. The variable's name, e.g. "t". */
    public final String variableName;

    /**
     * Valid when kind == VARIABLE and this leaf denotes a state reference
     * y[stateIndex]. Null for every other variable (e.g. "t" itself, or a
     * plain scalar parameter).
     */
    public final Integer stateIndex;

    /** Valid when kind == OP. '+', '-', '*', '/', '^', or a unary negation marker. */
    public final char opChar;

    /** Valid when kind == OP and this is a named function call, e.g. "sin". Null for plain operators. */
    public final String funcName;

    /** Valid when kind == OP. */
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

    /**
     * Renders this node, and its whole subtree, as JSON. Fields that don't
     * apply to this node's {@link Kind} are omitted rather than emitted as
     * null, so NUMBER nodes only carry {@code numberValue}, VARIABLE nodes
     * only carry {@code variableName}/{@code stateIndex}, and OP nodes only
     * carry {@code opChar}/{@code funcName}/{@code children}.
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        appendJson(sb);
        return sb.toString();
    }

    void appendJson(StringBuilder sb) {
        sb.append("{\"kind\":\"").append(kind).append('"');
        switch (kind) {
            case NUMBER:
                sb.append(",\"numberValue\":").append(numberValue);
                break;
            case VARIABLE:
                sb.append(",\"variableName\":");
                appendJsonString(sb, variableName);
                sb.append(",\"stateIndex\":").append(stateIndex == null ? "null" : stateIndex.toString());
                break;
            case OP:
                sb.append(",\"opChar\":");
                appendJsonString(sb, opChar == '\0' ? null : String.valueOf(opChar));
                sb.append(",\"funcName\":");
                appendJsonString(sb, funcName);
                sb.append(",\"children\":[");
                if (children != null) {
                    for (int i = 0; i < children.size(); i++) {
                        if (i > 0) {
                            sb.append(',');
                        }
                        children.get(i).appendJson(sb);
                    }
                }
                sb.append(']');
                break;
        }
        sb.append('}');
    }

    /**
     * Writes {@code s} as a JSON string literal (or the bare token {@code null})
     * onto {@code sb}, escaping quotes, backslashes, and control characters.
     * Package-private so {@link CoefficientExtractor}'s result types can reuse
     * it without duplicating escaping logic.
     */
    static void appendJsonString(StringBuilder sb, String s) {
        if (s == null) {
            sb.append("null");
            return;
        }
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
    }
}