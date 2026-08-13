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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common;

/**
 *
 * @author GBEMIRO
 */ 

import com.github.gbenroscience.parser.MathExpression.Token;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * Isolates one top-level argument's own postfix {@code Token[]} out of a
 * larger call's postfix — e.g. given the full scan of {@code
 * diffeqnHO(<equation>, t0, y0, tEnd, h, method)}, returns just the
 * {@code <equation>} tokens, still carrying their REAL VariableRegistry
 * frame indices from the one {@code MathExpression} the whole call was
 * parsed as.
 *
 * <h2>Why this replaces the old (.old package) ArgumentIsolator</h2>
 * The old isolator operated on raw text tokens ({@code List<String>}) and
 * returned text meant to be re-parsed through a fresh {@code
 * MathExpression} — exactly the "reconstruct a string, re-parse it, get a
 * fresh independent VariableRegistry with no required relationship to the
 * real execution frame" pattern {@link TokenTreeBuilder}'s own javadoc
 * identifies as a source of silent bugs once the two registries diverge.
 * This version never leaves the {@code Token} domain and never re-parses
 * anything, so there is no second registry to diverge from the first.
 * (Confirmed as the old convention — the raw equation argument is no
 * longer a lambda body either, so there is no opaque {@code
 * FUNCTION_HANDLE} indirection to work around: the equation's own tokens
 * sit directly inline in the call's postfix, like any other argument.)
 *
 * <h2>The algorithm</h2>
 * A naive "argument N ends where the running stack depth first reaches
 * N+1" is wrong: depth can revisit an earlier milestone value in the
 * middle of a single argument's own postfix (e.g. {@code a+b} reaches
 * depth 1 right after {@code a} alone, long before the argument actually
 * ends). The correct technique is the same stack-based subtree
 * reconstruction {@link TokenTreeBuilder} already uses to build {@code
 * ExprNode} trees — except tracking {@code [startIndex, endIndex]} index
 * spans instead of building nodes. Replaying the argument region as a
 * stack machine (each token either pushes a new one-token span, or pops
 * {@code arity}-many spans and pushes their combined span) leaves exactly
 * N spans on the stack at the end, one per argument, each span being
 * exactly that argument's own token range.
 */
public final class PostfixArgumentIsolator {

    private PostfixArgumentIsolator() {
    }

    /**
     * @param callPostfix the full postfix for the whole call, with the call
     *                    token itself as the LAST element (always true for
     *                    the outermost call in a parsed expression, or for
     *                    any sub-call whose own full postfix is passed in)
     * @param argIndex    0-based index of the argument to isolate
     * @return that argument's own postfix — a fresh array, a contiguous
     *         slice of callPostfix, tokens unmodified (same frameIndex,
     *         same everything)
     */
    public static Token[] isolateArgument(Token[] callPostfix, int argIndex) {
        if (callPostfix == null || callPostfix.length < 2) {
            throw new IllegalArgumentException("callPostfix must contain at least a call token and one argument");
        }
        Token callToken = callPostfix[callPostfix.length - 1];
        if (argIndex < 0 || argIndex >= callToken.arity) {
            throw new IllegalArgumentException(
                    "argIndex=" + argIndex + " out of range for call '" + callToken.name
                    + "' with arity=" + callToken.arity);
        }

        int regionEnd = callPostfix.length - 1; // exclusive of the call token itself
        Deque<int[]> spanStack = new ArrayDeque<>();

        for (int i = 0; i < regionEnd; i++) {
            Token t = callPostfix[i];
            int operandCount = operandCount(t);
            if (operandCount == 0) {
                spanStack.push(new int[]{i, i});
                continue;
            }
            if (spanStack.size() < operandCount) {
                throw new IllegalStateException(
                        "Malformed call postfix: '" + t.name + "' at index " + i + " needs " + operandCount
                        + " operand(s) but only " + spanStack.size() + " available.");
            }
            int start = i;
            for (int k = 0; k < operandCount; k++) {
                start = Math.min(start, spanStack.pop()[0]);
            }
            spanStack.push(new int[]{start, i});
        }

        int n = spanStack.size();
        if (n != callToken.arity) {
            throw new IllegalStateException(
                    "Malformed call postfix: found " + n + " argument(s) for '" + callToken.name
                    + "', expected " + callToken.arity + ".");
        }

        int[][] spansInOrder = new int[n][];
        for (int i = n - 1; i >= 0; i--) {
            spansInOrder[i] = spanStack.pop();
        }

        int[] target = spansInOrder[argIndex];
        return Arrays.copyOfRange(callPostfix, target[0], target[1] + 1);
    }

    /** How many stack items a token consumes: 0 for a leaf (NUMBER/VARIABLE/MATRIX), t.arity otherwise. */
    private static int operandCount(Token t) {
        switch (t.kind) {
            case Token.NUMBER:
            case Token.VARIABLE:
            case Token.MATRIX:
            case Token.FUNCTION_HANDLE:
            case Token.FUNCTION_HANDLE_UNDEFINED:
                return 0;
            case Token.OPERATOR:
            case Token.FUNCTION:
            case Token.METHOD:
                return t.arity;
            default:
                throw new IllegalArgumentException(
                        "Unexpected token kind inside a call's argument region: " + Token.getKind(t.kind));
        }
    }
}