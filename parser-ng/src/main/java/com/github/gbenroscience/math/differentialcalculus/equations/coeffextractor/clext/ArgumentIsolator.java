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
 
 
import java.util.ArrayList;
import java.util.List;

/**
 * @author GBEMIRO
 * Isolates the first top-level argument of a function call from its token
 * stream, e.g. given the full scan of
 * {@code diffeqn(<equation>, 1, 0, anon1)}, returns just the {@code
 * <equation>} tokens. If the input does not look like a single wrapping
 * call — e.g. it is already a bare expression — it is returned unchanged,
 * so CoefficientExtractor can accept either the full scanned diffeqn(...)
 * call or just the equation on its own through the same code path.
 *
 * <h2>Redundant outer grouping parens</h2>
 * A scanner may wrap an entire top-level statement in its own redundant
 * grouping parens — e.g. the whole scan being {@code ( diffeqn(...) )}
 * rather than just {@code diffeqn(...)}. Any number of such fully-spanning
 * outer parens are stripped first, before the call-detection rule below is
 * applied, so this works either way without the caller needing to know
 * which convention their scanner uses.
 *
 * <h2>Detection rule (after stripping outer grouping parens)</h2>
 * The remaining input is treated as a wrapping call only if: token(0) is an
 * identifier, token(1) is '(', and the parenthesis opened at token(1) closes
 * at the very last token — i.e. the whole remaining input is that one call's
 * argument list. This distinguishes {@code diffeqn(expr, 1, 0, anon1)} (a
 * wrapping call — isolate) from an expression that merely starts with a
 * function call, like {@code sin(x) + 1} (the parenthesis closes well before
 * the end — not a wrapping call, parse as-is).
 */
public final class ArgumentIsolator {

    private ArgumentIsolator() {
    }

    /**
     * @return the first top-level, comma-separated argument's tokens if the
     *         input is a single wrapping call (after stripping any redundant
     *         outer grouping parens), otherwise the input unchanged
     */
    public static List<String> isolateFirstArgument(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            throw new IllegalArgumentException("tokens must not be null or empty");
        }

        List<String> working = stripRedundantOuterParens(tokens);

        if (working.size() < 3 || !isIdentifier(working.get(0)) || !working.get(1).equals("(")) {
            return working; // does not start like a function call at all
        }

        int depth = 0;
        int closeIndex = -1;
        for (int i = 1; i < working.size(); i++) {
            String tok = working.get(i);
            if (tok.equals("(") || tok.equals("[")) {
                depth++;
            } else if (tok.equals(")") || tok.equals("]")) {
                depth--;
                if (depth == 0) {
                    closeIndex = i;
                    break;
                }
            }
        }

        if (closeIndex != working.size() - 1) {
            // The opening paren at index 1 closes before the end of the input
            // (e.g. "sin(x) + 1") -> this is not a single wrapping call.
            return working;
        }

        // working(2 .. closeIndex-1) is the full argument list; split on
        // top-level commas and take the first segment.
        List<String> firstArg = new ArrayList<>();
        int argDepth = 0;
        for (int i = 2; i < closeIndex; i++) {
            String tok = working.get(i);
            if (tok.equals("(") || tok.equals("[")) {
                argDepth++;
            } else if (tok.equals(")") || tok.equals("]")) {
                argDepth--;
            } else if (tok.equals(",") && argDepth == 0) {
                break; // end of the first argument
            }
            firstArg.add(tok);
        }

        if (firstArg.isEmpty()) {
            throw new IllegalArgumentException(
                    "Detected a wrapping call " + working.get(0) + "(...) but its first argument is empty.");
        }

        return firstArg;
    }

    /** Repeatedly strips a leading '(' that closes exactly at the last token — a redundant grouping wrapper. */
    private static List<String> stripRedundantOuterParens(List<String> tokens) {
        List<String> working = tokens;
        while (working.size() >= 2 && working.get(0).equals("(") && openingSpansToEnd(working)) {
            working = working.subList(1, working.size() - 1);
            if (working.isEmpty()) {
                throw new IllegalArgumentException("Empty parenthesized group after stripping outer parens.");
            }
        }
        return working;
    }

    /** True iff the '(' at index 0 closes exactly at the last index of tokens. */
    private static boolean openingSpansToEnd(List<String> tokens) {
        int depth = 0;
        for (int i = 0; i < tokens.size(); i++) {
            String tok = tokens.get(i);
            if (tok.equals("(") || tok.equals("[")) {
                depth++;
            } else if (tok.equals(")") || tok.equals("]")) {
                depth--;
                if (depth == 0) {
                    return i == tokens.size() - 1;
                }
            }
        }
        return false;
    }

    private static boolean isIdentifier(String tok) {
        return !tok.isEmpty() && (Character.isLetter(tok.charAt(0)) || tok.charAt(0) == '_');
    }
}