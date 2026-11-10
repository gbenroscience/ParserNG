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
package com.github.gbenroscience.math.differentialcalculus.pde;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Text-level front end: splits {@code LHS = RHS}, finds every
 * {@code diff(...)} call on an unknown function and replaces it by a
 * placeholder symbol. The rewritten text is then handed to ParserNG's own
 * parser/compiler, so no second expression language exists: placeholders are
 * ordinary ParserNG variables whose values the discretization supplies.
 */
final class EquationAnalyzer {

    /** A derivative of an unknown: spatial orders per axis plus a time order. */
    static final class DerivRef {

        final String comp;
        final int[] sp = new int[3];
        int tOrd;
        String placeholder;
        int index;

        DerivRef(String comp) {
            this.comp = comp;
        }

        boolean isSpatial() {
            return sp[0] + sp[1] + sp[2] > 0;
        }

        String key() {
            return comp + "|" + sp[0] + "," + sp[1] + "," + sp[2] + "|" + tOrd;
        }
    }

    /** Equation after placeholder substitution. */
    static final class Parsed {

        final String source;
        final String lhs;
        final String rhs;

        Parsed(String source, String lhs, String rhs) {
            this.source = source;
            this.lhs = lhs;
            this.rhs = rhs;
        }
    }

    private final String[] axisNames;
    private final String timeName;
    private final List<String> declaredUnknowns;
    final List<String> unknowns = new ArrayList<String>();
    final Map<String, DerivRef> byKey = new LinkedHashMap<String, DerivRef>();
    final Map<String, DerivRef> byPlaceholder = new LinkedHashMap<String, DerivRef>();

    EquationAnalyzer(String[] axisNames, String timeName, List<String> declaredUnknowns) {
        this.axisNames = axisNames;
        this.timeName = timeName;
        this.declaredUnknowns = declaredUnknowns;
        this.unknowns.addAll(declaredUnknowns);
    }

    Parsed parse(String equation) {
        int eq = topLevelEquals(equation);
        String l;
        String r;
        if (eq < 0) {
            l = equation;
            r = "0";
        } else {
            l = equation.substring(0, eq);
            r = equation.substring(eq + 1);
        }
        if (l.trim().isEmpty() || r.trim().isEmpty()) {
            throw new PDEConfigurationException("Equation '" + equation + "' has an empty side");
        }
        return new Parsed(equation, substitute(l.trim()), substitute(r.trim()));
    }

    // ------------------------------------------------------------ substitution
    private String substitute(String text) {
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < text.length()) {
            if (text.startsWith("diff(", i) && (i == 0 || !isIdentChar(text.charAt(i - 1)))) {
                int close = matchParen(text, i + 4);
                if (close < 0) {
                    throw new PDEConfigurationException("Unbalanced parentheses in '" + text + "'");
                }
                String inner = text.substring(i + 5, close);
                DerivRef ref = parseCall(inner, text);
                if (ref != null) {
                    out.append(placeholderFor(ref));
                    i = close + 1;
                    continue;
                }
            }
            out.append(text.charAt(i));
            i++;
        }
        return out.toString();
    }

    private String placeholderFor(DerivRef ref) {
        DerivRef existing = byKey.get(ref.key());
        if (existing != null) {
            return existing.placeholder;
        }
        ref.index = byKey.size();
        ref.placeholder = (ref.isSpatial() ? "pdeD" : "pdeT") + ref.index;
        byKey.put(ref.key(), ref);
        byPlaceholder.put(ref.placeholder, ref);
        return ref.placeholder;
    }

    /** Parses the inside of a {@code diff(...)} call; returns null if the target is not an unknown function. */
    private DerivRef parseCall(String inner, String context) {
        List<String> args = splitTop(inner);
        if (args.size() < 2) {
            throw new PDEConfigurationException("diff() needs at least a function and a variable in '" + context + "'");
        }
        String target = args.get(0).trim();
        DerivRef base;
        if (target.startsWith("diff(") && matchParen(target, 4) == target.length() - 1) {
            base = parseCall(target.substring(5, target.length() - 1), context);
            if (base == null) {
                return null;
            }
        } else if (isIdentifier(target)) {
            if (!declaredUnknowns.isEmpty() && !declaredUnknowns.contains(target)) {
                // e.g. diff(c,x) on a parameter, or a typo: refuse rather than silently differentiate nothing
                throw new PDEConfigurationException("diff() target '" + target + "' is not a declared unknown "
                        + declaredUnknowns + " in '" + context + "'");
            }
            if (!unknowns.contains(target)) {
                unknowns.add(target);
            }
            base = new DerivRef(target);
        } else {
            return null; // symbolic differentiation of an expression: leave to ParserNG
        }
        int idx = 1;
        while (idx < args.size()) {
            String var = args.get(idx).trim();
            int order = 1;
            if (idx + 1 < args.size() && isInteger(args.get(idx + 1).trim())) {
                order = Integer.parseInt(args.get(idx + 1).trim());
                idx += 2;
            } else {
                idx += 1;
            }
            if (order < 1) {
                throw new PDEConfigurationException("Derivative order must be >= 1 in '" + context + "'");
            }
            if (var.equals(timeName)) {
                base.tOrd += order;
            } else {
                int a = -1;
                for (int q = 0; q < axisNames.length; q++) {
                    if (axisNames[q].equals(var)) {
                        a = q;
                    }
                }
                if (a < 0) {
                    throw new PDEConfigurationException("'" + var + "' is not an independent variable in '" + context
                            + "' (expected one of " + java.util.Arrays.toString(axisNames) + " or " + timeName + ")");
                }
                base.sp[a] += order;
            }
        }
        return base;
    }

    // ------------------------------------------------------------ text helpers
    /** Index of the single top-level assignment '=' (not part of ==, <=, >=, !=), or -1. */
    static int topLevelEquals(String s) {
        int depth = 0;
        int found = -1;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                depth++;
            } else if (c == ')' || c == ']' || c == '}') {
                depth--;
            } else if (c == '=' && depth == 0) {
                char prev = i > 0 ? s.charAt(i - 1) : ' ';
                char next = i + 1 < s.length() ? s.charAt(i + 1) : ' ';
                if (prev == '<' || prev == '>' || prev == '!' || prev == '=' || next == '=') {
                    continue;
                }
                if (found >= 0) {
                    throw new PDEConfigurationException("Equation '" + s + "' contains more than one '='");
                }
                found = i;
            }
        }
        return found;
    }

    static int matchParen(String s, int open) {
        int depth = 0;
        for (int i = open; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                depth++;
            } else if (c == ')') {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }
        return -1;
    }

    static List<String> splitTop(String s) {
        List<String> parts = new ArrayList<String>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                depth++;
            } else if (c == ')' || c == ']' || c == '}') {
                depth--;
            } else if (c == ',' && depth == 0) {
                parts.add(s.substring(start, i));
                start = i + 1;
            }
        }
        parts.add(s.substring(start));
        return parts;
    }

    static boolean isIdentChar(char c) {
        return Character.isLetterOrDigit(c) || c == '_';
    }

    static boolean isIdentifier(String s) {
        if (s.isEmpty() || !(Character.isLetter(s.charAt(0)) || s.charAt(0) == '_')) {
            return false;
        }
        for (int i = 1; i < s.length(); i++) {
            if (!isIdentChar(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    static boolean isInteger(String s) {
        if (s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            if (!Character.isDigit(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    /** Placeholder identifiers appearing in already-substituted text. */
    List<String> placeholdersIn(String text) {
        List<String> out = new ArrayList<String>();
        int i = 0;
        while (i < text.length()) {
            if ((text.startsWith("pdeD", i) || text.startsWith("pdeT", i)) && (i == 0 || !isIdentChar(text.charAt(i - 1)))) {
                int j = i + 4;
                while (j < text.length() && Character.isDigit(text.charAt(j))) {
                    j++;
                }
                String name = text.substring(i, j);
                if (j > i + 4 && byPlaceholder.containsKey(name) && !out.contains(name)) {
                    out.add(name);
                }
                i = j;
            } else {
                i++;
            }
        }
        return out;
    }
}
