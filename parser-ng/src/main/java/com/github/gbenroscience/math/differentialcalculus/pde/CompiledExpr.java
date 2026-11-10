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

import com.github.gbenroscience.math.differentialcalculus.autodiff.SystemAutoDiffEvaluator;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.Variable;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A ParserNG expression parsed and compiled ONCE (to the turbo evaluator) and
 * evaluated against flat {@code double[]} frames. Variable names are resolved
 * to frame slots at construction time; evaluation does no lookup, parsing or
 * allocation.
 */
final class CompiledExpr {

    final String text;
    final MathExpression me;
    private SystemAutoDiffEvaluator ad;
    private boolean adTried;
    final FastCompositeExpression fn;
    /** Variable names in the expression. */
    final String[] names;
    /** Frame slot of each name. */
    final int[] slots;
    /** Frame size. */
    final int frameSize;
    /** Initial values of all slots (constants such as pi, e). */
    final double[] template;

    CompiledExpr(String text) {
        this.text = text;
        MathExpression me;
        final Map<String, String> toInternal = new LinkedHashMap<String, String>();
        final String isolated = isolateSymbols(normalizeScientific(text), toInternal);
        final Map<String, String> toOriginal = new HashMap<String, String>();
        for (Map.Entry<String, String> en : toInternal.entrySet()) {
            toOriginal.put(en.getValue(), en.getKey());
        }
        try {
            me = new MathExpression(isolated);
        } catch (RuntimeException ex) {
            throw new PDEConfigurationException("Cannot parse expression '" + text + "': " + ex.getMessage(), ex);
        }
        this.me = me;
        if (!me.isScannedAndOptimized()) {
            throw new PDEConfigurationException("Invalid expression '" + text + "': " + describeErrors(me));
        }
        try {
            this.fn = me.compileTurbo();
        } catch (Throwable ex) {
            throw new PDEConfigurationException("Cannot compile expression '" + text + "': " + ex, ex);
        }
        String[] vn = me.getVariablesNames();
        this.names = new String[vn.length];
        this.slots = new int[vn.length];
        int size = Math.max(1, me.getSlots().length);
        for (int i = 0; i < vn.length; i++) {
            Variable v = me.getVariable(vn[i]);
            names[i] = toOriginal.containsKey(vn[i]) ? toOriginal.get(vn[i]) : vn[i];
            slots[i] = v.getFrameIndex();
            size = Math.max(size, slots[i] + 1);
        }
        this.frameSize = size;
        this.template = new double[size];
        for (int i = 0; i < vn.length; i++) {
            // renamed symbols start at 0; only the genuine constants (pi, e, ...) carry a value
            template[slots[i]] = toOriginal.containsKey(vn[i]) ? 0.0 : me.getVariable(vn[i]).getValue();
        }
    }

    /**
     * ParserNG reads a lower-case {@code e} as Euler's number, so the literal
     * {@code 1e-3} would silently mean {@code 1*e-3}. In PDE input a number
     * immediately followed by {@code e}/{@code E} and an exponent is always
     * scientific notation; it is rewritten to the upper-case form ParserNG
     * understands ({@code 1E-3}).
     */
    static String normalizeScientific(String s) {
        StringBuilder out = null;
        int n = s.length();
        int i = 0;
        while (i < n) {
            char c = s.charAt(i);
            boolean startsNumber = (Character.isDigit(c) || (c == '.' && i + 1 < n && Character.isDigit(s.charAt(i + 1))))
                    && (i == 0 || !(Character.isLetterOrDigit(s.charAt(i - 1)) || s.charAt(i - 1) == '_' || s.charAt(i - 1) == '.'));
            if (!startsNumber) {
                if (out != null) {
                    out.append(c);
                }
                i++;
                continue;
            }
            int j = i;
            while (j < n && Character.isDigit(s.charAt(j))) {
                j++;
            }
            if (j < n && s.charAt(j) == '.') {
                j++;
                while (j < n && Character.isDigit(s.charAt(j))) {
                    j++;
                }
            }
            int mant = j;
            boolean sci = false;
            if (j < n && (s.charAt(j) == 'e' || s.charAt(j) == 'E')) {
                int k = j + 1;
                if (k < n && (s.charAt(k) == '+' || s.charAt(k) == '-')) {
                    k++;
                }
                if (k < n && Character.isDigit(s.charAt(k))) {
                    while (k < n && Character.isDigit(s.charAt(k))) {
                        k++;
                    }
                    if (k >= n || !(Character.isLetter(s.charAt(k)) || s.charAt(k) == '_')) {
                        sci = true;
                        if (out == null) {
                            out = new StringBuilder(s.substring(0, i));
                        }
                        out.append(s, i, mant).append('E').append(s, j + 1, k);
                        i = k;
                        continue;
                    }
                }
            }
            if (out != null) {
                out.append(s, i, mant);
            }
            i = mant;
        }
        return out == null ? s : out.toString();
    }

    private static boolean isConstantName(String n) {
        return n.equals("pi") || n.equals("PI") || n.equals("\u03c0") || n.equals("e") || n.equals("\u03c6");
    }

    /**
     * ParserNG keeps variables and functions in process-wide registries, so a
     * symbol such as {@code g}, {@code v} or {@code x} that the application
     * defined elsewhere would leak into (and silently change) a PDE
     * expression. Every symbol that is a VARIABLE here (an identifier not
     * followed by a parenthesis, and not a built-in constant) is therefore
     * renamed to a private name ({@code pdeQ0}, {@code pdeQ1}, ...) before the
     * text reaches the parser; {@link #names} reports the original names.
     */
    static String isolateSymbols(String s, Map<String, String> toInternal) {
        StringBuilder out = new StringBuilder(s.length() + 16);
        int n = s.length();
        int i = 0;
        boolean prevWasNumber = false;
        while (i < n) {
            char c = s.charAt(i);
            if (Character.isDigit(c) || (c == '.' && i + 1 < n && Character.isDigit(s.charAt(i + 1)))) {
                int j = i;
                while (j < n && (Character.isDigit(s.charAt(j)) || s.charAt(j) == '.')) {
                    j++;
                }
                if (j < n && s.charAt(j) == 'E') {
                    int k = j + 1;
                    if (k < n && (s.charAt(k) == '+' || s.charAt(k) == '-')) {
                        k++;
                    }
                    if (k < n && Character.isDigit(s.charAt(k))) {
                        while (k < n && Character.isDigit(s.charAt(k))) {
                            k++;
                        }
                        j = k;
                    }
                }
                out.append(s, i, j);
                i = j;
                prevWasNumber = true;
                continue;
            }
            if (Character.isLetter(c) || c == '_') {
                int j = i;
                while (j < n && (Character.isLetterOrDigit(s.charAt(j)) || s.charAt(j) == '_')) {
                    j++;
                }
                String id = s.substring(i, j);
                int k = j;
                while (k < n && s.charAt(k) == ' ') {
                    k++;
                }
                boolean isCall = k < n && s.charAt(k) == '(';
                if (isCall || isConstantName(id)) {
                    out.append(id);
                } else {
                    String internal = toInternal.get(id);
                    if (internal == null) {
                        internal = "pdeQ" + toInternal.size();
                        toInternal.put(id, internal);
                    }
                    if (prevWasNumber) {
                        out.append('*');
                    }
                    out.append(internal);
                }
                i = j;
                prevWasNumber = false;
                continue;
            }
            out.append(c);
            if (c != ' ') {
                prevWasNumber = false;
            }
            i++;
        }
        return out.toString();
    }

    private static String describeErrors(MathExpression me) {
        try {
            String s = String.valueOf(me.getErrorLog());
            return s == null || s.isEmpty() ? "syntax error" : s;
        } catch (RuntimeException ex) {
            return "syntax error";
        }
    }

    /**
     * ParserNG's forward-mode automatic differentiator (frame-based, GC-free,
     * thread-safe) for this expression, or null if the expression uses an
     * operator the differentiator does not support (e.g. comparisons inside
     * {@code if}). Built lazily, once.
     */
    synchronized SystemAutoDiffEvaluator ad() {
        if (!adTried) {
            adTried = true;
            try {
                ad = new SystemAutoDiffEvaluator(me, 2);
            } catch (RuntimeException ex) {
                ad = null;
            }
        }
        return ad;
    }

    /** @return slot of the named variable, or -1 if the expression does not use it. */
    int slotOf(String name) {
        for (int i = 0; i < names.length; i++) {
            if (names[i].equals(name)) {
                return slots[i];
            }
        }
        return -1;
    }

    /** @return a fresh frame initialised with the constants. */
    double[] newFrame() {
        return template.clone();
    }

    double eval(double[] frame) {
        return fn.applyScalar(frame);
    }
}
