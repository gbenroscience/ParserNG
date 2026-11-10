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

import com.github.gbenroscience.util.FunctionManager;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The MATHEMATICAL description of a PDE or PDE system, as recognised by
 * {@code MathExpression}: the equations, the unknown functions, and the
 * independent variables that appear in derivatives. It carries no grid, no
 * discretization and no solver choice, so one model can be solved with
 * different numerical methods; those belong to the numerical problem
 * ({@link PDEProblem}) that {@link #toProblem()} starts.
 * <p>
 * Obtain one from an expression:
 * <pre>{@code
 * MathExpression e = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");
 * PDEModel model = e.getPDEModel();            // or e.asPDE() for a ready problem builder
 * }</pre>
 * Recognised notation (see also {@link #classifyStatement(String)}): derivatives
 * {@code diff(u,t)}, {@code diff(u,x,2)}, {@code diff(u,x,y)},
 * {@code diff(diff(u,x),y)}; equations {@code LHS = RHS}; systems as several
 * statements separated by {@code ;}, several arguments of
 * {@code pde(eq1, eq2)}, or the array form {@code pde(@(2)("eq1","eq2"))}.
 */
public final class PDEModel implements Serializable {

    private static final long serialVersionUID = 1L;

    /** How a statement is read by the expression processor. */
    public enum StatementKind {
        /** Not a PDE: handled by the ordinary scalar / assignment machinery. */
        NOT_PDE,
        /** A bare equation such as {@code diff(u,t) = alpha*diff(u,x,2)}. */
        PDE_EQUATION,
        /** An explicit {@code pde(...)} command. */
        PDE_COMMAND
    }

    /** Result of inspecting the inside of one {@code diff(...)} call. */
    public static final class DerivativeCheck implements Serializable {

        private static final long serialVersionUID = 1L;
        /** True when the call has PDE shape: an unknown function differentiated with respect to named variables. */
        public final boolean pdeStyle;
        /** Non-null when the call is PDE-shaped but malformed. */
        public final String error;
        /** The unknown function being differentiated (null if not PDE-shaped). */
        public final String unknown;
        /** The variables differentiated with respect to, in order of appearance. */
        public final List<String> variables;

        DerivativeCheck(boolean pdeStyle, String error, String unknown, List<String> variables) {
            this.pdeStyle = pdeStyle;
            this.error = error;
            this.unknown = unknown;
            this.variables = variables;
        }
    }

    private final List<String> equations;
    private final List<String> unknowns;
    private final List<String> independents;
    private final Map<String, Double> boundParameters;
    private final PDEType type;
    private final ConservationForm conservation;
    private final String conservationReason;

    private PDEModel(List<String> equations, List<String> unknowns, List<String> independents,
            Map<String, Double> params, PDEType type, ConservationForm conservation, String reason) {
        this.equations = Collections.unmodifiableList(new ArrayList<String>(equations));
        this.unknowns = Collections.unmodifiableList(new ArrayList<String>(unknowns));
        this.independents = Collections.unmodifiableList(new ArrayList<String>(independents));
        this.boundParameters = Collections.unmodifiableMap(new LinkedHashMap<String, Double>(params));
        this.type = type;
        this.conservation = conservation;
        this.conservationReason = reason;
    }

    // ------------------------------------------------------------------ factory
    /**
     * Builds a model from equation texts.
     *
     * @param equations one or more {@code LHS = RHS} strings (or {@code expr}, meaning {@code expr = 0})
     * @param params constants defined next to the equations (for example by {@code alpha=0.1;});
     * only those that actually occur in an equation are kept
     * @throws PDEConfigurationException if no equation contains a PDE derivative, or an equation is malformed
     */
    public static PDEModel of(List<String> equations, Map<String, Double> params) {
        if (equations == null || equations.isEmpty()) {
            throw new PDEConfigurationException("A PDE model needs at least one equation");
        }
        List<String> eqs = new ArrayList<String>();
        Set<String> unk = new LinkedHashSet<String>();
        Set<String> ind = new LinkedHashSet<String>();
        int maxTime = 0;
        int maxSpace = 0;
        boolean anySpatial = false;
        for (String raw : equations) {
            String eq = stripQuotes(raw.trim());
            if (eq.isEmpty()) {
                throw new PDEConfigurationException("Empty equation in PDE model");
            }
            EquationAnalyzer.topLevelEquals(eq); // throws if more than one '='
            List<DerivativeCheck> calls = findDerivativeCalls(eq, true);
            boolean hasPde = false;
            for (DerivativeCheck c : calls) {
                if (!c.pdeStyle) {
                    continue;
                }
                if (c.error != null) {
                    throw new PDEConfigurationException(c.error + " in '" + eq + "'");
                }
                hasPde = true;
                unk.add(c.unknown);
                int order = 0;
                for (String v : c.variables) {
                    ind.add(v);
                    order++;
                    if (v.equals(PDEProblem.TIME)) {
                        maxTime = Math.max(maxTime, countOf(c.variables, v));
                    } else {
                        anySpatial = true;
                        maxSpace = Math.max(maxSpace, countOf(c.variables, v));
                    }
                }
            }
            if (!hasPde) {
                throw new PDEConfigurationException("Equation '" + eq + "' contains no derivative of an unknown function; "
                        + "write derivatives as diff(u,t), diff(u,x,2), diff(u,x,y)");
            }
            eqs.add(eq);
        }
        PDEType t;
        if (maxTime == 0) {
            t = PDEType.ELLIPTIC;
        } else if (maxTime >= 2) {
            t = PDEType.HYPERBOLIC;
        } else {
            t = maxSpace >= 2 || !anySpatial ? PDEType.PARABOLIC : PDEType.HYPERBOLIC;
        }
        // keep only the constants the equations really use
        Map<String, Double> used = new LinkedHashMap<String, Double>();
        if (params != null) {
            Set<String> idents = new LinkedHashSet<String>();
            for (String e : eqs) {
                collectIdentifiers(e, idents);
            }
            for (Map.Entry<String, Double> en : params.entrySet()) {
                if (idents.contains(en.getKey()) && !unk.contains(en.getKey()) && !ind.contains(en.getKey())) {
                    used.put(en.getKey(), en.getValue());
                }
            }
        }
        // conservation form u_t + div F(u) = S ?  (then the finite-volume solver can take the model)
        ConservationForm cf = null;
        String why = null;
        try {
            cf = parseConservation(eqs, new ArrayList<String>(unk));
        } catch (PDEConfigurationException ex) {
            why = ex.getMessage();
        }
        if (cf != null) {
            t = PDEType.HYPERBOLIC;
            for (String ax : cf.axes) {
                ind.add(ax);
            }
        }
        return new PDEModel(eqs, new ArrayList<String>(unk), new ArrayList<String>(ind), used, t, cf, why);
    }

    /** Convenience: a model from equation texts with no bound constants. */
    public static PDEModel of(String... equations) {
        return of(java.util.Arrays.asList(equations), null);
    }

    // ------------------------------------------------------------------ accessors
    /** @return the equations, in order. */
    public List<String> getEquations() {
        return equations;
    }

    /** @return the unknown functions found in derivatives, in order of first appearance. */
    public List<String> getUnknowns() {
        return unknowns;
    }

    /** @return the variables that occur in derivatives (e.g. x, y, t), in order of first appearance. */
    public List<String> getIndependentVariables() {
        return independents;
    }

    /** @return constants defined alongside the equations that the equations use (e.g. {@code alpha} in {@code alpha=0.1; ...}). */
    public Map<String, Double> getBoundParameters() {
        return boundParameters;
    }

    /**
     * @return the conservation-law structure {@code u_t + sum_d d/dx_d F_d(u) = S} of this model, or null if the
     * equations are not in that form (see {@link #getConservationFormReason()}). Flux terms are written as
     * {@code diff(F(u), x)} with an EXPRESSION of the unknowns as the first argument, e.g.
     * {@code diff(u,t) + diff(u^2/2, x) = 0} (Burgers) or {@code diff(u,t) + diff(a*u, x) = 0} (advection).
     */
    public ConservationForm getConservationForm() {
        return conservation;
    }

    /** @return why the model is not in conservation form (null if it is). */
    public String getConservationFormReason() {
        return conservationReason;
    }

    /** @return true if the model is a system of conservation laws that the finite-volume solver can take. */
    public boolean isConservationLaw() {
        return conservation != null;
    }

    /**
     * Starts a finite-volume problem for a model in conservation form: the law is built from the flux
     * expressions (wave speeds by automatic differentiation) and constants bound next to the equations.
     * Cells, initial data, reconstruction, numerical flux and boundaries are then added to the returned builder.
     *
     * @throws PDEConfigurationException if the model is not a conservation law
     */
    public FiniteVolumeSolver.Builder toFiniteVolume() {
        if (conservation == null) {
            throw new PDEConfigurationException("This PDE is not in conservation form" + (conservationReason == null ? "" : ": " + conservationReason)
                    + ". Conservation laws are written diff(u,t) + diff(F(u),x) = S, e.g. Burgers: diff(u,t) + diff(u^2/2,x) = 0");
        }
        final ConservationForm cf = conservation;
        final Map<String, Double> bound = boundParameters;
        FiniteVolumeSolver.Builder b = FiniteVolumeSolver.builder().lawFactory(new FiniteVolumeSolver.LawFactory() {
            @Override
            public ConservationLaw create(Map<String, Double> params) {
                // constants bound next to the equations, overridden by those given to the builder
                Map<String, Double> all = new LinkedHashMap<String, Double>(bound);
                all.putAll(params);
                return new ExpressionLaw(cf.unknowns.toArray(new String[0]), cf.axes.toArray(new String[0]),
                        cf.flux, cf.hasSource() ? cf.source : null, all);
            }
        });
        for (Map.Entry<String, Double> en : bound.entrySet()) {
            b.param(en.getKey(), en.getValue());
        }
        return b;
    }

    /** @return true for more than one equation. */
    public boolean isSystem() {
        return equations.size() > 1;
    }

    /** @return the coarse classification (elliptic / parabolic / hyperbolic) used for solver defaults. */
    public PDEType getType() {
        return type;
    }

    /** @return the canonical command form, e.g. {@code pde(diff(u,t) = alpha*diff(u,x,2))}. */
    public String toCanonicalString() {
        StringBuilder sb = new StringBuilder("pde(");
        for (int i = 0; i < equations.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(equations.get(i));
        }
        return sb.append(')').toString();
    }

    /**
     * Starts the numerical problem for this model: the equations plus any
     * constants defined next to them. Domain, conditions and solver choices are
     * then added to the returned builder.
     */
    public PDEProblem.Builder toProblem() {
        PDEProblem.Builder b = PDEProblem.builder(equations.toArray(new String[0]));
        for (Map.Entry<String, Double> en : boundParameters.entrySet()) {
            b.param(en.getKey(), en.getValue());
        }
        return b;
    }

    @Override
    public String toString() {
        return "PDEModel{" + type + ", unknowns=" + unknowns + ", variables=" + independents + ", " + toCanonicalString() + "}";
    }

    // ------------------------------------------------------------------ syntax recognition
    /**
     * Decides how one statement (as split on {@code ;}) is to be read.
     * <ul>
     * <li>{@code pde(...)} at the root: {@link StatementKind#PDE_COMMAND}.</li>
     * <li>A single top-level {@code =} whose left side is NOT an assignment
     * target (not {@code name}, not {@code name(params)} with a user name) and
     * which contains a PDE-shaped derivative: {@link StatementKind#PDE_EQUATION}.
     * A bare {@code u = diff(v,x)} stays an ordinary assignment (use
     * {@code pde(...)} for that), and {@code diff(F,x,2)} with a DEFINED
     * function {@code F} keeps its ordinary differentiation meaning.</li>
     * <li>Everything else, including {@code ==}, {@code <=} and any scalar
     * expression: {@link StatementKind#NOT_PDE}.</li>
     * </ul>
     */
    public static StatementKind classifyStatement(String code) {
        if (code == null) {
            return StatementKind.NOT_PDE;
        }
        String s = stripOuterParens(code.trim());
        if (isPdeCommand(s) || containsPdeCall(s)) {
            // a pde(...) call anywhere: let the expression validator judge it (root-only rule, balanced parentheses, ...)
            return StatementKind.PDE_COMMAND;
        }
        int eq;
        try {
            eq = EquationAnalyzer.topLevelEquals(s);
        } catch (PDEConfigurationException ex) {
            // several '=' : if it still looks like a PDE equation report it as one so the user gets a PDE diagnostic
            return containsPdeDerivative(s) && !isAssignmentTarget(s.substring(0, s.indexOf('='))) ? StatementKind.PDE_EQUATION : StatementKind.NOT_PDE;
        }
        if (eq <= 0 || eq >= s.length() - 1) {
            return StatementKind.NOT_PDE;
        }
        String lhs = s.substring(0, eq).trim();
        if (isAssignmentTarget(lhs)) {
            return StatementKind.NOT_PDE;
        }
        // 'diff' is an inbuilt name and can never be the name of a function being defined, so an equation whose
        // left side starts with diff(...) is a PDE attempt even if the derivative itself turns out to be malformed
        // (the PDE validator then explains what is wrong instead of the function-definition code)
        if (lhs.startsWith("diff") && lhs.substring(4).trim().startsWith("(")) {
            return StatementKind.PDE_EQUATION;
        }
        return containsPdeDerivative(s) ? StatementKind.PDE_EQUATION : StatementKind.NOT_PDE;
    }

    /**
     * @param s an equation text
     * @return the index of its single top-level assignment-style {@code =} (not part of
     * {@code ==}, {@code <=}, {@code >=}, {@code !=}), or -1 if there is none
     * @throws PDEConfigurationException if there is more than one
     */
    public static int topLevelEquals(String s) {
        return EquationAnalyzer.topLevelEquals(s);
    }

    /** @return {@code s} without any redundant outer grouping parentheses. */
    public static String unwrap(String s) {
        return stripOuterParens(s.trim());
    }

    /** @return the canonical {@code pde(eq1, eq2, ...)} text for the given equations. */
    public static String canonical(List<String> equations) {
        StringBuilder sb = new StringBuilder("pde(");
        for (int i = 0; i < equations.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(equations.get(i));
        }
        return sb.append(')').toString();
    }

    /** @return true if {@code s} is exactly one {@code pde(...)} call. */
    public static boolean isPdeCommand(String s) {
        String t = s.trim();
        if (!t.startsWith("pde") || t.length() < 5) {
            return false;
        }
        int i = 3;
        while (i < t.length() && Character.isWhitespace(t.charAt(i))) {
            i++;
        }
        if (i >= t.length() || t.charAt(i) != '(') {
            return false;
        }
        return EquationAnalyzer.matchParen(t, i) == t.length() - 1;
    }

    /** @return the equation texts inside a {@code pde(...)} command (handles several arguments and the {@code @(n)("..","..")} array form). */
    public static List<String> equationsOfCommand(String command) {
        String t = command.trim();
        int open = t.indexOf('(');
        String inner = t.substring(open + 1, t.length() - 1).trim();
        List<String> out = new ArrayList<String>();
        if (inner.startsWith("@(")) {
            out.addAll(PDEProblem.parseArraySyntax(inner));
            return out;
        }
        for (String part : EquationAnalyzer.splitTop(inner)) {
            String p = stripQuotes(part.trim());
            if (!p.isEmpty()) {
                out.add(p);
            }
        }
        return out;
    }

    private static boolean isAssignmentTarget(String lhs) {
        String l = lhs.trim();
        if (EquationAnalyzer.isIdentifier(l)) {
            return true;                         // x = ...
        }
        int p = l.indexOf('(');
        if (p > 0 && l.endsWith(")") && EquationAnalyzer.matchParen(l, p) == l.length() - 1) {
            String name = l.substring(0, p).trim();
            if (EquationAnalyzer.isIdentifier(name) && !com.github.gbenroscience.parser.methods.Declarations.isInbuiltMethodName(name)) {
                return true;                     // f(x,y) = ...   (a user function definition)
            }
        }
        return false;
    }

    /** @return true if {@code s} contains a call to the {@code pde} command (a name followed by an opening parenthesis). */
    static boolean containsPdeCall(String s) {
        int i = s.indexOf("pde");
        while (i >= 0) {
            boolean startOk = i == 0 || !EquationAnalyzer.isIdentChar(s.charAt(i - 1));
            int j = i + 3;
            while (j < s.length() && Character.isWhitespace(s.charAt(j))) {
                j++;
            }
            if (startOk && j < s.length() && s.charAt(j) == '(' && (i + 3 == j || true)) {
                // the name must be exactly 'pde': the character right after must not continue an identifier
                if (i + 3 >= s.length() || !EquationAnalyzer.isIdentChar(s.charAt(i + 3))) {
                    return true;
                }
            }
            i = s.indexOf("pde", i + 1);
        }
        return false;
    }

    private static boolean containsPdeDerivative(String s) {
        for (DerivativeCheck c : findDerivativeCalls(s, false)) {
            if (c.pdeStyle) {
                return true;
            }
        }
        return false;
    }

    /** All {@code diff(...)} calls (outermost) in {@code s}, each classified. */
    static List<DerivativeCheck> findDerivativeCalls(String s, boolean explicitPde) {
        List<DerivativeCheck> out = new ArrayList<DerivativeCheck>();
        int i = 0;
        while (i < s.length()) {
            if (s.startsWith("diff", i) && (i == 0 || !EquationAnalyzer.isIdentChar(s.charAt(i - 1)))) {
                int j = i + 4;
                while (j < s.length() && Character.isWhitespace(s.charAt(j))) {
                    j++;
                }
                if (j < s.length() && s.charAt(j) == '(') {
                    int close = EquationAnalyzer.matchParen(s, j);
                    if (close > 0) {
                        out.add(checkDerivativeCall(s.substring(j + 1, close), explicitPde));
                        i = close + 1;
                        continue;
                    }
                }
            }
            i++;
        }
        return out;
    }

    /**
     * Inspects the text between the parentheses of a {@code diff(...)} call.
     * PDE-shaped means: the first argument is an unknown function name (a
     * bare identifier that is not a defined ParserNG function) or a nested
     * PDE-shaped {@code diff(...)}, and the second argument is a variable name
     * (not a number). That separates {@code diff(u,x,2)} from ParserNG's own
     * {@code diff(F,2)} / {@code diff(F,x,n)} applied to a defined function.
     */
    public static DerivativeCheck checkDerivativeCall(String inner) {
        return checkDerivativeCall(inner, false);
    }

    /**
     * @param explicitPde true inside an explicit {@code pde(...)} command, where
     * the intent is unambiguous: a bare identifier is then always an unknown
     * function, even if the application happens to have defined a ParserNG
     * function of the same name. In a bare statement (false) a name that IS a
     * defined function keeps ParserNG's own meaning of {@code diff(F,...)}.
     */
    public static DerivativeCheck checkDerivativeCall(String inner, boolean explicitPde) {
        List<String> args = EquationAnalyzer.splitTop(inner);
        if (args.size() < 2) {
            return new DerivativeCheck(false, null, null, null);
        }
        String target = args.get(0).trim();
        String unknown;
        List<String> vars = new ArrayList<String>();
        if (target.startsWith("diff") && target.endsWith(")")) {
            int p = target.indexOf('(');
            if (p > 0 && target.substring(4, p).trim().isEmpty() && EquationAnalyzer.matchParen(target, p) == target.length() - 1) {
                DerivativeCheck nested = checkDerivativeCall(target.substring(p + 1, target.length() - 1), explicitPde);
                if (!nested.pdeStyle) {
                    return nested;
                }
                if (nested.error != null) {
                    return nested;
                }
                unknown = nested.unknown;
                vars.addAll(nested.variables);
            } else {
                return new DerivativeCheck(false, null, null, null);
            }
        } else if (EquationAnalyzer.isIdentifier(target)) {
            if (!explicitPde && FunctionManager.containsAny(target)) {
                return new DerivativeCheck(false, null, null, null);   // ParserNG's own diff(F, ...)
            }
            String second = args.get(1).trim();
            if (!EquationAnalyzer.isIdentifier(second)) {
                return new DerivativeCheck(false, null, null, null);   // diff(F, 2): ordinary
            }
            unknown = target;
        } else {
            return new DerivativeCheck(false, null, null, null);       // diff(x^2, x): symbolic differentiation
        }
        int idx = 1;
        while (idx < args.size()) {
            String var = args.get(idx).trim();
            if (!EquationAnalyzer.isIdentifier(var)) {
                return new DerivativeCheck(true, "diff(" + inner.trim() + "): expected a variable name but found '" + var + "'", unknown, vars);
            }
            int order = 1;
            if (idx + 1 < args.size() && EquationAnalyzer.isInteger(args.get(idx + 1).trim())) {
                order = Integer.parseInt(args.get(idx + 1).trim());
                idx += 2;
            } else {
                idx += 1;
            }
            if (order < 1 || order > 8) {
                return new DerivativeCheck(true, "diff(" + inner.trim() + "): derivative order must be between 1 and 8, got " + order, unknown, vars);
            }
            for (int q = 0; q < order; q++) {
                vars.add(var);
            }
        }
        return new DerivativeCheck(true, null, unknown, vars);
    }

    /**
     * Replaces every PDE-shaped {@code diff(...)} call in {@code s} by a plain
     * placeholder identifier ({@code pdeDeriv0}, {@code pdeDeriv1}, ...) so that
     * what remains is an ordinary scalar expression that the expression
     * validator can check. Calls that are not PDE-shaped (ParserNG's own
     * differentiation) are left untouched.
     *
     * @param count receives the number of PDE derivatives found (index 0)
     * @throws PDEConfigurationException if a PDE-shaped call is malformed
     */
    public static String isolateDerivatives(String s, int[] count) {
        StringBuilder out = new StringBuilder();
        int n = 0;
        int i = 0;
        while (i < s.length()) {
            if (s.startsWith("diff", i) && (i == 0 || !EquationAnalyzer.isIdentChar(s.charAt(i - 1)))) {
                int j = i + 4;
                while (j < s.length() && Character.isWhitespace(s.charAt(j))) {
                    j++;
                }
                if (j < s.length() && s.charAt(j) == '(') {
                    int close = EquationAnalyzer.matchParen(s, j);
                    if (close > 0) {
                        DerivativeCheck c = checkDerivativeCall(s.substring(j + 1, close), true);
                        if (c.pdeStyle) {
                            if (c.error != null) {
                                throw new PDEConfigurationException(c.error);
                            }
                            out.append("pdeDeriv").append(n++);
                            i = close + 1;
                            continue;
                        }
                    }
                }
            }
            out.append(s.charAt(i));
            i++;
        }
        count[0] = n;
        return out.toString();
    }

    // ------------------------------------------------------------------ conservation form
    /** The structure of a system of conservation laws recognised in a model. */
    public static final class ConservationForm implements Serializable {

        private static final long serialVersionUID = 1L;
        /** Conserved unknowns, in equation order. */
        public final List<String> unknowns;
        /** Space variables that carry fluxes (x, y, z order when named so), one per direction. */
        public final List<String> axes;
        /** {@code flux[direction][unknown]}: expression of the flux of the unknown along the direction. */
        public final String[][] flux;
        /** {@code source[unknown]}: right-hand-side source expression ("0" if none). */
        public final String[] source;

        ConservationForm(List<String> unknowns, List<String> axes, String[][] flux, String[] source) {
            this.unknowns = Collections.unmodifiableList(new ArrayList<String>(unknowns));
            this.axes = Collections.unmodifiableList(new ArrayList<String>(axes));
            this.flux = flux;
            this.source = source;
        }

        public boolean hasSource() {
            for (String s : source) {
                if (!"0".equals(s)) {
                    return true;
                }
            }
            return false;
        }
    }

    /** Splits {@code s} at its top-level {@code +} / {@code -} into signed terms (index 0 of each pair is "+" or "-"). */
    static List<String[]> splitAdditive(String s) {
        List<String[]> out = new ArrayList<String[]>();
        int depth = 0;
        int start = 0;
        String sign = "+";
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                depth++;
            } else if (c == ')' || c == ']' || c == '}') {
                depth--;
            } else if ((c == '+' || c == '-') && depth == 0) {
                int p = i - 1;
                while (p >= 0 && Character.isWhitespace(s.charAt(p))) {
                    p--;
                }
                if (p < 0 && out.isEmpty() && s.substring(0, i).trim().isEmpty()) {
                    // a sign at the very start of the expression is the sign of the first term, not a split point
                    sign = String.valueOf(c);
                    start = i + 1;
                    continue;
                }
                boolean unary = p < 0 || "(,*/^+-=<>".indexOf(s.charAt(p)) >= 0;
                boolean exponent = p >= 1 && (s.charAt(p) == 'E' || s.charAt(p) == 'e')
                        && (Character.isDigit(s.charAt(p - 1)) || s.charAt(p - 1) == '.') && isNumberStart(s, p - 1);
                if (unary || exponent) {
                    continue;
                }
                String piece = s.substring(start, i).trim();
                if (!piece.isEmpty()) {
                    out.add(new String[]{sign, piece});
                }
                sign = String.valueOf(c);
                start = i + 1;
            }
        }
        String last = s.substring(start).trim();
        if (!last.isEmpty()) {
            out.add(new String[]{sign, last});
        }
        return out;
    }

    private static boolean isNumberStart(String s, int i) {
        while (i >= 0 && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '.')) {
            i--;
        }
        return i < 0 || !EquationAnalyzer.isIdentChar(s.charAt(i));
    }

    private static String stripDiffCall(String term) {
        String t = term.trim();
        if (!t.startsWith("diff")) {
            return null;
        }
        int p = t.indexOf('(');
        if (p < 0 || !t.substring(4, p).trim().isEmpty() || EquationAnalyzer.matchParen(t, p) != t.length() - 1) {
            return null;
        }
        return t.substring(p + 1, t.length() - 1);
    }

    private static boolean mentionsAny(String expr, List<String> names) {
        Set<String> ids = new LinkedHashSet<String>();
        collectIdentifiers(expr, ids);
        for (String n : names) {
            if (ids.contains(n)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Recognises {@code diff(u,t) + sum diff(F(u), x_d) = S} equation by equation. Anything else (a
     * non-conservative derivative such as {@code diff(u,x)}, a coefficient on the time derivative, a second
     * derivative) is reported through {@link PDEConfigurationException} with the reason.
     */
    static ConservationForm parseConservation(List<String> eqs, List<String> unknownList) {
        int nc = unknownList.size();
        if (eqs.size() != nc) {
            throw new PDEConfigurationException("a conservation-law system needs as many equations as unknowns");
        }
        List<String> axes = new ArrayList<String>();
        // first pass to find the axes
        String[][] fluxTmp = new String[3][nc];
        String[] src = new String[nc];
        boolean[] seen = new boolean[nc];
        List<List<String[]>> fluxTerms = new ArrayList<List<String[]>>();    // per equation: {axis, expr(with sign)}
        for (int e = 0; e < eqs.size(); e++) {
            String eq = eqs.get(e);
            int idx = EquationAnalyzer.topLevelEquals(eq);
            String lhs = idx < 0 ? eq : eq.substring(0, idx);
            String rhs = idx < 0 ? "0" : eq.substring(idx + 1);
            List<String[]> terms = new ArrayList<String[]>();
            for (String[] t : splitAdditive(lhs)) {
                terms.add(t);
            }
            for (String[] t : splitAdditive(rhs)) {
                terms.add(new String[]{t[0].equals("+") ? "-" : "+", t[1]});
            }
            int timeIdx = -1;
            String uName = null;
            int flip = 1;
            for (int i = 0; i < terms.size(); i++) {
                String inner = stripDiffCall(terms.get(i)[1]);
                if (inner == null) {
                    continue;
                }
                DerivativeCheck c = checkDerivativeCall(inner, true);
                if (c.pdeStyle && c.error == null && c.variables.size() == 1 && c.variables.get(0).equals(PDEProblem.TIME)) {
                    if (timeIdx >= 0) {
                        throw new PDEConfigurationException("equation '" + eq + "' has more than one time derivative");
                    }
                    timeIdx = i;
                    uName = c.unknown;
                    flip = terms.get(i)[0].equals("+") ? 1 : -1;
                }
            }
            if (timeIdx < 0) {
                throw new PDEConfigurationException("equation '" + eq + "' has no first-order time derivative diff(u,t)");
            }
            int ui = unknownList.indexOf(uName);
            if (ui < 0 || seen[ui]) {
                throw new PDEConfigurationException("each unknown needs exactly one equation (problem with '" + uName + "')");
            }
            seen[ui] = true;
            StringBuilder other = new StringBuilder();
            List<String[]> fl = new ArrayList<String[]>();
            for (int i = 0; i < terms.size(); i++) {
                if (i == timeIdx) {
                    continue;
                }
                String sign = flip == 1 ? terms.get(i)[0] : (terms.get(i)[0].equals("+") ? "-" : "+");
                String text = terms.get(i)[1];
                String inner = stripDiffCall(text);
                if (inner != null) {
                    List<String> args = EquationAnalyzer.splitTop(inner);
                    if (args.size() != 2) {
                        throw new PDEConfigurationException("'" + text + "': a flux term is diff(F, x) with exactly two arguments");
                    }
                    String f = args.get(0).trim();
                    String ax = args.get(1).trim();
                    if (EquationAnalyzer.isIdentifier(f) && unknownList.contains(f)) {
                        throw new PDEConfigurationException("'" + text + "' is a non-conservative derivative of the unknown; "
                                + "write the flux inside the derivative, e.g. diff(a*u, x), or use the finite-difference solver");
                    }
                    if (!EquationAnalyzer.isIdentifier(ax) || ax.equals(PDEProblem.TIME) || unknownList.contains(ax)) {
                        throw new PDEConfigurationException("'" + text + "': the second argument must be a space variable");
                    }
                    if (!mentionsAny(f, unknownList)) {
                        throw new PDEConfigurationException("'" + text + "': a flux must depend on the unknowns " + unknownList);
                    }
                    if (!axes.contains(ax)) {
                        axes.add(ax);
                    }
                    fl.add(new String[]{ax, (sign.equals("-") ? "-" : "") + "(" + f + ")"});
                } else {
                    if (text.contains("diff(")) {
                        throw new PDEConfigurationException("'" + text + "': derivatives must appear as diff(u,t) or as flux terms diff(F(u),x)");
                    }
                    // the equation reads  u_t + div F + other = 0  =>  S = -sum(other)
                    other.append(sign.equals("+") ? "-" : "+").append("(").append(text).append(")");
                }
            }
            fluxTerms.add(fl);
            src[ui] = other.length() == 0 ? "0" : other.toString();
        }
        if (axes.isEmpty()) {
            throw new PDEConfigurationException("no flux term diff(F(u), x) found");
        }
        // order the axes x, y, z when so named
        List<String> ordered = new ArrayList<String>();
        for (String s : new String[]{"x", "y", "z"}) {
            if (axes.contains(s)) {
                ordered.add(s);
            }
        }
        for (String s : axes) {
            if (!ordered.contains(s)) {
                ordered.add(s);
            }
        }
        if (ordered.size() > 3) {
            throw new PDEConfigurationException("at most three space variables are supported");
        }
        String[][] flux = new String[ordered.size()][nc];
        for (String[] row : flux) {
            Arrays.fill(row, "0");
        }
        // equation e belongs to the unknown whose diff(u,t) it carries: recompute that mapping
        for (int e = 0; e < eqs.size(); e++) {
            String eq = eqs.get(e);
            int ui = -1;
            for (DerivativeCheck c : findDerivativeCalls(eq, true)) {
                if (c.pdeStyle && c.error == null && c.variables.size() == 1 && c.variables.get(0).equals(PDEProblem.TIME)) {
                    ui = unknownList.indexOf(c.unknown);
                }
            }
            for (String[] ft : fluxTerms.get(e)) {
                int d = ordered.indexOf(ft[0]);
                flux[d][ui] = "0".equals(flux[d][ui]) ? ft[1] : flux[d][ui] + "+" + ft[1];
            }
        }
        return new ConservationForm(unknownList, ordered, flux, src);
    }

    // ------------------------------------------------------------------ small helpers
    private static int countOf(List<String> vars, String v) {
        int c = 0;
        for (String s : vars) {
            if (s.equals(v)) {
                c++;
            }
        }
        return c;
    }

    private static String stripQuotes(String s) {
        if (s.length() >= 2 && ((s.startsWith("\"") && s.endsWith("\"")) || (s.startsWith("'") && s.endsWith("'")))) {
            return s.substring(1, s.length() - 1).trim();
        }
        return s;
    }

    private static String stripOuterParens(String s) {
        String t = s;
        while (t.startsWith("(") && EquationAnalyzer.matchParen(t, 0) == t.length() - 1) {
            t = t.substring(1, t.length() - 1).trim();
        }
        return t;
    }

    private static void collectIdentifiers(String s, Set<String> out) {
        int i = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (Character.isLetter(c) || c == '_') {
                int j = i;
                while (j < s.length() && EquationAnalyzer.isIdentChar(s.charAt(j))) {
                    j++;
                }
                int k = j;
                while (k < s.length() && s.charAt(k) == ' ') {
                    k++;
                }
                if (!(k < s.length() && s.charAt(k) == '(')) {
                    out.add(s.substring(i, j));
                }
                i = j;
            } else if (Character.isDigit(c)) {
                while (i < s.length() && (Character.isLetterOrDigit(s.charAt(i)) || s.charAt(i) == '.')) {
                    i++;
                }
            } else {
                i++;
            }
        }
    }
}
