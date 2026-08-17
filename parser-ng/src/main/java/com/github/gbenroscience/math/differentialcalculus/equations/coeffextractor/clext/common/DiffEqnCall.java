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

import com.github.gbenroscience.math.differentialcalculus.equations.standard.DifferentialEquations;

/**
 * Parsed shape of one diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call.
 *
 * <h2>equationTexts and equationArraySyntax</h2>
 * equationTexts holds one or more raw equation strings, each already
 * rearranged to {@code LHS-RHS} with {@code = 0} omitted — exactly the same
 * convention a single diffeqn/diffeqnHO equation already uses.
 * <p>
 * For the classic unwrapped single-equation form (equationArraySyntax ==
 * false), equationTexts is a length-1 array holding argument 0's raw text
 * verbatim, and {@link EquationRuntime} compiles it by directly isolating
 * argument 0's own postfix from the call's existing tokens — zero
 * re-parsing, the original behavior, unchanged.
 * <p>
 * For an explicit system (equationArraySyntax == true, via {@code
 * diffeqn(@(n)("eq1", ..., "eqN"), t0, y0, tEnd, ...)}), equationTexts holds
 * N independently-parseable equation strings. Each equation's divided-out
 * symbol is always {@code y[n]} where n == the system's component count
 * (== y0.length) — constant across every equation in the system, exactly
 * mirroring how a scalar diffeqn equation divides out {@code y[1]} and an
 * order-n HO equation divides out {@code y[n]}. Each string is compiled via
 * its own fresh MathExpression — safe because each is parsed exactly once
 * (no round-trip through an already-compiled tree, so there is no risk of
 * the diverging-VariableRegistry bug TokenTreeBuilder's javadoc warns
 * about), and each equation ends up with its own independent CanonicalFrame.
 */
public final class DiffEqnCall {

    public enum Kind {
        DIFFEQN, DIFFEQN_PATH, DIFFEQN_HO, DIFFEQN_PATH_HO
    }

    public final Kind kind;

    /**
     * One or more raw equation strings, each LHS-RHS with '=0' omitted.
     * Length 1 for the classic unwrapped form; length N (== y0.length) for
     * an explicit system given as @(N)("eq1", ..., "eqN").
     */
    public final String[] equationTexts;

    /**
     * True if argument 0 was supplied via the @(n)(...) array syntax (an
     * explicit system, even if N == 1), false for the classic unwrapped
     * single-equation form. Determines which compilation path
     * EquationRuntime takes — NOT simply equationTexts.length == 1, since a
     * wrapped single equation (@(1)("...")) still needs the reparse-from-
     * string path, not the direct-token-isolation path.
     */
    public final boolean equationArraySyntax;

    public final double t0;
    /** Length 1 for scalar diffeqn/diffeqnPath; length == order for diffeqnHO/diffeqnPathHO/an explicit system. */
    public final double[] y0;
    public final double tEnd;
    /** Step size (fixed methods) or initial step (rk45); {@link DiffEqnArgParser#DEFAULT_H} if omitted. */
    public final double h;
    public final ODESolverMethod method;
    /** Only meaningful for the *_PATH kinds; <= 0 means "solver's natural steps, no resampling". */
    public final int points;
    /**
     * {@link PresentationStrategy} field
     * Determines whether to show a matrix of t, y, y', y''(when this field equals {@link PresentationStrategy#STATE} ) etc 
     * or just t, y (when this field equals {@link PresentationStrategy#TRAJECTORY} ).
     * The default is {@link PresentationStrategy#TRAJECTORY}
     */
    public final PresentationStrategy presentationStrategy;

    public DiffEqnCall(Kind kind, String[] equationTexts, boolean equationArraySyntax, double t0, double[] y0,
                        double tEnd, double h, ODESolverMethod method, int points,
                        PresentationStrategy presentationStrategy) {
        this.kind = kind;
        this.equationTexts = equationTexts;
        this.equationArraySyntax = equationArraySyntax;
        this.t0 = t0;
        this.y0 = y0;
        this.tEnd = tEnd;
        this.h = h;
        this.method = method;
        this.points = points;
        this.presentationStrategy = presentationStrategy;
    }

    public boolean isPathVariant() {
        return kind == Kind.DIFFEQN_PATH || kind == Kind.DIFFEQN_PATH_HO;
    }

    public boolean isHigherOrder() {
        return kind == Kind.DIFFEQN_HO || kind == Kind.DIFFEQN_PATH_HO;
    }

    /** True for a genuine multi-equation system (equationTexts.length > 1). */
    public boolean isSystem() {
        return equationTexts.length > 1;
    }
}