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
 * Parsed shape of one diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call,
 * after {@link DiffEqnArgParser} has read the raw argument text off the
 * call's Token and turned it into typed values. Does not itself hold a
 * compiled {@code ODEFunction} — building that from the first argument (the
 * raw equation itself, {@code LHS-RHS} with {@code =} omitted) is {@link
 * EquationRuntime}'s job, once the equation's own postfix is isolated (see
 * {@link EquationCoefficientResolver}'s javadoc for the one open dependency
 * this pipeline still has).
 */
public final class DiffEqnCall {

    public enum Kind {
        DIFFEQN, DIFFEQN_PATH, DIFFEQN_HO, DIFFEQN_PATH_HO
    }

    public final Kind kind;
    /** rawArgs[0] as written — the raw equation's own text, e.g. "t*y[1] + 2*y[0] - sin(t)". */
    public final String rhsRawText;
    public final double t0;
    /** Length 1 for scalar diffeqn/diffeqnPath; length == order for diffeqnHO/diffeqnPathHO. */
    public final double[] y0;
    public final double tEnd;
    /** Step size (fixed methods) or initial step (rk45); {@link DiffEqnArgParser#DEFAULT_H} if omitted. */
    public final double h;
    public final ODESolverMethod method;
    /** Only meaningful for the *_PATH kinds; <= 0 means "solver's natural steps, no resampling". */
    public final int points;

    public DiffEqnCall(Kind kind, String rhsRawText, double t0, double[] y0, double tEnd,
                        double h, ODESolverMethod method, int points) {
        this.kind = kind;
        this.rhsRawText = rhsRawText;
        this.t0 = t0;
        this.y0 = y0;
        this.tEnd = tEnd;
        this.h = h;
        this.method = method;
        this.points = points;
    }

    public boolean isPathVariant() {
        return kind == Kind.DIFFEQN_PATH || kind == Kind.DIFFEQN_PATH_HO;
    }

    public boolean isHigherOrder() {
        return kind == Kind.DIFFEQN_HO || kind == Kind.DIFFEQN_PATH_HO;
    }
}