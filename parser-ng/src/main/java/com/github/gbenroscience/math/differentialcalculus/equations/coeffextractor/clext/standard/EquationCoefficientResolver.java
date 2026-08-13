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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard;

/**
 *
 * @author GBEMIRO
 */ 

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.ResolvedEquation;
import com.github.gbenroscience.parser.MathExpression.Token;

/**
 * The one integration point {@link EquationRuntime} cannot supply itself:
 * given the isolated postfix of a general-form equation (already {@code
 * LHS-RHS}, implicitly {@code = 0}) and the equation's order, extract the
 * linear top-order coefficient, divide it out, compile the resulting
 * top-derivative expression, and report which real frame slot backs each
 * of {@code t, y[0], ..., y[order-1]} (or {@link CanonicalFrame#NO_REAL_SLOT}
 * for any that never appear in the text at all).
 *
 * <h2>What this pipeline already has ready to receive it</h2>
 * Routing ({@link DiffEqnArgParser#classify}), argument parsing ({@link
 * DiffEqnArgParser#parse}), isolating this argument's own postfix ({@link
 * PostfixArgumentIsolator}), the canonical/real frame adapter ({@link
 * CanonicalFrame}, {@link FrameRemappingODEFunction}), solver dispatch, and
 * the exact analytic Jacobian are all fully implemented in {@link
 * EquationRuntime} and do not depend on this. Only the actual "linear form
 * extraction + compile" step does — the piece described earlier as {@code
 * CoefficientExtractor}/{@code LinearFormExtractor}/{@code ExprAlgebra},
 * being ported into this package, including the fix for the frame-ordering
 * bug (a resolver implementation is exactly where that fix's output —
 * the real frame slot for each {@code y[k]}, present or absent — gets
 * reported).
 */
@FunctionalInterface
public interface EquationCoefficientResolver {

    /**
     * @param equationPostfix the isolated equation argument's own postfix
     *                        (from {@link PostfixArgumentIsolator}),
     *                        already {@code LHS-RHS}, implicitly {@code = 0}
     * @param order           y0.length — the equation's order
     * @return 
     */
    ResolvedEquation resolve(Token[] equationPostfix, int order);
}