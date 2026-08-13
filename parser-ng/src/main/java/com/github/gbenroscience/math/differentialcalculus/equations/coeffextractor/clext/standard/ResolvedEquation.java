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
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ExprNode;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;

/**
 * Everything {@link EquationRuntime} needs from turning one isolated,
 * general-form equation ({@code A1(t)*y[n]+A2(t)*f1(y[n-1])+...=B(t)},
 * already rearranged to {@code LHS-RHS}) into a divided, callable
 * top-derivative — the shape {@code CoefficientExtractor} (ported into this
 * package per the earlier build plan) is expected to produce.
 */
public final class ResolvedEquation {

    /** The divided y[n] = ... expression, compiled against the REAL (possibly scattered) ParserNG frame. */
    public final ODEFunction topDerivativeRealFrame;

    /** The same expression's ExprNode tree — needed to build the exact AD Jacobian, not just evaluate numerically. */
    public final ExprNode topDerivativeTree;

    /**
     * canonicalToReal[0] = the real frame slot of t; canonicalToReal[1+k] =
     * the real frame slot of y[k] for k = 0..order-1, or {@link
     * CanonicalFrame#NO_REAL_SLOT} if y[k] never appears anywhere in the
     * equation text (the sparse-equation case flagged during this
     * pipeline's build — CoefficientExtractor must report this rather than
     * synthesize a fake frame index for a variable MathExpression never
     * actually scanned).
     */
    public final int[] canonicalToReal;

    /** Size of the real frame the equation was compiled against — needed to size CanonicalFrame's scratch buffer. */
    public final int realFrameSize;
/**
 * 
 * @param topDerivativeRealFrame
 * @param topDerivativeTree
 * @param canonicalToReal
 * @param realFrameSize 
 */
    public ResolvedEquation(ODEFunction topDerivativeRealFrame, ExprNode topDerivativeTree,
                             int[] canonicalToReal, int realFrameSize) {
        this.topDerivativeRealFrame = topDerivativeRealFrame;
        this.topDerivativeTree = topDerivativeTree;
        this.canonicalToReal = canonicalToReal;
        this.realFrameSize = realFrameSize;
    }
}