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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo;

 

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.ExprNode;
import java.lang.invoke.MethodHandle;

/**
 * Turbo-tier counterpart to {@link ResolvedEquation}: everything {@link
 * TurboEquationRuntime} needs from turning one isolated, general-form
 * equation into a divided, callable top-derivative — a {@code MethodHandle}
 * matching {@code (double[], double[])void} instead of an {@code
 * ODEFunction}, compiled via {@code ExprNodeCompiler.compileTurbo}.
 */
public final class ResolvedEquation {

    /** The divided y[n] = ... expression, compiled to a MethodHandle against the REAL (possibly scattered) frame. */
    public final MethodHandle topDerivativeRealFrame;

    /** The same expression's ExprNode tree — needed to build the exact AD Jacobian, not just evaluate numerically. */
    public final ExprNode topDerivativeTree;

    /** canonicalToReal[0] = t's real slot; canonicalToReal[1+k] = y[k]'s real slot, or {@link CanonicalFrame#NO_REAL_SLOT}. */
    public final int[] canonicalToReal;

    /** Size of the real frame the equation was compiled against. */
    public final int realFrameSize;

    public ResolvedEquation(MethodHandle topDerivativeRealFrame, ExprNode topDerivativeTree,
                                  int[] canonicalToReal, int realFrameSize) {
        this.topDerivativeRealFrame = topDerivativeRealFrame;
        this.topDerivativeTree = topDerivativeTree;
        this.canonicalToReal = canonicalToReal;
        this.realFrameSize = realFrameSize;
    }
}