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

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;

/**
 *
 * @author GBEMIRO
 */
public final class FrameRemapper implements ODEFunction {
 
    private final ODEFunction real;
    private final CanonicalFrame frame;
 
    public FrameRemapper(ODEFunction real, CanonicalFrame frame) {
        if (real == null) {
            throw new IllegalArgumentException("real ODEFunction must not be null");
        }
        if (frame == null) {
            throw new IllegalArgumentException("frame must not be null");
        }
        this.real = real;
        this.frame = frame;
    }
 
    @Override
    public void apply(double[] canonicalVars, double[] outDerivatives) {
        real.apply(frame.toReal(canonicalVars), outDerivatives);
    }
}
 