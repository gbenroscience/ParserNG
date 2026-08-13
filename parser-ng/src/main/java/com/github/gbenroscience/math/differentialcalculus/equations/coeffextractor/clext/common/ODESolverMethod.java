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
public enum ODESolverMethod {
    EULER, // Fast, O(h) error. Best for real-time graphics/particles.
    RK4, // Classical 4th Order fixed-step system workhorse.
    RK45_DORMAND_PRINCE,// Adaptive-step size system engine (Industry standard).
    IMPLICIT_EULER      // Backwards implicit setup optimized for stiff vector spaces.
}
