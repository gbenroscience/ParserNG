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

/**
 * Finite-difference scheme used for FIRST spatial derivatives. Derivatives of
 * order two and above always use central stencils.
 */
public enum DerivativeScheme {
    /** Symmetric stencil, even accuracy order (2, 4, 6...). Non-dissipative. */
    CENTRAL,
    /** One-sided stencil using points behind the node (upwind for a positive velocity). */
    BACKWARD,
    /** One-sided stencil using points ahead of the node (upwind for a negative velocity). */
    FORWARD
}
