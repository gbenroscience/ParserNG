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
import com.github.gbenroscience.parser.MathExpression.Token;

 

/**
 * Turbo-tier counterpart to {@link EquationCoefficientResolver} — same
 * contract, but resolving to a {@link TurboResolvedEquation} (a {@code
 * MethodHandle}-compiled top-derivative) instead of an {@code ODEFunction}
 * one. {@link TurboCoefficientExtractor#resolve} is the real implementation.
 */
@FunctionalInterface
public interface EquationCoefficientResolver {

    ResolvedEquation resolve(Token[] equationPostfix, int order);
}