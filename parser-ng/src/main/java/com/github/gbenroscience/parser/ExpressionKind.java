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
package com.github.gbenroscience.parser;

/**
 * What a {@link MathExpression} represents, as recognised when it is
 * constructed.
 * <ul>
 * <li>{@link #SCALAR}: an ordinary expression (numbers, vectors, matrices, text).</li>
 * <li>{@link #ODE} / {@link #ODE_SYSTEM}: a {@code diffeqn} family call, for a
 * single equation or an {@code @(n)("eq1",...)} system.</li>
 * <li>{@link #PDE} / {@link #PDE_SYSTEM}: a partial differential equation, or a
 * system of them, given as {@code diff(u,t) = alpha*diff(u,x,2)}, as
 * {@code pde(...)}, or as several such equations. A PDE expression is a
 * mathematical model: it is not evaluated to a number, and the numerical
 * problem is obtained from {@link MathExpression#asPDE()}.</li>
 * </ul>
 */
public enum ExpressionKind {
    SCALAR, ODE, ODE_SYSTEM, PDE, PDE_SYSTEM
}
