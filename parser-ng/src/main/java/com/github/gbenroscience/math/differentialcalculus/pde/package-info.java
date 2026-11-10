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
/**
 * Numerical partial-differential-equation engine for ParserNG.
 * <p>
 * Entry point: {@link com.github.gbenroscience.math.differentialcalculus.pde.PDE}.
 * Equations are written as ParserNG expressions ({@code diff(u,t) = alpha*diff(u,x,2)}),
 * compiled once, and solved on uniform 1-D, 2-D or 3-D grids by the method of
 * lines (time-dependent problems) or by iterative solvers (steady problems).
 * See {@code PDE.md} in the module root for the full guide.
 * <p>
 * Package map:
 * <ul>
 * <li>Public API: {@code PDE}, {@code PDEProblem}, {@code PDEOptions},
 * {@code PDESolver}, {@code PDEResult}, {@code SolverDiagnostics},
 * {@code Grid}, {@code Face}, {@code BoundaryCondition},
 * {@code MethodOfLinesSystem}, the method / policy enums and the exceptions.</li>
 * <li>Implementation (package-private): {@code Model} (compiled problem and
 * node kernels), {@code Boundaries}, {@code EvolutionEngine},
 * {@code EllipticEngine}, {@code NewtonKrylov}, {@code Stencils},
 * {@code EquationAnalyzer}, {@code CompiledExpr}, {@code WorkerPool}.</li>
 * </ul>
 */
package com.github.gbenroscience.math.differentialcalculus.pde;
