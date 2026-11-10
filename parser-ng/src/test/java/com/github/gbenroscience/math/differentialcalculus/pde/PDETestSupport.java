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
 * Shared helpers for the PDE test-suite.
 */
final class PDETestSupport {

    private PDETestSupport() {
    }

    /** Observed order of convergence from two errors on grids differing by {@code ratio} in resolution. */
    static double order(double coarseErr, double fineErr, double ratio) {
        return Math.log(coarseErr / fineErr) / Math.log(ratio);
    }

    /** Parabolic heat problem on [0,1] with homogeneous Dirichlet data and the sin(pi x) mode. */
    static PDEProblem.Builder heat1d(double alpha, int n) {
        return PDE.problem("diff(u,t) = alpha*diff(u,x,2)")
                .param("alpha", alpha)
                .domain("x", 0, 1, n)
                .initial("u", "sin(pi*x)")
                .dirichlet("u", "0");
    }

    static PDEResult.Exact heat1dExact(final double alpha) {
        return new PDEResult.Exact() {
            @Override
            public double at(double x, double y, double z, double t) {
                return Math.exp(-Math.PI * Math.PI * alpha * t) * Math.sin(Math.PI * x);
            }
        };
    }
}
