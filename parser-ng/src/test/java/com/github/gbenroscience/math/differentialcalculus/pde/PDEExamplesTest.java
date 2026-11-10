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

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/** Executes the documented examples and checks their answers. */
public class PDEExamplesTest {

    private static final double PI = Math.PI;

    @Test
    void heatEquation() {
        PDEResult r = PDEExamples.heatEquation();
        assertTrue(r.maxError("u", PDETestSupport.heat1dExact(0.1)) < 1e-4);
    }

    @Test
    void vibratingMembrane() {
        PDEResult r = PDEExamples.vibratingMembrane();
        assertTrue(r.isConverged());
        double max = 0;
        Grid g = r.getGrid();
        for (int i = 0; i < g.n(0); i++) {
            assertEquals(0.0, r.value("u", i, 0, 0), 0.0);
            assertEquals(0.0, r.value("u", i, g.n(1) - 1, 0), 0.0);
        }
        for (double v : r.values("u")) {
            max = Math.max(max, Math.abs(v));
        }
        assertTrue(max < 1.0 && max > 0.0, "amplitude stays bounded: " + max);
    }

    @Test
    void laplaceEquation() {
        PDEResult r = PDEExamples.laplaceEquation();
        assertTrue(r.isConverged());
        assertTrue(r.maxError("u", (x, y, z, t) -> Math.sin(PI * x) * Math.sinh(PI * y) / Math.sinh(PI)) < 2e-4);
    }

    @Test
    void waveEquation() {
        PDEResult r = PDEExamples.waveEquation();
        assertEquals(1.25, r.getTime(), 1e-12);
        assertTrue(r.maxError("u", (x, y, z, t) -> Math.sin(2 * PI * t) * Math.sin(2 * PI * x)) < 1e-6);
    }

    @Test
    void coupledSystem() {
        PDEResult r = PDEExamples.coupledSystem();
        double a = 0.1 * PI * PI;
        assertTrue(r.maxError("u", (x, y, z, t) -> Math.exp(-a * t) * Math.cos(t) * Math.sin(PI * x)) < 1e-4);
        assertTrue(r.maxError("v", (x, y, z, t) -> Math.exp(-a * t) * Math.sin(t) * Math.sin(PI * x)) < 1e-4);
    }

    @Test
    void grayScott() {
        PDEResult r = PDEExamples.grayScott();
        double uMin = 1e9;
        double uMax = -1e9;
        double vMax = 0;
        for (double v : r.values("u")) {
            uMin = Math.min(uMin, v);
            uMax = Math.max(uMax, v);
        }
        for (double v : r.values("v")) {
            vMax = Math.max(vMax, v);
        }
        assertTrue(uMin > 0.0 && uMax < 1.05, "u stays in the physical range: " + uMin + ".." + uMax);
        assertTrue(vMax > 0.0 && vMax < 1.0);
    }

    @Test
    void realTimeLoop() {
        double peak = PDEExamples.realTimeLoop(100);
        assertTrue(peak > 0 && peak <= 1.0 + 1e-9);
    }
}
