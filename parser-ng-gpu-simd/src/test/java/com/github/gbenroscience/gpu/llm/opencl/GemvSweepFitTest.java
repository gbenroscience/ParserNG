package com.github.gbenroscience.gpu.llm.opencl;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class GemvSweepFitTest {

    @Test
    void leastSquaresRecoversKnownCoefficients() {
        // t = 12 + 0.7*rows + 55*MB  with distinct shapes (rows and MB not collinear)
        List<double[]> rows = new ArrayList<>();
        double[][] shapes = {{512, 1.1}, {2048, 4.5}, {8192, 17.8}, {2048, 17.8}, {128256, 279.1}, {512, 17.8}, {32768, 17.8}};
        for (double[] s : shapes) {
            rows.add(new double[]{s[0], s[1], 12 + 0.7 * s[0] + 55 * s[1]});
        }
        double[] f = GemvSweep.fit(rows);
        assertEquals(12, f[0], 1e-6);
        assertEquals(0.7, f[1], 1e-9);
        assertEquals(55, f[2], 1e-8);
        assertEquals(1.0, f[3], 1e-12);
    }

    @Test
    void reportsNaNWhenShapesAreCollinear() {
        List<double[]> rows = new ArrayList<>();
        for (int i = 1; i < 6; i++) {
            rows.add(new double[]{i * 100, i * 2.0, 3.0 * i}); // rows and MB proportional -> cannot be separated
        }
        assertTrue(Double.isNaN(GemvSweep.fit(rows)[0]));
    }
}
