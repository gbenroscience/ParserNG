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
package com.github.gbenroscience.arrow;

import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF32;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;

import java.util.Arrays;
import java.util.Random;

/**
 * Quick ad-hoc benchmark comparing double vs float bulk evaluation
 * of the expression {@code 3*x^2 + sin(x^3)}.
 *
 * @author GBEMIRO
 */
public final class Main {

    private static final String EXPRESSION = "3*x^2+sin(x^3)";
    private static final int VECTOR_LEN = 1_000_000;
    private static final int WARMUP_ITERATIONS = 100;
    private static final int TIMED_ITERATIONS = 200;
    private static final int PREVIEW_COUNT = 50;

    private Main() {
    }

    public static void main(String[] args) throws Throwable {
        SIMDCommandF64.SIMDVectorCompositeExpression doubleExpr = SIMDCommandF64.getEvaluator(EXPRESSION);
        SIMDCommandF32.SIMDVectorCompositeExpression floatExpr = SIMDCommandF32.getEvaluator(EXPRESSION);

        double[] in = new double[VECTOR_LEN];
        double[] out = new double[VECTOR_LEN];
        float[] inf = new float[VECTOR_LEN];
        float[] outf = new float[VECTOR_LEN];

        Random r = new Random();
        for (int i = 0; i < VECTOR_LEN; i++) {
            in[i] = r.nextDouble();
            inf[i] = (float) in[i];
        }

        double doubleNanosPerRun = timeBulkApply(() -> doubleExpr.applyBulk(in, out));
        double floatNanosPerRun = timeBulkApply(() -> floatExpr.applyBulk(inf, outf));

        System.out.println("DOUBLE TIME: " + doubleNanosPerRun + " ns/run");
        System.out.println("FLOAT  TIME: " + floatNanosPerRun + " ns/run");
        System.out.println("DOUBLE RESULTS (first " + PREVIEW_COUNT + "): "
                + Arrays.toString(Arrays.copyOf(out, PREVIEW_COUNT)));
        System.out.println("FLOAT  RESULTS (first " + PREVIEW_COUNT + "): "
                + Arrays.toString(Arrays.copyOf(outf, PREVIEW_COUNT)));
    }

    private interface BulkOp {
        void run() throws Throwable;
    }

    private static double timeBulkApply(BulkOp op) throws Throwable {
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            op.run();
        }
        long start = System.nanoTime();
        for (int i = 0; i < TIMED_ITERATIONS; i++) {
            op.run();
        }
        return (System.nanoTime() - start) / (double) TIMED_ITERATIONS;
    }
}