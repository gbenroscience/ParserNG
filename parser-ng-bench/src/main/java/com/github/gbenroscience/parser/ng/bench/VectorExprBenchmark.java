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
package com.github.gbenroscience.parser.ng.bench;
 
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF32;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark comparing double vs float bulk evaluation
 * of the expression {@code 3*x^2 + sin(x^3)}.
 *
 * Both {@link SIMDCommandF64.SIMDVectorCompositeExpression} and its float
 * counterpart are {@code AutoCloseable}: the {@code getEvaluator(String, int)}
 * overload with a worker count can spin up a pinned background thread pool,
 * so both evaluators are requested with {@code numWorkers = 0} here (only the
 * single-threaded {@code applyBulk} is exercised — no {@code applyBulkParallel}
 * calls) and are explicitly {@code close()}d in {@link #tearDown()} rather than
 * left for the class's {@code Cleaner}-based GC safety net to reclaim.
 *
 * @author GBEMIRO
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS) 
@Fork(value = 2, jvmArgs = {
    "--add-modules=jdk.incubator.vector",
    "--enable-preview"
})
@State(Scope.Thread)
public class VectorExprBenchmark {

    private static final String EXPRESSION = "3*x^2+sin(x^3)";
    private static final int VECTOR_LEN = 1_000_000;
    private static final int NO_BACKGROUND_WORKERS = 0;

    private SIMDCommandF64.SIMDVectorCompositeExpression doubleExpr;
    private SIMDCommandF32.SIMDVectorCompositeExpression floatExpr;

    private double[] in;
    private double[] out;
    private float[] inf;
    private float[] outf;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        // numWorkers = 0: applyBulk() never touches the worker pool, so
        // requesting one here would only cost pinned-thread startup and
        // teardown time without ever being exercised.
        doubleExpr = SIMDCommandF64.getEvaluator(EXPRESSION, NO_BACKGROUND_WORKERS);
        floatExpr = SIMDCommandF32.getEvaluator(EXPRESSION, NO_BACKGROUND_WORKERS);

        in = new double[VECTOR_LEN];
        out = new double[VECTOR_LEN];
        inf = new float[VECTOR_LEN];
        outf = new float[VECTOR_LEN];

        Random r = new Random(42); // fixed seed: identical input across forks/runs
        for (int i = 0; i < VECTOR_LEN; i++) {
            in[i] = r.nextDouble();
            inf[i] = (float) in[i];
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        // Explicit release rather than relying on the Cleaner: promptly
        // frees whatever the evaluator holds (thread-locals, and worker
        // threads if numWorkers had been > 0) instead of leaving it to GC,
        // which is not guaranteed to run before the next trial/fork.
        if (doubleExpr != null) {
            doubleExpr.close();
        }
        if (floatExpr != null) {
            floatExpr.close();
        }
    }

    @Benchmark
    public void doubleBulkApply(Blackhole bh) {
        doubleExpr.applyBulk(in, out);
        bh.consume(out);
    }

    @Benchmark
    public void floatBulkApply(Blackhole bh) {
        floatExpr.applyBulk(inf, outf);
        bh.consume(outf);
    }

    /**
     * Convenience entry point so this can be run directly (e.g. {@code java -jar ...})
     * without requiring the JMH Maven/Gradle plugin's generated main-class shim.
     * @param args
     * @throws org.openjdk.jmh.runner.RunnerException
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(VectorExprBenchmark.class.getSimpleName())
                .build();
        new Runner(opt).run();
    }
}