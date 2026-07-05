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

import com.github.gbenroscience.parser.MathExpression;
import static com.github.gbenroscience.parser.ng.bench.ParserNgBench.EXPRESSION;
import static com.github.gbenroscience.parser.ng.bench.ParserNgBench.getVars;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/**
 *
 * @author GBEMIRO
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(value = 3, jvmArgs = {
    "-Xms5g", "-Xmx5g",
    "-XX:+UseG1GC",
    "-XX:-UseCompressedOops", // Avoids compressed oops artifacts
    "--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions"
})
@State(Scope.Thread)
public class QuickBench3 {

    public static interface JaninoMathFunction {

        double apply(double[] v);
    }

    @Param({
        "(sin(x1^3)-cos(x1^4)+tan(x1^0.5))/(2*x1^2+1)"
    })
    private String expression;

    @Param({"2000000"})
    private int dataSize;

    private double[] result;
    private SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression simdVectorTurbo;

    private int varCount;
    private double[] vars;
    protected static String[] expressionVars = getVars(EXPRESSION);
    //SoA dataSink layout for ParserNG
    private static double[][] dataSink;

    @Setup(Level.Trial)
    public void setup() {
        System.out.println("================EXPRESSION: " + expression + "================");
        MathExpression me = new MathExpression(expression);
        expressionVars = me.getVariablesNames();
        varCount = expressionVars.length;
        this.vars = new double[varCount];

        result = new double[dataSize];

        // Fill once per trial, not per iteration
        Random r = new Random(42);

        if (dataSink == null) {
            // 1. Allocate your native SoA format for ParserNG
            dataSink = new double[varCount][dataSize];

            for (int i = 0; i < dataSink.length; i++) {
                 for (int j = 0; j < dataSize; j++) {
                        double val = r.nextDouble(20);
                        // Populate ParserNG Data
                        dataSink[i][j] = val;
                    }
            }
            System.out.println("Repopulated both SoA and AoS data layouts successfully.");
        }

        setupParserNG(me);

        // Validation against the SoA structure
        for (int i = 0; i < dataSink.length; i++) {
            // Ensure your validate method is updated to handle double[][] 
            // instead of a single flat array
            simdVectorTurbo.validate(dataSink[i], result);
        }
    }

 


    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void parserNG(Blackhole bh) {
        // Read from the same stable scenario index—zero cursor modifications inside!
        final double[][] input = dataSink;
        final double[] res = result;

        // Execute core computation kernel
        simdVectorTurbo.applyBulk(input, res);
        bh.consume(res);
    }


    private void setupParserNG(MathExpression me) {
        try {
            simdVectorTurbo = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        } catch (Throwable ex) {
            throw new RuntimeException("Failed to compile SIMDVectorTurboEvaluator expressions", ex);
        }
    }

    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(QuickBench3.class.getSimpleName());

        Options configurations = opt.mode(Mode.AverageTime)
                .timeUnit(TimeUnit.NANOSECONDS)
                .warmupIterations(5)
                .warmupTime(TimeValue.milliseconds(200L))
                .measurementIterations(5)
                .measurementTime(TimeValue.milliseconds(500))
                .forks(2)
                .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                .jvmArgs("-Xms8g", "-Xmx8g", "-Dbenchmark.index=1")
                .jvmArgsAppend("--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions")
                .build();

        new Runner(configurations).run();
    }
}
