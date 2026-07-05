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
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import java.util.Random;
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
public class QuickBench {

 

    @Param({
        "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))"
    })
    private String expression;

    @Param({"2000000"})
    private int dataSize;

    private double[] flatInput; 
    private double[] result;
    private SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression simdVectorTurbo;

    private int varCount;
    private double[] vars;

    @Setup(Level.Trial)
    public void setup() {
        MathExpression me = new MathExpression(expression);
        String[] expressionVars = me.getVariablesNames();
        this.varCount = expressionVars.length;
        this.vars = new double[varCount];

        this.flatInput = new double[varCount * dataSize];
        this.result = new double[dataSize];

        // Safe initialization boundary (1.0 to 10.0) to prevent division by zero in expressions
        Random r = new Random(42);
        for (int i = 0; i < flatInput.length; i++) {
            flatInput[i] = 1.0 + (r.nextDouble() * 9.0);
        }
        
        setupParserNG(me);
    }

    @Benchmark
    public void parserNG(Blackhole bh) {
        // True measurement of high-performance vectorized vector logic execution
        simdVectorTurbo.applyBulk(flatInput, result);
        bh.consume(result); 
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
        opt.include(QuickBench.class.getSimpleName());
        
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