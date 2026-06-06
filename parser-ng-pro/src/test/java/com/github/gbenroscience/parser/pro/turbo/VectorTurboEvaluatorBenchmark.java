package com.github.gbenroscience.parser.pro.turbo;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator;
import org.openjdk.jmh.annotations.*;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
@State(Scope.Thread)
public class VectorTurboEvaluatorBenchmark {

    @Param({"1000", "50000", "500000"})
    private int dataSize;

    private double[][] variables;
    private double[] outputBuffer;

    private SIMDCompositeExpression linearExpr;
    private SIMDCompositeExpression gaussianExpr;
    private SIMDCompositeExpression conditionalExpr;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        Random rand = new Random(42);
        
        // Structure of Arrays (SoA): 3 variables (x1, x2, x3), each of length dataSize
        variables = new double[3][dataSize];
        outputBuffer = new double[dataSize];

        for (int i = 0; i < dataSize; i++) {
            variables[0][i] = 1.5 + rand.nextDouble() * 5.0;  // x1 (Must be >0 for div/pow)
            variables[1][i] = rand.nextDouble() * 5.0;        // x2
            variables[2][i] = rand.nextDouble() * 2.0;        // x3
        }

        // Compile expressions using the Vector Engine
        MathExpression meLinear = new MathExpression("12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2");
        linearExpr = (SIMDCompositeExpression) new VectorTurboEvaluator(meLinear).compile();

        MathExpression meGaussian = new MathExpression("(1 / (x1 * sqrt(2 * 3.141592653589793))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        gaussianExpr = (SIMDCompositeExpression) new VectorTurboEvaluator(meGaussian).compile();

        /*
        MathExpression meConditional = new MathExpression("if(x1 >= 2.5, sin(x1) % x2, x3 * vma(x1, x2, 1.5))");
        conditionalExpr = (SIMDCompositeExpression) new VectorTurboEvaluator(meConditional).compile();
        */
    }

    @Benchmark
    public double[] benchmarkLinearPolynomialBulk() {
        linearExpr.applyBulk(variables, outputBuffer);
        return outputBuffer;
    }

    @Benchmark
    public double[] benchmarkGaussianDistributionBulk() {
        gaussianExpr.applyBulk(variables, outputBuffer);
        return outputBuffer;
    }
/*
    @Benchmark
    public double[] benchmarkHardwareMaskConditionalBulk() {
        conditionalExpr.applyBulk(variables, outputBuffer);
        return outputBuffer;
    }
    */
    
    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
            opt.include(VectorTurboEvaluatorBenchmark.class.getSimpleName()); // Always include baseline
           // 4. Fluent, modern JMH Configuration
            Options configurations = opt.mode(Mode.AverageTime)
                    .timeUnit(TimeUnit.NANOSECONDS)
                    .warmupIterations(5)
                    .warmupTime(TimeValue.milliseconds(200L))
                    .measurementIterations(5)
                    .measurementTime(TimeValue.milliseconds(500))
                    .forks(2)
                    .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                    .jvmArgs("-Xms2g", "-Xmx2g", "-Dbenchmark.index=1")
                    .jvmArgsAppend("--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions")
                    .build();

            new Runner(configurations).run();
    }
}
