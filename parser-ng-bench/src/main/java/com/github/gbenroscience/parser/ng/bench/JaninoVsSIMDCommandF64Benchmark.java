package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Head-to-head: JaninoVectorTurboEvaluator vs. SIMDCommandF64, on the same
 * two expressions and the same input data, at {@code dataSize} = 1,000 and
 * 1,000,000. Both sides call the plain (single-threaded) {@code applyBulk}
 * entry point - same call shape {@link JaninoVectorTurboEvaluatorBenchmark}
 * already uses - so this isolates the two compilation/execution strategies
 * rather than mixing in parallel-dispatch differences.
 *
 * <p>Each expression is compiled from its own fresh {@link MathExpression}
 * instance per engine (not one instance shared across both compile()
 * calls) - I don't know whether MathExpression's internal state (e.g.
 * slot assignment) is safe to compile twice with two different backends,
 * so this avoids finding out the hard way in a benchmark run.
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
public class JaninoVsSIMDCommandF64Benchmark {

    private static final String LINEAR_EXPR = "12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2";
    private static final String GAUSSIAN_EXPR = "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))";

    @Param({"1000", "1000000"})
    private int dataSize;

    private double[][] variables;
    private double[] outputBuffer;

    // Janino path
    private JaninoVectorTurboEvaluator.JaninoBulkExpression janinoLinearExpr;
    private JaninoVectorTurboEvaluator.JaninoBulkExpression janinoGaussianExpr;

    // SIMDCommandF64 path
    private SIMDCommandF64.SIMDVectorCompositeExpression simdLinearExpr;
    private SIMDCommandF64.SIMDVectorCompositeExpression simdGaussianExpr;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        Random rand = new Random(42);
        // Structure of Arrays (SoA): 3 variables (x1, x2, x3), each of length dataSize
        variables = new double[3][dataSize];
        outputBuffer = new double[dataSize];

        for (int i = 0; i < dataSize; i++) {
            variables[0][i] = 1.5 + rand.nextDouble() * 5.0;  // x1 (must be >0 for div/log/exp denominator)
            variables[1][i] = rand.nextDouble() * 5.0;        // x2
            variables[2][i] = rand.nextDouble() * 2.0;        // x3
        }

        janinoLinearExpr = (JaninoVectorTurboEvaluator.JaninoBulkExpression)
                new JaninoVectorTurboEvaluator(new MathExpression(LINEAR_EXPR)).compile();
        janinoGaussianExpr = (JaninoVectorTurboEvaluator.JaninoBulkExpression)
                new JaninoVectorTurboEvaluator(new MathExpression(GAUSSIAN_EXPR)).compile();

        simdLinearExpr = SIMDCommandF64.getEvaluator(new MathExpression(LINEAR_EXPR));
        simdGaussianExpr = SIMDCommandF64.getEvaluator(new MathExpression(GAUSSIAN_EXPR));
    }

    /**
     * Same "sample every 64th element and sum" trick the original benchmark
     * uses, to force the JIT to actually execute every loop iteration
     * without paying full Blackhole.consume() cost per element.
     */
    private static double checksum(double[] out) {
        double c = 0.0;
        for (int i = 0; i < out.length; i += 64) {
            c += out[i];
        }
        return c;
    }

    @Benchmark
    public void benchmarkJaninoLinearBulk(Blackhole bh) {
        janinoLinearExpr.applyBulk(variables, outputBuffer);
        bh.consume(checksum(outputBuffer));
    }

    @Benchmark
    public void benchmarkSimdCommandF64LinearBulk(Blackhole bh) {
        simdLinearExpr.applyBulk(variables, outputBuffer);
        bh.consume(checksum(outputBuffer));
    }

    @Benchmark
    public void benchmarkJaninoGaussianBulk(Blackhole bh) {
        janinoGaussianExpr.applyBulk(variables, outputBuffer);
        bh.consume(checksum(outputBuffer));
    }

    @Benchmark
    public void benchmarkSimdCommandF64GaussianBulk(Blackhole bh) {
        simdGaussianExpr.applyBulk(variables, outputBuffer);
        bh.consume(checksum(outputBuffer));
    }

    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(JaninoVsSIMDCommandF64Benchmark.class.getSimpleName());

        Options configurations = opt.mode(Mode.AverageTime)
                .timeUnit(TimeUnit.NANOSECONDS)
                .warmupIterations(3)
                .warmupTime(TimeValue.seconds(2))
                .measurementIterations(5)
                .measurementTime(TimeValue.seconds(5))
                .forks(3)
                .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                .jvmArgs("-Xms8g", "-Xmx8g", "-Dbenchmark.index=1")
                .jvmArgsAppend("--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions")
                .build();

        new Runner(configurations).run();
    }
}