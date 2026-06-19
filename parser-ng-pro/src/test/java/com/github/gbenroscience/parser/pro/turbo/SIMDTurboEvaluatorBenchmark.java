package com.github.gbenroscience.parser.pro.turbo;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.SIMDVectorTurboEvaluator;
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
public class SIMDTurboEvaluatorBenchmark {

    @Param({"65536", "524288", "67108864"})
    private int dataSize;

    @Param({"true", "false"})
    private boolean tiledExecution;


    private double[] flatVariables;
    private double[][] variables;
    private double[] outputBuffer;

    private SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression linearExpr;
    private SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression gaussianExpr;
    private SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression conditionalExpr;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        Random rand = new Random(42);
      
        // Structure of Arrays (SoA): 3 variables (x1, x2, x3), each of length dataSize
        int stride = 3;
        variables = new double[stride][dataSize];
        flatVariables = new double[stride * dataSize];
        outputBuffer = new double[dataSize];

        for (int i = 0; i < dataSize; i++) {
            // 1. Populate the 2D SoA layout for the standard evaluator
            variables[0][i] = 1.5 + rand.nextDouble() * 5.0;  // x1
            variables[1][i] = rand.nextDouble() * 5.0;        // x2
            variables[2][i] = rand.nextDouble() * 2.0;        // x3

            // 2. Calculate the flat interleaved base address for timestep 'i'
            // For i=0, base=0. For i=1, base=3. For i=2, base=6...
            int base = i * stride;

            // 3. Populate the flat interleaved array perfectly
            for (int k = 0; k < stride; k++) {
                flatVariables[base + k] = variables[k][i];
            }
        }

        // Compile expressions using the Vector Engine
        MathExpression meLinear = new MathExpression("12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2");
        linearExpr = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(meLinear).compile();

        //MathExpression meGaussian = new MathExpression("(1 / (x1 * sqrt(2 * 3.141592653589793))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        MathExpression meGaussian = new MathExpression("0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))");
        gaussianExpr = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(meGaussian).compile();

        /*
        MathExpression meConditional = new MathExpression("if(x1 >= 2.5, sin(x1) % x2, x3 * vma(x1, x2, 1.5))");
        conditionalExpr = (SIMDCompositeExpression) new VectorTurboEvaluator(meConditional).compile();
         */
    }

    /*
    @Benchmark
    public void benchmarkLinearPolynomialBulk(org.openjdk.jmh.infra.Blackhole bh) {
        linearExpr.applyBulk(variables, outputBuffer);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
        double checksum = 0.0;
        for (int i = 0; i < outputBuffer.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += outputBuffer[i];
        }
        bh.consume(checksum);
    }

    @Benchmark
    public void benchmarkGaussianDistributionBulk(org.openjdk.jmh.infra.Blackhole bh) {
        gaussianExpr.applyBulk(variables, outputBuffer);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
        double checksum = 0.0;
        for (int i = 0; i < outputBuffer.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += outputBuffer[i];
        }
        bh.consume(checksum);
    }
     */

    @Benchmark
    public void benchmarkLinearPolynomialBulkFlatVars(org.openjdk.jmh.infra.Blackhole bh) {
        linearExpr.applyBulk(flatVariables, outputBuffer, tiledExecution);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
        double checksum = 0.0;
        for (int i = 0; i < outputBuffer.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += outputBuffer[i];
        }
        bh.consume(checksum);
    }

    @Benchmark
    public void benchmarkGaussianDistributionBulkFlatVars(org.openjdk.jmh.infra.Blackhole bh) {
        gaussianExpr.applyBulk(flatVariables, outputBuffer, tiledExecution);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
        double checksum = 0.0;
        for (int i = 0; i < outputBuffer.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += outputBuffer[i];
        }
        bh.consume(checksum);
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
        opt.include(SIMDTurboEvaluatorBenchmark.class.getSimpleName()); // Always include baseline
        // 4. Fluent, modern JMH Configuration
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
