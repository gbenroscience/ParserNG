package com.github.gbenroscience.parser.ng.bench;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/**
 * Use this to build in the parser-ng-pro directory mvn clean install -Dgpg.skip
 * Use this to run the benchmarks
 *
 * java -jar target/benchmarks.jar SIMDTurboBench -prof perfasm
 *
 * java -jar target/benchmarks.jar SIMDTurboBench -prof perfasm >
 * perf_output.txt Now find your specific method's assembly grep -A 50
 * "parserNG" perf_output.txt
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
public class VectorTurboBenchStrict {

    @Param({"12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2", "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))"})
    String expression;

    //static final String expression = "x1^3+x2^3+x3^3+x4^3+x5^3+x6^3";
    double[] result;

    @Param({"20000000"})
    private int dataSize;

    private double[] vars;
    private VectorTurboBench.JaninoMathFunction fastEvaluator;

  private VectorTurboEvaluator.BatchedVectorCompositeExpression vectorTurbo;
    private String[] expressionVars;
    private int varCount;

    private static double[][][] dataSink;

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
            // Allocate SoA: [5 scenarios][varCount variables][dataSize samples per variable]
            dataSink = new double[5][varCount][dataSize];

            for (int i = 0; i < dataSink.length; i++) {
                for (int v = 0; v < varCount; v++) {
                    for (int s = 0; s < dataSize; s++) {
                        dataSink[i][v][s] = r.nextDouble(20);
                    }
                }
            }
            System.out.println("Repopulated data sink with SoA layout: " + varCount + " variables per track.");
        }

        this.fastEvaluator = VectorTurboBench.setupJanino(expression, expressionVars);

        setupParserNG(me);

        // Validation against the SoA structure
        for (int i = 0; i < dataSink.length; i++) {
            // Ensure your validate method is updated to handle double[][] 
            // instead of a single flat array
            vectorTurbo.validate(dataSink[i], result);
        }
    }

   @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void janino(Blackhole bh) {
        final int limit = dataSize;
        final int stride = varCount;
        // Select one scenario (SoA matrix)
        final double[][] input = dataSink[ThreadLocalRandom.current().nextInt(dataSink.length)];
        final double[] localVars = this.vars;
        final VectorTurboBench.JaninoMathFunction evaluator = this.fastEvaluator;

        for (int i = 0; i < limit; i++) {
            // Linear access: read each variable for the current sample 'i'
            for (int j = 0; j < stride; j++) {
                localVars[j] = input[j][i];
            }
            bh.consume(evaluator.apply(localVars));
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void parserNG(Blackhole bh) {
        // Select one scenario (SoA matrix)
        // Ensure simdVectorTurbo.applyBulk is updated to accept double[][]
        final double[][] input = dataSink[ThreadLocalRandom.current().nextInt(dataSink.length)];

        // Measure execution of the SoA-optimized kernel
        vectorTurbo.applyBulk(input, result);

        bh.consume(result);
    }

    private void setupParserNG(MathExpression me) {
        try {
            vectorTurbo = (VectorTurboEvaluator.BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();
        } catch (Throwable ex) {
            System.getLogger(VectorTurboBenchStrict.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
    }

    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(VectorTurboBenchStrict.class.getSimpleName()); // Always include baseline
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
