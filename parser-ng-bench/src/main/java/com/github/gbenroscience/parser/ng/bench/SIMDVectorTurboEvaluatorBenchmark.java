package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import org.openjdk.jmh.annotations.*;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/**
 * Run from PowerShell for benchmarking with cpu pinning
 * cmd.exe /c start /affinity 0xFF /wait java -jar target/benchmarks.jar 'VectorTurboEvaluatorBenchmark'
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
public class SIMDVectorTurboEvaluatorBenchmark {

    @Param({"1000", "1000000", "67108864"})
    private int dataSize;

    @Param({
        "12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2", 
        "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))", 
        "sin(z-x)+3*sin(5*x^2 + 4*y^2)"
    })
    private String expression;

    private double[] flatVariables;
    private double[][] variables;
    private double[] outputBuffer;

    private SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression parserNG;
    private VectorTurboBench.JaninoMathFunction fastEvaluator;
    
    private int varCount;
    private double[] vars;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        Random rand = new Random(42);

        MathExpression me = new MathExpression(expression);
        String[] expressionVars = me.getVariablesNames();
        this.varCount = expressionVars.length;
        
        // Structure of Arrays (SoA) Allocation
        variables = new double[varCount][dataSize];
        flatVariables = new double[varCount * dataSize];
        outputBuffer = new double[dataSize];
        vars = new double[varCount];

        // Micro-optimized random initialization without string allocation overhead
        // Uses values from 1.0 to 9.0 to safely avoid division-by-zero exceptions in evaluations
        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < varCount; j++) {
                variables[j][i] = 1.0 + rand.nextInt(9);
            }
        }
        
        // Copy into a flat SoA buffer block
        for (int i = 0; i < variables.length; i++) {
            int sz = variables[i].length;
            System.arraycopy(variables[i], 0, flatVariables, i * sz, sz);
        }

        parserNG = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        fastEvaluator = VectorTurboBench.setupJanino(expression, expressionVars);
    }

    @Benchmark
    public void parserNG2DParallel(Blackhole bh) {
        
        final double[][] src = this.variables;
        final double[] out = this.outputBuffer;
        parserNG.applyBulkParallel(src, out);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
       /* double checksum = 0.0;
        for (int i = 0; i < out.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += out[i];
        }*/
        bh.consume(out);
    }

    @Benchmark
    public void parserNG2D(Blackhole bh) {
        
        final double[][] src = this.variables;
        final double[] out = this.outputBuffer;
        parserNG.applyBulk(src, out);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
       /* double checksum = 0.0;
        for (int i = 0; i < out.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += out[i];
        }*/
        bh.consume(out);
    }
    
        @Benchmark
    public void parserNG(Blackhole bh) {
        
        final double[] src = this.flatVariables;
        final double[] out = this.outputBuffer;
        parserNG.applyBulk(src, out);

        // FORCES THE JIT TO EXECUTE EVERY LOOP STEP:
        // By calculating a hash sum across the output, the compiler cannot optimize away intermediate indices.
    /* double checksum = 0.0;
        for (int i = 0; i < out.length; i += 64) { // Sample memory lines to reduce benchmark overhead
            checksum += out[i];
        }*/
        bh.consume(out);
    }

    @Benchmark
    public void janino(Blackhole bh) {
        // Localizing fields to minimize register loading inside the benchmark loop
        final int limit = this.dataSize;
        final int stride = this.varCount;
        final double[][] src = this.variables;
        final double[] currentVars = this.vars;
        final VectorTurboBench.JaninoMathFunction evaluator = this.fastEvaluator;

        for (int i = 0; i < limit; i++) {
            // Correctly reconstruct cross-sections out of the SoA data layout
            for (int j = 0; j < stride; j++) {
                currentVars[j] = src[j][i];
            }
            bh.consume(evaluator.apply(currentVars));
        }
    }

    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(SIMDVectorTurboEvaluatorBenchmark.class.getSimpleName());
        
        Options configurations = opt.mode(Mode.AverageTime) // Keep uniform with class-level metrics
                .timeUnit(TimeUnit.NANOSECONDS)
                .warmupIterations(3)
                .warmupTime(TimeValue.seconds(2))
                .measurementIterations(5)
                .measurementTime(TimeValue.seconds(2))
                .forks(1)
                .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                .jvmArgs("-Xms8g", "-Xmx8g", "-Dbenchmark.index=1")
                .jvmArgsAppend("--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions")
                .build();

        new Runner(configurations).run();
    }
}