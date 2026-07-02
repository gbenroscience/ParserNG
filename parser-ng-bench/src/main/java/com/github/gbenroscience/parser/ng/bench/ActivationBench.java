package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import org.openjdk.jmh.annotations.*;
import com.github.gbenroscience.simd.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.simd.turbo.tools.FlatMatrixF;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/**
 * Use this to build in the parser-ng-pro directory: mvn clean install -Dgpg.skip
Use this to run the benchmarks:

java -jar target/benchmarks.jar ActivationBench -prof perfasm
 *
 * @author GBEMIRO
 */
@BenchmarkMode(Mode.AverageTime) 
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Fork(value = 3, jvmArgsAppend = {
    "-Xms4g", "-Xmx4g",
    "-XX:+UnlockDiagnosticVMOptions",
    "--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions"
})
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 2)
@OperationsPerInvocation(ActivationBench.N) // Tells JMH to divide total time by N to output ns/element
public class ActivationBench {

    // Must be a compile-time constant to be used in the annotation above.
    // 16,777,216 elements represents a square matrix of 4096 x 4096.
    public static final int N = 4096 * 4096; 
    
    private final int sz = (int) Math.sqrt(N);
    private SIMDCompositeExpression evaluator;
    private FlatMatrixF in1;
    private FlatMatrixF in2;
    private FlatMatrixF out;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        MathExpression me = new MathExpression("x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))");
        evaluator = (SIMDCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        in1 = new FlatMatrixF(sz, sz);
        in2 = new FlatMatrixF(sz, sz);
        out = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in1);
        FlatMatrixF.randomFill(in2);
    }

    @Setup(Level.Iteration) // C2 can't optimize across iterations
    public void dirtyInput() {
        // JMH calls this before each measurement iteration to make inputs unpredictable
        for (int j = 0; j < N; j++) {
            in1.data[j] = ThreadLocalRandom.current().nextFloat();
            in2.data[j] = in1.data[j] * 2.13f;
        }
    }

    @Benchmark
    public void gelu(Blackhole bh) {
        evaluator.applyMatrixKernel(new FlatMatrixF[]{in1}, out, "gelu");
        bh.consume(out); // Completely safe, zero-overhead way to prevent Dead Code Elimination
    }

    @Benchmark
    public void swiglu(Blackhole bh) {
        evaluator.applyMatrixKernel(new FlatMatrixF[]{in1, in2}, out, "swiglu");
        bh.consume(out); 
    }
    
    @Benchmark
    public void geglu(Blackhole bh) {
        evaluator.applyMatrixKernel(new FlatMatrixF[]{in1, in2}, out, "geglu");
        bh.consume(out); 
    }
    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(ActivationBench.class.getSimpleName());

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