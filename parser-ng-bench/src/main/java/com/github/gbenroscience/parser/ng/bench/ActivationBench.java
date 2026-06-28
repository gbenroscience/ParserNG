package com.github.gbenroscience.parser.ng.bench;

/**
 *
 * @author GBEMIRO
 */
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
@BenchmarkMode(Mode.AverageTime) // ns/op
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Fork(value = 3, jvmArgsAppend = {
    "-Xms4g", "-Xmx4g",
    "-XX:+UnlockDiagnosticVMOptions",
    "--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions"
// "-XX:+PrintAssembly", "-XX:CompileCommand=print,*SIMDCompositeExpression.applyMatrixKernel" // uncomment if you have hsdis
})
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 2)
public class ActivationBench {

    @Param({"512", "1024", "2048", "4096"})
    public int sz;

    private SIMDCompositeExpression evaluator;
    private FlatMatrixF in1;
    private FlatMatrixF in2;
    private FlatMatrixF out;
    private int N;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        MathExpression me = new MathExpression("x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))");
        evaluator = (SIMDCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

        N = sz * sz;
        in1 = new FlatMatrixF(sz, sz);
        in2 = new FlatMatrixF(sz, sz);
        out = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in1);
        FlatMatrixF.randomFill(in2);
    }

    @Setup(Level.Iteration) // C2 can't optimize across iterations
    public void dirtyInput() {
        // JMH will call this before each measurement iteration
        // This makes the input unprovable without timing it
        for (int j = 0; j < N; j++) {
            in1.data[j] = ThreadLocalRandom.current().nextFloat();
            in2.data[j] = in1.data[j]*2.13f;
        }
    }

    @Benchmark
    public void gelu(Blackhole bh) {
        evaluator.applyMatrixKernel(new FlatMatrixF[]{in1}, out, "gelu");
        // Consume the whole output so C2 can't DCE
        /*
        for (int j = 0; j < N; j++) {
            bh.consume(Float.floatToRawIntBits(out.data[j]));
        }
        */
         bh.consume(out.data[0]);
    }
        @Benchmark
    public void swiglu(Blackhole bh) {
        evaluator.applyMatrixKernel(new FlatMatrixF[]{in1, in2}, out, "swiglu");
        // Consume the whole output so C2 can't DCE
               bh.consume(out.data[0]);
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
