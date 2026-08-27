package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.simdext.turbo.tools.SIMDEngineEvaluator;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmark comparing every storage path {@link SIMDEngineEvaluator}
 * now supports, on two representative expressions:
 *
 * <ul>
 *   <li><b>EXPR_ARITH</b> = {@code (x * y + x) / (y + 1)} — pure arithmetic,
 *       no transcendentals; mainly stresses load/store and the vectorized
 *       add/sub/mul/div fast paths.</li>
 *   <li><b>EXPR_TRIG</b> = {@code sin(x) * cos(y) + sqrt(x * x + y * y)} —
 *       exercises the transcendental (SVML/lanewise) paths in
 *       {@code VectorMath}/{@code VectorMathF}.</li>
 * </ul>
 *
 * For each expression, all seven storage paths are benchmarked:
 * {@code float[]}, {@code double[]}, {@code double[][]}, {@code float[][]},
 * {@code MemorySegment} of packed floats, {@code MemorySegment} of packed
 * doubles, {@code MemorySegment[]} of packed floats (one segment per
 * variable), and {@code MemorySegment[]} of packed doubles.
 *
 * <p>Run standalone with:
 * <pre>
 *   java -jar benchmarks.jar SIMDEngineEvaluatorBenchmark
 * </pre>
 * or via the {@code main} method below, or wire it into your existing JMH
 * build (annotation processor requires the {@code jmh-generator-annprocess}
 * dependency on the compile classpath).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {
        "--add-modules=jdk.incubator.vector",
        "--enable-preview"
})
public class SIMDEngineFloatBenchmark {

    private static final String EXPR_ARITH = "(x * y + x) / (y + 1)";
    private static final String EXPR_TRIG = "sin(x) * cos(y) + sqrt(x * x + y * y)";

    /**
     * Number of samples per invocation. 65536 is large enough to amortize
     * per-call overhead and to land comfortably above
     * {@code PARALLEL_OPS_THRESHOLD}, but the benchmarks below deliberately
     * call the single-threaded {@code applyBulk*} entry points (not
     * {@code applyBulkParallel*}) so the comparison isolates the
     * storage-format cost rather than thread-pool scheduling. Change to
     * {@code applyBulkParallel*} calls if you want to benchmark the
     * multi-threaded dispatch instead.
     */
    @Param({"65536"})
    public int n;

    // Compiled evaluators — one instance per expression, reused across all
    // storage-format benchmarks for that expression to avoid recompilation
    // overhead skewing results.
    private SIMDEngineEvaluator.SIMDVectorCompositeExpression arithEval;
    private SIMDEngineEvaluator.SIMDVectorCompositeExpression trigEval;

    private Arena arena;

    // ---- float[] (flat, 2 variables concatenated) ----
    private float[] flatF;
    private float[] outF;

    // ---- double[] (flat, 2 variables concatenated) ----
    private double[] flatD;
    private double[] outD;

    // ---- double[][] ----
    private double[][] vars2D;
    private double[] out2D;

    // ---- float[][] ----
    private float[][] vars2DF;
    private float[] out2DF;

    // ---- MemorySegment, packed floats (concatenated) ----
    private MemorySegment segFlatF;
    private MemorySegment segOutF;

    // ---- MemorySegment, packed doubles (concatenated) ----
    private MemorySegment segFlatD;
    private MemorySegment segOutD;

    // ---- MemorySegment[], packed floats (one per variable) ----
    private MemorySegment[] segArrF;
    private MemorySegment segArrOutF;

    // ---- MemorySegment[], packed doubles (one per variable) ----
    private MemorySegment[] segArrD;
    private MemorySegment segArrOutD;

    @Setup(Level.Trial)
    public void setup() throws Throwable {
        arithEval = SIMDEngineEvaluator.getEvaluator(EXPR_ARITH);
        trigEval = SIMDEngineEvaluator.getEvaluator(EXPR_TRIG);

        arena = Arena.ofShared();

        float[] xF = new float[n];
        float[] yF = new float[n];
        double[] xD = new double[n];
        double[] yD = new double[n];
        for (int i = 0; i < n; i++) {
            // Keep values well inside safe domains for sqrt/sin/cos/div.
            float fx = 0.01f + (i % 4096) * 0.001f;
            float fy = 1.0f + (i % 2048) * 0.0005f;
            xF[i] = fx;
            yF[i] = fy;
            xD[i] = fx;
            yD[i] = fy;
        }

        // float[]
        flatF = new float[2 * n];
        System.arraycopy(xF, 0, flatF, 0, n);
        System.arraycopy(yF, 0, flatF, n, n);
        outF = new float[n];

        // double[]
        flatD = new double[2 * n];
        System.arraycopy(xD, 0, flatD, 0, n);
        System.arraycopy(yD, 0, flatD, n, n);
        outD = new double[n];

        // double[][] / float[][]
        vars2D = new double[][]{xD, yD};
        out2D = new double[n];
        vars2DF = new float[][]{xF, yF};
        out2DF = new float[n];

        // MemorySegment, packed floats (concatenated: x block then y block)
        segFlatF = arena.allocate((long) 2 * n * Float.BYTES);
        MemorySegment.copy(xF, 0, segFlatF, ValueLayout.JAVA_FLOAT, 0L, n);
        MemorySegment.copy(yF, 0, segFlatF, ValueLayout.JAVA_FLOAT, (long) n * Float.BYTES, n);
        segOutF = arena.allocate((long) n * Float.BYTES);

        // MemorySegment, packed doubles (concatenated)
        segFlatD = arena.allocate((long) 2 * n * Double.BYTES);
        MemorySegment.copy(xD, 0, segFlatD, ValueLayout.JAVA_DOUBLE, 0L, n);
        MemorySegment.copy(yD, 0, segFlatD, ValueLayout.JAVA_DOUBLE, (long) n * Double.BYTES, n);
        segOutD = arena.allocate((long) n * Double.BYTES);

        // MemorySegment[], packed floats (one segment per variable — zero-copy)
        MemorySegment segX_F = arena.allocate((long) n * Float.BYTES);
        MemorySegment segY_F = arena.allocate((long) n * Float.BYTES);
        MemorySegment.copy(xF, 0, segX_F, ValueLayout.JAVA_FLOAT, 0L, n);
        MemorySegment.copy(yF, 0, segY_F, ValueLayout.JAVA_FLOAT, 0L, n);
        segArrF = new MemorySegment[]{segX_F, segY_F};
        segArrOutF = arena.allocate((long) n * Float.BYTES);

        // MemorySegment[], packed doubles (one segment per variable)
        MemorySegment segX_D = arena.allocate((long) n * Double.BYTES);
        MemorySegment segY_D = arena.allocate((long) n * Double.BYTES);
        MemorySegment.copy(xD, 0, segX_D, ValueLayout.JAVA_DOUBLE, 0L, n);
        MemorySegment.copy(yD, 0, segY_D, ValueLayout.JAVA_DOUBLE, 0L, n);
        segArrD = new MemorySegment[]{segX_D, segY_D};
        segArrOutD = arena.allocate((long) n * Double.BYTES);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (arithEval != null) {
            arithEval.close();
        }
        if (trigEval != null) {
            trigEval.close();
        }
        if (arena != null) {
            arena.close();
        }
    }

    // =====================================================================
    // EXPR_ARITH: (x * y + x) / (y + 1)
    // =====================================================================

    @Benchmark
    public void arith_floatArray(Blackhole bh) {
        arithEval.applyBulk(flatF, outF);
        bh.consume(outF);
    }

    @Benchmark
    public void arith_doubleArray(Blackhole bh) {
        arithEval.applyBulk(flatD, outD);
        bh.consume(outD);
    }

    @Benchmark
    public void arith_doubleArray2D(Blackhole bh) {
        arithEval.applyBulk(vars2D, out2D);
        bh.consume(out2D);
    }

    @Benchmark
    public void arith_floatArray2D(Blackhole bh) {
        arithEval.applyBulk(vars2DF, out2DF);
        bh.consume(out2DF);
    }

    @Benchmark
    public void arith_memSegFloat(Blackhole bh) {
        arithEval.applyBulkFloat(segFlatF, segOutF);
        bh.consume(segOutF);
    }

    @Benchmark
    public void arith_memSegDouble(Blackhole bh) {
        arithEval.applyBulk(segFlatD, segOutD);
        bh.consume(segOutD);
    }

    @Benchmark
    public void arith_memSegArrayFloat(Blackhole bh) {
        arithEval.applyBulkFloat(segArrF, segArrOutF);
        bh.consume(segArrOutF);
    }

    @Benchmark
    public void arith_memSegArrayDouble(Blackhole bh) {
        arithEval.applyBulk(segArrD, segArrOutD);
        bh.consume(segArrOutD);
    }

    // =====================================================================
    // EXPR_TRIG: sin(x) * cos(y) + sqrt(x * x + y * y)
    // =====================================================================

    @Benchmark
    public void trig_floatArray(Blackhole bh) {
        trigEval.applyBulk(flatF, outF);
        bh.consume(outF);
    }

    @Benchmark
    public void trig_doubleArray(Blackhole bh) {
        trigEval.applyBulk(flatD, outD);
        bh.consume(outD);
    }

    @Benchmark
    public void trig_doubleArray2D(Blackhole bh) {
        trigEval.applyBulk(vars2D, out2D);
        bh.consume(out2D);
    }

    @Benchmark
    public void trig_floatArray2D(Blackhole bh) {
        trigEval.applyBulk(vars2DF, out2DF);
        bh.consume(out2DF);
    }

    @Benchmark
    public void trig_memSegFloat(Blackhole bh) {
        trigEval.applyBulkFloat(segFlatF, segOutF);
        bh.consume(segOutF);
    }

    @Benchmark
    public void trig_memSegDouble(Blackhole bh) {
        trigEval.applyBulk(segFlatD, segOutD);
        bh.consume(segOutD);
    }

    @Benchmark
    public void trig_memSegArrayFloat(Blackhole bh) {
        trigEval.applyBulkFloat(segArrF, segArrOutF);
        bh.consume(segArrOutF);
    }

    @Benchmark
    public void trig_memSegArrayDouble(Blackhole bh) {
        trigEval.applyBulk(segArrD, segArrOutD);
        bh.consume(segArrOutD);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(SIMDEngineFloatBenchmark.class.getSimpleName())
                .build();
        new Runner(opt).run();
    }
}