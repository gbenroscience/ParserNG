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
/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression; 
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.utils.MathToJaninoConverter;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.codehaus.janino.ClassBodyEvaluator;
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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Phaser;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/**
 * JMH Benchmark comparing ParserNG, Janino on bulk evaluations Focus: repeated
 * evaluation of the same pre-compiled expression.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 3, jvmArgs = {
    "-Xms5g", "-Xmx5g",
    "-XX:+UseG1GC",
    "-XX:-UseCompressedOops", // Avoids compressed oops artifacts
    "--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions"
})
@Threads(1)
public class ParserNgBench {

    public static interface JaninoMathFunction {

        double apply(double[] v);
    }

    private ExecutorService janinoParallelExecutor;
    private Phaser barrier;

    // Thread-confined variable cache lines
    private double[] workerLocalVars;
    private double[] masterLocalVars;

    // Zero-allocation side-effect sinks to prevent JIT dead-code elimination
    private double[] workerResultSink;
    private double[] masterResultSink;

    // Pre-compiled instances (initialized in @Setup)
    protected MathExpression parserNG;

    // The expression to benchmark
    public static final String[] EXPRESSIONS = {
        "(sin(3) + cos(4 - sin(2))) ^ (-2)",
        "sin(3)+cos(5)-2.718281828459045^2",
        "((12+5)*3 - 2^3-13/12.23)^3.2",
        "5*sin(3+2)/(4*3-2)",
        "(1+1)*(1+2)*(3+4)*(8+9)*(6-1)*(4^3.14159265357)-(3+2)^1.8",
        "(sin(8+cos(3)) + 2 + ((27-5)/(8^3) * (3.14159 * 4^(14-10)) + sin(-3.141) + (0%4)) * 4/3 * 3/sqrt(4))+12",
        "((x1^2 + sin(x1)) / (1 + cos(x1^2))) * (exp(x1) / 10)",
        "((x1^2 + 3*sin(x1+5^3-1/4)) / (23/33 + cos(x1^2))) * (exp(x1) / 10)",
        "exp(5*4*3*2*1)",
        "1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20",
        "1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+sin(x1)",
        "2+3*4-5/2+sin(0)+cos(0)+sqrt(16)",
        "sin(7*x1+x2)+cos(7*x1-x2)-sin(4)+cos(5^6)",
        "((x1^2 + 3*sin(x1+5^3-1/4)) / (23/33 + cos(x1^2))) * (exp(x1) / 10) + (sin(3) + cos(4 - sin(2))) ^ (-2)",
        "(x1^2+x2^0.5)^4.2",
        "sin(x1^3+x2^3)-4*(x1-x2)",
        "(x1+x2+x3)^0+(x1+x2+x3)^1+(x1+x2+x3)^2+(x1+x2+x3)^3+(x1+x2+x3)^4+(x1+x2+x3)^5+(x1+x2+x3)^6+(x1+x2+x3)^7",
        "((x1^2 + 3*sin(x1+5^3-1/4+5*x2)) / (23/33 + cos(x1^2))) * (exp(x1+2*x3^2) / 10)",
        "sin(x1)+3*cos(x1)-4*x1^2-8*x1^3+9/(x1+1)+5*(x1-1)^3+12*x2",
        "sin((x1+x2+x3)^3.14)",
        "x1+x2+x3",
        "x1+x2+x3+sin(2)-cos(4)+exp(2^5)",
        "(x1+x2+x3)/(x1-x2+x3)",
        "sin((x1+x2+x3)/(x1-x2+x3))^3.14159265357",
        "sin(x1)+sin(x2)+sin(x3)-sin(x1+1)-sin(x1-1.1)-sin(x2-1)-sin(x2-1.1)+sin(x3+1)+sin(x3+2)+sin(x3+3*x1*x2*x3)",
        "sin(x1)+sin(x2)+sin(x3)-sin(x1+1)-sin(x1-1.1)-sin(x2-1)-sin(x2-1.1)+sin(x3+1)+sin(x3+2)+sin(x3+3*x1*x2*x3)+sin(x1)+sin(x2)+sin(x3)-sin(x1+1)-sin(x1-1.1)-sin(x2-1)-sin(x2-1.1)+sin(x3+1)+sin(x3+2)+sin(x3+3*x1*x2*x3)",
        "cos(x1+x2-5*x3-x4-2*x5)+sin(2*x1+4*x2-5*x3-x4-2*x5)",
        "cos(12*x1+3*x2-4*x3+5*x4-x5-4*x6+2*x7+x8-5*x9-x10-2*x11)+sin(2*x7+4*x8-5*x9^2-3*x10-2*x11)+sin(x9+x10-x7)+cos(x1+x2+x3)+12*x4",
        "sin(12*x1+3*x2-4*x3+5*x4-x5-4*x6+2*x7+x8-5*x9-x10-2*x11)+sin(2)-cos(3)+tan(1.5)-sinh(4.22)+cos(4.15)",
        "(12*x1+3*x2-4*x3+5*x4-x5-4*x6+2*x7+x8-5*x9-x10-2*x11)",
        "(x1^2/sin(2*3.14159265357/x2))-x1/2",
        "(cos(1+exp(x1))/sqrt(sin(x1)^2-cos(x1)^2))+atan(x1)",
        "x1^3+x2^3+x3^3+x4^3",
        "x1^3.21+x2^3.14+x3^3+x4^3+x5^3+x6^3",
        "(sin(x1^3)-cos(x1^4)+tan(x1^0.5))/(2*x1^2+1)",
        "(sin(x1) + 2 + ((7-5) * (3.14159 * x1^(14-10)) + sin(-3.141) + (0%x1)) * x1/3 * 3/sqrt(x1+12))",
        "x1^3+x2^3+x3^3+x4^3+x5^3+x6^3",
        "sin(sqrt(x1^2+x2^2+x3^2))",
        "x1/(1-x2)",
        "x1*exp(-(x2/(1-((x3-x4)/x3))))",
        "(1/(x1*sqrt(2*pi)))*exp((-(x2-x3)^2)/(2*x1^2))",
        "12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2",
        "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))",
        "(1/(x1*sqrt(2*pi)))*exp((-(x2-x3)-2)/(2*x1-2))",
        "(x1^2.5+(x1+2*x2)^3.14+4*x3^2.25+x4^1.98)",};

    static int index = 23;

    static {
        index = Integer.getInteger("benchmark.index", 23);
    }

    protected static final String EXPRESSION = getExpression();
    protected static String[] expressionVars = getVars(EXPRESSION);

    public static final String[] getVars(String e) {
        return new MathExpression(e).getVariablesNames();
    }
    protected int simpleCursor;

    String expression = EXPRESSION;

    double[] result;

    @Param({"2000000"})
    private int dataSize;

    private double[] vars;
    private JaninoMathFunction fastEvaluator;
 
    SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression simdVec;

    private int varCount;

    // 1. Add a single stable scenario index to your benchmark fields
    private int benchmarkScenario = 0;

    //SoA dataSink layout for ParserNG
    private static double[][][] dataSink;
    // Pre-flattened Array of Structures [scenario][sampleIndex][variables]
    private static double[][][] janinoDataSink;

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
            dataSink = new double[5][varCount][dataSize];
            // 2. Allocate the AoS format for Janino
            janinoDataSink = new double[5][dataSize][varCount];

            for (int i = 0; i < dataSink.length; i++) {
                for (int v = 0; v < varCount; v++) {
                    for (int s = 0; s < dataSize; s++) {
                        double val = r.nextDouble(20);
                        // Populate ParserNG Data
                        dataSink[i][v][s] = val;
                        // Populate Janino Data (Sequential Sample Row Layout)
                        janinoDataSink[i][s][v] = val;
                    }
                }
            }
            System.out.println("Repopulated both SoA and AoS data layouts successfully.");
        }

        // Exactly 2 permanent parties: 1 Master JMH thread + 1 Dedicated Background Worker
        this.barrier = new Phaser(2);

        // Pinned state blocks to isolate L1 data caches
        this.workerLocalVars = new double[varCount];
        this.masterLocalVars = new double[varCount];

        // Result sinks sized exactly to the split data constraints
        final int mid = dataSize / 2;
        this.workerResultSink = new double[mid];
        this.masterResultSink = new double[dataSize - mid];

        this.janinoParallelExecutor = Executors.newFixedThreadPool(1, v -> {
            Thread t = new Thread(v, "Janino-Parallel-Worker");
            t.setDaemon(true);
            return t;
        });

        this.fastEvaluator = setupJanino(expression, expressionVars);

        setupParserNG(me);

        // Validation against the SoA structure
        for (int i = 0; i < dataSink.length; i++) { 
            simdVec.validate(dataSink[i], result);
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (janinoParallelExecutor != null) {
            janinoParallelExecutor.shutdown();
            try {
                if (!janinoParallelExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    janinoParallelExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Setup(Level.Iteration)
    public void chooseIterationScenario() {
        // Pick or rotate a matrix ONLY between iterations, never inside the hot loop
        this.benchmarkScenario = ThreadLocalRandom.current().nextInt(5);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void janinoSoA(Blackhole bh) {
        final int limit = dataSize;
        final int stride = varCount;

        // Read from the static iteration-level scenario to ensure perfect JIT loop unrolling
        final double[][] input = dataSink[benchmarkScenario];
        final double[] localVars = this.vars;
        final JaninoMathFunction evaluator = this.fastEvaluator;

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
    public void janinoAoS(Blackhole bh) {
        final int limit = dataSize;
        // Read from a completely static scenario index
        final double[][] inputAoS = janinoDataSink[benchmarkScenario];
        final JaninoMathFunction evaluator = this.fastEvaluator;

        for (int i = 0; i < limit; i++) {
            bh.consume(evaluator.apply(inputAoS[i]));
        }
    }

    
    @Benchmark
    public void janinoAoSParallel(Blackhole bh) {
        final double[][] inputAoS = janinoDataSink[benchmarkScenario];
        final JaninoMathFunction evaluator = this.fastEvaluator;
        final int limit = dataSize;
        final int mid = limit / 2;
        
        final double[] wSink = this.workerResultSink;
        final double[] mSink = this.masterResultSink;

        // Dispatch chunk 1 to background loop
        janinoParallelExecutor.execute(() -> {
            try {
                for (int i = 0; i < mid; i++) {
                    wSink[i] = evaluator.apply(inputAoS[i]);
                }
            } finally {
                // Static arrival: decrements unarrived, waits if necessary, auto-resets phase
                barrier.arriveAndAwaitAdvance();
            }
        });

        // Master thread evaluates chunk 2 concurrently
        for (int i = mid; i < limit; i++) {
            mSink[i - mid] = evaluator.apply(inputAoS[i]);
        }

        // Master synchronization arrival barrier
        barrier.arriveAndAwaitAdvance();

        // Safe consumption on the native JMH harness thread post-barrier
        for (int i = 0; i < mid; i++) {
            bh.consume(wSink[i]);
        }
        for (int i = 0; i < mSink.length; i++) {
            bh.consume(mSink[i]);
        }
    }

    @Benchmark
    public void janinoSoAParallel(Blackhole bh) {
        final double[][] input = dataSink[benchmarkScenario];
        final int stride = varCount;
        final int limit = dataSize;
        final int mid = limit / 2;
        
        // Pin configurations to isolated array handles
        final double[] lvWorker = this.workerLocalVars;
        final double[] lvMaster = this.masterLocalVars;
        final double[] wSink = this.workerResultSink;
        final double[] mSink = this.masterResultSink;
        
        final JaninoMathFunction evaluator = this.fastEvaluator;

        janinoParallelExecutor.execute(() -> {
            try {
                for (int i = 0; i < mid; i++) {
                    for (int j = 0; j < stride; j++) {
                        lvWorker[j] = input[j][i];
                    }
                    wSink[i] = evaluator.apply(lvWorker);
                }
            } finally {
                barrier.arriveAndAwaitAdvance();
            }
        });

        // Master Thread Processing Segment
        for (int i = mid; i < limit; i++) {
            for (int j = 0; j < stride; j++) {
                lvMaster[j] = input[j][i];
            }
            mSink[i - mid] = evaluator.apply(lvMaster);
        }

        // Structural Thread Rendezvous Barrier
        barrier.arriveAndAwaitAdvance();

        // Linearized single-threaded Blackhole consumption phase
        for (int i = 0; i < mid; i++) {
            bh.consume(wSink[i]);
        }
        for (int i = 0; i < mSink.length; i++) {
            bh.consume(mSink[i]);
        }
    }
  



    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void simdVec(Blackhole bh) {
        // Read from the same stable scenario index—zero cursor modifications inside!
        final double[][] input = dataSink[benchmarkScenario];
        final double[] res = result;

        // Execute core computation kernel
        simdVec.applyBulk(input, res);
        bh.consume(res);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void simdVecParallel(Blackhole bh) {
        // Read from the same stable scenario index—zero cursor modifications inside!
        final double[][] input = dataSink[benchmarkScenario];
        final double[] res = result;

        // Execute core computation kernel
        simdVec.applyBulkParallel(input, res);
        bh.consume(res);
    }

    private void setupParserNG(MathExpression me) {
        try {
             //simdVec = SIMDVectorTurboEvaluator.getEvaluator(me, 2);
               simdVec = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me.copy()).compile();
        } catch (Throwable ex) {
            System.getLogger(ParserNgBench.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
    }

    public static final String getExpression() {
        return EXPRESSIONS[index];
    }

    /**
     * Varying workloads for duel! Expression 34 triggers gc on the SIMD
     * evaluators
     */
    public static final String EXPRESSIONS_MAP
            = """
            0====(sin(3) + cos(4 - sin(2))) ^ (-2)
            1====sin(3)+cos(5)-2.718281828459045^2
            2====((12+5)*3 - 2^3-13/12.23)^3.2
            3====5*sin(3+2)/(4*3-2)
            4====(1+1)*(1+2)*(3+4)*(8+9)*(6-1)*(4^3.14159265357)-(3+2)^1.8
            5====(sin(8+cos(3)) + 2 + ((27-5)/(8^3) * (3.14159 * 4^(14-10)) + sin(-3.141) + (0%4)) * 4/3 * 3/sqrt(4))+12
            6====((x1^2 + sin(x1)) / (1 + cos(x1^2))) * (exp(x1) / 10)
            7====((x1^2 + 3*sin(x1+5^3-1/4)) / (23/33 + cos(x1^2))) * (exp(x1) / 10)
            8====exp(5*4*3*2*1)
            9====1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20
            10====1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+sin(x1)
            11====2+3*4-5/2+sin(0)+cos(0)+sqrt(16)
            12====sin(7*x1+x2)+cos(7*x1-x2)-sin(4)+cos(5^6)
            13====((x1^2 + 3*sin(x1+5^3-1/4)) / (23/33 + cos(x1^2))) * (exp(x1) / 10) + (sin(3) + cos(4 - sin(2))) ^ (-2)
            14====(x1^2+x2^0.5)^4.2
            15====sin(x1^3+x2^3)-4*(x1-x2)
            16====(x1+x2+x3)^0+(x1+x2+x3)^1+(x1+x2+x3)^2+(x1+x2+x3)^3+(x1+x2+x3)^4+(x1+x2+x3)^5+(x1+x2+x3)^6+(x1+x2+x3)^7
            17====((x1^2 + 3*sin(x1+5^3-1/4+5*x2)) / (23/33 + cos(x1^2))) * (exp(x1+2*x3^2) / 10)
            18====sin(x1)+3*cos(x1)-4*x1^2-8*x1^3+9/(x1+1)+5*(x1-1)^3+12*x2
            19====sin((x1+x2+x3)^3.14)
            20====x1+x2+x3
            21====x1+x2+x3+sin(2)-cos(4)+exp(2^5)
            22====(x1+x2+x3)/(x1-x2+x3)
            23====sin((x1+x2+x3)/(x1-x2+x3))^3.14159265357
            24====sin(x1)+sin(x2)+sin(x3)-sin(x1+1)-sin(x1-1.1)-sin(x2-1)-sin(x2-1.1)+sin(x3+1)+sin(x3+2)+sin(x3+3*x1*x2*x3)
            25====sin(x1)+sin(x2)+sin(x3)-sin(x1+1)-sin(x1-1.1)-sin(x2-1)-sin(x2-1.1)+sin(x3+1)+sin(x3+2)+sin(x3+3*x1*x2*x3)+sin(x1)+sin(x2)+sin(x3)-sin(x1+1)-sin(x1-1.1)-sin(x2-1)-sin(x2-1.1)+sin(x3+1)+sin(x3+2)+sin(x3+3*x1*x2*x3)
            26====cos(x1+x2-5*x3-x4-2*x5)+sin(2*x1+4*x2-5*x3-x4-2*x5)
            27====cos(12*x1+3*x2-4*x3+5*x4-x5-4*x6+2*x7+x8-5*x9-x10-2*x11)+sin(2*x7+4*x8-5*x9^2-3*x10-2*x11)+sin(x9+x10-x7)+cos(x1+x2+x3)+12*x4
            28====sin(12*x1+3*x2-4*x3+5*x4-x5-4*x6+2*x7+x8-5*x9-x10-2*x11)+sin(2)-cos(3)+tan(1.5)-sinh(4.22)+cos(4.15)
            29====(12*x1+3*x2-4*x3+5*x4-x5-4*x6+2*x7+x8-5*x9-x10-2*x11)
            30====(x1^2/sin(2*3.14159265357/x2))-x1/2
            31====(cos(1+exp(x1))/sqrt(sin(x1)^2-cos(x1)^2))+atan(x1)
            32====x1^3+x2^3+x3^3+x4^3
            33====x1^3.21+x2^3.14+x3^3+x4^3+x5^3+x6^3
            34====(sin(x1^3)-cos(x1^4)+tan(x1^0.5))/(2*x1^2+1)
            35====(sin(x1) + 2 + ((7-5) * (3.14159 * x1^(14-10)) + sin(-3.141) + (0%x1)) * x1/3 * 3/sqrt(x1+12))
            36====x1^3+x2^3+x3^3+x4^3+x5^3+x6^3
            37====sin(sqrt(x1^2+x2^2+x3^2))  
            38====x1/(1-x2)  
            39====x1*exp(-(x2/(1-((x3-x4)/x3)))) 
            40====(1/(x1*sqrt(2*pi)))*exp((-(x2-x3)^2)/(2*x1^2))
            41====12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2
            42====0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))
            43====(1/(x1*sqrt(2*pi)))*exp((-(x2-x3)-2)/(2*x1-2))
            44====(x1^2.5+(x1+2*x2)^3.14+4*x3^2.25+x4^1.98)  
            """;

    public static final StringBuilder EXPR_MAP = new StringBuilder();

    static {
        int i = 0;
        for (String e : EXPRESSIONS) {
            EXPR_MAP.append(i++).append("====").append(e).append("\n");
        }
    }

    public static JaninoMathFunction setupJanino(String expression, String[] expressionVars) {
        String javaExpr = MathToJaninoConverter.convert(expression);

        for (int i = 0; i < expressionVars.length; i++) {
            String varName = expressionVars[i];
            String regex = "\\b" + java.util.regex.Pattern.quote(varName) + "\\b";
            javaExpr = javaExpr.replaceAll(regex, "v[" + i + "]");
        }

        String classBody = String.format("""
        @Override
        public double apply(double[] v) {
            return %s;
        }
        """, javaExpr);

        try {
            ClassBodyEvaluator cbe = new ClassBodyEvaluator();
            cbe.setImplementedInterfaces(new Class[]{JaninoMathFunction.class});
            cbe.cook(classBody);
            return (JaninoMathFunction) cbe.getClazz().getDeclaredConstructor().newInstance();
        } catch (Exception ex) {
            throw new RuntimeException("Failed to compile Janino function body expressions", ex);
        }
    }

    /**
     * Run this to generate the expressions map
     * above({@linkplain ParserNGWars#EXPRESSIONS_MAP}) whenever the array is
     * updated. this will help users to know the index of the expression to
     * reference in the array from the benchmarks in
     * {@linkplain com.github.gbenroscience.parser.wars.individual.*}
     *
     * @param args
     */
    public static void main(String[] args) {
        System.out.println("=========================================================");
        System.out.println("   Welcome to the ParserNG Bench: Where Code bench-presses Math   ");
        System.out.println("=========================================================\n");
        System.out.println("AVAILABLE EXPRESSIONS:\n" + EXPRESSIONS_MAP);

        // Use a single Scanner for the entire life of the main method
        Scanner scanner = new Scanner(System.in);

        // 1. Robust Expression Selection Loop
        while (true) {
            System.out.print("Choose an expression index (0 to " + (EXPRESSIONS.length - 1) + "): ");
            if (!scanner.hasNextInt()) {
                System.err.println("Invalid input! Please enter a valid integer.\n");
                scanner.next(); // Clear the bad token
                continue;
            }
            index = scanner.nextInt();
            if (index >= 0 && index < EXPRESSIONS.length) {
                break; // Valid choice!
            }
            System.err.println("Index out of bounds! Choose between 0 and " + (EXPRESSIONS.length - 1) + ".\n");
        }
        System.setProperty("benchmark.index", String.valueOf(index));
        scanner.nextLine(); // Critical: Consume the leftover newline character!

        System.out.println("\nSelected Expression: " + EXPRESSIONS[index] + "\n");

        // 3. Parse and Configure JMH Options
        try {
            OptionsBuilder opt = new OptionsBuilder();
  
            opt.include(ParserNgBench.class.getSimpleName());

            System.out.println("\n⚔️  MATCH MATCHUP: Janino vs ParserNG");
            System.out.println("🚀 LET THE GAMES BEGIN!\n");

            // 4. Fluent, modern JMH Configuration
            Options configurations = opt.mode(Mode.AverageTime)
                    .timeUnit(TimeUnit.NANOSECONDS)
                    .warmupIterations(5)
                    .warmupTime(TimeValue.milliseconds(200L))
                    .measurementIterations(5)
                    .measurementTime(TimeValue.milliseconds(500))
                    .forks(2)
                    .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
                    .jvmArgs("-Xms8g", "-Xmx8g", "-Dbenchmark.index=" + index)
                    .jvmArgsAppend("--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions"/*, "-XX:+LogCompilation", "-XX:+PrintInlining"*/)
                    .build();

            new Runner(configurations).run();

        } catch (NumberFormatException e) {
            System.err.println("Error: Input contains non-numeric characters inside the parser selections.");
        } catch (RunnerException ex) {
            System.getLogger(ParserNgBench.class.getName()).log(System.Logger.Level.ERROR, "JMH Execution failed", ex);
        }
    }

    public static void main2(String[] args) {//
        System.out.println(EXPR_MAP);
    }

}
