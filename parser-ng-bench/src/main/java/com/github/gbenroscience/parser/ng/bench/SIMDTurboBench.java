package com.github.gbenroscience.parser.ng.bench;



/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.SIMDVectorTurboEvaluator; 
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;
import java.util.Random;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/**
 * Use this to build in the parser-ng-pro directory mvn clean install -Dgpg.skip
Use this to run the benchmarks

java -jar target/benchmarks.jar SIMDTurboBench -prof perfasm

java -jar target/benchmarks.jar SIMDTurboBench -prof perfasm >
perf_output.txt 
Now find your specific method's assembly grep -A 50 "parserNG" perf_output.txt
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
public class SIMDTurboBench {


    @Param({"12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2", "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))"})
    String expression;

    //static final String expression = "x1^3+x2^3+x3^3+x4^3+x5^3+x6^3";
    double[] result;

    @Param({"1000000", "67108864"})
    private int dataSize;

    private double[] flatInput;
    private double[] vars;
    private VectorTurboBench.JaninoMathFunction fastEvaluator;

    SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression simdVectorTurbo;

    private String[] expressionVars;
    private int varCount;


    @Setup(Level.Trial)
    public void setup() {
        System.out.println("================EXPRESSION: " + expression+"================");
        MathExpression me = new MathExpression(expression);
        expressionVars = me.getVariablesNames();
        varCount = expressionVars.length;
        this.vars = new double[varCount];

        flatInput = new double[varCount * dataSize];
        result = new double[dataSize];

        // Fill once per trial, not per iteration
        Random r = new Random(42);
        for (int i = 0; i < flatInput.length; i++) {
            flatInput[i] = r.nextDouble() * 10 - 5;
        }

        this.fastEvaluator = VectorTurboBench.setupJanino(expression, expressionVars);

        setupParserNG(me);

    }
    
    
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void janino(Blackhole bh) {
        // Hoisting instance variables to local registers to eliminate field access overhead inside loop
        final int limit = this.dataSize;
        final int stride = this.varCount;
        final double[] input = this.flatInput;
        final double[] localVars = this.vars;
        final VectorTurboBench.JaninoMathFunction evaluator = this.fastEvaluator;

        for (int i = 0; i < limit; i++) {
            for(int j=0;j<stride;j++){
                localVars[j] = input[(j * limit) + i];
            }
            bh.consume(evaluator.apply(localVars));
        }  
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void parserNG(Blackhole bh) {
        // Correct usage: Call the bulk processor ONCE per benchmark invocation
        // This measures the true speed of your vectorization logic 
        simdVectorTurbo.applyBulk(flatInput, result);
        bh.consume(result);
    }

    private void setupParserNG(MathExpression me) {
        try {
            simdVectorTurbo = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        } catch (Throwable ex) {
            System.getLogger(SIMDTurboBench.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
    }

   
    
        public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(SIMDTurboBench.class.getSimpleName()); // Always include baseline
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
