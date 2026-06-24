package com.github.gbenroscience.parser.ng.bench;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.ng.bench.utils.MathToJaninoConverter;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
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
 * Use this to build in the parser-ng-pro directory:
mvn clean install -Dgpg.skip
* Use this to run the benchmarks:
java -jar target/benchmarks.jar VectorTurboBench -prof perfasm
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
public class VectorTurboBench {

    public static interface JaninoMathFunction {
        double apply(double[] v);
    }

    @Param({
        "12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2", 
        "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))"
    })
    private String expression;

    @Param({"1000000", "67108864"})
    private int dataSize;

    private double[] flatInput; 
    private double[] result;
    private JaninoMathFunction fastEvaluator;
    private VectorTurboEvaluator.BatchedVectorCompositeExpression vectorTurbo;

    private int varCount;
    private double[] vars;

    @Setup(Level.Trial)
    public void setup() {
        MathExpression me = new MathExpression(expression);
        String[] expressionVars = me.getVariablesNames();
        this.varCount = expressionVars.length;
        this.vars = new double[varCount];

        this.flatInput = new double[varCount * dataSize];
        this.result = new double[dataSize];

        // Safe initialization boundary (1.0 to 10.0) to prevent division by zero in expressions
        Random r = new Random(42);
        for (int i = 0; i < flatInput.length; i++) {
            flatInput[i] = 1.0 + (r.nextDouble() * 9.0);
        }
        
        this.fastEvaluator = setupJanino(expression, expressionVars);
        setupParserNG(me);
    }

    @Benchmark
    public void janino(Blackhole bh) {
        // Hoisting instance variables to local registers to eliminate field access overhead inside loop
        final int limit = this.dataSize;
        final int stride = this.varCount;
        final double[] input = this.flatInput;
        final double[] localVars = this.vars;
        final JaninoMathFunction evaluator = this.fastEvaluator;

        for (int i = 0; i < limit; i++) {
            for(int j=0;j<stride;j++){
                localVars[j] = input[(j * limit) + i];
            }
            bh.consume(evaluator.apply(localVars));
        }  
    }

    @Benchmark
    public void parserNG(Blackhole bh) {
        // True measurement of high-performance vectorized vector logic execution
        vectorTurbo.applyBulkParallel(flatInput, result);
        bh.consume(result); 
    }

    private void setupParserNG(MathExpression me) {
        try {
            vectorTurbo = (VectorTurboEvaluator.BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();
        } catch (Throwable ex) {
            throw new RuntimeException("Failed to compile VectorTurboEvaluator expressions", ex);
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
            org.codehaus.janino.ClassBodyEvaluator cbe = new org.codehaus.janino.ClassBodyEvaluator();
            cbe.setImplementedInterfaces(new Class[]{JaninoMathFunction.class});
            cbe.cook(classBody);
            return (JaninoMathFunction) cbe.getClazz().getDeclaredConstructor().newInstance();
        } catch (Exception ex) {
            throw new RuntimeException("Failed to compile Janino function body expressions", ex);
        }
    }
    
    public static void main(String[] args) throws RunnerException {
        OptionsBuilder opt = new OptionsBuilder();
        opt.include(VectorTurboBench.class.getSimpleName());
        
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