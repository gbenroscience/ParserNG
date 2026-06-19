package com.github.gbenroscience.parser.pro.turbo.tools.utils;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.SIMDVectorTurboEvaluator; 
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;
import java.util.Random;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Use this to build in the parser-ng-pro directory mvn clean install -Dgpg.skip
 * Use this to run the benchmarks
 *
 * java -jar target/benchmarks.jar MathEvalBenchmarkForSIMD -prof perfasm
 *
 * java -jar target/benchmarks.jar MathEvalBenchmarkForSIMD -prof perfasm >
 * perf_output.txt # Now find your specific method's assembly grep -A 50
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
@State(Scope.Benchmark)
public class MathEvalBenchmarkForSIMD {

    public static interface JaninoMathFunction {

        double apply(double[] v);
    }

    @Param({"12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2", "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))"})
    String expression;

    //static final String expression = "x1^3+x2^3+x3^3+x4^3+x5^3+x6^3";
    double[] result;

    @Param({"1000000", "67108864"})
    private int dataSize;

    private double[] flatInput;
    private JaninoMathFunction fastEvaluator;

    SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression simdVectorTurbo;

    private String[] expressionVars;
    private int varCount;

    private double vars[];

    @Setup(Level.Trial)
    public void setup() {
        System.out.println("================EXPRESSION: " + expression+"================");
        MathExpression me = new MathExpression(expression);
        expressionVars = me.getVariablesNames();
        varCount = expressionVars.length;
        vars = new double[varCount];

        flatInput = new double[varCount * dataSize];
        result = new double[dataSize];

        // Fill once per trial, not per iteration
        Random r = new Random(42);
        for (int i = 0; i < flatInput.length; i++) {
            flatInput[i] = r.nextDouble() * 10 - 5;
        }

        setupJanino();

        setupParserNG(me);

    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void janino(Blackhole bh) {
        // We use the flat array flatInput and pass segments or the whole array 
        // depending on your Janino interface. Assuming Janino takes a sliding window or full array:
        for (int i = 0; i < dataSize; i++) {
            // If Janino needs a specific variable buffer (vars):
            int base = i * varCount;
            System.arraycopy(flatInput, base, vars, 0, varCount);
            bh.consume(fastEvaluator.apply(vars));
        }

    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void parserNG(Blackhole bh) {
        // Correct usage: Call the bulk processor ONCE per benchmark invocation
        // This measures the true speed of your vectorization logic 
        simdVectorTurbo.applyBulkParallel(flatInput, result);
        bh.consume(result);
    }

    private void setupParserNG(MathExpression me) {
        try {
            simdVectorTurbo = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        } catch (Throwable ex) {
            System.getLogger(MathEvalBenchmarkForSIMD.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
    }

    private void setupJanino() {
        // Convert ParserNG syntax to Java syntax
        String javaExpr = MathToJaninoConverter.convert(expression);

        // Use an index loop to cleanly target array tracking positions
        for (int i = 0; i < expressionVars.length; i++) {
            String varName = expressionVars[i];
            // \b ensures we match exact variable tokens (e.g., matching "x1" but ignoring "x10")
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
            this.fastEvaluator = (JaninoMathFunction) cbe.getClazz()
                    .getDeclaredConstructor().newInstance();
        } catch (Exception ex) {
            // Rethrowing as RuntimeException prevents JMH from continuing silently with null states
            throw new RuntimeException("Failed to compile Janino function body expressions", ex);
        }
    }
}
