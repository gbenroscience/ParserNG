package com.github.gbenroscience.parser.pro.turbo;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.SIMDVectorTurboEvaluator;
import com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.parser.pro.turbo.tools.utils.MathToJaninoConverter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;



/**
 * Use this to build in the parser-ng-pro directory
 * mvn clean install -Dgpg.skip
 * Use this to run the benchmarks
 * 
 * java -jar target/benchmarks.jar MathEvalBenchmark -prof perfasm
 * 
 * java -jar target/benchmarks.jar MathEvalBenchmark -prof perfasm > perf_output.txt
#  Now find your specific method's assembly
 * grep -A 50 "parserNG" perf_output.txt
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
    "-XX:-UseCompressedOops" // Avoids compressed oops artifacts
})
@State(Scope.Benchmark)
public class ScalarBenchmarks {

    public static interface JaninoMathFunction {

        double apply(double[] v);
    }

   //static final String expression = "0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))";
   static final String expression = "12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2";
    
    //static final String expression = "x1^3+x2^3+x3^3+x4^3+x5^3+x6^3";

    double[] result;

    @Param({"1000000", "67108864"})
    private int dataSize;

    private double[] flatInput; 
    private JaninoMathFunction fastEvaluator;

    SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression vectorTurbo;

    private String[] expressionVars;
    private int varCount;
 
    protected double[] xValues;
    
    
    protected int simpleCursor;
    protected int[] randomData;
     

    @Setup(Level.Trial)
    public void setup() {

        initRandomData();
        MathExpression me = new MathExpression(expression);
        expressionVars = me.getVariablesNames();
        varCount = expressionVars.length;
         xValues = new double[varCount];

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
    
   protected void initRandomData() {
        this.randomData = splitLongIntoDigits(System.currentTimeMillis());
    }
     
   protected void generateInputs() {
        double base = randomData[simpleCursor++ % randomData.length];
        //double base = randomData[cursor.getAndIncrement() % randomData.length];//this is much slower!
        if (xValues.length != 0) {
            xValues[0] = base;
        }
        for (int i = 1; i < varCount; i++) {
            xValues[i] = base + (i % 2 == 0 ? 1.0 : -1.0) * (0.1 + (i % 10) * 0.1); // your original pattern
        }
    }


 
    
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void parserNG(Blackhole bh) {
        generateInputs();
        // Correct usage: Call the bulk processor ONCE per benchmark invocation
        // This measures the true speed of your vectorization logic  
         bh.consume(vectorTurbo.applyScalar(xValues)); 
    }
    
    
        @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void janino(Blackhole bh) {
        generateInputs();
        // We use the flat array flatInput and pass segments or the whole array 
        // depending on your Janino interface. Assuming Janino takes a sliding window or full array:
        bh.consume(fastEvaluator.apply(xValues));
      
 
    }

    
    
    
       private void setupParserNG(MathExpression me) {
        try {
            vectorTurbo = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();
        } catch (Throwable ex) {
            System.getLogger(ScalarBenchmarks.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
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

 
    public final static int[] splitLongIntoDigits(long n) {
        if (n == 0) {
            return new int[]{0}; // Special case for zero
        }

        boolean isNegative = n < 0;
        long temp = Math.abs(n); // Work with the absolute value
        List<Integer> digitList = new ArrayList<>();

        while (temp > 0) {
            // Get the last digit using modulo 10
            digitList.add((int) (temp % 10));
            // Remove the last digit using integer division
            temp /= 10;
        }

        // The digits are in reverse order, so reverse the list
        Collections.reverse(digitList);

        // Convert the ArrayList<Integer> to a primitive int[] array
        int[] digits = new int[digitList.size()];
        for (int i = 0; i < digitList.size(); i++) {
            digits[i] = digitList.get(i);
        }

        // Handle the sign if necessary (e.g. if the original number was negative, you might need 
        // to handle the representation of the sign explicitly, depending on requirements)
        // For simply getting the sequence of digits, the absolute value is sufficient.
        return digits;
    }
  public static void main(String[] args) throws RunnerException {

        /*
          Options opt = new OptionsBuilder()
            .include(MultiVariatePowerSeriesBenchmarks.class.getSimpleName())
            // First Mode: Latency (How fast is one?)
            .mode(Mode.AverageTime)
            .timeUnit(TimeUnit.NANOSECONDS)
            // Second Mode: Throughput (How many can we do?)
            .mode(Mode.Throughput)
            .timeUnit(TimeUnit.SECONDS) 
            .warmupIterations(5)
            .warmupTime(TimeValue.seconds(1))
            .measurementIterations(5)
            .measurementTime(TimeValue.seconds(2))
            .forks(1) 
            .addProfiler(org.openjdk.jmh.profile.GCProfiler.class)
            // Ensuring the JVM is in "High Performance" mode
            .jvmArgs("-Xms2g", "-Xmx2g", "-XX:+UseParallelGC", "-XX:+TieredCompilation")
            .build();
         */
        Options opt = new OptionsBuilder()
                .include(ScalarBenchmarks.class.getSimpleName())
                .mode(Mode.AverageTime)
                .timeUnit(TimeUnit.NANOSECONDS)
                .warmupIterations(5)
                .warmupTime(TimeValue.seconds(1))
                .measurementIterations(5)
                .measurementTime(TimeValue.seconds(2))
                .forks(1)
                .addProfiler(org.openjdk.jmh.profile.GCProfiler.class) 
                .jvmArgs("-Xms8g", "-Xmx8g", "-Dbenchmark.index=1", "-XX:+TieredCompilation")
                .jvmArgsAppend("--add-modules", "jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions")
                .build();

        new Runner(opt).run();
    }
}
