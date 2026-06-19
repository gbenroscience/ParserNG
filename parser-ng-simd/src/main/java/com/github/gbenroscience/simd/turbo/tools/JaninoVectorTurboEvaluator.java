package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.utils.MathToJaninoConverter;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import org.codehaus.janino.ClassBodyEvaluator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * Janino-based Turbo Evaluator - True zero-allocation at any scale.
 *
 * Generates bytecode at compile time, eliminating MethodHandle overhead. JIT
 * can fully inline and vectorize the generated code.
 *
 * Expected: 12.4ns scalar → ~12.8ns bulk (constant across all scales)
 * Allocations: 0 B/op ✓
 */
public class JaninoVectorTurboEvaluator extends ScalarTurboEvaluator1 {

    @FunctionalInterface
    public interface CompiledEvaluator {
        double eval(double x1, double x2, double x3);
        //double eval(double[] vars);
    }

    private final CompiledEvaluator janinoCompiledEval;
    private final double[] frameBuffer;

    public JaninoVectorTurboEvaluator(MathExpression me) throws Exception {
        super(me);
        int maxVars = (me.getSlots() != null) ? me.getSlots().length : 128;
        this.frameBuffer = new double[Math.max(maxVars, 128)];
        this.janinoCompiledEval = compileWithJanino(me);
    }

    /**
     * Convert postfix tokens to a direct Java expression string. Then compile
     * it with Janino to bytecode.
     */
    private CompiledEvaluator compileWithJanino(MathExpression me) throws Exception {
        String javaExpr = MathToJaninoConverter.convert(me.getExpression());
        String[] expressionVars = new MathExpression(me.getExpression()).getVariablesNames();
        // Use an index loop to cleanly target array tracking positions
        for (int i = 0; i < expressionVars.length; i++) {
            String varName = expressionVars[i];
            // \b ensures we match exact variable tokens (e.g., matching "x1" but ignoring "x10")
            String regex = "\\b" + java.util.regex.Pattern.quote(varName) + "\\b";
            javaExpr = javaExpr.replaceAll(regex, "x" + (i+1) + "");
        }

        // Generate a Janino class that evaluates the expression
        String javaCode = generateJaninoClass(javaExpr);

        System.out.println("Input Expression: " + me.getExpression());
        System.out.println("Output JAVA-CODE:\n" + javaCode);

        // Compile to bytecode and load
        ClassBodyEvaluator evaluator = new ClassBodyEvaluator();
        evaluator.setImplementedInterfaces(new Class[]{CompiledEvaluator.class});
        evaluator.cook(javaCode);

        // Create instance
        return (CompiledEvaluator) evaluator.getClazz()
                .getDeclaredConstructor().newInstance();

    }

    private String getJavaOperator(char op) {
        return switch (op) {
            case '+' ->
                "+";
            case '-' ->
                "-";
            case '*' ->
                "*";
            case '/' ->
                "/";
            case '^' ->
                "Math.pow";
            case '%' ->
                "%";
            default ->
                throw new IllegalArgumentException("Unknown operator: " + op);
        };
    }

    private String getJavaFunction(String name, int arity) {
        return switch (name) {
            case "sin" ->
                "Math.sin";
            case "cos" ->
                "Math.cos";
            case "tan" ->
                "Math.tan";
            case "sqrt" ->
                "Math.sqrt";
            case "exp" ->
                "Math.exp";
            case "log", "ln" ->
                "Math.log";
            case "log10" ->
                "Math.log10";
            case "abs" ->
                "Math.abs";
            case "floor" ->
                "Math.floor";
            case "ceil" ->
                "Math.ceil";
            case "asin" ->
                "Math.asin";
            case "acos" ->
                "Math.acos";
            case "atan" ->
                "Math.atan";
            case "sinh" ->
                "Math.sinh";
            case "cosh" ->
                "Math.cosh";
            case "tanh" ->
                "Math.tanh";
            case "cbrt" ->
                "Math.cbrt";
            case "min" ->
                "Math.min";
            case "max" ->
                "Math.max";
            default ->
                throw new IllegalArgumentException("Unknown function: " + name);
        };
    }

    private String generateJaninoClass(String expression) {
/*
        return String.format(
                """
             @Override
                public double eval(double[] vars) {
                    return %s;
                } 
            """, expression);
*/
                return String.format(
                """
             @Override
                public double eval(double x1, double x2, double x3) {
                    return %s;
                } 
            """, expression);

    }

    private int resolveSlotIndex(int tokenIndex, int[] slots) {
        if (slots != null) {
            for (int k = 0; k < slots.length; k++) {
                if (slots[k] == tokenIndex) {
                    return k;
                }
            }
        }
        return tokenIndex;
    }

    public FastCompositeExpression compile() throws Throwable {
        return new JaninoBulkExpression(janinoCompiledEval, frameBuffer);
    }

    public class JaninoBulkExpression implements FastCompositeExpression {

        private final JaninoVectorTurboEvaluator.CompiledEvaluator evaluator;
        private final double[] frameBuffer;

        JaninoBulkExpression(JaninoVectorTurboEvaluator.CompiledEvaluator eval, double[] buffer) {
            this.evaluator = eval;
            this.frameBuffer = buffer;
        }

        @Override
        public double applyScalar(double[] variables) {
            return evaluator.eval(variables[0],variables[1],variables[2]);
        }

        @Override
        public MathExpression.EvalResult apply(double[] variables) {
            return null;
        }

        @Override
        public String checkErrorLogs() {
            return "";
        }

        @Override
        public TurboExpressionEvaluator getCompiler() {
            return JaninoVectorTurboEvaluator.this;
        }

        /**
         * **TRUE ZERO-ALLOC with Janino**
         *
         * Janino generates direct bytecode, so the JIT sees: - Pure arithmetic
         * operations - Math library calls (already JIT-friendly) - NO
         * MethodHandle overhead - Full vectorization support
         */
        public void applyBulk(double[][] variables, double[] output) {
            applyBulkInternal(variables, output, 0, output.length);
        }

        public void applyBulk(double[][] variables, double[] output, int offset) {
            applyBulkInternal(variables, output, offset, output.length - offset);
        }

        private void applyBulkInternal(double[][] variables, double[] output, int startIdx, int length) {
            if (variables == null || variables.length == 0 || length <= 0) {
                return;
            }

            final int nVars = variables.length;
            final int endIdx = startIdx + length;
            final double[] frame = frameBuffer;

           /* // **Pure scalar loop - JIT will vectorize this**
            for (int i = startIdx; i < endIdx; i++) {
                for (int v = 0; v < nVars; v++) {
                    frame[v] = variables[v][i];
                }
                output[i] = evaluator.eval(frame);
            }
*/
            // In applyBulkInternal, replace the loop with:
            for (int i = startIdx; i < endIdx; i++) {
                output[i] = evaluator.eval(
                        variables[0][i],
                        variables[1][i],
                        variables[2][i]
                );
            }
        }

        public void applyBulk(double[][] variables, double[] output, ExecutorService executor) {
            if (variables == null || variables.length == 0 || executor == null) {
                applyBulkInternal(variables, output, 0, output.length);
                return;
            }

            final int length = output.length;
            if (length < 100_000) {
                applyBulkInternal(variables, output, 0, length);
                return;
            }

            final int nThreads = Math.min(
                    Runtime.getRuntime().availableProcessors(),
                    Math.max(1, length / 100_000)
            );
            final int chunkSize = (length + nThreads - 1) / nThreads;
            @SuppressWarnings("unchecked")
            final Future<Void>[] futures = new Future[nThreads];

            for (int t = 0; t < nThreads; t++) {
                final int start = t * chunkSize;
                final int end = Math.min(start + chunkSize, length);
                futures[t] = executor.submit(() -> {
                    applyBulkInternal(variables, output, start, end - start);
                    return null;
                });
            }

            for (Future<Void> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            int offset = 0;
            final int length = output.length;
            while (offset < length) {
                final int nextOffset = Math.min(offset + batchSize, length);
                applyBulkInternal(variables, output, offset, nextOffset - offset);
                offset = nextOffset;
            }
        }

        public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) {
            throw new UnsupportedOperationException("Not supported");
        }

        public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
            throw new UnsupportedOperationException("Not supported");
        }
    }
}
