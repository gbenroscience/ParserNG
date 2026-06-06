package com.github.gbenroscience.parser.pro.turbo.tools;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

/**
 * Enterprise SIMD Evaluator for ParserNG. Compiles a parallel Vector
 * MethodHandle tree for high-throughput block execution. * REQUIRES JVM FLAG:
 * --add-modules jdk.incubator.vector
 *
 * When running from command line: java --add-modules jdk.incubator.vector -cp
 * target/parserng.jar com.github.gbenroscience.parser.wars.ParserNGWars
 */
public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    // Let the JVM pick the optimal hardware vector size (e.g., AVX-256 or AVX-512)
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();

    private final MathExpression.Token[] postfix;

    public VectorTurboEvaluator(MathExpression me) {
        super(me);
        this.postfix = me.getCachedPostfix();
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        // 1. Inherit the highly optimized scalar handle for the tail-loop fallback
        final MethodHandle rawScalarHandle = super.compileScalar(this.postfix);

        MethodHandle vectorHandle = null;
        KernelInterceptException interceptedKernel = null;

        try {
            // Attempt to compile the new Hardware SIMD Vector Handle
            vectorHandle = compileVector(this.postfix);
        } catch (KernelInterceptException ex) {
            // We caught a matrix kernel signal!
            interceptedKernel = ex;
        }

        final MethodHandle finalVectorHandle = vectorHandle;
        final KernelInterceptException finalKernelSignal = interceptedKernel;

        return new SIMDCompositeExpression() {
            @Override
            public double applyScalar(double[] variables) {
                try {
                    return (double) rawScalarHandle.invokeExact(variables);
                } catch (Throwable t) {
                    throw new RuntimeException("Scalar execution failed", t);
                }
            }

            @Override
            public MathExpression.EvalResult apply(double[] variables) {
                return new MathExpression.EvalResult().wrap(applyScalar(variables));
            }

            @Override
            public String checkErrorLogs() {
                return "";
            }

            @Override
            public TurboExpressionEvaluator getCompiler() {
                return VectorTurboEvaluator.this;
            }

            /**
             * THE TURBO SIMD LOOP Executes 4, 8, or 16 operations
             * simultaneously per CPU cycle.
             */
            @Override
            public void applyBulk(double[][] variables, double[] output) {
                final int length = output.length;
                final int nVars = variables.length;

                // Guard #1: Tiny arrays or missing vector support. Go scalar.
                if (length < SPECIES.length() || finalVectorHandle == null) {
                    for (int i = 0; i < length; i++) {
                        // Instantiating inside the loop allows JVM Escape Analysis to allocate this on the stack
                        double[] scalarFrame = new double[nVars];
                        for (int v = 0; v < nVars; v++) {
                            scalarFrame[v] = variables[v][i];
                        }
                        try {
                            output[i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                        } catch (Throwable t) {
                            throw new RuntimeException("Scalar execution failed at index " + i, t);
                        }
                    }
                    return;
                }

                // Guard #2: For large N, go parallel. Threshold = 100k.
                if (length >= 100_000) {
                    applyBulkParallel(variables, output);
                    return;
                }

                // --- Hot path: Single-thread SIMD ---
                // Variables is ALREADY SoA. No transpose needed!
                final int loopBound = SPECIES.loopBound(length);
                final int laneWidth = SPECIES.length();
                int i = 0;

                try {
                    // 1. Hardware Vector Hot-Loop
                    for (; i < loopBound; i += laneWidth) {
                        DoubleVector resultLane = (DoubleVector) finalVectorHandle.invokeExact(variables, i);
                        resultLane.intoArray(output, i);
                    }

                    // 2. Scalar Tail Clean-up
                    if (i < length) {
                        for (; i < length; i++) {
                            double[] scalarFrame = new double[nVars]; // Inside loop for Escape Analysis
                            for (int v = 0; v < nVars; v++) {
                                scalarFrame[v] = variables[v][i];
                            }
                            output[i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                        }
                    }
                } catch (Throwable t) {
                    throw new RuntimeException("Hardware Vector SIMD Execution failed at loop index " + i, t);
                }
            }

            public void applyBulk(double[][] variables, double[] output, ExecutorService executor) {
                final int length = output.length;
                final int nVars = variables.length;

                // If it's too small, don't bother the ExecutorService. 
                // Just run it on the current thread using your existing optimized logic.
                if (length < 25_000) {
                    this.applyBulk(variables, output);
                    return;
                }
                // Only fallback if SIMD impossible. Respect user's executor otherwise.
                if (length < SPECIES.length() || finalVectorHandle == null) {
                    for (int i = 0; i < length; i++) {
                        double[] scalarFrame = new double[nVars];
                        for (int v = 0; v < nVars; v++) {
                            scalarFrame[v] = variables[v][i];
                        }
                        try {
                            output[i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                        } catch (Throwable t) {
                            throw new RuntimeException("Scalar execution failed at index " + i, t);
                        }
                    }
                    return;
                }

                // Use their executor for any size. Let them decide thread count.
                final int nThreads = Math.min(
                        Runtime.getRuntime().availableProcessors(),
                        Math.max(1, length / 25_000) // min 25k per thread to amortize overhead
                );
                final int chunk = (length + nThreads - 1) / nThreads;

                List<Future<?>> futures = new ArrayList<>(nThreads);

                for (int t = 0; t < nThreads; t++) {
                    final int threadId = t;
                    futures.add(executor.submit(() -> {
                        final int start = threadId * chunk;
                        final int end = Math.min(start + chunk, length);

                        final int localLength = end - start;
                        final int loopBound = start + SPECIES.loopBound(localLength);
                        final int laneWidth = SPECIES.length();
                        int i = start;

                        try {
                            for (; i < loopBound; i += laneWidth) {
                                DoubleVector resultLane = (DoubleVector) finalVectorHandle.invokeExact(variables, i);
                                resultLane.intoArray(output, i);
                            }
                            for (; i < end; i++) {
                                double[] scalarFrame = new double[nVars];
                                for (int v = 0; v < nVars; v++) {
                                    scalarFrame[v] = variables[v][i];
                                }
                                output[i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                            }
                        } catch (Throwable e) {
                            throw new RuntimeException("Executor task failed at index " + i + " thread " + threadId, e);
                        }
                    }));
                }

                // Block and propagate real exceptions
                for (Future<?> f : futures) {
                    try {
                        f.get();
                    } catch (ExecutionException e) {
                        Throwable cause = e.getCause() != null ? e.getCause() : e;
                        throw new RuntimeException("Parallel execution failed", cause);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted during parallel execution", e);
                    }
                }
            }

            private void applyBulkParallel(double[][] variables, double[] output) {
                final int length = output.length;
                final int nVars = variables.length;
                final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                        (length + 99_999) / 100_000);

                IntStream.range(0, nThreads).parallel().forEach(threadId -> {
                    final int chunk = (length + nThreads - 1) / nThreads;
                    final int start = threadId * chunk;
                    final int end = Math.min(start + chunk, length);

                    // CRITICAL FIX: Loop bound relative to chunk
                    final int localLength = end - start;
                    final int loopBound = start + SPECIES.loopBound(localLength);
                    final int laneWidth = SPECIES.length();
                    int i = start;

                    try {
                        // 1. Vector loop for this chunk
                        for (; i < loopBound; i += laneWidth) {
                            DoubleVector resultLane = (DoubleVector) finalVectorHandle.invokeExact(variables, i);
                            resultLane.intoArray(output, i);
                        }

                        // 2. Scalar tail for this chunk
                        if (i < end) {
                            for (; i < end; i++) {
                                double[] scalarFrame = new double[nVars]; // Inside loop for Escape Analysis
                                for (int v = 0; v < nVars; v++) {
                                    scalarFrame[v] = variables[v][i];
                                }
                                output[i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                            }
                        }
                    } catch (Throwable t) {
                        throw new RuntimeException("Parallel SIMD failed at index " + i + " on thread " + threadId, t);
                    }
                });
            }

            // =====================================================================
            // THE NEW MATRIX KERNEL DISPATCHER
            // =====================================================================
            @Override
            public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
                // If the user specified an operation manually, use it. 
                // Otherwise, use the one we intercepted from the AST.
                String kernelToRun = (op != null) ? op
                        : (finalKernelSignal != null) ? finalKernelSignal.getKernelName() : null;

                if (kernelToRun == null) {
                    throw new UnsupportedOperationException("No matrix kernel specified or intercepted.");
                }

                // Dispatch to FlatMatrix (The switch statement matching isMatrixKernel and FlatMatrix API)
                switch (kernelToRun.toLowerCase()) {

                    // ==========================================
                    // 1. STANDARD BINARY MATRIX ALGEBRA
                    // ==========================================
                    case "matmul":
                        if (inputs.length != 2) {
                            throw new IllegalArgumentException("matmul requires 2 inputs");
                        }
                        FlatMatrix.matmul(inputs[0], inputs[1], output);
                        break;

                    case "add":
                    case "matadd":
                    case "add_mat":
                        if (inputs.length != 2) {
                            throw new IllegalArgumentException("matadd requires 2 inputs");
                        }
                        FlatMatrix.add(inputs[0], inputs[1], output);
                        break;

                    // ==========================================
                    // 2. FUSED DEEP LEARNING KERNELS
                    // ==========================================
                    case "matmul_bias_gelu":
                    case "matmul_add_bias_gelu":
                        if (inputs.length != 3) {
                            throw new IllegalArgumentException("matmul_bias_gelu requires A, B, and Bias");
                        }
                        FlatMatrix.matmulAddBiasGelu(inputs[0], inputs[1], inputs[2], output);
                        break;

                    case "matmul_add_sin":
                        if (inputs.length != 3) {
                            throw new IllegalArgumentException("matmul_add_sin requires A, B, and Alpha (as 1x1 scalar matrix)");
                        }
                        // Alpha is passed as a constant through the AST, extract it from the 1x1 matrix wrapper
                        double alpha = inputs[2].data[0];
                        FlatMatrix.matmulAddSin(inputs[0], inputs[1], output, alpha);
                        break;

                    // ==========================================
                    // 3. IN-PLACE ELEMENT-WISE ACTIVATIONS
                    // ==========================================
                    case "softmax":
                    case "softmax_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("softmax requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.softmaxRowsInPlace();
                        break;

                    case "relu":
                    case "relu_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("relu requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.reluInPlace();
                        break;

                    case "gelu":
                    case "gelu_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("gelu requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.geluInPlace();
                        break;

                    case "exp":
                    case "exp_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("exp requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.expInPlace();
                        break;

                    case "log":
                    case "log_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("log requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.logInPlace();
                        break;

                    case "tanh":
                    case "tanh_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("tanh requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.tanhInPlace();
                        break;

                    case "sin":
                    case "sin_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("sin requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, 0, output.data, 0, output.data.length);
                        }
                        output.sinInPlace();
                        break;

                    default:
                        throw new UnsupportedOperationException("Unknown Matrix Kernel: " + kernelToRun);
                }
            }
        };
    }

    /**
     * Compiles the AST into a Hardware SIMD pipeline. Stack handles always have
     * the signature: (double[][] vars, int offset) -> DoubleVector
     */
    private MethodHandle compileVector(MathExpression.Token[] postfix) throws Throwable {
        Stack<MethodHandle> stack = new Stack<>();

        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    // Broadcast constant to all lanes: e.g., 2.0 -> [2.0, 2.0, 2.0, 2.0]
                    MethodHandle bcast = LOOKUP.findStatic(VectorTurboEvaluator.class, "broadcastConstant",
                            MethodType.methodType(DoubleVector.class, double.class));
                    bcast = MethodHandles.insertArguments(bcast, 0, t.value);
                    // Match signature: (double[][], int) -> DoubleVector
                    stack.push(MethodHandles.dropArguments(bcast, 0, double[][].class, int.class));
                    break;

                case MathExpression.Token.VARIABLE:
                    int finalIndex = resolveSlotIndex(t.frameIndex);
                    // Load slice from array into vector register
                    MethodHandle loadHandle = LOOKUP.findStatic(VectorTurboEvaluator.class, "loadVector",
                            MethodType.methodType(DoubleVector.class, double[][].class, int.class, int.class));
                    // Bind the slot index. Signature becomes: (double[][], int offset) -> DoubleVector
                    stack.push(MethodHandles.insertArguments(loadHandle, 1, finalIndex));
                    break;

                case MathExpression.Token.OPERATOR:
                    MethodHandle right = stack.pop();
                    MethodHandle left = stack.pop();
                    VectorOperators.Binary binOp = mapBinaryOp(t.opChar);
                    stack.push(applyVectorBinaryOp(binOp, left, right));
                    break;

                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD:

                    // 1. Check if it's a known Matrix Kernel Operation
                    if (isMatrixKernel(t.name, t.arity)) {
                        // We do NOT compile a MethodHandle tree for this.
                        // Instead, we throw a specific internal signal or return a specialized Matrix Handle
                        // that tells the runtime "Hey, execute this via applyMatrixKernel() instead of applyBulk()".
                        throw new KernelInterceptException(t.name, t.arity);
                    }

                    // For intrinsic functions like sin, cos, sqrt
                    if (t.arity == 1) {
                        MethodHandle operand = stack.pop();
                        VectorOperators.Unary unOp = mapUnaryOp(t.name);

                        if (unOp != null) {
                            stack.push(applyVectorUnaryOp(unOp, operand));
                        } else {
                            // If the Vector API doesn't support the function (e.g., sec, csc),
                            // we build a fallback handle that maps your scalar method lane-by-lane.
                            stack.push(createScalarFallbackUnary(t.name, operand));
                        }
                    } else if (t.arity == 2) {
                        MethodHandle r = stack.pop();
                        MethodHandle l = stack.pop();
                        VectorOperators.Binary bOp = mapBinaryOpName(t.name);
                        stack.push(applyVectorBinaryOp(bOp, l, r));
                    }
                    break;
            }
        }
        return stack.pop();
    }

    /**
     * Determines if a parsed function name maps to a high-speed FlatMatrix
     * kernel.
     *
     * * @param funcName The lowercase name of the function parsed from the
     * AST.
     * @param arity The number of arguments the user passed into the function.
     * @return true if it's a kernel operation, false if it should be standard
     * AST math.
     */
    private boolean isMatrixKernel(String funcName, int arity) {
        switch (funcName) {
            // --- Standard Matrix Algebra ---
            case "matmul":
                // Expects: matmul(A, B)
                return arity == 2;

            case "matadd": // Using matadd to distinguish from scalar add
            case "add_mat":
                // Expects: matadd(A, B)
                return arity == 2;

            // --- In-Place Activations (Element-wise) ---
            case "softmax":
            case "softmax_in_place":
            case "relu_in_place":
            case "gelu_in_place":
            case "exp_in_place":
            case "log_in_place":
            case "tanh_in_place":
            case "sin_in_place":
                // Expects: softmax(A)
                return arity == 1;

            // --- Fused Deep Learning Kernels (The Performance Multipliers) ---
            case "matmul_add_sin":
                // Expects: matmul_add_sin(A, B, alpha)
                return arity == 3;

            case "matmul_bias_gelu":
            case "matmul_add_bias_gelu":
                // Expects: matmul_bias_gelu(A, B, Bias)
                return arity == 3;

            default:
                return false;
        }
    }

    // =========================================================================
    // SIMD COMBINER MAGIC (Matching your zero-allocation fold logic)
    // =========================================================================
    private static MethodHandle applyVectorBinaryOp(VectorOperators.Binary op, MethodHandle left, MethodHandle right) throws Throwable {
        // Base logic: (DoubleVector a, DoubleVector b) -> a.lanewise(op, b)
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execBinaryVector",
                MethodType.methodType(DoubleVector.class, VectorOperators.Binary.class, DoubleVector.class, DoubleVector.class));
        MethodHandle boundOp = MethodHandles.insertArguments(baseOp, 0, op);

        // Permute to (rightValue, leftValue)
        MethodHandle fnSwapped = MethodHandles.permuteArguments(
                boundOp, MethodType.methodType(DoubleVector.class, DoubleVector.class, DoubleVector.class), 1, 0);

        // Filter the second argument to calculate `left(vars, offset)`
        MethodHandle combiner = MethodHandles.filterArguments(fnSwapped, 1, left);

        // Fold `right(vars, offset)` into the first argument
        return MethodHandles.foldArguments(combiner, right);
    }

    private static MethodHandle applyVectorUnaryOp(VectorOperators.Unary op, MethodHandle operand) throws Throwable {
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execUnaryVector",
                MethodType.methodType(DoubleVector.class, VectorOperators.Unary.class, DoubleVector.class));
        MethodHandle boundOp = MethodHandles.insertArguments(baseOp, 0, op);

        return MethodHandles.filterArguments(boundOp, 0, operand);
    }

    // =========================================================================
    // INTRINSIC VECTOR EXECUTION DELEGATES
    // =========================================================================
    public static DoubleVector loadVector(double[][] variables, int slot, int offset) {
        return DoubleVector.fromArray(SPECIES, variables[slot], offset);
    }

    public static DoubleVector broadcastConstant(double value) {
        return DoubleVector.broadcast(SPECIES, value);
    }

    public static DoubleVector execBinaryVector(VectorOperators.Binary op, DoubleVector a, DoubleVector b) {
        return a.lanewise(op, b);
    }

    public static DoubleVector execUnaryVector(VectorOperators.Unary op, DoubleVector a) {
        return a.lanewise(op);
    }

    // =========================================================================
    // HARDWARE OPERATOR MAPPINGS
    // =========================================================================
    private static VectorOperators.Binary mapBinaryOp(char op) {
        switch (op) {
            case '+':
                return VectorOperators.ADD;
            case '-':
                return VectorOperators.SUB;
            case '*':
                return VectorOperators.MUL;
            case '/':
                return VectorOperators.DIV;
            case '^':
                return VectorOperators.POW;
            default:
                throw new IllegalArgumentException("Unsupported SIMD binary operator: " + op);
        }
    }

    private static VectorOperators.Binary mapBinaryOpName(String name) {
        switch (name.toLowerCase()) {
            case "pow":
                return VectorOperators.POW;
            case "atan2":
                return VectorOperators.ATAN2;
            case "min":
                return VectorOperators.MIN;
            case "max":
                return VectorOperators.MAX;
            default:
                throw new IllegalArgumentException("Unsupported SIMD binary function: " + name);
        }
    }

    private static VectorOperators.Unary mapUnaryOp(String name) {
        switch (name.toLowerCase()) {
            case "sin":
                return VectorOperators.SIN;
            case "cos":
                return VectorOperators.COS;
            case "tan":
                return VectorOperators.TAN;
            case "asin":
                return VectorOperators.ASIN;
            case "acos":
                return VectorOperators.ACOS;
            case "atan":
                return VectorOperators.ATAN;
            case "sinh":
                return VectorOperators.SINH;
            case "cosh":
                return VectorOperators.COSH;
            case "tanh":
                return VectorOperators.TANH;
            case "sqrt":
            case "√":
                return VectorOperators.SQRT;
            case "cbrt":
                return VectorOperators.CBRT;
            case "abs":
                return VectorOperators.ABS;
            case "log":
            case "ln":
                return VectorOperators.LOG;
            case "lg":
                return VectorOperators.LOG10;
            case "exp":
                return VectorOperators.EXP;
            default:
                return null; // Fallback required
        }
    }

    private int resolveSlotIndex(int tokenIndex) {
        if (slots != null) {
            for (int k = 0; k < slots.length; k++) {
                if (slots[k] == tokenIndex) {
                    return k;
                }
            }
        }
        return tokenIndex;
    }

    // =========================================================================
    // SCALAR FALLBACK FOR NON-VECTORIZED FUNCTIONS (e.g., sec, csc, fact)
    // =========================================================================
    private MethodHandle createScalarFallbackUnary(String functionName, MethodHandle vectorOperand) throws Throwable {
        MethodHandle scalarFunc = getUnaryFunctionHandle(functionName); // Resolves from your parent class
        MethodHandle fallbackBridge = LOOKUP.findStatic(VectorTurboEvaluator.class, "laneByLaneFallback",
                MethodType.methodType(DoubleVector.class, MethodHandle.class, DoubleVector.class));

        MethodHandle boundBridge = MethodHandles.insertArguments(fallbackBridge, 0, scalarFunc);
        return MethodHandles.filterArguments(boundBridge, 0, vectorOperand);
    }

    public static DoubleVector laneByLaneFallback(MethodHandle scalarOp, DoubleVector input) throws Throwable {
        double[] chunk = new double[SPECIES.length()];
        input.intoArray(chunk, 0);
        for (int i = 0; i < chunk.length; i++) {
            chunk[i] = (double) scalarOp.invokeExact(chunk[i]);
        }
        return DoubleVector.fromArray(SPECIES, chunk, 0);
    }
}
