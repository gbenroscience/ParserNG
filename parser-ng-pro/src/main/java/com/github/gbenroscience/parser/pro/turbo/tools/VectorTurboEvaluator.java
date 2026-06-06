package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

/**
 * Enterprise SIMD Evaluator for ParserNG. Compiles a parallel Vector
 * MethodHandle tree for high-throughput block execution.
 *
 * REQUIRES JVM FLAG: --add-modules jdk.incubator.vector
 *
 * @author GBEMIRO
 */
public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
    private static final int VLEN = SPECIES.length();

    // Reusable allocation-free scratchpad for lane-by-lane custom scalar fallbacks
    private static final ThreadLocal<double[]> LANE_FALLBACK_BUFFER
            = ThreadLocal.withInitial(() -> new double[SPECIES.length()]);

    // Zero-allocation reusable buffers
    private static final ThreadLocal<double[]> SCALAR_FRAME
            = ThreadLocal.withInitial(() -> new double[512]); // Covers AVX-512 and future

    private static final ThreadLocal<Future<?>[]> FUTURES_CACHE
            = ThreadLocal.withInitial(() -> new Future[64]); // Max reasonable thread count

    // Zero-allocation reusable EvalResult wrapper to prevent object creation in single-row execution paths
    private static final ThreadLocal<MathExpression.EvalResult> EVAL_RESULT_CACHE
            = ThreadLocal.withInitial(() -> new MathExpression.EvalResult());

    private final MathExpression.Token[] postfix;

    public VectorTurboEvaluator(MathExpression me) {
        super(me);
        this.postfix = me.getCachedPostfix();
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        final MethodHandle rawScalarHandle = super.compileScalar(this.postfix);
        MethodHandle vectorHandle = null;
        KernelInterceptException interceptedKernel = null;

        try {
            vectorHandle = compileVector(this.postfix);
        } catch (KernelInterceptException ex) {
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
                // Fetch the persistent thread-local wrapper instead of allocating a new instance
                MathExpression.EvalResult cachedResult = EVAL_RESULT_CACHE.get();
                return cachedResult.wrap(applyScalar(variables));
            }

            @Override
            public String checkErrorLogs() {
                return "";
            }

            @Override
            public TurboExpressionEvaluator getCompiler() {
                return VectorTurboEvaluator.this;
            }

            @Override
            public void applyBulk(double[][] variables, double[] output) {
                final int length = output.length;
                final int nVars = variables.length;

                if (length < VLEN || finalVectorHandle == null) {
                    scalarBulk(variables, output, 0, length, nVars, rawScalarHandle);
                    return;
                }

                if (length >= 100_000) {
                    applyBulkParallel(variables, output);
                    return;
                }

                // SIMD hot path
                final int loopBound = SPECIES.loopBound(length);
                int i = 0;
                try {
                    for (; i < loopBound; i += VLEN) {
                        DoubleVector resultLane = (DoubleVector) finalVectorHandle.invokeExact(variables, i);
                        resultLane.intoArray(output, i);
                    }
                    if (i < length) {
                        scalarBulk(variables, output, i, length, nVars, rawScalarHandle);
                    }
                } catch (Throwable t) {
                    throw new RuntimeException("Hardware Vector SIMD Execution failed at loop index " + i, t);
                }
            }

            /**
             * High-Performance, zero-allocation bulk execution pass. The caller
             * completely manages the memory allocations by passing a
             * pre-allocated destination buffer.
             *
             * @param variables The 2D input variables matrix structured as a
             * Structure of Arrays (SoA).
             * @param outputBuffer The pre-allocated target array where the
             * evaluation results will be written.
             * @param offset The starting index position inside 'outputBuffer'
             * to begin writing results.
             */
            @Override
            public void applyBulk(double[][] variables, double[] outputBuffer, int offset) throws Throwable {
                // Enforce boundary safety constraints up front
                if (variables == null || variables.length == 0 || outputBuffer == null) {
                    return;
                }

                // Calculate length based on the total elements available in the first variable array slice
                final int length = variables[0].length;
                final int nVars = variables.length;

                // Verify destination array bounds can safely handle the incoming calculation payload
                if (offset + length > outputBuffer.length) {
                    throw new ArrayIndexOutOfBoundsException(
                            "Destination buffer overflow: Cannot write " + length + " elements starting at offset " + offset
                            + " into an output buffer of size " + outputBuffer.length
                    );
                }

                // Fetch the raw handles cached at compile time
                final MethodHandle rawScalarHandle = compileScalar(VectorTurboEvaluator.this.postfix);
                MethodHandle vectorHandle = null;
                try {
                    vectorHandle = compileVector(VectorTurboEvaluator.this.postfix);
                } catch (Throwable ignored) {
                    // Fall back cleanly if vector compiling encounters matrix kernels
                }

                // --- Fast Guard: If array sizes are too micro or hardware vector lanes aren't ready, go scalar ---
                if (length < VLEN || vectorHandle == null) {
                    // Pull a thread-local reusable scratchpad array to prevent heap allocation churn
                    double[] scalarFrame = LANE_FALLBACK_BUFFER.get();
                    for (int i = 0; i < length; i++) {
                        for (int v = 0; v < nVars; v++) {
                            scalarFrame[v] = variables[v][i];
                        }
                        try {
                            outputBuffer[offset + i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                        } catch (Throwable t) {
                            throw new RuntimeException("Scalar execution fallback failed inside bulk override at index " + i, t);
                        }
                    }
                    return;
                }

                final int loopBound = SPECIES.loopBound(length);
                int i = 0;
                try {
                    // 1. Main Hardware SIMD Vector Lane Loop
                    // Pipes data directly out of variable memory tracks straight into your destination buffer indices
                    for (; i < loopBound; i += VLEN) {
                        DoubleVector resultLane = (DoubleVector) vectorHandle.invokeExact(variables, i);
                        resultLane.intoArray(outputBuffer, offset + i);
                    }

                    // 2. Fixed Tail Scalar Remainder Guard
                    if (i < length) {
                        double[] scalarFrame = LANE_FALLBACK_BUFFER.get();
                        for (; i < length; i++) {
                            for (int v = 0; v < nVars; v++) {
                                scalarFrame[v] = variables[v][i];
                            }
                            outputBuffer[offset + i] = (double) rawScalarHandle.invokeExact(scalarFrame);
                        }
                    }
                } catch (Throwable t) {
                    throw new RuntimeException("Hardware Vector SIMD Memory-Reuse loop failed at index " + i, t);
                }
            }

            public void applyBulk(double[][] variables, double[] output, ExecutorService executor) {
                final int length = output.length;
                final int nVars = variables.length;

                if (length < 25_000) {
                    this.applyBulk(variables, output);
                    return;
                }

                if (length < VLEN || finalVectorHandle == null) {
                    scalarBulk(variables, output, 0, length, nVars, rawScalarHandle);
                    return;
                }

                final int nThreads = Math.min(
                        Runtime.getRuntime().availableProcessors(),
                        Math.max(1, length / 25_000)
                );

                Future<?>[] futures = FUTURES_CACHE.get();
                if (futures.length < nThreads) {
                    futures = new Future<?>[nThreads];
                    FUTURES_CACHE.set(futures);
                }

                final int chunk = (length + nThreads - 1) / nThreads;

                for (int t = 0; t < nThreads; t++) {
                    final int threadId = t;
                    final int start = t * chunk;
                    final int end = Math.min(start + chunk, length);
                    futures[t] = executor.submit(() -> {
                        final int loopBound = start + SPECIES.loopBound(end - start);
                        int i = start;
                        try {
                            for (; i < loopBound; i += VLEN) {
                                DoubleVector resultLane = (DoubleVector) finalVectorHandle.invokeExact(variables, i);
                                resultLane.intoArray(output, i);
                            }
                            if (i < end) {
                                scalarBulk(variables, output, i, end, nVars, rawScalarHandle);
                            }
                        } catch (Throwable e) {
                            throw new RuntimeException("Executor task failed at index " + i + " thread " + threadId, e);
                        }
                    });
                }

                for (int t = 0; t < nThreads; t++) {
                    try {
                        futures[t].get();
                        futures[t] = null; // Help GC
                    } catch (ExecutionException e) {
                        Throwable cause = e.getCause() != null ? e.getCause() : e;
                        throw new RuntimeException("Parallel execution failed", cause);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted during parallel execution", e);
                    }
                }
            }

            private void scalarBulk(double[][] variables, double[] output, int from, int to, int nVars, MethodHandle scalarHandle) {
                double[] frame = SCALAR_FRAME.get();
                if (frame.length < nVars) {
                    frame = new double[nVars];
                    SCALAR_FRAME.set(frame);
                }
                for (int i = from; i < to; i++) {
                    for (int v = 0; v < nVars; v++) {
                        frame[v] = variables[v][i];
                    }
                    try {
                        output[i] = (double) scalarHandle.invokeExact(frame);
                    } catch (Throwable t) {
                        throw new RuntimeException("Scalar fallback failed at index " + i, t);
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
                    final int loopBound = start + SPECIES.loopBound(end - start);

                    int i = start;
                    try {
                        for (; i < loopBound; i += VLEN) {
                            DoubleVector resultLane = (DoubleVector) finalVectorHandle.invokeExact(variables, i);
                            resultLane.intoArray(output, i);
                        }
                        if (i < end) {
                            scalarBulk(variables, output, i, end, nVars, rawScalarHandle);
                        }
                    } catch (Throwable t) {
                        throw new RuntimeException("Parallel SIMD failed at index " + i + " on thread " + threadId, t);
                    }
                });
            }

            @Override
            public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
                String kernelToRun = (op != null) ? op
                        : (finalKernelSignal != null) ? finalKernelSignal.getKernelName() : null;

                if (kernelToRun == null) {
                    throw new UnsupportedOperationException("No matrix kernel specified or intercepted.");
                }

                switch (kernelToRun.toLowerCase()) {
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
                    case "matmul_bias_gelu":
                    case "matmul_add_bias_gelu":
                        if (inputs.length != 3) {
                            throw new IllegalArgumentException("matmul_bias_gelu requires 3 inputs");
                        }
                        FlatMatrix.matmulAddBiasGelu(inputs[0], inputs[1], inputs[2], output);
                        break;
                    case "matmul_add_sin":
                        if (inputs.length != 3) {
                            throw new IllegalArgumentException("matmul_add_sin requires 3 inputs");
                        }
                        double alpha = inputs[2].get(0, 0);
                        FlatMatrix.matmulAddSin(inputs[0], inputs[1], output, alpha);
                        break;
                    case "softmax":
                    case "softmax_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("softmax requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.softmaxRowsInPlace();
                        break;
                    case "relu":
                    case "relu_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("relu requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.reluInPlace();
                        break;
                    case "gelu":
                    case "gelu_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("gelu requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.geluInPlace();
                        break;
                    case "exp":
                    case "exp_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("exp requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.expInPlace();
                        break;
                    case "log":
                    case "log_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("log requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.logInPlace();
                        break;
                    case "tanh":
                    case "tanh_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("tanh requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.tanhInPlace();
                        break;
                    case "sin":
                    case "sin_in_place":
                        if (inputs.length != 1) {
                            throw new IllegalArgumentException("sin requires 1 input");
                        }
                        if (inputs[0] != output) {
                            System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        }
                        output.sinInPlace();
                        break;
                    default:
                        throw new UnsupportedOperationException("Unknown Matrix Kernel: " + kernelToRun);
                }

            }

            public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op, ExecutorService executor) {
                if (executor == null || !isElementWiseOp(op)) {
                    applyMatrixKernel(inputs, output, op);
                    return;
                }
                runElementWiseParallel(inputs[0], output, op, executor);
            }

            private boolean isElementWiseOp(String op) {
                if (op == null) {
                    return false;
                }
                return switch (op.toLowerCase()) {
                    case "relu", "gelu", "exp", "log", "tanh", "sin", "softmax" ->
                        true;
                    default ->
                        false;
                };
            }

            private void runElementWiseParallel(FlatMatrix in, FlatMatrix out, String op, ExecutorService executor) {
                final int length = in.data.length;
                final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                        Math.max(1, length / 50_000));
                final int chunk = (length + nThreads - 1) / nThreads;
                Future<?>[] futures = FUTURES_CACHE.get();
                if (futures.length < nThreads) {
                    futures = new Future<?>[nThreads];
                    FUTURES_CACHE.set(futures);
                }

                for (int t = 0; t < nThreads; t++) {
                    final int start = t * chunk;
                    final int end = Math.min(start + chunk, length);
                    futures[t] = executor.submit(() -> applyElementWiseKernel(in.data, out.data, start, end, op));
                }

                for (int t = 0; t < nThreads; t++) {
                    try {
                        futures[t].get();
                        futures[t] = null;
                    } catch (ExecutionException e) {
                        throw new RuntimeException("Element-wise kernel failed", e.getCause() != null ? e.getCause() : e);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Interrupted", e);
                    }
                }
            }

            private void applyElementWiseKernel(double[] in, double[] out, int start, int end, String op) {
                int i = start;
                if (VLEN > 1) {
                    int loopBound = start + SPECIES.loopBound(end - start);
                    for (; i < loopBound; i += VLEN) {
                        var v = DoubleVector.fromArray(SPECIES, in, i);
                        var r = switch (op.toLowerCase()) {
                            case "relu" ->
                                v.max(0.0);
                            case "gelu" ->
                                FlatMatrix.VectorMath.geluVector(v);
                            case "exp" ->
                                v.lanewise(VectorOperators.EXP);
                            case "log" ->
                                v.lanewise(VectorOperators.LOG);
                            case "tanh" ->
                                v.lanewise(VectorOperators.TANH);
                            case "sin" ->
                                v.lanewise(VectorOperators.SIN);
                            default ->
                                throw new UnsupportedOperationException(op);
                        };
                        r.intoArray(out, i);
                    }
                }
                for (; i < end; i++) {
                    out[i] = switch (op.toLowerCase()) {
                        case "relu" ->
                            Math.max(0, in[i]);
                        case "exp" ->
                            Math.exp(in[i]);
                        case "log" ->
                            Math.log(in[i]);
                        case "tanh" ->
                            Math.tanh(in[i]);
                        case "sin" ->
                            Math.sin(in[i]);
                        default ->
                            throw new UnsupportedOperationException(op);
                    };
                }
            }
        };
    }

    // ==================== VECTOR COMPILATION ====================
    private MethodHandle compileVector(MathExpression.Token[] postfix) throws Throwable {
        Stack<MethodHandle> stack = new Stack<>();
        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    MethodHandle bcast = LOOKUP.findStatic(VectorTurboEvaluator.class, "broadcastConstant",
                            MethodType.methodType(DoubleVector.class, double.class));
                    bcast = MethodHandles.insertArguments(bcast, 0, t.value);
                    stack.push(MethodHandles.dropArguments(bcast, 0, double[][].class, int.class));
                    break;

                case MathExpression.Token.VARIABLE:
                    int finalIndex = resolveSlotIndex(t.frameIndex);
                    MethodHandle loadHandle = LOOKUP.findStatic(VectorTurboEvaluator.class, "loadVector",
                            MethodType.methodType(DoubleVector.class, double[][].class, int.class, int.class));
                    stack.push(MethodHandles.insertArguments(loadHandle, 1, finalIndex));
                    break;

                case MathExpression.Token.OPERATOR:
                    MethodHandle right = stack.pop();
                    MethodHandle left = stack.pop();

                    // Retrieve parameter counts to check if these are absolute literal nodes
                    int lParams = left.type().parameterCount();
                    int rParams = right.type().parameterCount();

                    if (t.opChar == '%') {
                        // Check if both sides are compile-time constants (e.g., 10 % 3)
                        if (lParams == 0 && rParams == 0) {
                            double lVal = getConstantValuePrimitive(left);
                            double rVal = getConstantValuePrimitive(right);
                            if (!Double.isNaN(lVal) && !Double.isNaN(rVal)) {
                                double result = lVal % rVal;
                                stack.push(createVectorBroadcastConstantNode(result));
                                break;
                            }
                        }
                        stack.push(applyVectorRemainderOp(left, right));
                    } else if (isRelationalOp(t.opChar)) {
                        // Fold constant comparisons (e.g., 5 > 2 -> broadcasts 1.0)
                        if (lParams == 0 && rParams == 0) {
                            double lVal = getConstantValuePrimitive(left);
                            double rVal = getConstantValuePrimitive(right);
                            if (!Double.isNaN(lVal) && !Double.isNaN(rVal)) {
                                boolean matches = evaluateScalarComparison(t.opChar, lVal, rVal);
                                double result = matches ? 1.0 : 0.0;
                                stack.push(createVectorBroadcastConstantNode(result));
                                break;
                            }
                        }
                        VectorOperators.Comparison compOp = mapComparisonOp(t.opChar);
                        stack.push(applyVectorComparisonOp(compOp, left, right));
                    } else {
                        // Standard Arithmetic Folding (e.g., 2 * pi)
                        if (lParams == 0 && rParams == 0) {
                            double lVal = getConstantValuePrimitive(left);
                            double rVal = getConstantValuePrimitive(right);
                            if (!Double.isNaN(lVal) && !Double.isNaN(rVal)) {
                                double result = evaluateScalarArithmetic(t.opChar, lVal, rVal);
                                stack.push(createVectorBroadcastConstantNode(result));
                                break;
                            }
                        }
                        VectorOperators.Binary binOp = mapBinaryOp(t.opChar);
                        stack.push(applyVectorBinaryOp(binOp, left, right));
                    }
                    break;

                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD:
                    if (isMatrixKernel(t.name, t.arity)) {
                        throw new KernelInterceptException(t.name, t.arity);
                    }
                    // CRITICAL ENTERPRISE ARITY VALIDATION GUARDS
                    // Intercepts wrong argument counts BEFORE stack corruption can occur.

                    if ("vma".equalsIgnoreCase(t.name) && t.arity != 3) {
                        throw new MalformedExpressionException(
                                "Invalid compilation arity: The fused vector multiply-add function 'vma(A, B, C)' requires exactly 3 arguments. Found: " + t.arity
                        );
                    }
                    if ("if".equalsIgnoreCase(t.name) && t.arity != 3) {
                        throw new MalformedExpressionException(
                                "Invalid compilation arity: The hardware conditional masking function 'if(condition, trueBranch, falseBranch)' requires exactly 3 arguments. Found: " + t.arity
                        );
                    }
                    if (t.arity == 1) {
                        MethodHandle operand = stack.pop();
                        VectorOperators.Unary unOp = mapUnaryOp(t.name);
                        if (unOp != null) {
                            stack.push(applyVectorUnaryOp(unOp, operand));
                        } else {
                            stack.push(createScalarFallbackUnary(t.name, operand));
                        }
                    } else if (t.arity == 2) {
                        MethodHandle r = stack.pop();
                        MethodHandle l = stack.pop();
                        VectorOperators.Binary bOp = mapBinaryOpName(t.name);
                        stack.push(applyVectorBinaryOp(bOp, l, r));
                    } else if (t.arity == 3 && "vma".equalsIgnoreCase(t.name)) {
                        // vma(A, B, C) -> A * B + C
                        MethodHandle cHandle = stack.pop();
                        MethodHandle bHandle = stack.pop();
                        MethodHandle aHandle = stack.pop();
                        stack.push(applyVectorFmaOp(aHandle, bHandle, cHandle));
                    } else if (t.arity == 3 && "if".equalsIgnoreCase(t.name)) {
                        // if(condition, trueBranch, falseBranch)
                        // e.g if(x > 0.5, x * 2, 0)
                        MethodHandle falseBranch = stack.pop();
                        MethodHandle trueBranch = stack.pop();
                        MethodHandle condition = stack.pop();
                        // UPGRADED: Routes through the branch-elimination optimizer 
                        // instead of blindly building full mask combinator trees.
                        stack.push(optimizeVectorConditionalMask(condition, trueBranch, falseBranch));
                    }
                    break;

            }
        }
        return stack.pop();
    }

    private boolean isMatrixKernel(String funcName, int arity) {
        return switch (funcName.toLowerCase()) {
            case "matmul", "matadd", "add_mat" ->
                arity == 2;
            case "softmax", "softmax_in_place", "relu_in_place", "gelu_in_place", "exp_in_place", "log_in_place", "tanh_in_place", "sin_in_place" ->
                arity == 1;
            case "matmul_add_sin", "matmul_bias_gelu", "matmul_add_bias_gelu" ->
                arity == 3;
            default ->
                false;
        };
    }

    // ==================== NATIVE HARDWARE VECTOR EXECUTIONS ====================
    public static DoubleVector execVectorFma(DoubleVector a, DoubleVector b, DoubleVector c) {
        // Maps directly to CPU opcode VFMADD231PD/VFMADD132PD 
        return a.fma(b, c);
    }

    public static DoubleVector execVectorMask(DoubleVector condition, DoubleVector trueVal, DoubleVector falseVal) {
        // Generates an optimization lane condition mask based on comparisons against 0.0
        // If condition value > 0.0, select lane from trueVal, else select lane from falseVal
        VectorMask<Double> mask = condition.compare(VectorOperators.GT, 0.0);
        return falseVal.blend(trueVal, mask);
    }

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

    private static VectorOperators.Binary mapBinaryOp(char op) {
        return switch (op) {
            case '+' ->
                VectorOperators.ADD;
            case '-' ->
                VectorOperators.SUB;
            case '*' ->
                VectorOperators.MUL;
            case '/' ->
                VectorOperators.DIV;
            case '^' ->
                VectorOperators.POW;
            default ->
                throw new IllegalArgumentException("Unsupported SIMD binary operator: " + op);
        };
    }

    private static VectorOperators.Binary mapBinaryOpName(String name) {
        return switch (name.toLowerCase()) {
            case "pow" ->
                VectorOperators.POW;
            case "atan2" ->
                VectorOperators.ATAN2;
            case "min" ->
                VectorOperators.MIN;
            case "max" ->
                VectorOperators.MAX;
            default ->
                throw new IllegalArgumentException("Unsupported SIMD binary function: " + name);
        };
    }
    
    private static VectorOperators.Unary mapUnaryOp(String name) {
        return switch (name.toLowerCase()) {
            case "sin" ->
                VectorOperators.SIN;
            case "cos" ->
                VectorOperators.COS;
            case "tan" ->
                VectorOperators.TAN;
            case "asin" ->
                VectorOperators.ASIN;
            case "acos" ->
                VectorOperators.ACOS;
            case "atan" ->
                VectorOperators.ATAN;
            case "sinh" ->
                VectorOperators.SINH;
            case "cosh" ->
                VectorOperators.COSH;
            case "tanh" ->
                VectorOperators.TANH;
            case "sqrt", "√" ->
                VectorOperators.SQRT;
            case "cbrt" ->
                VectorOperators.CBRT;
            case "abs" ->
                VectorOperators.ABS;
            case "log", "ln" ->
                VectorOperators.LOG;
            case "lg" ->
                VectorOperators.LOG10;
            case "exp" ->
                VectorOperators.EXP;
            default ->
                null;
        };
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

    private static boolean isRelationalOp(char op) {
        return op == '>' || op == '<' || op == '=' || op == '!' || op == 'g' || op == 'l';
    }

    private static VectorOperators.Comparison mapComparisonOp(char op) {
        return switch (op) {
            case '>' ->
                VectorOperators.GT; // Greater Than
            case '<' ->
                VectorOperators.LT; // Less Than
            case '=' ->
                VectorOperators.EQ; // Equal To
            case '!' ->
                VectorOperators.NE; // Not Equal To
            case 'g' ->
                VectorOperators.GE; // Greater Than or Equal To (sentinel 'g' or '≥')
            case 'l' ->
                VectorOperators.LE; // Less Than or Equal To (sentinel 'l' or '≤')
            default ->
                throw new IllegalArgumentException("Unsupported SIMD relational operator: " + op);
        };
    }

    ////////////
      /**
     * Compiles a Fused Multiply-Add (FMA) vector operation: (A * B) + C.
     * Executes inside a single CPU hardware cycle bypassing intermediate
     * rounding.
     */
        private static MethodHandle applyVectorFmaOp(MethodHandle a, MethodHandle b, MethodHandle c) throws Throwable {
        MethodHandle target = LOOKUP.findStatic(VectorTurboEvaluator.class, "pipeFmaVector",
                MethodType.methodType(DoubleVector.class, MethodHandle.class, MethodHandle.class, MethodHandle.class, double[][].class, int.class));
        return MethodHandles.insertArguments(target, 0, a, b, c);
    }

    /**
     * Compiles a hardware-accelerated relational comparison operator. Evaluates
     * lanes concurrently and outputs 1.0 for true conditions, 0.0 for false.
     * Supports GT, LT, EQ, NE, GE, and LE operations.
     */
    private static MethodHandle applyVectorComparisonOp(VectorOperators.Comparison op, MethodHandle left, MethodHandle right) throws Throwable {
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execVectorComparison",
                MethodType.methodType(DoubleVector.class, VectorOperators.Comparison.class, DoubleVector.class, DoubleVector.class));
        MethodHandle boundOp = MethodHandles.insertArguments(baseOp, 0, op);

        MethodHandle target = LOOKUP.findStatic(VectorTurboEvaluator.class, "pipeBinaryVectors",
                MethodType.methodType(DoubleVector.class, MethodHandle.class, MethodHandle.class, MethodHandle.class, double[][].class, int.class));
        return MethodHandles.insertArguments(target, 0, boundOp, left, right);
    }

    public static DoubleVector pipeFmaVector(MethodHandle a, MethodHandle b, MethodHandle c, double[][] vars, int offset) throws Throwable {
        DoubleVector aVec = (DoubleVector) a.invokeExact(vars, offset);
        DoubleVector bVec = (DoubleVector) b.invokeExact(vars, offset);
        DoubleVector cVec = (DoubleVector) c.invokeExact(vars, offset);
        return aVec.fma(bVec, cVec);
    }

    ///////////////////////////////////////////////////////////////////////
    
    
    
        private static MethodHandle applyVectorBinaryOp(VectorOperators.Binary op, MethodHandle left, MethodHandle right) throws Throwable {
        // 1. Get the base operator handle: (VectorOperators.Binary, DoubleVector, DoubleVector)DoubleVector
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execBinaryVector",
                MethodType.methodType(DoubleVector.class, VectorOperators.Binary.class, DoubleVector.class, DoubleVector.class));

        // 2. Bind the specific operator: (DoubleVector, DoubleVector)DoubleVector
        MethodHandle boundOp = MethodHandles.insertArguments(baseOp, 0, op);

        // 3. Instead of permute/filter/fold combinations that append signatures,
        // compile a flat target adapter method via MethodHandles.collectArguments.
        // This explicitly pipes (double[][], int) straight down to the execution nodes.
        MethodHandle target = LOOKUP.findStatic(VectorTurboEvaluator.class, "pipeBinaryVectors",
                MethodType.methodType(DoubleVector.class, MethodHandle.class, MethodHandle.class, MethodHandle.class, double[][].class, int.class));

        return MethodHandles.insertArguments(target, 0, boundOp, left, right);
    }

    private static MethodHandle applyVectorUnaryOp(VectorOperators.Unary op, MethodHandle operand) throws Throwable {
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execUnaryVector",
                MethodType.methodType(DoubleVector.class, VectorOperators.Unary.class, DoubleVector.class));

        MethodHandle boundOp = MethodHandles.insertArguments(baseOp, 0, op);

        MethodHandle target = LOOKUP.findStatic(VectorTurboEvaluator.class, "pipeUnaryVector",
                MethodType.methodType(DoubleVector.class, MethodHandle.class, MethodHandle.class, double[][].class, int.class));

        return MethodHandles.insertArguments(target, 0, boundOp, operand);
    }

    // ==================== FAST RUNTIME VECTOR PIPING ENGINES ====================
    public static DoubleVector pipeBinaryVectors(MethodHandle op, MethodHandle left, MethodHandle right, double[][] vars, int offset) throws Throwable {
        // Evaluate the left and right nodes straight out of memory registers natively
        DoubleVector lVec = (DoubleVector) left.invokeExact(vars, offset);
        DoubleVector rVec = (DoubleVector) right.invokeExact(vars, offset);
        // Execute the binary lanewise vector operation
        return (DoubleVector) op.invokeExact(lVec, rVec);
    }

    public static DoubleVector pipeUnaryVector(MethodHandle op, MethodHandle operand, double[][] vars, int offset) throws Throwable {
        DoubleVector val = (DoubleVector) operand.invokeExact(vars, offset);
        return (DoubleVector) op.invokeExact(val);
    }

    ///////////////////////////////////////////////////////////////////////////
    

    /**
     *
     * Compiles low-latency conditional masking. Translates 'if(cond,
     * trueVal,falseVal)' straight into a hardware VectorMask. Analyzes the
     * condition slot of an 'if' function node during compilation. If the
     * condition is an absolute constant, it eliminates the dead branch entirely
     * and returns the optimized active branch handle. Otherwise, it compiles
     * the standard hardware-vectorized blending mask handle.
     */
    private static MethodHandle optimizeVectorConditionalMask(MethodHandle condition, MethodHandle trueBranch, MethodHandle falseBranch) throws Throwable {

        // Check if the condition node requires ZERO parameters (meaning it's an absolute literal)
        if (condition.type().parameterCount() == 0) {
            try {
                // Safely extract the constant condition value using zero-allocation invokeExact
                double conditionValue = (double) condition.invokeExact();

                // ParserNG Semantic Guard: If value > 0.0, the condition is permanently TRUE.
                // Discard the false branch completely and return the true branch.
                if (conditionValue > 0.0) {
                    return trueBranch;
                } else {
                    // Otherwise, condition is permanently FALSE. Discard the true branch.
                    return falseBranch;
                }
            } catch (Throwable t) {
                // Fall back gracefully to the standard dynamic vector path if extraction fails
            }
        }

        // --- Standard Dynamic Path ---
        // If the condition is variable-dependent, compile the full register-level blending mask
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execVectorMask",
                MethodType.methodType(DoubleVector.class, DoubleVector.class, DoubleVector.class, DoubleVector.class));

        MethodHandle filter = MethodHandles.filterArguments(baseOp, 0, condition, trueBranch);
        MethodHandle permuted = MethodHandles.permuteArguments(filter,
                MethodType.methodType(DoubleVector.class, double[][].class, int.class, double[][].class, int.class, DoubleVector.class), 0, 1, 0, 1, 2);

        MethodHandle combined = MethodHandles.filterArguments(permuted, 2, falseBranch);
        return MethodHandles.foldArguments(MethodHandles.foldArguments(combined, trueBranch), condition);
    }

    // NATIVE VECTOR COMPARISON LAYER 
    public static DoubleVector execVectorComparison(VectorOperators.Comparison op, DoubleVector a, DoubleVector b) {
        // Generates the hardware mask vector matching your target comparison operator
        VectorMask<Double> mask = a.compare(op, b);

        // Zero-allocation lane blending directly inside CPU vector registers
        DoubleVector vTrue = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector vFalse = DoubleVector.zero(SPECIES);
        return vFalse.blend(vTrue, mask);
    }

    private static MethodHandle applyVectorRemainderOp(MethodHandle left, MethodHandle right) throws Throwable {
        MethodHandle baseOp = LOOKUP.findStatic(VectorTurboEvaluator.class, "execVectorRemainder",
                MethodType.methodType(DoubleVector.class, DoubleVector.class, DoubleVector.class));

        // Swapping parameters so foldArguments pipes evaluation tokens into slots correctly
        MethodHandle fnSwapped = MethodHandles.permuteArguments(baseOp,
                MethodType.methodType(DoubleVector.class, DoubleVector.class, DoubleVector.class), 1, 0);

        MethodHandle combiner = MethodHandles.filterArguments(fnSwapped, 1, left);
        return MethodHandles.foldArguments(combiner, right);
    }

    /**
     *
     * Executes a high-performance hardware-vectorized remainder operation (A %
     * B). Compiles down to an optimized div-trunc-mul-sub assembly pipeline.
     *
     * @param a
     * @param b
     * @return
     */
    public static DoubleVector execVectorRemainder(DoubleVector a, DoubleVector b) {
        // 1. Compute floating-point division: a / b
        DoubleVector quotient = a.div(b);

        // 2. Hardware-Level Truncation toward zero:
        // Convert DoubleVector -> LongVector (discards fractional parts)
        VectorSpecies<Long> longSpecies = LongVector.SPECIES_PREFERRED;
        LongVector truncatedLongs = (LongVector) quotient.convert(VectorOperators.Conversion.ofCast(double.class, long.class), 0);

        // Convert LongVector back -> DoubleVector
        DoubleVector truncatedQuotient = (DoubleVector) truncatedLongs.convert(VectorOperators.Conversion.ofCast(long.class, double.class), 0);

        // 3. Fused Multiply-Subtract: a - (truncatedQuotient * b)
        return a.sub(truncatedQuotient.mul(b));
    }

    private MethodHandle createScalarFallbackUnary(String functionName, MethodHandle vectorOperand) throws Throwable {
        MethodHandle scalarFunc = getUnaryFunctionHandle(functionName);
        MethodHandle fallbackBridge = LOOKUP.findStatic(VectorTurboEvaluator.class, "laneByLaneFallback",
                MethodType.methodType(DoubleVector.class, MethodHandle.class, DoubleVector.class));
        MethodHandle boundBridge = MethodHandles.insertArguments(fallbackBridge, 0, scalarFunc);
        return MethodHandles.filterArguments(boundBridge, 0, vectorOperand);
    }

    public static DoubleVector laneByLaneFallback(MethodHandle scalarOp, DoubleVector input) throws Throwable {
        double[] chunk = SCALAR_FRAME.get();
        input.intoArray(chunk, 0);
        for (int i = 0; i < chunk.length; i++) {
            chunk[i] = (double) scalarOp.invokeExact(chunk[i]);
        }
        return DoubleVector.fromArray(SPECIES, chunk, 0);
    }

    /**
     * Synthesizes a completely flat MethodHandle that drops incoming array
     * arguments and broadcasts a pre-calculated scalar constant straight into
     * hardware vector registers.
     */
    private static MethodHandle createVectorBroadcastConstantNode(double constantValue) throws Throwable {
        MethodHandle bcast = LOOKUP.findStatic(VectorTurboEvaluator.class, "broadcastConstant",
                MethodType.methodType(DoubleVector.class, double.class));
        bcast = MethodHandles.insertArguments(bcast, 0, constantValue);
        // Signature: (double[][], int)DoubleVector — pipes seamlessly into your existing stack
        return MethodHandles.dropArguments(bcast, 0, double[][].class, int.class);
    }

    /**
     * Pre-evaluates standard arithmetic operators during the compilation token
     * pass.
     */
    private static double evaluateScalarArithmetic(char op, double left, double right) {
        return switch (op) {
            case '+' ->
                left + right;
            case '-' ->
                left - right;
            case '*' ->
                left * right;
            case '/' ->
                left / right;
            case '%' ->
                left % right;
            case '^' ->
                Math.pow(left, right);
            default ->
                throw new IllegalArgumentException("Unsupported compile-time folding operator: " + op);
        };
    }

    /**
     * Pre-evaluates relational operators during the compilation token pass.
     */
    private static boolean evaluateScalarComparison(char op, double left, double right) {
        return switch (op) {
            case '>' ->
                left > right;
            case '<' ->
                left < right;
            case '=' ->
                left == right;
            case '!' ->
                left != right;
            case 'g' ->
                left >= right;
            case 'l' ->
                left <= right;
            default ->
                throw new IllegalArgumentException("Unsupported compile-time folding comparison: " + op);
        };
    }

    /**
     * Extracts primitive values from intermediate constant handles with zero
     * boxing.
     */
    private static double getConstantValuePrimitive(MethodHandle handle) {
        if (handle != null) {
            try {
                // If it is a nested dropArguments constant node, unwrap its base value
                return (double) MethodHandles.reflectAs(java.lang.reflect.Method.class, handle).invoke(null);
            } catch (Throwable t) {
                // If reflection is blocked or restricted, execute a zero-argument invocation check
                if (handle.type().parameterCount() == 0) {
                    try {
                        return (double) handle.invokeExact();
                    } catch (Throwable ignored) {
                    }
                }
            }
        }
        return Double.NaN;
    }

    /**
     * Thrown when an expression violates semantic boundaries or structural
     * arity constraints during ParserNG Turbo compilation.
     */
    static public class MalformedExpressionException extends RuntimeException {

        public MalformedExpressionException(String message) {
            super(message);
        }
    }

}
