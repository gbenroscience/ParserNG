package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;

/**
 * SIMD Turbo Evaluator - High-Performance Block Vectorization Engine *
 * Strategy: Processes data in cache-friendly column-major blocks (e.g., 2048
 * elements) per instruction stream opcode. This eliminates runtime allocations
 * completely, flattens interpreter switch branches, and allows HotSpot JIT
 * Superword Level Parallelism (SLP) to natively auto-vectorize the inner
 * primitive primitive loops via AVX/Neon instructions.
 *
 * * @author GBEMIRO
 */
public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    // Opcode Constants
    private static final int OP_CONST = 1;
    private static final int OP_LOAD = 2;
    private static final int OP_ADD = 3;
    private static final int OP_SUB = 4;
    private static final int OP_MUL = 5;
    private static final int OP_DIV = 6;
    private static final int OP_POW = 7;
    private static final int OP_SIN = 8;
    private static final int OP_COS = 9;
    private static final int OP_TAN = 10;
    private static final int OP_ASIN = 11;
    private static final int OP_ACOS = 12;
    private static final int OP_ATAN = 13;
    private static final int OP_SINH = 14;
    private static final int OP_COSH = 15;
    private static final int OP_TANH = 16;
    private static final int OP_ABS = 17;
    private static final int OP_EXP = 18;
    private static final int OP_SQRT = 19;
    private static final int OP_CBRT = 20;
    private static final int OP_LOG = 21;
    private static final int OP_LOG10 = 22;
    private static final int OP_VMA = 23;
    private static final int OP_REM = 24;
    private static final int OP_IF = 25;
    private static final int OP_GT = 26;
    private static final int OP_LT = 27;
    private static final int OP_EQ = 28;
    private static final int OP_NE = 29;
    private static final int OP_GE = 30;
    private static final int OP_LE = 31;

    // Pre-allocated compilation state
    private final MathExpression.Token[] postfix;
    private final MethodHandle compiledScalarHandle;
    private int[] opcodes;
    private int[] targetSlots;
    private double[] literalConstants;
    private int instructionCount;
    private KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator(MathExpression me) throws Throwable {
        super(me);
        this.postfix = me.getCachedPostfix();
        this.compiledScalarHandle = compileScalar(postfix);
        compileToPrimitiveProgram();
    }

    private void compileToPrimitiveProgram() {
        int len = postfix.length;
        this.opcodes = new int[len];
        this.targetSlots = new int[len];
        this.literalConstants = new double[len];
        this.instructionCount = 0;
        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER -> {
                    opcodes[instructionCount] = OP_CONST;
                    literalConstants[instructionCount] = t.value;
                    instructionCount++;
                }
                case MathExpression.Token.VARIABLE -> {
                    opcodes[instructionCount] = OP_LOAD;
                    targetSlots[instructionCount] = resolveSlotIndex(t.frameIndex);
                    instructionCount++;
                }
                case MathExpression.Token.OPERATOR -> {
                    opcodes[instructionCount] = switch (t.opChar) {
                        case '+' ->
                            OP_ADD;
                        case '-' ->
                            OP_SUB;
                        case '*' ->
                            OP_MUL;
                        case '/' ->
                            OP_DIV;
                        case '^' ->
                            OP_POW;
                        case '%' ->
                            OP_REM;
                        case '>' ->
                            OP_GT;
                        case '<' ->
                            OP_LT;
                        default ->
                            throw new IllegalArgumentException("Unknown operator: " + t.opChar);
                    };
                    instructionCount++;
                }
                case MathExpression.Token.FUNCTION, MathExpression.Token.METHOD -> {
                    if (isMatrixKernel(t.name, t.arity)) {
                        interceptedKernel = new KernelInterceptException(t.name, t.arity);
                        return;
                    }
                    String name = t.name.toLowerCase();
                    opcodes[instructionCount] = switch (name) {
                        case "sin", "sin_rad" ->
                            OP_SIN;
                        case "cos", "cos_rad" ->
                            OP_COS;
                        case "tan", "tan_rad" ->
                            OP_TAN;
                        case "asin", "asin_rad" ->
                            OP_ASIN;
                        case "acos", "acos_rad" ->
                            OP_ACOS;
                        case "atan", "atan_rad" ->
                            OP_ATAN;
                        case "sinh" ->
                            OP_SINH;
                        case "cosh" ->
                            OP_COSH;
                        case "tanh" ->
                            OP_TANH;
                        case "abs" ->
                            OP_ABS;
                        case "exp" ->
                            OP_EXP;
                        case "sqrt" ->
                            OP_SQRT;
                        case "cbrt" ->
                            OP_CBRT;
                        case "log", "ln" ->
                            OP_LOG;
                        case "log10", "lg" ->
                            OP_LOG10;
                        case "vma" ->
                            OP_VMA;
                        case "if" ->
                            OP_IF;
                        default ->
                            throw new IllegalArgumentException("Unknown function: " + t.name);
                    };
                    instructionCount++;
                }
            }
        }
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        return new BatchedVectorCompositeExpression(compiledScalarHandle, opcodes, targetSlots,
                literalConstants, instructionCount);
    }

    public final class BatchedVectorCompositeExpression implements FastCompositeExpression {

        private final MethodHandle scalarHandle;
        private final int[] opcodes;
        private final int[] targetSlots;
        private final double[] literalConstants;
        private final int instructionCount;
        private final double[] scalarFrameBuffer;

        // Optimized block size designed to comfortably fit into standard CPU L1/L2 caches (2048 * 8 bytes = 16KB)
        private static final int BLOCK_SIZE = 2048;

        // Zero-allocation reusable intermediate stack pads pooled per thread
        private static final ThreadLocal<double[][]> SCRATCH_STACKS = ThreadLocal.withInitial(() -> {
            // Accommodates max expression stack depth evaluation of up to 64 nested levels
            double[][] blocks = new double[64][BLOCK_SIZE];
            return blocks;
        });

        BatchedVectorCompositeExpression(MethodHandle handle, int[] ops, int[] slots,
                double[] consts, int count) {
            this.scalarHandle = handle;
            this.opcodes = ops;
            this.targetSlots = slots;
            this.literalConstants = consts;
            this.instructionCount = count;
            this.scalarFrameBuffer = new double[128];
        }

        @Override
        public double applyScalar(double[] variables) {
            try {
                return (double) scalarHandle.invokeExact(variables);
            } catch (Throwable t) {
                throw new RuntimeException("Turbo evaluation failed", t);
            }
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
            return VectorTurboEvaluator.this;
        }

        public void applyBulk(double[][] variables, double[] output) {
            applyBulkInternal(variables, output, 0, output.length);
        }

        public void applyBulk(double[][] variables, double[] output, int offset) {
            applyBulkInternal(variables, output, offset, output.length - offset);
        }

        /**
         * Core Column-Major Vectorized Loop Processor
         */
        private void applyBulkInternal(double[][] variables, double[] output, int startIdx, int length) {
            if (variables == null || variables.length == 0 || length <= 0) {
                return;
            }

            final double[][] executionStack = SCRATCH_STACKS.get();
            final int endIdx = startIdx + length;

            // Step 1: Chunk operations into L1/L2 cache-friendly blocks
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                int sp = 0; // Flat local stack pointer resetting per block segment

                // Step 2: Traverse opcodes sequentially across the entire current data chunk
                for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                    final int opcode = opcodes[instIdx];

                    switch (opcode) {
                        case OP_CONST -> {
                            final double val = literalConstants[instIdx];
                            final double[] stackArr = executionStack[sp++];
                            Arrays.fill(stackArr, 0, currentBlockSize, val);
                        }

                        case OP_LOAD -> {
                            final int slotIdx = targetSlots[instIdx];
                            final double[] varArray = variables[slotIdx];
                            final double[] stackArr = executionStack[sp++];
                            System.arraycopy(varArray, blockStart, stackArr, 0, currentBlockSize);
                        }

                        case OP_ADD -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            // Clean primitive loop structure trivial for Hotspot JIT SIMD optimization
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] + r[k];
                            }
                        }

                        case OP_SUB -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] - r[k];
                            }
                        }

                        case OP_MUL -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] * r[k];
                            }
                        }

                        case OP_DIV -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] / r[k];
                            }
                        }

                        case OP_POW -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = pow(l[k], r[k]);
                            }
                        }

                        case OP_REM -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] % r[k];
                            }
                        }

                        case OP_SIN -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.sin(src[k]);
                            }
                        }

                        case OP_COS -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.cos(src[k]);
                            }
                        }

                        case OP_TAN -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.tan(src[k]);
                            }
                        }

                        case OP_ASIN -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.asin(src[k]);
                            }
                        }

                        case OP_ACOS -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.acos(src[k]);
                            }
                        }

                        case OP_ATAN -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.atan(src[k]);
                            }
                        }

                        case OP_SINH -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.sinh(src[k]);
                            }
                        }

                        case OP_COSH -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.cosh(src[k]);
                            }
                        }

                        case OP_TANH -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.tanh(src[k]);
                            }
                        }

                        case OP_ABS -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.abs(src[k]);
                            }
                        }

                        case OP_EXP -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.exp(src[k]);
                            }
                        }

                        case OP_SQRT -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.sqrt(src[k]);
                            }
                        }

                        case OP_CBRT -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.cbrt(src[k]);
                            }
                        }

                        case OP_LOG -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.log(src[k]);
                            }
                        }

                        case OP_LOG10 -> {
                            final double[] src = executionStack[sp - 1];
                            for (int k = 0; k < currentBlockSize; k++) {
                                src[k] = Math.log10(src[k]);
                            }
                        }

                        case OP_GT -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] > r[k] ? 1.0 : 0.0;
                            }
                        }

                        case OP_LT -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] < r[k] ? 1.0 : 0.0;
                            }
                        }

                        case OP_EQ -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] == r[k] ? 1.0 : 0.0;
                            }
                        }

                        case OP_NE -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] != r[k] ? 1.0 : 0.0;
                            }
                        }

                        case OP_GE -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] >= r[k] ? 1.0 : 0.0;
                            }
                        }

                        case OP_LE -> {
                            final double[] r = executionStack[--sp];
                            final double[] l = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = l[k] <= r[k] ? 1.0 : 0.0;
                            }
                        }

                        case OP_VMA -> {
                            final double[] c = executionStack[--sp];
                            final double[] b = executionStack[--sp];
                            final double[] a = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = a[k] * b[k] + c[k];
                            }
                        }

                        case OP_IF -> {
                            final double[] falseVal = executionStack[--sp];
                            final double[] trueVal = executionStack[--sp];
                            final double[] cond = executionStack[--sp];
                            final double[] res = executionStack[sp++];
                            for (int k = 0; k < currentBlockSize; k++) {
                                res[k] = (cond[k] != 0.0) ? trueVal[k] : falseVal[k];
                            }
                        }

                        default ->
                            throw new UnsupportedOperationException("Unknown opcode: " + opcode);
                    }
                }

                // Step 3: Extract final calculation results from index 0 on stack directly to output destination
                System.arraycopy(executionStack[0], 0, output, blockStart, currentBlockSize);
            }
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
   
   /**
     * Optimized power utility for performance-critical math kernels.
     */
    public static double pow(double base, double exp) {
        // --- Integer Powers (-4 to 4) ---
        if (exp == 2.0) return base * base;
        if (exp == 3.0) return base * base * base;
        if (exp == 4.0) { double sq = base * base; return sq * sq; }
        if (exp == 1.0) return base;
        if (exp == 0.0) return 1.0;
        if (exp == -1.0) return 1.0 / base;
        if (exp == -2.0) return 1.0 / (base * base);
        if (exp == -3.0) { double sq = base * base; return 1.0 / (sq * base); }
        if (exp == -4.0) { double sq = base * base; return 1.0 / (sq * sq); }

        // --- Roots (Square & Cube) ---
        // Square Root
        if (exp == 0.5) return Math.sqrt(base);
        
        // Cube Root (handling direct 1/3 and common decimal approximation)
        if (Math.abs(exp - 0.3333333333333333) < 1e-15 || exp == (1.0 / 3.0)) {
            return Math.cbrt(base);
        }

        // Fallback to standard library
        return Math.pow(base, exp);
    }

     
    private boolean isMatrixKernel(String name, int arity) {
        return switch (name.toLowerCase()) {
            // BLAS
            case "matmul", "gemm", "mm" ->
                arity == 3;
            case "gemv", "matvec" ->
                arity == 2;
            case "dot", "inner", "axpy" ->
                arity == 3;
            case "transpose", "t" ->
                arity == 1;

            // Elementwise + activations
            case "add", "sub", "mul", "div", "pow" ->
                arity == 2;
            case "relu", "gelu", "sigmoid", "tanh", "silu", "swish", "mish" ->
                arity == 1;
            case "exp", "log", "sqrt", "rsqrt", "sin", "cos" ->
                arity == 1;
            case "leaky_relu", "elu", "clip", "clamp" ->
                arity == 2;
            case "abs", "neg", "square" ->
                arity == 1;

            // Reductions
            case "sum", "mean", "prod", "max", "min" ->
                arity == 1;
            case "softmax", "log_softmax" ->
                arity == 1;
            case "layer_norm" ->
                arity == 3;
            case "rms_norm" ->
                arity == 2;

            // Fused - big wins
            case "fma" ->
                arity == 3;
            case "bias_add" ->
                arity == 2;
            case "linear", "matmul_bias" ->
                arity == 3;
            case "matmul_bias_relu", "matmul_bias_gelu" ->
                arity == 3;
            case "matmul_bias_add" ->
                arity == 4;

            // Loss
            case "mse", "mae", "cross_entropy" ->
                arity == 2;

            default ->
                false;
        };
    }
}
