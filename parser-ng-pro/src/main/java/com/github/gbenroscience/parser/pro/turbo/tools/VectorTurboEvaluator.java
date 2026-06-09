package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.concurrent.Future;

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

    // Aggressive pre-sized caches
    private static final ThreadLocal<double[]> SCALAR_FRAME = ThreadLocal.withInitial(() -> new double[128]);
    private static final ThreadLocal<Future<?>[]> FUTURES_CACHE = ThreadLocal.withInitial(() -> new Future[128]);
    private static final ThreadLocal<MathExpression.EvalResult> RESULT_CACHE = ThreadLocal.withInitial(MathExpression.EvalResult::new);

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
    public SIMDCompositeExpression compile() throws Throwable {
        return new BatchedVectorCompositeExpression(compiledScalarHandle, opcodes, targetSlots,
                literalConstants, instructionCount);
    }

    public final class BatchedVectorCompositeExpression implements SIMDCompositeExpression {

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
         * Core Column-Major Vectorized Loop Processor..
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

        @Override
        public void applyBulk(double[][] variables, double[] output, java.util.concurrent.ExecutorService executor) {
            int dataSize = variables[0].length;
            int cores = Runtime.getRuntime().availableProcessors();
            int chunkSize = (dataSize + cores - 1) / cores; // Round up

            java.util.List<java.util.concurrent.Callable<Void>> tasks = new java.util.ArrayList<>();

            for (int i = 0; i < cores; i++) {
                final int start = i * chunkSize;
                final int end = Math.min(start + chunkSize, dataSize);

                if (start < end) {
                    tasks.add(() -> {
                        applyBulkInternal(variables, output, start, end - start);
                        return null;
                    });
                }
            }

            try {
                executor.invokeAll(tasks);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            int dataSize = variables[0].length;

            // Process in chunks of 'batchSize' to keep data hot in L1/L2 cache
            for (int start = 0; start < dataSize; start += batchSize) {
                int length = Math.min(batchSize, dataSize - start);
                applyBulkInternal(variables, output, start, length);
            }
        }

        @Override
        public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) {
            String kernelToRun = (op != null) ? op : (interceptedKernel != null ? interceptedKernel.getKernelName() : null);
            if (kernelToRun == null) {
                throw new UnsupportedOperationException("No kernel");
            }

            switch (kernelToRun.toLowerCase()) {
                case "matmul" ->
                    FlatMatrixF.matmul(inputs[0], inputs[1], output);
                case "add", "matadd", "add_mat" ->
                    FlatMatrixF.add(inputs[0], inputs[1], output);
                case "matmul_bias_gelu", "matmul_add_bias_gelu" ->
                    FlatMatrixF.matmulAddBiasGelu(inputs[0], inputs[1], inputs[2], output);
                case "matmul_add_sin" -> {
                    float alpha = inputs[2].get(0, 0);
                    FlatMatrixF.matmulAddSin(inputs[0], inputs[1], output, alpha);
                }
                case "softmax", "softmax_in_place" -> {
                    if (inputs[0] != output) {
                        System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                    }
                    output.softmaxRowsInPlace();
                }
                case "relu", "relu_in_place" -> {
                    if (inputs[0] != output) {
                        System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                    }
                    output.reluInPlace();
                }
                case "gelu", "gelu_in_place" -> {
                    if (inputs[0] != output) {
                        System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                    }
                    output.geluInPlace();
                }

                // === New Q8 + Attention kernels ===
                case "q8_quantize" -> {
                    // inputs[0] = src float, inputs[1] = scale float[1], output = dst byte wrapped as float
                    float[] src = inputs[0].data;
                    int srcOff = inputs[0].offset;
                    int len = inputs[0].rows * inputs[0].cols;
                    float scale = inputs[1].data[inputs[1].offset];
                    byte[] dst = output.asByteArray(); // FlatMatrix needs asByteArray() helper
                    KernelsInt8.quantize_f32_to_i8(src, srcOff, len, dst, output.offset, scale);
                }
                case "q8_dequant" -> {
                    // inputs[0] = src byte wrapped as float, inputs[1] = scale float[1], output = dst float
                    byte[] src = inputs[0].asByteArray();
                    int srcOff = inputs[0].offset;
                    int len = inputs[0].rows * inputs[0].cols;
                    float scale = inputs[1].data[inputs[1].offset];
                    float[] dst = output.data;
                    KernelsInt8.dequant_i8_to_f32(src, srcOff, len, dst, output.offset, scale);
                }
                case "q8_absmax_quant" -> {
                    // inputs[0] = src float, output[0] = dst byte, output[1] = scale float[1]
                    float[] src = inputs[0].data;
                    int srcOff = inputs[0].offset;
                    int len = inputs[0].rows * inputs[0].cols;
                    float amax = KernelsInt8.absmax(src, srcOff, len);
                    float scale = amax / 127.0f;
                    if (scale == 0.0f) {
                        scale = 1e-6f;
                    }
                    byte[] dst = output.asByteArray();
                    KernelsInt8.quantize_f32_to_i8(src, srcOff, len, dst, output.offset, scale);
                    // write scale to inputs[1] which caller must provide as writable
                    inputs[1].data[inputs[1].offset] = scale;
                }
                case "rope_split" -> {
                    // inputs[0] = buf [Q||K], inputs[1] = cos, inputs[2] = sin, inputs[3] = meta[pos, qHeads, kHeads, head_dim]
                    float[] buf = inputs[0].data;
                    float[] cos = inputs[1].data;
                    float[] sin = inputs[2].data;
                    int pos = (int) inputs[3].data[inputs[3].offset + 0];
                    int qHeads = (int) inputs[3].data[inputs[3].offset + 1];
                    int kHeads = (int) inputs[3].data[inputs[3].offset + 2];
                    int head_dim = (int) inputs[3].data[inputs[3].offset + 3];
                    int qOff = inputs[0].offset;
                    int kOff = qOff + qHeads * head_dim;
                    KernelsFloat.rope_inplace_split_ws(buf, qOff, qHeads, kOff, kHeads, head_dim, pos, cos, sin);
                    if (output != inputs[0]) {
                        System.arraycopy(buf, qOff, output.data, output.offset, (qHeads + kHeads) * head_dim);
                    }
                }
                case "accum_v" -> {
                    // output += inputs[0] * inputs[1].get(0,0)
                    float scale = (float) inputs[1].data[inputs[1].offset];
                    float[] src = inputs[0].data;
                    float[] dst = output.data;
                    int len = inputs[0].rows * inputs[0].cols;
                    KernelsFloat.accumulate_v_f32(dst, output.offset, src, inputs[0].offset, len, scale);
                }
                case "matmul_1xn_axpy" -> {
                    // output = inputs[0][1,K] @ inputs[1][K,N], inputs[0] is workspace slice
                    // inputs[2] = meta[K, N] as float[2]
                    int K = (int) inputs[2].data[inputs[2].offset + 0];
                    int N = (int) inputs[2].data[inputs[2].offset + 1];
                    KernelsFloat.matmul_f32_1xN_axpy(
                            inputs[0].data, inputs[1].data, output.data, output.offset,
                            K, N, inputs[0].offset
                    );
                }
                case "matmul_bias_relu" ->
                    FlatMatrixF.matmulAddBiasRelu(inputs[0], inputs[1], inputs[2], output);
                case "rms_norm" -> {
                    // inputs[0]=x, inputs[1]=weight, inputs[2]=eps[1], output=y
                    float eps = (float) inputs[2].data[inputs[2].offset];
                    FlatMatrixF.rmsNorm(inputs[0], inputs[1], output, eps);
                }
                case "layer_norm" -> {
                    // inputs[0]=x, inputs[1]=gamma, inputs[2]=beta, inputs[3]=eps[1]
                    float eps = (float) inputs[3].data[inputs[3].offset];
                    FlatMatrixF.layerNorm(inputs[0], inputs[1], inputs[2], output, eps);
                }
                case "matmul_bias_silu" ->
                    FlatMatrixF.matmulAddBiasSilu(inputs[0], inputs[1], inputs[2], output);
                case "swiglu" -> {
                    // inputs[0]=gate, inputs[1]=up, output=gate*SiLU(up)
                    FlatMatrixF.swiGLU(inputs[0], inputs[1], output);
                }

                case "mha_attention" -> {
                    // inputs: [Q, K, V, scale[1]], output=attn_out
                    float scale = (float) inputs[3].data[inputs[3].offset];
                    FlatMatrixF.mhaAttention(inputs[0], inputs[1], inputs[2], output, scale);
                }

                default ->
                    throw new UnsupportedOperationException("Unknown kernel: " + kernelToRun);
            }
        }

        @Override
        public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
            String kernelToRun = (op != null) ? op : (interceptedKernel != null ? interceptedKernel.getKernelName() : null);
            if (kernelToRun == null) {
                throw new UnsupportedOperationException("No kernel");
            }

        }
    }
/**
 * Turbo-flattens a jagged array into a destination array.
 * Reuses the provided 'flatBuffer' to ensure ZERO heap allocation.
 */
public static void flatten(double[][] jagged, double[] flatBuffer) {
    int offset = 0;
    for (int i = 0; i < jagged.length; i++) {
        int length = jagged[i].length;
        // Native-level memory copy - this is the fastest way to move data in JVM
        System.arraycopy(jagged[i], 0, flatBuffer, offset, length);
        offset += length;
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
        if (exp == 2.0) {
            return base * base;
        }
        if (exp == 3.0) {
            return base * base * base;
        }
        if (exp == 4.0) {
            double sq = base * base;
            return sq * sq;
        }
        if (exp == 1.0) {
            return base;
        }
        if (exp == 0.0) {
            return 1.0;
        }
        if (exp == -1.0) {
            return 1.0 / base;
        }
        if (exp == -2.0) {
            return 1.0 / (base * base);
        }
        if (exp == -3.0) {
            double sq = base * base;
            return 1.0 / (sq * base);
        }
        if (exp == -4.0) {
            double sq = base * base;
            return 1.0 / (sq * sq);
        }

        // --- Roots (Square & Cube) ---
        // Square Root
        if (exp == 0.5) {
            return Math.sqrt(base);
        }

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
