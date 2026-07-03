package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorizedCodyMath;
import java.util.function.DoubleUnaryOperator;
import jdk.incubator.vector.*;

/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 */
public class SIMDVectorTurboEvaluator extends VectorTurboEvaluator {

    public SIMDVectorTurboEvaluator(MathExpression me) throws Throwable {
        super(me);
    }

    @Override
    public BatchedVectorCompositeExpression compile() throws Throwable {
        return new SIMDVectorCompositeExpression();
    }

    public final class SIMDVectorCompositeExpression extends BatchedVectorCompositeExpression {

        private final ThreadLocal<double[]> SINGLE_SCALAR_BUFFER = ThreadLocal.withInitial(()
                -> new double[SIMDVectorTurboEvaluator.this.stackDepth]
        );
        private final ThreadLocal<double[]> MATRIX_FLATTEN_BUFFER = ThreadLocal.withInitial(()
                -> new double[0]
        );

        SIMDVectorCompositeExpression() {
            super(compiledScalarHandle, opcodes, targetSlots,
                    literalConstants, instructionCount, varCount);
        }

        @Override
        public MathExpression.EvalResult apply(double[] vars) {
            return null;
        }

        @Override
        public String checkErrorLogs() {
            return "";
        }

        @Override
        public TurboExpressionEvaluator getCompiler() {
            return SIMDVectorTurboEvaluator.this;
        }

        // --- 2D Matrix Channels ---
        private double[] getOrCreateFlattenBuffer(int totalSize) {
            double[] buf = MATRIX_FLATTEN_BUFFER.get();
            if (buf == null || buf.length < totalSize) {
                buf = new double[totalSize];
                MATRIX_FLATTEN_BUFFER.set(buf);
            }
            return buf;
        }

        @Override
        public void applyBulk(double[][] variables, double[] output) {
            int numSamples = variables[0].length;

            applyBulkInternal(variables, numSamples, output, 0, numSamples);
        }

        @Override
        public void applyBulkParallel(double[][] variables, double[] output) {
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;
            if (numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulk(variables, output);
                return;
            }

            dispatchToWorkerRing(variables, output, numSamples);
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            int numSamples = variables[0].length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(variables, numSamples, output, start, length);
            }
        }

        // --- 1D Flat Contiguous Frameworks ---
        @Override
        public void applyBulk(double[] flatVariables, double[] output) {
            applyBulkInternal(flatVariables, output.length, output, 0, output.length);
        }

        @Override
        public void applyBulkParallel(double[] flatVariables, double[] output) {
            int numSamples = output.length;
            if (numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, numSamples, output, 0, numSamples);
                return;
            }
            dispatchToWorkerRing(flatVariables, output, numSamples);
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize) {
            int numSamples = output.length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, numSamples, output, start, length);
            }
        }

        /**
         * Core column-major vectorized loop processor utilizing a flat memory
         * architecture. Bypasses pointer-chasing and allocation overhead
         * inherent to jagged multidimensional arrays (e.g., {@code double[][]})
         * to execute loops at maximum hardware throughput.
         * <p>
         * Based on the {@code tiledExecution} execution parameter, this method
         * delegates to either a cache-conscious tiledExecution implementation
         * or a direct flat streaming evaluation strategy.
         * </p>
         *
         * @param flatVariables a contiguous, single-dimensional array
         * containing concatenated variable tracks back-to-back (e.g.,
         * {@code [x1, x2...xn, y1, y2...yn, z1, z2...zn]})
         * @param dataSize the total number of elements per individual variable
         * row, used as the stride offset to hop between distinct variable
         * memory blocks
         * @param output the target destination array where the resulting
         * mathematical evaluations are written
         * @param startIdx the baseline index indicating where the batch
         * processing slice begins
         * @param length the absolute number of elements to process within this
         * batch window Tiles by default: If execution should be routed
         *
         * through a block memory pattern optimized for L1/L2 cache locality;
         * {@code false} for flat, non-tiledExecution bulk processing
         *
         * If you are processing a total dataset of 1,000,000 elements (dataSize
         * = 1000000), but a specific thread or tile is only processing a chunk
         * of 500 elements starting at element 200,000:
         *
         * startIdx = 200000
         *
         * length = 500
         *
         * It tells the engine: "Grab 500 elements starting at offset 200,000
         * from each variable segment in the input, compute them, and write them
         * into indices 200,000 through 200,499 of the output array."
         */
        private void applyBulkInternal(double[] flatVariables, int dataSize, double[] output, int startIdx, int length) {
            // Thread-local scratch space acquisition & sizing guard
            double[] scratch = FLAT_SCRATCH_STACK.get();
            final int requiredScratchSize = Math.min(BLOCK_SIZE, length);
            if (scratch == null || scratch.length < requiredScratchSize) {
                scratch = new double[requiredScratchSize];
                FLAT_SCRATCH_STACK.set(scratch);
            }

            // Cache-aligned Loop Tiling
            final int endIdx = startIdx + length;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                evaluateBlock(flatVariables, dataSize, output, blockStart, currentBlockSize, scratch);
            }
        }

        private void applyBulkInternal(double[][] variables, int dataSize, double[] output, int startIdx, int length) {

            // Thread-local scratch space acquisition & sizing guard
            double[] scratch = FLAT_SCRATCH_STACK.get();
            if (scratch == null || scratch.length < (SIMDVectorTurboEvaluator.this.stackDepth * BLOCK_SIZE)) {
                scratch = new double[SIMDVectorTurboEvaluator.this.stackDepth * BLOCK_SIZE];
                FLAT_SCRATCH_STACK.set(scratch);
            }

            // Cache-aligned Loop Tiling
            final int endIdx = startIdx + length;
            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                evaluateBlock(variables, output, blockStart, currentBlockSize, scratch);
            }
        }

        ////////////////////////////////////////////
        /**
         * Core interpretation stream. Leverages explicit AVX-bound Incubator
         * Vector API instructions where possible, falling back to clean
         * primitive loops for auto-vectorization across the tile window.
         */
        private void evaluateBlock(double[] flatVariables,
                int dataSize,
                double[] output,
                int blockStart,
                int currentBlockSize,
                double[] scratch) {

            final int n = currentBlockSize; // Local alias for loop bounds
            int sp = 0;

            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        final double val = literalConstants[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        DoubleVector valVec = DoubleVector.broadcast(SPECIES, val);
                        for (; k < upperBound; k += VLEN) {
                            valVec.intoArray(scratch, stackOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[stackOffset + k] = val;
                        }
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;
                        if (stackOffset + n > scratch.length) {
                            throw new IllegalStateException(
                                    String.format("Scratch buffer overflow! dataSize=%d, stackOffset=%d, n=%d, scratch.length=%d, BLOCK_SIZE=%d, sp=%d",
                                            dataSize, stackOffset, n, scratch.length, BLOCK_SIZE, sp));
                        }
                        final int flatOffset = (slotIdx * dataSize) + blockStart;
                        System.arraycopy(flatVariables, flatOffset, scratch, stackOffset, n);
                    }

                    // Binary Operators
                    case OP_ADD -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.add(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] + scratch[rOffset + k];
                        }
                    }

                    case OP_SUB -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.sub(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] - scratch[rOffset + k];
                        }
                    }

                    case OP_MUL -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.mul(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] * scratch[rOffset + k];
                        }
                    }

                    case OP_DIV -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.div(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] / scratch[rOffset + k];
                        }
                    }

                    case OP_POW -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = pow(scratch[lOffset + k], scratch[rOffset + k]);
                        }
                    }
                    case OP_SWIGLU_2 -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // Hoist constants outside the hot loop
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

                        // 1. Vectorized main loop
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector y = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);

                            // Calculate SwiGLU: (x * y) / (1.0 + exp(-x))
                            DoubleVector expNegX = TurboVectorMath.fastVectorExp(x.neg()); // x.neg() simply flips the sign bit (fast)
                            DoubleVector denom = expNegX.add(ONE);
                            DoubleVector result = x.mul(y).div(denom);     // Fused mul + hardware div 

                            result.intoArray(scratch, resOffset + k);
                        }

                        // 2. Scalar tail loop for any remaining elements
                        for (; k < n; k++) {
                            scratch[resOffset + k] = Maths.swiglu(scratch[lOffset + k], scratch[rOffset + k]);
                        }
                    }
                    case OP_GEGLU_2 -> {
                        sp -= 2; // Adjust stack pointer
                        final int lOffset = sp * BLOCK_SIZE;
                        final int rOffset = lOffset + BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);

                        final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

                        int k = 0;
                        for (; k < loopBound; k += SPECIES.length()) {
                            // Load two inputs simultaneously
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector y = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);

                            // GeLU(y) = 0.5 * y * (1 + erf(y / sqrt(2)))
                            // We reuse your high-precision vectorizedErf utility
                            DoubleVector erfVal = TurboVectorMath.vectorizedErf(y.mul(INV_SQRT_2));
                            DoubleVector geluY = y.mul(HALF).mul(erfVal.add(ONE));

                            // Result = x * GeLU(y)
                            DoubleVector result = x.mul(geluY);

                            // Store back to lOffset (simulating stack push)
                            result.intoArray(scratch, lOffset + k);
                        }

                        // Tail cleanup
                        for (; k < n; k++) {
                            scratch[lOffset + k] = Maths.geglu(scratch[lOffset + k], scratch[rOffset + k]);
                        }

                        sp++; // Finalize stack pointer adjustment
                    }
                    case OP_REM -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] % scratch[rOffset + k];
                        }
                    }

                    // Base Unary Operations (In-place scalar evaluation)
                    case OP_SIN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.sin(scratch[srcOffset + k]);
                        }
                    }

                    case OP_COS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.cos(scratch[srcOffset + k]);
                        }
                    }

                    case OP_TAN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.tan(scratch[srcOffset + k]);
                        }
                    }

                    case OP_SINH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.sinh(scratch[srcOffset + k]);
                        }
                    }

                    case OP_COSH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.cosh(scratch[srcOffset + k]);
                        }
                    }

                    case OP_TANH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.tanh(scratch[srcOffset + k]);
                        }
                    }

                    case OP_ABS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.abs(scratch[srcOffset + k]);
                        }
                    }

                    case OP_EXP -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.exp(scratch[srcOffset + k]);
                        }
                    }

                    case OP_SQRT -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.sqrt(scratch[srcOffset + k]);
                        }
                    }
                    case OP_SWIGLU -> {
                        final int base = (sp - 1) * BLOCK_SIZE;

                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // Hoist constant outside the hot loop
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

                        // 1. Vectorized main loop (In-place modification)
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, base + k);

                            // Calculate 1-arg SwiGLU (Swish/SiLU): x / (1.0 + exp(-x))
                            DoubleVector expNegX = TurboVectorMath.fastVectorExp(x.neg());
                            DoubleVector denom = expNegX.add(ONE);
                            DoubleVector result = x.div(denom);

                            // Write directly back to the same offset
                            result.intoArray(scratch, base + k);
                        }

                        // 2. Scalar tail loop for remaining elements
                        for (; k < n; k++) {
                            scratch[base + k] = Maths.swiglu(scratch[base + k]);
                        }
                    }
                    case OP_GELU, OP_GEGLU, OP_GELU_FAST -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // Common constants
                        final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector TWO = DoubleVector.broadcast(SPECIES, 2.0);

                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, base + k);
                            DoubleVector result;

                            if (opcode == OP_GELU) {
                                // GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
                                final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);
                                // Note: Reuse your vectorizedErf here
                                result = x.mul(HALF).mul(TurboVectorMath.vectorizedErf(x.mul(INV_SQRT_2)).add(ONE));
                            } else if (opcode == OP_GELU_FAST) {
                                // FAST GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                                final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
                                final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);

                                DoubleVector x3 = x.mul(x).mul(x);
                                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);

                                // Using your optimized fastVectorExp for tanh calculation
                                DoubleVector exp2z = TurboVectorMath.fastVectorExp(z.mul(TWO));
                                DoubleVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                                result = x.mul(HALF).mul(tanhZ.add(ONE));
                            } else { // OP_GEGLU
                                // Placeholder for GEGLU logic (Gated Error Linear Unit)
                                result = x;
                            }

                            result.intoArray(scratch, base + k);
                        }

                        // Scalar fallback
                        for (; k < n; k++) {
                            if (opcode == OP_GELU) {
                                scratch[base + k] = Maths.gelu(scratch[base + k]);
                            } else if (opcode == OP_GELU_FAST) {
                                scratch[base + k] = Maths.fastGelu(scratch[base + k]);
                            } else {
                                scratch[base + k] = Maths.geglu(scratch[base + k]);
                            }
                        }
                    }
                    case OP_ERF -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // 1. Vectorized main loop
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, base + k);

                            // Use your optimized piecewise vectorizedErf
                            DoubleVector result = TurboVectorMath.vectorizedErf(x);

                            result.intoArray(scratch, base + k);
                        }

                        // 2. Scalar tail loop for remaining elements
                        for (; k < n; k++) {
                            scratch[base + k] = Maths.erf(scratch[base + k]);
                        }
                    }
                    case OP_CBRT -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.cbrt(scratch[srcOffset + k]);
                        }
                    }

                    case OP_LOG -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.log(scratch[srcOffset + k]);
                        }
                    }

                    case OP_LOG10 -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.log10(scratch[srcOffset + k]);
                        }
                    }

                    // --- Inverse Radians (Routed to MethodSack for Alias Mapping) ---
                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        MethodSack.asinRad((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        MethodSack.acosRad((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        MethodSack.atanRad((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    // --- DEGREE / GRADIAN TRIG VARIANTS ---
                    case OP_SIN_DEG ->
                        MethodSack.sinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_DEG ->
                        MethodSack.cosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_DEG ->
                        MethodSack.tanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SIN_GRAD ->
                        MethodSack.sinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_GRAD ->
                        MethodSack.cosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_GRAD ->
                        MethodSack.tanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE DEGREE / GRADIAN VARIANTS ---
                    case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG ->
                        MethodSack.asinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG ->
                        MethodSack.acosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG ->
                        MethodSack.atanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD ->
                        MethodSack.asinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD ->
                        MethodSack.acosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD ->
                        MethodSack.atanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- RECIPROCAL TRIG (SEC, CSC, COT) VARIANTS ---
                    case OP_SEC ->
                        MethodSack.sec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_DEG ->
                        MethodSack.secDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_GRAD ->
                        MethodSack.secGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC ->
                        MethodSack.csc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_DEG ->
                        MethodSack.cscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_GRAD ->
                        MethodSack.cscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT ->
                        MethodSack.cot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_DEG ->
                        MethodSack.cotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_GRAD ->
                        MethodSack.cotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE RECIPROCAL TRIG VARIANTS ---
                    case OP_ARC_SEC, OP_ARC_SEC_ALT ->
                        MethodSack.asec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG ->
                        MethodSack.asecDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD ->
                        MethodSack.asecGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC, OP_ARC_COSEC_ALT ->
                        MethodSack.acsc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG ->
                        MethodSack.acscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD ->
                        MethodSack.acscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT, OP_ARC_COT_ALT ->
                        MethodSack.acot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG ->
                        MethodSack.acotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD ->
                        MethodSack.acotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- HYPERBOLIC INVERSES ---
                    case OP_ASINH, OP_ASINH_ALT ->
                        MethodSack.asinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOSH, OP_ACOSH_ALT ->
                        MethodSack.acosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATANH, OP_ATANH_ALT ->
                        MethodSack.atanh((sp - 1) * BLOCK_SIZE, n, scratch);

                    // Conditional Comparisons
                    case OP_GT -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] > scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_LT -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] < scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_EQ -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] == scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_NE -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] != scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_GE -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] >= scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_LE -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] <= scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    // Ternary Operations (Manual FMA & Logical Selection)
                    case OP_VMA -> {
                        final int cOffset = (--sp) * BLOCK_SIZE;
                        final int bOffset = (--sp) * BLOCK_SIZE;
                        final int aOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[aOffset + k] * scratch[bOffset + k] + scratch[cOffset + k];
                        }
                    }

                    case OP_IF -> {
                        final int falseOffset = (--sp) * BLOCK_SIZE;
                        final int trueOffset = (--sp) * BLOCK_SIZE;
                        final int condOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = (scratch[condOffset + k] != 0.0) ? scratch[trueOffset + k] : scratch[falseOffset + k];
                        }
                    }

                    default ->
                        throw new UnsupportedOperationException("Unknown opcode: " + opcode);
                }
            }

            // Flush the final computation results out to the destination block segment
            System.arraycopy(scratch, 0, output, blockStart, n);
        }

        private void evaluateBlock(double[][] _2DVariables,
                double[] output,
                int blockStart,
                int currentBlockSize,
                double[] scratch) {

            final int n = currentBlockSize; // Local alias for loop bounds
            int sp = 0;

            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        final double val = literalConstants[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        DoubleVector valVec = DoubleVector.broadcast(SPECIES, val);
                        for (; k < upperBound; k += VLEN) {
                            valVec.intoArray(scratch, stackOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[stackOffset + k] = val;
                        }
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;
                        /*if (stackOffset + n > scratch.length) {
                            throw new IllegalStateException(
                            String.format("Scratch buffer overflow! dataSize=%d, stackOffset=%d, n=%d, scratch.length=%d, BLOCK_SIZE=%d, sp=%d",
                                            dataSize, stackOffset, n, scratch.length, BLOCK_SIZE, sp));
                        }
                        //final int flatOffset = (slotIdx * dataSize) + blockStart;
                        //System.arraycopy(_2DVariables, flatOffset, scratch, stackOffset, n);
                         */
 /*Works
                        for (int k = 0; k < n; k++) {
                            scratch[stackOffset + k] = _2DVariables[slotIdx][blockStart + k];
                        }*/
                        System.arraycopy(_2DVariables[slotIdx], blockStart, scratch, stackOffset, n);
                    }

                    // Binary Operators
                    case OP_ADD -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.add(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] + scratch[rOffset + k];
                        }
                    }

                    case OP_SUB -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.sub(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] - scratch[rOffset + k];
                        }
                    }

                    case OP_MUL -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.mul(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] * scratch[rOffset + k];
                        }
                    }

                    case OP_DIV -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        int k = 0;
                        int upperBound = SPECIES.loopBound(n);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.div(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] / scratch[rOffset + k];
                        }
                    }

                    case OP_POW -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = pow(scratch[lOffset + k], scratch[rOffset + k]);
                        }
                    }
                    case OP_SWIGLU_2 -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // Hoist constants outside the hot loop
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

                        // 1. Vectorized main loop
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector y = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);

                            // Calculate SwiGLU: (x * y) / (1.0 + exp(-x))
                            DoubleVector expNegX = TurboVectorMath.fastVectorExp(x.neg()); // x.neg() simply flips the sign bit (fast)
                            DoubleVector denom = expNegX.add(ONE);
                            DoubleVector result = x.mul(y).div(denom);     // Fused mul + hardware div 

                            result.intoArray(scratch, resOffset + k);
                        }

                        // 2. Scalar tail loop for any remaining elements
                        for (; k < n; k++) {
                            scratch[resOffset + k] = Maths.swiglu(scratch[lOffset + k], scratch[rOffset + k]);
                        }
                    }
                    case OP_GEGLU_2 -> {
                        sp -= 2; // Adjust stack pointer
                        final int lOffset = sp * BLOCK_SIZE;
                        final int rOffset = lOffset + BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);

                        final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);

                        int k = 0;
                        for (; k < loopBound; k += SPECIES.length()) {
                            // Load two inputs simultaneously
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector y = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);

                            // GeLU(y) = 0.5 * y * (1 + erf(y / sqrt(2)))
                            // We reuse your high-precision vectorizedErf utility
                            DoubleVector erfVal = TurboVectorMath.vectorizedErf(y.mul(INV_SQRT_2));
                            DoubleVector geluY = y.mul(HALF).mul(erfVal.add(ONE));

                            // Result = x * GeLU(y)
                            DoubleVector result = x.mul(geluY);

                            // Store back to lOffset (simulating stack push)
                            result.intoArray(scratch, lOffset + k);
                        }

                        // Tail cleanup
                        for (; k < n; k++) {
                            scratch[lOffset + k] = Maths.geglu(scratch[lOffset + k], scratch[rOffset + k]);
                        }

                        sp++; // Finalize stack pointer adjustment
                    }
                    case OP_REM -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] % scratch[rOffset + k];
                        }
                    }

                    // Base Unary Operations (In-place scalar evaluation)
                    case OP_SIN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.sin(scratch[srcOffset + k]);
                        }
                    }

                    case OP_COS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.cos(scratch[srcOffset + k]);
                        }
                    }

                    case OP_TAN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.tan(scratch[srcOffset + k]);
                        }
                    }

                    case OP_SINH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.sinh(scratch[srcOffset + k]);
                        }
                    }

                    case OP_COSH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.cosh(scratch[srcOffset + k]);
                        }
                    }

                    case OP_TANH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.tanh(scratch[srcOffset + k]);
                        }
                    }

                    case OP_ABS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.abs(scratch[srcOffset + k]);
                        }
                    }

                    /*  case OP_EXP -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;

                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.exp(scratch[srcOffset + k]);
                        }
                    }*/
                    case OP_EXP -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;

                        int k = 0;
                        final int upperBound = SPECIES.loopBound(n);

                        // Turbo SIMD Lane
                        for (; k < upperBound; k += SPECIES.length()) {
                            final int targetIdx = srcOffset + k;
                            DoubleVector.fromArray(SPECIES, scratch, targetIdx)
                                    .lanewise(VectorOperators.EXP)
                                    .intoArray(scratch, targetIdx);
                        }

                        // Scalar Tail Cleanup
                        for (; k < n; k++) {
                            final int targetIdx = srcOffset + k;
                            scratch[targetIdx] = Math.exp(scratch[targetIdx]);
                        }
                    }

                    case OP_SQRT -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        VectorTranscendentals.evaluateNative(
                                scratch, // src array
                                srcOffset, // srcOffset
                                scratch, // dest array (in-place)
                                srcOffset, // destOffset
                                n, // element count
                                VectorOperators.SQRT // unary operator
                        );
                    }

                    case OP_CBRT -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        VectorTranscendentals.evaluateNative(
                                scratch, // src array
                                srcOffset, // srcOffset
                                scratch, // dest array (in-place)
                                srcOffset, // destOffset
                                n, // element count
                                VectorOperators.CBRT // unary operator
                        );
                    }

                    case OP_SWIGLU -> {
                        final int base = (sp - 1) * BLOCK_SIZE;

                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // Hoist constant outside the hot loop
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

                        // 1. Vectorized main loop (In-place modification)
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, base + k);

                            // Calculate 1-arg SwiGLU (Swish/SiLU): x / (1.0 + exp(-x))
                            DoubleVector expNegX = TurboVectorMath.fastVectorExp(x.neg());
                            DoubleVector denom = expNegX.add(ONE);
                            DoubleVector result = x.div(denom);

                            // Write directly back to the same offset
                            result.intoArray(scratch, base + k);
                        }

                        // 2. Scalar tail loop for remaining elements
                        for (; k < n; k++) {
                            scratch[base + k] = Maths.swiglu(scratch[base + k]);
                        }
                    }
                    case OP_GELU, OP_GEGLU, OP_GELU_FAST -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // Common constants
                        final DoubleVector HALF = DoubleVector.broadcast(SPECIES, 0.5);
                        final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector TWO = DoubleVector.broadcast(SPECIES, 2.0);

                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, base + k);
                            DoubleVector result;

                            if (opcode == OP_GELU) {
                                // GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
                                final DoubleVector INV_SQRT_2 = DoubleVector.broadcast(SPECIES, 0.7071067811865476);
                                // Note: Reuse your vectorizedErf here
                                result = x.mul(HALF).mul(TurboVectorMath.vectorizedErf(x.mul(INV_SQRT_2)).add(ONE));
                            } else if (opcode == OP_GELU_FAST) {
                                // FAST GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                                final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
                                final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);

                                DoubleVector x3 = x.mul(x).mul(x);
                                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);

                                // Using your optimized fastVectorExp for tanh calculation
                                DoubleVector exp2z = TurboVectorMath.fastVectorExp(z.mul(TWO));
                                DoubleVector tanhZ = exp2z.sub(ONE).div(exp2z.add(ONE));
                                result = x.mul(HALF).mul(tanhZ.add(ONE));
                            } else { // OP_GEGLU
                                // Placeholder for GEGLU logic (Gated Error Linear Unit)
                                result = x;
                            }

                            result.intoArray(scratch, base + k);
                        }

                        // Scalar fallback
                        for (; k < n; k++) {
                            if (opcode == OP_GELU) {
                                scratch[base + k] = Maths.gelu(scratch[base + k]);
                            } else if (opcode == OP_GELU_FAST) {
                                scratch[base + k] = Maths.fastGelu(scratch[base + k]);
                            } else {
                                scratch[base + k] = Maths.geglu(scratch[base + k]);
                            }
                        }
                    }
                    case OP_ERF -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        final int loopBound = SPECIES.loopBound(n);
                        int k = 0;

                        // 1. Vectorized main loop
                        for (; k < loopBound; k += SPECIES.length()) {
                            DoubleVector x = DoubleVector.fromArray(SPECIES, scratch, base + k);

                            // Use your optimized piecewise vectorizedErf
                            DoubleVector result = TurboVectorMath.vectorizedErf(x);

                            result.intoArray(scratch, base + k);
                        }
                        // 2. Scalar tail loop for remaining elements
                        for (; k < n; k++) {
                            scratch[base + k] = Maths.erf(scratch[base + k]);
                        }
                    }

                    case OP_LOG -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.log(scratch[srcOffset + k]);
                        }
                    }

                    case OP_LOG10 -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.log10(scratch[srcOffset + k]);
                        }
                    }

                    // --- Inverse Radians (Routed to MethodSack for Alias Mapping) ---
                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        MethodSack.asinRad((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        MethodSack.acosRad((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        MethodSack.atanRad((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    // --- DEGREE / GRADIAN TRIG VARIANTS ---
                    case OP_SIN_DEG ->
                        MethodSack.sinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_DEG ->
                        MethodSack.cosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_DEG ->
                        MethodSack.tanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SIN_GRAD ->
                        MethodSack.sinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_GRAD ->
                        MethodSack.cosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_GRAD ->
                        MethodSack.tanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE DEGREE / GRADIAN VARIANTS ---
                    case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG ->
                        MethodSack.asinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG ->
                        MethodSack.acosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG ->
                        MethodSack.atanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD ->
                        MethodSack.asinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD ->
                        MethodSack.acosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD ->
                        MethodSack.atanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- RECIPROCAL TRIG (SEC, CSC, COT) VARIANTS ---
                    case OP_SEC ->
                        MethodSack.sec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_DEG ->
                        MethodSack.secDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_GRAD ->
                        MethodSack.secGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC ->
                        MethodSack.csc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_DEG ->
                        MethodSack.cscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_GRAD ->
                        MethodSack.cscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT ->
                        MethodSack.cot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_DEG ->
                        MethodSack.cotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_GRAD ->
                        MethodSack.cotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE RECIPROCAL TRIG VARIANTS ---
                    case OP_ARC_SEC, OP_ARC_SEC_ALT ->
                        MethodSack.asec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG ->
                        MethodSack.asecDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD ->
                        MethodSack.asecGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC, OP_ARC_COSEC_ALT ->
                        MethodSack.acsc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG ->
                        MethodSack.acscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD ->
                        MethodSack.acscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT, OP_ARC_COT_ALT ->
                        MethodSack.acot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG ->
                        MethodSack.acotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD ->
                        MethodSack.acotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- HYPERBOLIC INVERSES ---
                    case OP_ASINH, OP_ASINH_ALT ->
                        MethodSack.asinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOSH, OP_ACOSH_ALT ->
                        MethodSack.acosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATANH, OP_ATANH_ALT ->
                        MethodSack.atanh((sp - 1) * BLOCK_SIZE, n, scratch);

                    // Conditional Comparisons
                    case OP_GT -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] > scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_LT -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] < scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_EQ -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] == scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_NE -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] != scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_GE -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] >= scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_LE -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] <= scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    // Ternary Operations (Manual FMA & Logical Selection)
                    case OP_VMA -> {
                        final int cOffset = (--sp) * BLOCK_SIZE;
                        final int bOffset = (--sp) * BLOCK_SIZE;
                        final int aOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[aOffset + k] * scratch[bOffset + k] + scratch[cOffset + k];
                        }
                    }

                    case OP_IF -> {
                        final int falseOffset = (--sp) * BLOCK_SIZE;
                        final int trueOffset = (--sp) * BLOCK_SIZE;
                        final int condOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = (scratch[condOffset + k] != 0.0) ? scratch[trueOffset + k] : scratch[falseOffset + k];
                        }
                    }

                    default ->
                        throw new UnsupportedOperationException("Unknown opcode: " + opcode);
                }
            }

            // Flush the final computation results out to the destination block segment
            System.arraycopy(scratch, 0, output, blockStart, n);
        }
    }

    public static final class VectorTranscendentals {

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

        public static void evaluateNative(double[] src, int srcOffset, double[] dest, int destOffset, int n, VectorOperators.Unary op) {
            int vl = SPECIES.length();
            int limit = SPECIES.loopBound(n);
            int i = 0;

            // Vector Loop
            for (; i < limit; i += vl) {
                DoubleVector va  = DoubleVector.fromArray(SPECIES, src, srcOffset + i);
                va.lanewise(op).intoArray(dest, destOffset + i);
            }

            // Clean Masked Tail
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector va  = DoubleVector.fromArray(SPECIES, src, srcOffset + i, mask);
                va.lanewise(op).intoArray(dest, destOffset + i, mask);
            }
        }
    }

    /**
     * High-performance vectorized math utilities for bulk array operations.
     * Optimized for a math expression parser.
     *
     * * @author GBEMIRO (refined)
     */
    public static final class TurboVectorMath {

        private TurboVectorMath() {
        }

        private static final double DEG_TO_RAD = Math.PI / 180.0;
        private static final double RAD_TO_DEG = 180.0 / Math.PI;
        private static final double GRAD_TO_RAD = Math.PI / 200.0;
        private static final double RAD_TO_GRAD = 200.0 / Math.PI;

        // ===================== Core Orchestration Helpers =====================
        private static void process(int base, int n, double[] array,
                VectorOperators.Unary op,
                DoubleUnaryOperator scalarOp) {
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, array, base + i);
                v.lanewise(op).intoArray(array, base + i);
            }
            for (; i < n; i++) {
                array[base + i] = scalarOp.applyAsDouble(array[base + i]);
            }
        }

        /**
         * High-performance vectorized exp() using Cody-Waite range reduction +
         * 6th-degree minimax polynomial + direct bit manipulation for 2^k.
         */
        /**
         * High-performance vectorized exp() using Cody-Waite range reduction +
         * 6th-degree minimax polynomial + direct bit manipulation for 2^k.
         */
        /**
         * High-performance vectorized exp() using magic-number rounding +
         * 6th-degree minimax polynomial + fast bit manipulation for 2^k.
         */
        private static DoubleVector fastVectorExp(DoubleVector x) {
            // 1. Clamp
            x = x.lanewise(VectorOperators.MAX, -40.0)
                    .lanewise(VectorOperators.MIN, 40.0);

            // 2. Range reduction
            DoubleVector invLn2 = DoubleVector.broadcast(SPECIES, 1.4426950408889634074);
            DoubleVector ln2Hi = DoubleVector.broadcast(SPECIES, -0.6931471805599453);
            DoubleVector ln2Lo = DoubleVector.broadcast(SPECIES, -2.8235290563031574E-13);

            // Magic number rounding to nearest integer
            DoubleVector magic = DoubleVector.broadcast(SPECIES, 4503599627370496.0); // 2^52
            DoubleVector k = x.mul(invLn2).add(magic).sub(magic);

            DoubleVector r = x.add(k.mul(ln2Hi)).add(k.mul(ln2Lo));

            // 3. 6th-degree minimax polynomial (optimized coefficients)
            DoubleVector p = r.mul(0.001398199650)
                    .add(0.0088632903)
                    .mul(r).add(0.04166666666)
                    .mul(r).add(0.16666666666)
                    .mul(r).add(0.5)
                    .mul(r).add(1.0)
                    .mul(r).add(1.0);

            // 4. Fast 2^k via bit manipulation
            LongVector kLong = (LongVector) k.convert(VectorOperators.D2L, 0);
            LongVector exponent = kLong.add(1023).lanewise(VectorOperators.LSHL, 52);
            DoubleVector twoK = (DoubleVector) exponent.convert(VectorOperators.REINTERPRET_L2D, 0);

            return p.mul(twoK);
        }

        // Define these at the class level
        private static final double THRESHOLD_LOW = 0.46875;
        private static final double THRESHOLD_HIGH = 4.0;
        private static final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0);

        private static DoubleVector vectorizedErf(DoubleVector x) {
            DoubleVector absX = x.abs();

            // 1. Masks for intervals
            VectorMask<Double> maskLow = absX.compare(VectorOperators.LE, THRESHOLD_LOW);
            VectorMask<Double> maskHigh = absX.compare(VectorOperators.LE, THRESHOLD_HIGH);

            // 2. Interval 1 (Low)
            DoubleVector xSq = absX.mul(absX);
            DoubleVector p = xSq.mul(0.260194122534674).add(30.59022585250011).mul(xSq)
                    .add(573.9507736045833).mul(xSq).add(2801.752391065013).mul(xSq).add(3204.677458505002);
            DoubleVector q = xSq.add(159.0884090976454).mul(xSq).add(1422.080683811422).mul(xSq)
                    .add(4423.613442045816).mul(xSq).add(3204.677458506958);

            // Result for low interval with sign preserved
            DoubleVector resLow = x.mul(p.div(q));

            // 3. Intervals 2 & 3 (Medium/Large)
            // We pass your required signatures: (vX, vAbsX, mask)
            DoubleVector resMed = VectorizedCodyMath.evaluateMediumVector(x, absX, maskHigh);
            DoubleVector resHigh = VectorizedCodyMath.evaluateLargeVector(x, absX, maskHigh.not());
            DoubleVector erfcVal = resMed.blend(resHigh, maskHigh.not());

            // erfc(absX) -> erf(absX) = 1 - erfc(absX)
            DoubleVector resMidHigh = ONE.sub(erfcVal);

            // 4. Apply sign ONCE at the end
            VectorMask<Double> isNegative = x.compare(VectorOperators.LT, 0.0);
            resMidHigh = resMidHigh.blend(resMidHigh.neg(), isNegative);

            // 5. Final Blend
            return resMidHigh.blend(resLow, maskLow);
        }

        private static void scaleInputAndApply(int base, int n, double[] array, double scale,
                VectorOperators.Unary op,
                DoubleUnaryOperator scalarOp) {
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, scale);
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, array, base + i).mul(vScale);
                v.lanewise(op).intoArray(array, base + i);
            }
            for (; i < n; i++) {
                array[base + i] = scalarOp.applyAsDouble(array[base + i] * scale);
            }
        }

        private static void applyAndScaleOutput(int base, int n, double[] array,
                VectorOperators.Unary op,
                DoubleUnaryOperator scalarOp,
                double scale) {
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, scale);
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, array, base + i);
                v.lanewise(op).mul(vScale).intoArray(array, base + i);
            }
            for (; i < n; i++) {
                array[base + i] = scalarOp.applyAsDouble(array[base + i]) * scale;
            }
        }

        // ===================== Conditional Branching =====================
        public static void if3(int base, int tileN, double[] s, int block) {
            final int cond = base + block;
            final int trueVal = base + 2 * block;
            final int falseVal = base + 3 * block;
            final int res = base;

            int bound = SPECIES.loopBound(tileN);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0)
                        .and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i);
            }
            for (; i < tileN; i++) {
                double c = s[cond + i];
                s[res + i] = (c != 0.0 && !Double.isNaN(c)) ? s[trueVal + i] : s[falseVal + i];
            }
        }

        // ===================== Trigonometric =====================
        public static void sin(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.SIN, Math::sin);
        }

        public static void cos(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.COS, Math::cos);
        }

        public static void tan(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.TAN, Math::tan);
        }

        public static void sinDeg(int base, int n, double[] s) {
            scaleInputAndApply(base, n, s, DEG_TO_RAD, VectorOperators.SIN, Math::sin);
        }

        public static void cosDeg(int base, int n, double[] s) {
            scaleInputAndApply(base, n, s, DEG_TO_RAD, VectorOperators.COS, Math::cos);
        }

        public static void tanDeg(int base, int n, double[] s) {
            scaleInputAndApply(base, n, s, DEG_TO_RAD, VectorOperators.TAN, Math::tan);
        }

        public static void sinGrad(int base, int n, double[] s) {
            scaleInputAndApply(base, n, s, GRAD_TO_RAD, VectorOperators.SIN, Math::sin);
        }

        public static void cosGrad(int base, int n, double[] s) {
            scaleInputAndApply(base, n, s, GRAD_TO_RAD, VectorOperators.COS, Math::cos);
        }

        public static void tanGrad(int base, int n, double[] s) {
            scaleInputAndApply(base, n, s, GRAD_TO_RAD, VectorOperators.TAN, Math::tan);
        }

        // ===================== Inverse Trigonometric =====================
        public static void asinRad(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.ASIN, Math::asin);
        }

        public static void acosRad(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.ACOS, Math::acos);
        }

        public static void atanRad(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.ATAN, Math::atan);
        }

        public static void asinDeg(int base, int n, double[] s) {
            applyAndScaleOutput(base, n, s, VectorOperators.ASIN, Math::asin, RAD_TO_DEG);
        }

        public static void acosDeg(int base, int n, double[] s) {
            applyAndScaleOutput(base, n, s, VectorOperators.ACOS, Math::acos, RAD_TO_DEG);
        }

        public static void atanDeg(int base, int n, double[] s) {
            applyAndScaleOutput(base, n, s, VectorOperators.ATAN, Math::atan, RAD_TO_DEG);
        }

        public static void asinGrad(int base, int n, double[] s) {
            applyAndScaleOutput(base, n, s, VectorOperators.ASIN, Math::asin, RAD_TO_GRAD);
        }

        public static void acosGrad(int base, int n, double[] s) {
            applyAndScaleOutput(base, n, s, VectorOperators.ACOS, Math::acos, RAD_TO_GRAD);
        }

        public static void atanGrad(int base, int n, double[] s) {
            applyAndScaleOutput(base, n, s, VectorOperators.ATAN, Math::atan, RAD_TO_GRAD);
        }

        // ===================== Reciprocal Trig =====================
        public static void sec(int base, int n, double[] s) {
            processReciprocal(base, n, s, VectorOperators.COS, Math::cos);
        }

        public static void csc(int base, int n, double[] s) {
            processReciprocal(base, n, s, VectorOperators.SIN, Math::sin);
        }

        public static void cot(int base, int n, double[] s) {
            processReciprocal(base, n, s, VectorOperators.TAN, Math::tan);
        }

        public static void secDeg(int base, int n, double[] s) {
            scaleInputReciprocal(base, n, s, DEG_TO_RAD, VectorOperators.COS, Math::cos);
        }

        public static void cscDeg(int base, int n, double[] s) {
            scaleInputReciprocal(base, n, s, DEG_TO_RAD, VectorOperators.SIN, Math::sin);
        }

        public static void cotDeg(int base, int n, double[] s) {
            scaleInputReciprocal(base, n, s, DEG_TO_RAD, VectorOperators.TAN, Math::tan);
        }

        // ===================== Inverse Reciprocal =====================
        public static void asec(int base, int n, double[] s) {
            processInverseReciprocal(base, n, s, VectorOperators.ACOS, Math::acos);
        }

        public static void acsc(int base, int n, double[] s) {
            processInverseReciprocal(base, n, s, VectorOperators.ASIN, Math::asin);
        }

        public static void acot(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector halfPi = DoubleVector.broadcast(SPECIES, Math.PI / 2.0);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                halfPi.sub(v.lanewise(VectorOperators.ATAN)).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.PI / 2.0 - Math.atan(s[base + i]);
            }
        }

        public static void asecDeg(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, RAD_TO_DEG);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                one.div(v).lanewise(VectorOperators.ACOS).mul(vScale).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]) * RAD_TO_DEG;
            }
        }

        public static void acscDeg(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, RAD_TO_DEG);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                one.div(v).lanewise(VectorOperators.ASIN).mul(vScale).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]) * RAD_TO_DEG;
            }
        }

        public static void asecGrad(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, RAD_TO_GRAD);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                one.div(v).lanewise(VectorOperators.ACOS).mul(vScale).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acscGrad(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, RAD_TO_GRAD);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                one.div(v).lanewise(VectorOperators.ASIN).mul(vScale).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Hyperbolic Operations =====================
        public static void sinh(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector expX = x.lanewise(VectorOperators.EXP);
                DoubleVector expMX = x.neg().lanewise(VectorOperators.EXP);
                expX.sub(expMX).mul(half).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sinh(s[base + i]);
            }
        }

        public static void cosh(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector expX = x.lanewise(VectorOperators.EXP);
                DoubleVector expMX = x.neg().lanewise(VectorOperators.EXP);
                expX.add(expMX).mul(half).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cosh(s[base + i]);
            }
        }

        public static void tanh(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector exp2x = x.add(x).lanewise(VectorOperators.EXP);
                exp2x.sub(one).div(exp2x.add(one)).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tanh(s[base + i]);
            }
        }

        // ===================== Reciprocal Trig (Gradian) =====================
        public static void secGrad(int base, int n, double[] s) {
            scaleInputReciprocal(base, n, s, GRAD_TO_RAD, VectorOperators.COS, Math::cos);
        }

        public static void cscGrad(int base, int n, double[] s) {
            scaleInputReciprocal(base, n, s, GRAD_TO_RAD, VectorOperators.SIN, Math::sin);
        }

        public static void cotGrad(int base, int n, double[] s) {
            scaleInputReciprocal(base, n, s, GRAD_TO_RAD, VectorOperators.TAN, Math::tan);
        }

        // ===================== Inverse Reciprocal (Deg/Grad) =====================
        public static void acotDeg(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector halfPi = DoubleVector.broadcast(SPECIES, Math.PI / 2.0);
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, RAD_TO_DEG);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                halfPi.sub(v.lanewise(VectorOperators.ATAN)).mul(vScale).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (Math.PI / 2.0 - Math.atan(s[base + i])) * RAD_TO_DEG;
            }
        }

        public static void acotGrad(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector halfPi = DoubleVector.broadcast(SPECIES, Math.PI / 2.0);
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, RAD_TO_GRAD);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                halfPi.sub(v.lanewise(VectorOperators.ATAN)).mul(vScale).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = (Math.PI / 2.0 - Math.atan(s[base + i])) * RAD_TO_GRAD;
            }
        }

        // ===================== Inverse Hyperbolic =====================
        public static void asinh(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + i);
                // ln(x + sqrt(x^2 + 1))
                DoubleVector expr = x.add(x.mul(x).add(one).lanewise(VectorOperators.SQRT));
                expr.lanewise(VectorOperators.LOG).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = Math.log(x + Math.sqrt(x * x + 1.0));
            }
        }

        public static void acosh(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + i);
                // ln(x + sqrt(x^2 - 1))
                DoubleVector expr = x.add(x.mul(x).sub(one).lanewise(VectorOperators.SQRT));
                expr.lanewise(VectorOperators.LOG).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                if (x < 1.0) {
                    s[base + i] = Double.NaN;
                } else {
                    s[base + i] = Math.log(x + Math.sqrt(x * x - 1.0));
                }
            }
        }

        public static void atanh(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector x = DoubleVector.fromArray(SPECIES, s, base + i);
                // 0.5 * ln((1 + x) / (1 - x))
                DoubleVector div = one.add(x).div(one.sub(x));
                div.lanewise(VectorOperators.LOG).mul(half).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = 0.5 * Math.log((1.0 + x) / (1.0 - x));
            }
        }

        // ===================== Power & Exponential =====================
        public static void pow(int base, int n, double[] s, double exponent) {
            DoubleVector vExp = DoubleVector.broadcast(SPECIES, exponent);
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                v.lanewise(VectorOperators.POW, vExp).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.pow(s[base + i], exponent);
            }
        }

        public static void exp(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.EXP, Math::exp);
        }

        public static void ln(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.LOG, Math::log);
        }

        public static void lg(int base, int n, double[] s) {
            process(base, n, s, VectorOperators.LOG10, Math::log10);
        }

        // ===================== Stirling's Factorial Approximation =====================
        public static void stirling(int base, int n, double[] s) {
            int bound = SPECIES.loopBound(n);
            DoubleVector pi2 = DoubleVector.broadcast(SPECIES, 2.0 * Math.PI);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = one.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);
                result.intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                if (x <= 0) {
                    s[base + i] = Double.NaN;
                    continue;
                }
                double lnFact = x * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI * x) + 1.0 / (12.0 * x);
                s[base + i] = Math.exp(lnFact);
            }
        }

        // ===================== Private Vector Engine Enforcers =====================
        private static void processReciprocal(int base, int n, double[] s,
                VectorOperators.Unary op,
                DoubleUnaryOperator scalar) {
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                one.div(v.lanewise(op)).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / scalar.applyAsDouble(s[base + i]);
            }
        }

        private static void scaleInputReciprocal(int base, int n, double[] s, double scale,
                VectorOperators.Unary op,
                DoubleUnaryOperator scalar) {
            DoubleVector vScale = DoubleVector.broadcast(SPECIES, scale);
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i).mul(vScale);
                one.div(v.lanewise(op)).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / scalar.applyAsDouble(s[base + i] * scale);
            }
        }

        private static void processInverseReciprocal(int base, int n, double[] s,
                VectorOperators.Unary op,
                DoubleUnaryOperator scalar) {
            DoubleVector one = DoubleVector.broadcast(SPECIES, 1.0);
            int bound = SPECIES.loopBound(n);
            int i = 0;
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                one.div(v).lanewise(op).intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = scalar.applyAsDouble(1.0 / s[base + i]);
            }
        }
    }

}
