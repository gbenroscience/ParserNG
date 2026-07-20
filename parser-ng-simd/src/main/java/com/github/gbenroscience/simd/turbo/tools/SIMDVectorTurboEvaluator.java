package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.utils.VectorConfig.*;

import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import com.github.gbenroscience.simd.turbo.tools.utils.VectorizedCodyMath;
import jdk.incubator.vector.*;
import java.util.stream.IntStream;

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

        // Pre-allocated ONCE during initialization/compilation
        private final int NUM_WORKERS;

        SIMDVectorCompositeExpression() {
            super(compiledScalarHandle, opcodes, targetSlots,
                    literalConstants, instructionCount, varCount);

            if (numWorkers <= 2) {
                this.NUM_WORKERS = numWorkers;
            } else {
                this.NUM_WORKERS = numWorkers - 1;
            }
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

        // --- JDK 21+ ForkJoinPool Dispatchers ---
        @Override
        protected void dispatchToWorkerRing(double[] flatVariables, double[] output, int dataSize) {
            int numCores = NUM_WORKERS;
            int chunkSize = (dataSize + numCores - 1) / numCores;

            IntStream.range(0, numCores).parallel().forEach(workerId -> {
                int start = workerId * chunkSize;
                int end = Math.min(start + chunkSize, dataSize);
                if (start < end) {
                    applyBulkInternal(flatVariables, dataSize, output, start, end - start);
                }
            });
        }

        @Override
        protected void dispatchToWorkerRing(double[][] variables, double[] output, int dataSize) {
            int numCores = NUM_WORKERS;
            int chunkSize = (dataSize + numCores - 1) / numCores;

            IntStream.range(0, numCores).parallel().forEach(workerId -> {
                int start = workerId * chunkSize;
                int end = Math.min(start + chunkSize, dataSize);
                if (start < end) {
                    applyBulkInternal(variables, dataSize, output, start, end - start);
                }
            });
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
                        final int expOffset = (sp - 1) * BLOCK_SIZE;
                        final int baseOffset = (sp - 2) * BLOCK_SIZE;

                        // Keep the switch thin! Delegate out to highly optimized,
                        // easily inlined method kernels to preserve Escape Analysis.
                        VectorMath.executePowerBlended(scratch, baseOffset, expOffset, n);

                        sp--;
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
                            DoubleVector expNegX = VectorMath.fastVectorExp(x.neg()); // x.neg() simply flips the sign bit (fast)
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
                            DoubleVector erfVal = VectorMath.vectorizedErf(y.mul(INV_SQRT_2));
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
                        VectorMath.sin((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_COS -> {
                        VectorMath.cos((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_TAN -> {
                        VectorMath.tan((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_SINH -> {
                        VectorMath.sinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_COSH -> {
                        VectorMath.cosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_TANH -> {
                        VectorMath.tanh((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ABS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[srcOffset + k] = Math.abs(scratch[srcOffset + k]);
                        }
                    }

                    case OP_EXP -> {
                        VectorMath.exp((sp - 1) * BLOCK_SIZE, n, scratch);
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
                            DoubleVector expNegX = VectorMath.fastVectorExp(x.neg());
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
                                result = x.mul(HALF).mul(VectorMath.vectorizedErf(x.mul(INV_SQRT_2)).add(ONE));
                            } else if (opcode == OP_GELU_FAST) {
                                // FAST GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                                final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
                                final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);

                                DoubleVector x3 = x.mul(x).mul(x);
                                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);

                                // Using your optimized fastVectorExp for tanh calculation
                                DoubleVector exp2z = VectorMath.fastVectorExp(z.mul(TWO));
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
                            DoubleVector result = VectorMath.vectorizedErf(x);

                            result.intoArray(scratch, base + k);
                        }

                        // 2. Scalar tail loop for remaining elements
                        for (; k < n; k++) {
                            scratch[base + k] = Maths.erf(scratch[base + k]);
                        }
                    }

                    case OP_LOG -> {
                        VectorMath.ln((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_LOG10 -> {
                        VectorMath.log10((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    // --- Inverse Radians (Routed to VectorMath for Alias Mapping) ---
                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        VectorMath.asin((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        VectorMath.acos((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        VectorMath.atan((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    // --- DEGREE / GRADIAN TRIG VARIANTS ---
                    case OP_SIN_DEG ->
                        VectorMath.sinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_DEG ->
                        VectorMath.cosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_DEG ->
                        VectorMath.tanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SIN_GRAD ->
                        VectorMath.sinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_GRAD ->
                        VectorMath.cosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_GRAD ->
                        VectorMath.tanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE DEGREE / GRADIAN VARIANTS ---
                    case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG ->
                        VectorMath.asinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG ->
                        VectorMath.acosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG ->
                        VectorMath.atanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD ->
                        VectorMath.asinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD ->
                        VectorMath.acosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD ->
                        VectorMath.atanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- RECIPROCAL TRIG (SEC, CSC, COT) VARIANTS ---
                    case OP_SEC ->
                        VectorMath.sec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_DEG ->
                        VectorMath.secDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_GRAD ->
                        VectorMath.secGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC ->
                        VectorMath.csc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_DEG ->
                        VectorMath.cscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_GRAD ->
                        VectorMath.cscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT ->
                        VectorMath.cot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_DEG ->
                        VectorMath.cotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_GRAD ->
                        VectorMath.cotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE RECIPROCAL TRIG VARIANTS ---
                    case OP_ARC_SEC, OP_ARC_SEC_ALT ->
                        VectorMath.asec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG ->
                        VectorMath.asecDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD ->
                        VectorMath.asecGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC, OP_ARC_COSEC_ALT ->
                        VectorMath.acsc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG ->
                        VectorMath.acscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD ->
                        VectorMath.acscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT, OP_ARC_COT_ALT ->
                        VectorMath.acot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG ->
                        VectorMath.acotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD ->
                        VectorMath.acotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- HYPERBOLIC INVERSES ---
                    case OP_ASINH, OP_ASINH_ALT ->
                        VectorMath.asinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOSH, OP_ACOSH_ALT ->
                        VectorMath.acosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATANH, OP_ATANH_ALT ->
                        VectorMath.atanh((sp - 1) * BLOCK_SIZE, n, scratch);

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

                    case OP_VMA -> {
                        final int cOffset = (--sp) * BLOCK_SIZE;
                        final int bOffset = (--sp) * BLOCK_SIZE;
                        final int aOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;

                        final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

                        int k = 0;
                        // 1. Process as many full vectors as possible
                        int bound = SPECIES.loopBound(n);
                        for (; k < bound; k += SPECIES.length()) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, aOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, bOffset + k);
                            DoubleVector vc = DoubleVector.fromArray(SPECIES, scratch, cOffset + k);

                            va.fma(vb, vc).intoArray(scratch, resOffset + k);
                        }

                        // 2. Handle the "tail" using a Mask
                        // This creates a mask where only indices < n are enabled
                        if (k < n) {
                            VectorMask<Double> mask = SPECIES.indexInRange(k, n);

                            // Use 'fromArray' with a mask to safely load only valid elements
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, aOffset + k, mask);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, bOffset + k, mask);
                            DoubleVector vc = DoubleVector.fromArray(SPECIES, scratch, cOffset + k, mask);

                            // FMA with mask - only updates enabled lanes
                            va.fma(vb, vc).intoArray(scratch, resOffset + k, mask);
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
                        final int expOffset = (sp - 1) * BLOCK_SIZE;
                        final int baseOffset = (sp - 2) * BLOCK_SIZE;

                        // Keep the switch thin! Delegate out to highly optimized,
                        // easily inlined method kernels to preserve Escape Analysis.
                        VectorMath.executePowerBlended(scratch, baseOffset, expOffset, n);

                        sp--;
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
                            DoubleVector expNegX = VectorMath.fastVectorExp(x.neg()); // x.neg() simply flips the sign bit (fast)
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
                            DoubleVector erfVal = VectorMath.vectorizedErf(y.mul(INV_SQRT_2));
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
                        VectorMath.sin((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_COS -> {
                        VectorMath.cos((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_TAN -> {
                        VectorMath.tan((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_SINH -> {
                        VectorMath.sinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_COSH -> {
                        VectorMath.cosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_TANH -> {
                        VectorMath.tanh((sp - 1) * BLOCK_SIZE, n, scratch);
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
                        VectorMath.exp((sp - 1) * BLOCK_SIZE, n, scratch);
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
                            DoubleVector expNegX = VectorMath.fastVectorExp(x.neg());
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
                                result = x.mul(HALF).mul(VectorMath.vectorizedErf(x.mul(INV_SQRT_2)).add(ONE));
                            } else if (opcode == OP_GELU_FAST) {
                                // FAST GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                                final DoubleVector SQRT_2_OVER_PI = DoubleVector.broadcast(SPECIES, 0.7978845608028654);
                                final DoubleVector COEF = DoubleVector.broadcast(SPECIES, 0.044715);

                                DoubleVector x3 = x.mul(x).mul(x);
                                DoubleVector z = x3.mul(COEF).add(x).mul(SQRT_2_OVER_PI);

                                // Using your optimized fastVectorExp for tanh calculation
                                DoubleVector exp2z = VectorMath.fastVectorExp(z.mul(TWO));
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
                            DoubleVector result = VectorMath.vectorizedErf(x);

                            result.intoArray(scratch, base + k);
                        }
                        // 2. Scalar tail loop for remaining elements
                        for (; k < n; k++) {
                            scratch[base + k] = Maths.erf(scratch[base + k]);
                        }
                    }

                    case OP_LOG -> {
                        VectorMath.ln((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_LOG10 -> {
                        VectorMath.log10((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    // --- Inverse Radians (Routed to VectorMath for Alias Mapping) ---
                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        VectorMath.asin((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        VectorMath.acos((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        VectorMath.atan((sp - 1) * BLOCK_SIZE, n, scratch);
                    }

                    // --- DEGREE / GRADIAN TRIG VARIANTS ---
                    case OP_SIN_DEG ->
                        VectorMath.sinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_DEG ->
                        VectorMath.cosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_DEG ->
                        VectorMath.tanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SIN_GRAD ->
                        VectorMath.sinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COS_GRAD ->
                        VectorMath.cosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_TAN_GRAD ->
                        VectorMath.tanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE DEGREE / GRADIAN VARIANTS ---
                    case OP_ASIN_DEG, OP_ASIN_DEG_ALT, OP_ARC_SIN_ALT_DEG ->
                        VectorMath.asinDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_DEG, OP_ACOS_DEG_ALT, OP_ARC_COS_ALT_DEG ->
                        VectorMath.acosDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_DEG, OP_ATAN_DEG_ALT, OP_ARC_TAN_ALT_DEG ->
                        VectorMath.atanDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ASIN_GRAD, OP_ASIN_GRAD_ALT, OP_ARC_SIN_ALT_GRAD ->
                        VectorMath.asinGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOS_GRAD, OP_ACOS_GRAD_ALT, OP_ARC_COS_ALT_GRAD ->
                        VectorMath.acosGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATAN_GRAD, OP_ATAN_GRAD_ALT, OP_ARC_TAN_ALT_GRAD ->
                        VectorMath.atanGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- RECIPROCAL TRIG (SEC, CSC, COT) VARIANTS ---
                    case OP_SEC ->
                        VectorMath.sec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_DEG ->
                        VectorMath.secDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_SEC_GRAD ->
                        VectorMath.secGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC ->
                        VectorMath.csc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_DEG ->
                        VectorMath.cscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COSEC_GRAD ->
                        VectorMath.cscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT ->
                        VectorMath.cot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_DEG ->
                        VectorMath.cotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_COT_GRAD ->
                        VectorMath.cotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- INVERSE RECIPROCAL TRIG VARIANTS ---
                    case OP_ARC_SEC, OP_ARC_SEC_ALT ->
                        VectorMath.asec((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_DEG, OP_ARC_SEC_ALT_DEG ->
                        VectorMath.asecDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_SEC_GRAD, OP_ARC_SEC_ALT_GRAD ->
                        VectorMath.asecGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC, OP_ARC_COSEC_ALT ->
                        VectorMath.acsc((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_DEG, OP_ARC_COSEC_ALT_DEG ->
                        VectorMath.acscDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COSEC_GRAD, OP_ARC_COSEC_ALT_GRAD ->
                        VectorMath.acscGrad((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT, OP_ARC_COT_ALT ->
                        VectorMath.acot((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_DEG, OP_ARC_COT_ALT_DEG ->
                        VectorMath.acotDeg((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ARC_COT_GRAD, OP_ARC_COT_ALT_GRAD ->
                        VectorMath.acotGrad((sp - 1) * BLOCK_SIZE, n, scratch);

                    // --- HYPERBOLIC INVERSES ---
                    case OP_ASINH, OP_ASINH_ALT ->
                        VectorMath.asinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOSH, OP_ACOSH_ALT ->
                        VectorMath.acosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATANH, OP_ATANH_ALT ->
                        VectorMath.atanh((sp - 1) * BLOCK_SIZE, n, scratch);

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

                        int k = 0;
                        // 1. Process as many full vectors as possible
                        int bound = SPECIES.loopBound(n);
                        for (; k < bound; k += SPECIES.length()) {
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, aOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, bOffset + k);
                            DoubleVector vc = DoubleVector.fromArray(SPECIES, scratch, cOffset + k);

                            va.fma(vb, vc).intoArray(scratch, resOffset + k);
                        }

                        // 2. Handle the "tail" using a Mask
                        // This creates a mask where only indices < n are enabled
                        if (k < n) {
                            VectorMask<Double> mask = SPECIES.indexInRange(k, n);

                            // Use 'fromArray' with a mask to safely load only valid elements
                            DoubleVector va  = DoubleVector.fromArray(SPECIES, scratch, aOffset + k, mask);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, bOffset + k, mask);
                            DoubleVector vc = DoubleVector.fromArray(SPECIES, scratch, cOffset + k, mask);

                            // FMA with mask - only updates enabled lanes
                            va.fma(vb, vc).intoArray(scratch, resOffset + k, mask);
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

    public final class VectorMath {

        private VectorMath() {
        }

        private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
        public static int VECTOR_THRESHOLD = 256;

        // Angle conversions
        private static final double DEG_TO_RAD = Math.PI / 180.0;
        private static final double RAD_TO_DEG = 180.0 / Math.PI;
        private static final double GRAD_TO_RAD = Math.PI / 200.0;
        private static final double RAD_TO_GRAD = 200.0 / Math.PI;

        private static final DoubleVector V_DEG_TO_RAD = DoubleVector.broadcast(SPECIES, DEG_TO_RAD);
        private static final DoubleVector V_RAD_TO_DEG = DoubleVector.broadcast(SPECIES, RAD_TO_DEG);
        private static final DoubleVector V_GRAD_TO_RAD = DoubleVector.broadcast(SPECIES, GRAD_TO_RAD);
        private static final DoubleVector V_RAD_TO_GRAD = DoubleVector.broadcast(SPECIES, RAD_TO_GRAD);

        // Core constants
        private static final DoubleVector V_ONE = DoubleVector.broadcast(SPECIES, 1.0);
        private static final DoubleVector V_NEG_ONE = DoubleVector.broadcast(SPECIES, -1.0);
        private static final DoubleVector V_HALF = DoubleVector.broadcast(SPECIES, 0.5);
        private static final DoubleVector V_HALF_PI = DoubleVector.broadcast(SPECIES, Math.PI / 2.0);
        private static final DoubleVector V_NEG_HALF_PI = DoubleVector.broadcast(SPECIES, -Math.PI / 2.0);
        private static final DoubleVector V_NAN = DoubleVector.broadcast(SPECIES, Double.NaN);
        private static final DoubleVector ZERO = DoubleVector.broadcast(SPECIES, 0.0);

        private static final double THRESHOLD_LOW = 0.46875;
        private static final double THRESHOLD_HIGH = 4.0;

        // ========================================================================
        // NO-LAMBDA DIRECT OPERATIONS
        // ========================================================================
        // Radian
        public static void sin(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(s[base + i]);
            }
        }

        public static void cos(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(s[base + i]);
            }
        }

        public static void tan(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void sinDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cosDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void tanDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void sinGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sin(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cosGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cos(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void tanGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.TAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tan(s[base + i] * GRAD_TO_RAD);
            }
        }

        // ===================== Reciprocal Trigonometric =====================
        // Radian
        public static void sec(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(s[base + i]);
            }
        }

        public static void csc(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(s[base + i]);
            }
        }

        public static void cot(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(s[base + i]);
            }
        }

        // Degree
        public static void secDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(Math.toRadians(s[base + i]));
            }
        }

        public static void cscDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(Math.toRadians(s[base + i]));
            }
        }

        public static void cotDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_DEG_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(Math.toRadians(s[base + i]));
            }
        }

        // Grad
        public static void secGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.COS))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.cos(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cscGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD)
                        .lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.sin(s[base + i] * GRAD_TO_RAD);
            }
        }

        public static void cotGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i)
                        .mul(V_GRAD_TO_RAD);
                v.lanewise(VectorOperators.COS)
                        .div(v.lanewise(VectorOperators.SIN))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = 1.0 / Math.tan(s[base + i] * GRAD_TO_RAD);
            }
        }

        // ===================== Inverse Trigonometric =====================
        // Radian
        public static void asin(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(s[base + i]);
            }
        }

        public static void acos(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(s[base + i]);
            }
        }

        public static void atan(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(s[base + i]);
            }
        }

        // Degree
        public static void asinDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.asin(s[base + i]));
            }
        }

        public static void acosDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.acos(s[base + i]));
            }
        }

        public static void atanDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.atan(s[base + i]));
            }
        }

        // Grad
        public static void asinGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acosGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void atanGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Inverse Reciprocal Trigonometric =====================
        // Radian
        public static void acsc(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]);
            }
        }

        public static void asec(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]);
            }
        }

        public static void acot(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(1.0 / s[base + i]);
            }
        }

        // Degree
        public static void acscDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.asin(1.0 / s[base + i]));
            }
        }

        public static void asecDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.acos(1.0 / s[base + i]));
            }
        }

        public static void acotDeg(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_DEG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.toDegrees(Math.atan(1.0 / s[base + i]));
            }
        }

        // Grad
        public static void acscGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ASIN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.asin(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void asecGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ACOS)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.acos(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        public static void acotGrad(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                V_ONE.div(DoubleVector.fromArray(SPECIES, s, base + i))
                        .lanewise(VectorOperators.ATAN)
                        .mul(V_RAD_TO_GRAD)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.atan(1.0 / s[base + i]) * RAD_TO_GRAD;
            }
        }

        // ===================== Hyperbolic =====================
        public static void sinh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.SINH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.sinh(s[base + i]);
            }
        }

        public static void cosh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.COSH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.cosh(s[base + i]);
            }
        }

        public static void tanh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.TANH)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.tanh(s[base + i]);
            }
        }

        // ===================== Inverse Hyperbolic =====================
        public static void asinh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsinhImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log(s[base + i] + Math.sqrt(s[base + i] * s[base + i] + 1.0));
            }
        }

        public static void acosh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcoshImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = x < 1.0 ? Double.NaN : Math.log(x + Math.sqrt(x * x - 1.0));
            }
        }

        public static void atanh(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAtanhImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = 0.5 * Math.log((1.0 + x) / (1.0 - x));
            }
        }

        public static void asech(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAsechImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = (x <= 0.0 || x > 1.0) ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) - 1.0));
            }
        }

        public static void acsch(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcschImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = x == 0.0 ? Double.NaN : Math.log((1.0 / x) + Math.sqrt((1.0 / (x * x)) + 1.0));
            }
        }

        public static void acoth(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                vectorAcothImpl(DoubleVector.fromArray(SPECIES, s, base + i))
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                double x = s[base + i];
                s[base + i] = Math.abs(x) <= 1.0 ? Double.NaN : 0.5 * Math.log((1.0 + (1.0 / x)) / (1.0 - (1.0 / x)));
            }
        }

        // ===================== Exponential and Logarithmic =====================
        public static void exp(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.EXP)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.exp(s[base + i]);
            }
        }

        public static void ln(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.LOG)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log(s[base + i]);
            }
        }

        public static void log10(int base, int n, double[] s) {
            int i = 0;
            int limit = SPECIES.loopBound(n);
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector.fromArray(SPECIES, s, base + i)
                        .lanewise(VectorOperators.LOG10)
                        .intoArray(s, base + i);
            }
            for (; i < n; i++) {
                s[base + i] = Math.log10(s[base + i]);
            }
        }

        private static boolean isExponentUniform(double[] scratch, int offset, int n) {
            if (n <= 1) {
                return true;
            }

            final double first = scratch[offset];
            if (Double.isNaN(first)) {
                // All must be NaN
                final int vl = SPECIES.length();
                int i = 0;
                int bound = SPECIES.loopBound(n);
                for (; i < bound; i += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i);
                    if (v.compare(VectorOperators.EQ, v).anyTrue()) {
                        return false;
                    }
                }
                int remaining = n - i;
                if (remaining > 0) {
                    var mask = SPECIES.indexInRange(0, remaining);
                    DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i, mask);
                    if (v.compare(VectorOperators.EQ, v, mask).anyTrue()) {
                        return false;
                    }
                }
                return true;
            }

            final DoubleVector target = DoubleVector.broadcast(SPECIES, first);
            final int vl = SPECIES.length();
            int i = 0;
            int bound = SPECIES.loopBound(n);

            for (; i < bound; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i);
                if (v.compare(VectorOperators.NE, target).anyTrue()) {
                    return false;
                }
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, scratch, offset + i, mask);
                if (v.compare(VectorOperators.NE, target, mask).anyTrue()) {
                    return false;
                }
            }
            return true;
        }

        public static void evaluateVariableExponent(double[] base, int bOffset, double[] exp, int eOffset,
                double[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            int i = 0;
            final int limit = SPECIES.loopBound(n);

            // === 1. Core Vector Loop: exp(y * ln(x)) ===
            for (; i < limit; i += SPECIES.length()) {
                DoubleVector vBase = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                DoubleVector vExp = DoubleVector.fromArray(SPECIES, exp, eOffset + i);

                // Execute algebraic transcendental transformation
                DoubleVector log = vBase.lanewise(VectorOperators.LOG);
                DoubleVector scaled = log.mul(vExp);
                scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
            }

            // === 2. Masked Tail Pass ===
            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector vBase = DoubleVector.fromArray(SPECIES, base, bOffset + i, mask);
                DoubleVector vExp = DoubleVector.fromArray(SPECIES, exp, eOffset + i, mask);

                // Apply masks to intermediate operators to maintain lane isolation
                DoubleVector log = vBase.lanewise(VectorOperators.LOG, mask);
                DoubleVector scaled = log.mul(vExp, mask);
                DoubleVector res = scaled.lanewise(VectorOperators.EXP, mask);

                res.intoArray(dest, dOffset + i, mask);
            }
        }

        public static void executePowerBlended(double[] scratch, int baseOffset, int expOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (isExponentUniform(scratch, expOffset, n)) {
                double uniformExp = scratch[expOffset];

                if (uniformExp == 0.5) {
                    VectorTranscendentals.evaluateNative(scratch, baseOffset, scratch, baseOffset, n, VectorOperators.SQRT);
                    return;
                }
                if (uniformExp == 2.0) {
                    computeSquare(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }
                if (uniformExp == 3.0) {
                    computeCube(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }
                if (uniformExp == 4.0) {
                    computeFourthPower(scratch, baseOffset, scratch, baseOffset, n);
                    return;
                }

                // Isolated fallback for uniform constants
                evaluateUniformExponent(scratch, baseOffset, uniformExp, scratch, baseOffset, n);
            } else {
                // Isolated fallback for variable exponents
                evaluateVariableExponent(scratch, baseOffset, scratch, expOffset, scratch, baseOffset, n);
            }
        }

// ==========================================
// Isolated Fast-Path Micro-Methods (EA Safe)
// ==========================================
        private static void computeSquare(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                v.mul(v).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                v.mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeCube(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                v.mul(v).mul(v).intoArray(dest, destOff + k);
            }

            int remaining = k - n; // Wait, original had n - k, let's keep it safe:
            remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                v.mul(v).mul(v).intoArray(dest, destOff + k, mask);
            }
        }

        private static void computeFourthPower(double[] src, int srcOff, double[] dest, int destOff, int n) {
            int k = 0;
            final int limit = SPECIES.loopBound(n);
            final int vl = SPECIES.length();

            for (; k < limit; k += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k);
                DoubleVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k);
            }

            int remaining = n - k;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, src, srcOff + k, mask);
                DoubleVector sq = v.mul(v);
                sq.mul(sq).intoArray(dest, destOff + k, mask);
            }
        }

        public static void evaluateUniformExponent(double[] base, int bOffset, double exp,
                double[] dest, int dOffset, int n) {
            if (n <= 0) {
                return;
            }

            if (exp == 1.0) {
                if (base != dest || bOffset != dOffset) {
                    System.arraycopy(base, bOffset, dest, dOffset, n);
                }
                return;
            }
            if (exp == 2.0) {
                computeSquare(base, bOffset, dest, dOffset, n);
                return;
            }
            if (exp == 3.0) {
                computeCube(base, bOffset, dest, dOffset, n);
                return;
            }
            if (exp == 4.0) {
                computeFourthPower(base, bOffset, dest, dOffset, n);
                return;
            }

            if (exp == 0.5) {
                VectorTranscendentals.evaluateNative(base, bOffset, dest, dOffset, n, VectorOperators.SQRT);
                return;
            }

            // Delegate the highly complex log/exp routines to a separate compilation target
            evaluateComplexUniformExponent(base, bOffset, exp, dest, dOffset, n);
        }

        private static void evaluateComplexUniformExponent(double[] base, int bOffset, double exp,
                double[] dest, int dOffset, int n) {
            final int vl = SPECIES.length();
            final int limit = SPECIES.loopBound(n);
            int i = 0;

            if (exp == 0.0) {
                for (; i < limit; i += vl) {
                    V_ONE.intoArray(dest, dOffset + i);
                }
            } else if (exp == -1.0) {
                for (; i < limit; i += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                    V_ONE.div(v).intoArray(dest, dOffset + i);
                }
            } else {
                final DoubleVector vExp = DoubleVector.broadcast(SPECIES, exp);
                if (exp % 1.0 == 0.0) {
                    if (exp % 2.0 != 0.0) {
                        // Scenario 1: Odd Integer (FIXED: targetIdx bug resolved)
                        for (; i < limit; i += vl) {
                            DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                            var isNegativeMask = v.compare(VectorOperators.LT, 0.0);
                            DoubleVector log = v.abs().lanewise(VectorOperators.LOG);
                            DoubleVector scaled = log.mul(vExp);
                            DoubleVector resAbs = scaled.lanewise(VectorOperators.EXP);
                            resAbs.blend(resAbs.neg(), isNegativeMask).intoArray(dest, dOffset + i);
                        }
                    } else {
                        // Scenario 2: Even Integer
                        for (; i < limit; i += vl) {
                            DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                            DoubleVector log = v.abs().lanewise(VectorOperators.LOG);
                            DoubleVector scaled = log.mul(vExp);
                            scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                        }
                    }
                } else {
                    // Scenario 3: Non-Integer
                    for (; i < limit; i += vl) {
                        DoubleVector v = DoubleVector.fromArray(SPECIES, base, bOffset + i);
                        DoubleVector log = v.lanewise(VectorOperators.LOG);
                        DoubleVector scaled = log.mul(vExp);
                        scaled.lanewise(VectorOperators.EXP).intoArray(dest, dOffset + i);
                    }
                }
            }

            // Clean Scalar Tail Pass
            for (; i < n; i++) {
                final double b = base[bOffset + i];
                dest[dOffset + i] = (exp == 0.0) ? 1.0 : (exp == -1.0) ? 1.0 / b : Math.pow(b, exp);
            }
        }

        // ========================================================================
        // Specialized Mathematical Transcendentals
        // ========================================================================
        /**
         * High-performance vectorized exp() using magic-number rounding +
         * 6th-degree minimax polynomial via FMA + fast bit manipulation for
         * 2^k.
         */
        private static DoubleVector fastVectorExp(DoubleVector x) {
            x = x.lanewise(VectorOperators.MAX, -745.13).lanewise(VectorOperators.MIN, 709.78);

            DoubleVector invLn2 = DoubleVector.broadcast(SPECIES, 1.4426950408889634074);
            DoubleVector ln2Hi = DoubleVector.broadcast(SPECIES, -0.6931471805599453);
            DoubleVector ln2Lo = DoubleVector.broadcast(SPECIES, -2.8235290563031574E-13);

            DoubleVector magic = DoubleVector.broadcast(SPECIES, 4503599627370496.0); // 2^52
            DoubleVector k = x.mul(invLn2).add(magic).sub(magic);
            DoubleVector r = x.add(k.mul(ln2Hi)).add(k.mul(ln2Lo));

            DoubleVector p = r.mul(0.001398199650).add(0.0088632903);
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.04166666666));
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.16666666666));
            p = r.lanewise(VectorOperators.FMA, p, DoubleVector.broadcast(SPECIES, 0.5));
            p = r.lanewise(VectorOperators.FMA, p, V_ONE);
            p = r.lanewise(VectorOperators.FMA, p, V_ONE);

            LongVector kLong = (LongVector) k.convert(VectorOperators.D2L, 0);
            LongVector exponent = kLong.add(1023).lanewise(VectorOperators.LSHL, 52);
            DoubleVector twoK = (DoubleVector) exponent.convert(VectorOperators.REINTERPRET_L2D, 0);

            return p.mul(twoK);
        }

        private static DoubleVector vectorizedErf(DoubleVector x) {
            DoubleVector absX = x.abs();

            VectorMask<Double> maskLow = absX.compare(VectorOperators.LE, THRESHOLD_LOW);
            VectorMask<Double> maskHigh = absX.compare(VectorOperators.LE, THRESHOLD_HIGH);

            DoubleVector xSq = absX.mul(absX);
            DoubleVector p = xSq.mul(0.260194122534674).add(30.59022585250011).mul(xSq)
                    .add(573.9507736045833).mul(xSq).add(2801.752391065013).mul(xSq).add(3204.677458505002);
            DoubleVector q = xSq.add(159.0884090976454).mul(xSq).add(1422.080683811422).mul(xSq)
                    .add(4423.613442045816).mul(xSq).add(3204.677458506958);

            DoubleVector resLow = x.mul(p.div(q));

            // Note: Assumes VectorizedCodyMath is available in your classpath
            DoubleVector resMed = VectorizedCodyMath.evaluateMediumVector(x, absX, maskHigh);
            DoubleVector resHigh = VectorizedCodyMath.evaluateLargeVector(x, absX, maskHigh.not());
            DoubleVector erfcVal = resMed.blend(resHigh, maskHigh.not());

            DoubleVector resMidHigh = V_ONE.sub(erfcVal);
            VectorMask<Double> isNegative = x.compare(VectorOperators.LT, 0.0);
            resMidHigh = resMidHigh.blend(resMidHigh.neg(), isNegative);

            return resMidHigh.blend(resLow, maskLow);
        }

        // ===================== Stirling's Factorial Approximation =====================
        public static void stirling(int base, int n, double[] s) {
            int vl = SPECIES.length();
            int bound = SPECIES.loopBound(n);
            DoubleVector pi2 = DoubleVector.broadcast(SPECIES, 2.0 * Math.PI);
            DoubleVector nanVec = DoubleVector.broadcast(SPECIES, Double.NaN);
            int i = 0;

            for (; i < bound; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = V_ONE.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0);
                result.blend(nanVec, invalidMask).intoArray(s, base + i);
            }

            int remaining = n - i;
            if (remaining > 0) {
                var mask = SPECIES.indexInRange(0, remaining);
                DoubleVector v = DoubleVector.fromArray(SPECIES, s, base + i, mask);
                DoubleVector lnN = v.lanewise(VectorOperators.LOG);
                DoubleVector term1 = v.mul(lnN).sub(v);
                DoubleVector term2 = pi2.mul(v).lanewise(VectorOperators.LOG).mul(0.5);
                DoubleVector term3 = V_ONE.div(v.mul(12.0));
                DoubleVector result = term1.add(term2).add(term3).lanewise(VectorOperators.EXP);

                var invalidMask = v.compare(VectorOperators.LE, 0.0);
                result.blend(nanVec, invalidMask).intoArray(s, base + i, mask);
            }
        }

        // ===================== Conditional Branching =====================
        public static void if3(int base, int tileN, double[] s, int block) {
            final int cond = base + block;
            final int trueVal = base + 2 * block;
            final int falseVal = base + 3 * block;
            final int res = base;

            int vl = SPECIES.length();
            int bound = SPECIES.loopBound(tileN);
            int i = 0;

            for (; i < bound; i += vl) {
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i);
            }

            int remaining = tileN - i;
            if (remaining > 0) {
                var maskTail = SPECIES.indexInRange(0, remaining);
                DoubleVector vc = DoubleVector.fromArray(SPECIES, s, cond + i, maskTail);
                DoubleVector vt = DoubleVector.fromArray(SPECIES, s, trueVal + i, maskTail);
                DoubleVector vf = DoubleVector.fromArray(SPECIES, s, falseVal + i, maskTail);
                VectorMask<Double> mask = vc.compare(VectorOperators.NE, 0.0).and(vc.compare(VectorOperators.EQ, vc));
                vf.blend(vt, mask).intoArray(s, res + i, maskTail);
            }
        }

        // ========================================================================
        // Vectorized Inverse Hyperbolic Implementations
        // ========================================================================
        private static DoubleVector vectorAsinhImpl(DoubleVector x) {
            return x.add(x.mul(x).add(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
        }

        private static DoubleVector vectorAcoshImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.GE, V_ONE);
            DoubleVector result = x.add(x.mul(x).sub(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAtanhImpl(DoubleVector x) {
            VectorMask<Double> valid = x.abs().compare(VectorOperators.LT, V_ONE);
            DoubleVector result = V_ONE.add(x).div(V_ONE.sub(x))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAsechImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.GT, 0.0)
                    .and(x.compare(VectorOperators.LE, V_ONE));
            DoubleVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).sub(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAcschImpl(DoubleVector x) {
            VectorMask<Double> valid = x.compare(VectorOperators.NE, 0.0);
            DoubleVector result = V_ONE.div(x).add(V_ONE.div(x.mul(x)).add(V_ONE).lanewise(VectorOperators.SQRT))
                    .lanewise(VectorOperators.LOG);
            return result.blend(V_NAN, valid.not());
        }

        private static DoubleVector vectorAcothImpl(DoubleVector x) {
            VectorMask<Double> valid = x.abs().compare(VectorOperators.GT, V_ONE);
            DoubleVector result = V_ONE.add(V_ONE.div(x)).div(V_ONE.sub(V_ONE.div(x)))
                    .lanewise(VectorOperators.LOG)
                    .mul(V_HALF);
            return result.blend(V_NAN, valid.not());
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

}
