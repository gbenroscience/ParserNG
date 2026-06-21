//CURRENT CRASHING VERSION
package com.github.gbenroscience.simd.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.*;

import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import java.util.function.DoubleUnaryOperator;
import jdk.incubator.vector.*;

/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 */
public class SIMDVectorTurboEvaluator extends VectorTurboEvaluator {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    // private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final int VLEN = SPECIES.length();

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
        public double applyScalar(double[] variables) {
            double[] stack = SINGLE_SCALAR_BUFFER.get();
            int sp = 0;
            for (int i = 0; i < instructionCount; i++) {
                switch (opcodes[i]) {
                    case OP_CONST ->
                        stack[sp++] = literalConstants[i];
                    case OP_LOAD ->
                        stack[sp++] = variables[targetSlots[i]];
                    case OP_ADD -> {
                        double b = stack[--sp];
                        stack[sp - 1] += b;
                    }
                    case OP_SUB -> {
                        double b = stack[--sp];
                        stack[sp - 1] -= b;
                    }
                    case OP_MUL -> {
                        double b = stack[--sp];
                        stack[sp - 1] *= b;
                    }
                    case OP_DIV -> {
                        double b = stack[--sp];
                        stack[sp - 1] /= b;
                    }
                    case OP_POW -> {
                        double b = stack[--sp];
                        stack[sp - 1] = pow(stack[sp - 1], b);
                    }
                    case OP_REM -> {
                        double b = stack[--sp];
                        stack[sp - 1] = stack[sp - 1] % b;
                    }
                    case OP_SIN ->
                        stack[sp - 1] = Math.sin(stack[sp - 1]);
                    case OP_COS ->
                        stack[sp - 1] = Math.cos(stack[sp - 1]);
                    case OP_TAN ->
                        stack[sp - 1] = Math.tan(stack[sp - 1]);
                    case OP_ASIN ->
                        stack[sp - 1] = Math.asin(stack[sp - 1]);
                    case OP_ACOS ->
                        stack[sp - 1] = Math.acos(stack[sp - 1]);
                    case OP_ATAN ->
                        stack[sp - 1] = Math.atan(stack[sp - 1]);
                    case OP_SINH ->
                        stack[sp - 1] = Math.sinh(stack[sp - 1]);
                    case OP_COSH ->
                        stack[sp - 1] = Math.cosh(stack[sp - 1]);
                    case OP_TANH ->
                        stack[sp - 1] = Math.tanh(stack[sp - 1]);
                    case OP_ABS ->
                        stack[sp - 1] = Math.abs(stack[sp - 1]);
                    case OP_EXP ->
                        stack[sp - 1] = Math.exp(stack[sp - 1]);
                    case OP_SQRT ->
                        stack[sp - 1] = Math.sqrt(stack[sp - 1]);
                    case OP_CBRT ->
                        stack[sp - 1] = Math.cbrt(stack[sp - 1]);
                    case OP_LOG ->
                        stack[sp - 1] = Math.log(stack[sp - 1]);
                    case OP_LOG10 ->
                        stack[sp - 1] = Math.log10(stack[sp - 1]);
                }
            }
            return stack[0];
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
        public void applyBulk(double[][] variables, double[] output, boolean tiledExecution) {
            checkError(variables, output);
            int numSamples = variables[0].length;
            double[] flatVars = getOrCreateFlattenBuffer(varCount * numSamples);
            for (int i = 0; i < varCount; i++) {
                System.arraycopy(variables[i], 0, flatVars, i * numSamples, numSamples);
            }
            applyBulkInternal(flatVars, numSamples, output, 0, numSamples, tiledExecution);
        }

        @Override
        public void applyBulkParallel(double[][] variables, double[] output) {
            checkError(variables, output);
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;
            if (numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulk(variables, output, false);
                return;
            }
            double[] flatVars = getOrCreateFlattenBuffer(varCount * numSamples);
            for (int i = 0; i < varCount; i++) {
                System.arraycopy(variables[i], 0, flatVars, i * numSamples, numSamples);
            }
            dispatchToWorkerRing(flatVars, output, numSamples, true);
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize, boolean tiledExecution) {
            checkError(variables, output);
            int numSamples = variables[0].length;
            double[] flatVars = getOrCreateFlattenBuffer(varCount * numSamples);
            for (int i = 0; i < varCount; i++) {
                System.arraycopy(variables[i], 0, flatVars, i * numSamples, numSamples);
            }
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVars, numSamples, output, start, length, tiledExecution);
            }
        }

        // --- 1D Flat Contiguous Frameworks ---
        @Override
        public void applyBulk(double[] flatVariables, double[] output, boolean tiledExecution) {
            checkError(flatVariables, output);
            applyBulkInternal(flatVariables, output.length, output, 0, output.length, tiledExecution);
        }

        @Override
        public void applyBulkParallel(double[] flatVariables, double[] output) {
            checkError(flatVariables, output);
            int numSamples = output.length;
            if (numSamples < PARALLEL_OPS_THRESHOLD) {
                applyBulkInternal(flatVariables, numSamples, output, 0, numSamples, false);
                return;
            }
            dispatchToWorkerRing(flatVariables, output, numSamples, true);
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize, boolean tiledExecution) {
            checkError(flatVariables, output);
            int numSamples = output.length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, numSamples, output, start, length, tiledExecution);
            }
        }

        //////////////////////////////////////////////////
        
        
     /**
         * Core Column-Major Vectorized Loop Processor (Flat Memory
         * Architecture) Bypasses jagged array pointers to execute at maximum
         * hardware throughput.
         *
         * @param flatVariables Grouped/Contiguous array of variables e.g.
         * [x1,x2..xn, y1,y2..yn, z1,z2..zn] back-to-back. concatenated.
         * @param dataSize The total number of elements per variable row (used
         * for stride offset).
         */
    
            private void applyBulkInternalWithTiles(double[] flatVariables, int dataSize, double[] output, int startIdx, int length) {
            // Keep this check: Guard against multi-threaded / batch slicing arithmetic bugs
            if (startIdx + length > dataSize || startIdx + length > output.length) {
                throw new IllegalArgumentException(String.format(
                        "Slice bounds violation: startIdx=%d, length=%d exceeds dataSize=%d or output.length=%d",
                        startIdx, length, dataSize, output.length
                ));
            }

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
                int relativeOffset = blockStart - startIdx;

                evaluateBlock(flatVariables, dataSize, output, startIdx, currentBlockSize, scratch);
            }
        }

        /**
         * Core Column-Major Vectorized Loop Processor (Flat Memory
         * Architecture) Bypasses jagged array pointers to execute at maximum
         * hardware throughput.
         *
         * @param flatVariables Grouped/Contiguous array of variables e.g.
         * [x1,x2..xn, y1,y2..yn, z1,z2..zn] back-to-back. concatenated.
         * @param dataSize The total number of elements per variable row (used
         * for stride offset).
         */
        private void applyBulkInternalNoTiles(double[] flatVariables, int dataSize, double[] output, int startIdx) {
            double[] scratch = FLAT_SCRATCH_STACK.get();

            // 1. Precise scale mapping without guesswork
            long totalRequiredSize = (long) SIMDVectorTurboEvaluator.this.stackDepth * dataSize;
            // 2. Clear fail-fast guard
            if (totalRequiredSize > (Integer.MAX_VALUE - 8)) {
                throw new UnsupportedOperationException("Dataset scale requires off-heap execution or loop tiling block models.");
            }

            int rss = (int) totalRequiredSize;

            // 3. Allocate exactly what is needed, down to the exact slot
            if (scratch == null || scratch.length < rss) {
                scratch = new double[rss];
                FLAT_SCRATCH_STACK.set(scratch);
            }

            evaluateBlock(flatVariables, dataSize, output, startIdx, dataSize, scratch);
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
         * batch window
         * @param tiledExecution {@code true} if execution should be routed
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
        private void applyBulkInternal(double[] flatVariables, int dataSize, double[] output, int startIdx, int length, boolean tiledExecution) {
            if (tiledExecution) {
                applyBulkInternalWithTiles(flatVariables, dataSize, output, startIdx, length);
            } else {
                if (mustUseTiling(dataSize)) {
                    applyBulkInternalWithTiles(flatVariables, dataSize, output, startIdx, length);
                } else {
                    applyBulkInternalNoTiles(flatVariables, dataSize, output, startIdx);
                }
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

            final int BLOCK_SIZE = currentBlockSize;
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
                            scratch[resOffset + k] = Math.pow(scratch[lOffset + k], scratch[rOffset + k]);
                        }
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

        @Override
        public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
            super.applyMatrixKernel(inputs, output, op);
        }

        @Override
        public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) {
            super.applyMatrixKernel(inputs, output, op);
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
