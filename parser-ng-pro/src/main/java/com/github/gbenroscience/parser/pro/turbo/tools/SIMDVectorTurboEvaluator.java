package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator.*;
import static com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator.*;

import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;


/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 */
public class SIMDVectorTurboEvaluator extends VectorTurboEvaluator {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
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
                -> new double[MAX_STACK_DEPTH]
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
        public void applyBulk(double[][] variables, double[] output, boolean tiledExecution, boolean useWorkers) {
            checkError(variables, output);
            if (variables == null || variables.length == 0 || output == null) {
                return;
            }
            int numSamples = variables[0].length;
            if (!useWorkers || numSamples < 2048) {
                applyBulk(variables, output, tiledExecution);
                return;
            }
            double[] flatVars = getOrCreateFlattenBuffer(varCount * numSamples);
            for (int i = 0; i < varCount; i++) {
                System.arraycopy(variables[i], 0, flatVars, i * numSamples, numSamples);
            }
            dispatchToWorkerRing(flatVars, output, numSamples, tiledExecution);
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
        public void applyBulk(double[] flatVariables, double[] output, boolean tiledExecution, boolean useWorkers) {
             checkError(flatVariables, output);
            int numSamples = output.length;
            if (!useWorkers || numSamples < 2048) {
                applyBulkInternal(flatVariables, numSamples, output, 0, numSamples, tiledExecution);
                return;
            }
            dispatchToWorkerRing(flatVariables, output, numSamples, tiledExecution);
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

        
        private void applyBulkInternal(double[] flatVariables, int dataSize, double[] output, int startIdx, int length, boolean tiled) {
            if (flatVariables == null || length <= 0) {
                return;
            }

            final double[] scratch = FLAT_SCRATCH_STACK.get();
            final int endIdx = startIdx + length;

            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                int relativeOffset = blockStart - startIdx;

                evaluateBlock(flatVariables, dataSize, output, startIdx, blockStart, currentBlockSize, scratch);
            }
        }

        /**
         * Core interpretation stream. Leverages explicit AVX-bound Incubator
         * Vector API instructions where possible, falling back to clean
         * primitive loops for auto-vectorization across the tile window.
         */
        private void evaluateBlock(double[] flatVariables,
                int dataSize,
                double[] output,
                int startIdx,
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
            super.applyMatrixKernel(inputs, output,  op);
        }

        @Override
        public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) {
            super.applyMatrixKernel(inputs, output,  op);
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
}
