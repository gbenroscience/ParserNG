package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.utils.HardwareDetector;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;

import java.util.Arrays;
import java.util.concurrent.locks.LockSupport;

/**
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization 
 * with a zero-allocation primitive stack interpreter.
 * Completely eliminates the scalar parser overhead and task object allocations on the hot path.
 */
public class SIMDVectorTurboEvaluator extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final int VLEN = SPECIES.length();
    
    // Opcode Constants from VectorTurbo
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
    private static final int OP_REM = 24;

    // Compilation State
    private final MathExpression.Token[] postfix;
    private int[] opcodes;
    private int[] targetSlots;
    private double[] literalConstants;
    private int varCount;
    private int instructionCount;

    public SIMDVectorTurboEvaluator(MathExpression me) throws Throwable {
        super(me);
        this.postfix = me.getCachedPostfix();
        this.varCount = (me.getVariablesNames() != null) ? me.getVariablesNames().length : 1;
        compileToPrimitiveProgram();
    }

    private void compileToPrimitiveProgram() {
        int len = postfix.length;
        this.opcodes = new int[len];
        this.targetSlots = new int[len];
        this.literalConstants = new double[len];
        this.instructionCount = 0;
        Arrays.fill(targetSlots, -1);

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
                        case '+' -> OP_ADD;
                        case '-' -> OP_SUB;
                        case '*' -> OP_MUL;
                        case '/' -> OP_DIV;
                        case '^' -> OP_POW;
                        case '%' -> OP_REM;
                        default -> throw new IllegalArgumentException("Unknown operator: " + t.opChar);
                    };
                    instructionCount++;
                }
                case MathExpression.Token.METHOD -> {
                    String name = t.name.toLowerCase();
                    opcodes[instructionCount] = switch (name) {
                        case "sin", "sin_rad" -> OP_SIN;
                        case "cos", "cos_rad" -> OP_COS;
                        case "tan", "tan_rad" -> OP_TAN;
                        case "asin", "asin_rad" -> OP_ASIN;
                        case "acos", "acos_rad" -> OP_ACOS;
                        case "atan", "atan_rad" -> OP_ATAN;
                        case "sinh" -> OP_SINH;
                        case "cosh" -> OP_COSH;
                        case "tanh" -> OP_TANH;
                        case "abs" -> OP_ABS;
                        case "exp" -> OP_EXP;
                        case "sqrt" -> OP_SQRT;
                        case "cbrt" -> OP_CBRT;
                        case "log", "ln" -> OP_LOG;
                        case "log10", "lg" -> OP_LOG10;
                        default -> throw new IllegalArgumentException("Unknown function: " + t.name);
                    };
                    instructionCount++;
                }
            }
        }
        this.opcodes = Arrays.copyOf(opcodes, instructionCount);
        this.targetSlots = Arrays.copyOf(targetSlots, instructionCount);
        this.literalConstants = Arrays.copyOf(literalConstants, instructionCount);
    }

    @Override
    public SIMDCompositeExpression compile() throws Throwable {
        return new SIMDVectorCompositeExpression();
    }

    public final class SIMDVectorCompositeExpression implements SIMDCompositeExpression {

        private static final int BLOCK_SIZE = 512; // Aligned with standard L1/L2 cache prefetching
        private static final int MAX_STACK_DEPTH = 64;

        // Zero-allocation thread pools for internal interpretation layers
        private final ThreadLocal<double[]> FLAT_SCRATCH_STACK = ThreadLocal.withInitial(() -> 
            new double[MAX_STACK_DEPTH * BLOCK_SIZE]
        );
        private final ThreadLocal<double[]> SINGLE_SCALAR_BUFFER = ThreadLocal.withInitial(() -> 
            new double[MAX_STACK_DEPTH]
        );
        private final ThreadLocal<double[]> MATRIX_FLATTEN_BUFFER = ThreadLocal.withInitial(() -> 
            new double[0]
        );

        // Coordination Signaling Registers
        private static final int STATE_IDLE = 0;
        private static final int STATE_RUNNING = 1;
        private static final int STATE_FINISHED = 2;

        private final int cores;
        private final WorkerThread[] workers;

        private volatile Thread masterThread;
        private volatile double[] currentFlatVars;
        private volatile double[] currentOutput;
        private volatile int currentDataSize;
        private volatile boolean currentTiled;

        SIMDVectorCompositeExpression() {
            int detected = HardwareDetector.detectPhysicalCores();
            this.cores = Math.min(detected, Runtime.getRuntime().availableProcessors());
            this.workers = new WorkerThread[cores];

            for (int i = 0; i < cores; i++) {
                workers[i] = new WorkerThread(i);
                workers[i].setDaemon(true);
                workers[i].start();
            }
        }

        @Override
        public double applyScalar(double[] variables) {
            double[] stack = SINGLE_SCALAR_BUFFER.get();
            int sp = 0;
            for (int i = 0; i < instructionCount; i++) {
                switch (opcodes[i]) {
                    case OP_CONST -> stack[sp++] = literalConstants[i];
                    case OP_LOAD -> stack[sp++] = variables[targetSlots[i]];
                    case OP_ADD -> { double b = stack[--sp]; stack[sp - 1] += b; }
                    case OP_SUB -> { double b = stack[--sp]; stack[sp - 1] -= b; }
                    case OP_MUL -> { double b = stack[--sp]; stack[sp - 1] *= b; }
                    case OP_DIV -> { double b = stack[--sp]; stack[sp - 1] /= b; }
                    case OP_POW -> { double b = stack[--sp]; stack[sp - 1] = VectorTurboEvaluator.pow(stack[sp - 1], b); }
                    case OP_REM -> { double b = stack[--sp]; stack[sp - 1] = stack[sp - 1] % b; }
                    case OP_SIN -> stack[sp - 1] = Math.sin(stack[sp - 1]);
                    case OP_COS -> stack[sp - 1] = Math.cos(stack[sp - 1]);
                    case OP_TAN -> stack[sp - 1] = Math.tan(stack[sp - 1]);
                    case OP_ASIN -> stack[sp - 1] = Math.asin(stack[sp - 1]);
                    case OP_ACOS -> stack[sp - 1] = Math.acos(stack[sp - 1]);
                    case OP_ATAN -> stack[sp - 1] = Math.atan(stack[sp - 1]);
                    case OP_SINH -> stack[sp - 1] = Math.sinh(stack[sp - 1]);
                    case OP_COSH -> stack[sp - 1] = Math.cosh(stack[sp - 1]);
                    case OP_TANH -> stack[sp - 1] = Math.tanh(stack[sp - 1]);
                    case OP_ABS -> stack[sp - 1] = Math.abs(stack[sp - 1]);
                    case OP_EXP -> stack[sp - 1] = Math.exp(stack[sp - 1]);
                    case OP_SQRT -> stack[sp - 1] = Math.sqrt(stack[sp - 1]);
                    case OP_CBRT -> stack[sp - 1] = Math.cbrt(stack[sp - 1]);
                    case OP_LOG -> stack[sp - 1] = Math.log(stack[sp - 1]);
                    case OP_LOG10 -> stack[sp - 1] = Math.log10(stack[sp - 1]);
                }
            }
            return stack[0];
        }

        @Override public MathExpression.EvalResult apply(double[] vars) { return null; }
        @Override public String checkErrorLogs() { return ""; }
        @Override public TurboExpressionEvaluator getCompiler() { return SIMDVectorTurboEvaluator.this; }

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
            int numSamples = variables[0].length;
            double[] flatVars = getOrCreateFlattenBuffer(varCount * numSamples);
            for (int i = 0; i < varCount; i++) {
                System.arraycopy(variables[i], 0, flatVars, i * numSamples, numSamples);
            }
            applyBulkInternal(flatVars, numSamples, output, 0, numSamples, tiledExecution);
        }

        @Override
        public void applyBulk(double[][] variables, double[] output, int offset, boolean tiledExecution) {
            int numSamples = variables[0].length;
            double[] flatVars = getOrCreateFlattenBuffer(varCount * numSamples);
            for (int i = 0; i < varCount; i++) {
                System.arraycopy(variables[i], 0, flatVars, i * numSamples, numSamples);
            }
            applyBulkInternal(flatVars, numSamples, output, offset, numSamples - offset, tiledExecution);
        }

        @Override
        public void applyBulk(double[][] variables, double[] output, boolean tiledExecution, boolean useWorkers) {
            if (variables == null || variables.length == 0 || output == null) return;
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
            applyBulkInternal(flatVariables, output.length, output, 0, output.length, tiledExecution);
        }

        @Override
        public void applyBulk(double[] flatVariables, double[] output, int offset, boolean tiledExecution) {
            applyBulkInternal(flatVariables, output.length, output, offset, output.length - offset, tiledExecution);
        }

        @Override
        public void applyBulk(double[] flatVariables, double[] output, boolean tiledExecution, boolean useWorkers) {
            int numSamples = output.length;
            if (!useWorkers || numSamples < 2048) {
                applyBulkInternal(flatVariables, numSamples, output, 0, numSamples, tiledExecution);
                return;
            }
            dispatchToWorkerRing(flatVariables, output, numSamples, tiledExecution);
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize, boolean tiledExecution) {
            int numSamples = output.length;
            for (int start = 0; start < numSamples; start += batchSize) {
                int length = Math.min(batchSize, numSamples - start);
                applyBulkInternal(flatVariables, numSamples, output, start, length, tiledExecution);
            }
        }

        private void dispatchToWorkerRing(double[] flatVariables, double[] output, int dataSize, boolean tiledExecution) {
            this.masterThread = Thread.currentThread();
            this.currentFlatVars = flatVariables;
            this.currentOutput = output;
            this.currentDataSize = dataSize;
            this.currentTiled = tiledExecution;

            for (int i = 0; i < cores; i++) {
                workers[i].state = STATE_RUNNING;
                LockSupport.unpark(workers[i]);
            }

            for (int i = 0; i < cores; i++) {
                WorkerThread worker = workers[i];
                while (worker.state != STATE_FINISHED) {
                    LockSupport.park();
                }
                worker.state = STATE_IDLE;
            }

            this.masterThread = null;
            this.currentFlatVars = null;
            this.currentOutput = null;
        }

        private void applyBulkInternal(double[] flatVariables, int dataSize, double[] output, int startIdx, int length, boolean tiled) {
            if (flatVariables == null || length <= 0) return;

            final double[] scratch = FLAT_SCRATCH_STACK.get();
            final int endIdx = startIdx + length;

            for (int blockStart = startIdx; blockStart < endIdx; blockStart += BLOCK_SIZE) {
                final int currentBlockSize = Math.min(BLOCK_SIZE, endIdx - blockStart);
                int relativeOffset = blockStart - startIdx;

                evaluateBlock(flatVariables, dataSize, output, startIdx, blockStart, currentBlockSize, scratch);
            }
        }

        /**
         * Core interpretation stream. Leverages explicit AVX-bound Incubator Vector API instructions 
         * where possible, falling back to clean primitive loops for auto-vectorization across the tile window.
         */
        private void evaluateBlock(double[] flatVariables, int dataSize, double[] output, int startIdx, int blockStart, int currentBlockSize, double[] scratch) {
            int sp = 0;

            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        final double val = literalConstants[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;
                        int k = 0;
                        int upperBound = SPECIES.loopBound(currentBlockSize);
                        DoubleVector valVec = DoubleVector.broadcast(SPECIES, val);
                        for (; k < upperBound; k += VLEN) {
                            valVec.intoArray(scratch, stackOffset + k);
                        }
                        for (; k < currentBlockSize; k++) {
                            scratch[stackOffset + k] = val;
                        }
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;
                        final int flatOffset = (slotIdx * dataSize) + blockStart;
                        System.arraycopy(flatVariables, flatOffset, scratch, stackOffset, currentBlockSize);
                    }

                    case OP_ADD -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        int k = 0;
                        int upperBound = SPECIES.loopBound(currentBlockSize);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.add(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < currentBlockSize; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] + scratch[rOffset + k];
                        }
                    }

                    case OP_SUB -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        int k = 0;
                        int upperBound = SPECIES.loopBound(currentBlockSize);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.sub(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < currentBlockSize; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] - scratch[rOffset + k];
                        }
                    }

                    case OP_MUL -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        int k = 0;
                        int upperBound = SPECIES.loopBound(currentBlockSize);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.mul(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < currentBlockSize; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] * scratch[rOffset + k];
                        }
                    }

                    case OP_DIV -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        int k = 0;
                        int upperBound = SPECIES.loopBound(currentBlockSize);
                        for (; k < upperBound; k += VLEN) {
                            DoubleVector va = DoubleVector.fromArray(SPECIES, scratch, lOffset + k);
                            DoubleVector vb = DoubleVector.fromArray(SPECIES, scratch, rOffset + k);
                            va.div(vb).intoArray(scratch, resOffset + k);
                        }
                        for (; k < currentBlockSize; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] / scratch[rOffset + k];
                        }
                    }

                    case OP_POW -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < currentBlockSize; k++) {
                            scratch[resOffset + k] = Math.pow(scratch[lOffset + k], scratch[rOffset + k]);
                        }
                    }

                    case OP_REM -> {
                        final int rOffset = (--sp) * BLOCK_SIZE;
                        final int lOffset = (--sp) * BLOCK_SIZE;
                        final int resOffset = sp * BLOCK_SIZE;
                        sp++;
                        for (int k = 0; k < currentBlockSize; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] % scratch[rOffset + k];
                        }
                    }

                    case OP_SIN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.sin(scratch[srcOffset + k]);
                    }
                    case OP_COS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.cos(scratch[srcOffset + k]);
                    }
                    case OP_TAN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.tan(scratch[srcOffset + k]);
                    }
                    case OP_ASIN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.asin(scratch[srcOffset + k]);
                    }
                    case OP_ACOS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.acos(scratch[srcOffset + k]);
                    }
                    case OP_ATAN -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.atan(scratch[srcOffset + k]);
                    }
                    case OP_SINH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.sinh(scratch[srcOffset + k]);
                    }
                    case OP_COSH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.cosh(scratch[srcOffset + k]);
                    }
                    case OP_TANH -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.tanh(scratch[srcOffset + k]);
                    }
                    case OP_ABS -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.abs(scratch[srcOffset + k]);
                    }
                    case OP_EXP -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.exp(scratch[srcOffset + k]);
                    }
                    case OP_SQRT -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.sqrt(scratch[srcOffset + k]);
                    }
                    case OP_CBRT -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.cbrt(scratch[srcOffset + k]);
                    }
                    case OP_LOG -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.log(scratch[srcOffset + k]);
                    }
                    case OP_LOG10 -> {
                        final int srcOffset = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < currentBlockSize; k++) scratch[srcOffset + k] = Math.log10(scratch[srcOffset + k]);
                    }
                }
            }

            // Flush the computation results from the top of the stack segment out to the target position
            System.arraycopy(scratch, 0, output, blockStart, currentBlockSize);
        }

        private class WorkerThread extends Thread {
            private final int workerId;
            volatile int state = STATE_IDLE;

            public WorkerThread(int workerId) {
                super("SIMDVectorTurbo-WorkerRing-" + workerId);
                this.workerId = workerId;
            }

            @Override
            public void run() {
                while (true) {
                    while (state != STATE_RUNNING) {
                        LockSupport.park();
                    }

                    double[] flatVars = currentFlatVars;
                    double[] out = currentOutput;
                    int dataSize = currentDataSize;
                    boolean tiled = currentTiled;

                    int chunkSize = (dataSize + cores - 1) / cores;
                    int start = workerId * chunkSize;
                    int end = Math.min(start + chunkSize, dataSize);

                    if (start < end) {
                        applyBulkInternal(flatVars, dataSize, out, start, end - start, tiled);
                    }

                    state = STATE_FINISHED;
                    LockSupport.unpark(masterThread);
                }
            }
        }

        @Override public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) { throw new UnsupportedOperationException(); }
        @Override public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) { throw new UnsupportedOperationException(); }
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