package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.pro.turbo.tools.utils.HardwareDetector;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.concurrent.locks.LockSupport;

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

    public class ThreadLocalBufferPool {
        // ThreadLocal cache to keep threads from colliding in multi-threaded environments

        private static final ThreadLocal<double[]> BUFFER_CACHE = new ThreadLocal<>();

        public static double[] getOrCreateBuffer(int totalSize) {
            double[] buffer = BUFFER_CACHE.get();
            if (buffer == null || buffer.length < totalSize) {
                // Only allocates on the very first pass (or if data size scales up)
                buffer = new double[totalSize];
                BUFFER_CACHE.set(buffer);
            }
            return buffer;
        }
    }

// Thread-local cache allocation helper for flat translation buffers
    private static final ThreadLocal<double[]> FLAT_BUFFER_CACHE = ThreadLocal.withInitial(() -> new double[0]);

    // Opcode Constants
    static final int OP_CONST = 1;
    static final int OP_LOAD = 2;
    static final int OP_ADD = 3;
    static final int OP_SUB = 4;
    static final int OP_MUL = 5;
    static final int OP_DIV = 6;
    static final int OP_POW = 7;

    static final int OP_SIN = 8;
    static final int OP_COS = 9;
    static final int OP_TAN = 10;

    static final int OP_SIN_DEG = 11;
    static final int OP_COS_DEG = 12;
    static final int OP_TAN_DEG = 13;

    static final int OP_SIN_GRAD = 14;
    static final int OP_COS_GRAD = 15;
    static final int OP_TAN_GRAD = 16;

    static final int OP_ASIN = 17;
    static final int OP_ACOS = 18;
    static final int OP_ATAN = 19;

    static final int OP_ASIN_ALT = 20;
    static final int OP_ACOS_ALT = 21;
    static final int OP_ATAN_ALT = 22;

    static final int OP_ASIN_DEG = 23;
    static final int OP_ACOS_DEG = 24;
    static final int OP_ATAN_DEG = 25;

    static final int OP_ASIN_DEG_ALT = 26;
    static final int OP_ACOS_DEG_ALT = 27;
    static final int OP_ATAN_DEG_ALT = 28;

    static final int OP_ASIN_GRAD = 29;
    static final int OP_ACOS_GRAD = 30;
    static final int OP_ATAN_GRAD = 31;

    static final int OP_ASIN_GRAD_ALT = 32;
    static final int OP_ACOS_GRAD_ALT = 33;
    static final int OP_ATAN_GRAD_ALT = 34;

    static final int OP_SEC = 35;
    static final int OP_SEC_DEG = 36;
    static final int OP_SEC_GRAD = 37;

    static final int OP_COSEC = 38;
    static final int OP_COSEC_DEG = 39;
    static final int OP_COSEC_GRAD = 40;

    static final int OP_COT = 41;
    static final int OP_COT_DEG = 42;
    static final int OP_COT_GRAD = 43;

    static final int OP_ARC_SEC = 44;
    static final int OP_ARC_SEC_DEG = 45;
    static final int OP_ARC_SEC_GRAD = 46;

    static final int OP_ARC_COSEC = 47;
    static final int OP_ARC_COSEC_DEG = 48;
    static final int OP_ARC_COSEC_GRAD = 49;

    static final int OP_ARC_COT = 50;
    static final int OP_ARC_COT_DEG = 51;
    static final int OP_ARC_COT_GRAD = 52;

    static final int OP_ARC_SIN_ALT = 53;
    static final int OP_ARC_SIN_ALT_DEG = 54;
    static final int OP_ARC_SIN_ALT_GRAD = 55;

    static final int OP_ARC_COS_ALT = 56;
    static final int OP_ARC_COS_ALT_DEG = 57;
    static final int OP_ARC_COS_ALT_GRAD = 58;

    static final int OP_ARC_TAN_ALT = 59;
    static final int OP_ARC_TAN_ALT_DEG = 60;
    static final int OP_ARC_TAN_ALT_GRAD = 61;

    static final int OP_ARC_SEC_ALT = 62;
    static final int OP_ARC_SEC_ALT_DEG = 63;
    static final int OP_ARC_SEC_ALT_GRAD = 64;

    static final int OP_ARC_COSEC_ALT = 65;
    static final int OP_ARC_COSEC_ALT_DEG = 66;
    static final int OP_ARC_COSEC_ALT_GRAD = 67;

    static final int OP_ARC_COT_ALT = 68;
    static final int OP_ARC_COT_ALT_DEG = 69;
    static final int OP_ARC_COT_ALT_GRAD = 70;

    static final int OP_SINH = 71;
    static final int OP_COSH = 72;
    static final int OP_TANH = 73;

    static final int OP_ASINH = 74;
    static final int OP_ACOSH = 75;
    static final int OP_ATANH = 76;

    static final int OP_ASINH_ALT = 77;
    static final int OP_ACOSH_ALT = 78;
    static final int OP_ATANH_ALT = 79;

    static final int OP_ABS = 80;
    static final int OP_EXP = 81;
    static final int OP_SQRT = 82;
    static final int OP_CBRT = 83;
    static final int OP_LOG = 84;
    static final int OP_LOG10 = 85;
    static final int OP_VMA = 86;
    static final int OP_REM = 87;
    static final int OP_IF = 88;
    static final int OP_GT = 89;
    static final int OP_LT = 90;
    static final int OP_EQ = 91;
    static final int OP_NE = 92;
    static final int OP_GE = 93;
    static final int OP_LE = 94;

    // Pre-allocated compilation state
    protected MathExpression.Token[] postfix;
    protected final MethodHandle compiledScalarHandle;
    protected int[] opcodes;
    protected int[] targetSlots;
    protected double[] literalConstants;
    protected int[] argumentCount;

    protected int varCount;
    protected int instructionCount;
    protected KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator(MathExpression me) throws Throwable {
        super(me);
        this.postfix = me.getCachedPostfix();
        this.varCount = me.getVariablesNames().length;
        this.compiledScalarHandle = compileScalar(postfix);
        compileToPrimitiveProgram();
    }

    protected final void compileToPrimitiveProgram() {
        int len = postfix.length;
        this.opcodes = new int[len];
        this.targetSlots = new int[len];
        this.literalConstants = new double[len];
        this.argumentCount = new int[len];
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
                case MathExpression.Token.METHOD -> {
                    argumentCount[instructionCount] = t.arity;
                    String name = t.name.toLowerCase();

                    opcodes[instructionCount] = switch (name) {
                        // Core Math Functions
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
                        case "vma", "fma" ->
                            OP_VMA;
                        case "rem", "mod" ->
                            OP_REM;
                        case "if" ->
                            OP_IF;

                        // Relational Word Aliases
                        case "gt" ->
                            OP_GT;
                        case "lt" ->
                            OP_LT;
                        case "eq" ->
                            OP_EQ;
                        case "ne" ->
                            OP_NE;
                        case "ge" ->
                            OP_GE;
                        case "le" ->
                            OP_LE;

                        // Standard Trig (Radians)
                        case "sin", "sin_rad" ->
                            OP_SIN;
                        case "cos", "cos_rad" ->
                            OP_COS;
                        case "tan", "tan_rad" ->
                            OP_TAN;

                        // Standard Trig (Degrees)
                        case "sin_deg", "sind" ->
                            OP_SIN_DEG;
                        case "cos_deg", "cosd" ->
                            OP_COS_DEG;
                        case "tan_deg", "tand" ->
                            OP_TAN_DEG;

                        // Standard Trig (Gradians)
                        case "sin_grad", "sing" ->
                            OP_SIN_GRAD;
                        case "cos_grad", "cosg" ->
                            OP_COS_GRAD;
                        case "tan_grad", "tang" ->
                            OP_TAN_GRAD;

                        // Inverse Trig (Radians) [e.g., "sin-¹"]
                        case "sin-¹", "sin-¹_rad", "arcsin" ->
                            OP_ASIN;
                        case "cos-¹", "cos-¹_rad", "arccos" ->
                            OP_ACOS;
                        case "tan-¹", "tan-¹_rad", "arctan" ->
                            OP_ATAN;

                        // Inverse Trig (Degrees)
                        case "sin-¹_deg", "arcsin_deg" ->
                            OP_ASIN_DEG;
                        case "cos-¹_deg", "arccos_deg" ->
                            OP_ACOS_DEG;
                        case "tan-¹_deg", "arctan_deg" ->
                            OP_ATAN_DEG;

                        // Inverse Trig (Gradians)
                        case "sin-¹_grad", "arcsin_grad" ->
                            OP_ASIN_GRAD;
                        case "cos-¹_grad", "arccos_grad" ->
                            OP_ACOS_GRAD;
                        case "tan-¹_grad", "arctan_grad" ->
                            OP_ATAN_GRAD;

                        // Inverse Trig Alt (Short Prefix "asin" Form) - Radians
                        case "asin", "asin_rad" ->
                            OP_ASIN_ALT;
                        case "acos", "acos_rad" ->
                            OP_ACOS_ALT;
                        case "atan", "atan_rad" ->
                            OP_ATAN_ALT;

                        // Inverse Trig Alt (Short Prefix) - Degrees
                        case "asin_deg", "asind" ->
                            OP_ASIN_DEG_ALT;
                        case "acos_deg", "acosd" ->
                            OP_ACOS_DEG_ALT;
                        case "atan_deg", "atand" ->
                            OP_ATAN_DEG_ALT;

                        // Inverse Trig Alt (Short Prefix) - Gradians
                        case "asin_grad", "asing" ->
                            OP_ASIN_GRAD_ALT;
                        case "acos_grad", "acosg" ->
                            OP_ACOS_GRAD_ALT;
                        case "atan_grad", "atang" ->
                            OP_ATAN_GRAD_ALT;

                        // Reciprocal Trig (Secant, Cosecant, Cotangent)
                        case "sec", "sec_rad" ->
                            OP_SEC;
                        case "sec_deg", "secd" ->
                            OP_SEC_DEG;
                        case "sec_grad" ->
                            OP_SEC_GRAD;
                        case "cosec", "csc", "csc_rad" ->
                            OP_COSEC;
                        case "cosec_deg", "cscd" ->
                            OP_COSEC_DEG;
                        case "cosec_grad" ->
                            OP_COSEC_GRAD;
                        case "cot", "cot_rad" ->
                            OP_COT;
                        case "cot_deg", "cotd" ->
                            OP_COT_DEG;
                        case "cot_grad" ->
                            OP_COT_GRAD;

                        // Inverse Reciprocal Trig (Standard Form)
                        case "sec-¹", "sec-¹_rad", "arcsec" ->
                            OP_ARC_SEC;
                        case "sec-¹_deg", "arcsec_deg" ->
                            OP_ARC_SEC_DEG;
                        case "sec-¹_grad", "arcsec_grad" ->
                            OP_ARC_SEC_GRAD;
                        case "csc-¹", "csc-¹_rad", "arccsc" ->
                            OP_ARC_COSEC;
                        case "csc-¹_deg", "arccsc_deg" ->
                            OP_ARC_COSEC_DEG;
                        case "csc-¹_grad", "arccsc_grad" ->
                            OP_ARC_COSEC_GRAD;
                        case "cot-¹", "cot-¹_rad", "arccot" ->
                            OP_ARC_COT;
                        case "cot-¹_deg", "arccot_deg" ->
                            OP_ARC_COT_DEG;
                        case "cot-¹_grad", "arccot_grad" ->
                            OP_ARC_COT_GRAD;

                        // Inverse Trig Explicit Alt Chains (Opcodes 53 - 61)
                        case "arc_sin_alt" ->
                            OP_ARC_SIN_ALT;
                        case "arc_sin_alt_deg" ->
                            OP_ARC_SIN_ALT_DEG;
                        case "arc_sin_alt_grad" ->
                            OP_ARC_SIN_ALT_GRAD;
                        case "arc_cos_alt" ->
                            OP_ARC_COS_ALT;
                        case "arc_cos_alt_deg" ->
                            OP_ARC_COS_ALT_DEG;
                        case "arc_cos_alt_grad" ->
                            OP_ARC_COS_ALT_GRAD;
                        case "arc_tan_alt" ->
                            OP_ARC_TAN_ALT;
                        case "arc_tan_alt_deg" ->
                            OP_ARC_TAN_ALT_DEG;
                        case "arc_tan_alt_grad" ->
                            OP_ARC_TAN_ALT_GRAD;

                        // Inverse Reciprocal Explicit Alt Chains (Opcodes 62 - 70)
                        case "asec", "asec_rad", "arc_sec_alt" ->
                            OP_ARC_SEC_ALT;
                        case "asec_deg", "arc_sec_alt_deg" ->
                            OP_ARC_SEC_ALT_DEG;
                        case "asec_grad", "arc_sec_alt_grad" ->
                            OP_ARC_SEC_ALT_GRAD;
                        case "acsc", "acsc_rad", "arc_cosec_alt" ->
                            OP_ARC_COSEC_ALT;
                        case "acsc_deg", "arc_cosec_alt_deg" ->
                            OP_ARC_COSEC_ALT_DEG;
                        case "acsc_grad", "arc_cosec_alt_grad" ->
                            OP_ARC_COSEC_ALT_GRAD;
                        case "acot", "acot_rad", "arc_cot_alt" ->
                            OP_ARC_COT_ALT;
                        case "acot_deg", "arc_cot_alt_deg" ->
                            OP_ARC_COT_ALT_DEG;
                        case "acot_grad", "arc_cot_alt_grad" ->
                            OP_ARC_COT_GRAD;

                        // Hyperbolic Functions
                        case "sinh" ->
                            OP_SINH;
                        case "cosh" ->
                            OP_COSH;
                        case "tanh" ->
                            OP_TANH;

                        // Hyperbolic Inverses (Standard Form)
                        case "sinh-¹", "arcsinh" ->
                            OP_ASINH;
                        case "cosh-¹", "arccosh" ->
                            OP_ACOSH;
                        case "tanh-¹", "arctanh" ->
                            OP_ATANH;

                        // Hyperbolic Inverses (Alt Short Prefix Form)
                        case "asinh" ->
                            OP_ASINH_ALT;
                        case "acosh" ->
                            OP_ACOSH_ALT;
                        case "atanh" ->
                            OP_ATANH_ALT;

                        default ->
                            throw new IllegalArgumentException("Unknown function: " + t.name);
                    };
                    instructionCount++;
                }
            }
        }

        // CRITICAL: Trim all tracking arrays to matching exact instruction boundary
        this.opcodes = Arrays.copyOf(opcodes, instructionCount);
        this.targetSlots = Arrays.copyOf(targetSlots, instructionCount);
        this.literalConstants = Arrays.copyOf(literalConstants, instructionCount);
        this.argumentCount = Arrays.copyOf(argumentCount, instructionCount);
    }

    public int getVarCount() {
        return varCount;
    }

    @Override
    public SIMDCompositeExpression compile() throws Throwable {
        return new BatchedVectorCompositeExpression(compiledScalarHandle, opcodes, targetSlots,
                literalConstants, instructionCount, varCount);
    }


    public class BatchedVectorCompositeExpression implements SIMDCompositeExpression {

        // Optimized block size designed to comfortably fit into standard CPU L1/L2 caches (2048 * 8 bytes = 16KB)
        protected static final int BLOCK_SIZE = 256;
        protected static final int MAX_STACK_DEPTH = 64;

        private final MethodHandle scalarHandle;
        private final int[] opcodes;
        private final int[] targetSlots;
        private final double[] literalConstants;
        private final int instructionCount;
        private int varCount;

        ////NEW FIELDS
        // --- ZERO-ALLOCATION MULTI-THREADING SUBSYSTEM ---
        protected static final int STATE_IDLE = 0;
        protected static final int STATE_RUNNING = 1;
        protected static final int STATE_FINISHED = 2;


        protected int cores;
        protected WorkerThread[] workers;

        // Volatile transfer registers (allows zero-heap overhead parameter passing)
        protected volatile Thread masterThread;
        protected volatile double[] currentFlatVars;
        protected volatile double[] currentOutput;
        protected volatile int currentDataSize;
        protected volatile boolean currentTiled;

        /**
         * Zero-allocation thread pools for internal interpretation layers
         * Allocates a perfectly flat, contiguous scratch block (64 nested stack
         * frames * 2048 block size)
         */
        protected static final ThreadLocal<double[]> FLAT_SCRATCH_STACK = ThreadLocal.withInitial(()
                -> new double[MAX_STACK_DEPTH * BLOCK_SIZE]
        );

        public int[] getOpcodes() {
            return opcodes;
        }

        public int[] getTargetSlots() {
            return targetSlots;
        }

        BatchedVectorCompositeExpression(MethodHandle handle, int[] ops, int[] targetSlots,
                double[] consts, int count, int varCount) {
            this.scalarHandle = handle;
            // DEFENSIVE: Make copies to isolate from VectorTurboEvaluator mutations
            this.opcodes = Arrays.copyOf(ops, ops.length);
            this.targetSlots = Arrays.copyOf(targetSlots, targetSlots.length);
            this.literalConstants = Arrays.copyOf(consts, consts.length);
            this.instructionCount = count;
            this.varCount = varCount;

            // Initialize the dedicated worker threads ONCE when this expression is compiled/created
            int dcores = HardwareDetector.detectPhysicalCores();
            // Optional: cap to logical in case override lies
            this.cores = Math.min(dcores, Runtime.getRuntime().availableProcessors());
            this.workers = new WorkerThread[cores];

            for (int i = 0; i < cores; i++) {
                workers[i] = new WorkerThread(i);
                workers[i].setDaemon(true); // Ensures the JVM can shutdown cleanly
                workers[i].start();
            }
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

        /**
         * Private encapsulated Inner Class to handle the raw CPU-bound math
         * slices
         */
        protected class WorkerThread extends Thread {

            private final int workerId;
            volatile int state = STATE_IDLE;

            public WorkerThread(int workerId) {
                super("VectorTurbo-Worker-" + workerId);
                this.workerId = workerId;
            }

            @Override
            public void run() {
                while (true) {
                    // Spin/Park until the master thread throws work into the registers
                    while (state != STATE_RUNNING) {
                        LockSupport.park();
                    }

                    // Volatile reads establish a memory barrier (Happens-Before guarantee)
                    double[] flatVars = currentFlatVars;
                    double[] out = currentOutput;
                    int dataSize = currentDataSize;
                    boolean tiled = currentTiled;

                    // Execute your exact coarse-grained thread slicing logic
                    int chunkSize = (dataSize + cores - 1) / cores;
                    int start = workerId * chunkSize;
                    int end = Math.min(start + chunkSize, dataSize);

                    if (start < end) {
                        applyBulkInternal(flatVars, dataSize, out, start, end - start, true);
                    }

                    // Signal the master thread that this core is finished
                    state = STATE_FINISHED;
                    LockSupport.unpark(masterThread);
                }
            }
        }

        protected void checkError(double[][] variables, double[] output) {
             int numSamples = variables!=null && variables.length>0&&output!=null&&output.length>0 ? variables[0].length : -1;
            int stride = this.varCount;
            if (stride != variables.length || numSamples != output.length) {
                throw new IllegalStateException(String.format("varCount mismatch[stride=%d, variables-count-from-input=%d] || array sizes not correct(elements-per-variable=%d vs output-array-size=%d)", stride,variables.length, numSamples,output.length));
            }
        }

        protected void checkError(double[] flatVariables, double[] output) {
            int totalSamples = flatVariables!=null && flatVariables.length>0&&output!=null&&output.length>0 ? flatVariables.length : -1;
            int stride = this.varCount;
            if (totalSamples != stride*output.length) {
                throw new IllegalStateException(String.format("array sizes not correct[totalSamples=%d vs computed(var-count*output-array-size)=%d]", 
                        totalSamples,stride*output.length));
            }
  
        }

        //////////////////////////////////////////////////////////////////
        ///                                                            ///
        ///          Uses double[][] Arrays For Convenience          ///
        ///         Speed may drop by as much as 4ns                   ///
        ///                                                            ///
        //////////////////////////////////////////////////////////////////
        
        @Override
        public void applyBulk(double[][] variables, double[] output, boolean useBlocks) {
            checkError(variables, output);
            int numSamples = variables[0].length;
            int stride = this.varCount;

            // Rent buffer. Don't zero it. We overwrite all of it.
            // 
            double[] flatVariables = ThreadLocalBufferPool.getOrCreateBuffer(stride * numSamples);
            for (int i = 0; i < stride; i++) {
                System.arraycopy(variables[i], 0, flatVariables, i * numSamples, numSamples);
            }
            // FIX: Pass dataSize explicitly. Don't let applyBulk guess from flatVariables.length
            applyBulkInternal(flatVariables, numSamples, output, 0, numSamples, useBlocks);
        }

    
        @Override
        public void applyBulkParallel(double[][] variables, double[] output) {
            checkError(variables, output);

            final int numSamples = variables[0].length;
            final int stride = this.varCount;
          

            // Zero-allocation buffer retrieval from your thread-local pool
            double[] flatVariables = ThreadLocalBufferPool.getOrCreateBuffer(stride * numSamples);

            // Flatten 2D array to 1D (sequential block transfers)
            for (int i = 0; i < stride; i++) {
                System.arraycopy(variables[i], 0, flatVariables, i * numSamples, numSamples);
            }

            // Sequential bypass for small workloads
            if (numSamples < 2048) {
                applyBulkInternal(flatVariables, numSamples, output, 0, numSamples, false);
                return;
            }

            // Coordinate via the EXACT SAME zero-alloc threading subsystem
            dispatchToWorkerRing(flatVariables, output, numSamples, true);
        }
        
   

        /**
         *  Centralized coordination mechanism to eliminate code duplication
         * @param flatVariables
         * @param output
         * @param dataSize
         * @param useBlocks 
         */
        protected void dispatchToWorkerRing(double[] flatVariables, double[] output, int dataSize, boolean useBlocks) {
            // 1. Publish parameters to volatile registers (No allocations)
            this.masterThread = Thread.currentThread();
            this.currentFlatVars = flatVariables;
            this.currentOutput = output;
            this.currentDataSize = dataSize;
            this.currentTiled = useBlocks;

            // 2. Wake up the pre-allocated worker threads via OS permits
            for (int i = 0; i < cores; i++) {
                workers[i].state = STATE_RUNNING;
                LockSupport.unpark(workers[i]);
            }

            // 3. Wait loop (0 bytes allocated on heap while sleeping)
            for (int i = 0; i < cores; i++) {
                WorkerThread worker = workers[i];
                while (worker.state != STATE_FINISHED) {
                    LockSupport.park();
                }
                worker.state = STATE_IDLE;
            }

            // 4. Memory leak protection (Comment out lines below IF benchmarking this specific class long-term)
            this.masterThread = null;
            this.currentFlatVars = null;
            this.currentOutput = null;
        }

        @Override
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize, boolean useBlocks) {
            checkError(variables, output);
            int dataSize = variables[0].length;
            int stride = this.varCount;
            double[] flatVariables = ThreadLocalBufferPool.getOrCreateBuffer(stride * dataSize);
            for (int i = 0; i < stride; i++) {
                System.arraycopy(variables[i], 0, flatVariables, i * dataSize, dataSize);
            }
            // FIX: Don't delegate to flat overload. Loop here.
            for (int start = 0; start < dataSize; start += batchSize) {
                int length = Math.min(batchSize, dataSize - start);
                applyBulkInternal(flatVariables, dataSize, output, start, length, useBlocks);
            }
        }

        //////////////////////////////////////////////////////////////////////////////
        ///                      double[][] Arrays ends                            ///
        ///               Uses Flat interleaved Arrays for speed(0.8ns-1.8ns)      ///
        ///                  [x1,x2..xn, y1,y2..yn, z1,z2..zn]                     ///
        ///                                                                        ///                   
        //////////////////////////////////////////////////////////////////////////////
        
     
        
        
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
        private void applyBulkInternalWithBlocks(double[] flatVariables, int dataSize, double[] output, int startIdx, int length) {
            // Top-level API boundary validations (executed once per pipeline request)
        

            if (dataSize * this.varCount > flatVariables.length) {
                throw new IllegalArgumentException("flatVariables too small: need " + (dataSize * varCount) + " got " + flatVariables.length);
            }

            final double[] scratch = FLAT_SCRATCH_STACK.get();
            final int endIdx = startIdx + length;
            // 1. SHORT-CIRCUIT: Small datasets bypass tiling infrastructure entirely
            if (length <= BLOCK_SIZE) {
                evaluateTile(flatVariables, dataSize, output, startIdx, length, scratch);
            } // 2. TILING STRATEGY: Process large datasets in cache-aligned blocks
            else {
                for (int tileStart = startIdx; tileStart < endIdx; tileStart += BLOCK_SIZE) {
                    final int currentTileSize = Math.min(BLOCK_SIZE, endIdx - tileStart);
                    evaluateTile(flatVariables, dataSize, output, tileStart, currentTileSize, scratch);
                }
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
        private void applyBulkInternalNoBlocks(double[] flatVariables, int dataSize, double[] output,
                int startIdx) {
            // Retrieve the completely flat contiguous scratch space for maximum CPU cache prefetching
            final double[] scratch = FLAT_SCRATCH_STACK.get();

            evaluateTile(flatVariables,
                    dataSize,
                    output,
                    startIdx,
                    dataSize,
                    scratch);
        }

        /**
         * Core column-major vectorized loop processor utilizing a flat memory
         * architecture. Bypasses pointer-chasing and allocation overhead
         * inherent to jagged multidimensional arrays (e.g., {@code double[][]})
         * to execute loops at maximum hardware throughput.
         * <p>
         * Based on the {@code useBlocks} execution parameter, this method
         * delegates to either a cache-conscious useBlocks implementation or a
         * direct flat streaming evaluation strategy.
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
         * @param useBlocks {@code true} if execution should be routed through a
         * useBlocks memory pattern optimized for L1/L2 cache locality;
         * {@code false} for flat, non-useBlocks bulk processing
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
        private void applyBulkInternal(double[] flatVariables, int dataSize, double[] output, int startIdx, int length, boolean useBlocks) {
            if (useBlocks) {
                applyBulkInternalWithBlocks(flatVariables, dataSize, output, startIdx, length);
            } else {
                applyBulkInternalNoBlocks(flatVariables, dataSize, output, startIdx);
            }
        }

        /**
         *
         * @param flatVariables
         * @param dataSize
         * @param output
         * @param startIdx
         * @param tileStart
         * @param currentTileSize
         * @param scratch
         */
        //@jdk.internal.vm.annotation.ForceInline
        private void evaluateTile(double[] flatVariables,
                int dataSize,
                double[] output,
                int tileStart,
                int currentTileSize,
                double[] scratch) {

            final int n = currentTileSize; // hoist for C2 unroll
            int sp = 0;

            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        final double val = literalConstants[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[stackOffset + k] = val;
                        }
                    }

                    case OP_LOAD -> {
                        final int slotIdx = targetSlots[instIdx];
                        final int stackOffset = sp * BLOCK_SIZE;
                        sp++;
                        final int flatOffset = (slotIdx * dataSize) + tileStart;

                        System.arraycopy(flatVariables, flatOffset, scratch, stackOffset, n);
                    }

                    case OP_ADD -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] + scratch[rOffset + k];
                        }
                    }

                    case OP_SUB -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] - scratch[rOffset + k];
                        }
                    }

                    case OP_MUL -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] * scratch[rOffset + k];
                        }
                    }

                    case OP_DIV -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] / scratch[rOffset + k];
                        }
                    }

                    case OP_POW -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = pow(scratch[lOffset + k], scratch[rOffset + k]);
                        }
                    }

                    case OP_REM -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;

                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] % scratch[rOffset + k];
                        }
                    }

                    // Unary ops - in-place, no sp shuffle
                    case OP_SIN -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.sin(scratch[base + k]);
                        }
                    }

                    case OP_COS -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.cos(scratch[base + k]);
                        }
                    }

                    case OP_TAN -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.tan(scratch[base + k]);
                        }
                    }

                    case OP_ASIN, OP_ASIN_ALT, OP_ARC_SIN_ALT -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        MethodSack.asinRad(base, n, scratch);
                    }

                    case OP_ACOS, OP_ACOS_ALT, OP_ARC_COS_ALT -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        MethodSack.acosRad(base, n, scratch);
                    }

                    case OP_ATAN, OP_ATAN_ALT, OP_ARC_TAN_ALT -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        MethodSack.atanRad(base, n, scratch);
                    }

                    case OP_SINH -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.sinh(scratch[base + k]);
                        }
                    }

                    case OP_COSH -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.cosh(scratch[base + k]);
                        }
                    }

                    case OP_TANH -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.tanh(scratch[base + k]);
                        }
                    }

                    case OP_ABS -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.abs(scratch[base + k]);
                        }
                    }

                    case OP_EXP -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.exp(scratch[base + k]);
                        }
                    }

                    case OP_SQRT -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.sqrt(scratch[base + k]);
                        }
                    }

                    case OP_CBRT -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.cbrt(scratch[base + k]);
                        }
                    }

                    case OP_LOG -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.log(scratch[base + k]);
                        }
                    }

                    case OP_LOG10 -> {
                        final int base = (sp - 1) * BLOCK_SIZE;
                        for (int k = 0; k < n; k++) {
                            scratch[base + k] = Math.log10(scratch[base + k]);
                        }
                    }

                    // Comparisons
                    case OP_GT -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] > scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_LT -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] < scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_EQ -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] == scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_NE -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] != scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_GE -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] >= scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_LE -> {
                        sp -= 2;
                        final int base = sp * BLOCK_SIZE;
                        final int lOffset = base;
                        final int rOffset = base + BLOCK_SIZE;
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[lOffset + k] <= scratch[rOffset + k] ? 1.0 : 0.0;
                        }
                    }

                    case OP_VMA -> {
                        sp -= 3;
                        final int base = sp * BLOCK_SIZE;
                        final int aOffset = base;
                        final int bOffset = base + BLOCK_SIZE;
                        final int cOffset = base + (2 * BLOCK_SIZE);
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = scratch[aOffset + k] * scratch[bOffset + k] + scratch[cOffset + k];
                        }
                    }

                    case OP_IF -> {
                        sp -= 3;
                        final int base = sp * BLOCK_SIZE;
                        final int condOffset = base;
                        final int trueOffset = base + BLOCK_SIZE;
                        final int falseOffset = base + (2 * BLOCK_SIZE);
                        final int resOffset = base;
                        sp++;
                        for (int k = 0; k < n; k++) {
                            scratch[resOffset + k] = (scratch[condOffset + k] != 0.0) ? scratch[trueOffset + k] : scratch[falseOffset + k];
                        }
                    }

                    // --- NEW DEGREE / GRADIAN TRIG VARIANTS ---
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

                    // --- NEW INVERSE DEGREE / GRADIAN VARIANTS ---
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

                    // --- NEW RECIPROCAL TRIG (SEC, CSC, COT) VARIANTS ---
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

                    // --- NEW INVERSE RECIPROCAL TRIG VARIANTS ---
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

                    // --- NEW HYPERBOLIC INVERSES ---
                    case OP_ASINH, OP_ASINH_ALT ->
                        MethodSack.asinh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ACOSH, OP_ACOSH_ALT ->
                        MethodSack.acosh((sp - 1) * BLOCK_SIZE, n, scratch);
                    case OP_ATANH, OP_ATANH_ALT ->
                        MethodSack.atanh((sp - 1) * BLOCK_SIZE, n, scratch);

                    default ->
                        throw new UnsupportedOperationException("Unknown opcode: " + opcode);
                }
            }

            System.arraycopy(scratch, 0, output, tileStart, n);
        }

        /**
         * OVERLOAD 1: Flat 1D Array Input (Guaranteed 0 B/op)
         */
        @Override
        public void applyBulkParallel(double[] flatVariables, double[] output) {
            checkError(flatVariables, output);

            final int length = output.length;

            if (length < 2048) {
                applyBulkInternal(flatVariables, length, output, 0, length, false);
                return;
            }

            // Coordinate via the zero-alloc threading subsystem
            dispatchToWorkerRing(flatVariables, output, length, true);
        }

        @Override
        public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize, boolean useBlocks) {
            checkError(flatVariables, output);
            if (batchSize <= 0) {
                return;
            }

            final int totalLength = output.length; // FIX: use output.length
            final int dataSize = totalLength;      // FIX: each var has totalLength elements

            for (int start = 0; start < totalLength; start += batchSize) {
                int length = Math.min(batchSize, totalLength - start);
                applyBulkInternal(flatVariables, dataSize, output, start, length, useBlocks);
            }
        }

        @Override
        public void applyBulk(double[] flatVariables, double[] output, boolean useBlocks) {
            checkError(flatVariables, output);
            final int length = output.length; // This is the contract: output.length = elements to compute
            applyBulkInternal(flatVariables, length, output, 0, length, useBlocks);
        }

   

        ///////////////////////////////////////////////////////////
        ///                    MATRIX KERNELS                   ///
        ///////////////////////////////////////////////////////////
        
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
                    float[] src = inputs[0].data;
                    int srcOff = inputs[0].offset;
                    int len = inputs[0].rows * inputs[0].cols;
                    float scale = inputs[1].data[inputs[1].offset];
                    byte[] dst = output.asByteArray();
                    KernelsInt8.quantize_f32_to_i8(src, srcOff, len, dst, output.offset, scale);
                }
                case "q8_dequant" -> {
                    byte[] src = inputs[0].asByteArray();
                    int srcOff = inputs[0].offset;
                    int len = inputs[0].rows * inputs[0].cols;
                    float scale = inputs[1].data[inputs[1].offset];
                    float[] dst = output.data;
                    KernelsInt8.dequant_i8_to_f32(src, srcOff, len, dst, output.offset, scale);
                }
                case "q8_absmax_quant" -> {
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
                    inputs[1].data[inputs[1].offset] = scale;
                }
                case "rope_split" -> {
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
                    float scale = (float) inputs[1].data[inputs[1].offset];
                    float[] src = inputs[0].data;
                    float[] dst = output.data;
                    int len = inputs[0].rows * inputs[0].cols;
                    KernelsFloat.accumulate_v_f32(dst, output.offset, src, inputs[0].offset, len, scale);
                }
                case "matmul_1xn_axpy" -> {
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
                    float eps = (float) inputs[2].data[inputs[2].offset];
                    FlatMatrixF.rmsNorm(inputs[0], inputs[1], output, eps);
                }
                case "layer_norm" -> {
                    float eps = (float) inputs[3].data[inputs[3].offset];
                    FlatMatrixF.layerNorm(inputs[0], inputs[1], inputs[2], output, eps);
                }
                case "matmul_bias_silu" ->
                    FlatMatrixF.matmulAddBiasSilu(inputs[0], inputs[1], inputs[2], output);
                case "swiglu" -> {
                    FlatMatrixF.swiGLU(inputs[0], inputs[1], output);
                }
                case "mha_attention" -> {
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
     * Turbo-flattens a jagged array into a destination array. Reuses the
     * provided 'flatBuffer' to ensure ZERO heap allocation.
     */
    public static final void flatten(double[][] jagged, double[] flatBuffer) {
        int offset = 0;
        for (int i = 0; i < jagged.length; i++) {
            int length = jagged[i].length;
            // Native-level memory copy - this is the fastest way to move data in JVM
            System.arraycopy(jagged[i], 0, flatBuffer, offset, length);
            offset += length;
        }
    }

    private double[] getWorkingFlatBuffer(int requiredSize) {
        double[] buf = FLAT_BUFFER_CACHE.get();
        if (buf.length < requiredSize) {
            buf = new double[requiredSize];
            FLAT_BUFFER_CACHE.set(buf);
        }
        return buf;
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
    protected static double pow(double base, double exp) {
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

    static final class MethodSack {

        private static final double DEG_TO_RAD = Math.PI / 180.0;
        private static final double RAD_TO_DEG = 180.0 / Math.PI;
        private static final double GRAD_TO_RAD = Math.PI / 200.0;
        private static final double RAD_TO_GRAD = 200.0 / Math.PI;

        private MethodSack() {
        } // no instantiation

        static void if3(int base, int tileN, double[] s, int block) {
            final int cond = base + block;
            final int trueVal = base + 2 * block;
            final int falseVal = base + 3 * block;
            final int res = base;
            for (int k = 0; k < tileN; k++) {
                double c = s[cond + k];
                s[res + k] = c != 0.0 && !Double.isNaN(c) ? s[trueVal + k] : s[falseVal + k];
            }
        }

        // --- Standard Radians Fallbacks ---
        static void asinRad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.asin(s[base + k]);
            }
        }

        static void acosRad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.acos(s[base + k]);
            }
        }

        static void atanRad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.atan(s[base + k]);
            }
        }

        // --- Degree Conversions ---
        static void sinDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.sin(s[base + k] * DEG_TO_RAD);
            }
        }

        static void cosDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.cos(s[base + k] * DEG_TO_RAD);
            }
        }

        static void tanDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.tan(s[base + k] * DEG_TO_RAD);
            }
        }

        static void asinDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.asin(s[base + k]) * RAD_TO_DEG;
            }
        }

        static void acosDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.acos(s[base + k]) * RAD_TO_DEG;
            }
        }

        static void atanDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.atan(s[base + k]) * RAD_TO_DEG;
            }
        }

        // --- Gradian Conversions ---
        static void sinGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.sin(s[base + k] * GRAD_TO_RAD);
            }
        }

        static void cosGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.cos(s[base + k] * GRAD_TO_RAD);
            }
        }

        static void tanGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.tan(s[base + k] * GRAD_TO_RAD);
            }
        }

        static void asinGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.asin(s[base + k]) * RAD_TO_GRAD;
            }
        }

        static void acosGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.acos(s[base + k]) * RAD_TO_GRAD;
            }
        }

        static void atanGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.atan(s[base + k]) * RAD_TO_GRAD;
            }
        }

        // --- Reciprocals (Secant, Cosecant, Cotangent) ---
        static void sec(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.cos(s[base + k]);
            }
        }

        static void secDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.cos(s[base + k] * DEG_TO_RAD);
            }
        }

        static void secGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.cos(s[base + k] * GRAD_TO_RAD);
            }
        }

        static void csc(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.sin(s[base + k]);
            }
        }

        static void cscDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.sin(s[base + k] * DEG_TO_RAD);
            }
        }

        static void cscGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.sin(s[base + k] * GRAD_TO_RAD);
            }
        }

        static void cot(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.tan(s[base + k]);
            }
        }

        static void cotDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.tan(s[base + k] * DEG_TO_RAD);
            }
        }

        static void cotGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = 1.0 / Math.tan(s[base + k] * GRAD_TO_RAD);
            }
        }

        // --- Inverse Reciprocals ---
        static void asec(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.acos(1.0 / s[base + k]);
            }
        }

        static void asecDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.acos(1.0 / s[base + k]) * RAD_TO_DEG;
            }
        }

        static void asecGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.acos(1.0 / s[base + k]) * RAD_TO_GRAD;
            }
        }

        static void acsc(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.asin(1.0 / s[base + k]);
            }
        }

        static void acscDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.asin(1.0 / s[base + k]) * RAD_TO_DEG;
            }
        }

        static void acscGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.asin(1.0 / s[base + k]) * RAD_TO_GRAD;
            }
        }

        static void acot(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.atan(1.0 / s[base + k]);
            }
        }

        static void acotDeg(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.atan(1.0 / s[base + k]) * RAD_TO_DEG;
            }
        }

        static void acotGrad(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                s[base + k] = Math.atan(1.0 / s[base + k]) * RAD_TO_GRAD;
            }
        }

        // --- Hyperbolic Inverses ---
        static void asinh(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                final double val = s[base + k];
                s[base + k] = Math.log(val + Math.sqrt(val * val + 1.0));
            }
        }

        static void acosh(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                final double val = s[base + k];
                s[base + k] = Math.log(val + Math.sqrt(val * val - 1.0));
            }
        }

        static void atanh(int base, int n, double[] s) {
            for (int k = 0; k < n; k++) {
                final double val = s[base + k];
                s[base + k] = 0.5 * Math.log((1.0 + val) / (1.0 - val));
            }
        }
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
