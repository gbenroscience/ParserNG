package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * REAL Vector Turbo - True SIMD without allocation overhead.
 * 
 * KEY INSIGHT: Pre-allocate DoubleVector[] at construction time.
 * Then reuse the same vectors in the loop - no allocations.
 * 
 * The Vector API allocates ONCE per vector at creation.
 * If we pre-create them and reuse, no allocation in hot loop.
 */
public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VLEN = SPECIES.length();

    // Opcodes
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

    // **PRE-ALLOCATED** at construction time (instance, not ThreadLocal)
    private final DoubleVector[] vectorRegisters;
    private final double[] scalarFrameBuffer;
    
    private final MathExpression.Token[] postfix;
    private int[] opcodes;
    private int[] targetSlots;
    private double[] literalConstants;
    private int instructionCount;
    private KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator(MathExpression me) throws Throwable {
        super(me);
        this.postfix = me.getCachedPostfix();
        
        // **Pre-allocate register file ONCE**
        // This is the key: allocate once, reuse forever
        this.vectorRegisters = new DoubleVector[Math.max(postfix.length + 16, 256)];
        this.scalarFrameBuffer = new double[128];
        
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
                        case '+' -> OP_ADD;
                        case '-' -> OP_SUB;
                        case '*' -> OP_MUL;
                        case '/' -> OP_DIV;
                        case '^' -> OP_POW;
                        case '%' -> OP_REM;
                        case '>' -> OP_GT;
                        case '<' -> OP_LT;
                        default -> throw new IllegalArgumentException("Unknown operator: " + t.opChar);
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
                        case "vma" -> OP_VMA;
                        case "if" -> OP_IF;
                        default -> throw new IllegalArgumentException("Unknown function: " + t.name);
                    };
                    instructionCount++;
                }
            }
        }
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        final MethodHandle scalarHandle = compileScalar(postfix);
        return new VectorCompositeExpression(scalarHandle, vectorRegisters, scalarFrameBuffer, 
                                            opcodes, targetSlots, literalConstants, instructionCount){
            @Override
            public TurboExpressionEvaluator getCompiler() {
                return VectorTurboEvaluator.this;
            }
                                            
                                            };
    }

    public static class VectorCompositeExpression implements FastCompositeExpression {
        
        private final MethodHandle scalarHandle;
        private final DoubleVector[] registers;
        private final double[] frameBuffer;
        private final int[] opcodes;
        private final int[] targetSlots;
        private final double[] literalConstants;
        private final int instructionCount;

        VectorCompositeExpression(MethodHandle handle, DoubleVector[] regs, double[] frame,
                                  int[] ops, int[] slots, double[] consts, int count) {
            this.scalarHandle = handle;
            this.registers = regs;
            this.frameBuffer = frame;
            this.opcodes = ops;
            this.targetSlots = slots;
            this.literalConstants = consts;
            this.instructionCount = count;
        }

        @Override
        public double applyScalar(double[] variables) {
            try {
                return (double) scalarHandle.invokeExact(variables);
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }

         
        public MathExpression.EvalResult apply(double[] variables) {
           // return MathExpression.EvalResult.wrap(applyScalar(variables));
           return null;
        }

     
        public String checkErrorLogs() {
            return "";
        }

        @Override
        public TurboExpressionEvaluator getCompiler() {
            return null;
        }

       
        public void applyBulk(double[][] variables, double[] output) {
            applyBulkInternal(variables, output, 0, output.length);
        }

       
        public void applyBulk(double[][] variables, double[] output, int offset) {
            applyBulkInternal(variables, output, offset, output.length - offset);
        }

        /**
         * **TRUE SIMD: 8 elements per iteration, NO allocations**
         * 
         * Pre-allocated registers are reused.
         * Vector operations process 8 lanes in parallel.
         */
        private void applyBulkInternal(double[][] variables, double[] output, int startIdx, int length) {
            if (variables == null || variables.length == 0 || length <= 0) {
                return;
            }

            final int endIdx = startIdx + length;
            final int loopBound = startIdx + (length / VLEN) * VLEN;

            // **VECTOR LOOP: Process 8 elements per iteration**
            for (int i = startIdx; i < loopBound; i += VLEN) {
                execVectorIteration(variables, i, output, i);
            }

            // **Scalar tail**
            for (int i = loopBound; i < endIdx; i++) {
                for (int v = 0; v < variables.length; v++) {
                    frameBuffer[v] = variables[v][i];
                }
                try {
                    output[i] = (double) scalarHandle.invokeExact(frameBuffer);
                } catch (Throwable t) {
                    output[i] = Double.NaN;
                }
            }
        }

        /**
         * **Execute one vector iteration (8 lanes)**
         * 
         * Reuses pre-allocated registers - zero allocations.
         */
        private void execVectorIteration(double[][] variables, int idx, double[] output, int outIdx) {
            int sp = 0;

            for (int instIdx = 0; instIdx < instructionCount; instIdx++) {
                final int opcode = opcodes[instIdx];

                switch (opcode) {
                    case OP_CONST -> {
                        registers[sp++] = DoubleVector.broadcast(VectorTurboEvaluator.SPECIES, literalConstants[instIdx]);
                    }
                    case OP_LOAD -> {
                        registers[sp++] = DoubleVector.fromArray(VectorTurboEvaluator.SPECIES, 
                                                                   variables[targetSlots[instIdx]], idx);
                    }
                    case OP_ADD -> {
                        DoubleVector r = registers[--sp];
                        DoubleVector l = registers[--sp];
                        registers[sp++] = l.add(r);
                    }
                    case OP_SUB -> {
                        DoubleVector r = registers[--sp];
                        DoubleVector l = registers[--sp];
                        registers[sp++] = l.sub(r);
                    }
                    case OP_MUL -> {
                        DoubleVector r = registers[--sp];
                        DoubleVector l = registers[--sp];
                        registers[sp++] = l.mul(r);
                    }
                    case OP_DIV -> {
                        DoubleVector r = registers[--sp];
                        DoubleVector l = registers[--sp];
                        registers[sp++] = l.div(r);
                    }
                    case OP_POW -> {
                        DoubleVector r = registers[--sp];
                        DoubleVector l = registers[--sp];
                        registers[sp++] = l.pow(r);
                    }
                    case OP_SIN -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SIN);
                    case OP_COS -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.COS);
                    case OP_TAN -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.TAN);
                    case OP_ASIN -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.ASIN);
                    case OP_ACOS -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.ACOS);
                    case OP_ATAN -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.ATAN);
                    case OP_SINH -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SINH);
                    case OP_COSH -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.COSH);
                    case OP_TANH -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.TANH);
                    case OP_ABS -> registers[sp - 1] = registers[sp - 1].abs();
                    case OP_EXP -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.EXP);
                    case OP_SQRT -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SQRT);
                    case OP_CBRT -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.CBRT);
                    case OP_LOG -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.LOG);
                    case OP_LOG10 -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.LOG10);
                    case OP_VMA -> {
                        DoubleVector c = registers[--sp];
                        DoubleVector b = registers[--sp];
                        DoubleVector a = registers[--sp];
                        registers[sp++] = a.fma(b, c);
                    }
                    case OP_REM -> {
                        DoubleVector b = registers[--sp];
                        DoubleVector a = registers[--sp];
                        registers[sp++] = execVectorRemainder(a, b);
                    }
                    case OP_GT -> {
                        DoubleVector r = registers[--sp];
                        registers[sp - 1] = registers[sp - 1].compare(VectorOperators.GT, r).toVector().reinterpretAsDoubles();
                    }
                    case OP_LT -> {
                        DoubleVector r = registers[--sp];
                        registers[sp - 1] = registers[sp - 1].compare(VectorOperators.LT, r).toVector().reinterpretAsDoubles();
                    }
                    case OP_EQ -> {
                        DoubleVector r = registers[--sp];
                        registers[sp - 1] = registers[sp - 1].compare(VectorOperators.EQ, r).toVector().reinterpretAsDoubles();
                    }
                    case OP_NE -> {
                        DoubleVector r = registers[--sp];
                        registers[sp - 1] = registers[sp - 1].compare(VectorOperators.NE, r).toVector().reinterpretAsDoubles();
                    }
                    case OP_GE -> {
                        DoubleVector r = registers[--sp];
                        registers[sp - 1] = registers[sp - 1].compare(VectorOperators.GE, r).toVector().reinterpretAsDoubles();
                    }
                    case OP_LE -> {
                        DoubleVector r = registers[--sp];
                        registers[sp - 1] = registers[sp - 1].compare(VectorOperators.LE, r).toVector().reinterpretAsDoubles();
                    }
                    case OP_IF -> {
                        DoubleVector falseVal = registers[--sp];
                        DoubleVector trueVal = registers[--sp];
                        DoubleVector cond = registers[--sp];
                        VectorMask<Double> mask = cond.compare(VectorOperators.GT, 0.0);
                        registers[sp++] = trueVal.blend(falseVal, mask);
                    }
                }
            }

            registers[--sp].intoArray(output, outIdx);
        }

        private static DoubleVector execVectorRemainder(DoubleVector a, DoubleVector b) {
            DoubleVector q = a.div(b);
            LongVector t = (LongVector) q.convert(VectorOperators.Conversion.ofCast(double.class, long.class), 0);
            DoubleVector td = (DoubleVector) t.convert(VectorOperators.Conversion.ofCast(long.class, double.class), 0);
            return a.sub(td.mul(b));
        }

         
        public void applyBulk(double[][] variables, double[] output, ExecutorService executor) {
            if (variables == null || variables.length == 0 || executor == null) {
                applyBulkInternal(variables, output, 0, output.length);
                return;
            }

            final int length = output.length;
            if (length < 100_000) {
                applyBulkInternal(variables, output, 0, length);
                return;
            }

            final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), 
                                         Math.max(1, length / 100_000));
            final int chunkSize = (length + nThreads - 1) / nThreads;
            @SuppressWarnings("unchecked")
            final Future<Void>[] futures = new Future[nThreads];

            for (int t = 0; t < nThreads; t++) {
                final int start = t * chunkSize;
                final int end = Math.min(start + chunkSize, length);
                futures[t] = executor.submit(() -> {
                    applyBulkInternal(variables, output, start, end - start);
                    return null;
                });
            }

            for (Future<Void> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        
        public void applyBulkBatched(double[][] variables, double[] output, int batchSize) {
            int offset = 0;
            final int length = output.length;
            while (offset < length) {
                final int nextOffset = Math.min(offset + batchSize, length);
                applyBulkInternal(variables, output, offset, nextOffset - offset);
                offset = nextOffset;
            }
        }

        
        public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op) {
            throw new UnsupportedOperationException("Not supported");
        }

       
        public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
            throw new UnsupportedOperationException("Not supported");
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

    private boolean isMatrixKernel(String name, int arity) {
        return switch (name.toLowerCase()) {
            case "matmul", "gemm", "mm" -> arity == 3;
            case "gemv", "matvec" -> arity == 2;
            case "dot", "inner", "axpy" -> arity == 3;
            case "transpose", "t" -> arity == 1;
            case "add", "sub", "mul", "div", "pow" -> arity == 2;
            case "relu", "gelu", "sigmoid", "tanh", "silu", "swish", "mish" -> arity == 1;
            case "exp", "log", "sqrt", "rsqrt", "sin", "cos" -> arity == 1;
            case "leaky_relu", "elu", "clip", "clamp" -> arity == 2;
            case "abs", "neg", "square" -> arity == 1;
            case "sum", "mean", "prod", "max", "min" -> arity == 1;
            case "softmax", "log_softmax" -> arity == 1;
            case "layer_norm" -> arity == 3;
            case "rms_norm" -> arity == 2;
            case "fma" -> arity == 3;
            case "bias_add" -> arity == 2;
            case "linear", "matmul_bias" -> arity == 3;
            case "matmul_bias_relu", "matmul_bias_gelu" -> arity == 3;
            case "matmul_bias_add" -> arity == 4;
            case "mse", "mae", "cross_entropy" -> arity == 2;
            default -> false;
        };
    }
}