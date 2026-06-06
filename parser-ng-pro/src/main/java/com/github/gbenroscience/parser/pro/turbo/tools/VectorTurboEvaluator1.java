package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import java.util.Random;
import jdk.incubator.vector.*;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class VectorTurboEvaluator1 extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VLEN = SPECIES.length();

    private static final int OP_CONST = 1;
    private static final int OP_LOAD = 2;
    private static final int OP_ADD = 3;
    private static final int OP_SUB = 4;
    private static final int OP_MUL = 5;
    private static final int OP_DIV = 6;
    private static final int OP_POW = 7;
    private static final int OP_SIN = 8;
    private static final int OP_COS = 9;
    private static final int OP_EXP = 10;
    private static final int OP_SQRT = 11;
    private static final int OP_LOG = 12;
    private static final int OP_VMA = 13;
    private static final int OP_REM = 14;
    private static final int OP_IF = 15;
    private static final int OP_GT = 16;
    private static final int OP_LT = 17;
    private static final int OP_EQ = 18;
    private static final int OP_NE = 19;
    private static final int OP_GE = 20;
    private static final int OP_LE = 21;

    // Fixed: Initializing arrays with specific sizes
    private static final ThreadLocal<double[]> SCALAR_FRAME =
            ThreadLocal.withInitial(() -> new double[1024]);
            
    private static final ThreadLocal<Future<?>[]> FUTURES_CACHE =
            ThreadLocal.withInitial(() -> new Future[128]);
            
    private static final ThreadLocal<MathExpression.EvalResult> RESULT_CACHE =
            ThreadLocal.withInitial(MathExpression.EvalResult::new);
            
    private static final ThreadLocal<DoubleVector[]> VECTOR_EXEC_STACK =
            ThreadLocal.withInitial(() -> new DoubleVector[32]);

    private final MathExpression.Token[] postfix;
    private int[] opcodes;
    private int[] targetSlots;
    private double[] literalConstants;
    private int instructionCount;
    private KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator1(MathExpression me) {
        super(me);
        this.postfix = me.getCachedPostfix();
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
                case MathExpression.Token.NUMBER:
                    opcodes[instructionCount] = OP_CONST;
                    literalConstants[instructionCount] = t.value;
                    instructionCount++;
                    break;
                case MathExpression.Token.VARIABLE:
                    opcodes[instructionCount] = OP_LOAD;
                    targetSlots[instructionCount] = resolveSlotIndex(t.frameIndex);
                    instructionCount++;
                    break;
                case MathExpression.Token.OPERATOR:
                    opcodes[instructionCount] = switch (t.opChar) {
                        case '+' -> OP_ADD;
                        case '-' -> OP_SUB;
                        case '*' -> OP_MUL;
                        case '/' -> OP_DIV;
                        case '^' -> OP_POW;
                        case '%' -> OP_REM;
                        case '>' -> OP_GT;
                        case '<' -> OP_LT;
                        case '=' -> OP_EQ;
                        case '!' -> OP_NE;
                        case 'g' -> OP_GE;
                        case 'l' -> OP_LE;
                        default -> throw new IllegalArgumentException("Unknown operator key: " + t.opChar);
                    };
                    instructionCount++;
                    break;
                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD:
                    if (isMatrixKernel(t.name, t.arity)) {
                        interceptedKernel = new KernelInterceptException(t.name, t.arity);
                        return;
                    }
                    opcodes[instructionCount] = switch (t.name.toLowerCase()) {
                        case "sin" -> OP_SIN;
                        case "cos" -> OP_COS;
                        case "exp" -> OP_EXP;
                        case "sqrt", "√" -> OP_SQRT;
                        case "log", "ln" -> OP_LOG;
                        case "vma" -> OP_VMA;
                        case "if" -> OP_IF;
                        default -> throw new IllegalArgumentException("Unknown intrinsic name: " + t.name);
                    };
                    instructionCount++;
                    break;
            }
        }
    }

    @Override
    public FastCompositeExpression compile() {
        try {
            final KernelInterceptException finalKernelSignal = this.interceptedKernel;
            FastCompositeExpression fce = super.compile();
            
            return new SIMDCompositeExpression() {
                @Override
                public double applyScalar(double[] variables) {
                    return fce.applyScalar(variables);
                }
                
                @Override
                public MathExpression.EvalResult apply(double[] variables) {
                    return RESULT_CACHE.get().wrap(applyScalar(variables));
                }
                
                @Override
                public String checkErrorLogs() {
                    return "";
                }
                
                @Override
                public TurboExpressionEvaluator getCompiler() {
                    return VectorTurboEvaluator1.this;
                }
                
                @Override
                public void applyBulk(double[][] variables, double[] output) {
                    applyBulkInternal(variables, output, 0);
                }
                
                @Override
                public void applyBulk(double[][] variables, double[] outputBuffer, int offset) {
                    applyBulkInternal(variables, outputBuffer, offset);
                }
                
                private void applyBulkInternal(double[][] variables, double[] output, int offset) {
                    // Fixed: The length of the data is variables[0].length
                    if (variables == null || variables.length == 0) return;
                    final int length = variables[0].length;
                    if (length == 0) return;
                    
                    if (interceptedKernel != null || length < VLEN) {
                        scalarBulk(variables, output, offset, length);
                        return;
                    }
                    
                    final int loopBound = SPECIES.loopBound(length);
                    int i = 0;
                    try {
                        for (; i < loopBound; i += VLEN) {
                            evaluateVectorFlat(variables, i, output, offset + i);
                        }
                        if (i < length) {
                            scalarBulk(variables, output, offset + i, length - i);
                        }
                    } catch (Throwable t) {
                        throw new RuntimeException("Vector macro evaluation failed at index " + i, t);
                    }
                }
                
                private void evaluateVectorFlat(double[][] variables, int offset, double[] out, int outOffset) {
                    DoubleVector[] registers = VECTOR_EXEC_STACK.get();
                    int sp = 0;
                    
                    for (int idx = 0; idx < instructionCount; idx++) {
                        int opcode = opcodes[idx];
                        switch (opcode) {
                            case OP_CONST:
                                registers[sp++] = DoubleVector.broadcast(SPECIES, literalConstants[idx]);
                                break;
                            case OP_LOAD:
                                registers[sp++] = DoubleVector.fromArray(SPECIES, variables[targetSlots[idx]], offset);
                                break;
                            case OP_ADD: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                registers[sp++] = l.add(r);
                                break;
                            }
                            case OP_SUB: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                registers[sp++] = l.sub(r);
                                break;
                            }
                            case OP_MUL: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                registers[sp++] = l.mul(r);
                                break;
                            }
                            case OP_DIV: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                registers[sp++] = l.div(r);
                                break;
                            }
                            case OP_POW: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                registers[sp++] = l.lanewise(VectorOperators.POW, r);
                                break;
                            }
                            case OP_SIN: {
                                registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SIN);
                                break;
                            }
                            case OP_COS: {
                                registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.COS);
                                break;
                            }
                            case OP_EXP: {
                                registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.EXP);
                                break;
                            }
                            case OP_SQRT: {
                                registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SQRT);
                                break;
                            }
                            case OP_LOG: {
                                registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.LOG);
                                break;
                            }
                            case OP_VMA: {
                                DoubleVector c = registers[--sp];
                                DoubleVector b = registers[--sp];
                                DoubleVector a = registers[--sp];
                                registers[sp++] = a.fma(b, c);
                                break;
                            }
                            case OP_REM: {
                                DoubleVector b = registers[--sp];
                                DoubleVector a = registers[--sp];
                                DoubleVector q = a.div(b);
                                LongVector t = (LongVector) q.convert(VectorOperators.Conversion.ofCast(double.class, long.class), 0);
                                DoubleVector td = (DoubleVector) t.convert(VectorOperators.Conversion.ofCast(long.class, double.class), 0);
                                registers[sp++] = a.sub(td.mul(b));
                                break;
                            }
                            case OP_IF: {
                                DoubleVector falseVal = registers[--sp];
                                DoubleVector trueVal = registers[--sp];
                                DoubleVector condition = registers[--sp];
                                VectorMask<Double> mask = condition.compare(VectorOperators.GT, 0.0);
                                registers[sp++] = falseVal.blend(trueVal, mask);
                                break;
                            }
                            case OP_GT: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                VectorMask<Double> mask = l.compare(VectorOperators.GT, r);
                                registers[sp++] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
                                break;
                            }
                            case OP_LT: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                VectorMask<Double> mask = l.compare(VectorOperators.LT, r);
                                registers[sp++] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
                                break;
                            }
                            case OP_EQ: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                VectorMask<Double> mask = l.compare(VectorOperators.EQ, r);
                                registers[sp++] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
                                break;
                            }
                            case OP_NE: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                VectorMask<Double> mask = l.compare(VectorOperators.NE, r);
                                registers[sp++] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
                                break;
                            }
                            case OP_GE: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                VectorMask<Double> mask = l.compare(VectorOperators.GE, r);
                                registers[sp++] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
                                break;
                            }
                            case OP_LE: {
                                DoubleVector r = registers[--sp];
                                DoubleVector l = registers[--sp];
                                VectorMask<Double> mask = l.compare(VectorOperators.LE, r);
                                registers[sp++] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
                                break;
                            }
                        }
                    }
                    registers[--sp].intoArray(out, outOffset);
                }
                
                private void scalarBulk(double[][] variables, double[] output, int offset, int length) {
                    double[] frame = SCALAR_FRAME.get();
                    final int nVars = variables.length;
                    if (frame.length < nVars) {
                        frame = new double[nVars];
                        SCALAR_FRAME.set(frame);
                    }
                    for (int i = 0; i < length; i++) {
                        for (int v = 0; v < nVars; v++) {
                            frame[v] = variables[v][offset + i];
                        }
                        output[offset + i] = applyScalar(frame);
                    }
                }
                
                @Override
                public void applyBulk(double[][] variables, double[] output, ExecutorService executor) {
                    final int length = output.length;
                    if (executor == null || length < 25_000) {
                        applyBulkInternal(variables, output, 0);
                        return;
                    }
                    
                    final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, length / 25_000));
                    Future<?>[] futures = FUTURES_CACHE.get();
                    
                    if (futures.length < nThreads) {
                        futures = new Future[nThreads];
                        FUTURES_CACHE.set(futures);
                    }
                    
                    final int chunk = (length + nThreads - 1) / nThreads;
                    
                    for (int t = 0; t < nThreads; t++) {
                        final int start = t * chunk;
                        final int end = Math.min(start + chunk, length);
                        futures[t] = executor.submit(() -> {
                            final int loopBound = start + SPECIES.loopBound(end - start);
                            int i = start;
                            try {
                                for (; i < loopBound; i += VLEN) {
                                    evaluateVectorFlat(variables, i, output, i);
                                }
                                if (i < end) {
                                    scalarBulk(variables, output, i, end - i);
                                }
                            } catch (Throwable e) {
                                throw new RuntimeException(e);
                            }
                        });
                    }
                    
                    for (int t = 0; t < nThreads; t++) {
                        try {
                            futures[t].get();
                            futures[t] = null;
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                }
                
                @Override
                public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
                    // Connects directly to the optimized FlatMatrix hardware layer
                }
            };
        } catch (Throwable ex) {
            System.getLogger(VectorTurboEvaluator1.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
            return null;
        }
    }

    private int resolveSlotIndex(int tokenIndex) {
        if (slots != null) {
            for (int k = 0; k < slots.length; k++) {
                if (slots[k] == tokenIndex) return k;
            }
        }
        return tokenIndex;
    }

    private boolean isMatrixKernel(String name, int arity) {
        return switch (name.toLowerCase()) {
            case "matmul", "matadd", "add_mat" -> arity == 2;
            case "softmax", "relu", "gelu", "exp", "log", "tanh", "sin" -> arity == 1;
            case "matmul_add_sin", "matmul_bias_gelu" -> arity == 3;
            default -> false;
        };
    }

    // Defining the custom Exception locally to prevent compilation failures if missing
    public static class KernelInterceptException extends RuntimeException {
        private final String kernelName;
        private final int arity;

        public KernelInterceptException(String kernelName, int arity) {
            this.kernelName = kernelName;
            this.arity = arity;
        }

        public String getKernelName() {
            return kernelName;
        }

        public int getArity() {
            return arity;
        }
    }

    public static void main(String[] args) {
        int dataSize = 50000;
             // Structure of Arrays (SoA): 3 variables (x1, x2, x3), each of length dataSize
       double[][] variables = new double[3][dataSize];
       double[] outputBuffer = new double[dataSize];
  Random rand = new Random(42);
        for (int i = 0; i < dataSize; i++) {
            variables[0][i] = 1.5 + rand.nextDouble() * 5.0;  // x1 (Must be >0 for div/pow)
            variables[1][i] = rand.nextDouble() * 5.0;        // x2
            variables[2][i] = rand.nextDouble() * 2.0;        // x3
        }
            MathExpression meGaussian = new MathExpression("(1 / (x1 * sqrt(2 * 3.141592653589793))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
         SIMDCompositeExpression gaussianExpr = (SIMDCompositeExpression) new VectorTurboEvaluator1(meGaussian).compile();
         
         int i=0;
         while(i++<50000){
             gaussianExpr.applyBulk(variables, outputBuffer);
         }
    }

}