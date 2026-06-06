package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
    private static final int VLEN = SPECIES.length();
    private static final ThreadLocal<double[]> SCALAR_FRAME = ThreadLocal.withInitial(() -> new double[32]);
    private static final ThreadLocal<Future<?>[]> FUTURES_CACHE = ThreadLocal.withInitial(() -> new Future<?>[64]);
    private static final ThreadLocal<MathExpression.EvalResult> RESULT_CACHE = ThreadLocal.withInitial(MathExpression.EvalResult::new);
    private static final ThreadLocal<double[]> VECTOR_REGISTER_BANK = ThreadLocal.withInitial(() -> new double[1024]);
    
    private final MathExpression.Token[] postfix;
    private KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator(MathExpression me) {
        super(me);
        this.postfix = me.getCachedPostfix();
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        final MethodHandle rawScalarHandle = super.compileScalar(this.postfix);
        MethodHandle vectorTreeHandle = null;
        try {
            vectorTreeHandle = compileVectorTree(this.postfix);
        } catch (KernelInterceptException ex) {
            this.interceptedKernel = ex;
        }
        final MethodHandle finalVectorHandle = vectorTreeHandle;

        return new SIMDCompositeExpression() {
            @Override
            public double applyScalar(double[] variables) {
                try {
                    return (double) rawScalarHandle.invokeExact(variables);
                } catch (Throwable t) {
                    throw new RuntimeException(t);
                }
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
                return VectorTurboEvaluator.this;
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
                final int length = variables[0].length;
                if (length == 0) return;
                
                if (interceptedKernel != null || length < VLEN || finalVectorHandle == null) {
                    scalarBulk(variables, output, offset, length);
                    return;
                }
                
                final int loopBound = SPECIES.loopBound(length);
                int i = 0;
                try {
                    for (; i < loopBound; i += VLEN) {
                        finalVectorHandle.invokeExact(variables, i, output, offset + i);
                    }
                    if (i < length) {
                        scalarBulk(variables, output, offset + i, length - i);
                    }
                } catch (Throwable t) {
                    throw new RuntimeException("Vector MethodHandle execution failed at index " + i, t);
                }
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
                if (executor == null || length < 25_000 || finalVectorHandle == null) {
                    applyBulkInternal(variables, output, 0);
                    return;
                }
                
                final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, length / 25_000));
                Future<?>[] futures = FUTURES_CACHE.get();
                if (futures.length < nThreads) {
                    futures = new Future<?>[nThreads];
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
                                finalVectorHandle.invokeExact(variables, i, output, i);
                            }
                            if (i < end) {
                                scalarBulk(variables, output, i, end - i);
                            }
                        } catch (Throwable e) {
                            throw new RuntimeException("Parallel execution failed", e);
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
            public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {}
        };
    }

    private MethodHandle compileVectorTree(MathExpression.Token[] postfix) throws Throwable {
        Stack<Integer> registerStack = new Stack<>();
        List<MethodHandle> handles = new ArrayList<>();
        int regCounter = 0;
        
        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER: {
                    int destReg = regCounter++;
                    MethodHandle h = LOOKUP.findStatic(VectorTurboEvaluator.class, "execConst", 
                        MethodType.methodType(void.class, double.class, int.class, double[][].class, int.class, double[].class, int.class));
                    h = MethodHandles.insertArguments(h, 0, t.value, destReg);
                    handles.add(h);
                    registerStack.push(destReg);
                    break;
                }
                case MathExpression.Token.VARIABLE: {
                    int destReg = regCounter++;
                    int slot = resolveSlotIndex(t.frameIndex);
                    MethodHandle h = LOOKUP.findStatic(VectorTurboEvaluator.class, "execLoad", 
                        MethodType.methodType(void.class, int.class, int.class, double[][].class, int.class, double[].class, int.class));
                    h = MethodHandles.insertArguments(h, 0, slot, destReg);
                    handles.add(h);
                    registerStack.push(destReg);
                    break;
                }
                case MathExpression.Token.OPERATOR: {
                    int rReg = registerStack.pop();
                    int lReg = registerStack.pop();
                    int destReg = regCounter++;
                    String opMethod = switch (t.opChar) {
                        case '+' -> "execAdd";
                        case '-' -> "execSub";
                        case '*' -> "execMul";
                        case '/' -> "execDiv";
                        case '^' -> "execPow";
                        case '%' -> "execRem";
                        default -> throw new UnsupportedOperationException();
                    };
                    MethodHandle h = LOOKUP.findStatic(VectorTurboEvaluator.class, opMethod, 
                        MethodType.methodType(void.class, int.class, int.class, int.class, double[][].class, int.class, double[].class, int.class));
                    h = MethodHandles.insertArguments(h, 0, lReg, rReg, destReg);
                    handles.add(h);
                    registerStack.push(destReg);
                    break;
                }
                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD: {
                    if (isMatrixKernel(t.name, t.arity)) {
                        throw new KernelInterceptException(t.name, t.arity);
                    }
                    if (t.arity == 1) {
                        int srcReg = registerStack.pop();
                        int destReg = regCounter++;
                        String unMethod = switch (t.name.toLowerCase()) {
                            case "sin" -> "execSin";
                            case "cos" -> "execCos";
                            case "exp" -> "execExp";
                            case "sqrt" -> "execSqrt";
                            case "log" -> "execLog";
                            default -> throw new UnsupportedOperationException(t.name);
                        };
                        MethodHandle h = LOOKUP.findStatic(VectorTurboEvaluator.class, unMethod, 
                            MethodType.methodType(void.class, int.class, int.class, double[][].class, int.class, double[].class, int.class));
                        h = MethodHandles.insertArguments(h, 0, srcReg, destReg);
                        handles.add(h);
                        registerStack.push(destReg);
                    } else if (t.arity == 3 && "vma".equalsIgnoreCase(t.name)) {
                        int cReg = registerStack.pop();
                        int bReg = registerStack.pop();
                        int aReg = registerStack.pop();
                        int destReg = regCounter++;
                        MethodHandle h = LOOKUP.findStatic(VectorTurboEvaluator.class, "execVma", 
                            MethodType.methodType(void.class, int.class, int.class, int.class, int.class, double[][].class, int.class, double[].class, int.class));
                        h = MethodHandles.insertArguments(h, 0, aReg, bReg, cReg, destReg);
                        handles.add(h);
                        registerStack.push(destReg);
                    }
                    break;
                }
            }
        }
        int finalReg = registerStack.pop();
        MethodHandle store = LOOKUP.findStatic(VectorTurboEvaluator.class, "execStore", 
            MethodType.methodType(void.class, int.class, double[][].class, int.class, double[].class, int.class));
        store = MethodHandles.insertArguments(store, 0, finalReg);
        handles.add(store);
        
        MethodHandle finalTree = handles.get(0);
        for (int i = 1; i < handles.size(); i++) {
            finalTree = MethodHandles.foldArguments(handles.get(i), finalTree);
        }
        return finalTree;
    }

    // Static helper methods for MethodHandles
    public static void execConst(double val, int dest, double[][] vars, int off, double[] out, int outOff) {
        DoubleVector.broadcast(SPECIES, val).intoArray(VECTOR_REGISTER_BANK.get(), dest * VLEN);
    }

    public static void execLoad(int slot, int dest, double[][] vars, int off, double[] out, int outOff) {
        DoubleVector.fromArray(SPECIES, vars[slot], off).intoArray(VECTOR_REGISTER_BANK.get(), dest * VLEN);
    }

    public static void execAdd(int srcL, int srcR, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, srcL * VLEN).add(DoubleVector.fromArray(SPECIES, r, srcR * VLEN)).intoArray(r, dest * VLEN);
    }

    public static void execSub(int srcL, int srcR, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, srcL * VLEN).sub(DoubleVector.fromArray(SPECIES, r, srcR * VLEN)).intoArray(r, dest * VLEN);
    }

    public static void execMul(int srcL, int srcR, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, srcL * VLEN).mul(DoubleVector.fromArray(SPECIES, r, srcR * VLEN)).intoArray(r, dest * VLEN);
    }

    public static void execDiv(int srcL, int srcR, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, srcL * VLEN).div(DoubleVector.fromArray(SPECIES, r, srcR * VLEN)).intoArray(r, dest * VLEN);
    }

    public static void execPow(int srcL, int srcR, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, srcL * VLEN).lanewise(VectorOperators.POW, DoubleVector.fromArray(SPECIES, r, srcR * VLEN)).intoArray(r, dest * VLEN);
    }

    public static void execSin(int src, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, src * VLEN).lanewise(VectorOperators.SIN).intoArray(r, dest * VLEN);
    }

    public static void execCos(int src, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, src * VLEN).lanewise(VectorOperators.COS).intoArray(r, dest * VLEN);
    }

    public static void execExp(int src, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, src * VLEN).lanewise(VectorOperators.EXP).intoArray(r, dest * VLEN);
    }

    public static void execSqrt(int src, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, src * VLEN).lanewise(VectorOperators.SQRT).intoArray(r, dest * VLEN);
    }

    public static void execLog(int src, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, src * VLEN).lanewise(VectorOperators.LOG).intoArray(r, dest * VLEN);
    }

    public static void execRem(int srcL, int srcR, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector a = DoubleVector.fromArray(SPECIES, r, srcL * VLEN);
        DoubleVector b = DoubleVector.fromArray(SPECIES, r, srcR * VLEN);
        DoubleVector q = a.div(b);
        LongVector t = (LongVector) q.convert(VectorOperators.Conversion.ofCast(double.class, long.class), 0);
        DoubleVector td = (DoubleVector) t.convert(VectorOperators.Conversion.ofCast(long.class, double.class), 0);
        a.sub(td.mul(b)).intoArray(r, dest * VLEN);
    }

    public static void execVma(int srcA, int srcB, int srcC, int dest, double[][] vars, int off, double[] out, int outOff) {
        double[] r = VECTOR_REGISTER_BANK.get();
        DoubleVector.fromArray(SPECIES, r, srcA * VLEN).fma(DoubleVector.fromArray(SPECIES, r, srcB * VLEN), DoubleVector.fromArray(SPECIES, r, srcC * VLEN)).intoArray(r, dest * VLEN);
    }

    public static void execStore(int srcReg, double[][] vars, int off, double[] out, int outOff) {
        DoubleVector.fromArray(SPECIES, VECTOR_REGISTER_BANK.get(), srcReg * VLEN).intoArray(out, outOff);
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
        return false;
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
         while(i++<5){
             gaussianExpr.applyBulk(variables, outputBuffer);
         }
    }
}