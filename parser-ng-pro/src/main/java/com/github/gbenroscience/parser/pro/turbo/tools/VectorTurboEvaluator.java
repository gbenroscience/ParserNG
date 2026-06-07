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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * Original low-allocation VectorTurboEvaluator + FlatMatrix integration.
 */
public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
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

    private static final ThreadLocal<double[]> SCALAR_FRAME = ThreadLocal.withInitial(() -> new double[0]);
    private static final ThreadLocal<Future<?>[]> FUTURES_CACHE = ThreadLocal.withInitial(() -> new Future[0]);
    private static final ThreadLocal<MathExpression.EvalResult> RESULT_CACHE = ThreadLocal.withInitial(MathExpression.EvalResult::new);

    private final MathExpression.Token[] postfix;
    private int[] opcodes;
    private int[] targetSlots;
    private double[] literalConstants;
    private int instructionCount;
    private KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator(MathExpression me) {
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
                case MathExpression.Token.FUNCTION, MathExpression.Token.METHOD -> {
                    if (isMatrixKernel(t.name, t.arity)) {
                        interceptedKernel = new KernelInterceptException(t.name, t.arity);
                        return;
                    }
                    opcodes[instructionCount] = switch (t.name.toLowerCase()) {
                        case "sin" -> OP_SIN;
                        case "cos" -> OP_COS;
                        case "exp" -> OP_EXP;
                        case "sqrt" -> OP_SQRT;
                        case "log" -> OP_LOG;
                        case "vma" -> OP_VMA;
                        default -> throw new IllegalArgumentException("Unknown function: " + t.name);
                    };
                    instructionCount++;
                }
            }
        }
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        final MethodHandle rawScalarHandle = super.compileScalar(this.postfix);
        MethodHandle vectorTreeHandle = compileVectorTree();
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
                final int length = variables.length;
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
            public void applyBulk(double[][] variables, double[] output, java.util.concurrent.ExecutorService executor) {
                final int length = output.length;
                if (executor == null || length < 25_000 || finalVectorHandle == null) {
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
            public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
                String kernelToRun = (op != null) ? op : (interceptedKernel != null ? interceptedKernel.getKernelName() : null);
                if (kernelToRun == null) throw new UnsupportedOperationException("No kernel");

                switch (kernelToRun.toLowerCase()) {
                    case "matmul" -> FlatMatrix.matmul(inputs[0], inputs[1], output);
                    case "add", "matadd", "add_mat" -> FlatMatrix.add(inputs[0], inputs[1], output);
                    case "matmul_bias_gelu", "matmul_add_bias_gelu" -> FlatMatrix.matmulAddBiasGelu(inputs[0], inputs[1], inputs[2], output);
                    case "matmul_add_sin" -> {
                        double alpha = inputs[2].get(0, 0);
                        FlatMatrix.matmulAddSin(inputs[0], inputs[1], output, alpha);
                    }
                    case "softmax", "softmax_in_place" -> {
                        if (inputs[0] != output) System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        output.softmaxRowsInPlace();
                    }
                    case "relu", "relu_in_place" -> {
                        if (inputs[0] != output) System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        output.reluInPlace();
                    }
                    case "gelu", "gelu_in_place" -> {
                        if (inputs[0] != output) System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                        output.geluInPlace();
                    }
                    // Add more cases as needed
                    default -> throw new UnsupportedOperationException("Unknown kernel: " + kernelToRun);
                }
            }
        };
    }

    private MethodHandle compileVectorTree() throws Throwable {
        MethodHandle h = LOOKUP.findStatic(VectorTurboEvaluator.class, "execFusedVectorProgram",
                MethodType.methodType(void.class, int[].class, int[].class, double[].class, int.class,
                        double[][].class, int.class, double[].class, int.class));
        return MethodHandles.insertArguments(h, 0, this.opcodes, this.targetSlots, this.literalConstants, this.instructionCount);
    }

    public static void execFusedVectorProgram(int[] opcodes, int[] targetSlots, double[] literalConstants, int instructionCount,
                                              double[][] vars, int off, double[] out, int outOff) {
        DoubleVector[] registers = new DoubleVector[instructionCount + 2];
        int sp = 0;
        for (int idx = 0; idx < instructionCount; idx++) {
            int opcode = opcodes[idx];
            switch (opcode) {
                case OP_CONST -> registers[sp++] = DoubleVector.broadcast(SPECIES, literalConstants[idx]);
                case OP_LOAD -> registers[sp++] = DoubleVector.fromArray(SPECIES, vars[targetSlots[idx]], off);
                case OP_ADD -> { DoubleVector r = registers[--sp]; DoubleVector l = registers[--sp]; registers[sp++] = l.add(r); }
                case OP_SUB -> { DoubleVector r = registers[--sp]; DoubleVector l = registers[--sp]; registers[sp++] = l.sub(r); }
                case OP_MUL -> { DoubleVector r = registers[--sp]; DoubleVector l = registers[--sp]; registers[sp++] = l.mul(r); }
                case OP_DIV -> { DoubleVector r = registers[--sp]; DoubleVector l = registers[--sp]; registers[sp++] = l.div(r); }
                case OP_POW -> { DoubleVector r = registers[--sp]; DoubleVector l = registers[--sp]; registers[sp++] = l.lanewise(VectorOperators.POW, r); }
                case OP_SIN -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SIN);
                case OP_COS -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.COS);
                case OP_EXP -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.EXP);
                case OP_SQRT -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.SQRT);
                case OP_LOG -> registers[sp - 1] = registers[sp - 1].lanewise(VectorOperators.LOG);
                case OP_VMA -> {
                    DoubleVector c = registers[--sp];
                    DoubleVector b = registers[--sp];
                    DoubleVector a = registers[--sp];
                    registers[sp++] = a.fma(b, c);
                }
                case OP_REM -> {
                    DoubleVector b = registers[--sp];
                    DoubleVector a = registers[--sp];
                    DoubleVector q = a.div(b);
                    LongVector t = (LongVector) q.convert(VectorOperators.Conversion.ofCast(double.class, long.class), 0);
                    DoubleVector td = (DoubleVector) t.convert(VectorOperators.Conversion.ofCast(long.class, double.class), 0);
                    registers[sp++] = a.sub(td.mul(b));
                }
            }
        }
        registers[--sp].intoArray(out, outOff);
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
}