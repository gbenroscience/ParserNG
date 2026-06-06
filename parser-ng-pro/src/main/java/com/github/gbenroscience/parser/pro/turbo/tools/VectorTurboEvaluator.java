package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

/**
 * High-performance Vector Interpreter for ParserNG.
 * Direct vector dispatch with minimal allocation.
 *
 * @author GBEMIRO
 */
public class VectorTurboEvaluator extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VLEN = SPECIES.length();

    private static final ThreadLocal<double[]> SCALAR_FRAME = ThreadLocal.withInitial(() -> new double[1024]);
    private static final ThreadLocal<Future<?>[]> FUTURES_CACHE = ThreadLocal.withInitial(() -> new Future[128]);
    private static final ThreadLocal<MathExpression.EvalResult> RESULT_CACHE = ThreadLocal.withInitial(MathExpression.EvalResult::new);

    private final MathExpression.Token[] postfix;
    private final List<Instruction> instructions = new ArrayList<>();
    private KernelInterceptException interceptedKernel;

    public VectorTurboEvaluator(MathExpression me) {
        super(me);
        this.postfix = me.getCachedPostfix();
        compileToInstructions();
    }

    private void compileToInstructions() {
        Stack<Instruction> stack = new Stack<>();
        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    stack.push(new ConstantInstr(t.value));
                    break;
                case MathExpression.Token.VARIABLE:
                    stack.push(new LoadInstr(t.frameIndex));
                    break;
                case MathExpression.Token.OPERATOR:
                    Instruction right = stack.pop();
                    Instruction left = stack.pop();
                    if (t.opChar == '%') {
                        stack.push(new RemainderInstr(left, right));
                    } else if (isRelationalOp(t.opChar)) {
                        stack.push(new ComparisonInstr(left, right, t.opChar));
                    } else {
                        stack.push(new BinaryInstr(left, right, t.opChar));
                    }
                    break;
                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD:
                    if (isMatrixKernel(t.name, t.arity)) {
                        interceptedKernel = new KernelInterceptException(t.name, t.arity);
                        return;
                    }
                    if (t.arity == 1) {
                        stack.push(new UnaryInstr(stack.pop(), t.name));
                    } else if (t.arity == 2) {
                        Instruction r = stack.pop();
                        Instruction l = stack.pop();
                        stack.push(new BinaryFuncInstr(l, r, t.name));
                    } else if (t.arity == 3 && "vma".equalsIgnoreCase(t.name)) {
                        Instruction c = stack.pop();
                        Instruction b = stack.pop();
                        Instruction a = stack.pop();
                        stack.push(new FmaInstr(a, b, c));
                    }
                    break;
            }
        }
        if (!stack.isEmpty()) {
            instructions.add(stack.pop());
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
                    
                    if (interceptedKernel != null || length < VLEN) {
                        scalarBulk(variables, output, offset, length);
                        return;
                    }
                    
                    final int loopBound = SPECIES.loopBound(length);
                    int i = 0;
                    try {
                        for (; i < loopBound; i += VLEN) {
                            DoubleVector result = evaluateVector(variables, i);
                            result.intoArray(output, offset + i);
                        }
                        if (i < length) {
                            scalarBulk(variables, output, offset + i, length);
                        }
                    } catch (Throwable t) {
                        throw new RuntimeException("Vector evaluation failed at index " + i, t);
                    }
                }
                
                private DoubleVector evaluateVector(double[][] variables, int offset) {
                    DoubleVector[] stack = new DoubleVector[32]; // Fixed size - sufficient for most expressions
                    int sp = 0;
                    
                    for (Instruction instr : instructions) {
                        instr.execute(variables, offset, stack, sp);
                        sp++;
                    }
                    return stack[sp - 1];
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
                            frame[v] = variables[v][i];
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
                        futures = new Future<?>[nThreads];
                        FUTURES_CACHE.set(futures);
                    }
                    
                    final int chunk = (length + nThreads - 1) / nThreads;
                    
                    for (int t = 0; t < nThreads; t++) {
                        final int threadId = t;
                        final int start = t * chunk;
                        final int end = Math.min(start + chunk, length);
                        futures[t] = executor.submit(() -> {
                            final int loopBound = start + SPECIES.loopBound(end - start);
                            int i = start;
                            try {
                                for (; i < loopBound; i += VLEN) {
                                    DoubleVector res = evaluateVector(variables, i);
                                    res.intoArray(output, i);
                                }
                                if (i < end) {
                                    scalarBulk(variables, output, i, end);
                                }
                            } catch (Throwable e) {
                                throw new RuntimeException("Executor task failed at index " + i + " thread " + threadId, e);
                            }
                        });
                    }
                    
                    for (int t = 0; t < nThreads; t++) {
                        try {
                            futures[t].get();
                            futures[t] = null;
                        } catch (ExecutionException e) {
                            Throwable cause = e.getCause() != null ? e.getCause() : e;
                            throw new RuntimeException("Parallel execution failed", cause);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException("Interrupted during parallel execution", e);
                        }
                    }
                }
                
                @Override
                public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op) {
                    String kernelToRun = (op != null) ? op
                            : (interceptedKernel != null) ? interceptedKernel.getKernelName() : null;
                    
                    if (kernelToRun == null) {
                        throw new UnsupportedOperationException("No matrix kernel specified or intercepted.");
                    }
                    
                    switch (kernelToRun.toLowerCase()) {
                        case "matmul":
                            if (inputs.length != 2) throw new IllegalArgumentException("matmul requires 2 inputs");
                            FlatMatrix.matmul(inputs[0], inputs[1], output);
                            break;
                        case "add":
                        case "matadd":
                        case "add_mat":
                            if (inputs.length != 2) throw new IllegalArgumentException("matadd requires 2 inputs");
                            FlatMatrix.add(inputs[0], inputs[1], output);
                            break;
                        case "matmul_bias_gelu":
                        case "matmul_add_bias_gelu":
                            if (inputs.length != 3) throw new IllegalArgumentException("matmul_bias_gelu requires 3 inputs");
                            FlatMatrix.matmulAddBiasGelu(inputs[0], inputs[1], inputs[2], output);
                            break;
                        case "matmul_add_sin":
                            if (inputs.length != 3) throw new IllegalArgumentException("matmul_add_sin requires 3 inputs");
                            double alpha = inputs[2].get(0, 0);
                            FlatMatrix.matmulAddSin(inputs[0], inputs[1], output, alpha);
                            break;
                        case "softmax":
                        case "softmax_in_place":
                            if (inputs.length != 1) throw new IllegalArgumentException("softmax requires 1 input");
                            if (inputs[0] != output) {
                                System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                            }
                            output.softmaxRowsInPlace();
                            break;
                        case "relu":
                        case "relu_in_place":
                            if (inputs.length != 1) throw new IllegalArgumentException("relu requires 1 input");
                            if (inputs[0] != output) {
                                System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                            }
                            output.reluInPlace();
                            break;
                        case "gelu":
                        case "gelu_in_place":
                            if (inputs.length != 1) throw new IllegalArgumentException("gelu requires 1 input");
                            if (inputs[0] != output) {
                                System.arraycopy(inputs[0].data, inputs[0].offset, output.data, output.offset, output.rows * output.cols);
                            }
                            output.geluInPlace();
                            break;
                        default:
                            throw new UnsupportedOperationException("Unknown Matrix Kernel: " + kernelToRun);
                    }
                }
                
                public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String op, ExecutorService executor) {
                    if (executor == null || !isElementWiseOp(op)) {
                        applyMatrixKernel(inputs, output, op);
                        return;
                    }
                    runElementWiseParallel(inputs[0], output, op, executor);
                }
                
                private boolean isElementWiseOp(String op) {
                    if (op == null) return false;
                    return switch (op.toLowerCase()) {
                        case "relu", "gelu", "exp", "log", "tanh", "sin", "softmax" -> true;
                        default -> false;
                    };
                }
                
                private void runElementWiseParallel(FlatMatrix in, FlatMatrix out, String op, ExecutorService executor) {
                    final int length = in.data.length;
                    final int nThreads = Math.min(Runtime.getRuntime().availableProcessors(), Math.max(1, length / 50_000));
                    final int chunk = (length + nThreads - 1) / nThreads;
                    Future<?>[] futures = FUTURES_CACHE.get();
                    if (futures.length < nThreads) {
                        futures = new Future<?>[nThreads];
                        FUTURES_CACHE.set(futures);
                    }
                    
                    for (int t = 0; t < nThreads; t++) {
                        final int start = t * chunk;
                        final int end = Math.min(start + chunk, length);
                        futures[t] = executor.submit(() -> applyElementWiseKernel(in.data, out.data, start, end, op));
                    }
                    
                    for (int t = 0; t < nThreads; t++) {
                        try {
                            futures[t].get();
                            futures[t] = null;
                        } catch (ExecutionException e) {
                            throw new RuntimeException("Element-wise kernel failed", e.getCause() != null ? e.getCause() : e);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException("Interrupted", e);
                        }
                    }
                }
                
                private void applyElementWiseKernel(double[] in, double[] out, int start, int end, String op) {
                    int i = start;
                    if (VLEN > 1) {
                        int loopBound = start + SPECIES.loopBound(end - start);
                        for (; i < loopBound; i += VLEN) {
                            var v = DoubleVector.fromArray(SPECIES, in, i);
                            var r = switch (op.toLowerCase()) {
                                case "relu" -> v.max(0.0);
                                case "gelu" -> FlatMatrix.VectorMath.geluVector(v);
                                case "exp" -> v.lanewise(VectorOperators.EXP);
                                case "log" -> v.lanewise(VectorOperators.LOG);
                                case "tanh" -> v.lanewise(VectorOperators.TANH);
                                case "sin" -> v.lanewise(VectorOperators.SIN);
                                default -> throw new UnsupportedOperationException(op);
                            };
                            r.intoArray(out, i);
                        }
                    }
                    for (; i < end; i++) {
                        out[i] = switch (op.toLowerCase()) {
                            case "relu" -> Math.max(0, in[i]);
                            case "exp" -> Math.exp(in[i]);
                            case "log" -> Math.log(in[i]);
                            case "tanh" -> Math.tanh(in[i]);
                            case "sin" -> Math.sin(in[i]);
                            default -> throw new UnsupportedOperationException(op);
                        };
                    }
                }
            };
        } catch (Throwable ex) {
            System.getLogger(VectorTurboEvaluator.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
            return null;
        }
    }

  
    // ====================== FULL INSTRUCTION SET ======================
    private interface Instruction {
        void execute(double[][] vars, int offset, DoubleVector[] stack, int sp);
    }

    private record ConstantInstr(double value) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            stack[sp] = DoubleVector.broadcast(SPECIES, value);
        }
    }

    private record LoadInstr(int slot) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            stack[sp] = DoubleVector.fromArray(SPECIES, vars[slot], offset);
        }
    }

    private record BinaryInstr(Instruction left, Instruction right, char op) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            right.execute(vars, offset, stack, sp);
            DoubleVector r = stack[sp];
            left.execute(vars, offset, stack, sp);
            DoubleVector l = stack[sp];

            DoubleVector res = switch (op) {
                case '+' -> l.add(r);
                case '-' -> l.sub(r);
                case '*' -> l.mul(r);
                case '/' -> l.div(r);
                case '^' -> l.pow(r);
                default -> throw new UnsupportedOperationException("Binary op: " + op);
            };
            stack[sp] = res;
        }
    }

    private record BinaryFuncInstr(Instruction left, Instruction right, String name) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            right.execute(vars, offset, stack, sp);
            DoubleVector r = stack[sp];
            left.execute(vars, offset, stack, sp);
            DoubleVector l = stack[sp];

            DoubleVector res = switch (name.toLowerCase()) {
                case "pow" -> l.pow(r);
                case "min" -> l.min(r);
                case "max" -> l.max(r);
                default -> throw new UnsupportedOperationException("Binary func: " + name);
            };
            stack[sp] = res;
        }
    }

    private record UnaryInstr(Instruction operand, String name) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            operand.execute(vars, offset, stack, sp);
            DoubleVector v = stack[sp];

            DoubleVector res = switch (name.toLowerCase()) {
                case "sin" -> v.lanewise(VectorOperators.SIN);
                case "cos" -> v.lanewise(VectorOperators.COS);
                case "tanh" -> v.lanewise(VectorOperators.TANH);
                case "exp" -> v.lanewise(VectorOperators.EXP);
                case "log" -> v.lanewise(VectorOperators.LOG);
                case "sqrt", "√" -> v.lanewise(VectorOperators.SQRT);
                case "abs" -> v.abs();
                default -> throw new UnsupportedOperationException("Unary: " + name);
            };
            stack[sp] = res;
        }
    }

    private record RemainderInstr(Instruction left, Instruction right) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            right.execute(vars, offset, stack, sp);
            DoubleVector b = stack[sp];
            left.execute(vars, offset, stack, sp);
            DoubleVector a = stack[sp];
            stack[sp] = execVectorRemainder(a, b);
        }
    }

    private record ComparisonInstr(Instruction left, Instruction right, char op) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            right.execute(vars, offset, stack, sp);
            DoubleVector r = stack[sp];
            left.execute(vars, offset, stack, sp);
            DoubleVector l = stack[sp];

            VectorMask<Double> mask = switch (op) {
                case '>' -> l.compare(VectorOperators.GT, r);
                case '<' -> l.compare(VectorOperators.LT, r);
                case '=' -> l.compare(VectorOperators.EQ, r);
                case '!' -> l.compare(VectorOperators.NE, r);
                case 'g' -> l.compare(VectorOperators.GE, r);
                case 'l' -> l.compare(VectorOperators.LE, r);
                default -> throw new UnsupportedOperationException("Comparison: " + op);
            };
            stack[sp] = DoubleVector.zero(SPECIES).blend(DoubleVector.broadcast(SPECIES, 1.0), mask);
        }
    }

    private record FmaInstr(Instruction a, Instruction b, Instruction c) implements Instruction {
        public void execute(double[][] vars, int offset, DoubleVector[] stack, int sp) {
            c.execute(vars, offset, stack, sp);
            DoubleVector cv = stack[sp];
            b.execute(vars, offset, stack, sp);
            DoubleVector bv = stack[sp];
            a.execute(vars, offset, stack, sp);
            DoubleVector av = stack[sp];
            stack[sp] = av.fma(bv, cv);
        }
    }

    public static DoubleVector execVectorRemainder(DoubleVector a, DoubleVector b) {
        DoubleVector q = a.div(b);
        LongVector t = (LongVector) q.convert(VectorOperators.Conversion.ofCast(double.class, long.class), 0);
        DoubleVector td = (DoubleVector) t.convert(VectorOperators.Conversion.ofCast(long.class, double.class), 0);
        return a.sub(td.mul(b));
    }

    private boolean isRelationalOp(char op) {
        return op == '>' || op == '<' || op == '=' || op == '!' || op == 'g' || op == 'l';
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