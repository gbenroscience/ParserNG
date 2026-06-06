package com.github.gbenroscience.parser.pro.turbo.tools;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.pro.turbo.SIMDCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import jdk.incubator.vector.*;

import java.util.*;
import java.util.concurrent.*;

/**
 * Full-featured SIMD evaluator for ParserNG. Uses a generic postfix-to-vector
 * compiler and ScalarTurboEvaluator1 for scalar fallback.
 */
public class VectorTurboEvaluator1 extends ScalarTurboEvaluator1 {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VLEN = SPECIES.length();

    @FunctionalInterface
    interface VectorNode {

        DoubleVector eval(double[][] vars, int index);
    }

    private final MathExpression expr;
    private final MathExpression.Token[] postfix;
    private final VectorNode compiledVector;

    // Thread-local cache for EvalResult
    private static final ThreadLocal<MathExpression.EvalResult> EVAL_RESULT_CACHE
            = ThreadLocal.withInitial(MathExpression.EvalResult::new);

    public VectorTurboEvaluator1(MathExpression me) {
        super(me);
        this.expr = me;
        this.postfix = me.getCachedPostfix();
        this.compiledVector = compileVector(postfix);
    }

    @Override
    public FastCompositeExpression compile() {

        try {
            final FastCompositeExpression fce = super.compile();

            return new SIMDCompositeExpression() {
                @Override
                public double applyScalar(double[] variables) {
                    return fce.applyScalar(variables);
                }

                @Override
                public MathExpression.EvalResult apply(double[] variables) {
                    return EVAL_RESULT_CACHE.get().wrap(applyScalar(variables));
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
                public void applyBulk(double[][] vars, double[] out) {
                    applyBulkInternal(vars, out, 0);
                }

                @Override
                public void applyBulk(double[][] vars, double[] out, int offset) {
                    applyBulkInternal(vars, out, offset);
                }

                private void applyBulkInternal(double[][] vars, double[] out, int offset) {
                    int length = vars[0].length;
                    int loopBound = SPECIES.loopBound(length);
                    int i = 0;
                    for (; i < loopBound; i += VLEN) {
                        DoubleVector v = compiledVector.eval(vars, i);
                        v.intoArray(out, offset + i);
                    }
                    // tail handled by ScalarTurboEvaluator1
                    for (; i < length; i++) {
                        out[offset + i] = fce.applyScalar(vars[i]);
                    }
                }

                @Override
                public void applyBulk(double[][] vars, double[] out, ExecutorService executor) {
                    if (executor == null || out.length < 25000) {
                        applyBulkInternal(vars, out, 0);
                        return;
                    }
                    int length = out.length;
                    int nThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                            Math.max(1, length / 25000));
                    int chunk = (length + nThreads - 1) / nThreads;
                    Future<?>[] futures = new Future<?>[nThreads];

                    for (int t = 0; t < nThreads; t++) {
                        final int start = t * chunk;
                        final int end = Math.min(start + chunk, length);
                        futures[t] = executor.submit(() -> {
                            int loopBound = start + SPECIES.loopBound(end - start);
                            int i = start;
                            for (; i < loopBound; i += VLEN) {
                                DoubleVector v = compiledVector.eval(vars, i);
                                v.intoArray(out, i);
                            }
                            for (; i < end; i++) {
                                out[i] = fce.applyScalar(vars[i]);
                            }
                        });
                    }

                    for (Future<?> f : futures) {
                        try {
                            f.get();
                        } catch (ExecutionException | InterruptedException e) {
                            throw new RuntimeException("Parallel execution failed", e);
                        }
                    }
                }
            };
        } catch (Throwable ex) {
            System.getLogger(VectorTurboEvaluator.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
            return null;
        }
    }

    // === Generic postfix-to-vector compiler ===
    private VectorNode compileVector(MathExpression.Token[] postfix) {
        Deque<VectorNode> stack = new ArrayDeque<>();
        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER -> {
                    double val = t.value;
                    stack.push((vars, i) -> DoubleVector.broadcast(SPECIES, val));
                }
                case MathExpression.Token.VARIABLE -> {
                    int idx = t.frameIndex;
                    stack.push((vars, i) -> DoubleVector.fromArray(SPECIES, vars[idx], i));
                }
                case MathExpression.Token.OPERATOR -> {
                    VectorNode r = stack.pop();
                    VectorNode l = stack.pop();
                    switch (t.opChar) {
                        case '+' ->
                            stack.push((vars, i) -> l.eval(vars, i).add(r.eval(vars, i)));
                        case '-' ->
                            stack.push((vars, i) -> l.eval(vars, i).sub(r.eval(vars, i)));
                        case '*' ->
                            stack.push((vars, i) -> l.eval(vars, i).mul(r.eval(vars, i)));
                        case '/' ->
                            stack.push((vars, i) -> l.eval(vars, i).div(r.eval(vars, i)));
                        // add more ops as needed
                    }
                }
                case MathExpression.Token.FUNCTION -> {
                    if (t.arity == 1) {
                        VectorNode arg = stack.pop();
                        switch (t.name.toLowerCase()) {
                            case "exp" ->
                                stack.push((vars, i) -> arg.eval(vars, i).lanewise(VectorOperators.EXP));
                            case "log" ->
                                stack.push((vars, i) -> arg.eval(vars, i).lanewise(VectorOperators.LOG));
                            case "sin" ->
                                stack.push((vars, i) -> arg.eval(vars, i).lanewise(VectorOperators.SIN));
                            case "tanh" ->
                                stack.push((vars, i) -> arg.eval(vars, i).lanewise(VectorOperators.TANH));
                            case "relu" ->
                                stack.push((vars, i) -> arg.eval(vars, i).max(0.0));
                        }
                    } else if (t.arity == 3 && "vma".equalsIgnoreCase(t.name)) {
                        VectorNode c = stack.pop();
                        VectorNode b = stack.pop();
                        VectorNode a = stack.pop();
                        stack.push((vars, i) -> a.eval(vars, i).fma(b.eval(vars, i), c.eval(vars, i)));
                    } else if (t.arity == 3 && "if".equalsIgnoreCase(t.name)) {
                        VectorNode f = stack.pop();
                        VectorNode tr = stack.pop();
                        VectorNode cond = stack.pop();
                        stack.push((vars, i) -> {
                            DoubleVector c = cond.eval(vars, i);
                            DoubleVector tBranch = tr.eval(vars, i);
                            DoubleVector fBranch = f.eval(vars, i);

                            // Compare returns a VectorMask<Double>
                            VectorMask<Double> mask = c.compare(VectorOperators.GT, 0.0);

                            // FIX: Call blend on the FALSE branch vector, passing the TRUE branch vector and the mask.
                            // This means: Start with fBranch, and select lanes from tBranch where mask is true.
                            return fBranch.blend(tBranch, mask);
                        });
                    }
                }
            }
        }
        return stack.pop();
    }
}
