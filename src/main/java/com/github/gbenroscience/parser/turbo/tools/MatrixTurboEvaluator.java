/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser.turbo.tools;

/**
 *
 * @author GBEMIRO
 */
import com.github.gbenroscience.math.geom.Direction; 
import com.github.gbenroscience.math.geom.Point;
import com.github.gbenroscience.math.geom.ROTOR;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.EvalResult;
import com.github.gbenroscience.parser.ParserResult;
import com.github.gbenroscience.parser.TYPE;
import static com.github.gbenroscience.parser.TYPE.ALGEBRAIC_EXPRESSION;
import static com.github.gbenroscience.parser.TYPE.MATRIX;
import com.github.gbenroscience.parser.Variable;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.Method;
import com.github.gbenroscience.parser.methods.MethodRegistry;
import com.github.gbenroscience.util.ErrorLog;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.VariableManager;
import java.lang.invoke.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Allocation-free Turbo compiler optimized for ParserNG's flat-array Matrix
 * implementation. Uses compile-time bound ResultCaches and Zero-Argument
 * MethodHandle trees to maximize JIT inlining and execution speed.
 *
 * @author GBEMIRO
 */
public final class MatrixTurboEvaluator implements TurboExpressionEvaluator {

    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
    private boolean willFoldConstants;

    protected final double[] turboArgs;
    protected final int[] slots;
    private ErrorLog errorLog = new ErrorLog();
    private MathExpression.Token[] postfix;
    private MethodHandle finalHandle;
    // Enables True AST Inlining for user-defined functions
    private MethodHandle[] inlinedVariables;

    public MatrixTurboEvaluator(MathExpression me) {
        this.postfix = me.getCachedPostfix();
        this.willFoldConstants = me.isWillFoldConstants();
        int num_vars = me.getVariablesNames().length;
        slots = me.getSlots();
        turboArgs = new double[num_vars];
        me.copyErrorLogTo(errorLog);
    }

    // 1. ThreadLocal holding a reusable array of EvalResults to avoid GC pressure
    private static final ThreadLocal<MathExpression.EvalResult[]> WRAPPER_CACHE
            = ThreadLocal.withInitial(() -> {
                MathExpression.EvalResult[] arr = new MathExpression.EvalResult[12];
                for (int i = 0; i < 12; i++) {
                    arr[i] = new MathExpression.EvalResult();
                }
                return arr;
            });

    // ========== THE RESULT CACHE ==========
    /**
     * Holds the mutable state for a single node in the execution tree. Bound
     * into the MethodHandle chain at compile-time.
     */
    public static final class ResultCache {

        public final EvalResult result = new EvalResult();
        private double lastDeterminant;
        public double[] matrixData;
        public Matrix matrix;

        public double[] eigenValueBuffer;
        public int[] intBuffer;
        // Secondary buffer for re-entrant operations like Matrix Power
        private double[] matrixData2;
        private Matrix matrix2;

        // Secondary buffer for re-entrant operations like Matrix Power
        private double[] matrixData3;
        private Matrix matrix3;

        public Matrix getMatrixBuffer(int rows, int cols) {
            int size = rows * cols;
            if (matrixData == null || matrixData.length != size) {
                matrixData = new double[size];
                matrix = new Matrix(matrixData, rows, cols);
            } else if (matrix.getRows() != rows || matrix.getCols() != cols) {
                matrix = new Matrix(matrixData, rows, cols);
            }
            return matrix;
        }

        public Matrix getSecondaryBuffer(int rows, int cols) {
            int size = rows * cols;
            if (matrixData2 == null || matrixData2.length != size) {
                matrixData2 = new double[size];
                matrix2 = new Matrix(matrixData2, rows, cols);
            } else if (matrix2.getRows() != rows || matrix2.getCols() != cols) {
                matrix2 = new Matrix(matrixData2, rows, cols);
            }
            return matrix2;
        }

        public Matrix getTertiaryBuffer(int rows, int cols) {
            int size = rows * cols;
            if (matrixData3 == null || matrixData3.length != size) {
                matrixData3 = new double[size];
                matrix3 = new Matrix(matrixData3, rows, cols);
            } else if (matrix3.getRows() != rows || matrix3.getCols() != cols) {
                matrix3 = new Matrix(matrixData3, rows, cols);
            }
            return matrix3;
        }

        public double[] getEigenBuffer(int n) {
            if (eigenValueBuffer == null || eigenValueBuffer.length != n) {
                eigenValueBuffer = new double[n];
            }
            return eigenValueBuffer;
        }

        public int[] getIntBuffer(int n) {
            if (intBuffer == null || intBuffer.length != n) {
                intBuffer = new int[n];
            }
            return intBuffer;
        }

        public EvalResult[] argsBuffer;

        public EvalResult[] getArgsBuffer(int arity) {
            if (argsBuffer == null || argsBuffer.length != arity) {
                argsBuffer = new EvalResult[arity];
            }
            return argsBuffer;
        }

    }

    public void setWillFoldConstants(boolean willFoldConstants) {
        this.willFoldConstants = willFoldConstants;
    }

    public boolean isWillFoldConstants() {
        return willFoldConstants;
    }

    public double[] getTurboArgs() {
        return turboArgs;
    }

    public static MathExpression.EvalResult invokeRegistryMethod(int methodId, EvalResult[] argsValues) {
        int arity = argsValues.length;
        MathExpression.EvalResult resultContainer = new MathExpression.EvalResult();
        MethodRegistry.getAction(methodId).calc(resultContainer, arity, argsValues);
        return resultContainer;
    }

    // ========== COMPILER CORE ==========
    @Override
    public FastCompositeExpression compile() throws Throwable {

        Stack<MethodHandle> stack = new Stack<>();

        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    MethodHandle leaf;
                    if (t.name != null && !t.name.isEmpty()) {
                        if (t.v != null) {
                            // TRUE AST INLINING: If we are inside an inlined function, grab the caller's handle!
                            if (inlinedVariables != null && t.frameIndex < inlinedVariables.length) {
                                leaf = inlinedVariables[t.frameIndex];
                            } else {
                                leaf = compileVariableLookupByIndex(t.frameIndex);
                            }
                        } else {
                            Function f = FunctionManager.lookUp(t.name);
                            if (f != null && f.getMatrix() != null) {
                                MathExpression.EvalResult res = new MathExpression.EvalResult().wrap(f.getMatrix());
                                leaf = MethodHandles.constant(MathExpression.EvalResult.class, res);
                            } else {
                                MathExpression.EvalResult res = new MathExpression.EvalResult().wrap(t.name);
                                leaf = MethodHandles.constant(MathExpression.EvalResult.class, res);
                            }
                        }
                    } else {
                        MathExpression.EvalResult res = new MathExpression.EvalResult().wrap(t.value);
                        leaf = MethodHandles.constant(MathExpression.EvalResult.class, res);
                    }
                    stack.push(leaf);
                    break;
                case MathExpression.Token.OPERATOR:
                    if (t.isPostfix) {
                        MethodHandle operand = stack.pop();
                        stack.push(compileUnaryOpOnEvalResult(t.opChar, operand));
                    } else {
                        MethodHandle right = stack.pop();
                        MethodHandle left = stack.pop();
                        stack.push(compileBinaryOpOnEvalResult(t.opChar, left, right));
                    }
                    break;

                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD:
                    String name = t.name.toLowerCase();

                    if (Method.isPureStatsMethod(name)) {
                        for (int i = 0; i < t.arity; i++) {
                            stack.pop();
                        }
                        String[] rawArgs = t.getRawArgs();
                        double[] data = new double[t.arity];
                        for (int i = 0; i < t.arity; i++) {
                            data[i] = Double.parseDouble(rawArgs[i]);
                        }

                        EvalResult precalculated;
                        if (name.equals(Declarations.SORT) || name.equals(Declarations.MODE)) {
                            double[] vecResult = executeVectorStatAtCompileTime(name, data);
                            precalculated = new EvalResult();
                            precalculated.wrap(vecResult);
                        } else {
                            double scalarResult = executeScalarStatAtCompileTime(name, data);
                            precalculated = new EvalResult();
                            precalculated.wrap(scalarResult);
                        }
                        stack.push(MethodHandles.constant(EvalResult.class, precalculated));
                        break;
                    } else if (t.name.equalsIgnoreCase("print")) {
                        for (int i = 0; i < t.arity; i++) {
                            stack.pop();
                        }

                        MethodHandle bridge = LOOKUP.findStatic(MatrixTurboEvaluator.class, "executePrint",
                                MethodType.methodType(EvalResult.class, String[].class));
                        String[] rawArgs = t.getRawArgs();
                        stack.push(MethodHandles.insertArguments(bridge, 0, (Object) rawArgs));

                    } else if (Method.isMatrixMethod(t.name) || t.name.equals(Declarations.ROTOR)) {
                        MethodHandle[] args = new MethodHandle[t.arity];
                        for (int i = t.arity - 1; i >= 0; i--) {
                            args[i] = stack.pop();
                        }
                        stack.push(compileMatrixFunction(t, args));
                    } else {
                        MethodHandle[] args = new MethodHandle[t.arity];
                        for (int i = t.arity - 1; i >= 0; i--) {
                            args[i] = stack.pop();
                        }

                        Function userFunc = FunctionManager.lookUp(t.name);
                        if (userFunc != null) {
                            MathExpression bodyExpr = userFunc.getMathExpression();
                            MatrixTurboEvaluator inlineEvaluator = new MatrixTurboEvaluator(bodyExpr);
                            inlineEvaluator.inlinedVariables = args;
                            inlineEvaluator.compile();

                            MethodHandle inlinedHandle = inlineEvaluator.finalHandle;

                            // CRITICAL FIX: Ensure the inlined handle matches (double[])EvalResult
                            final MethodType expectedType = MethodType.methodType(EvalResult.class, double[].class);
                            if (!inlinedHandle.type().equals(expectedType)) {
                                if (inlinedHandle.type().parameterCount() == 0) {
                                    inlinedHandle = MethodHandles.dropArguments(inlinedHandle, 0, double[].class);
                                } else {
                                    inlinedHandle = inlinedHandle.asType(expectedType);
                                }
                            }

                            stack.push(mergeDoubleArrays(inlinedHandle));
                        } else {
                            stack.push(compileFunction(t, args));
                        }
                    }
                    break;
                default:
                    errorLog.info("Unknown Token Kind: " + t.kind + " Name: " + t.name);
                    break;
            }
        }

        if (stack.size() != 1) {
            String err = "Invalid postfix stack state.";
            errorLog.info(err);
            throw new IllegalArgumentException(err);
        }

        this.finalHandle = stack.pop();

        // Ensure ALL redundant double[] arguments are securely merged down to 1
        finalHandle = mergeDoubleArrays(finalHandle);

        // Enforce the strict public interface (double[])EvalResult
        final MethodType expectedType = MethodType.methodType(EvalResult.class, double[].class);
        if (!finalHandle.type().equals(expectedType)) {
            if (finalHandle.type().parameterCount() == 0) {
                finalHandle = MethodHandles.dropArguments(finalHandle, 0, double[].class);
            } else {
                finalHandle = finalHandle.asType(expectedType);
            }
        }

        return new FastCompositeExpression() {
            double args[] = new double[turboArgs == null ? 0 : turboArgs.length];

            private void loadVars(double[] variables) {
                // Safety check: ensure we don't null pointer if arrays aren't initialized
                if (variables == null || turboArgs == null || slots == null) {
                    return;
                }

                // Use the smallest length to prevent print-of-bounds access
                int limit = Math.min(variables.length, Math.min(args.length, slots.length));

                for (int i = 0; i < limit; i++) {
                    args[slots[i]] = variables[i];
                }
            }

            @Override
            public EvalResult apply(double[] variables) {
                try {
                    loadVars(variables);
                    // OPTIMIZED: Invoke with the single double[] array.
                    return (EvalResult) finalHandle.invokeExact(args);
                } catch (Throwable e) {
                    errorLog.error(e);
                    throw new RuntimeException("Turbo matrix execution failed", e);
                }
            }

            @Override
            public double applyScalar(double[] variables) {
                return apply(variables).scalar;
            }

            @Override
            public String checkErrorLogs() {
                String logs = errorLog.getLogs();
                errorLog.print();
                return logs;
            }

            @Override
            public TurboExpressionEvaluator getCompiler() {
                return MatrixTurboEvaluator.this;
            }
        };
    }

    /**
     * Safely collapses any duplicate double[] arguments down to a single
     * instance. Prevents MethodHandle signature explosion during AST
     * construction.
     */
    private static MethodHandle mergeDoubleArrays(MethodHandle mh) {
        MethodType type = mh.type();
        int doubleArrayCount = 0;
        for (int i = 0; i < type.parameterCount(); i++) {
            if (type.parameterType(i) == double[].class) {
                doubleArrayCount++;
            }
        }

        if (doubleArrayCount <= 1) {
            return mh; // Nothing to merge
        }

        Class<?>[] newParams = new Class<?>[type.parameterCount() - doubleArrayCount + 1];
        int[] reorder = new int[type.parameterCount()];

        int firstDoubleIdx = -1;
        int newIdx = 0;

        for (int i = 0; i < type.parameterCount(); i++) {
            if (type.parameterType(i) == double[].class) {
                if (firstDoubleIdx == -1) {
                    firstDoubleIdx = newIdx;
                    newParams[newIdx] = double[].class;
                    reorder[i] = newIdx;
                    newIdx++;
                } else {
                    reorder[i] = firstDoubleIdx;
                }
            } else {
                newParams[newIdx] = type.parameterType(i);
                reorder[i] = newIdx;
                newIdx++;
            }
        }

        MethodType newType = MethodType.methodType(type.returnType(), newParams);
        return MethodHandles.permuteArguments(mh, newType, reorder);
    }

    /**
     * Safely locates the next target EvalResult parameter and binds it, merging
     * arrays afterward.
     */
    private static MethodHandle collectNextEvalResult(MethodHandle target, MethodHandle valueToInject) {
        int fillPos = -1;
        for (int p = 0; p < target.type().parameterCount(); p++) {
            if (target.type().parameterType(p) == EvalResult.class) {
                fillPos = p;
                break;
            }
        }
        if (fillPos == -1) {
            throw new IllegalStateException("Internal Error: No EvalResult parameter found to collect into.");
        }
        MethodHandle collected = MethodHandles.collectArguments(target, fillPos, valueToInject);
        return mergeDoubleArrays(collected);
    }

    private MethodHandle compileVariableLookupByIndex(final int index) throws NoSuchMethodException, IllegalAccessException {
        if (inlinedVariables != null && index >= 0 && index < inlinedVariables.length) {
            return inlinedVariables[index];
        }

        MethodHandle getter = LOOKUP.findStatic(MatrixTurboEvaluator.class, "getVar",
                MethodType.methodType(double.class, double[].class, int.class));
        getter = MethodHandles.insertArguments(getter, 1, index);

        MethodHandle wrapper = LOOKUP.findStatic(MatrixTurboEvaluator.class, "wrapDouble",
                MethodType.methodType(EvalResult.class, double.class, ResultCache.class));
        wrapper = MethodHandles.insertArguments(wrapper, 1, new ResultCache());

        // Chain: (double[]) -> double -> EvalResult
        return MethodHandles.filterReturnValue(getter, wrapper);
    }

    /**
     * Rebuilt using the unified collectNextEvalResult logic!
     */
    private static MethodHandle combineArgsToArray(MethodHandle[] args, ResultCache cache) throws Throwable {
        int arity = args.length;
        MethodHandle packer;

        if (arity <= 5) {
            Class<?>[] ptypes = new Class<?>[arity + 1];
            ptypes[0] = ResultCache.class;
            for (int i = 0; i < arity; i++) {
                ptypes[i + 1] = EvalResult.class;
            }
            packer = LOOKUP.findStatic(MatrixTurboEvaluator.class, "packArgs" + arity,
                    MethodType.methodType(EvalResult[].class, ptypes));
        } else {
            packer = LOOKUP.findStatic(MatrixTurboEvaluator.class, "packArgsN",
                    MethodType.methodType(EvalResult[].class, ResultCache.class, EvalResult[].class)).asVarargsCollector(EvalResult[].class);
            Class<?>[] ptypes = new Class<?>[arity + 1];
            ptypes[0] = ResultCache.class;
            for (int i = 0; i < arity; i++) {
                ptypes[i + 1] = EvalResult.class;
            }
            packer = packer.asType(MethodType.methodType(EvalResult[].class, ptypes));
        }

        MethodHandle result = MethodHandles.insertArguments(packer, 0, cache);

        for (int i = 0; i < arity; i++) {
            result = collectNextEvalResult(result, args[i]);
        }
        return result;
    }

    public static EvalResult wrapDouble(double val, ResultCache cache) {
        return cache.result.wrap(val);
    }

    public static double getVar(double[] vars, int index) {
        return vars[index];
    }

    private MethodHandle compileBinaryOpOnEvalResult(char op, MethodHandle left, MethodHandle right) throws Throwable {
        MethodHandle dispatcher = LOOKUP.findStatic(MatrixTurboEvaluator.class, "dispatchBinaryOp",
                MethodType.methodType(EvalResult.class, char.class, EvalResult.class, EvalResult.class, ResultCache.class));

        ResultCache nodeCache = new ResultCache();
        dispatcher = MethodHandles.insertArguments(dispatcher, 3, nodeCache);
        dispatcher = MethodHandles.insertArguments(dispatcher, 0, op);

        // Fill Left then Right using the safe array-merging collector
        dispatcher = collectNextEvalResult(dispatcher, left);
        dispatcher = collectNextEvalResult(dispatcher, right);

        return dispatcher;
    }

    private MethodHandle compileUnaryOpOnEvalResult(char op, MethodHandle operand) throws Throwable {
        MethodHandle dispatcher = LOOKUP.findStatic(MatrixTurboEvaluator.class, "dispatchUnaryOp",
                MethodType.methodType(EvalResult.class, char.class, EvalResult.class, ResultCache.class));

        ResultCache nodeCache = new ResultCache();
        dispatcher = MethodHandles.insertArguments(dispatcher, 2, nodeCache);
        dispatcher = MethodHandles.insertArguments(dispatcher, 0, op);

        return collectNextEvalResult(dispatcher, operand);
    }

    private MethodHandle compileMatrixFunction(MathExpression.Token t, MethodHandle[] args) throws Throwable {
        String funcName = t.name.toLowerCase();
        MethodHandle dispatcher = LOOKUP.findStatic(MatrixTurboEvaluator.class, "dispatchMatrixFunction",
                MethodType.methodType(EvalResult.class, EvalResult[].class, String.class, ResultCache.class));

        ResultCache nodeCache = new ResultCache();
        MethodHandle boundBridge = MethodHandles.insertArguments(dispatcher, 1, funcName, nodeCache);

        MethodHandle combinedArgs = combineArgsToArray(args, nodeCache);
        dispatcher = MethodHandles.collectArguments(boundBridge, 0, combinedArgs);
        return mergeDoubleArrays(dispatcher);
    }

    private static MethodHandle compileFunction(MathExpression.Token t, MethodHandle[] args) throws Throwable {
        int methodId = MethodRegistry.getMethodID(t.name);
        MethodHandle bridge = LOOKUP.findStatic(MatrixTurboEvaluator.class, "invokeRegistryMethod",
                MethodType.methodType(MathExpression.EvalResult.class, int.class, ResultCache.class, EvalResult[].class));

        ResultCache nodeCache = new ResultCache();
        MethodHandle boundBridge = MethodHandles.insertArguments(bridge, 0, methodId, nodeCache);

        MethodHandle combinedArgs = combineArgsToArray(args, nodeCache);
        MethodHandle dispatcher = MethodHandles.collectArguments(boundBridge, 0, combinedArgs);
        return mergeDoubleArrays(dispatcher);
    }

    static MathExpression.EvalResult executePrint(String[] args) {
        MathExpression.EvalResult ctx = new EvalResult();
        StringBuilder sb = new StringBuilder();
        for (String arg : args) {
            Function v = FunctionManager.lookUp(arg);
            if (v != null) {
                switch (v.getType()) {
                    case ALGEBRAIC_EXPRESSION:
                        sb.append(v.toString()).append("\n");
                        break;
                    case MATRIX:
                        sb.append(v.getName()).append("=").append(v.getMatrix() != null ? v.getMatrix().toString() : "null").append("\n");
                        break;
                    default:
                        sb.append(v.toString()).append("\n");
                        break;
                }
                continue;
            }
            Variable myVar = VariableManager.lookUp(arg);
            if (myVar != null) {
                sb.append(myVar.toString()).append("\n");
            } else if (com.github.gbenroscience.parser.Number.isNumber(arg)) {
                sb.append(arg).append("\n");
            } else {
                sb.append("null").append("\n");
            }
        }
        String v = sb.toString();
        System.out.println(v);
        return ctx.wrap(v);
    }

    public static EvalResult[] collectArgsArray(EvalResult... args) {
        return args;
    }

    /////////////////////////////////////////////////////////////////
    public static MathExpression.EvalResult invokeRegistryMethod(int methodId, ResultCache cache, EvalResult[] argsValues) {
        MethodRegistry.getAction(methodId).calc(cache.result, argsValues.length, argsValues);
        return cache.result;
    }

    public static EvalResult[] packArgs0(ResultCache cache) {
        return cache.getArgsBuffer(0);
    }

    public static EvalResult[] packArgs1(ResultCache cache, EvalResult a) {
        EvalResult[] buf = cache.getArgsBuffer(1);
        buf[0] = a;
        return buf;
    }

    public static EvalResult[] packArgs2(ResultCache cache, EvalResult a, EvalResult b) {
        EvalResult[] buf = cache.getArgsBuffer(2);
        buf[0] = a;
        buf[1] = b;
        return buf;
    }

    public static EvalResult[] packArgs3(ResultCache cache, EvalResult a, EvalResult b, EvalResult c) {
        EvalResult[] buf = cache.getArgsBuffer(3);
        buf[0] = a;
        buf[1] = b;
        buf[2] = c;
        return buf;
    }

    public static EvalResult[] packArgs4(ResultCache cache, EvalResult a, EvalResult b, EvalResult c, EvalResult d) {
        EvalResult[] buf = cache.getArgsBuffer(4);
        buf[0] = a;
        buf[1] = b;
        buf[2] = c;
        buf[3] = d;
        return buf;
    }

    public static EvalResult[] packArgs5(ResultCache cache, EvalResult a, EvalResult b, EvalResult c, EvalResult d, EvalResult e) {
        EvalResult[] buf = cache.getArgsBuffer(5);
        buf[0] = a;
        buf[1] = b;
        buf[2] = c;
        buf[3] = d;
        buf[4] = e;
        return buf;
    }

    public static EvalResult[] packArgsN(ResultCache cache, EvalResult... args) {
        EvalResult[] buf = cache.getArgsBuffer(args.length);
        System.arraycopy(args, 0, buf, 0, args.length);
        return buf;
    }

    //////////////////////////////

    // ========== RUNTIME DISPATCHERS ==========
    private static EvalResult dispatchBinaryOp(char op, EvalResult left, EvalResult right, ResultCache cache) {
        int leftType = left.type;
        int rightType = right.type;

        switch (op) {
            case '+':
                if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(left.scalar + right.scalar);
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    cache.result.wrap(flatMatrixAdd(left.matrix, right.matrix, cache));
                }
                break;

            case '-':
                if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(left.scalar - right.scalar);
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    cache.result.wrap(flatMatrixSubtract(left.matrix, right.matrix, cache));
                }
                break;

            case '*':
                if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(left.scalar * right.scalar);
                } else if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_MATRIX) {
                    cache.result.wrap(flatMatrixScalarMultiply(left.scalar, right.matrix, cache));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(flatMatrixScalarMultiply(right.scalar, left.matrix, cache));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    Matrix out = cache.getMatrixBuffer(left.matrix.getRows(), right.matrix.getCols());
                    cache.result.wrap(flatMatrixMultiply(left.matrix, right.matrix, out));
                }
                break;

            case '/':
                if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(left.scalar / right.scalar);
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    double det = computeDeterminantTurbo(right.matrix, cache);
                    if (Math.abs(det) < 1e-15) {
                        throw new ArithmeticException("Matrix B is singular");
                    }
                    Matrix A = left.matrix;
                    Matrix B = right.matrix;
                    // 1. Calculate the inverse first. 
                    // This will occupy the primary MatrixBuffer.
                    Matrix invB = flatMatrixInverseLUTurbo(B, cache);
                    // 2. Request a DIFFERENT buffer for the final output.
                    // Using getTertiaryBuffer ensures there is no overlap with invB.
                    Matrix out = cache.getTertiaryBuffer(A.getRows(), invB.getCols());
                    // 3. Perform the multiplication.
                    // A (source), invB (primary buffer), out (tertiary buffer) are now independent.
                    return cache.result.wrap(flatMatrixMultiply(A, invB, out));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
                    if (Math.abs(right.scalar) < 1e-15) {
                        throw new ArithmeticException("Division by zero scalar");
                    }
                    cache.result.wrap(flatMatrixScalarMultiply(1.0 / right.scalar, left.matrix, cache));
                } else if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_MATRIX) {
                    // Use turbo determinant
                    //double det = right.matrix.determinant(); // Or use computeDeterminantTurbo if available for the full matrix
                    double det = computeDeterminantTurbo(right.matrix, cache);
                    if (Math.abs(det) < 1e-15) {
                        throw new ArithmeticException("Matrix is singular");
                    }

                    // Use turbo adjoint buffers instead of right.matrix.adjoint()
                    int n = right.matrix.getRows();
                    Matrix adjB = cache.getMatrixBuffer(n, n);
                    Matrix minorBuf = cache.getSecondaryBuffer(n - 1, n - 1);
                    computeAdjointInto(right.matrix, adjB, minorBuf, cache);

                    cache.result.wrap(flatMatrixScalarMultiply(left.scalar / det, adjB, cache));
                }
                break;

            case '^':
                if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(Math.pow(left.scalar, right.scalar));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(flatMatrixPower(left.matrix, right.scalar, cache));
                }
                break;

            default:
                throw new UnsupportedOperationException("Operator not implemented: " + op);
        }
        return cache.result;
    }

    private static void computeAdjointInto(Matrix input, Matrix adjoint, Matrix minorBuf, ResultCache cache) {
        int n = input.getRows();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fillMinor(input.getFlatArray(), minorBuf.getFlatArray(), i, j, n);
                double minorDet = computeDeterminantTurbo(minorBuf, cache);
                // This safely applies the alternating signs you were missing
                adjoint.getFlatArray()[j * n + i] = (((i + j) % 2 == 0) ? 1.0 : -1.0) * minorDet;
            }
        }
    }

    private static Matrix flatMatrixScalarMultiply(double scalar, Matrix m, ResultCache cache) {
        double[] mF = m.getFlatArray();
        Matrix out = cache.getMatrixBuffer(m.getRows(), m.getCols());
        double[] resF = out.getFlatArray();
        for (int i = 0; i < mF.length; i++) {
            resF[i] = scalar * mF[i];
        }
        return out;
    }

    private static Matrix flatMatrixInverseLUTurbo(Matrix input, ResultCache cache) {
        final int n = input.getRows();
        if (n != input.getCols()) {
            throw new IllegalArgumentException("Matrix must be square for inversion.");
        }

        Matrix luMat = cache.getSecondaryBuffer(n, n);
        Matrix invMat = cache.getMatrixBuffer(n, n);
        double[] lu = luMat.getFlatArray();
        double[] inv = invMat.getFlatArray();
        double[] work = cache.getEigenBuffer(n);
        int[] pivot = cache.getIntBuffer(n);

        double det = 1.0; // Track the determinant
        int swapCount = 0;

        System.arraycopy(input.getFlatArray(), 0, lu, 0, n * n);
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }

        // === LU DECOMPOSITION ===
        for (int k = 0; k < n; k++) {
            int kOff = k * n;
            int pivotRow = k;
            double maxVal = Math.abs(lu[kOff + k]);

            for (int i = k + 1; i < n; i++) {
                double absVal = Math.abs(lu[i * n + k]);
                if (absVal > maxVal) {
                    maxVal = absVal;
                    pivotRow = i;
                }
            }

            if (maxVal < 1e-16) { // Slightly more lenient threshold
                cache.lastDeterminant = 0;
                throw new ArithmeticException("Matrix is singular.");
            }

            if (pivotRow != k) {
                int tmp = pivot[k];
                pivot[k] = pivot[pivotRow];
                pivot[pivotRow] = tmp;

                int pOff = pivotRow * n;
                // FIX: Must swap the FULL row (start from 0) to keep L consistent
                for (int j = 0; j < n; j++) {
                    double t = lu[kOff + j];
                    lu[kOff + j] = lu[pOff + j];
                    lu[pOff + j] = t;
                }
                swapCount++; // Increment swap counter
            }

            // The diagonal element of U is lu[kOff + k]
            det *= lu[kOff + k];
            double invPivot = 1.0 / lu[kOff + k];
            for (int i = k + 1; i < n; i++) {
                int iOff = i * n;
                double mult = (lu[iOff + k] *= invPivot);
                for (int j = k + 1; j < n; j++) {
                    lu[iOff + j] -= mult * lu[kOff + j];
                }
            }
        }

        double[] uInvDiag = cache.getTertiaryBuffer(1, n).getFlatArray();
        for (int i = 0; i < n; i++) {
            uInvDiag[i] = 1.0 / lu[i * n + i];
        }

        if ((swapCount & 1) != 0) {
            det = -det;
        }
        cache.lastDeterminant = det; // Cache the determinant for reuse

        // === SOLVE PHASE ===
        // Loop over the current pivot positions instead of columns
        for (int pIdx = 0; pIdx < n; pIdx++) {
            // pivot[pIdx] tells us exactly which original column this represents
            int j = pivot[pIdx];

            Arrays.fill(work, 0.0);
            work[pIdx] = 1.0;

            // Forward substitution: Solve L*y = Pb
            // Start from pIdx + 1 because work[0...pIdx-1] is all zeros
            for (int i = pIdx + 1; i < n; i++) {
                double sum = work[i]; // effectively 0.0 initially
                int iOff = i * n;
                for (int k = pIdx; k < i; k++) {
                    sum -= lu[iOff + k] * work[k];
                }
                work[i] = sum;
            }

            // Backward substitution: Solve U*x = y
            for (int i = n - 1; i >= 0; i--) {
                double sum = work[i];
                int iOff = i * n;
                for (int k = i + 1; k < n; k++) {
                    sum -= lu[iOff + k] * work[k];
                }
                work[i] = sum * uInvDiag[i];
            }

            // Store result strictly into the correct original column 'j'
            for (int i = 0; i < n; i++) {
                inv[i * n + j] = work[i];
            }
        }

        return invMat;
    }

    private static double computeDeterminantTurbo(Matrix minor, ResultCache cache) {
        if (minor.getRows() != minor.getCols()) {
            throw new IllegalArgumentException("Determinant is only defined for square matrices.");
        }
        int n = minor.getRows();
        if (n == 1) {
            return minor.getFlatArray()[0];
        }
        Matrix work = cache.getTertiaryBuffer(n, n);
        System.arraycopy(minor.getFlatArray(), 0, work.getFlatArray(), 0, n * n);
        double det = 1.0;
        double[] data = work.getFlatArray();
        for (int i = 0; i < n; i++) {
            int pivot = i;
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(data[j * n + i]) > Math.abs(data[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (pivot != i) {
                work.swapRow(i, pivot);
                det *= -1;
            }
            double v = data[i * n + i];
            if (Math.abs(v) < 1e-14) {
                return 0;
            }
            det *= v;
            for (int row = i + 1; row < n; row++) {
                double factor = data[row * n + i] / v;
                for (int col = i + 1; col < n; col++) {
                    data[row * n + col] -= factor * data[i * n + col];
                }
            }
        }
        return det;
    }

    private static void solveEquationInto(Matrix input, Matrix solnMatrix, ResultCache cache) {
        int rows = input.getRows();
        int cols = input.getCols();
        Matrix matrixLoader = cache.getSecondaryBuffer(rows, cols);
        System.arraycopy(input.getFlatArray(), 0, matrixLoader.getFlatArray(), 0, rows * cols);
        matrixLoader.reduceToTriangularMatrixInPlace();
        double[] mArr = matrixLoader.getFlatArray();
        double[] sArr = solnMatrix.getFlatArray();
        for (int row = rows - 1; row >= 0; row--) {
            double sum = 0;
            for (int col = row + 1; col < cols - 1; col++) {
                sum += mArr[row * cols + col] * sArr[col];
            }
            double b = mArr[row * cols + (cols - 1)];
            double coefficient = mArr[row * cols + row];
            sArr[row] = (b - sum) / coefficient;
        }
    }

    private static void fillMinor(double[] src, double[] dst, int rowExc, int colExc, int n) {
        int d = 0;
        for (int i = 0; i < n; i++) {
            if (i == rowExc) {
                continue;
            }
            int rowOff = i * n;
            for (int j = 0; j < n; j++) {
                if (j == colExc) {
                    continue;
                }
                dst[d++] = src[rowOff + j];
            }
        }
    }

    public static EvalResult dispatchMatrixFunction(EvalResult[] args, String funcName, ResultCache cache) {
        switch (funcName) {
            case Declarations.MATRIX_ADD:
                return cache.result.wrap(flatMatrixAdd(args[0].matrix, args[1].matrix, cache));
            case Declarations.MATRIX_SUBTRACT:
                return cache.result.wrap(flatMatrixSubtract(args[0].matrix, args[1].matrix, cache));
            case Declarations.MATRIX_MULTIPLY:
            case "matrix_multiply":
                if (args[0].type == EvalResult.TYPE_SCALAR && args[1].type == EvalResult.TYPE_SCALAR) {
                    return cache.result.wrap(args[0].scalar * args[1].scalar);
                } else if (args[0].type == EvalResult.TYPE_SCALAR && args[1].type == EvalResult.TYPE_MATRIX) {
                    return cache.result.wrap(flatMatrixScalarMultiply(args[0].scalar, args[1].matrix, cache));
                } else if (args[0].type == EvalResult.TYPE_MATRIX && args[1].type == EvalResult.TYPE_SCALAR) {
                    return cache.result.wrap(flatMatrixScalarMultiply(args[1].scalar, args[0].matrix, cache));
                } else if (args[0].type == EvalResult.TYPE_MATRIX && args[1].type == EvalResult.TYPE_MATRIX) {
                    Matrix out = cache.getMatrixBuffer(args[0].matrix.getRows(), args[1].matrix.getCols());
                    return cache.result.wrap(flatMatrixMultiply(args[0].matrix, args[1].matrix, out));
                }
            case Declarations.DETERMINANT:
                return cache.result.wrap(args[0].matrix.determinant());
            case Declarations.MATRIX_POWER:
                return cache.result.wrap(flatMatrixPower(args[0].matrix, args[1].scalar, cache));
            case Declarations.INVERSE_MATRIX:
                return cache.result.wrap(flatMatrixInverseLUTurbo(args[0].matrix, cache));
            case Declarations.MATRIX_TRANSPOSE:
                return cache.result.wrap(args[0].matrix.transpose());
            case Declarations.TRIANGULAR_MATRIX:
                args[0].matrix.reduceToTriangularMatrixInPlace();
                return cache.result.wrap(args[0].matrix);
            case Declarations.MATRIX_EIGENVALUES:
                return cache.result.wrap(EigenProvider.getEigenvalues(args[0].matrix.getRows(), args[0].matrix.getFlatArray()));
            case Declarations.MATRIX_EIGENVEC:
                return cache.result.wrap(EigenProvider.getEigenvectors(args[0].matrix.getRows(), args[0].matrix.getFlatArray()));
            case Declarations.LINEAR_SYSTEM:
                Matrix input;
                if (args.length == 1) {
                    if (args[0] == null || args[0].matrix == null) {
                        return cache.result.wrap(ParserResult.UNDEFINED_ARG);
                    }
                    input = args[0].matrix;
                } else {
                    int n = (int) ((-1 + Math.sqrt(1 + 4 * args.length)) / 2.0);
                    input = cache.getMatrixBuffer(n, n + 1);
                    double[] flat = input.getFlatArray();
                    for (int i = 0; i < args.length; i++) {
                        flat[i] = args[i].scalar;
                    }
                }
                Matrix result = cache.getMatrixBuffer(input.getRows(), 1);
                solveEquationInto(input, result, cache);
                return cache.result.wrap(result);
            case Declarations.MATRIX_COFACTORS:
                input = args[0].matrix;
                int n = input.getRows();
                result = cache.getMatrixBuffer(n, n);
                Matrix minorBuf = cache.getSecondaryBuffer(n - 1, n - 1);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        fillMinor(input.getFlatArray(), minorBuf.getFlatArray(), i, j, n);
                        double minorDet = computeDeterminantTurbo(minorBuf, cache);
                        result.getFlatArray()[i * n + j] = (((i + j) % 2 == 0) ? 1.0 : -1.0) * minorDet;
                    }
                }
                return cache.result.wrap(result);
            case Declarations.MATRIX_ADJOINT:
                input = args[0].matrix;
                n = input.getRows();
                Matrix adjoint = cache.getMatrixBuffer(n, n);
                minorBuf = cache.getSecondaryBuffer(n - 1, n - 1);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        fillMinor(input.getFlatArray(), minorBuf.getFlatArray(), i, j, n);
                        double minorDet = computeDeterminantTurbo(minorBuf, cache);
                        adjoint.getFlatArray()[j * n + i] = (((i + j) % 2 == 0) ? 1.0 : -1.0) * minorDet;
                    }
                }
                return cache.result.wrap(adjoint);
            case Declarations.MATRIX_DIVIDE:
                Matrix A = args[0].matrix;
                Matrix B = args[1].matrix;

                // 1. Calculate the inverse FIRST. 
                // This internally locks up getSecondaryBuffer (for LU) and getMatrixBuffer (for the returned inverse).
                Matrix invB = flatMatrixInverseLUTurbo(B, cache);

                // 2. Safely grab a completely independent buffer for the final output.
                Matrix out = cache.getTertiaryBuffer(A.getRows(), invB.getCols());

                // 3. Multiply and wrap. A (source), invB (primary buffer), out (tertiary buffer) -> No collisions!
                return cache.result.wrap(flatMatrixMultiply(A, invB, out));
            case Declarations.MATRIX_EDIT:
                Matrix target = args[0].matrix;
                int row = (int) args[1].scalar,
                 col = (int) args[2].scalar;
                result = cache.getMatrixBuffer(target.getRows(), target.getCols());
                System.arraycopy(target.getFlatArray(), 0, result.getFlatArray(), 0, target.getFlatArray().length);
                result.getFlatArray()[row * target.getCols() + col] = args[3].scalar;
                return cache.result.wrap(result);
            case Declarations.ROTOR:
                //rot(F,a,O,D) function, angle, origin, direction vector
                //We have 2 scenarios:
                /**
                 * 1. rot(F, a, O, D) function, angle, origin, direction vector
                 * 2. rot(P1, P2, a, O, D) First Point, Second Point, angle,
                 * Origin and Direction These scenarios are further broken down
                 * into 2: The args may come in raw form. e.g F, O and D may
                 * come as function handles or as the raw data(more uncommon,
                 * but happens in MatrixTurboEvaluator)
                 */
                if (args.length == 4) {
                    //args[0] is always either a function handle or a Matrix of points
                    Matrix pointsVector = args[0].matrix;
                    double angle = args[1].scalar;
                    Matrix origin = args[2].matrix;
                    Matrix dir = args[3].matrix;
                    Function f = null;
                    if (args[0].type == EvalResult.TYPE_STRING) {
                        f = FunctionManager.lookUp(args[0].textRes);
                        if (f == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (f.isMatrix()) {
                            pointsVector = f.getMatrix();
                        }
                    }
                    Function v = null;
                    if (args[2].type == EvalResult.TYPE_STRING) {
                        v = FunctionManager.lookUp(args[2].textRes);
                        if (v == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (v.isMatrix()) {
                            origin = v.getMatrix();
                        }
                    }
                    if (args[3].type == EvalResult.TYPE_STRING) {
                        v = FunctionManager.lookUp(args[3].textRes);
                        if (v == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (v.isMatrix()) {
                            dir = v.getMatrix();
                        }
                    }
                    double[] pArr = origin.getRowMatrix(0).getFlatArray();
                    Point p = new Point(pArr[0], pArr[1], pArr[2]);
                    double[] dirArr = dir.getFlatArray();
                    Direction d = new Direction(dirArr[0], dirArr[1], dirArr[2]);
                    if (origin.getRows() != 1) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }
                    if (origin.getCols() != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    if (dir.getRows() != 1) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }
                    if (dir.getCols() != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    if (pointsVector != null) {
                        //rotate a set of points

                        ROTOR r = new ROTOR(angle, p, d);
                        int rows = pointsVector.getRows();
                        int cols = pointsVector.getCols();//@(n,3)

                        if (cols == 3) {
                            double outArr[] = new double[3 * rows];
                            int j = 0;
                            for (int i = 0; i < rows; i++) {
                                double[] pm = pointsVector.getRowMatrix(i).getFlatArray();
                                Point pt = new Point(pm[0], pm[1], pm[2]);
                                Point rotP = r.rotate(pt);
                                outArr[j] = rotP.x;
                                outArr[j + 1] = rotP.y;
                                outArr[j + 2] = rotP.z;
                                j += 3;
                            }
                            return cache.result.wrap(new Matrix(outArr, rows, 3));
                        } else {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                    }
                    if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
                        ArrayList<Variable> vars = f.getIndependentVariables();
                        int siz = vars.size();
                        if (siz > 3) {
                            return MathExpression.EvalResult.ERROR;
                        }
                        String expr = f.getMathExpression().getExpression();
                        String fullExpr = f.getName() + "=" + expr;
                        String transformedFuncName = f.getName();

                        ROTOR r = new ROTOR(angle, p, d);

                        if (siz == 2) {
                            r.setXAxisName(vars.get(0).getName());
                            r.setYAxisName(vars.get(1).getName());
                            r.setZAxisName(transformedFuncName);
                        }
                        if (siz == 1) {
                            r.setXAxisName(vars.get(0).getName());
                            r.setYAxisName(transformedFuncName);
                        }

                        String res = r.rotate(fullExpr);
                        return cache.result.wrap(res);
                    }

                    return null;

                }
                if (args.length == 5) {
                    Matrix p1Vector = args[0].matrix;
                    Matrix p2Vector = args[1].matrix;

                    double angle = args[2].scalar;
                    Matrix origin = args[3].matrix;
                    Matrix dir = args[4].matrix;
                    Function p1 = null;
                    if (args[0].type == EvalResult.TYPE_STRING) {
                        p1 = FunctionManager.lookUp(args[0].textRes);
                        if (p1 == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (p1.isMatrix()) {
                            p1Vector = p1.getMatrix();
                        }
                    }

                    Function p2 = null;
                    if (args[1].type == EvalResult.TYPE_STRING) {
                        p2 = FunctionManager.lookUp(args[1].textRes);
                        if (p2 == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (p2.isMatrix()) {
                            p2Vector = p2.getMatrix();
                        }
                    }

                    Function v = null;
                    if (args[3].type == EvalResult.TYPE_STRING) {
                        v = FunctionManager.lookUp(args[3].textRes);
                        if (v == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (v.isMatrix()) {
                            origin = v.getMatrix();
                        }
                    }
                    if (args[4].type == EvalResult.TYPE_STRING) {
                        v = FunctionManager.lookUp(args[4].textRes);
                        if (v == null) {
                            return cache.result.wrap(EvalResult.ERROR);
                        }
                        if (v.isMatrix()) {
                            dir = v.getMatrix();
                        }
                    }

                    double[] pArr = origin.getRowMatrix(0).getFlatArray();
                    Point p = new Point(pArr[0], pArr[1], pArr[2]);
                    double[] dirArr = dir.getFlatArray();
                    Direction d = new Direction(dirArr[0], dirArr[1], dirArr[2]);
                    ROTOR r = new ROTOR(angle, p, d);
                    int rows = p1Vector.getRows();
                    int cols = p1Vector.getCols();//@(n,3)
                    if (rows != 1) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }
                    if (cols != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    rows = p2Vector.getRows();
                    cols = p2Vector.getCols();//@(n,3)
                    if (rows != 1) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }
                    if (cols != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    double p1Arr[] = p1Vector.getFlatArray();
                    double p2Arr[] = p2Vector.getFlatArray();
                    Point rotP1 = r.rotate(new Point(p1Arr[0], p1Arr[1], p1Arr[2]));
                    Point rotP2 = r.rotate(new Point(p2Arr[0], p2Arr[1], p2Arr[2]));
                    Matrix outMatrix = new Matrix(new double[]{rotP1.x, rotP1.y, rotP1.z, rotP2.x, rotP2.y, rotP2.z}, 2, 3);//BREAKING CHANGE---CHANGED 2 point rotation to 2x3 Matrix output
                    return cache.result.wrap(outMatrix);
                }

            default:
                throw new UnsupportedOperationException("Function: `" + funcName + "` not supported");
        }

    }

    private static double executeScalarStatAtCompileTime(String method, double[] args) {
        int n = args.length;
        if (n == 0 && !method.equals(Declarations.RANDOM)) {
            return Double.NaN; // Safety guard for empty arrays
        }

        switch (method) {
            case Declarations.LIST_SUM:
            case Declarations.SUM: {
                double total = 0.0;
                // The JIT compiler will aggressively unroll this primitive loop
                for (int i = 0; i < n; i++) {
                    total += args[i];
                }
                return total;
            }

            case Declarations.PROD: {
                double prod = 1.0;
                for (int i = 0; i < n; i++) {
                    prod *= args[i];
                }
                return prod;
            }

            case Declarations.AVG:
            case Declarations.MEAN: {
                double mTotal = 0.0;
                for (int i = 0; i < n; i++) {
                    mTotal += args[i];
                }
                return mTotal / n;
            }

            case Declarations.MEDIAN: {
                // In-place sort is incredibly fast for small arrays and avoids allocation
                Arrays.sort(args);
                int mid = n / 2;
                if (n % 2 == 0) {
                    return (args[mid - 1] + args[mid]) * 0.5;
                }
                return args[mid];
            }

            case Declarations.MIN: {
                double min = args[0];
                for (int i = 1; i < n; i++) {
                    if (args[i] < min) {
                        min = args[i];
                    }
                }
                return min;
            }

            case Declarations.MAX: {
                double max = args[0];
                for (int i = 1; i < n; i++) {
                    if (args[i] > max) {
                        max = args[i];
                    }
                }
                return max;
            }
            case Declarations.NOW: {
                return System.currentTimeMillis();
            }
            case Declarations.NANOS: {
                return System.nanoTime();
            }
            case Declarations.RANGE: {
                double rMin = args[0];
                double rMax = args[0];
                for (int i = 1; i < n; i++) {
                    if (args[i] < rMin) {
                        rMin = args[i];
                    } else if (args[i] > rMax) {
                        rMax = args[i];
                    }
                }
                return rMax - rMin;
            }

            case Declarations.MID_RANGE: {
                double mrMin = args[0];
                double mrMax = args[0];
                for (int i = 1; i < n; i++) {
                    if (args[i] < mrMin) {
                        mrMin = args[i];
                    } else if (args[i] > mrMax) {
                        mrMax = args[i];
                    }
                }
                return (mrMax + mrMin) * 0.5;
            }

            case Declarations.VARIANCE: {
                if (n < 2) {
                    return 0.0;
                }
                // Welford's Algorithm: One-pass, cache-friendly, numerically stable
                double mean = 0.0;
                double M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                return M2 / (n - 1); // Sample variance
            }

            case Declarations.STD_DEV: {
                if (n < 2) {
                    return 0.0;
                }
                double mean = 0.0;
                double M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                return Math.sqrt(M2 / (n - 1));
            }

            case Declarations.STD_ERR: {
                if (n < 2) {
                    return 0.0;
                }
                double mean = 0.0;
                double M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                double stdDev = Math.sqrt(M2 / (n - 1));
                return stdDev / Math.sqrt(n);
            }

            case Declarations.COEFFICIENT_OF_VARIATION: {
                if (n < 2) {
                    return Double.NaN;
                }
                double mean = 0.0;
                double M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                if (mean == 0.0) {
                    return Double.NaN; // Guard against /0
                }
                double stdDev = Math.sqrt(M2 / (n - 1));
                return stdDev / mean;
            }

            case Declarations.ROOT_MEAN_SQUARED: {
                double sumSq = 0.0;
                for (int i = 0; i < n; i++) {
                    sumSq += (args[i] * args[i]);
                }
                return Math.sqrt(sumSq / n);
            }

            case Declarations.RANDOM:
                // ThreadLocalRandom is vastly superior to Math.random() for high-throughput calls
                return ThreadLocalRandom.current().nextDouble();

            default:
                return Double.NaN;
        }
    }

    private static double[] executeVectorStatAtCompileTime(String method, double[] args) {
        if (method.equals(Declarations.SORT)) {
            Arrays.sort(args);
            return args;
        }
        if (method.equals(Declarations.MODE)) {
            Arrays.sort(args);
            int maxC = 0, currC = 1;
            for (int i = 1; i < args.length; i++) {
                if (args[i] == args[i - 1]) {
                    currC++;
                } else {
                    maxC = Math.max(maxC, currC);
                    currC = 1;
                }
            }
            maxC = Math.max(maxC, currC);
            double[] tmp = new double[args.length];
            int idx = 0;
            currC = 1;
            for (int i = 1; i < args.length; i++) {
                if (args[i] == args[i - 1]) {
                    currC++;
                } else {
                    if (currC == maxC) {
                        tmp[idx++] = args[i - 1];
                    }
                    currC = 1;
                }
            }
            if (currC == maxC) {
                tmp[idx++] = args[args.length - 1];
            }
            return Arrays.copyOf(tmp, idx);
        }
        return null;
    }

    public static EvalResult dispatchUnaryOp(EvalResult operand, char op, ResultCache cache) {
        if (op == '²') {
            if (operand.type == EvalResult.TYPE_SCALAR) {
                return cache.result.wrap(operand.scalar * operand.scalar);
            }
            Matrix out = cache.getMatrixBuffer(operand.matrix.getRows(), operand.matrix.getCols());
            return cache.result.wrap(flatMatrixMultiply(operand.matrix, operand.matrix, out));
        }
        throw new UnsupportedOperationException("Unary op: " + op);
    }

    private static Matrix flatMatrixAdd(Matrix a, Matrix b, ResultCache cache) {
        double[] aF = a.getFlatArray(), bF = b.getFlatArray();
        Matrix out = cache.getMatrixBuffer(a.getRows(), a.getCols());
        for (int i = 0; i < aF.length; i++) {
            out.getFlatArray()[i] = aF[i] + bF[i];
        }
        return out;
    }

    private static Matrix flatMatrixSubtract(Matrix a, Matrix b, ResultCache cache) {
        double[] aF = a.getFlatArray(), bF = b.getFlatArray();
        Matrix out = cache.getMatrixBuffer(a.getRows(), a.getCols());
        for (int i = 0; i < aF.length; i++) {
            out.getFlatArray()[i] = aF[i] - bF[i];
        }
        return out;
    }

    private static Matrix flatMatrixMultiply1(Matrix a, Matrix b, Matrix out) {
        int aR = a.getRows(), aC = a.getCols(), bC = b.getCols();
        double[] aF = a.getFlatArray(), bF = b.getFlatArray(), resF = out.getFlatArray();
        // Required for i-k-j because it accumulates results
        out.reset();
        for (int i = 0; i < aR; i++) {
            int iRow = i * aC, outRow = i * bC;
            for (int j = 0; j < bC; j++) {
                double s = 0;
                for (int k = 0; k < aC; k++) {
                    s += aF[iRow + k] * bF[k * bC + j];
                }
                resF[outRow + j] = s;
            }
        }
        return out;
    }

    private static Matrix flatMatrixMultiply(Matrix a, Matrix b, Matrix out) {
        int aR = a.getRows(), aC = a.getCols(), bC = b.getCols();
        double[] aF = a.getFlatArray(), bF = b.getFlatArray(), resF = out.getFlatArray();

        // Required for i-k-j because it accumulates results
        out.reset();

        for (int i = 0; i < aR; i++) {
            int iRow = i * aC;
            int outRow = i * bC;
            for (int k = 0; k < aC; k++) {
                double aVal = aF[iRow + k]; // Load once, use for the whole row
                int kRow = k * bC;

                // This loop is the performance heart: perfectly sequential access
                for (int j = 0; j < bC; j++) {
                    resF[outRow + j] += aVal * bF[kRow + j];
                }
            }
        }
        return out;
    }

    private static Matrix flatMatrixPower(Matrix m, double exponent, ResultCache cache) {
        int p = (int) exponent;
        if (p < 0) {
            throw new UnsupportedOperationException("Negative power not supported.");
        }
        if (p == 0) {
            return identity(m.getRows(), cache);
        }
        Matrix res = identity(m.getRows(), new ResultCache()), base = new Matrix(m.getFlatArray().clone(), m.getRows(), m.getCols());
        while (p > 0) {
            if ((p & 1) == 1) {
                res = flatMatrixMultiply(res, base, new Matrix(new double[m.getRows() * m.getCols()], m.getRows(), m.getCols()));
            }
            p >>= 1;
            if (p > 0) {
                base = flatMatrixMultiply(base, base, new Matrix(new double[m.getRows() * m.getCols()], m.getRows(), m.getCols()));
            }
        }
        System.arraycopy(res.getFlatArray(), 0, cache.getMatrixBuffer(m.getRows(), m.getCols()).getFlatArray(), 0, m.getRows() * m.getCols());
        return cache.matrix;
    }

    private static Matrix identity(int dim, ResultCache cache) {
        Matrix id = cache.getMatrixBuffer(dim, dim);
        Arrays.fill(id.getFlatArray(), 0.0);
        for (int i = 0; i < dim; i++) {
            id.getFlatArray()[i * dim + i] = 1.0;
        }
        return id;
    }

    public static final class EigenProvider {

        private static final EigenEngineTurbo ENGINE = new EigenEngineTurbo();

        public static Matrix getEigenvalues(int n, double[] data) {
            return new Matrix(ENGINE.getEigenvalues(n, data), n, 2);
        }

        public static Matrix getEigenvectors(int n, double[] data) {
            Matrix m = new Matrix(data, n, n);
            return m.getEigenVectorMatrix(m.computeEigenValues());
        }
    }
}
