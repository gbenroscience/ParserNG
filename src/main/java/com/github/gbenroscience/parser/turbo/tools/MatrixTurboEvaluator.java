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
import com.github.gbenroscience.math.geom.*;
import com.github.gbenroscience.math.geom.Point;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.EvalResult;
import com.github.gbenroscience.parser.ParserResult;

import com.github.gbenroscience.parser.TYPE;
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
 */
public final class MatrixTurboEvaluator implements TurboExpressionEvaluator {

    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
    private boolean willFoldConstants;

    protected final int[] slots;
    private ErrorLog errorLog = new ErrorLog();
    private MathExpression.Token[] postfix;
    private MethodHandle finalHandle;
    // Enables True AST Inlining for user-defined functions
    private MethodHandle[] inlinedVariables;

    public MatrixTurboEvaluator(MathExpression me) {
        this.postfix = me.getCachedPostfix();
        this.willFoldConstants = me.isWillFoldConstants();
        slots = me.getSlots();
        me.copyErrorLogTo(errorLog);
    }

    private static final ThreadLocal<MathExpression.EvalResult[]> WRAPPER_CACHE
            = ThreadLocal.withInitial(() -> {
                MathExpression.EvalResult[] arr = new MathExpression.EvalResult[12];
                for (int i = 0; i < 12; i++) {
                    arr[i] = new MathExpression.EvalResult();
                }
                return arr;
            });

    public static final class ResultCache {

        public final EvalResult result = new EvalResult();
        private double lastDeterminant;
        public double[] matrixData;
        public Matrix matrix;

        public double[] eigenValueBuffer;
        public int[] intBuffer;
        private double[] matrixData2;
        private Matrix matrix2;

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

    public static MathExpression.EvalResult invokeRegistryMethod(int methodId, EvalResult[] argsValues) {
        int arity = argsValues.length;
        MathExpression.EvalResult resultContainer = new MathExpression.EvalResult();
        MethodRegistry.getAction(methodId).calc(resultContainer, arity, argsValues);
        return resultContainer;
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {

        Stack<MethodHandle> stack = new Stack<>();

        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    MethodHandle leaf;
                    if (t.name != null && !t.name.isEmpty()) {
                        if (t.v != null) {
                            if (inlinedVariables != null && t.frameIndex < inlinedVariables.length) {
                                leaf = inlinedVariables[t.frameIndex];
                            } else {
                                leaf = compileVariableLookupByIndex(t.frameIndex);
                            }
                        } else {
                            Function f = FunctionManager.lookUp(t.name);
                            if (f != null && f.getMatrix() != null) {
                                EvalResult res = new EvalResult().wrap(f.getMatrix());
                                leaf = createConstantHandle(res);
                            } else {
                                EvalResult res = new EvalResult().wrap(t.name);
                                leaf = createConstantHandle(res);
                            }
                        }
                    } else {
                        EvalResult res = new EvalResult().wrap(t.value);
                        leaf = createConstantHandle(res);
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

                        EvalResult precalculated = new EvalResult();
                        if (name.equals(Declarations.SORT) || name.equals(Declarations.MODE)) {
                            double[] vecResult = executeVectorStatAtCompileTime(name, data);
                            precalculated.wrap(vecResult);
                        } else {
                            double scalarResult = executeScalarStatAtCompileTime(name, data);
                            precalculated.wrap(scalarResult);
                        }
                        stack.push(createConstantHandle(precalculated));
                        break;
                    } else if (t.name.equalsIgnoreCase("print")) {
                        for (int i = 0; i < t.arity; i++) {
                            stack.pop();
                        }

                        MethodHandle bridge = LOOKUP.findStatic(MatrixTurboEvaluator.class, "evaluatePrint",
                                MethodType.methodType(EvalResult.class, String[].class, double[].class));
                        stack.push(MethodHandles.insertArguments(bridge, 0, (Object) t.getRawArgs()));

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

                            // The inner compiler produces a handle that perfectly takes (double[]) -> EvalResult
                            stack.push(inlineEvaluator.finalHandle);
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

        return new FastCompositeExpression() {
            // Internal 'args[]' tracking array is completely eliminated to guarantee thread-safety!

            @Override
            public EvalResult apply(double[] variables) {
                try {
                    // ZERO runtime loops. ZERO array copying. ZERO bounds matching overhead.
                    // finalHandle now reads straight from the user's input variable coordinates pointer.
                    return (EvalResult) finalHandle.invokeExact(variables);
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

    ///        //////////////////////////////////////////////////////////////////////////

// =========================================================================
// EXECUTION BRIDGES (Bypasses Android ART bugs with MethodHandle Combinators)
// =========================================================================

private static MethodHandle createConstantHandle(EvalResult res) {
        MethodHandle c = MethodHandles.constant(EvalResult.class, res);
        return MethodHandles.dropArguments(c, 0, double[].class);
    }

    private static EvalResult evaluateVariable(int index, ResultCache cache, double[] vars) {
        return cache.result.wrap(vars[index]);
    }

    private MethodHandle compileVariableLookupByIndex(final int frameIndex) throws Throwable {
        if (inlinedVariables != null && frameIndex >= 0 && frameIndex < inlinedVariables.length) {
            return inlinedVariables[frameIndex];
        }

        // Determine which index in the user's incoming 'variables' array matches this frameIndex
        int targetVariableIndex = -1;
        if (slots != null) {
            for (int i = 0; i < slots.length; i++) {
                if (slots[i] == frameIndex) {
                    targetVariableIndex = i;
                    break;
                }
            }
        }
        // Fallback to frameIndex if it is an unmapped execution parameter slot
        int finalSourceIndex = (targetVariableIndex != -1) ? targetVariableIndex : frameIndex;

        MethodHandle evaluator = LOOKUP.findStatic(MatrixTurboEvaluator.class, "evaluateVariable",
                MethodType.methodType(EvalResult.class, int.class, ResultCache.class, double[].class));

        ResultCache nodeCache = new ResultCache();
        // Insert the directly calculated finalSourceIndex to allow register-level hardware indexing
        return MethodHandles.insertArguments(evaluator, 0, finalSourceIndex, nodeCache);
    }

    private static EvalResult evaluateBinary(char op, MethodHandle left, MethodHandle right, ResultCache cache, double[] vars) throws Throwable {
        EvalResult l = (EvalResult) left.invokeExact(vars);
        EvalResult r = (EvalResult) right.invokeExact(vars);
        return dispatchBinaryOp(op, l, r, cache);
    }

    private MethodHandle compileBinaryOpOnEvalResult(char op, MethodHandle left, MethodHandle right) throws Throwable {
        MethodHandle evaluator = LOOKUP.findStatic(MatrixTurboEvaluator.class, "evaluateBinary",
                MethodType.methodType(EvalResult.class, char.class, MethodHandle.class, MethodHandle.class, ResultCache.class, double[].class));
        ResultCache nodeCache = new ResultCache();
        return MethodHandles.insertArguments(evaluator, 0, op, left, right, nodeCache);
    }

    private static EvalResult evaluateUnary(char op, MethodHandle operand, ResultCache cache, double[] vars) throws Throwable {
        EvalResult val = (EvalResult) operand.invokeExact(vars);
        return dispatchUnaryOp(val, op, cache);
    }

    private MethodHandle compileUnaryOpOnEvalResult(char op, MethodHandle operand) throws Throwable {
        MethodHandle evaluator = LOOKUP.findStatic(MatrixTurboEvaluator.class, "evaluateUnary",
                MethodType.methodType(EvalResult.class, char.class, MethodHandle.class, ResultCache.class, double[].class));
        ResultCache nodeCache = new ResultCache();
        return MethodHandles.insertArguments(evaluator, 0, op, operand, nodeCache);
    }

    private static EvalResult evaluateMatrixFunction(String funcName, MethodHandle[] args, ResultCache cache, double[] vars) throws Throwable {
        EvalResult[] buf = cache.getArgsBuffer(args.length);
        for (int i = 0; i < args.length; i++) {
            buf[i] = (EvalResult) args[i].invokeExact(vars);
        }
        return dispatchMatrixFunction(buf, funcName, cache);
    }

    private MethodHandle compileMatrixFunction(MathExpression.Token t, MethodHandle[] args) throws Throwable {
        MethodHandle evaluator = LOOKUP.findStatic(MatrixTurboEvaluator.class, "evaluateMatrixFunction",
                MethodType.methodType(EvalResult.class, String.class, MethodHandle[].class, ResultCache.class, double[].class));
        ResultCache nodeCache = new ResultCache();
        return MethodHandles.insertArguments(evaluator, 0, t.name.toLowerCase(), args, nodeCache);
    }

    private static EvalResult evaluateRegistryFunction(int methodId, MethodHandle[] args, ResultCache cache, double[] vars) throws Throwable {
        EvalResult[] buf = cache.getArgsBuffer(args.length);
        for (int i = 0; i < args.length; i++) {
            buf[i] = (EvalResult) args[i].invokeExact(vars);
        }
        return invokeRegistryMethod(methodId, cache, buf);
    }

    private static MethodHandle compileFunction(MathExpression.Token t, MethodHandle[] args) throws Throwable {
        int methodId = MethodRegistry.getMethodID(t.name);
        MethodHandle evaluator = LOOKUP.findStatic(MatrixTurboEvaluator.class, "evaluateRegistryFunction",
                MethodType.methodType(EvalResult.class, int.class, MethodHandle[].class, ResultCache.class, double[].class));
        ResultCache nodeCache = new ResultCache();
        return MethodHandles.insertArguments(evaluator, 0, methodId, args, nodeCache);
    }

    private static EvalResult evaluatePrint(String[] args, double[] vars) {
        return executePrint(args);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    private static MethodHandle mergeDoubleArrays(MethodHandle mh) {
        MethodType type = mh.type();
        int doubleArrayCount = 0;
        for (int i = 0; i < type.parameterCount(); i++) {
            if (type.parameterType(i) == double[].class) {
                doubleArrayCount++;
            }
        }

        if (doubleArrayCount <= 1) {
            return mh;
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

    public static EvalResult wrapDouble(double val, ResultCache cache) {
        return cache.result.wrap(val);
    }

    public static double getVar(double[] vars, int index) {
        return vars[index];
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

    // ========== RUNTIME DISPATCHERS ==========
    private static EvalResult dispatchBinaryOp(char op, EvalResult left, EvalResult right, ResultCache cache) {
        final int leftType = left.type;
        final int rightType = right.type;

        // =========================================================================
        // ⚡ ULTRA-TURBO FAST-PATH FOR PURE SCALAR ARITHMETIC
        // Removes all internal conditional type-checking gates for scalar operations.
        // =========================================================================
        if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_SCALAR) {
            switch (op) {
                case '+':
                    cache.result.wrap(left.scalar + right.scalar);
                    break;
                case '-':
                    cache.result.wrap(left.scalar - right.scalar);
                    break;
                case '*':
                    cache.result.wrap(left.scalar * right.scalar);
                    break;
                case '/':
                    cache.result.wrap(left.scalar / right.scalar);
                    break;
                case '^':
                    cache.result.wrap(Math.pow(left.scalar, right.scalar));
                    break;
                default:
                    throw new UnsupportedOperationException("Operator not implemented: " + op);
            }
            return cache.result;
        }

        // =========================================================================
        // 📂 MATRIX AND MIXED-TYPE OPERATIONS PATH
        // Executed only when at least one operand is a Matrix structure.
        // =========================================================================
        switch (op) {
            case '+':
                if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    cache.result.wrap(flatMatrixAdd(left.matrix, right.matrix, cache));
                }
                break;

            case '-':
                if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    cache.result.wrap(flatMatrixSubtract(left.matrix, right.matrix, cache));
                }
                break;

            case '*':
                if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_MATRIX) {
                    cache.result.wrap(flatMatrixScalarMultiply(left.scalar, right.matrix, cache));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
                    cache.result.wrap(flatMatrixScalarMultiply(right.scalar, left.matrix, cache));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    Matrix out = cache.getMatrixBuffer(left.matrix.getRows(), right.matrix.getCols());
                    cache.result.wrap(flatMatrixMultiply(left.matrix, right.matrix, out));
                }
                break;

            case '/':
                if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_MATRIX) {
                    double det = computeDeterminantTurbo(right.matrix, cache);
                    if (Math.abs(det) < 1e-15) {
                        throw new ArithmeticException("Matrix B is singular");
                    }
                    Matrix A = left.matrix;
                    Matrix B = right.matrix;
                    Matrix invB = flatMatrixInverseLUTurbo(B, cache);
                    Matrix out = cache.getTertiaryBuffer(A.getRows(), invB.getCols());
                    cache.result.wrap(flatMatrixMultiply(A, invB, out));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
                    if (Math.abs(right.scalar) < 1e-15) {
                        throw new ArithmeticException("Division by zero scalar");
                    }
                    cache.result.wrap(flatMatrixScalarMultiply(1.0 / right.scalar, left.matrix, cache));
                } else if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_MATRIX) {
                    double det = computeDeterminantTurbo(right.matrix, cache);
                    if (Math.abs(det) < 1e-15) {
                        throw new ArithmeticException("Matrix is singular");
                    }
                    int n = right.matrix.getRows();
                    Matrix adjB = cache.getMatrixBuffer(n, n);
                    Matrix minorBuf = cache.getSecondaryBuffer(n - 1, n - 1);
                    computeAdjointInto(right.matrix, adjB, minorBuf, cache);

                    cache.result.wrap(flatMatrixScalarMultiply(left.scalar / det, adjB, cache));
                }
                break;

            case '^':
                if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
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

        double det = 1.0;
        int swapCount = 0;

        System.arraycopy(input.getFlatArray(), 0, lu, 0, n * n);
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }

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

            if (maxVal < 1e-16) {
                cache.lastDeterminant = 0;
                throw new ArithmeticException("Matrix is singular.");
            }

            if (pivotRow != k) {
                int tmp = pivot[k];
                pivot[k] = pivot[pivotRow];
                pivot[pivotRow] = tmp;

                int pOff = pivotRow * n;
                for (int j = 0; j < n; j++) {
                    double t = lu[kOff + j];
                    lu[kOff + j] = lu[pOff + j];
                    lu[pOff + j] = t;
                }
                swapCount++;
            }

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
        cache.lastDeterminant = det;

        for (int pIdx = 0; pIdx < n; pIdx++) {
            int j = pivot[pIdx];

            Arrays.fill(work, 0.0);
            work[pIdx] = 1.0;

            for (int i = pIdx + 1; i < n; i++) {
                double sum = work[i];
                int iOff = i * n;
                for (int k = pIdx; k < i; k++) {
                    sum -= lu[iOff + k] * work[k];
                }
                work[i] = sum;
            }

            for (int i = n - 1; i >= 0; i--) {
                double sum = work[i];
                int iOff = i * n;
                for (int k = i + 1; k < n; k++) {
                    sum -= lu[iOff + k] * work[k];
                }
                work[i] = sum * uInvDiag[i];
            }

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

                Matrix invB = flatMatrixInverseLUTurbo(B, cache);
                Matrix out = cache.getTertiaryBuffer(A.getRows(), invB.getCols());
                return cache.result.wrap(flatMatrixMultiply(A, invB, out));
            case Declarations.MATRIX_EDIT:
                Matrix target = args[0].matrix;
                int row = (int) args[1].scalar,
                 col = (int) args[2].scalar;
                result = cache.getMatrixBuffer(target.getRows(), target.getCols());
                System.arraycopy(target.getFlatArray(), 0, result.getFlatArray(), 0, target.getFlatArray().length);
                result.getFlatArray()[row * target.getCols() + col] = args[3].scalar;
                return cache.result.wrap(result);
            case Declarations.SUB_MATRIX:
                Matrix main = args[0].matrix;
                 row = (int) args[1].scalar;
                 col = (int) args[2].scalar;
                Matrix sm = getSubmatrix(main, row, col, cache);
                return cache.result.wrap(sm);
            case Declarations.RANDOM_MATRIX:
                 n = (int) args[0].scalar;
                 row = (int) args[1].scalar;
                 col = (int) args[2].scalar;
                 sm = randomFillTurbo(n, row, col, cache);
                return cache.result.wrap(sm);
            case Declarations.MATRIX_MINOR:
                 main = args[0].matrix;
                 row = (int) args[1].scalar;
                 col = (int) args[2].scalar;
                 sm = minor(main, row, col, cache);
                return cache.result.wrap(sm);
            case Declarations.ROTOR:
                if (args.length == 4) {
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
                    if (origin.getRows() != 1 || origin.getCols() != 3 || dir.getRows() != 1 || dir.getCols() != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    if (pointsVector != null) {
                        ROTOR r = new ROTOR(angle, p, d);
                        int rows = pointsVector.getRows();
                        int cols = pointsVector.getCols();

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
                    int cols = p1Vector.getCols();
                    if (rows != 1 || cols != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    rows = p2Vector.getRows();
                    cols = p2Vector.getCols();
                    if (rows != 1 || cols != 3) {
                        return cache.result.wrap(EvalResult.ERROR);
                    }

                    double p1Arr[] = p1Vector.getFlatArray();
                    double p2Arr[] = p2Vector.getFlatArray();
                    Point rotP1 = r.rotate(new Point(p1Arr[0], p1Arr[1], p1Arr[2]));
                    Point rotP2 = r.rotate(new Point(p2Arr[0], p2Arr[1], p2Arr[2]));
                    Matrix outMatrix = new Matrix(new double[]{rotP1.x, rotP1.y, rotP1.z, rotP2.x, rotP2.y, rotP2.z}, 2, 3);
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

        out.reset();

        for (int i = 0; i < aR; i++) {
            int iRow = i * aC;
            int outRow = i * bC;
            for (int k = 0; k < aC; k++) {
                double aVal = aF[iRow + k];
                int kRow = k * bC;

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

    private static Matrix minor(Matrix A, int r, int c, ResultCache cache) {
        int aR = A.getRows();
        int aC = A.getCols();

        if (aR <= 1 || aC <= 1) {
            throw new IllegalArgumentException("Matrix is too small to extract a submatrix.");
        }
        if (r < 0 || r >= aR || c < 0 || c >= aC) {
            throw new IndexOutOfBoundsException("Row or column index out of bounds.");
        }

        // The new matrix is 1 row and 1 column smaller
        int bR = aR - 1;
        int bC = aC - 1;

        // Borrow buffer to avoid GC pressure
        Matrix B = cache.getMatrixBuffer(bR, bC);

        double[] aF = A.getFlatArray();
        double[] bF = B.getFlatArray();

        int destRow = 0;
        for (int i = 0; i < aR; i++) {
            // Skip the target row 'r'
            if (i == r) {
                continue;
            }

            int srcOffset = i * aC;
            int destOffset = destRow * bC;

            // 1. Copy the left chunk (everything before column 'c')
            if (c > 0) {
                System.arraycopy(aF, srcOffset, bF, destOffset, c);
            }

            // 2. Copy the right chunk (everything after column 'c')
            if (c < aC - 1) {
                System.arraycopy(aF, srcOffset + c + 1, bF, destOffset + c, aC - c - 1);
            }

            destRow++;
        }

        return B;
    }

    /**
     * Computes and returns specifically, the bottom-right sub-block from a
     * given coordinate to the end of the matrix), not a Minor submatrix
     *
     * @param A The matrix
     * @param r The row to copy from(zero based)
     * @param c The column to copy from(zero based)
     * @param cache
     * @return specifically the bottom-right sub-block from a given coordinate
     * to the end of the matrix
     */
    private static Matrix getSubmatrix(Matrix A, int r, int c, ResultCache cache) {
        int aR = A.getRows();
        int aC = A.getCols();

        // Bounds validation
        if (r < 0 || r >= aR || c < 0 || c >= aC) {
            throw new IndexOutOfBoundsException("Row or column index out of bounds.");
        }

        // Calculate dimensions of the new sub-block
        int newR = aR - r;
        int newC = aC - c;

        // Borrow a buffer of the exact required size
        Matrix B = cache.getMatrixBuffer(newR, newC);

        double[] aF = A.getFlatArray();
        double[] bF = B.getFlatArray();

        // Copy row by row using System.arraycopy
        for (int i = 0; i < newR; i++) {
            // Source index: (Current original row * total columns) + starting column
            int srcOffset = (r + i) * aC + c;

            // Destination index: (Current new row * new columns)
            int destOffset = i * newC;

            // Copy the entire continuous width of the new row in one instruction
            System.arraycopy(aF, srcOffset, bF, destOffset, newC);
        }

        return B;
    }

    public static Matrix randomFillTurbo(int n, int rowSize, int colSize, ResultCache cache) {
        // 1. Zero Allocation: Borrow a matrix from the cache instead of using 'new'
        Matrix m = cache.getMatrixBuffer(rowSize, colSize);

        // 2. Extract the flat array for sequential memory access
        double[] array = m.getFlatArray();
        int len = array.length;

        // 3. The Secret Weapon: ThreadLocalRandom
        // This bypasses all atomic lock contention, making it up to 10x faster than java.util.Random
        ThreadLocalRandom ran = ThreadLocalRandom.current();

        // 4. Hot Loop
        for (int i = 0; i < len; i++) {
            array[i] = 1 + ran.nextInt(n);
        }
        return m; // Return the borrowed, filled matrix
    }

    public static void randomFillTurbo(Matrix m, int n) {
        double[] array = m.getFlatArray();
        int len = array.length;
        ThreadLocalRandom ran = ThreadLocalRandom.current();

        // Loop unrolling for extreme sizes (optional, but very "Turbo")
        int i = 0;
        for (; i <= len - 4; i += 4) {
            array[i] = 1 + ran.nextInt(n);
            array[i + 1] = 1 + ran.nextInt(n);
            array[i + 2] = 1 + ran.nextInt(n);
            array[i + 3] = 1 + ran.nextInt(n);
        }
        // Clean up any remaining elements
        for (; i < len; i++) {
            array[i] = 1 + ran.nextInt(n);
        }
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
