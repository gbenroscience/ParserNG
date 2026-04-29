/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
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
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.EvalResult;
import com.github.gbenroscience.parser.ParserResult;
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
        public double[] matrixData;
        public Matrix matrix;

        public double[] eigenValueBuffer;
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
                            // OPTIMIZED: Bound Variable Lookup (Zero arguments)
                            leaf = compileVariableLookupByIndex(t.frameIndex);
                        } else {
                            Function f = FunctionManager.lookUp(t.name);
                            if (f != null && f.getMatrix() != null) {
                                MathExpression.EvalResult res = new MathExpression.EvalResult();
                                res.wrap(f.getMatrix());
                                leaf = MethodHandles.constant(MathExpression.EvalResult.class, res);
                            } else {
                                MathExpression.EvalResult res = new MathExpression.EvalResult();
                                res.wrap(t.name);
                                leaf = MethodHandles.constant(MathExpression.EvalResult.class, res);
                            }
                        }
                    } else {
                        MathExpression.EvalResult res = new MathExpression.EvalResult();
                        res.wrap(t.value);
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

                    } else if (Method.isMatrixMethod(t.name)) {
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
                        stack.push(compileFunction(t, args));
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

        // OPTIMIZED: The final handle now takes zero arguments () -> EvalResult
        final MethodHandle finalHandle = stack.pop().asType(MethodType.methodType(EvalResult.class));

        return new FastCompositeExpression() {
            private void loadVars(double[] variables) {
                for (int i = 0; i < turboArgs.length; i++) {
                    turboArgs[slots[i]] = variables[i];
                }
            }

            @Override
            public EvalResult apply(double[] variables) {
                try {
                    loadVars(variables);
                    // OPTIMIZED: Invoke without arguments. turboArgs is already bound inside.
                    return (EvalResult) finalHandle.invokeExact();
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

    private MethodHandle compileVariableLookupByIndex(int frameIndex) throws NoSuchMethodException, IllegalAccessException {
        MethodHandle getter = LOOKUP.findStatic(MatrixTurboEvaluator.class,
                "getVariableFromFrame",
                MethodType.methodType(EvalResult.class, double[].class, int.class)
        );

        // OPTIMIZED: Bind the array AND the index. The resulting handle is () -> EvalResult
        return MethodHandles.insertArguments(getter, 0, turboArgs, frameIndex);
    }

    private static EvalResult getVariableFromFrame(double[] variables, int index) {
        MathExpression.EvalResult res = new MathExpression.EvalResult();
        double val = (index < variables.length) ? variables[index] : 0.0;
        res.wrap(val);
        return res;
    }

    private MethodHandle compileBinaryOpOnEvalResult(char op, MethodHandle left, MethodHandle right) throws Throwable {
        MethodHandle dispatcher = LOOKUP.findStatic(MatrixTurboEvaluator.class, "dispatchBinaryOp",
                MethodType.methodType(EvalResult.class, char.class, EvalResult.class, EvalResult.class, ResultCache.class));

        ResultCache nodeCache = new ResultCache();
        dispatcher = MethodHandles.insertArguments(dispatcher, 3, nodeCache);
        dispatcher = MethodHandles.insertArguments(dispatcher, 0, op);

        // OPTIMIZED: Use collectArguments to chain the zero-argument operands
        dispatcher = MethodHandles.collectArguments(dispatcher, 0, left);
        dispatcher = MethodHandles.collectArguments(dispatcher, 0, right);

        return dispatcher;
    }

    private MethodHandle compileUnaryOpOnEvalResult(char op, MethodHandle operand) throws Throwable {
        MethodHandle dispatcher = LOOKUP.findStatic(MatrixTurboEvaluator.class, "dispatchUnaryOp",
                MethodType.methodType(EvalResult.class, char.class, EvalResult.class, ResultCache.class));

        ResultCache nodeCache = new ResultCache();
        dispatcher = MethodHandles.insertArguments(dispatcher, 2, nodeCache);
        dispatcher = MethodHandles.insertArguments(dispatcher, 0, op);

        return MethodHandles.collectArguments(dispatcher, 0, operand);
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

    private MethodHandle compileMatrixFunction(MathExpression.Token t, MethodHandle[] args) throws Throwable {
        String funcName = t.name.toLowerCase();

        MethodHandle dispatcher = LOOKUP.findStatic(MatrixTurboEvaluator.class, "dispatchMatrixFunction",
                MethodType.methodType(EvalResult.class, EvalResult[].class, String.class, ResultCache.class));

        ResultCache nodeCache = new ResultCache();
        dispatcher = MethodHandles.insertArguments(dispatcher, 1, funcName, nodeCache);

        MethodHandle collector = LOOKUP.findStatic(MatrixTurboEvaluator.class, "collectArgsArray",
                MethodType.methodType(EvalResult[].class, EvalResult[].class)).asVarargsCollector(EvalResult[].class);

        collector = collector.asType(MethodType.methodType(EvalResult[].class,
                Collections.nCopies(t.arity, EvalResult.class).toArray(new Class[0])));

        MethodHandle finalFunc = MethodHandles.collectArguments(dispatcher, 0, collector);

        // Chain the zero-argument operands into the collector
        for (int i = 0; i < args.length; i++) {
            finalFunc = MethodHandles.collectArguments(finalFunc, 0, args[i]);
        }

        return finalFunc;
    }

    public static EvalResult[] collectArgsArray(EvalResult... args) {
        return args;
    }

    private static MethodHandle compileFunction(MathExpression.Token t, MethodHandle[] args) throws Throwable {
        int methodId = MethodRegistry.getMethodID(t.name);

        MethodHandle bridge = LOOKUP.findStatic(MatrixTurboEvaluator.class, "invokeRegistryMethod",
                MethodType.methodType(MathExpression.EvalResult.class, int.class, EvalResult[].class));

        MethodHandle boundBridge = MethodHandles.insertArguments(bridge, 0, methodId);
        MethodHandle collector = boundBridge.asCollector(EvalResult[].class, args.length);

        for (int i = 0; i < args.length; i++) {
            collector = MethodHandles.collectArguments(collector, 0, args[i]);
        }

        return collector;
    }

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
                    double det = right.matrix.determinant();
                    if (Math.abs(det) < 1e-15) {
                        throw new ArithmeticException("Matrix B is singular");
                    }
                    Matrix adjB = right.matrix.adjoint();
                    Matrix invB = flatMatrixScalarMultiply(1.0 / det, adjB, cache);
                    Matrix out = cache.getMatrixBuffer(left.matrix.getRows(), invB.getCols());
                    cache.result.wrap(flatMatrixMultiply(left.matrix, invB, out));
                } else if (leftType == EvalResult.TYPE_MATRIX && rightType == EvalResult.TYPE_SCALAR) {
                    if (Math.abs(right.scalar) < 1e-15) {
                        throw new ArithmeticException("Division by zero scalar");
                    }
                    cache.result.wrap(flatMatrixScalarMultiply(1.0 / right.scalar, left.matrix, cache));
                } else if (leftType == EvalResult.TYPE_SCALAR && rightType == EvalResult.TYPE_MATRIX) {
                    double det = right.matrix.determinant();
                    if (Math.abs(det) < 1e-15) {
                        throw new ArithmeticException("Matrix is singular");
                    }
                    Matrix adjB = right.matrix.adjoint();
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

    private static Matrix flatMatrixScalarMultiply(double scalar, Matrix m, ResultCache cache) {
        double[] mF = m.getFlatArray();
        Matrix out = cache.getMatrixBuffer(m.getRows(), m.getCols());
        double[] resF = out.getFlatArray();
        for (int i = 0; i < mF.length; i++) {
            resF[i] = scalar * mF[i];
        }
        return out;
    }

    private static double computeDeterminantTurbo(Matrix minor, ResultCache cache) {
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
            if (Math.abs(v) < 1e-18) {
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
                Matrix out = cache.getMatrixBuffer(args[0].matrix.getRows(), args[1].matrix.getCols());
                return cache.result.wrap(flatMatrixMultiply(args[0].matrix, args[1].matrix, out));
            case Declarations.DETERMINANT:
                return cache.result.wrap(args[0].matrix.determinant());
            case Declarations.INVERSE_MATRIX:
                return cache.result.wrap(args[0].matrix.inverse());
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
                Matrix A = args[0].matrix,
                 B = args[1].matrix;
                n = B.getRows();
                int m = A.getCols();
                Matrix augmented = cache.getSecondaryBuffer(n, n + m);
                double[] augF = augmented.getFlatArray();
                for (int i = 0; i < n; i++) {
                    System.arraycopy(B.getFlatArray(), i * n, augF, i * (n + m), n);
                    System.arraycopy(A.getFlatArray(), i * m, augF, i * (n + m) + n, m);
                }
                augmented.reduceToTriangularMatrixInPlace();
                result = cache.getMatrixBuffer(n, m);
                for (int k = 0; k < m; k++) {
                    for (int i = n - 1; i >= 0; i--) {
                        double sum = 0;
                        for (int j = i + 1; j < n; j++) {
                            sum += augF[i * (n + m) + j] * result.getFlatArray()[j * m + k];
                        }
                        result.getFlatArray()[i * m + k] = (augF[i * (n + m) + n + k] - sum) / augF[i * (n + m) + i];
                    }
                }
                return cache.result.wrap(result);
            case Declarations.MATRIX_EDIT:
                Matrix target = args[0].matrix;
                int row = (int) args[1].scalar,
                 col = (int) args[2].scalar;
                result = cache.getMatrixBuffer(target.getRows(), target.getCols());
                System.arraycopy(target.getFlatArray(), 0, result.getFlatArray(), 0, target.getFlatArray().length);
                result.getFlatArray()[row * target.getCols() + col] = args[3].scalar;
                return cache.result.wrap(result);
            default:
                throw new UnsupportedOperationException("Function: " + funcName);
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

    private static Matrix flatMatrixMultiply(Matrix a, Matrix b, Matrix out) {
        int aR = a.getRows(), aC = a.getCols(), bC = b.getCols();
        double[] aF = a.getFlatArray(), bF = b.getFlatArray(), resF = out.getFlatArray();
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
