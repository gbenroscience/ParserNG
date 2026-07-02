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

import com.github.gbenroscience.interfaces.Savable;
import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.math.differentialcalculus.Derivative;
import com.github.gbenroscience.math.differentialcalculus.equations.DifferentialEquations;
import com.github.gbenroscience.math.geom.Direction;
import com.github.gbenroscience.math.geom.Line3D;
import com.github.gbenroscience.math.geom.Point;
import com.github.gbenroscience.math.geom.ROTOR;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.math.numericalmethods.NumericalIntegrator;
import com.github.gbenroscience.math.numericalmethods.TurboRootFinder;
import com.github.gbenroscience.math.quadratic.QuadraticSolver;
import com.github.gbenroscience.math.quadratic.Quadratic_Equation;
import com.github.gbenroscience.math.tartaglia.Tartaglia_Equation;
import com.github.gbenroscience.parser.Bracket;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.TYPE;
import static com.github.gbenroscience.parser.TYPE.ALGEBRAIC_EXPRESSION;
import static com.github.gbenroscience.parser.TYPE.MATRIX;
import com.github.gbenroscience.parser.Variable;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.Method;
import com.github.gbenroscience.parser.methods.MethodRegistry;
import com.github.gbenroscience.util.ErrorLog;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.Utils;
import com.github.gbenroscience.util.io.ByteArrayBuilder;
import java.lang.invoke.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Turbo compiler optimized for PURE SCALAR expressions. Uses an array based
 * technique to pass variables to the evaluator
 *
 * @author GBEMIRO
 */
public class ScalarTurboEvaluator1 implements TurboExpressionEvaluator, Savable {

    private static final long serialVersionUID = 1L;
    protected boolean willFoldConstants;
    protected final int[] slots;
    protected MathExpression.VariableRegistry registry;

    public static final MethodHandle SCALAR_GATEKEEPER_HANDLE;
    public static final MethodHandle VECTOR_GATEKEEPER_HANDLE;
    public static final MethodHandle VECTOR_2_GATEKEEPER_HANDLE;

    // 1. ThreadLocal workspace: persists across evaluations on the same thread
    private static final ThreadLocal<double[]> WORKSPACE = ThreadLocal.withInitial(() -> new double[256]);

    static {
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup();

            // Define the method types: (String, double[]) -> double/double[]
            MethodType scalarType = MethodType.methodType(double.class, String.class, double[].class);
            MethodType vectorType = MethodType.methodType(double[].class, String.class, double[].class);
            MethodType vector2Type = MethodType.methodType(double[].class, String.class, String.class);

            /**
             * For non stats methods that return a double array
             */
            SCALAR_GATEKEEPER_HANDLE = lookup.findStatic(ScalarTurboEvaluator1.class, "scalarStatsGatekeeper", scalarType);

            /**
             * For stats methods that return a double array
             */
            VECTOR_GATEKEEPER_HANDLE = lookup.findStatic(ScalarTurboEvaluator1.class, "vectorStatsGatekeeper", vectorType);

            /**
             * For non stats methods that return a double array
             */
            VECTOR_2_GATEKEEPER_HANDLE = lookup.findStatic(ScalarTurboEvaluator1.class, "vectorNonStatsGatekeeper", vector2Type);
//public static double[] vectorNonStatsGatekeeper(String method, String funcHandle)
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new ExceptionInInitializerError("Failed to initialize Stats Gatekeepers: " + e.getMessage());
        }
    }

    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();

    // Common method types
    private static final MethodType MT_DOUBLE_D = MethodType.methodType(double.class, double.class);
    private static final MethodType MT_DOUBLE_DD = MethodType.methodType(double.class, double.class, double.class);
    private static final MethodType MT_SAFE_WRAP = MethodType.methodType(double.class, double[].class);

    // 1. ThreadLocal holding a reusable array of EvalResults to avoid GC pressure
    private static final ThreadLocal<MathExpression.EvalResult[]> WRAPPER_CACHE
            = ThreadLocal.withInitial(() -> {
                MathExpression.EvalResult[] arr = new MathExpression.EvalResult[8];
                for (int i = 0; i < 8; i++) {
                    arr[i] = new MathExpression.EvalResult();
                }
                return arr;
            });

    protected MathExpression.Token[] postfix;

    private static final Set<String> FAST_PATH_METHODS = new HashSet<>(Arrays.asList(
            "acsc_deg", "sech", "tan-¹_rad", "acot_rad",
            "cos-¹_deg", "asin_grad", "sqrt", "acos_grad",
            "acot_deg", "cube", "exp", "ln-¹", "asec_rad",
            "asec_deg", "acosh", "sec_grad", "ceil",
            "csc_rad", "tanh-¹", "comb", "tan-¹_deg", "acsch",
            "acot_grad", "sec_rad", "fact", "acoth", "atanh",
            "log", "tan_grad", "tan-¹_grad", "coth-¹",
            "min", "log-¹", "cot-¹_grad", "sech-¹", "pow",
            "gelu", "swiglu", "geglu", "fast_gelu", "erf",
            "csc_deg", "cos-¹_rad", "tan_rad", "max", "sin-¹_deg",
            "cot-¹_deg", "alog", "acsc_rad", "abs",
            "sin-¹_rad", "tan_deg", "lg", "sec_deg", "atan_deg",
            "ln", "sinh-¹", "asin_rad", "acos_deg",
            "atan_rad", "asech", "cos_grad", "cot-¹_rad",
            "asec_grad", "acos_rad", "alg", "aln",
            "sinh", "cos_rad", "rnd", "rng",
            "acsc_grad", "square", "csch-¹",
            "sec-¹_grad", "asin_deg", "cos_deg", "perm",
            "csc_grad", "sec-¹_deg", "sin_deg", "sin-¹_grad",
            "cot_deg", "coth", "cbrt", "sec-¹_rad", "tanh",
            "cos-¹_grad", "lg-¹",
            "cot_rad", "atan_grad", "sin_grad", "cot_grad",
            "csc-¹_grad", "length", "csc-¹_deg", "cosh-¹", "cosh",
            "csc-¹_rad", "sin_rad", "csch", "asinh"
    ));

    ////////EvalResult Pool params start/////////////
        private static final int INIT_POOL_SIZE = 64;
    // A simple pre-allocated array of results to act as a stack
    private MathExpression.EvalResult[] pool = new MathExpression.EvalResult[INIT_POOL_SIZE];
    private int poolPointer = 0;
    protected ErrorLog errorLog = new ErrorLog();

    ////////EvalResult Pool params ends/////////////
/**
 * 
 * @param me 
 */
    public ScalarTurboEvaluator1(MathExpression me) {
        for (int i = 0; i < INIT_POOL_SIZE; i++) {
            pool[i] = new MathExpression.EvalResult();
        }
        this.postfix = me.getCachedPostfix();
        this.willFoldConstants = me.isWillFoldConstants();
        slots = me.getSlots();
        registry = me.getRegistry().clone();
        me.copyErrorLogTo(errorLog);
    }

// In ExpressionSolver.getNextResult():
    public MathExpression.EvalResult getNextResult() {
        // 1. Check if we need to expand (using a local copy for thread safety)
        MathExpression.EvalResult[] currentPool = this.pool;
        if (poolPointer >= currentPool.length) {
            synchronized (this) {
                // Double-check pattern to prevent multi-thread OOM
                if (poolPointer >= pool.length) {
                    int newSize = pool.length * 2;
                    MathExpression.EvalResult[] newPool = new MathExpression.EvalResult[newSize];
                    System.arraycopy(pool, 0, newPool, 0, pool.length);
                    for (int i = pool.length; i < newSize; i++) {
                        newPool[i] = new MathExpression.EvalResult();
                    }
                    this.pool = newPool; // Now the pointer won't OOM
                }
            }
        }

        // 2. Fetch and Reset
        MathExpression.EvalResult result = pool[poolPointer++];
        result.reset(); // Clear old state (crucial for complex objects!)
        return result;
    }

// Add a reset method to clear the pool between evaluations
    private void resetPool() {
        poolPointer = 0;
    }

    public void setWillFoldConstants(boolean willFoldConstants) {
        this.willFoldConstants = willFoldConstants;
    }

    public boolean isWillFoldConstants() {
        return willFoldConstants;
    }

    /**
     * Hardened production bridge. Zero allocation for arity <= 8. Safely scales
     * for any arity without crashing.
     */
    public static MathExpression.EvalResult invokeRegistryMethod(int methodId, double[] argsValues) {
        MathExpression.EvalResult[] wrappers = WRAPPER_CACHE.get();
        int arity = argsValues.length;

        if (arity > wrappers.length) {
            int newSize = Math.max(arity, wrappers.length * 2);
            MathExpression.EvalResult[] newWrappers = new MathExpression.EvalResult[newSize];
            System.arraycopy(wrappers, 0, newWrappers, 0, wrappers.length);
            for (int i = wrappers.length; i < newSize; i++) {
                newWrappers[i] = new MathExpression.EvalResult();
            }
            wrappers = newWrappers;
            WRAPPER_CACHE.set(wrappers);
        }

        for (int i = 0; i < arity; i++) {
            wrappers[i].wrap(argsValues[i]);
        }

        MathExpression.EvalResult resultContainer = new MathExpression.EvalResult();

        // Execute the registry action
        MethodRegistry.getAction(methodId).calc(resultContainer, arity, wrappers);

        // BULLETPROOF: Return the whole object. 
        // If it's a number, the caller can cast it; if it's an array, it's preserved.
        return resultContainer;
    }

    /**
     * Compile scalar expression to EvalResult-wrapped bytecode.
     *
     * The compilation process: 1. Build MethodHandle chain from postfix tokens
     * 2. Each handle transforms (double[]) -> double 3. Binary ops: combine
     * left & right, permute args 4. Wrap result in EvalResult.wrap(scalar)
     *
     * @return A FastCompositeExpression that returns wrapped scalar
     * @throws Throwable if compilation fails
     *
     * @Override public FastCompositeExpression compile() throws Throwable { //
     * This now yields a handle with signature (double[])Object MethodHandle
     * scalarHandle = compileScalar(postfix);
     *
     * return new FastCompositeExpression() {
     * @Override public MathExpression.EvalResult apply(double[] variables) {
     * try { // invoke() now returns Double (boxed) or double[] Object result =
     * scalarHandle.invoke(variables);
     *
     * if (result instanceof double[]) { return new
     * MathExpression.EvalResult().wrap((double[]) result); } return new
     * MathExpression.EvalResult().wrap(((Number) result).doubleValue()); }
     * catch (Throwable t) { throw new RuntimeException("Turbo evaluation
     * failed", t); } }
     *
     * @Override public double applyScalar(double[] variables) { try { Object
     * result = scalarHandle.invoke(variables);
     *
     * if (result instanceof Number) { return ((Number) result).doubleValue(); }
     * // Coercion: If the user calls a vector function in a scalar context, //
     * we return the first element to prevent a crash. double[] arr = (double[])
     * result; return (arr != null && arr.length > 0) ? arr[0] : Double.NaN; }
     * catch (Throwable t) { throw new RuntimeException("Turbo primitive
     * execution failed", t); } } }; }
     */
    ///////////////////////////////////////////UPDATE STARTS////////////////////////////////////////////////////////////////////////////////
    
        @Override
    public FastCompositeExpression compile() throws Throwable {
        // 1. The RAW handle: Returns primitive double, accepts double[]
        // Modified via compileScalar to directly target user input array indices.
        final MethodHandle rawScalarHandle = compileScalar(postfix);

        // 2. The GENERIC handle: Returns Object, accepts double[]
        final MethodHandle genericHandle = rawScalarHandle.asType(
                java.lang.invoke.MethodType.methodType(Object.class, double[].class)
        );

        return new FastCompositeExpression() {
            // NOTE: The internal 'args[]' tracking array is completely eliminated!

            @Override
            public double applyScalar(double[] variables) {
                try {
                    // ZERO loops. ZERO array copying. ZERO bounds matching overhead.
                    // The MethodHandle pipeline reads straight from the input array.
                    return (double) rawScalarHandle.invokeExact(variables);
                } catch (Throwable t) {
                    t.printStackTrace();
                    throw new RuntimeException("Turbo evaluation failed", t);
                }
            }

            @Override
            public MathExpression.EvalResult apply(double[] variables) {
                try {
                    return handleResult(genericHandle.invokeExact(variables));
                } catch (Throwable t) {
                    errorLog.error(t);
                    return execute(variables);
                }
            }

            private MathExpression.EvalResult execute(double[] variables) {
                try {
                    //Object result = genericHandle.invokeExact(MathExpression.EvalResult.ERROR);
                    Object result = genericHandle.invokeExact(variables);
                    return handleResult(result);
                } catch (Throwable t) {
                    errorLog.error(t);
                    return MathExpression.EvalResult.ERROR;
                }
            }

            private MathExpression.EvalResult handleResult(Object result) {
                if (result instanceof MathExpression.EvalResult) {
                    return (MathExpression.EvalResult) result;
                }
                if (result instanceof double[]) {
                    return getNextResult().wrap((double[]) result);
                }
                if (result instanceof Double) {
                    return getNextResult().wrap((double) result);
                }
                return MathExpression.EvalResult.ERROR;
            }

            @Override
            public String checkErrorLogs() {
                String logs = errorLog.getLogs();
                errorLog.print();
                return logs;
            }

            @Override
            public TurboExpressionEvaluator getCompiler() {
                return ScalarTurboEvaluator1.this;
            }
        };
    }

    private static boolean isIntrinsic(String name, int arity) {
        if (arity < 1 || arity > 2) {
            return false;
        }

        String lowerName = name.toLowerCase();
        // 1. Direct match (e.g., "sin_deg", "sqrt")
        if (FAST_PATH_METHODS.contains(lowerName)) {
            return true;
        }

        // 2. Base match (e.g., "sin" is intrinsic, so "sin_anything" is intrinsic)
        if (lowerName.contains("_")) {
            String base = lowerName.split("_")[0];
            // Only allow splitting if the base itself is a known intrinsic
            return FAST_PATH_METHODS.contains(base);
        }

        return false;
    }

    private void pushAndVerify(Stack<MethodHandle> stack, MethodHandle handle) {
        // CRITICAL: Any handle entering the stack MUST be (double[])double
        if (handle.type().parameterCount() != 1 || handle.type().parameterType(0) != double[].class) {
            throw new IllegalStateException("CORRUPTED HANDLE PUSHED TO STACK! Signature: " + handle.type());
        }
        stack.push(handle);
    }

    protected MethodHandle compileScalar(MathExpression.Token[] postfix) throws Throwable {
        Stack<MethodHandle> stack = new Stack<>();
        for (MathExpression.Token t : postfix) {
            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    // Start with: () -> double
                    MethodHandle constant = MethodHandles.constant(double.class, t.value);
                    // Transform to: (double[]) -> double
                    pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));
                    break;
                case MathExpression.Token.VARIABLE:
                    // Determine which index of the runtime 'variables' array contains this variable's value
                    int targetVariableIndex = -1;
                    if (slots != null) {
                        for (int k = 0; k < slots.length; k++) {
                            if (slots[k] == t.frameIndex) {
                                targetVariableIndex = k;
                                break;
                            }
                        }
                    }
                    // Fallback to frameIndex if it's an unmapped execution slot parameter
                    int finalIndex = (targetVariableIndex != -1) ? targetVariableIndex : t.frameIndex;

                    // Signature: (double[]) -> double (Reading straight from target index position)
                    MethodHandle arrayGetter = AndroidFriendlyMethodHandles.getDoubleArrayGetter();
                    System.out.println("");
                    pushAndVerify(stack, MethodHandles.insertArguments(arrayGetter, 1, finalIndex));
                    break;

                case MathExpression.Token.MATRIX:
                    //TODO: DETERMNINE IF TO ROUTE TO ParserNG Standard Matrix Handler OR WHETHER TO throw Exception and direct user to MatrixTurboEvaluator
                    // Start with: () -> double
                    constant = MethodHandles.constant(double.class, t.value);
                    // Transform to: (double[]) -> double
                    pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));
                    break;
                case MathExpression.Token.FUNCTION_HANDLE:
                    // Start with: () -> double
                    constant = MethodHandles.constant(double.class, t.value);
                    // Transform to: (double[]) -> double
                    pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));
                    break;
                case MathExpression.Token.FUNCTION_HANDLE_UNDEFINED:
                    // Start with: () -> double
                    constant = MethodHandles.constant(double.class, t.value);
                    // Transform to: (double[]) -> double
                    pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));
                    break;

                case MathExpression.Token.OPERATOR:
                    if (t.isPostfix || t.opChar == '√') {
                        pushAndVerify(stack, applyUnaryOp(t.opChar, ensurePrimitive(stack.pop())));
                    } else if (t.opChar == '^') {
                        MethodHandle exponentHandle = ensurePrimitive(stack.pop());
                        MethodHandle baseHandle = ensurePrimitive(stack.pop());
                        double constantExponent = getConstantValue(exponentHandle);

                        if (!Double.isNaN(constantExponent)) {
                            // Fetch the direct, fast (double)double handle for this exponent
                            MethodHandle optimizedUnary = getOptimizedPowerHandle(constantExponent);

                            // If base and exponent are literal constant numbers, fold them immediately
                            if (baseHandle.type().parameterCount() == 0 && exponentHandle.type().parameterCount() == 0) {
                                double baseVal = (double) baseHandle.invokeExact();
                                double result = (double) optimizedUnary.invokeExact(baseVal);
                                pushAndVerify(stack, MethodHandles.constant(double.class, result));
                            } else {
                                // Pipe the (double[])double output of the base straight into the optimized power handle
                                pushAndVerify(stack, MethodHandles.filterReturnValue(baseHandle, optimizedUnary));
                            }
                        } else {
                            // Fall back to standard binary operation processing if the exponent is a variable (e.g. x^y)
                            pushAndVerify(stack, applyBinaryOpNoPermute(t.opChar, baseHandle, exponentHandle));
                        }
                    } else {
                        MethodHandle right = ensurePrimitive(stack.pop());
                        MethodHandle left = ensurePrimitive(stack.pop());
                        pushAndVerify(stack, applyBinaryOpNoPermute(t.opChar, left, right));
                    }
                    break;

                case MathExpression.Token.METHOD:
                    String name = t.name.toLowerCase();
                    Function exec = null;
                    if (Method.isPureStatsMethod(name)) {
                        int arity = t.arity;
                        String[] rawArgs = t.getRawArgs();

                        ByteArrayBuilder bab = new ByteArrayBuilder();

                        for (int i = 0; i < arity; i++) {
                            stack.pop();
                            String arg = rawArgs[i];

                            if (com.github.gbenroscience.parser.Number.isNumber(arg)) {
                                bab.append(com.github.gbenroscience.parser.Number.fastParseDouble(arg));
                            } else {
                                MathExpression.EvalResult ev = new MathExpression(arg).solveGeneric();//compound args 
                                if (ev.type == MathExpression.EvalResult.TYPE_SCALAR) {
                                    bab.append(ev.scalar);
                                } else if (ev.type == MathExpression.EvalResult.TYPE_VECTOR) {
                                    bab.append(ev.vector);
                                }
                            }
                        }
                        double[] data = bab.getAsDoubleArray();

                        MethodHandle finalOp;
                        if (name.equals(Declarations.SORT) || name.equals(Declarations.MODE)) {
                            finalOp = MethodHandles.insertArguments(VECTOR_GATEKEEPER_HANDLE, 0, name, data);
                            finalOp = finalOp.asType(finalOp.type().changeReturnType(double[].class));
                        } else {
                            finalOp = MethodHandles.insertArguments(SCALAR_GATEKEEPER_HANDLE, 0, name, data);
                            finalOp = finalOp.asType(finalOp.type().changeReturnType(double.class));
                        }
                        finalOp = MethodHandles.dropArguments(finalOp, 0, double[].class);

                        pushAndVerify(stack, finalOp);
                        break;
                    } else if (name.equals(Declarations.QUADRATIC) || name.equals(Declarations.TARTAGLIA_ROOTS)) {
                        int arity = t.arity;
                        String[] rawArgs = t.getRawArgs();
                        stack.pop();

                        MethodHandle finalOp = MethodHandles.insertArguments(VECTOR_2_GATEKEEPER_HANDLE, 0, name, rawArgs[0]);
                        finalOp = finalOp.asType(finalOp.type().changeReturnType(double[].class));
                        finalOp = MethodHandles.dropArguments(finalOp, 0, double[].class);

                        pushAndVerify(stack, finalOp);
                        break;
                    } else if (name.equals(Declarations.GENERAL_ROOT)) {
                        int arity = t.arity;
                        for (int i = 0; i < arity; i++) {
                            stack.pop();
                        }
                        String[] args = t.getRawArgs();
                        if (args.length != arity) {
                            throw new RuntimeException("Invalid input. Expression did not pass token compiler phase");
                        }
                        if (args.length > 4) {
                            throw new RuntimeException("Invalid input. Argument count for general root is invalid. Expected: <=4 Found " + args.length);
                        }

                        String fNameOrExpr = args[0];
                        Function f = Variable.isVariableString(fNameOrExpr) ? FunctionManager.lookUp(fNameOrExpr) : FunctionManager.add(fNameOrExpr);

                        String varName = f.getIndependentVariables().get(0).getName();
                        double lower = args.length > 1 ? Double.parseDouble(args[1]) : -2;
                        double upper = args.length > 2 ? Double.parseDouble(args[2]) : 2;
                        int iterations = args.length > 3 ? Integer.parseInt(args[3]) : TurboRootFinder.DEFAULT_ITERATIONS;

                        MathExpression innerExpr = f.getMathExpression();
                        int xSlot = innerExpr.getVariable(varName).getFrameIndex();
                        MethodHandle targetHandle = compileScalar(innerExpr.getCachedPostfix());

                        MethodHandle derivHandle = null;
                        try {
                            String diffExpr = "diff(" + fNameOrExpr + ",1)";
                            String derivString = Derivative.eval(diffExpr).textRes;
                            derivHandle = compileScalar(FunctionManager.lookUp(derivString).getMathExpression().getCachedPostfix());
                        } catch (Exception e) {
                            errorLog.error(e);
                            e.printStackTrace();
                            derivHandle = null;
                        }

                        MethodHandle bridge = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "executeTurboRoot",
                                MethodType.methodType(double.class, MethodHandle.class, MethodHandle.class,
                                        int.class, double.class, double.class, int.class));

                        MethodHandle currentHandle = MethodHandles.insertArguments(bridge, 0,
                                targetHandle, derivHandle, xSlot, lower, upper, iterations);

                        currentHandle = MethodHandles.dropArguments(currentHandle, 0, double[].class);
                        pushAndVerify(stack, currentHandle);
                        break;
                    } else if (name.equals("print")) {
                        int arity = t.arity;

                        for (int i = 0; i < arity; i++) {
                            stack.pop();
                        }

                        String[] rawArgs = t.getRawArgs();

                        if (rawArgs == null || rawArgs.length != arity) {
                            throw new RuntimeException("Compile Error: Print arity mismatch.");
                        }

                        try {
                            MathExpression.EvalResult soln = executePrint(getNextResult(), rawArgs, registry);
                            constant = MethodHandles.constant(MathExpression.EvalResult.class, soln);
                            pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));

                        } catch (Exception e) {
                            throw new RuntimeException("Failed to bind print handle: " + e.getMessage(), e);
                        }
                        break;
                    } else if (name.equals("intg")) {
                        int arity = t.arity;
                        for (int i = 0; i < arity; i++) {
                            stack.pop();
                        }

                        String[] rawArgs = t.getRawArgs();
                        if (rawArgs.length != arity) {
                            throw new RuntimeException("Invalid input. Expression did not pass token compiler phase");
                        }
                        if (rawArgs.length != 3 && rawArgs.length != 4) {
                            throw new RuntimeException("Invalid input. Incomplete arguments for definite integral function: `intg`");
                        }

                        Function f = FunctionManager.lookUp(rawArgs[0]);
                        MathExpression innerExpr = f.getMathExpression();
                        MethodHandle compiledInner = compileScalar(innerExpr.getCachedPostfix());

                        double lower = Double.parseDouble(rawArgs[1]);
                        double upper = Double.parseDouble(rawArgs[2]);
                        int iterations = (int) ((arity == 4) ? (int) Double.parseDouble(rawArgs[3]) : (int) ((upper - lower) / 0.05));
                        String[] vars = innerExpr.getVariablesNames();
                        int[] slots = innerExpr.getSlots();

                        MethodHandle bridge = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "executeTurboIntegral",
                                MethodType.methodType(double.class, Function.class, MethodHandle.class, double.class, double.class, int.class,
                                        String[].class, int[].class));

                        MethodHandle finalIntgHandle = MethodHandles.insertArguments(bridge, 0, f, compiledInner, lower, upper, iterations, vars, slots);

                        pushAndVerify(stack, MethodHandles.dropArguments(finalIntgHandle, 0, double[].class));
                        break;
                    } else if (name.equals("diff")) {
                        for (int i = 0; i < t.arity; i++) {
                            stack.pop();
                        }

                        String[] args = t.getRawArgs();

                        if (args == null || args.length == 0) {
                            throw new IllegalArgumentException("Method 'diff' requires arguments.");
                        }
                        if (args.length != t.arity) {
                            throw new RuntimeException("Invalid input. Expression did not pass token compiler phase");
                        }
                        if (args.length > 3) {
                            throw new RuntimeException("Invalid input. Argument count for general root is invalid. Expected: <=3 Found " + args.length);
                        }

                        String returnHandle = null;
                        double evalPoint = -1;
                        int order = -1;
                        String targetExpr = args[0];

                        MathExpression.EvalResult solution = null;
                        switch (args.length) {
                            case 1:
                                targetExpr = args[0];
                                order = 1;
                                solution = Derivative.eval("diff(" + targetExpr + "," + order + ")");
                                break;
                            case 2:
                                targetExpr = args[0];
                                if (com.github.gbenroscience.parser.Number.isNumber(args[1])) {
                                    order = Integer.parseInt(args[1]);
                                    solution = Derivative.eval("diff(" + targetExpr + "," + order + ")");
                                } else if (Variable.isVariableString(args[1])) {
                                    returnHandle = args[1];
                                    FunctionManager.lockDown(returnHandle, args);
                                    solution = Derivative.eval("diff(" + targetExpr + "," + returnHandle + ")");
                                }
                                break;
                            case 3:
                                targetExpr = args[0];
                                if (com.github.gbenroscience.parser.Number.isNumber(args[2])) {
                                    order = Integer.parseInt(args[2]);
                                } else if (Variable.isVariableString(args[1])) {
                                    throw new RuntimeException("The 3rd argument of the diff command is the order of differentiation! It must be a whole number!");
                                }

                                if (com.github.gbenroscience.parser.Number.isNumber(args[1])) {
                                    evalPoint = Integer.parseInt(args[1]);
                                    solution = Derivative.eval("diff(" + targetExpr + "," + evalPoint + "," + order + ")");
                                } else if (Variable.isVariableString(args[1])) {
                                    returnHandle = args[1];
                                    FunctionManager.lockDown(returnHandle, args);
                                    solution = Derivative.eval("diff(" + targetExpr + "," + returnHandle + "," + order + ")");
                                }
                                break;

                            default:
                                throw new AssertionError();
                        }

                        if (solution.getType() == TYPE.NUMBER) {
                            // Outcome A: Evaluated at a static point, returns a hard number.
                            constant = MethodHandles.constant(double.class, solution.scalar);
                            pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class)); // FIXED SIGNATURE
                        } else if (solution.getType() == TYPE.STRING || solution.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
                            // Outcome B: Symbolic Derivative! (e.g., "cos(x)")
                            // We compile the new algebraic string directly into the current MethodHandle tree!
                            constant = MethodHandles.constant(MathExpression.EvalResult.class, solution);
                            pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));
                            /*
                            MathExpression solutionExpr = new MathExpression(solution.textRes, true);
                            MethodHandle compiledDeriv = compileScalar(solutionExpr.getCachedPostfix());
                            stack.push(ensurePrimitive(compiledDeriv));
                             */
                        } else {
                            String err = "Invalid expression passed to `diff` method: " + targetExpr;
                            errorLog.info(err);
                            throw new RuntimeException(err);
                        }
                        break;
                    } else if (name.equals(Declarations.DIFF_EQN)) {
                        for (int i = 0; i < t.arity; i++) {
                            stack.pop();
                        }

                        String[] args = t.getRawArgs();
                        if (args == null || args.length < 4 || args.length > 5) {
                            String err = "Invalid input for `diffeq`";
                            errorLog.info(err);
                            throw new RuntimeException(err);
                        }

                    } else if (name.equals("rot")) {
                        for (int i = 0; i < t.arity; i++) {
                            stack.pop();
                        }
                        String[] args = t.getRawArgs();
                        if (args == null || args.length < 4 || args.length > 5) {
                            String err = "Invalid input for `rot`";
                            errorLog.info(err);
                            throw new RuntimeException(err);
                        }

                        MathExpression.EvalResult solution = executeRotor(t.arity, args);

                        if (solution.getType() == TYPE.STRING || solution.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
                            constant = MethodHandles.constant(MathExpression.EvalResult.class, solution);
                            // The pipeline ALWAYS inputs a double[] array, so we drop double[].class
                            pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));

                        } else if (solution.getType() == TYPE.VECTOR) {
                            constant = MethodHandles.constant(double[].class, solution.vector);
                            pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));

                        } else {
                            // Fallback for EvalResult Error types
                            constant = MethodHandles.constant(MathExpression.EvalResult.class, solution);
                            pushAndVerify(stack, MethodHandles.dropArguments(constant, 0, double[].class));
                        }
                        break;
                    } else if (name.equals("log")) {
                        System.out.println("log!!! found!!!! arity=" + t.arity + ", args=" + t.getRawArgs().length + ",\n args=" + Arrays.toString(t.getRawArgs()));
                    } else if (isIntrinsic(name, t.arity)) {
                        // Optimization patch for your isIntrinsic block
                        if (t.arity == 1) {
                            MethodHandle operand = ensurePrimitive(stack.pop());
                            MethodHandle fn = getUnaryFunctionHandle(name);

                            // If the operand is a constant literal, fold the entire intrinsic operation!
                            if (operand.type().parameterCount() == 0) {
                                double innerConstant = (double) operand.invokeExact();
                                // Pre-calculate the result (e.g., sqrt(6.28) -> 2.50)
                                double foldedResult = (double) fn.invokeExact(innerConstant);

                                pushAndVerify(stack, MethodHandles.constant(double.class, foldedResult));
                            } else {
                                // Fall back to standard runtime handle generation
                                pushAndVerify(stack, MethodHandles.filterArguments(fn, 0, operand));
                            }
                        } else if (t.arity == 2) {
                            MethodHandle right = ensurePrimitive(stack.pop());
                            MethodHandle left = ensurePrimitive(stack.pop());
                            MethodHandle fn = getBinaryFunctionHandle(name);

                            int lParams = left.type().parameterCount();
                            int rParams = right.type().parameterCount();

                            // === CONSTANT FOLDING: Upgraded with strict primitive invokeExact extractions ===
                            if (lParams == 0 && rParams == 0) {
                                double lVal = (double) left.invokeExact();
                                double rVal = (double) right.invokeExact();
                                double result = (double) fn.invokeExact(lVal, rVal);
                                pushAndVerify(stack, MethodHandles.constant(double.class, result));
                            } else if (lParams == 1 && rParams == 0) {
                                double val = (double) right.invokeExact();
                                pushAndVerify(stack, MethodHandles.filterArguments(MethodHandles.insertArguments(fn, 1, val), 0, left));
                            } else if (lParams == 0 && rParams == 1) {
                                double val = (double) left.invokeExact();
                                pushAndVerify(stack, MethodHandles.filterArguments(MethodHandles.insertArguments(fn, 0, val), 0, right));
                            } else {
                                // Order arguments matching your original working method signature
                                // Assuming signature is: applyBinaryOpNoPermute(char op, MethodHandle left, MethodHandle right)
                                // If passing 'fn' directly, adjust your method parameters accordingly!
                                pushAndVerify(stack, applyBinaryOpNoPermute(t.opChar, left, right));
                            }
                        }
                    }// Inside your FUNCTION/METHOD case handler for user-defined functions
                    else {
                        // 1. Pop arguments
                        List<MethodHandle> argumentHandles = new ArrayList<>(t.arity);
                        for (int i = 0; i < t.arity; i++) {
                            argumentHandles.add(0, stack.pop());
                        }

                        // 2. Fallback to the reliable Registry Method invocation
                        // This avoids the type-mismatch errors inherent in manual inlining
                        pushAndVerify(stack, compileFunction(t, argumentHandles));
                        break;
                    }
            }
        }

        if (stack.size() != 1) {
            String err = "Invalid postfix expression: stack size = " + stack.size();
            errorLog.info(err);
            throw new IllegalArgumentException(err);
        }

        MethodHandle result = stack.pop();
        if (result.type().parameterCount() == 0) {
            result = MethodHandles.dropArguments(result, 0, double[].class);
        }
        if (result.type().returnType() == double.class) {
            return result;
        } else {
            // For both double[] and EvalResult returns, we wrap the return type to Object,
            // BUT the input parameter MUST remain double[].class for the turboArgs pipeline.
            return result.asType(MethodType.methodType(Object.class, double[].class));
        }
    }

    /**
     * Extracts the primitive scalar value from a MethodHandle if it represents
     * a structural constant literal node (i.e., a zero-argument constant
     * handle). Returns null if the handle depends on variable parameters.
     *
     * @param handle The MethodHandle node to check.
     * @return The constant double value, or null if it is a variable
     * expression.
     */
    /**
     * Extracts the primitive value from a constant handle without any object
     * boxing. Returns Double.NaN if the handle is dynamic or requires parameter
     * variables.
     */
    private static double getConstantValue(MethodHandle handle) {
        // A handle can only be a constant literal if it requires ZERO parameters
        if (handle != null && handle.type().parameterCount() == 0) {
            try {
                // Returns a raw primitive double. Zero allocation. Zero boxing.
                return (double) handle.invokeExact();
            } catch (Throwable t) {
                return Double.NaN;
            }
        }
        return Double.NaN;
    }

    /**
     * Combines multiple handles of type (double[])double into a single handle
     * of type (double[])double[].
     *
     * @param argumentHandles
     * @throws Exception
     */
    public static MethodHandle combineArgs(List<MethodHandle> argumentHandles) throws Exception {
        int arity = argumentHandles.size();

        // JVM Slot Limit check: 1 double = 2 slots. 
        // We cap at 120 to be safe (leaving room for 'this' and other overhead).
        if (arity <= 120) {
            // --- FAST PATH: Recursive Inlining ---
            MethodHandle packer;
            if (arity <= 3) {
                packer = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "pack" + arity,
                        MethodType.methodType(double[].class, Collections.nCopies(arity, double.class)));
            } else {
                // General packer for 4 to 120 args
                MethodHandle base = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "packN",
                        MethodType.methodType(double[].class, double[].class));
                packer = base.asVarargsCollector(double[].class)
                        .asType(MethodType.methodType(double[].class, Collections.nCopies(arity, double.class)));
            }

            MethodHandle combined = packer;
            for (int i = 0; i < arity; i++) {
                combined = MethodHandles.collectArguments(combined, i, argumentHandles.get(i));
            }

            int[] reorder = new int[arity]; // All point to parameter 0 (global double[] vars)
            return MethodHandles.permuteArguments(combined,
                    MethodType.methodType(double[].class, double[].class), reorder);

        } else {
            // --- ROBUST PATH: Massive Arity (120+ args) ---
            // We cannot expand these into a MethodType descriptor, 
            // so we pass them as an array to a loop-based evaluator.
            MethodHandle[] handleArray = argumentHandles.toArray(new MethodHandle[0]);
            MethodHandle massive = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "massivePacker",
                    MethodType.methodType(double[].class, MethodHandle[].class, double[].class));

            // Bind the handle array so the final signature is just (double[] vars) -> double[]
            return MethodHandles.insertArguments(massive, 0, (Object) handleArray);
        }
    }

// Optimization: Faster array creation for common math functions
    private static double[] pack1(double a) {
        return new double[]{a};
    }

    private static double[] pack2(double a, double b) {
        return new double[]{a, b};
    }

    private static double[] pack3(double a, double b, double c) {
        return new double[]{a, b, c};
    }

// Fallback for functions with many arguments
    private static double[] packN(double... args) {
        return args;
    }

    /**
     * The "Future-Proof" evaluator for functions with hundreds of arguments. It
     * bypasses MethodType slot limits by using array iteration.
     */
    private static double[] massivePacker(MethodHandle[] subExprs, double[] vars) throws Throwable {
        double[] results = new double[subExprs.length];
        for (int i = 0; i < subExprs.length; i++) {
            results[i] = (double) subExprs[i].invokeExact(vars);
        }
        return results;
    }

    private static MethodHandle compileFunction(MathExpression.Token t, List<MethodHandle> argumentHandles) throws Throwable {
        int methodId = MethodRegistry.getMethodID(t.name);

        // Bridge that handles the result mapping
        MethodHandle bridge = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "invokeRegistryMethod",
                MethodType.methodType(MathExpression.EvalResult.class, int.class, double[].class));

        MethodHandle boundBridge = MethodHandles.insertArguments(bridge, 0, methodId);

        // If any argument is NOT a primitive double (e.g., a nested sort() returning double[]),
        // we must use the "Massive/Complex" packer instead of the primitive collector.
        boolean hasComplexArgs = argumentHandles.stream().anyMatch(h -> h.type().returnType() != double.class);

        if (hasComplexArgs) {
            // Use a variant of our bridge that can handle Object[] inputs
            MethodHandle massive = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "invokeComplexStats",
                    MethodType.methodType(Object.class, MethodHandle.class, String.class, Object[].class, double[].class));

            // 1. Determine the appropriate gatekeeper
            String lowerName = t.name.toLowerCase();
            MethodHandle gatekeeper;
            if (lowerName.equals(Declarations.SORT) || lowerName.equals(Declarations.MODE)) {
                gatekeeper = VECTOR_GATEKEEPER_HANDLE;
            } else if (lowerName.equals(Declarations.QUADRATIC) || lowerName.equals(Declarations.TARTAGLIA_ROOTS)) {
                gatekeeper = VECTOR_2_GATEKEEPER_HANDLE;
            } else {
                gatekeeper = SCALAR_GATEKEEPER_HANDLE;
            }

            // 2. THE FIX: Bind starting from index 0, injecting the gatekeeper first!
            return MethodHandles.insertArguments(massive, 0, gatekeeper, t.name, argumentHandles.toArray());
        }
        // Standard primitive path
        MethodHandle collector = boundBridge.asCollector(double[].class, argumentHandles.size());
        for (int i = 0; i < argumentHandles.size(); i++) {
            collector = MethodHandles.collectArguments(collector, i, argumentHandles.get(i));
        }
        return collector;
    }

    public static Object invokeComplexStats1(MethodHandle gatekeeper, String name, Object[] parts, double[] vars) throws Throwable {
        int totalSize = 0;
        Object[] evaluated = new Object[parts.length];

        // Phase 1: Dynamic Evaluation & Unboxing
        for (int i = 0; i < parts.length; i++) {
            Object p = parts[i];
            // Static double[] are used directly; handles are executed
            Object res = (p instanceof double[]) ? p : ((MethodHandle) p).invokeExact(vars);

            if (res instanceof MathExpression.EvalResult) {
                MathExpression.EvalResult er = (MathExpression.EvalResult) res;
                res = (er.type == MathExpression.EvalResult.TYPE_VECTOR) ? er.vector : er.scalar;
            }

            evaluated[i] = res;
            if (res instanceof double[]) {
                totalSize += ((double[]) res).length;
            } else {
                totalSize++;
            }
        }

        // Phase 2: High-Speed Flattening
        double[] data = new double[totalSize];
        int cursor = 0;
        for (Object obj : evaluated) {
            if (obj instanceof double[]) {
                double[] d = (double[]) obj;
                System.arraycopy(d, 0, data, cursor, d.length);
                cursor += d.length;
            } else {
                data[cursor++] = ((Number) obj).doubleValue();
            }
        }
        return gatekeeper.invoke(name, data);
    }

    public static Object invokeComplexStats(MethodHandle gatekeeper, String name, Object[] parts, double[] vars) throws Throwable {
        int totalSize = 0;
        Object[] evaluated = new Object[parts.length];

// Phase 1: Dynamic Evaluation
        for (int i = 0; i < parts.length; i++) {
            Object p = parts[i];
            Object res;

            if (p instanceof double[]) {
                res = p;
            } else {
                MethodHandle mh = (MethodHandle) p;
                // CRITICAL: We check the return type of the handle BEFORE calling it
                Class<?> returnType = mh.type().returnType();

                if (returnType == double.class) {
                    // This is the "fast path" - no boxing, no object creation
                    res = (double) mh.invokeExact(vars);
                } else if (returnType == MathExpression.EvalResult.class) {
                    MathExpression.EvalResult er = (MathExpression.EvalResult) mh.invokeExact(vars);
                    res = (er.type == MathExpression.EvalResult.TYPE_VECTOR) ? er.vector
                            : (er.type == MathExpression.EvalResult.TYPE_MATRIX ? er.matrix.getFlatArray() : er.scalar);
                } else {
                    // Fallback for types that aren't primitives
                    res = mh.invoke(vars);
                }
            }

            evaluated[i] = res;
            totalSize += (res instanceof double[]) ? ((double[]) res).length : 1;
        }

        // Phase 2: High-Speed Flattening
        double[] data = new double[totalSize];
        int cursor = 0;
        for (Object obj : evaluated) {
            if (obj instanceof double[]) {
                double[] d = (double[]) obj;
                System.arraycopy(d, 0, data, cursor, d.length);
                cursor += d.length;
            } else {
                data[cursor++] = ((Number) obj).doubleValue();
            }
        }
        return gatekeeper.invoke(name, data);
    }

    // ========== ARITY REDUCTION & FOLDING ==========
    private static MethodHandle applyUnaryOp(char op, MethodHandle operand) throws Throwable {
        // Constant Folding: √4 -> 2.0
        if (operand.type().parameterCount() == 0) {
            double val = (double) operand.invokeExact();
            return MethodHandles.constant(double.class, (double) getUnaryOpHandle(op).invokeExact(val));
        }
        return MethodHandles.filterArguments(getUnaryOpHandle(op), 0, operand);
    }

    /**
     * Apply binary operator WITHOUT using permuteArguments. Uses foldArguments
     * for natural data flow that the JIT can inline.
     *
     */
    private static MethodHandle applyBinaryOpNoPermute(char op, MethodHandle left, MethodHandle right) throws Throwable {
        MethodHandle opHandle = getBinaryOpHandle(op);
        int lParams = left.type().parameterCount();
        int rParams = right.type().parameterCount();

        // === CONSTANT FOLDING ===
        if (lParams == 0 && rParams == 0) {
            double lVal = (double) left.invokeExact();
            double rVal = (double) right.invokeExact();
            double result = (double) opHandle.invokeExact(lVal, rVal);
            return MethodHandles.constant(double.class, result);
        }

        // === ARITY REDUCTION: One constant, one variable ===
        if (lParams == 0 && rParams >= 1) {
            double lVal = (double) left.invokeExact();
            MethodHandle boundFn = MethodHandles.insertArguments(opHandle, 0, lVal);
            return MethodHandles.filterArguments(boundFn, 0, right);
        }

        if (lParams >= 1 && rParams == 0) {
            double rVal = (double) right.invokeExact();
            MethodHandle boundFn = MethodHandles.insertArguments(opHandle, 1, rVal);
            return MethodHandles.filterArguments(boundFn, 0, left);
        }

        // === BOTH VARIABLES: Direct Combiner Generation ===
        MethodHandle combiner = createBinaryCombiner(opHandle, left);
        MethodHandle out = MethodHandles.foldArguments(combiner, right);
        return out;
    }

    /**
     * Create a combiner for binary operations.
     *
     * Combiner signature: (double rightValue, double[] array) -> double
     * Behavior: Computes fn(left(array), rightValue)
     *
     * This avoids permutation by threading array through naturally.
     */
    private static MethodHandle createBinaryCombiner(MethodHandle fn, MethodHandle left) throws Throwable {
        // We need: (double, double[]) -> double
        // That computes: fn(left(array), rightValue)

        // Step 1: Extract left value from array
        // Create: (double[] array) -> double by calling left
        MethodHandle left_from_array = left;  // left: (double[]) -> double

        // Step 2: Create (double rightValue, double[] array) -> (double, double)
        // We want to compute: (leftValue, rightValue) for fn
        // Where leftValue = left(array)
        // Use filterArguments to inject left evaluation:
        // filterArguments(fn, 1, left):
        //   - fn: (double, double) -> double (leftValue, rightValue)
        //   - Position 1 gets replaced with left(array)
        //   - Position 0 (rightValue) stays
        //   Result: (double rightValue, double[] array) -> double
        //   BUT this gives us: fn(rightValue, left(array))
        //   We need: fn(left(array), rightValue)
        // So we need to swap the arguments to fn first:
        MethodHandle fn_swapped = MethodHandles.permuteArguments(
                fn,
                MethodType.methodType(double.class, double.class, double.class),
                1, 0 // Arguments to fn come in positions (1, 0), i.e., swapped
        );
        // fn_swapped: (double rightValue, double leftValue) -> fn(leftValue, rightValue)

        // Step 3: Now apply filterArguments to inject left computation
        // Position 1 in fn_swapped will be replaced with left(array)
        MethodHandle combiner = MethodHandles.filterArguments(
                fn_swapped,
                1, // Position 1 of fn_swapped signature: replace with left(array)
                left_from_array
        );
        // combiner: (double rightValue, double[] array) -> double

        return combiner;
    }

    /**
     * Safe runtime execution bridge for custom FunctionManager functions.
     *
     * @param funcName
     * @param argsValues
     * @return
     */
    public static double invokeCustomFunction(String funcName, double[] argsValues) {
        Function f = FunctionManager.lookUp(funcName);
        if (f == null) {
            return Double.NaN;
        }

        int arity = argsValues.length;

        // We instantiate fresh EvalResults here instead of using the ThreadLocal WRAPPER_CACHE.
        // Why? Because custom functions can call other custom functions recursively!
        // If we use a shared cache, a nested function call will overwrite the variables of the parent call.
        MathExpression.EvalResult[] wrappers = new MathExpression.EvalResult[arity];
        for (int i = 0; i < arity; i++) {
            wrappers[i] = new MathExpression.EvalResult().wrap(argsValues[i]);
        }

        MathExpression.EvalResult resultContainer = new MathExpression.EvalResult();
        f.calc(resultContainer, arity, wrappers);

        return resultContainer.scalar;
    }

    public static MethodHandle createConstantHandle(double value) {
        // 1. Create a handle that just returns the literal value: ()double
        MethodHandle c = MethodHandles.constant(double.class, value);

        // 2. Transform it to: (double[])double
        // It accepts the array but ignores it, always returning 'value'
        return MethodHandles.dropArguments(c, 0, double[].class);
    }

    private static MethodHandle ensurePrimitive(MethodHandle handle) {
        if (handle.type().returnType() == double.class) {
            return handle;
        }
        // If it's an EvalResult (from registry), we need to extract the scalar.
        if (handle.type().returnType() == MathExpression.EvalResult.class || handle.type().returnType() == Object.class) {
            try {
                // Bridge: (Object) -> double
                MethodHandle unwrapper = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "unwrapToDouble",
                        MethodType.methodType(double.class, Object.class));
                return MethodHandles.filterReturnValue(handle, unwrapper);
            } catch (Throwable e) {
                e.printStackTrace();
                return handle.asType(handle.type().changeReturnType(double.class));
            }
        }
        return handle.asType(handle.type().changeReturnType(double.class));
    }

    // Helper for the unwrapper above
    public static double unwrapToDouble(Object obj) {
        if (obj instanceof MathExpression.EvalResult) {
            return ((MathExpression.EvalResult) obj).scalar;
        }
        if (obj instanceof Number) {
            return ((Number) obj).doubleValue();
        }
        if (obj instanceof double[]) {
            double[] arr = (double[]) obj;
            return arr.length > 0 ? arr[0] : Double.NaN;
        }
        return Double.NaN;
    }

    //////////////////////////////////////////////WIDE-METHODS///////////////////////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * Get the MethodHandle for a binary operator.
     */
    private static MethodHandle getBinaryOpHandle(char op) throws Throwable {
        switch (op) {
            case '+':
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "add", MT_DOUBLE_DD);
            case '-':
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "subtract", MT_DOUBLE_DD);
            case '*':
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "multiply", MT_DOUBLE_DD);
            case '/':
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "divide", MT_DOUBLE_DD);
            case '%':
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "modulo", MT_DOUBLE_DD);
            case '^':
                return LOOKUP.findStatic(Math.class, "pow", MT_DOUBLE_DD);
            case 'P':
                return LOOKUP.findStatic(Maths.class, "permutation", MT_DOUBLE_DD);
            case 'C':
                return LOOKUP.findStatic(Maths.class, "combination", MT_DOUBLE_DD);
            default:
                throw new IllegalArgumentException("Unsupported binary operator: " + op);
        }
    }

    static MathExpression.EvalResult executePrint(MathExpression.EvalResult ctx, String[] args, MathExpression.VariableRegistry registry) {

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
            Variable myVar = registry.lookUp(arg, false);
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

    public static double executeTurboIntegral(Function f, MethodHandle handle, double lower, double upper, int iterations, String[] vars, int[] slots) throws Throwable {
        MethodHandle primitiveHandle = handle.asType(
                MethodType.methodType(double.class, double[].class)
        );
        boolean shouldSwap = lower > upper;
        if (shouldSwap) {
            NumericalIntegrator numericalIntegrator = new NumericalIntegrator(f, primitiveHandle, upper, lower, vars, slots);
            return numericalIntegrator.integrate(f);
        } else {
            NumericalIntegrator numericalIntegrator = new NumericalIntegrator(f, primitiveHandle, lower, upper, vars, slots);
            return numericalIntegrator.integrate(f);
        }

    }

    public static double scalarStatsGatekeeper(String method, double[] data) {
        return executeScalarReturningStatsMethod(null, data, method);
    }

// For things like SORT, MODE
    public static double[] vectorStatsGatekeeper(String method, double[] data) {
        return executeVectorReturningStatsMethod(null, data, method);
    }

    public static double[] vectorNonStatsGatekeeper(String method, String funcHandle) {
        if (method.equals(Declarations.QUADRATIC)) {
            return executeQuadraticRoot(funcHandle);
        } else if (method.equals(Declarations.TARTAGLIA_ROOTS)) {
            return executeTartagliaRoot(funcHandle);
        }
        return new double[]{};
    }

    private static double executeScalarReturningStatsMethod(MethodHandle handle, double[] args, String method) {
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

    private static double[] executeVectorReturningStatsMethod(MethodHandle handle, double[] args, String method) {
        int n = args.length;

        switch (method) {
            case Declarations.MODE: {
                Arrays.sort(args); // Sort first to group identical values

                // First pass: Find the maximum frequency
                int maxCount = 0;
                int currentCount = 1;
                for (int i = 1; i < n; i++) {
                    if (args[i] == args[i - 1]) {
                        currentCount++;
                    } else {
                        if (currentCount > maxCount) {
                            maxCount = currentCount;
                        }
                        currentCount = 1;
                    }
                }
                if (currentCount > maxCount) {
                    maxCount = currentCount;
                }

                // Second pass: Collect all values that match maxCount
                // We use a temporary list or a precisely sized array
                double[] tempModes = new double[n];
                int modeIdx = 0;
                currentCount = 1;

                // Handle single element case
                if (n == 1) {
                    return new double[]{args[0]};
                }

                for (int i = 1; i < n; i++) {
                    if (args[i] == args[i - 1]) {
                        currentCount++;
                    } else {
                        if (currentCount == maxCount) {
                            tempModes[modeIdx++] = args[i - 1];
                        }
                        currentCount = 1;
                    }
                }
                if (currentCount == maxCount) {
                    tempModes[modeIdx++] = args[n - 1];
                }

                // Return a trimmed array containing only the modes
                return Arrays.copyOf(tempModes, modeIdx);
            }
            case Declarations.SORT: {
                // Arrays.sort mutates the array in-place extremely fast.
                Arrays.sort(args);
                // Since this method strictly returns a double, returning the array itself isn't possible here.
                // Returning args[0] gives the caller the first element, while the array remains sorted in memory.
                return args;
            }

            default:
                return null;
        }
    }

    /**
     * Execution bridge for the TurboRootFinder. This is invoked by the compiled
     * MethodHandle chain.
     *
     * @param baseHandle
     * @param derivHandle
     * @param xSlot
     * @param iterations
     * @param lower
     * @param upper
     * @return
     */
    public static double executeTurboRoot(MethodHandle baseHandle, MethodHandle derivHandle,
            int xSlot, double lower, double upper, int iterations) {
        // We use a default iteration cap of 1000 for the turbo version
        TurboRootFinder trf = new TurboRootFinder(baseHandle, derivHandle, xSlot, lower, upper, iterations);
        return trf.findRoots();
    }

    public static double[] executeQuadraticRoot(String funcHandle) {
        Function f = FunctionManager.lookUp(funcHandle);
        String input = f.expressionForm();
        input = input.substring(1);//remove the @
        int closeBracOfAt = Bracket.getComplementIndex(true, 0, input);
        input = input.substring(closeBracOfAt + 1);
        if (input.startsWith("(")) {
            input = input.substring(1, input.length() - 1);
            input = input.concat("=0");
        }
        Quadratic_Equation solver = new Quadratic_Equation(input);
        QuadraticSolver alg = solver.getAlgorithm();
        if (alg.isComplex()) {
            return alg.solutions;
        } else {
            return new double[]{alg.solutions[0], alg.solutions[2]};
        }
    }

    public static double[] executeTartagliaRoot(String funcHandle) {
        Function f = FunctionManager.lookUp(funcHandle);
        String input = f.expressionForm();
        input = input.substring(1);//remove the @
        int closeBracOfAt = Bracket.getComplementIndex(true, 0, input);
        input = input.substring(closeBracOfAt + 1);
        if (input.startsWith("(")) {
            input = input.substring(1, input.length() - 1);
            input = input.concat("=0");
        }
        Tartaglia_Equation solver = new Tartaglia_Equation(input);
        solver.getAlgorithm().solve();
        double[] solns = solver.getAlgorithm().solutions;
        return solns;
    }

    static final MathExpression.EvalResult executeRotor(int arity, String[] args) {
        MathExpression.EvalResult ctx = new MathExpression.EvalResult();
        int sz = args.length;
        if (args.length == 4) {//rot(F,a,O,D) function, angle, origin, direction vector
            //confirm the last 3 other args
            double angle = Variable.getConstantValue(args[1]);
            if (Double.isNaN(angle)) {
                angle = Double.parseDouble(args[1]);
            }
            String anonFuncOrig = args[2];
            String anonFuncDir = args[3];
            Function origFun = FunctionManager.lookUp(anonFuncOrig);
            Function dirFun = FunctionManager.lookUp(anonFuncDir);
            if (origFun == null) {
                return MathExpression.EvalResult.ERROR;
            }
            Point origin;
            Matrix origVector = origFun.getMatrix();
            int rows = origVector.getRows();
            int cols = origVector.getCols();//@(1,3)
            if ((rows == 1 && cols == 3) || (rows == 3 && cols == 1)) {
                double[] arr = origVector.getFlatArray();
                origin = new Point(arr[0], arr[1], arr[2]);
            } else {
                return MathExpression.EvalResult.ERROR;
            }

            Matrix dirVector = dirFun.getMatrix();
            rows = dirVector.getRows();
            cols = dirVector.getCols();//@(1,3)
            Direction dir;
            if ((rows == 1 && cols == 3) || (rows == 3 && cols == 1)) {
                double[] arr = dirVector.getFlatArray();
                dir = new Direction(arr[0], arr[1], arr[2]);
            } else {
                return MathExpression.EvalResult.ERROR;
            }

            Function f = FunctionManager.lookUp(args[0]);
            if (f == null) {
                return MathExpression.EvalResult.ERROR;
            }
            ArrayList<Variable> vars = f.getIndependentVariables();
            int siz = vars.size();
            if (siz > 3) {
                return MathExpression.EvalResult.ERROR;
            }
            if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
                String expr = f.getMathExpression().getExpression();
                String fullExpr = f.getName() + "=" + expr;
                String transformedFuncName = f.getName();

                ROTOR r = new ROTOR(angle, origin, dir);

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
                return ctx.wrap(res);
            }
            if (f.getType() == TYPE.MATRIX) {
                //rotate a set of points
                Matrix pointVectors = f.getMatrix();
                ROTOR r = new ROTOR(angle, origin, dir);
                rows = pointVectors.getRows();
                cols = pointVectors.getCols();//@(n,3)

                if (cols == 3) {
                    int n = rows;
                    double outArr[] = new double[3 * n];
                    int j = 0;
                    for (int i = 0; i < n; i++) {
                        double[] pm = pointVectors.getRowMatrix(i).getFlatArray();
                        Point p = new Point(pm[0], pm[1], pm[2]);
                        Point rotP = r.rotate(p);
                        outArr[j] = rotP.x;
                        outArr[j + 1] = rotP.y;
                        outArr[j + 2] = rotP.z;
                        j += 3;
                    }
                    return ctx.wrap(new Matrix(outArr, rows, 3));
                } else {
                    return MathExpression.EvalResult.ERROR;
                }
            }
        } else if (args.length == 5) {//rot(P1,P2,a,O,D) function, angle, origin, direction vector---- rotates lines, P1 and P2 are point matrices that define a line
            //confirm the last 3 other args

            double angle = Variable.getConstantValue(args[2]);
            if (Double.isNaN(angle)) {
                angle = Double.parseDouble(args[2]);
            }
            String anonFuncOrig = args[3];
            String anonFuncDir = args[4];
            Function origFun = FunctionManager.lookUp(anonFuncOrig);
            Function dirFun = FunctionManager.lookUp(anonFuncDir);
            if (origFun == null) {
                return MathExpression.EvalResult.ERROR;
            }
            Point origin;
            Matrix origVector = origFun.getMatrix();
            int rows = origVector.getRows();
            int cols = origVector.getCols();//@(1,3)
            if ((rows == 1 && cols == 3) || (rows == 3 && cols == 1)) {
                double[] arr = origVector.getFlatArray();
                origin = new Point(arr[0], arr[1], arr[2]);
            } else {
                return MathExpression.EvalResult.ERROR;
            }

            Matrix dirVector = dirFun.getMatrix();
            rows = dirVector.getRows();
            cols = dirVector.getCols();//@(1,3)
            Direction dir;
            if ((rows == 1 && cols == 3) || (rows == 3 && cols == 1)) {
                double[] arr = dirVector.getFlatArray();
                dir = new Direction(arr[0], arr[1], arr[2]);
            } else {
                return MathExpression.EvalResult.ERROR;
            }

            Function p1 = FunctionManager.lookUp(args[0]);
            Function p2 = FunctionManager.lookUp(args[1]);
            if (p1 == null) {
                return MathExpression.EvalResult.ERROR;
            }

            if (p1.getType() == TYPE.MATRIX && p2.getType() == TYPE.MATRIX) {
                ROTOR r = new ROTOR(angle, origin, dir);

                //rotate a point
                Matrix p1Vector = p1.getMatrix();
                Matrix p2Vector = p2.getMatrix();
                int r1 = p1Vector.getRows();
                int c1 = p1Vector.getCols();//@(1,3)
                int r2 = p2Vector.getRows();
                int c2 = p2Vector.getCols();//@(1,3)
                if (((r1 == 1 && c1 == 3) || (r1 == 3 && c1 == 1)) && ((r2 == 1 && c2 == 3) || (r2 == 3 && c2 == 1))) {
                    double[] arr1 = p1Vector.getFlatArray();
                    Point p11 = new Point(arr1[0], arr1[1], arr1[2]);
                    double[] arr2 = p2Vector.getFlatArray();
                    Point p22 = new Point(arr2[0], arr2[1], arr2[2]);

                    Line3D l3D = new Line3D(p11, p22);

                    Line3D rotL3D = r.rotate(l3D);

                    Point p11Rot = r.rotate(p11);
                    Point p22Rot = r.rotate(p22);
                    Matrix m = new Matrix(new double[]{p11Rot.x, p11Rot.y, p11Rot.z, p22Rot.x, p22Rot.y, p22Rot.z}, 2, 3);
                    return ctx.wrap(m);
                } else {
                    return MathExpression.EvalResult.ERROR;
                }
            }

        } else {
            return MathExpression.EvalResult.ERROR;
        }
        return MathExpression.EvalResult.ERROR;

    }

    /**
     * Get the MethodHandle for a unary operator.
     */
    private static MethodHandle getUnaryOpHandle(char op) throws Throwable {
        switch (op) {
            case '√':
                // Optimization: Square root uses the CPU's fsqrt instruction
                return LOOKUP.findStatic(Math.class, "sqrt", MT_DOUBLE_D);
            case 'R':
                // Cube root (internal representation for ³√)
                return LOOKUP.findStatic(Math.class, "cbrt", MT_DOUBLE_D);
            case '²':
                // Optimization: Maps to x * x via specialized kernel
                return getOptimizedPowerHandle(2.0);
            case '³':
                // Optimization: Maps to x * x * x via specialized kernel
                return getOptimizedPowerHandle(3.0);
            case '!':
                // Factorial (External utility)
                return LOOKUP.findStatic(Maths.class, "fact", MT_DOUBLE_D);
            default:
                throw new IllegalArgumentException("Unsupported unary operator: " + op);
        }
    }

    public static MethodHandle getOptimizedPowerHandle(double n) throws Throwable {
        // Case: x^0
        if (n == 0.0) {
            return MethodHandles.constant(double.class, 1.0);
        }
        // Case: x^1
        if (n == 1.0) {
            //return MethodHandles.identity(double.class);
            return MethodHandlePolyfill.identity(double.class);
        }
        // Case: x^0.5 (Square Root)
        if (n == 0.5) {
            return LOOKUP.findStatic(Math.class, "sqrt", MethodType.methodType(double.class, double.class));
        }
        // Case: x^0.333... (Cube Root)
        if (Math.abs(n - (1.0 / 3.0)) < 1E-9) {
            return LOOKUP.findStatic(Math.class, "cbrt", MethodType.methodType(double.class, double.class));
        }
        // Case: x^2
        if (n == 2.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "square", MethodType.methodType(double.class, double.class));
        }
        // Case: x^3
        if (n == 3.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "cube", MethodType.methodType(double.class, double.class));
        }       // Case: x^4
        if (n == 4.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "quad", MethodType.methodType(double.class, double.class));
        }

        // Fallback: The "Heavy" Math.pow
        MethodHandle pow = LOOKUP.findStatic(Math.class, "pow",
                MethodType.methodType(double.class, double.class, double.class));
        return MethodHandles.insertArguments(pow, 1, n);
    }

    // ========== FUNCTIONS ==========
    /**
     * Checks if the prefix is a standard math function that supports unit
     * suffixes like _deg, _rad, or _grad.
     */
    private static boolean isStandardMathBase(String base) {
        switch (base.toLowerCase()) {
            case "sin":
            case "cos":
            case "tan":
            case "asin":
            case "acos":
            case "atan":
            case "sec":
            case "csc":
            case "cot":
            case "asec":
            case "acsc":
            case "acot":
            case "sinh":
            case "cosh":
            case "tanh":
            case "sech":
            case "csch":
            case "coth":
            case "asinh":
            case "acosh":
            case "atanh":
            case "asech":
            case "acsch":
            case "acoth":
            case "sin-¹":
            case "cos-¹":
            case "tan-¹":
            case "sec-¹":
            case "csc-¹":
            case "cot-¹":
            case "sinh-¹":
            case "cosh-¹":
            case "tanh-¹":
            case "sech-¹":
            case "csch-¹":
            case "coth-¹":
            case "log":
            case "ln":
            case "lg":
            case "log-¹":
            case "ln-¹":
            case "lg-¹":
                return true;
            default:
                return false;
        }
    }

    /**
     * Get unary function handle (arity 1).
     *
     * Supports: - Trigonometric: sin, cos, tan, asin, acos, atan (with DRG
     * variants) - Logarithmic: log, ln, log10 - Power/Root: sqrt, cbrt -
     * Rounding: floor, ceil, abs - Other: exp, fact
     */
    protected static MethodHandle getUnaryFunctionHandle(String name) throws Throwable {
        // Check for single-character operators like !
        if (name.length() == 1) {
            char op = name.charAt(0);
            if (op == '!') {
                return LOOKUP.findStatic(Maths.class, "fact", MT_DOUBLE_D);
            }
            // Handle unary plus/minus if needed
            if (op == '-') {
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "negate", MT_DOUBLE_D);
            }
            if (op == '+') {
                //return MethodHandles.identity(double.class);
                return MethodHandlePolyfill.identity(double.class);
            }
        }
        String lower = name.toLowerCase();
        String base;
        String unit = "rad";

        if (lower.contains("_")) {
            String[] parts = lower.split("_");
            String candidateBase = parts[0];
            // Only split if the first part is a known trig/math function
            if (isStandardMathBase(candidateBase)) {
                base = candidateBase;
                unit = parts[1];
            } else {
                base = lower; // Treat the whole thing (like t_root) as the base
            }
        } else {
            base = lower;
        }

        MethodHandle mh;

        switch (base) {
            case "²":
                // Optimization: Maps the postfix ² to the inlined square(x) method
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "square", MT_DOUBLE_D);

            case "³":
                // Optimization: Maps the postfix ³ to the inlined cube(x) method
                return LOOKUP.findStatic(ScalarTurboEvaluator1.class, "cube", MT_DOUBLE_D);
            case "sin":
                mh = LOOKUP.findStatic(Math.class, "sin", MT_DOUBLE_D);
                break;
            case "cos":
                mh = LOOKUP.findStatic(Math.class, "cos", MT_DOUBLE_D);
                break;
            case "tan":
                mh = LOOKUP.findStatic(Math.class, "tan", MT_DOUBLE_D);
                break;
            case "asin":
            case "sin-¹":
                mh = LOOKUP.findStatic(Math.class, "asin", MT_DOUBLE_D);
                break;
            case "acos":
            case "cos-¹":
                mh = LOOKUP.findStatic(Math.class, "acos", MT_DOUBLE_D);
                break;
            case "atan":
            case "tan-¹":
                mh = LOOKUP.findStatic(Math.class, "atan", MT_DOUBLE_D);
                break;
            case "sec":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "sec", MT_DOUBLE_D);
                break;
            case "csc":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "csc", MT_DOUBLE_D);
                break;
            case "cot":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "cot", MT_DOUBLE_D);
                break;

            case "asec":
            case "sec-¹":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "asec", MT_DOUBLE_D);
                break;
            case "acsc":
            case "csc-¹":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "acsc", MT_DOUBLE_D);
                break;
            case "acot":
            case "cot-¹":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "acot", MT_DOUBLE_D);
                break;
            case "sinh":
                mh = LOOKUP.findStatic(Math.class, "sinh", MT_DOUBLE_D);
                break;
            case "cosh":
                mh = LOOKUP.findStatic(Math.class, "cosh", MT_DOUBLE_D);
                break;
            case "tanh":
                mh = LOOKUP.findStatic(Math.class, "tanh", MT_DOUBLE_D);
                break;
            case "sech":
                mh = LOOKUP.findStatic(Maths.class, "sech", MT_DOUBLE_D); // Assuming Maths helper
                break;
            case "csch":
                mh = LOOKUP.findStatic(Maths.class, "csch", MT_DOUBLE_D);
                break;
            case "coth":
                mh = LOOKUP.findStatic(Maths.class, "coth", MT_DOUBLE_D);
                break;

// --- INVERSE HYPERBOLICS ---
            case "asinh":
            case "sinh-¹":
                mh = LOOKUP.findStatic(Maths.class, "asinh", MT_DOUBLE_D);
                break;
            case "acosh":
            case "cosh-¹":
                mh = LOOKUP.findStatic(Maths.class, "acosh", MT_DOUBLE_D);
                break;
            case "atanh":
            case "tanh-¹":
                mh = LOOKUP.findStatic(Maths.class, "atanh", MT_DOUBLE_D);
                break;
            case "asech":
            case "sech-¹":
                mh = LOOKUP.findStatic(Maths.class, "asech", MT_DOUBLE_D);
                break;
            case "acsch":
            case "csch-¹":
                mh = LOOKUP.findStatic(Maths.class, "acsch", MT_DOUBLE_D);
                break;
            case "acoth":
            case "coth-¹":
                mh = LOOKUP.findStatic(Maths.class, "acoth", MT_DOUBLE_D);
                break;
// --- ADDITIONAL LOG / EXP ---

            case "lg":
                mh = LOOKUP.findStatic(Math.class, "log10", MT_DOUBLE_D);
                break;
            case "ln":
            case "log":
                mh = LOOKUP.findStatic(Math.class, "log", MT_DOUBLE_D);
                break;
            case "alg": // Anti-log base 10
            case "lg-¹":
                mh = LOOKUP.findStatic(Maths.class, "antiLog10", MT_DOUBLE_D);
                break;
            case "exp":
            case "aln":
            case "ln-¹":
                mh = LOOKUP.findStatic(Math.class, "exp", MT_DOUBLE_D); // ln-¹ is just e^x
                break;

// --- ROUNDING / MISC ---
            case "ceil":
                mh = LOOKUP.findStatic(Math.class, "ceil", MT_DOUBLE_D);
                break;
            case "rnd":
                mh = LOOKUP.findStatic(Math.class, "round", MT_DOUBLE_D); // Note: might need cast logic
                break;
            case "√":
            case "sqrt":
                mh = LOOKUP.findStatic(Math.class, "sqrt", MT_DOUBLE_D);
                break;
            case "cbrt":
                mh = LOOKUP.findStatic(Math.class, "cbrt", MT_DOUBLE_D);
                break;
            case "abs":
                mh = LOOKUP.findStatic(Math.class, "abs", MT_DOUBLE_D);
                break;
            case "fact":
                mh = LOOKUP.findStatic(Maths.class, "fact", MT_DOUBLE_D);
                break;
            case "square":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "square", MT_DOUBLE_D);
                break;
            case "cube":
                mh = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "cube", MT_DOUBLE_D);
                break;
            case "erf":
                mh = LOOKUP.findStatic(Maths.class, "erf", MT_DOUBLE_D);
                break;
            case "gelu":
                mh = LOOKUP.findStatic(Maths.class, "gelu", MT_DOUBLE_D);
                break;
            case "fast_gelu":
                mh = LOOKUP.findStatic(Maths.class, "fastGelu", MT_DOUBLE_D);
                break;
            case "swiglu":
                mh = LOOKUP.findStatic(Maths.class, "swiglu", MT_DOUBLE_D);
                break;
            case "geglu":
                mh = LOOKUP.findStatic(Maths.class, "geglu", MT_DOUBLE_D);
                break;

            default:
                throw new UnsupportedOperationException("No fast-path for: " + base);
        }

        // Bake the unit conversion into the MethodHandle chain
        switch (unit) {
            case "deg":
                return chainToRadians(mh);
            case "grad":
                return chainGradToRadians(mh);
            default:
                return mh;
        }
    }

    public static double sec(double x) {
        return 1.0 / Math.cos(x);
    }

    public static double csc(double x) {
        return 1.0 / Math.sin(x);
    }

    public static double cot(double x) {
        return 1.0 / Math.tan(x);
    }

    public static double asec(double x) {
        return Math.acos(1.0 / x);
    }

    public static double acsc(double x) {
        return Math.asin(1.0 / x);
    }

    public static double acot(double x) {
        return Math.atan(1.0 / x);
    }

    public static double getDoubleFromArray(double[] array, int index) {
        return array[index];
    }

    /**
     * Chain Math.toRadians into a trigonometric function. This keeps the
     * conversion in the compiled bytecode.
     *
     * Pattern: trigOp(toRadians(x)) is compiled as a single chain.
     */
    private static MethodHandle chainToRadians(MethodHandle trigOp) throws Throwable {
        MethodHandle toRad = LOOKUP.findStatic(Math.class, "toRadians", MT_DOUBLE_D);
        return MethodHandles.filterArguments(trigOp, 0, toRad);
    }

    private static MethodHandle chainGradToRadians(MethodHandle trigOp) throws Throwable {
        // 1 grad = PI / 200 radians
        MethodHandle toRad = MethodHandles.filterArguments(
                LOOKUP.findStatic(ScalarTurboEvaluator1.class, "multiply", MT_DOUBLE_DD), // Custom multiply or use a constant
                0, MethodHandles.constant(double.class, Math.PI / 200.0)
        );
        // Simplified: Just use a helper method for clarity
        MethodHandle gradToRad = LOOKUP.findStatic(ScalarTurboEvaluator1.class, "gradToRad", MT_DOUBLE_D);
        return MethodHandles.filterArguments(trigOp, 0, gradToRad);
    }

    public static double gradToRad(double grads) {
        return grads * (Math.PI / 200.0);
    }

    /**
     * Get binary function handle (arity 2).
     *
     * Supports: - Power operations: pow - Trigonometric: atan2 - Comparison:
     * min, max
     */
    private static MethodHandle getBinaryFunctionHandle(String name) throws Throwable {
        // If it's a single character, it's likely one of your defined operators
        if (name.length() == 1) {
            try {
                return getBinaryOpHandle(name.charAt(0));
            } catch (IllegalArgumentException e) {
                // Not a known operator char, fall through to check function names
            }
        }
        switch (name.toLowerCase()) {
            case "pow":
                return LOOKUP.findStatic(Math.class, "pow", MT_DOUBLE_DD);
            case "atan2":
                return LOOKUP.findStatic(Math.class, "atan2", MT_DOUBLE_DD);
            case "min":
                return LOOKUP.findStatic(Math.class, "min", MT_DOUBLE_DD);
            case "max":
                return LOOKUP.findStatic(Math.class, "max", MT_DOUBLE_DD);
            case "geglu":
                return LOOKUP.findStatic(Maths.class, "geglu", MT_DOUBLE_DD);
            case "swiglu":
                return LOOKUP.findStatic(Maths.class, "swiglu", MT_DOUBLE_DD);
            case "log":
                // Fix: Binary log is usually log(x) / log(base)
                // If your Maths library has a log(a, b) method, point to it here.
                // Otherwise, remove this case to let it go through the legacy bridge.
                return LOOKUP.findStatic(Maths.class, "logToAnyBase", MT_DOUBLE_DD);
            case "alog":
            case "log-¹":
                return LOOKUP.findStatic(Maths.class, "antiLogToAnyBase", MT_DOUBLE_DD);
            case "comb":
            case "perm":
                return LOOKUP.findStatic(Maths.class,
                        name.equals("comb") ? "combination" : "permutation", MT_DOUBLE_DD);
            // --- FIX FOR POSTFIX SQUARED ---
            case "²":
                // Discards the second argument (the exponent 2.0) and calls square(x)
                return MethodHandles.dropArguments(
                        LOOKUP.findStatic(ScalarTurboEvaluator1.class, "square", MT_DOUBLE_D),
                        1, double.class);
            // --- FIX FOR POSTFIX CUBED ---
            case "³":
                // Discards the second argument (the exponent 3.0) and calls cube(x)
                return MethodHandles.dropArguments(
                        LOOKUP.findStatic(ScalarTurboEvaluator1.class, "cube", MT_DOUBLE_D),
                        1, double.class);
            default:
                // This is where "root", "diff", and "intg" will now fall through
                // if they are not in FAST_PATH_METHODS.
                throw new UnsupportedOperationException("Binary fast-path not found: " + name);
        }
    }

    // ========== INLINE ARITHMETIC HELPERS ==========
    /**
     * Inlined addition: a + b
     */
    public static double add(double a, double b) {
        return a + b;
    }

    /**
     * Inlined subtraction: a - b
     */
    public static double subtract(double a, double b) {
        return a - b;
    }

    /**
     * Inlined multiplication: a * b
     */
    public static double multiply(double a, double b) {
        return a * b;
    }

    /**
     * Inlined division: a / b with zero-check
     */
    public static double divide(double a, double b) {
        if (b == 0) {
            return a >= 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }
        return a / b;
    }

    /**
     * Inlined modulo: a % b
     */
    public static double modulo(double a, double b) {
        return a % b;
    }

    // Zero-overhead square
    public static double square(double x) {
        return x * x;
    }

    // Zero-overhead cube
    public static double cube(double x) {
        return x * x * x;
    }

    // Optional: Quad power (x^4) - still faster than Math.pow
    public static double quad(double x) {
        double x2 = x * x;
        return x2 * x2;
    }

    public static final class AndroidFriendlyMethodHandles {

        private static final MethodHandle DOUBLE_ARRAY_GETTER;

        static {
            MethodHandle getter = null;
            if (!Utils.isAndroid()) {
                try {
                    // Standard JVM path
                    getter = MethodHandles.arrayElementGetter(double[].class);
                } catch (Throwable t) {
                    // Fallback if the JVM environment is restricted
                }
            }

            if (getter == null) {
                try {
                    // Guaranteed safe path for Android
                    getter = MethodHandles.lookup().findStatic(
                            AndroidFriendlyMethodHandles.class,
                            "fallbackGetDouble",
                            MethodType.methodType(double.class, double[].class, int.class)
                    );
                } catch (NoSuchMethodException | IllegalAccessException e) {
                    throw new ExceptionInInitializerError("Failed to initialize polyfill: " + e.getMessage());
                }
            }
            DOUBLE_ARRAY_GETTER = getter;
        }

        public static double fallbackGetDouble(double[] array, int index) {
            if (array == null) {
                throw new IllegalArgumentException("Internal Error: Array passed to MethodHandle is null");
            }
            return array[index];
        }

        public static MethodHandle getDoubleArrayGetter() {
            return DOUBLE_ARRAY_GETTER;
        }
    }

    public static final class MethodHandlePolyfill {

        private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();

        // 1. Define the bare-metal static identity targets for Android fallback
        private static double idDouble(double x) {
            return x;
        }

        private static int idInt(int x) {
            return x;
        }

        private static long idLong(long x) {
            return x;
        }

        private static Object idObj(Object x) {
            return x;
        }

        // 2. Cache the handles
        private static final MethodHandle DOUBLE_ID;
        private static final MethodHandle INT_ID;
        private static final MethodHandle LONG_ID;
        private static final MethodHandle OBJ_ID;

        static {
            try {
                if (Utils.isAndroid()) {
                    // Android-specific polyfill paths
                    DOUBLE_ID = LOOKUP.findStatic(MethodHandlePolyfill.class, "idDouble", MethodType.methodType(double.class, double.class));
                    INT_ID = LOOKUP.findStatic(MethodHandlePolyfill.class, "idInt", MethodType.methodType(int.class, int.class));
                    LONG_ID = LOOKUP.findStatic(MethodHandlePolyfill.class, "idLong", MethodType.methodType(long.class, long.class));
                    OBJ_ID = LOOKUP.findStatic(MethodHandlePolyfill.class, "idObj", MethodType.methodType(Object.class, Object.class));
                } else {
                    // Standard JVM - Use native highly-optimized identity handles
                    DOUBLE_ID = MethodHandles.identity(double.class);
                    INT_ID = MethodHandles.identity(int.class);
                    LONG_ID = MethodHandles.identity(long.class);
                    OBJ_ID = MethodHandles.identity(Object.class);
                }
            } catch (NoSuchMethodException | IllegalAccessException e) {
                throw new ExceptionInInitializerError(e);
            }
        }

        /**
         * Drop-in replacement for MethodHandles.identity() that chooses the
         * most performant path based on the environment.
         *
         * @param type
         * @return
         */
        public static MethodHandle identity(Class<?> type) {
            if (type == double.class) {
                return DOUBLE_ID;
            }
            if (type == int.class) {
                return INT_ID;
            }
            if (type == long.class) {
                return LONG_ID;
            }

            if (!type.isPrimitive()) {
                // For reference types, adapt the cached Object identity to the specific class
                return OBJ_ID.asType(MethodType.methodType(type, type));
            }

            // Fallback for less common primitives (boolean, float, etc.)
            // This is safe on the JVM, but will still crash on older Android if called for these types.
            return MethodHandles.identity(type);
        }
    }

    public static double executeTurboODE(MethodHandle dy_dt, int tSlot, int ySlot, int frameSize,
            double t0, double y0, double tEnd, double initialStep,
            DifferentialEquations.ODESolverMethod method) throws Throwable {

        // 1. Pack the scalar y0 into a 1-element state vector array
        double[] y0Vector = new double[]{y0};
        int systemSize = 1;

        // 2. Dedeprecate fixed steps from initialStep for fixed-step solvers
        // If initialStep is h, then steps = (tEnd - t0) / h
        int steps = (int) Math.round((tEnd - t0) / initialStep);
        if (steps <= 0) {
            steps = 1; // Prevent division-by-zero or negative steps
        }
        double[] resultVector;

        // 3. Route parameters correctly matching your vectorized solvers
        switch (method) {
            case EULER:
                resultVector = DifferentialEquations.stepEuler(dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            case RK4:
                resultVector = DifferentialEquations.stepRK4(dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            case RK45_DORMAND_PRINCE:
                // Adaptive solver takes initialStep directly as initialH
                resultVector = DifferentialEquations.stepRK45Adaptive(dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, initialStep);
                break;
            case IMPLICIT_EULER:
                resultVector = DifferentialEquations.stepImplicitEuler(dy_dt, tSlot, ySlot, systemSize, frameSize, t0, y0Vector, tEnd, steps);
                break;
            default:
                throw new IllegalArgumentException("Unsupported ODE method: " + method);
        }

        // 4. Unpack and return the scalar result
        return resultVector[0];
    }

}
