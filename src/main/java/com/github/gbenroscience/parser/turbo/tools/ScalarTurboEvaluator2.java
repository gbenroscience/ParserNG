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

import com.github.gbenroscience.interfaces.Savable;
import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.math.differentialcalculus.Derivative;
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
import com.github.gbenroscience.util.VariableManager;

import java.lang.invoke.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Turbo compiler optimized for PURE SCALAR expressions. Uses widening
 * techniques to pass variables to the evaluator. May be intensely fast. But has
 * limits due to the JVMs limits of 256 slots(so 126-127 args for double args)
 * for method argument count Will balance this with
 * {@linkplain com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator1}
 * which is array based, to allow usage of expressions with thousands of
 * variables and more.
 *
 * @author GBEMIRO
 */
public class ScalarTurboEvaluator2 implements TurboExpressionEvaluator, Savable {

    private static final long serialVersionUID = 1L;
    private boolean willFoldConstants;

    protected final int[] slots;

    private MathExpression.Token[] postfix;
    private ErrorLog errorLog = new ErrorLog();
    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();
    private static final MethodHandle QUAD_HANDLE;
    private static final MethodHandle TARTAGLIA_HANDLE;

    static {
        try {
            QUAD_HANDLE = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeQuadraticRoot",
                    MethodType.methodType(double[].class, String.class));
            TARTAGLIA_HANDLE = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeTartagliaRoot",
                    MethodType.methodType(double[].class, String.class));
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new ExceptionInInitializerError("Failed to initialize Stats Gatekeepers: " + e.getMessage());
        }
    }

    private static final MethodType MT_DOUBLE_D = MethodType.methodType(double.class, double.class);
    private static final MethodType MT_DOUBLE_DD = MethodType.methodType(double.class, double.class, double.class);
    private static final MethodType MT_DOUBLE_OBJ = MethodType.methodType(double.class, Object.class);

    private static final ThreadLocal<MathExpression.EvalResult[]> WRAPPER_CACHE = ThreadLocal.withInitial(() -> {
        MathExpression.EvalResult[] arr = new MathExpression.EvalResult[8];
        for (int i = 0; i < 8; i++) {
            arr[i] = new MathExpression.EvalResult();
        }
        return arr;
    });

    private static final Map<String, MethodHandle> UNARY_MAP = new HashMap<>(128);
    private static final Map<String, MethodHandle> BINARY_MAP = new HashMap<>(32);
    // Define this at the top of ScalarTurboEvaluator2
    private static final MethodHandle MULTIPLY_HANDLE;
    private static final MethodHandle RECIPROCAL_HANDLE;
    private static final MethodHandle SQRT_HANDLE;

    static {
        try {
            MULTIPLY_HANDLE = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "multiply", MT_DOUBLE_DD);
            RECIPROCAL_HANDLE = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "reciprocal",
                    MethodType.methodType(double.class, double.class));
            SQRT_HANDLE = LOOKUP.findStatic(Math.class, "sqrt", MT_DOUBLE_D);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    static {
        try {
            // BINARY OPERATORS
            BINARY_MAP.put("+", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "add", MT_DOUBLE_DD));
            BINARY_MAP.put("-", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "subtract", MT_DOUBLE_DD));
            BINARY_MAP.put("*", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "multiply", MT_DOUBLE_DD));
            BINARY_MAP.put("/", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "divide", MT_DOUBLE_DD));
            BINARY_MAP.put("Р", LOOKUP.findStatic(Maths.class, "permutation", MT_DOUBLE_DD));
            BINARY_MAP.put("Č", LOOKUP.findStatic(Maths.class, "combination", MT_DOUBLE_DD));
            BINARY_MAP.put("%", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "modulo", MT_DOUBLE_DD));
            BINARY_MAP.put("^", LOOKUP.findStatic(Math.class, "pow", MT_DOUBLE_DD));
            BINARY_MAP.put("pow", BINARY_MAP.get("^"));
            BINARY_MAP.put("min", LOOKUP.findStatic(Math.class, "min", MT_DOUBLE_DD));
            BINARY_MAP.put("max", LOOKUP.findStatic(Math.class, "max", MT_DOUBLE_DD));
            BINARY_MAP.put("comb", LOOKUP.findStatic(Maths.class, "combination", MT_DOUBLE_DD));
            BINARY_MAP.put("perm", LOOKUP.findStatic(Maths.class, "permutation", MT_DOUBLE_DD));
            BINARY_MAP.put("log", LOOKUP.findStatic(Maths.class, "logToAnyBase", MT_DOUBLE_DD));
            BINARY_MAP.put("alog", LOOKUP.findStatic(Maths.class, "antiLogToAnyBase", MT_DOUBLE_DD));
            BINARY_MAP.put("log-¹", LOOKUP.findStatic(Maths.class, "antiLogToAnyBase", MT_DOUBLE_DD));

            // UNARY INTRINSICS
            UNARY_MAP.put("√", LOOKUP.findStatic(Math.class, "sqrt", MT_DOUBLE_D));
            UNARY_MAP.put("R", LOOKUP.findStatic(Math.class, "cbrt", MT_DOUBLE_D));
            UNARY_MAP.put("sqrt", UNARY_MAP.get("√"));
            UNARY_MAP.put("²", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "square", MT_DOUBLE_D));
            UNARY_MAP.put("square", UNARY_MAP.get("²"));
            UNARY_MAP.put("³", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "cube", MT_DOUBLE_D));
            UNARY_MAP.put("!", LOOKUP.findStatic(Maths.class, "fact", MT_DOUBLE_D));
            UNARY_MAP.put("cube", UNARY_MAP.get("³"));
            UNARY_MAP.put("cbrt", LOOKUP.findStatic(Math.class, "cbrt", MT_DOUBLE_D));

            UNARY_MAP.put("abs", LOOKUP.findStatic(Math.class, "abs", MT_DOUBLE_D));
            UNARY_MAP.put("ceil", LOOKUP.findStatic(Math.class, "ceil", MT_DOUBLE_D));
            UNARY_MAP.put("rnd", LOOKUP.findStatic(Math.class, "rint", MT_DOUBLE_D));
            UNARY_MAP.put("fact", LOOKUP.findStatic(Maths.class, "fact", MT_DOUBLE_D));

            MethodHandle exp = LOOKUP.findStatic(Math.class, "exp", MT_DOUBLE_D);
            MethodHandle log10 = LOOKUP.findStatic(Math.class, "log10", MT_DOUBLE_D);
            MethodHandle logE = LOOKUP.findStatic(Math.class, "log", MT_DOUBLE_D);
            MethodHandle alg = LOOKUP.findStatic(Maths.class, "antiLog10", MT_DOUBLE_D);

            UNARY_MAP.put("exp", exp);
            UNARY_MAP.put("ln", logE);
            UNARY_MAP.put("aln", exp);
            UNARY_MAP.put("lg", log10);
            UNARY_MAP.put("alg", alg);
            UNARY_MAP.put("ln-¹", exp);
            UNARY_MAP.put("lg-¹", alg);

            Map<String, MethodHandle> baseTrig = new HashMap<>();
            baseTrig.put("sin", LOOKUP.findStatic(Math.class, "sin", MT_DOUBLE_D));
            baseTrig.put("cos", LOOKUP.findStatic(Math.class, "cos", MT_DOUBLE_D));
            baseTrig.put("tan", LOOKUP.findStatic(Math.class, "tan", MT_DOUBLE_D));
            baseTrig.put("sec", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "sec", MT_DOUBLE_D));
            baseTrig.put("csc", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "csc", MT_DOUBLE_D));
            baseTrig.put("cot", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "cot", MT_DOUBLE_D));

            baseTrig.put("asin", LOOKUP.findStatic(Math.class, "asin", MT_DOUBLE_D));
            baseTrig.put("acos", LOOKUP.findStatic(Math.class, "acos", MT_DOUBLE_D));
            baseTrig.put("atan", LOOKUP.findStatic(Math.class, "atan", MT_DOUBLE_D));
            baseTrig.put("asec", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "asec", MT_DOUBLE_D));
            baseTrig.put("acsc", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "acsc", MT_DOUBLE_D));
            baseTrig.put("acot", LOOKUP.findStatic(ScalarTurboEvaluator2.class, "acot", MT_DOUBLE_D));

            BINARY_MAP.put("atan2", LOOKUP.findStatic(Math.class, "atan2", MT_DOUBLE_DD));

            baseTrig.put("sinh", LOOKUP.findStatic(Math.class, "sinh", MT_DOUBLE_D));
            baseTrig.put("cosh", LOOKUP.findStatic(Math.class, "cosh", MT_DOUBLE_D));
            baseTrig.put("tanh", LOOKUP.findStatic(Math.class, "tanh", MT_DOUBLE_D));
            baseTrig.put("sech", LOOKUP.findStatic(Maths.class, "sech", MT_DOUBLE_D));
            baseTrig.put("csch", LOOKUP.findStatic(Maths.class, "csch", MT_DOUBLE_D));
            baseTrig.put("coth", LOOKUP.findStatic(Maths.class, "coth", MT_DOUBLE_D));

            baseTrig.put("asinh", LOOKUP.findStatic(Maths.class, "asinh", MT_DOUBLE_D));
            baseTrig.put("acosh", LOOKUP.findStatic(Maths.class, "acosh", MT_DOUBLE_D));
            baseTrig.put("atanh", LOOKUP.findStatic(Maths.class, "atanh", MT_DOUBLE_D));
            baseTrig.put("asech", LOOKUP.findStatic(Maths.class, "asech", MT_DOUBLE_D));
            baseTrig.put("acsch", LOOKUP.findStatic(Maths.class, "acsch", MT_DOUBLE_D));
            baseTrig.put("acoth", LOOKUP.findStatic(Maths.class, "acoth", MT_DOUBLE_D));

            MethodHandle toRad = LOOKUP.findStatic(Math.class, "toRadians", MT_DOUBLE_D);
            MethodHandle gradToRad = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "gradToRad", MT_DOUBLE_D);

            for (Map.Entry<String, MethodHandle> entry : baseTrig.entrySet()) {
                String name = entry.getKey();
                MethodHandle handle = entry.getValue();

                UNARY_MAP.put(name, handle);
                UNARY_MAP.put(name + "_rad", handle);
                MethodHandle degHandle = MethodHandles.filterArguments(handle, 0, toRad);
                UNARY_MAP.put(name + "_deg", degHandle);
                MethodHandle gradHandle = MethodHandles.filterArguments(handle, 0, gradToRad);
                UNARY_MAP.put(name + "_grad", gradHandle);

                if (name.startsWith("a") && name.length() >= 4) {
                    String baseName = name.substring(1);
                    String inverseAlias = baseName + "-¹";
                    UNARY_MAP.put(inverseAlias, handle);
                    UNARY_MAP.put(inverseAlias + "_rad", handle);
                    UNARY_MAP.put(inverseAlias + "_deg", degHandle);
                    UNARY_MAP.put(inverseAlias + "_grad", gradHandle);
                }
            }
        } catch (Throwable t) {
            throw new ExceptionInInitializerError("Failed to initialize Turbo Handle Registry: " + t.getMessage());
        }
    }

    public ScalarTurboEvaluator2(MathExpression me) {
        if (ScalarTurboEvaluator.SUPPORTS_WIDENING) {
            this.postfix = me.getCachedPostfix();
            this.willFoldConstants = me.isWillFoldConstants();
            slots = me.getSlots();
            me.copyErrorLogTo(errorLog);
        } else {
            throw new UnsupportedOperationException("This evaluator does not support adaptive widening of method signatures\nPlease use ScalarTurboEvaluator1");
        }
    }

    private static MethodHandle getUnaryHandle(String op) {
        MethodHandle mh = UNARY_MAP.get(op);
        if (mh == null) {
            throw new UnsupportedOperationException("No unary fast-path for: " + op);
        }
        return mh;
    }

    private static MethodHandle getBinaryHandle(String op) {
        MethodHandle mh = BINARY_MAP.get(op);
        if (mh == null) {
            throw new UnsupportedOperationException("No binary fast-path for: " + op);
        }
        return mh;
    }

    public void setWillFoldConstants(boolean willFoldConstants) {
        this.willFoldConstants = willFoldConstants;
    }

    public boolean isWillFoldConstants() {
        return willFoldConstants;
    }

    public static Object invokeRegistryMethod(int methodId, double[] argsValues) {
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
        MethodRegistry.getAction(methodId).calc(resultContainer, arity, wrappers);

        if (resultContainer.getType() == TYPE.NUMBER) {
            return resultContainer.scalar;
        } else if (resultContainer.getType() == TYPE.VECTOR) {
            return resultContainer.vector;
        }
        return resultContainer;
    }

    @Override
    public FastCompositeExpression compile() throws Throwable {
        int varCount = countVariables(postfix); // Counts maxIndex + 1
        boolean foldConstants = this.willFoldConstants;
        try {
            // 1. The core math handle: accepts 'varCount' double arguments
            MethodHandle wideHandle = compileScalarWide(postfix, varCount, foldConstants);
            Class<?> returnType = wideHandle.type().returnType();

            // 2. Create a "Scalar-Only" version of the handle (returns primitive double)
            MethodHandle scalarOnlyHandle = wideHandle;
            if (returnType == double[].class) {
                MethodHandle getFirst = ScalarTurboEvaluator1.AndroidFriendlyMethodHandles.getDoubleArrayGetter();
                scalarOnlyHandle = MethodHandles.filterReturnValue(wideHandle,
                        MethodHandles.insertArguments(getFirst, 1, 0));
            } else if (returnType == MathExpression.EvalResult.class) {
                MethodHandle extractor = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "extractScalarFromResult",
                        MethodType.methodType(double.class, MathExpression.EvalResult.class));
                scalarOnlyHandle = MethodHandles.filterReturnValue(wideHandle, extractor);
            } else {
                scalarOnlyHandle = scalarOnlyHandle.asType(scalarOnlyHandle.type().changeReturnType(double.class));
            }

            // 3. Ultra-Turbo Compile-Time Inverse Spreader Strategy
            MethodHandle scalarSpreader;
            MethodHandle genericSpreader;

            if (varCount == 0) {
                scalarSpreader = MethodHandles.dropArguments(scalarOnlyHandle, 0, double[].class);
                MethodHandle genericBase = wideHandle.asType(wideHandle.type().changeReturnType(Object.class));
                genericSpreader = MethodHandles.dropArguments(genericBase, 0, double[].class);
            } else {
                MethodHandle baseScalar = scalarOnlyHandle;
                MethodHandle baseGeneric = wideHandle.asType(wideHandle.type().changeReturnType(Object.class));

                // Create an array of filters mapping straight to the user's incoming indices
                MethodHandle arrayGetter = ScalarTurboEvaluator1.AndroidFriendlyMethodHandles.getDoubleArrayGetter();
                MethodHandle[] filters = new MethodHandle[varCount];

                for (int k = 0; k < varCount; k++) {
                    // Find which index in the user's 'variables' array matches this frameIndex parameter 'k'
                    int targetVariableIndex = -1;
                    if (slots != null) {
                        for (int i = 0; i < slots.length; i++) {
                            if (slots[i] == k) {
                                targetVariableIndex = i;
                                break;
                            }
                        }
                    }

                    // Fallback to k if unmapped
                    int finalSourceIndex = (targetVariableIndex != -1) ? targetVariableIndex : k;

                    // Direct binding: tells the handle to pull straight from user input array position
                    filters[k] = MethodHandles.insertArguments(arrayGetter, 1, finalSourceIndex);
                }

                // Apply positional filters
                scalarSpreader = MethodHandles.filterArguments(baseScalar, 0, filters);
                genericSpreader = MethodHandles.filterArguments(baseGeneric, 0, filters);

                // Collapse all individual double[] argument requirements into ONE clean double[] argument
                int[] reorder = new int[varCount]; // Initialize as [0, 0, 0, ...]
                scalarSpreader = MethodHandles.permuteArguments(scalarSpreader,
                        MethodType.methodType(double.class, double[].class), reorder);
                genericSpreader = MethodHandles.permuteArguments(genericSpreader,
                        MethodType.methodType(Object.class, double[].class), reorder);
            }

            final MethodHandle finalScalar = scalarSpreader;
            final MethodHandle finalGeneric = genericSpreader;

            // Return the optimized execution frame without internal array state management
            return new FastCompositeExpression() {

                @Override
                public double applyScalar(double[] variables) {
                    try {
                        // Zero loops, zero array copies, zero Math.min checks.
                        // Variables flow directly from your iteration loop into CPU registers.
                        return (double) finalScalar.invokeExact(variables);
                    } catch (Throwable t) {
                        errorLog.error(t);
                        throw new RuntimeException("Turbo scalar execution failed", t);
                    }
                }

                @Override
                public MathExpression.EvalResult apply(double[] variables) {
                    try {
                        Object result = finalGeneric.invokeExact(variables);
                        MathExpression.EvalResult res = new MathExpression.EvalResult();
                        if (result instanceof double[]) {
                            res.type = MathExpression.EvalResult.TYPE_VECTOR;
                            res.vector = (double[]) result;
                        } else if (result instanceof MathExpression.EvalResult) {
                            return (MathExpression.EvalResult) result;
                        } else {
                            res.type = MathExpression.EvalResult.TYPE_SCALAR;
                            res.scalar = (double) result;
                        }
                        return res;
                    } catch (Throwable t) {
                        errorLog.error(t);
                        throw new RuntimeException("Turbo execution failed", t);
                    }
                }

                @Override
                public String checkErrorLogs() {
                    String logs = errorLog.getLogs();
                    errorLog.print();
                    return logs;
                }

                @Override
                public TurboExpressionEvaluator getCompiler() {
                    return ScalarTurboEvaluator2.this;
                }
            };
        } catch (Throwable t) {
            errorLog.error(t);
            throw new RuntimeException("Failed to compile Turbo expression", t);
        }
    }

    private static boolean isIntrinsic(String name, int arity) {
        if (arity == 1) {
            return UNARY_MAP.containsKey(name);
        } else if (arity == 2) {
            return BINARY_MAP.containsKey(name);
        }
        return false;
    }

    private MethodHandle compileScalarWide(MathExpression.Token[] postfix, int varCount, boolean foldConstants) throws Throwable {

        if (postfix == null || postfix.length == 0) {
            return MethodHandles.constant(double.class, 0.0);
        }

        Class<?>[] pTypes = new Class<?>[varCount];
        Arrays.fill(pTypes, double.class);

        int[] mask = new int[0];
        if (varCount > 0) {
            mask = new int[varCount * 2];
            for (int i = 0; i < varCount; i++) {
                mask[i] = i;
                mask[i + varCount] = i;
            }
        }

        Deque<MethodHandle> stack = new ArrayDeque<>();
        for (MathExpression.Token t : postfix) {

            switch (t.kind) {
                case MathExpression.Token.NUMBER:
                    stack.push(liftConstant(t.value, varCount));
                    break;
                case MathExpression.Token.VARIABLE:
                    stack.push(liftVariable(t.frameIndex, varCount));
                    break;
                case MathExpression.Token.MATRIX:
                    //TODO: DETERMNINE IF TO CHANNEL TO ParserNG Standard Matrix Handler OR WHETHER TO throw Exception and direct user to MatrixTurboEvaluator
                    stack.push(liftConstant(t.value, varCount));
                    break;
                case MathExpression.Token.FUNCTION_HANDLE:
                    stack.push(liftConstant(t.value, varCount));
                    break;
                case MathExpression.Token.FUNCTION_HANDLE_UNDEFINED:
                    stack.push(liftConstant(t.value, varCount));
                    break;
                case MathExpression.Token.OPERATOR:
                    if (t.isPostfix || t.opChar == '√') {
                        applyUnaryWide(t, stack, foldConstants, varCount);
                    } else if (t.opChar == '^') {
                        MethodHandle exponentHandle = stack.pop();
                        MethodHandle baseHandle = stack.pop();
                        Double constantExponent = getConstantValue(exponentHandle);

                        if (constantExponent != null) {
                            // Get the primitive (double)double handle for the specific exponent
                            MethodHandle optimizedUnary = getOptimizedPowerHandle(constantExponent);

                            // Case A: Everything is a structural constant (e.g., 2^3) -> Fully fold
                            if (foldConstants && baseHandle.type().parameterCount() == 0 && exponentHandle.type().parameterCount() == 0) {
                                double baseVal = (double) baseHandle.invokeExact();
                                double result = (double) optimizedUnary.invokeExact(baseVal);
                                stack.push(liftConstant(result, varCount));
                            } // Case B: Base is variable/expression, exponent is constant (e.g., x1^3)
                            else {
                                // Extract and lift ONLY if the base is truly a zero-argument constant literal node
                                if (baseHandle.type().parameterCount() == 0) {
                                    double constantBaseVal = (double) baseHandle.invokeExact();
                                    if (varCount > 0) {
                                        baseHandle = liftConstant(constantBaseVal, varCount);
                                    }
                                }

                                // High-Performance Pipe: optimizedUnary(baseHandle(vars))
                                // filterReturnValue pipes the scalar double output of baseHandle directly into optimizedUnary
                                MethodHandle chained = MethodHandles.filterReturnValue(baseHandle, optimizedUnary);
                                stack.push(chained);
                            }
                        } else {
                            // Standard Binary Path for dynamic powers (e.g., x^y) -> Falls back to native Math.pow
                            stack.push(baseHandle);
                            stack.push(exponentHandle);
                            applyBinaryWide(t, stack, foldConstants, varCount, pTypes, mask);
                        }
                    } else {
                        applyBinaryWide(t, stack, foldConstants, varCount, pTypes, mask);
                    }
                    break;

                case MathExpression.Token.FUNCTION:
                case MathExpression.Token.METHOD:
                    Function userFunc = FunctionManager.getFunction(t.name);

                    if (userFunc != null) {
                        int callArity = t.arity;
                        int defArity = userFunc.getArity();

                        if (callArity != defArity) {
                            String err = "Function " + t.name + " expects " + defArity + " arguments but received " + callArity;
                            errorLog.info(err);
                            throw new RuntimeException(err);
                        }

                        List<MethodHandle> argumentHandles = new ArrayList<>(callArity);
                        for (int i = 0; i < callArity; i++) {
                            MethodHandle h = stack.pop();
                            // Ensure every argument is widened to the outer scope
                            if (h.type().parameterCount() != pTypes.length) {
                                h = MethodHandles.dropArguments(h, 0, pTypes);
                            }
                            argumentHandles.add(0, h);
                        }

                        MathExpression bodyExpr = userFunc.getMathExpression();
                        MethodHandle bodyHandle = compileScalarWide(bodyExpr.getCachedPostfix(), defArity, willFoldConstants);

                        if (defArity == 0) {
                            stack.push(MethodHandles.dropArguments(bodyHandle, 0, pTypes));
                        } else {
                            int M = pTypes.length; // Number of variables in the calling scope
                            int N = defArity;      // Number of arguments the function accepts

                            if (N * M > 120) {
                                // JVM 255-slot safety net
                                stack.addAll(argumentHandles);
                                MethodHandle legacy = compileComplexFunction(t);
                                stack.push(adaptToWideSignature(legacy, callArity, stack, pTypes));
                            } else {
                                // FIX: Pipe arguments into the body using collectArguments
                                MethodHandle combined = bodyHandle;
                                for (int i = 0; i < N; i++) {
                                    // Because each collection replaces 1 parameter with M parameters,
                                    // the index shifts to the right by exactly M every iteration.
                                    combined = MethodHandles.collectArguments(combined, i * M, argumentHandles.get(i));
                                }

                                // Collapse the duplicated variables
                                int[] reorder = new int[N * M];
                                for (int i = 0; i < N; i++) {
                                    for (int j = 0; j < M; j++) {
                                        reorder[i * M + j] = j;
                                    }
                                }
                                MethodType targetType = MethodType.methodType(double.class, pTypes);
                                stack.push(MethodHandles.permuteArguments(combined, targetType, reorder));
                            }
                        }
                        break;
                    }
                    String name = t.name.toLowerCase();

                    if (name.equals(Declarations.QUADRATIC) || name.equals(Declarations.TARTAGLIA_ROOTS)
                            || name.equals(Declarations.GENERAL_ROOT) || name.equals(Declarations.DIFFERENTIATION) || name.equals(Declarations.INTEGRATION)
                            || name.equals(Declarations.DIFF_EQN)
                            || name.equals(Declarations.PRINT) || name.equals(Declarations.ROTOR)) {
                        MethodHandle legacy = compileComplexFunction(t);

                        if (legacy.type().parameterCount() == 0) {
                            if (varCount > 0) {
                                legacy = MethodHandles.dropArguments(legacy, 0, pTypes); // <-- FIX HERE
                            }
                            stack.push(legacy);
                        } else {
                            MethodHandle widened = adaptToWideSignature(legacy, t.arity, stack, pTypes);
                            stack.push(widened);
                        }
                        break;
                    } else if (isIntrinsic(name, t.arity)) {
                        if (t.arity == 1) {
                            applyUnaryWide(t, stack, foldConstants, varCount);
                        } else if (t.arity == 2) {
                            applyBinaryWide(t, stack, foldConstants, varCount, pTypes, mask);
                        }
                    } else if (Method.isListReturningStatsMethod(name) || Method.isNumberReturningStatsMethod(name)) {
                        MethodHandle[] argFilters = new MethodHandle[t.arity];
                        for (int i = t.arity - 1; i >= 0; i--) {
                            argFilters[i] = stack.pop();
                        }

                        boolean isVector = name.equals(Declarations.MODE) || name.equals(Declarations.SORT);
                        MethodHandle computation = isVector
                                ? LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeVectorReturningStatsMethod", MethodType.methodType(double[].class, double[].class, String.class))
                                : LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeScalarReturningStatsMethod", MethodType.methodType(double.class, double[].class, String.class));

                        MethodHandle bound = MethodHandles.insertArguments(computation, 1, name);

                        // Use flattenObjects to accept mixed types (Object[]) instead of strict double[]
                        MethodHandle arrayCollector = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "flattenObjects", MethodType.methodType(double[].class, Object[].class))
                                .asCollector(Object[].class, t.arity);

                        MethodHandle combined = MethodHandles.collectArguments(bound, 0, arrayCollector);

                        for (int i = t.arity - 1; i >= 0; i--) {
                            // Fix liftConstant to support constant vectors (e.g., from an inner sort(...) with no variables)
                            if (argFilters[i].type().parameterCount() == 0 && varCount > 0) {
                                Object val = argFilters[i].invoke();
                                if (val instanceof double[]) {
                                    MethodHandle c = MethodHandles.constant(double[].class, val);
                                    argFilters[i] = MethodHandles.dropArguments(c, 0, pTypes);
                                } else if (val instanceof Number) {
                                    argFilters[i] = liftConstant(((Number) val).doubleValue(), varCount);
                                } else {
                                    MethodHandle c = MethodHandles.constant(val.getClass(), val);
                                    argFilters[i] = MethodHandles.dropArguments(c, 0, pTypes);
                                }
                            }

                            // Dynamically align the combined handle's parameter type to match the exact return type of the filter.
                            // This prevents the IllegalArgumentException by handling auto-boxing for scalars.
                            MethodType currentType = combined.type();
                            MethodType targetType = currentType.changeParameterType(i, argFilters[i].type().returnType());
                            combined = combined.asType(targetType);

                            combined = MethodHandles.collectArguments(combined, i, argFilters[i]);
                        }

                        if (varCount > 0) {
                            int[] mergeMask = new int[t.arity * varCount];
                            for (int i = 0; i < t.arity; i++) {
                                for (int j = 0; j < varCount; j++) {
                                    mergeMask[i * varCount + j] = j;
                                }
                            }
                            combined = MethodHandles.permuteArguments(combined, MethodType.methodType(combined.type().returnType(), pTypes), mergeMask);
                        }
                        stack.push(combined);
                    } // Inside compileScalarWide -> case FUNCTION/METHOD:
                    else {
                        name = t.name.toLowerCase();
                        MethodHandle legacy = compileComplexFunction(t);
                        stack.push(adaptToWideSignature(legacy, t.arity, stack, pTypes));
                    }
                    break;
            }
        }

        if (stack.isEmpty()) {
            throw new IllegalStateException("Expression evaluation resulted in empty stack");
        }
        return stack.pop();
    }

    public static double[] collectToArray(double... values) {
        return values;
    }

    public static double[] flattenObjects(Object... args) {
        int len = 0;
        // Calculate final array size
        for (Object arg : args) {
            if (arg instanceof double[]) {
                len += ((double[]) arg).length;
            } else if (arg instanceof MathExpression.EvalResult) {
                MathExpression.EvalResult res = (MathExpression.EvalResult) arg;
                len += (res.type == MathExpression.EvalResult.TYPE_VECTOR) ? res.vector.length : 1;
            } else {
                len++; // Accounts for Number or NaN fallback
            }
        }

        double[] result = new double[len];
        int idx = 0;

        // Populate the array
        for (Object arg : args) {
            if (arg instanceof double[]) {
                double[] arr = (double[]) arg;
                System.arraycopy(arr, 0, result, idx, arr.length);
                idx += arr.length;
            } else if (arg instanceof Number) {
                result[idx++] = ((Number) arg).doubleValue();
            } else if (arg instanceof MathExpression.EvalResult) {
                MathExpression.EvalResult res = (MathExpression.EvalResult) arg;
                if (res.type == MathExpression.EvalResult.TYPE_VECTOR) {
                    System.arraycopy(res.vector, 0, result, idx, res.vector.length);
                    idx += res.vector.length;
                } else {
                    result[idx++] = res.scalar;
                }
            } else {
                result[idx++] = Double.NaN;
            }
        }
        return result;
    }

    public static double extractScalarFromResult(MathExpression.EvalResult res) {
        if (res == null) {
            return Double.NaN;
        }
        if (res.type == MathExpression.EvalResult.TYPE_SCALAR) {
            return res.scalar;
        }
        if (res.type == MathExpression.EvalResult.TYPE_VECTOR && res.vector != null && res.vector.length > 0) {
            return res.vector[0];
        }
        return Double.NaN; // Fallback for Strings (like print results) or undefined
    }

    private static MethodHandle compileRotorHandle(MathExpression.Token t) throws Throwable {
        int arity = t.arity;
        String[] args = t.getRawArgs();
        if (args == null || args.length == 0) {
            throw new IllegalArgumentException("Method 'rot' requires arguments.");
        }
        if (args.length != t.arity) {
            throw new RuntimeException("Invalid input. Expression did not pass token compiler phase");
        }
        if (args.length != 4 && args.length != 5) {
            throw new RuntimeException("Invalid input. Argument count for general root is invalid. Expected: 4 or 5 args: Found " + t.arity + " args");
        }
        MathExpression.EvalResult solution = ScalarTurboEvaluator1.executeRotor(arity, args);
        MethodHandle constant = MethodHandles.constant(MathExpression.EvalResult.class, solution);
        return constant;
    }

    private static MethodHandle compilePrintHandle(MathExpression.Token t) throws Throwable {
        String[] rawArgs = t.getRawArgs();
        MathExpression.EvalResult e = new MathExpression.EvalResult();
        e = executePrint(e, rawArgs);
        MethodHandle constant = MethodHandles.constant(e.getClass(), e);
        return constant;
    }

    private static MethodHandle compileDerivativeHandle(MathExpression.Token t) throws Throwable {
        String[] args = t.getRawArgs();
        if (args == null || args.length == 0) {
            throw new IllegalArgumentException("Method 'diff' requires arguments.");
        }

        String targetExpr = args[0];
        int order = 1;
        double evalPoint = -1;
        MathExpression.EvalResult solution = null;

        switch (args.length) {
            case 1:
                solution = Derivative.eval("diff(" + targetExpr + ",1)");
                break;
            case 2:
                if (com.github.gbenroscience.parser.Number.isNumber(args[1])) {
                    order = Integer.parseInt(args[1]);
                    solution = Derivative.eval("diff(" + targetExpr + "," + order + ")");
                } else {
                    solution = Derivative.eval("diff(" + targetExpr + "," + args[1] + ")");
                }
                break;
            case 3:
                order = Integer.parseInt(args[2]);
                if (com.github.gbenroscience.parser.Number.isNumber(args[1])) {
                    evalPoint = Double.parseDouble(args[1]);
                    solution = Derivative.eval("diff(" + targetExpr + "," + evalPoint + "," + order + ")");
                } else {
                    solution = Derivative.eval("diff(" + targetExpr + "," + args[1] + "," + order + ")");
                }
                break;
        }

        if (solution.getType() == TYPE.NUMBER) {
            return createConstantHandle(solution.scalar);
        } else if (solution.getType() == TYPE.STRING) {
            MethodHandle constant = MethodHandles.constant(MathExpression.EvalResult.class, solution);
            return constant;
        } else {
            throw new RuntimeException("Invalid expression passed to `diff` method: " + FunctionManager.lookUp(targetExpr));
        }
    }

    private MethodHandle compileIntegrationHandle(MathExpression.Token t) throws Throwable {
        String[] rawArgs = t.getRawArgs();
        Function f = FunctionManager.lookUp(rawArgs[0]);
        MathExpression innerExpr = f.getMathExpression();

        MethodHandle compiledInner = compileScalar(innerExpr.getCachedPostfix());

        double lower = Double.parseDouble(rawArgs[1]);
        double upper = Double.parseDouble(rawArgs[2]);
        int iterations = (rawArgs.length == 4) ? (int) Double.parseDouble(rawArgs[3]) : (int) ((upper - lower) / 0.05);

        String[] vars = innerExpr.getVariablesNames();
        int[] slots = innerExpr.getSlots();

        MethodHandle bridge = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeTurboIntegral",
                MethodType.methodType(double.class, Function.class, MethodHandle.class, double.class, double.class, int.class, String[].class, int[].class));

        return MethodHandles.insertArguments(bridge, 0, f, compiledInner, lower, upper, iterations, vars, slots);
    }

    private MethodHandle compileRootHandle(MathExpression.Token t) throws Throwable {
        String[] args = t.getRawArgs();
        String fNameOrExpr = args[0];
        Function f = Variable.isVariableString(fNameOrExpr) ? FunctionManager.lookUp(fNameOrExpr) : FunctionManager.add(fNameOrExpr);

        String varName = f.getIndependentVariables().get(0).getName();
        // Fallback default bounds widened so Brent's method doesn't miss the 2.06 Root
        double lower = args.length > 1 ? Double.parseDouble(args[1]) : -10.0;
        double upper = args.length > 2 ? Double.parseDouble(args[2]) : 10.0;
        int iterations = args.length > 3 ? Integer.parseInt(args[3]) : TurboRootFinder.DEFAULT_ITERATIONS;

        int xSlot = f.getMathExpression().getVariable(varName).getFrameIndex();
        MethodHandle targetHandle = compileScalar(f.getMathExpression().getCachedPostfix());

        MethodHandle derivHandle = null;
        try {
            // Use fNameOrExpr to avoid the internal NPE in Parser.getFunction()
            MathExpression.EvalResult diffRes = Derivative.eval("diff(" + fNameOrExpr + ",1)");

            if (diffRes != null && diffRes.textRes != null) {
                String derivString = diffRes.textRes;

                // If the derivative evaluator returned a registered function alias, 
                // we must compile its internal math expression, not the wrapper.
                Function df = FunctionManager.lookUp(derivString);
                if (df != null) {
                    derivHandle = compileScalar(df.getMathExpression().getCachedPostfix());
                } else {
                    derivHandle = compileScalar(new MathExpression(derivString, true).getCachedPostfix());
                }
            }
        } catch (Throwable e) {
            // Silently swallow any calculus errors. 
            // Passing null allows TurboRootFinder to seamlessly use Brent's/Secant method.
            derivHandle = null;
        }

        MethodHandle bridge = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeGeneralRoot",
                MethodType.methodType(double.class, MethodHandle.class, MethodHandle.class, int.class, double.class, double.class, int.class));

        return MethodHandles.insertArguments(bridge, 0, targetHandle, derivHandle, xSlot, lower, upper, iterations);
    }

    public static double executeGeneralRoot(MethodHandle targetHandle, MethodHandle derivHandle, int xSlot, double lower, double upper, int iterations) throws Throwable {
        return executeTurboRoot(targetHandle, derivHandle, xSlot, lower, upper, iterations);
    }

    private MethodHandle compileComplexFunction(MathExpression.Token t) throws Throwable {
        String name = t.name.toLowerCase();

        switch (name) {
            case Declarations.DIFFERENTIATION:
                return compileDerivativeHandle(t);
            case Declarations.INTEGRATION:
                return compileIntegrationHandle(t);
            case Declarations.GENERAL_ROOT:
                return compileRootHandle(t);
            case Declarations.ROTOR:
                return compileRotorHandle(t);
            case Declarations.PRINT:
                return compilePrintHandle(t);
            case Declarations.QUADRATIC:
            case Declarations.TARTAGLIA_ROOTS:
                String funcHandle = t.getRawArgs()[0];
                MethodHandle solver = name.equals(Declarations.QUADRATIC) ? QUAD_HANDLE : TARTAGLIA_HANDLE;
                MethodHandle bound = MethodHandles.insertArguments(solver, 0, funcHandle);
                return bound.asType(bound.type().changeReturnType(Object.class));
            default:
                if (Method.isPureStatsMethod(name)) {
                    return buildLegacyStatsHandle(t);
                }
                return buildLegacyRegistryHandle(t);
        }
    }

    private static Double getConstantValue(MethodHandle handle) {
        if (handle == null) {
            return null;
        }
        try {
            if (handle.type().parameterCount() == 0) {
                Object result = handle.invoke();
                if (result instanceof Number) {
                    return ((Number) result).doubleValue();
                }
            }
        } catch (Throwable ignored) {
        }
        return null;
    }

    public MethodHandle compileScalar(MathExpression.Token[] postfix) throws Throwable {
        if (postfix == null || postfix.length == 0) {
            return MethodHandles.constant(double.class, 0.0);
        }
        int varCount = countVariables(postfix);
        return compileScalarWide(postfix, varCount, true);
    }

    private static int countVariables(MathExpression.Token[] postfix) {
        int maxIndex = -1;
        for (MathExpression.Token t : postfix) {
            if (t.isVariable()) {
                maxIndex = Math.max(maxIndex, t.frameIndex);
            }
        }
        return maxIndex + 1;
    }

    public static double gradToRad(double grads) {
        return grads * (Math.PI / 200.0);
    }

    private static MethodHandle adaptToWideSignature(MethodHandle target, int arity, Deque<MethodHandle> stack, Class<?>[] wideTypes) throws Throwable {
        MethodHandle[] argFilters = new MethodHandle[arity];
        for (int i = arity - 1; i >= 0; i--) {
            argFilters[i] = stack.pop();
        }

        if (target.type().parameterCount() == 0) {
            MethodHandle result = wideTypes.length == 0 ? target : MethodHandles.dropArguments(target, 0, wideTypes);
            if (result.type().returnType() != Object.class) {
                result = result.asType(result.type().changeReturnType(Object.class));
            }
            return result;
        }

        if (wideTypes.length > 0 && target.type().parameterCount() == wideTypes.length && target.type().parameterType(0) == double.class) {
            if (target.type().returnType() != Object.class) {
                return target.asType(target.type().changeReturnType(Object.class));
            }
            return target;
        }

        if (target.type().parameterCount() == 1 && target.type().parameterType(0) == double[].class) {
            MethodHandle collector = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "collectToArray", MethodType.methodType(double[].class, double[].class))
                    .asCollector(double[].class, arity);
            target = MethodHandles.collectArguments(target, 0, collector);
        }

        MethodHandle widened = target;
        for (int i = arity - 1; i >= 0; i--) {
            if (argFilters[i].type().parameterCount() == 0 && wideTypes.length > 0) {
                argFilters[i] = liftConstant((double) argFilters[i].invokeExact(), wideTypes.length);
            }
            widened = MethodHandles.collectArguments(widened, i, argFilters[i]);
        }

        if (wideTypes.length > 0) {
            int varCount = wideTypes.length;
            int[] mergeMask = new int[arity * varCount];
            for (int i = 0; i < arity; i++) {
                for (int j = 0; j < varCount; j++) {
                    mergeMask[i * varCount + j] = j;
                }
            }
            widened = MethodHandles.permuteArguments(widened, MethodType.methodType(target.type().returnType(), wideTypes), mergeMask);
        }

        if (widened.type().returnType() != Object.class) {
            widened = widened.asType(widened.type().changeReturnType(Object.class));
        }
        return widened;
    }

    public static MethodHandle createConstantHandle(double value) {
        MethodHandle c = MethodHandles.constant(double.class, value);
        return MethodHandles.dropArguments(c, 0, double[].class);
    }

    private static void applyUnaryWide(MathExpression.Token t, Deque<MethodHandle> stack, boolean fold, int varCount) throws Throwable {
        MethodHandle input = stack.pop();
        String name = (t.kind == MathExpression.Token.OPERATOR) ? String.valueOf(t.opChar) : t.name;
        MethodHandle op = getUnaryHandle(name);

        // Constant Folding: Use invokeExact to eliminate signature check overhead
        if (fold && input.type().parameterCount() == 0) {
            double result = (double) op.invokeExact((double) input.invokeExact());
            stack.push(liftConstant(result, varCount));
            return;
        }

        if (input.type().parameterCount() == 0 && varCount > 0) {
            input = liftConstant((double) input.invokeExact(), varCount);
        }

        stack.push(MethodHandles.collectArguments(op, 0, input));
    }

    private static void applyBinaryWide(MathExpression.Token t, Deque<MethodHandle> stack,
            boolean fold, int varCount, Class<?>[] pTypes, int[] mask) throws Throwable {

        MethodHandle right = stack.pop();
        MethodHandle left = stack.pop();
        String name = (t.kind == MathExpression.Token.OPERATOR) ? String.valueOf(t.opChar) : t.name;
        MethodHandle op = getBinaryHandle(name);

        // Constant Folding Optimization
        if (fold && left.type().parameterCount() == 0 && right.type().parameterCount() == 0) {
            double valL = (double) left.invokeExact();
            double valR = (double) right.invokeExact();
            stack.push(liftConstant((double) op.invokeExact(valL, valR), varCount));
            return;
        }

        if (left.type().parameterCount() == 0 && varCount > 0) {
            left = liftConstant((double) left.invokeExact(), varCount);
        }
        if (right.type().parameterCount() == 0 && varCount > 0) {
            right = liftConstant((double) right.invokeExact(), varCount);
        }

        MethodHandle widened = op;
        widened = MethodHandles.collectArguments(widened, 1, right);
        widened = MethodHandles.collectArguments(widened, 0, left);

        if (varCount > 0) {
            // FIXED: Leverage the pre-allocated 'mask' parameter completely.
            // Eliminates redundant 'new int[]' heap allocation and loop churn per operator node.
            widened = MethodHandles.permuteArguments(widened, MethodType.methodType(double.class, pTypes), mask);
        }
        stack.push(widened);
    }

    static MathExpression.EvalResult executePrint(MathExpression.EvalResult ctx, String[] args) {

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

    private static MethodHandle liftVariable(int index, int total) {
        if (total == 0) {
            return MethodHandles.constant(double.class, 0.0);
        }

        //MethodHandle getter = MethodHandles.identity(double.class);
        MethodHandle getter = MethodHandlePolyfill.identity(double.class);
        if (index > 0) {
            Class<?>[] prefix = new Class<?>[index];
            Arrays.fill(prefix, double.class);
            getter = MethodHandles.dropArguments(getter, 0, prefix);
        }
        int suffixCount = total - index - 1;
        if (suffixCount > 0) {
            Class<?>[] suffix = new Class<?>[suffixCount];
            Arrays.fill(suffix, double.class);
            getter = MethodHandles.dropArguments(getter, index + 1, suffix);
        }
        return getter;
    }

    private static MethodHandle liftConstant(double val, int varCount) {
        MethodHandle c = MethodHandles.constant(double.class, val);
        if (varCount == 0) {
            return c;
        }
        Class<?>[] params = new Class<?>[varCount];
        Arrays.fill(params, double.class);
        return MethodHandles.dropArguments(c, 0, params);
    }

    private static MethodHandle buildLegacyRegistryHandle(MathExpression.Token t) throws Throwable {
        int methodId = MethodRegistry.getMethodID(t.name);
        MethodHandle bridge = LOOKUP.findStatic(ScalarTurboEvaluator2.class, "invokeRegistryMethod",
                MethodType.methodType(Object.class, int.class, double[].class));
        MethodHandle bound = MethodHandles.insertArguments(bridge, 0, methodId);
        return bound.asType(bound.type().changeReturnType(double.class));
    }

    public static double extractFirstScalar(Object result) {
        if (result instanceof Number) {
            return ((Number) result).doubleValue();
        }
        if (result instanceof double[]) {
            double[] arr = (double[]) result;
            return arr.length > 0 ? arr[0] : Double.NaN;
        }
        return Double.NaN;
    }

    public static double executeTurboIntegral(Function f, MethodHandle handle, double lower, double upper, int iterations, String[] vars, int[] slots) throws Throwable {
        MethodHandle primitiveHandle;
        int paramCount = handle.type().parameterCount();

        if (handle.type().returnType() == Object.class) {
            handle = MethodHandles.filterReturnValue(handle.asType(MethodType.methodType(Object.class, handle.type().parameterArray())),
                    LOOKUP.findStatic(ScalarTurboEvaluator2.class, "extractFirstScalar", MT_DOUBLE_OBJ)
            );
            paramCount = handle.type().parameterCount();
        }

        if (paramCount == 0) {
            primitiveHandle = MethodHandles.dropArguments(handle, 0, double[].class);
        } else if (paramCount == 1 && handle.type().parameterType(0) == double[].class) {
            primitiveHandle = handle;
        } else if (paramCount == 1 && handle.type().parameterType(0) == double.class) {
            MethodHandle getter = ScalarTurboEvaluator1.AndroidFriendlyMethodHandles.getDoubleArrayGetter();
            MethodHandle indexBinder = MethodHandles.insertArguments(getter, 1, 0);
            primitiveHandle = MethodHandles.collectArguments(handle, 0, indexBinder);
        } else {
            primitiveHandle = handle.asSpreader(double[].class, paramCount)
                    .asType(MethodType.methodType(double.class, double[].class));
        }

        boolean shouldSwap = lower > upper;
        if (shouldSwap) {
            NumericalIntegrator numericalIntegrator = new NumericalIntegrator(f, primitiveHandle, upper, lower, vars, slots);
            return numericalIntegrator.integrate(f);
        } else {
            NumericalIntegrator numericalIntegrator = new NumericalIntegrator(f, primitiveHandle, lower, upper, vars, slots);
            return numericalIntegrator.integrate(f);
        }
    }

    public static double executeTurboRoot(MethodHandle baseHandle, MethodHandle derivHandle,
            int xSlot, double lower, double upper, int iterations) {
        try {
            int baseParamCount = baseHandle.type().parameterCount();

            if (baseHandle.type().returnType() == Object.class) {
                baseHandle = MethodHandles.filterReturnValue(baseHandle.asType(MethodType.methodType(Object.class, baseHandle.type().parameterArray())),
                        LOOKUP.findStatic(ScalarTurboEvaluator2.class, "extractFirstScalar", MT_DOUBLE_OBJ)
                );
                baseParamCount = baseHandle.type().parameterCount();
            }

            MethodHandle wideBase;
            if (baseParamCount == 0) {
                wideBase = MethodHandles.dropArguments(baseHandle, 0, double[].class);
            } else if (baseParamCount == 1 && baseHandle.type().parameterType(0) == double[].class) {
                wideBase = baseHandle;
            } else if (baseParamCount == 1 && baseHandle.type().parameterType(0) == double.class) {
                MethodHandle getter = ScalarTurboEvaluator1.AndroidFriendlyMethodHandles.getDoubleArrayGetter();
                MethodHandle indexBinder = MethodHandles.insertArguments(getter, 1, xSlot);
                wideBase = MethodHandles.collectArguments(baseHandle, 0, indexBinder);
            } else {
                wideBase = baseHandle.asSpreader(double[].class, baseParamCount)
                        .asType(MethodType.methodType(double.class, double[].class));
            }

            MethodHandle wideDeriv = null;
            if (derivHandle != null) {
                int derivParamCount = derivHandle.type().parameterCount();

                if (derivHandle.type().returnType() == Object.class) {
                    derivHandle = MethodHandles.filterReturnValue(derivHandle.asType(MethodType.methodType(Object.class, derivHandle.type().parameterArray())),
                            LOOKUP.findStatic(ScalarTurboEvaluator2.class, "extractFirstScalar", MT_DOUBLE_OBJ)
                    );
                    derivParamCount = derivHandle.type().parameterCount();
                }

                if (derivParamCount == 0) {
                    wideDeriv = MethodHandles.dropArguments(derivHandle, 0, double[].class);
                } else if (derivParamCount == 1 && derivHandle.type().parameterType(0) == double[].class) {
                    wideDeriv = derivHandle;
                } else if (derivParamCount == 1 && derivHandle.type().parameterType(0) == double.class) {
                    MethodHandle getter = ScalarTurboEvaluator1.AndroidFriendlyMethodHandles.getDoubleArrayGetter();
                    MethodHandle indexBinder = MethodHandles.insertArguments(getter, 1, xSlot);
                    wideDeriv = MethodHandles.collectArguments(derivHandle, 0, indexBinder);
                } else {
                    wideDeriv = derivHandle.asSpreader(double[].class, derivParamCount)
                            .asType(MethodType.methodType(double.class, double[].class));
                }
            }

            TurboRootFinder trf = new TurboRootFinder(wideBase, wideDeriv, xSlot, lower, upper, iterations);
            return trf.findRoots();
        } catch (Throwable t) {
            throw new RuntimeException("Root finding failed: " + t.getMessage(), t);
        }
    }

    private static MethodHandle buildLegacyStatsHandle(MathExpression.Token t) throws Throwable {
        String name = t.name.toLowerCase();
        boolean isVector = name.equals(Declarations.MODE) || name.equals(Declarations.SORT);
        MethodHandle computation = isVector
                ? LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeVectorReturningStatsMethod", MethodType.methodType(double[].class, double[].class, String.class))
                : LOOKUP.findStatic(ScalarTurboEvaluator2.class, "executeScalarReturningStatsMethod", MethodType.methodType(double.class, double[].class, String.class));
        MethodHandle bound = MethodHandles.insertArguments(computation, 1, name);
        return bound.asType(bound.type().changeReturnType(Object.class));
    }

    private static double executeScalarReturningStatsMethod(double[] args, String method) {
        int n = args.length;
        if (n == 0 && !method.equals(Declarations.RANDOM)) {
            return Double.NaN;
        }

        switch (method) {
            case Declarations.LIST_SUM:
            case Declarations.SUM: {
                double total = 0.0;
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
            case Declarations.NOW:
                return System.currentTimeMillis();
            case Declarations.NANOS:
                return System.nanoTime();
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
                double mean = 0.0, M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                return M2 / (n - 1);
            }
            case Declarations.STD_DEV: {
                if (n < 2) {
                    return 0.0;
                }
                double mean = 0.0, M2 = 0.0;
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
                double mean = 0.0, M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                return Math.sqrt(M2 / (n - 1)) / Math.sqrt(n);
            }
            case Declarations.COEFFICIENT_OF_VARIATION: {
                if (n < 2) {
                    return Double.NaN;
                }
                double mean = 0.0, M2 = 0.0;
                for (int i = 0; i < n; i++) {
                    double delta = args[i] - mean;
                    mean += delta / (i + 1);
                    M2 += delta * (args[i] - mean);
                }
                if (mean == 0.0) {
                    return Double.NaN;
                }
                return Math.sqrt(M2 / (n - 1)) / mean;
            }
            case Declarations.ROOT_MEAN_SQUARED: {
                double sumSq = 0.0;
                for (int i = 0; i < n; i++) {
                    sumSq += (args[i] * args[i]);
                }
                return Math.sqrt(sumSq / n);
            }
            case Declarations.RANDOM:
                return ThreadLocalRandom.current().nextDouble();
            default:
                return Double.NaN;
        }
    }

    private static double[] executeVectorReturningStatsMethod(double[] args, String method) {
        int n = args.length;
        switch (method) {
            case Declarations.MODE: {
                Arrays.sort(args);
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

                double[] tempModes = new double[n];
                int modeIdx = 0;
                currentCount = 1;
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
                return Arrays.copyOf(tempModes, modeIdx);
            }
            case Declarations.SORT: {
                Arrays.sort(args);
                return args;
            }
            default:
                return null;
        }
    }

    public static double[] executeQuadraticRoot(String funcHandle) {
        Function f = FunctionManager.lookUp(funcHandle);
        String input = f.expressionForm();
        input = input.substring(1);
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
        }
        return new double[]{alg.solutions[0], alg.solutions[2]};
    }

    public static double[] executeTartagliaRoot(String funcHandle) {
        Function f = FunctionManager.lookUp(funcHandle);
        String input = f.expressionForm();
        input = input.substring(1);
        int closeBracOfAt = Bracket.getComplementIndex(true, 0, input);
        input = input.substring(closeBracOfAt + 1);
        if (input.startsWith("(")) {
            input = input.substring(1, input.length() - 1);
            input = input.concat("=0");
        }
        Tartaglia_Equation solver = new Tartaglia_Equation(input);
        solver.getAlgorithm().solve();
        return solver.getAlgorithm().solutions;
    }

    public static MethodHandle getOptimizedPowerHandle(double n) throws Throwable {
        if (n == 0.0) {
            return MethodHandles.constant(double.class, 1.0);
        }
        // Negative integer powers
        if (n == -1.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "reciprocal", MethodType.methodType(double.class, double.class));
        }
        if (n == -2.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "invSquare", MethodType.methodType(double.class, double.class));
        }
        if (n == -3.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "invCube", MethodType.methodType(double.class, double.class));
        }

        // Positive integer powers
        if (n == 1.0) {
            //return MethodHandles.identity(double.class);
            return MethodHandlePolyfill.identity(double.class);
        }
        if (n == 2.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "square", MethodType.methodType(double.class, double.class));
        }
        if (n == 3.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "cube", MethodType.methodType(double.class, double.class));
        }
        if (n == 4.0) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "quad", MethodType.methodType(double.class, double.class));
        }

        // Roots and Fractional Powers
        if (n == 0.5) {
            return LOOKUP.findStatic(Math.class, "sqrt", MethodType.methodType(double.class, double.class));
        }
        if (n == -0.5) {
            return LOOKUP.findStatic(ScalarTurboEvaluator2.class, "invSqrt", MethodType.methodType(double.class, double.class));
        }
        if (n == 0.25) {
            // x^0.25 -> sqrt(sqrt(x))
            MethodHandle sqrt = LOOKUP.findStatic(Math.class, "sqrt", MethodType.methodType(double.class, double.class));
            return MethodHandles.filterReturnValue(sqrt, sqrt);
        }
        if (Math.abs(n - (1.0 / 3.0)) < 1E-9) {
            return LOOKUP.findStatic(Math.class, "cbrt", MethodType.methodType(double.class, double.class));
        }

        // Fallback for everything else
        MethodHandle pow = LOOKUP.findStatic(Math.class, "pow", MethodType.methodType(double.class, double.class, double.class));
        return MethodHandles.insertArguments(pow, 1, n);
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

    public static double add(double a, double b) {
        return a + b;
    }

    public static double subtract(double a, double b) {
        return a - b;
    }

    public static double multiply(double a, double b) {
        return a * b;
    }

    public static double divide(double a, double b) {
        if (b == 0) {
            return a >= 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }
        return a / b;
    }

    public static double modulo(double a, double b) {
        return a % b;
    }

    public static double square(double x) {
        return x * x;
    }

    public static double cube(double x) {
        return x * x * x;
    }

    public static double quad(double x) {
        double x2 = x * x;
        return x2 * x2;
    }

    // The actual logic
    public static double reciprocal(double x) {
        return 1.0 / x;
    }

    public static double invSquare(double x) {
        return 1.0 / (x * x);
    }

    public static double invCube(double x) {
        return 1.0 / (x * x * x);
    }

    public static double invSqrt(double x) {
        return 1.0 / Math.sqrt(x);
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

    private MethodHandle optimizePower(MethodHandle base, double exp) {
        // 1. Handle Square: x^2 -> x * x
        if (exp == 2.0) {
            return MethodHandles.filterArguments(BINARY_MAP.get("*"), 0, base, base);
        }

        // 2. Handle Cube: x^3 -> (x * x) * x
        if (exp == 3.0) {
            MethodHandle square = MethodHandles.filterArguments(BINARY_MAP.get("*"), 0, base, base);
            return MethodHandles.filterArguments(BINARY_MAP.get("*"), 0, square, base);
        }

        // 3. Handle Reciprocal: x^-1 -> 1 / x
        if (exp == -1.0) {
            return MethodHandles.filterReturnValue(base, RECIPROCAL_HANDLE);
        }

        // 4. Handle Square Root: x^0.5 -> sqrt(x)
        if (exp == 0.5) {
            return MethodHandles.filterReturnValue(base, SQRT_HANDLE);
        }

        // 5. Handle Inverse Square: x^-2 -> 1 / (x * x)
        if (exp == -2.0) {
            MethodHandle square = MethodHandles.filterArguments(BINARY_MAP.get("*"), 0, base, base);
            return MethodHandles.filterReturnValue(square, RECIPROCAL_HANDLE);
        }

        // Fallback: Math.pow(base, constant_exponent)
        MethodHandle constExp = MethodHandles.constant(double.class, exp);
        return MethodHandles.filterArguments(BINARY_MAP.get("^"), 0, base, constExp);
    }

}
