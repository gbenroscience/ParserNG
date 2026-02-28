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
package com.github.gbenroscience.parser.methods;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.math.differentialcalculus.Derivative;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.math.numericalmethods.NumericalIntegral;
import com.github.gbenroscience.math.numericalmethods.RootFinder;
import com.github.gbenroscience.math.quadratic.QuadraticSolver;
import com.github.gbenroscience.math.quadratic.Quadratic_Equation;
import com.github.gbenroscience.math.tartaglia.Tartaglia_Equation;
import com.github.gbenroscience.parser.Bracket;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.util.FunctionManager;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author GBEMIRO
 */
public class MethodRegistry {

    // Functional interface for method execution
    public interface MethodAction {

        /**
         * Allows methods like diff(fn,args) to differentiate an expression
         * using the MethodRegistry interface Is the intersection between
         * methods and user-defined functions
         *
         * @param ctx The {@link MathExpression}
         * @param funcName The name of the function
         * @param arity The number of argument to read from the args array
         * @param args The args passed to the function
         * @return
         */
        MathExpression.EvalResult execute(MathExpression ctx, String funcName, int arity, MathExpression.EvalResult[] args);
    }

    // 1. Map for compilation phase (String -> ID)
    private static final Map<String, Integer> methodIds = new HashMap<>();

    // 2. Array/List for evaluation phase (ID -> Action)
    // Using an ArrayList backed by an array for dynamic growth, 
    // but accessed via index for extreme speed.
    private static final List<MethodAction> actions = new ArrayList<>();

    private static int nextId = 0;

    // --- Core Methods ---
    // Called when a user (or the system) registers a new method
    public static void registerMethod(String name, MethodAction action) {
        if (!methodIds.containsKey(name)) {
            methodIds.put(name, nextId);
            actions.add(action); // Placed exactly at index 'nextId'
            nextId++;
        } else {
            // Overwrite existing method logic if re-registered
            int id = methodIds.get(name);
            actions.set(id, action);
        }
    }

    // Called during the translate() phase
    public static int getMethodID(String name) {
        Integer id = methodIds.get(name);
        return (id != null) ? id : -1; // -1 means method not found
    }

    // Called during the evaluate() hot loop
    public static MethodAction getAction(int id) {
        return actions.get(id); // O(1) array access
    }

    // --- Pre-register Built-ins ---
   static{
     loadInBuiltMethods();
   }
    private static void loadInBuiltMethods(){
        registerMethod(Declarations.SIN, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.sinDegToRad(args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Math.sin(args[0].scalar) : Maths.sinGradToRad(args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.COS, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.cosDegToRad(args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Math.cos(args[0].scalar) : Maths.cosGradToRad(args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.TAN, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.tanDegToRad(args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Math.tan(args[0].scalar) : Maths.tanGradToRad(args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.SEC, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(1.0 / getAction(getMethodID(Declarations.COS)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });
        registerMethod(Declarations.COSEC, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(1.0 / getAction(getMethodID(Declarations.SIN)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });
        registerMethod(Declarations.COT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(1.0 / getAction(getMethodID(Declarations.TAN)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });

        registerMethod(Declarations.ARC_SIN, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.asinRadToDeg(args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Math.asin(args[0].scalar) : Maths.asinRadToGrad(args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.ARC_SIN_ALT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(getAction(getMethodID(Declarations.ARC_SIN)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });
        registerMethod(Declarations.ARC_COS, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.acosRadToDeg(args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Math.acos(args[0].scalar) : Maths.acosRadToGrad(args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.ARC_COS_ALT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(getAction(getMethodID(Declarations.ARC_COS)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });
        registerMethod(Declarations.ARC_TAN, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.atanRadToDeg(args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Math.atan(args[0].scalar) : Maths.atanRadToGrad(args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.ARC_TAN_ALT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(getAction(getMethodID(Declarations.ARC_TAN)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });

        registerMethod(Declarations.ARC_SEC, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.acosRadToDeg(1 / args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Maths.acos(1 / args[0].scalar) : Maths.acosRadToGrad(1 / args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.ARC_SEC_ALT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(getAction(getMethodID(Declarations.ARC_SEC)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });
        registerMethod(Declarations.ARC_COSEC, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.asinRadToDeg(1 / args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Maths.asin(1 / args[0].scalar) : Maths.asinRadToGrad(1 / args[0].scalar)));
            return result;

        });
        registerMethod(Declarations.ARC_COSEC_ALT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(getAction(getMethodID(Declarations.ARC_COSEC)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });
        registerMethod(Declarations.ARC_COT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(ctx.getDRG() == DRG_MODE.DEG ? Maths.atanRadToDeg(1 / args[0].scalar) : (ctx.getDRG() == DRG_MODE.RAD ? Maths.atan(1 / args[0].scalar) : Maths.atanRadToGrad(1 / args[0].scalar)));
            return result;
        });
        registerMethod(Declarations.ARC_COT_ALT, (ctx, funcName, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(getAction(getMethodID(Declarations.ARC_COT)).execute(ctx, funcName, arity, args).scalar);
            return result;
        });

        registerMethod(Declarations.SINH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.sinh(args[0].scalar)));
        registerMethod(Declarations.COSH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.cosh(args[0].scalar)));
        registerMethod(Declarations.TANH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.tanh(args[0].scalar)));
        registerMethod(Declarations.SECH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(1.0 / Math.cosh(args[0].scalar)));
        registerMethod(Declarations.COSECH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(1.0 / Math.sinh(args[0].scalar)));
        registerMethod(Declarations.COTH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(1.0 / Math.tanh(args[0].scalar)));

        registerMethod(Declarations.ARC_SINH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.asinh(args[0].scalar)));
        registerMethod(Declarations.ARC_SINH_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.asinh(args[0].scalar)));
        registerMethod(Declarations.ARC_COSH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.acosh(args[0].scalar)));
        registerMethod(Declarations.ARC_COSH_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.acosh(args[0].scalar)));
        registerMethod(Declarations.ARC_TANH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.atanh(args[0].scalar)));
        registerMethod(Declarations.ARC_TANH_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.atanh(args[0].scalar)));

        registerMethod(Declarations.ARC_SECH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.asech(args[0].scalar)));
        registerMethod(Declarations.ARC_SECH_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.asech(args[0].scalar)));
        registerMethod(Declarations.ARC_COSECH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.acsch(args[0].scalar)));
        registerMethod(Declarations.ARC_COSECH_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.acsch(args[0].scalar)));
        registerMethod(Declarations.ARC_COTH, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.acoth(args[0].scalar)));
        registerMethod(Declarations.ARC_COTH_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.acoth(args[0].scalar)));

        registerMethod(Declarations.CUBE, (ctx, funcName, arity, args) -> {
            double x = args[0].scalar;
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(x * x * x);
            return result;
        });
        registerMethod(Declarations.SQUARE, (ctx, funcName, arity, args) -> {
            double x = args[0].scalar;
            MathExpression.EvalResult result = ctx.getNextResult();
            result.wrap(x * x);
            return result;
        });
        registerMethod(Declarations.CBRT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.cbrt(args[0].scalar)));
        registerMethod(Declarations.SQRT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.sqrt(args[0].scalar)));
        registerMethod(Declarations.POW, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.pow(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.EXP, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.exp(args[0].scalar)));

        registerMethod(Declarations.LG, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.log10(args[0].scalar)));
        registerMethod(Declarations.LG_INV, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.pow(10, args[0].scalar)));
        registerMethod(Declarations.LG_INV_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.pow(10, args[0].scalar)));
        registerMethod(Declarations.LOG, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.logToAnyBase(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.LOG_INV, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.antiLogToAnyBase(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.LOG_INV_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.antiLogToAnyBase(args[0].scalar, args[1].scalar)));

        registerMethod(Declarations.LN, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.log(args[0].scalar)));
        registerMethod(Declarations.LN_INV, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.exp(args[0].scalar)));
        registerMethod(Declarations.LN_INV_ALT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Math.exp(args[0].scalar)));
        registerMethod(Declarations.INVERSE, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(1.0 / args[0].scalar));

        registerMethod(Declarations.FACT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.fact(args[0].scalar)));
        registerMethod(Declarations.COMBINATION, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.combination(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.PERMUTATION, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(Maths.permutation(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.DIFFERENTIATION, (ctx, funcName, arity, args) -> {
            int sz = args.length;
            switch (sz) {
                case 1: {
                    String solution = Derivative.eval("diff(" + funcName + ",1)");
                    return ctx.getNextResult().wrap(com.github.gbenroscience.parser.Number.isNumber(solution) ? Double.parseDouble(solution) : Double.NaN);
                }
                case 2: {
                    String anonFunc = funcName;
                    double value = args[0].scalar;

                    String solution = Derivative.eval("diff(" + anonFunc + "," + value + ")");
                    return ctx.getNextResult().wrap(com.github.gbenroscience.parser.Number.isNumber(solution) ? Double.parseDouble(solution) : Double.NaN);
                }
                case 3: {
                    String anonFunc = funcName;
                    double value = args[1].scalar;
                    int order = (int) args[2].scalar;
                    /*  NumericalDerivative der = new NumericalDerivative(FunctionManager.lookUp(data.get(0)),Double.parseDouble(data.get(1)));
                return der.findDerivativeByPolynomialExpander();
                     */
                    String solution = Derivative.eval("diff(" + anonFunc + "," + value + "," + order + ")");
                    return ctx.getNextResult().wrap(com.github.gbenroscience.parser.Number.isNumber(solution) ? Double.parseDouble(solution) : Double.NaN);
                }
                default:
                    return ctx.getNextResult().wrap(Double.NaN);
            }
        });
        registerMethod(Declarations.INTEGRATION, (ctx, funcName, arity, args) -> {
            boolean has2NumberArguments = args.length == 2;
            boolean has3NumberArguments = args.length == 3;
            if (has2NumberArguments) {
                NumericalIntegral intg = new NumericalIntegral(args[0].scalar, args[1].scalar, 0, funcName);
                return ctx.getNextResult().wrap(intg.findHighRangeIntegral());
            }//end if
            else if (has3NumberArguments) {
                NumericalIntegral intg = new NumericalIntegral(args[0].scalar, args[1].scalar, (int) args[2].scalar, funcName);
                return ctx.getNextResult().wrap(intg.findHighRangeIntegral());
            }//end else if
            return ctx.getNextResult().wrap(Double.NaN);
        });
        registerMethod(Declarations.PLOT, (ctx, funcName, arity, args) -> ctx.getNextResult().wrap(-1));

        registerMethod(Declarations.PRINT, (ctx, funcName, arity, args) -> {
            for (MathExpression.EvalResult arg : args) {
                System.out.println(arg);
            }
            return ctx.getNextResult().wrap(-1);
        });
        /*registerMethod(Declarations.SUM, (ctx, funcName, arity, args) -> {
            double x = 0;
            for (MathExpression.EvalResult i : args) {
                x += i.scalar;
            }
            return ctx.getNextResult().wrap(x);
        });
         */
        registerMethod(Declarations.LIST_SUM, (ctx, name, arity, args) -> {
            MathExpression.EvalResult result = ctx.getNextResult();
            double total = 0.0;
           //    System.out.println("arg-type-in-registryp-call: "+args[0].type+", arg: "+args[0].toString());
            for (MathExpression.EvalResult arg : args) {
              //  System.out.println("arg-type-in-registryp-call: "+arg.type+", arg: "+arg.toString());
                if (arg.type == MathExpression.EvalResult.TYPE_SCALAR) {                    // scalar
                    total += arg.scalar;
                } else if (arg.type == MathExpression.EvalResult.TYPE_VECTOR && arg.vector != null) {   // list/vector from sort/mode
                    for (double v : arg.vector) {
                        total += v;
                    }
                }
            }
 
            result.wrap(total);
            return result;
        });
        registerMethod(Declarations.PROD, (ctx, funcName, arity, args) -> {
            double x = 1;
            for (MathExpression.EvalResult i : args) {
                x *= i.scalar;
            }
            return ctx.getNextResult().wrap(x);
        });
        registerMethod(Declarations.MEAN, (ctx, funcName, arity, args) -> {
            double n = args.length;
            return ctx.getNextResult().wrap(getAction(getMethodID(Declarations.SUM)).execute(ctx, funcName, arity, args).scalar / n);
        });
      //Registered under BasicNumeral
      //registerMethod(Declarations.AVG, (ctx, funcName, arity, args) -> getAction(getMethodID(Declarations.MEAN)).execute(ctx, funcName, arity, args));

        registerMethod(Declarations.MEDIAN, (ctx, funcName, arity, args) -> {
            if (args == null || args.length == 0) {
                return ctx.getNextResult().wrap(-1);
            }
            int n = args.length;

            if (n % 2 != 0) {
                // Odd length: Find the exact middle
                return ctx.getNextResult().wrap(quickSelect(args, 0, n - 1, n / 2));
            } else {
                // Even length: Average of the two middle elements
                // Note: After the first select, the array is partially partitioned,
                // making the second select significantly faster.
                double m1 = quickSelect(args, 0, n - 1, n / 2 - 1);
                double m2 = quickSelect(args, 0, n - 1, n / 2);
                return ctx.getNextResult().wrap((m1 + m2) / 2.0);
            }
        });
        registerMethod(Declarations.MODE, (ctx, funcName, arity, args) -> {
            if (arity == 0) {
                return ctx.getNextResult().wrap(new double[0]);
            }

            double[] vals = new double[arity];
            for (int i = 0; i < arity; i++) {
                vals[i] = args[i].scalar;
            }
            Arrays.sort(vals);

            // Find the highest frequency
            int maxFreq = 0, currFreq = 0;
            for (int i = 0; i < vals.length; i++) {
                currFreq = (i > 0 && vals[i] == vals[i - 1]) ? currFreq + 1 : 1;
                if (currFreq > maxFreq) {
                    maxFreq = currFreq;
                }
            }

            // Collect all values matching that frequency (supports multimodal)
            java.util.List<Double> modes = new java.util.ArrayList<>();
            currFreq = 0;
            for (int i = 0; i < vals.length; i++) {
                currFreq = (i > 0 && vals[i] == vals[i - 1]) ? currFreq + 1 : 1;
                if (currFreq == maxFreq) {
                    modes.add(vals[i]);
                }
            }

            double[] result = new double[modes.size()];
            for (int i = 0; i < modes.size(); i++) {
                result[i] = modes.get(i);
            }

            return ctx.getNextResult().wrap(result);
        });
        registerMethod(Declarations.RANGE, (ctx, funcName, arity, args) -> {
            // Handle empty or null datasets
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0);
                return res;
            }

            double min = args[0].scalar;
            double max = args[0].scalar;

            // Single pass O(n) - The fastest possible way to find range
            for (int i = 1; i < args.length; i++) {
                double val = args[i].scalar;
                if (val < min) {
                    min = val;
                } else if (val > max) {
                    max = val;
                }
            }

            // Standard Range calculation
            double range = max - min;

            // Wrap as a scalar and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(range);
            return res;
        });
        registerMethod(Declarations.MID_RANGE, (ctx, funcName, arity, args) -> {
            // Handle empty or null datasets
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0);
                return res;
            }

            double min = args[0].scalar;
            double max = args[0].scalar;

            // Single pass O(n) scan to find extrema
            for (int i = 1; i < args.length; i++) {
                double val = args[i].scalar;
                if (val < min) {
                    min = val;
                } else if (val > max) {
                    max = val;
                }
            }

            // Mid-range calculation: (Max + Min) / 2
            double midRange = (max + min) / 2.0;

            // Wrap as a scalar and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(midRange);
            return res;
        });
        registerMethod(Declarations.RANDOM, (ctx, funcName, arity, args) -> {
            double result;

            // Use ThreadLocalRandom for peak performance and thread safety
            java.util.concurrent.ThreadLocalRandom rng = java.util.concurrent.ThreadLocalRandom.current();

            if (args == null || args.length == 0) {
                // Case: random() -> [0.0, 1.0)
                result = rng.nextDouble();
            } else {
                // Case: random(n) -> [0.0, n)
                result = rng.nextDouble() * args[0].scalar;
            }

            // Wrap as a scalar and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(result);
            return res;
        });
        registerMethod(Declarations.STD_DEV, (ctx, funcName, arity, args) -> {
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0); // StdDev of 0 or 1 element is 0
                return res;
            }

            double mean = 0.0;
            double s = 0.0;
            int n = args.length;

            // Welford's Algorithm: Single pass, numerically stable
            for (int i = 0; i < n; i++) {
                double x = args[i].scalar;
                double oldMean = mean;
                mean += (x - oldMean) / (i + 1);
                s += (x - oldMean) * (x - mean);
            }

            // Variance = s / (n - 1)
            // StdDev = sqrt(Variance)
            double stdDev = Math.sqrt(s / (n - 1));

            // Wrap as a scalar and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(stdDev);
            return res;
        });
        registerMethod(Declarations.STD_ERR, (ctx, funcName, arity, args) -> {
            // SEM requires at least 2 data points for a sample deviation
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0);
                return res;
            }

            double mean = 0.0;
            double s = 0.0;
            int n = args.length;

            // Welford's Algorithm for numerical stability
            for (int i = 0; i < n; i++) {
                double x = args[i].scalar;
                double oldMean = mean;
                mean += (x - oldMean) / (i + 1);
                s += (x - oldMean) * (x - mean);
            }

            // Standard Deviation (Sample) = sqrt(s / (n - 1))
            // Standard Error = StdDev / sqrt(n)
            // Optimized as: sqrt(s / (n * (n - 1))) to reduce Math.sqrt calls
            double stdError = Math.sqrt(s / ((double) n * (n - 1)));

            // Wrap as a scalar and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(stdError);
            return res;
        });
        registerMethod(Declarations.VARIANCE, (ctx, funcName, arity, args) -> {
            // Variance requires at least 2 data points for a sample calculation
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0);
                return res;
            }

            double mean = 0.0;
            double s = 0.0;
            int n = args.length;

            // Welford's Algorithm: Single pass, numerically stable
            // Accumulates the sum of squares of differences from the mean
            for (int i = 0; i < n; i++) {
                double x = args[i].scalar;
                double oldMean = mean;
                mean += (x - oldMean) / (i + 1);
                s += (x - oldMean) * (x - mean);
            }

            // Sample Variance formula: s / (n - 1)
            double variance = s / (n - 1);

            // Wrap as a scalar and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(variance);
            return res;
        });

        registerMethod(Declarations.SORT, (ctx, funcName, arity, args) -> {
            // 1. Handle edge cases
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(new double[0]);
                return res;
            }

            // 2. Clone the array to ensure the original dataset (args) remains 
            // unchanged for other parts of the expression evaluation.
            MathExpression.EvalResult[] sortedData = args.clone();

            // 3. Perform the high-speed Dual-Pivot Quicksort 
            java.util.Arrays.sort(sortedData, (MathExpression.EvalResult o1, MathExpression.EvalResult o2) -> {
                // Assuming primary sorting by scalar (type 0) - adjust if needed for other types
                // If non-scalar, treat as equal or handle specifically (e.g., compare lengths for vectors)
                double v1 = (o1.type == 0) ? o1.scalar : 0.0;
                double v2 = (o2.type == 0) ? o2.scalar : 0.0;
                return Double.compare(v1, v2);  // Ascending order
            });
            double[] out = new double[sortedData.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = sortedData[i].scalar;
            }

            // 4. Wrap the result as a Vector (type 1) and return from the pool
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(out);
            return res;
        });

        registerMethod(Declarations.MIN, (ctx, funcName, arity, args) -> {
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(Double.NaN); // Or 0.0, depending on your parser's policy
                return res;
            }

            double min = args[0].scalar;
            // Start loop from 1 since index 0 is already the initial min
            for (int i = 1; i < args.length; i++) {
                if (args[i].scalar < min) {
                    min = args[i].scalar;
                }
            }

            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(min);
            return res;
        });

        registerMethod(Declarations.MAX, (ctx, funcName, arity, args) -> {
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(Double.NaN);
                return res;
            }

            double max = args[0].scalar;
            for (int i = 1; i < args.length; i++) {
                if (args[i].scalar > max) {
                    max = args[i].scalar;
                }
            }

            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(max);
            return res;
        });

        registerMethod(Declarations.ROOT_MEAN_SQUARED, (ctx, funcName, arity, args) -> {
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0);
                return res;
            }

            double sumSquares = 0.0;
            int n = args.length;

            // Standard O(n) pass
            for (MathExpression.EvalResult val : args) {
                sumSquares += val.scalar * val.scalar;
            }

            double rms = Math.sqrt(sumSquares / n);

            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(rms);
            return res;
        });
        registerMethod(Declarations.COEFFICIENT_OF_VARIATION, (ctx, funcName, arity, args) -> {
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(0.0);
                return res;
            }

            double mean = 0.0;
            double s = 0.0;
            int n = args.length;

            // Welford's Algorithm for stability
            for (int i = 0; i < n; i++) {
                double x = args[i].scalar;
                double oldMean = mean;
                mean += (x - oldMean) / (i + 1);
                s += (x - oldMean) * (x - mean);
            }

            // If mean is zero, CV is mathematically undefined (or Infinite)
            if (Math.abs(mean) < 1e-15) {
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(Double.NaN);
                return res;
            }

            double stdDev = Math.sqrt(s / (n - 1));
            double cv = stdDev / mean;

            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(cv);
            return res;
        });

        registerMethod(Declarations.GENERAL_ROOT, (ctx, funcName, arity, args) -> {
            RootFinder rf = new RootFinder(FunctionManager.lookUp(funcName), args[0].scalar);
            String root = rf.findRoots();
            MathExpression.EvalResult res = ctx.getNextResult();
            res.wrap(com.github.gbenroscience.parser.Number.isNumber(root) ? Double.parseDouble(root) : Double.NaN);
            return res;
        });
        registerMethod(Declarations.QUADRATIC, (ctx, funcName, arity, args) -> {
            QuadraticSolver qs = new QuadraticSolver(args[0].scalar, args[1].scalar, args[2].scalar);
            MathExpression.EvalResult res = ctx.getNextResult();
            if (qs.isComplex()) {
                // Return a vector [real1, imag1, real2, imag2]
                res.wrap(qs.solutions);
            } else {
                // Return a vector [root1, root2]
                res.wrap(new double[]{qs.solutions[0], qs.solutions[1]});
            }
            return res;
        });
        registerMethod(Declarations.QUADRATIC, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
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
            return ctx.getNextResult().wrap(alg.solutions);
        });
        registerMethod(Declarations.TARTAGLIA_ROOTS, (ctx, funcName, arity, args) -> {

            Function f = FunctionManager.lookUp(funcName);

            String input = f.expressionForm();
            input = input.substring(1);//remove the @
            int closeBracOfAt = Bracket.getComplementIndex(true, 0, input);
            input = input.substring(closeBracOfAt + 1);
            if (input.startsWith("(")) {
                input = input.substring(1, input.length() - 1);
                input = input.concat("=0");
            }

            Tartaglia_Equation solver = new Tartaglia_Equation(input);
            // Equation: c*x^3 + a*x + b = 0
            double c_coeff = solver.getAlgorithm().getA(); // c
            double a_coeff = solver.getAlgorithm().getB(); // a
            double b_coeff = solver.getAlgorithm().getC(); // b

            if (Math.abs(c_coeff) < 1e-12) {
                // Fallback to linear: ax + b = 0
                MathExpression.EvalResult res = ctx.getNextResult();
                res.wrap(-b_coeff / a_coeff);
                return res;
            }

            // Normalize to x^3 + px + q = 0
            double p = a_coeff / c_coeff;
            double q = b_coeff / c_coeff;

            // Cardano's Discriminant for depressed cubic
            double discriminant = (q * q / 4.0) + (p * p * p / 27.0);

            MathExpression.EvalResult res = ctx.getNextResult();

            if (discriminant > 0) {
                // 1 Real Root
                double sqrtD = Math.sqrt(discriminant);
                double u = Math.cbrt(-q / 2.0 + sqrtD);
                double v = Math.cbrt(-q / 2.0 - sqrtD);
                res.wrap(u + v); // Returns scalar
            } else if (discriminant == 0) {
                // Multiple Real Roots
                double u = Math.cbrt(-q / 2.0);
                res.wrap(new double[]{2 * u, -u}); // Returns vector
            } else {
                // 3 Distinct Real Roots (Trigonometric solution)
                double r = Math.sqrt(-(p * p * p) / 27.0);
                double phi = Math.acos(-q / (2.0 * r));
                double s = 2.0 * Math.pow(r, 1.0 / 3.0);

                double root1 = s * Math.cos(phi / 3.0);
                double root2 = s * Math.cos((phi + 2 * Math.PI) / 3.0);
                double root3 = s * Math.cos((phi + 4 * Math.PI) / 3.0);

                res.wrap(new double[]{root1, root2, root3}); // Returns vector
            }
            return res;
        });

        registerMethod(Declarations.HELP, (ctx, funcName, arity, args) -> {
            return ctx.getNextResult().wrap(Double.POSITIVE_INFINITY);
        });
        registerMethod(Declarations.DETERMINANT, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix();
            return ctx.getNextResult().wrap(m.determinant());
        });
        registerMethod(Declarations.LINEAR_SYSTEM, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().solveEquation();
            return ctx.getNextResult().wrap(m);
        });

        registerMethod(Declarations.MATRIX_ADD, (ctx, funcName, arity, args) -> {
            int commaIndex = funcName.indexOf(",");
            String matrixA = funcName.substring(0, commaIndex);
            String matrixB = funcName.substring(commaIndex + 1);
            Function fA = FunctionManager.lookUp(matrixA);
            Function fB = FunctionManager.lookUp(matrixB);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.getNextResult().wrap(mA.add(mB));
        });
        registerMethod(Declarations.MATRIX_ADJOINT, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().adjoint();
            return ctx.getNextResult().wrap(m);
        });
        registerMethod(Declarations.MATRIX_COFACTORS, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().getCofactorMatrix();
            return ctx.getNextResult().wrap(m);
        });
        registerMethod(Declarations.MATRIX_DIVIDE, (ctx, funcName, arity, args) -> {
            int commaIndex = funcName.indexOf(",");
            String matrixA = funcName.substring(0, commaIndex);
            String matrixB = funcName.substring(commaIndex + 1);
            Function fA = FunctionManager.lookUp(matrixA);
            Function fB = FunctionManager.lookUp(matrixB);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.getNextResult().wrap(Matrix.multiply(mA, mB.inverse()));
        });
        registerMethod(Declarations.MATRIX_EDIT, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix();
            int row = (int) args[0].scalar;
            int col = (int) args[1].scalar;
            double val = args[2].scalar;
            boolean updated = m.update(val, row, col);
            return ctx.getNextResult().wrap(m);
        });
        registerMethod(Declarations.MATRIX_EIGENPOLY, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            String v = f.getMatrix().getCharacteristicPolynomialForEigenVector();
            return ctx.getNextResult().wrap(v);
        });

        registerMethod(Declarations.MATRIX_EIGENVEC, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix();
            double eigenValues[] = m.computeEigenValues();
            int n = eigenValues.length;

// 2. Prepare a Matrix to hold all eigenvectors as columns
// Column 0 corresponds to lambda[0], Column 1 to lambda[1], etc.
            double[][] eigenvectorMatrix = new double[n][n];

            for (int i = 0; i < n; i++) {
                double lambda = eigenValues[i];
                double[] v = m.computeEigenVector(lambda);
                // Store v as a COLUMN in the result matrix
                for (int row = 0; row < n; row++) {
                    eigenvectorMatrix[row][i] = v[row];
                }
            }
            Matrix eigVectorMatrix = new Matrix(eigenvectorMatrix);
            return ctx.getNextResult().wrap(eigVectorMatrix);
        });
        registerMethod(Declarations.MATRIX_MULTIPLY, (ctx, funcName, arity, args) -> {
            int commaIndex = funcName.indexOf(",");
            String matrixA = funcName.substring(0, commaIndex);
            String matrixB = funcName.substring(commaIndex + 1);
            Function fA = FunctionManager.lookUp(matrixA);
            Function fB = FunctionManager.lookUp(matrixB);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.getNextResult().wrap(Matrix.multiply(mA, mB));
        });
        registerMethod(Declarations.MATRIX_POWER, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix mA = f.getMatrix();
            return ctx.getNextResult().wrap(Matrix.pow(mA, (int) args[0].scalar));

        });
        registerMethod(Declarations.MATRIX_SUBTRACT, (ctx, funcName, arity, args) -> {
            int commaIndex = funcName.indexOf(",");
            String matrixA = funcName.substring(0, commaIndex);
            String matrixB = funcName.substring(commaIndex + 1);
            Function fA = FunctionManager.lookUp(matrixA);
            Function fB = FunctionManager.lookUp(matrixB);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.getNextResult().wrap(mA.subtract(mB));
        });
        registerMethod(Declarations.MATRIX_TRANSPOSE, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().transpose();
            return ctx.getNextResult().wrap(m);
        });
        registerMethod(Declarations.INVERSE_MATRIX, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().inverse();
            return ctx.getNextResult().wrap(m);
        });
        registerMethod(Declarations.TRIANGULAR_MATRIX, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().reduceToTriangularMatrix();
            return ctx.getNextResult().wrap(m);
        });
        registerMethod(Declarations.ECHELON_MATRIX, (ctx, funcName, arity, args) -> {
            Function f = FunctionManager.lookUp(funcName);
            Matrix m = f.getMatrix().reduceToRowEchelonMatrix();
            return ctx.getNextResult().wrap(m);
        });

        // Users can call MethodRegistry.registerMethod("custom", (ctx, funcName, arity, args) -> ...) at runtime!
    
    }
    public static boolean isInBuiltMethod(String name) {
        return methodIds.containsKey(name);
    }

    private static double quickSelect(MathExpression.EvalResult[] arr, int left, int right, int k) {
        while (left < right) {
            if (right - left < 10) { // Optimization: Use Insertion Sort for tiny ranges
                insertionSort(arr, left, right);
                return arr[k].scalar;
            }
            int pivotIndex = partition(arr, left, right);
            if (k <= pivotIndex) {
                right = pivotIndex;
            } else {
                left = pivotIndex + 1;
            }
        }
        return arr[left].scalar;
    }

    private static int partition(MathExpression.EvalResult[] arr, int left, int right) {
        double pivot = arr[left + (right - left) / 2].scalar;
        int i = left - 1;
        int j = right + 1;
        while (true) {
            do {
                i++;
            } while (arr[i].scalar < pivot);
            do {
                j--;
            } while (arr[j].scalar > pivot);
            if (i >= j) {
                return j;
            }
            double temp = arr[i].scalar;
            arr[i].scalar = arr[j].scalar;
            arr[j].scalar = temp;
        }
    }

    private static void insertionSort(MathExpression.EvalResult[] arr, int left, int right) {
        for (int i = left + 1; i <= right; i++) {
            double val = arr[i].scalar;
            int j = i - 1;
            while (j >= left && arr[j].scalar > val) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1].scalar = val;
        }
    }
}
