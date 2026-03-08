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
import com.github.gbenroscience.util.Utils;
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

    public interface MethodAction {

        /**
         * Allows methods like diff(fn,args) to differentiate an expression
         * using the MethodRegistry interface Is the intersection between
         * methods and user-defined functions
         *
         * @param nextResult
         * @param arity The number of argument to read from the args array
         * @param args The args passed to the function
         * @return
         */
        MathExpression.EvalResult calc(MathExpression.EvalResult nextResult, int arity, MathExpression.EvalResult... args);
    }

    private static final Map<String, Integer> methodIds = new HashMap<>();

    // Using a raw array for the absolute fastest access
    private static MethodAction[] actions = new MethodAction[128];

    private static int nextId = 0;

    // --- Core Methods ---
    public static void registerMethod(String name, MethodAction action) {
        Integer existingId = methodIds.get(name);

        if (existingId == null) {
            int id = nextId++;
            ensureCapacity(id);
            methodIds.put(name, id);
            actions[id] = action;
        } else {
            // Overwrite existing method logic
            actions[existingId] = action;
        }
    }

    /**
     * Ensures the array is large enough to hold the given index. Uses a
     * standard growth factor of 1.5x to balance memory and performance.
     */
    private static void ensureCapacity(int index) {
        if (index >= actions.length) {
            int newSize = actions.length + (actions.length >> 1); // 1.5x growth
            if (newSize <= index) {
                newSize = index + 1;
            }
            actions = Arrays.copyOf(actions, newSize);
        }
    }

    // Called during the translate() phase
    public static int getMethodID(String name) {
        Integer id = methodIds.get(name);
        return (id != null) ? id : -1;
    }

    // --- THE HOT LOOP TARGET ---
    /**
     * This is now a direct array access. The JIT compiler can easily inline
     * this and potentially eliminate bounds checks if it recognizes the loop
     * patterns.
     * @param id 
     */
    public static MethodAction getAction(int id) {
        return actions[id];
    }

    // --- Pre-register Built-ins ---
    static {
        loadInBuiltMethods();
    }

    private static void loadInBuiltMethods() {

        String sinDeg = Declarations.getTrigFuncDRGVariant(Declarations.SIN, DRG_MODE.DEG);
        String sinRad = Declarations.getTrigFuncDRGVariant(Declarations.SIN, DRG_MODE.RAD);
        String sinGrad = Declarations.getTrigFuncDRGVariant(Declarations.SIN, DRG_MODE.GRAD);
        String cosDeg = Declarations.getTrigFuncDRGVariant(Declarations.COS, DRG_MODE.DEG);
        String cosRad = Declarations.getTrigFuncDRGVariant(Declarations.COS, DRG_MODE.RAD);
        String cosGrad = Declarations.getTrigFuncDRGVariant(Declarations.COS, DRG_MODE.GRAD);
        String tanDeg = Declarations.getTrigFuncDRGVariant(Declarations.TAN, DRG_MODE.DEG);
        String tanRad = Declarations.getTrigFuncDRGVariant(Declarations.TAN, DRG_MODE.RAD);
        String tanGrad = Declarations.getTrigFuncDRGVariant(Declarations.TAN, DRG_MODE.GRAD);

        String asinDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN, DRG_MODE.DEG);
        String asinRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN, DRG_MODE.RAD);
        String asinGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN, DRG_MODE.GRAD);
        String acosDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS, DRG_MODE.DEG);
        String acosRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS, DRG_MODE.RAD);
        String acosGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS, DRG_MODE.GRAD);
        String atanDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN, DRG_MODE.DEG);
        String atanRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN, DRG_MODE.RAD);
        String atanGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN, DRG_MODE.GRAD);

        String asinAltDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN_ALT, DRG_MODE.DEG);
        String asinAltRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN_ALT, DRG_MODE.RAD);
        String asinAltGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN_ALT, DRG_MODE.GRAD);
        String acosAltDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS_ALT, DRG_MODE.DEG);
        String acosAltRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS_ALT, DRG_MODE.RAD);
        String acosAltGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS_ALT, DRG_MODE.GRAD);
        String atanAltDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN_ALT, DRG_MODE.DEG);
        String atanAltRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN_ALT, DRG_MODE.RAD);
        String atanAltGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN_ALT, DRG_MODE.GRAD);

        String secDeg = Declarations.getTrigFuncDRGVariant(Declarations.SEC, DRG_MODE.DEG);
        String secRad = Declarations.getTrigFuncDRGVariant(Declarations.SEC, DRG_MODE.RAD);
        String secGrad = Declarations.getTrigFuncDRGVariant(Declarations.SEC, DRG_MODE.GRAD);
        String cscDeg = Declarations.getTrigFuncDRGVariant(Declarations.COSEC, DRG_MODE.DEG);
        String cscRad = Declarations.getTrigFuncDRGVariant(Declarations.COSEC, DRG_MODE.RAD);
        String cscGrad = Declarations.getTrigFuncDRGVariant(Declarations.COSEC, DRG_MODE.GRAD);
        String cotDeg = Declarations.getTrigFuncDRGVariant(Declarations.COT, DRG_MODE.DEG);
        String cotRad = Declarations.getTrigFuncDRGVariant(Declarations.COT, DRG_MODE.RAD);
        String cotGrad = Declarations.getTrigFuncDRGVariant(Declarations.COT, DRG_MODE.GRAD);

        String asecDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.DEG);
        String asecRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.RAD);
        String asecGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.GRAD);
        String acscDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.DEG);
        String acscRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.RAD);
        String acscGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.GRAD);
        String acotDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.DEG);
        String acotRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.RAD);
        String acotGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.GRAD);

        String asecAltDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.DEG);
        String asecAltRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.RAD);
        String asecAltGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.GRAD);
        String acscAltDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.DEG);
        String acscAltRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.RAD);
        String acscAltGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.GRAD);
        String acotAltDeg = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.DEG);
        String acotAltRad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.RAD);
        String acotAltGrad = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.GRAD);

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        registerMethod(sinDeg, (ctx, arity, args) -> ctx.wrap(Maths.sinDegToRad(args[0].scalar)));
        registerMethod(sinRad, (ctx, arity, args) -> ctx.wrap(Math.sin(args[0].scalar)));
        //registerMethod(sinGrad, (ctx, arity, args) -> ctx.wrap(Maths.sinGradToRad(args[0].scalar)));
        registerMethod(sinGrad, (ctx, arity, args) -> {
            return ctx.wrap(Maths.sinGradToRad(args[0].scalar));
        });

        registerMethod(cosDeg, (ctx, arity, args) -> ctx.wrap(Maths.cosDegToRad(args[0].scalar)));
        registerMethod(cosRad, (ctx, arity, args) -> ctx.wrap(Math.cos(args[0].scalar)));
        registerMethod(cosGrad, (ctx, arity, args) -> ctx.wrap(Maths.cosGradToRad(args[0].scalar)));

        registerMethod(tanDeg, (ctx, arity, args) -> ctx.wrap(Maths.tanDegToRad(args[0].scalar)));
        registerMethod(tanRad, (ctx, arity, args) -> ctx.wrap(Math.tan(args[0].scalar)));
        registerMethod(tanGrad, (ctx, arity, args) -> ctx.wrap(Maths.tanGradToRad(args[0].scalar)));

        registerMethod(asinDeg, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToDeg(args[0].scalar)));
        registerMethod(asinRad, (ctx, arity, args) -> ctx.wrap(Math.asin(args[0].scalar)));
        registerMethod(asinGrad, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToGrad(args[0].scalar)));

        registerMethod(asinAltDeg, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToDeg(args[0].scalar)));
        registerMethod(asinAltRad, (ctx, arity, args) -> ctx.wrap(Math.asin(args[0].scalar)));
        registerMethod(asinAltGrad, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToGrad(args[0].scalar)));

        registerMethod(acosDeg, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToDeg(args[0].scalar)));
        registerMethod(acosRad, (ctx, arity, args) -> ctx.wrap(Math.acos(args[0].scalar)));
        registerMethod(acosGrad, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToGrad(args[0].scalar)));

        registerMethod(acosAltDeg, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToDeg(args[0].scalar)));
        registerMethod(acosAltRad, (ctx, arity, args) -> ctx.wrap(Math.acos(args[0].scalar)));
        registerMethod(acosAltGrad, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToGrad(args[0].scalar)));

        registerMethod(atanDeg, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToDeg(args[0].scalar)));
        registerMethod(atanRad, (ctx, arity, args) -> ctx.wrap(Math.atan(args[0].scalar)));
        registerMethod(atanGrad, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToGrad(args[0].scalar)));

        registerMethod(atanAltDeg, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToDeg(args[0].scalar)));
        registerMethod(atanAltRad, (ctx, arity, args) -> ctx.wrap(Math.atan(args[0].scalar)));
        registerMethod(atanAltGrad, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToGrad(args[0].scalar)));

        registerMethod(secDeg, (ctx, arity, args) -> ctx.wrap(1.0 / Maths.cosDegToRad(args[0].scalar)));
        registerMethod(secRad, (ctx, arity, args) -> ctx.wrap(1.0 / Math.sin(args[0].scalar)));
        registerMethod(secGrad, (ctx, arity, args) -> ctx.wrap(1.0 / Maths.sinGradToRad(args[0].scalar)));

        registerMethod(cscDeg, (ctx, arity, args) -> ctx.wrap(1.0 / Maths.sinDegToRad(args[0].scalar)));
        registerMethod(cscRad, (ctx, arity, args) -> ctx.wrap(1.0 / Math.sin(args[0].scalar)));
        registerMethod(cscGrad, (ctx, arity, args) -> ctx.wrap(1.0 / Maths.sinGradToRad(args[0].scalar)));

        registerMethod(cotDeg, (ctx, arity, args) -> ctx.wrap(1.0 / Maths.tanDegToRad(args[0].scalar)));
        registerMethod(cotRad, (ctx, arity, args) -> ctx.wrap(1.0 / Math.tan(args[0].scalar)));
        registerMethod(cotGrad, (ctx, arity, args) -> ctx.wrap(1.0 / Maths.tanGradToRad(args[0].scalar)));

        registerMethod(asecDeg, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToDeg(1 / args[0].scalar)));
        registerMethod(asecRad, (ctx, arity, args) -> ctx.wrap(Maths.acos(1 / args[0].scalar)));
        registerMethod(asecGrad, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToGrad(1 / args[0].scalar)));

        registerMethod(asecAltDeg, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToDeg(1 / args[0].scalar)));
        registerMethod(asecAltRad, (ctx, arity, args) -> ctx.wrap(Maths.acos(1 / args[0].scalar)));
        registerMethod(asecAltGrad, (ctx, arity, args) -> ctx.wrap(Maths.acosRadToGrad(1 / args[0].scalar)));

        registerMethod(acscDeg, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToDeg(1 / args[0].scalar)));
        registerMethod(acscRad, (ctx, arity, args) -> ctx.wrap(Maths.asin(1 / args[0].scalar)));
        registerMethod(acscGrad, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToGrad(1 / args[0].scalar)));

        registerMethod(acscAltDeg, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToDeg(1 / args[0].scalar)));
        registerMethod(acscAltRad, (ctx, arity, args) -> ctx.wrap(Maths.asin(1 / args[0].scalar)));
        registerMethod(acscAltGrad, (ctx, arity, args) -> ctx.wrap(Maths.asinRadToGrad(1 / args[0].scalar)));

        registerMethod(acotDeg, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToDeg(1 / args[0].scalar)));
        registerMethod(acotRad, (ctx, arity, args) -> ctx.wrap(Maths.atan(1 / args[0].scalar)));
        registerMethod(acotGrad, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToGrad(1 / args[0].scalar)));

        registerMethod(acotAltDeg, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToDeg(1 / args[0].scalar)));
        registerMethod(acotAltRad, (ctx, arity, args) -> ctx.wrap(Maths.atan(1 / args[0].scalar)));
        registerMethod(acotAltGrad, (ctx, arity, args) -> ctx.wrap(Maths.atanRadToGrad(1 / args[0].scalar)));

        registerMethod(Declarations.SINH, (ctx, arity, args) -> ctx.wrap(Math.sinh(args[0].scalar)));
        registerMethod(Declarations.COSH, (ctx, arity, args) -> ctx.wrap(Math.cosh(args[0].scalar)));
        registerMethod(Declarations.TANH, (ctx, arity, args) -> ctx.wrap(Math.tanh(args[0].scalar)));
        registerMethod(Declarations.SECH, (ctx, arity, args) -> ctx.wrap(1.0 / Math.cosh(args[0].scalar)));
        registerMethod(Declarations.COSECH, (ctx, arity, args) -> ctx.wrap(1.0 / Math.sinh(args[0].scalar)));
        registerMethod(Declarations.COTH, (ctx, arity, args) -> ctx.wrap(1.0 / Math.tanh(args[0].scalar)));

        registerMethod(Declarations.ARC_SINH, (ctx, arity, args) -> ctx.wrap(Maths.asinh(args[0].scalar)));
        registerMethod(Declarations.ARC_SINH_ALT, (ctx, arity, args) -> ctx.wrap(Maths.asinh(args[0].scalar)));
        registerMethod(Declarations.ARC_COSH, (ctx, arity, args) -> ctx.wrap(Maths.acosh(args[0].scalar)));
        registerMethod(Declarations.ARC_COSH_ALT, (ctx, arity, args) -> ctx.wrap(Maths.acosh(args[0].scalar)));
        registerMethod(Declarations.ARC_TANH, (ctx, arity, args) -> ctx.wrap(Maths.atanh(args[0].scalar)));
        registerMethod(Declarations.ARC_TANH_ALT, (ctx, arity, args) -> ctx.wrap(Maths.atanh(args[0].scalar)));

        registerMethod(Declarations.ARC_SECH, (ctx, arity, args) -> ctx.wrap(Maths.asech(args[0].scalar)));
        registerMethod(Declarations.ARC_SECH_ALT, (ctx, arity, args) -> ctx.wrap(Maths.asech(args[0].scalar)));
        registerMethod(Declarations.ARC_COSECH, (ctx, arity, args) -> ctx.wrap(Maths.acsch(args[0].scalar)));
        registerMethod(Declarations.ARC_COSECH_ALT, (ctx, arity, args) -> ctx.wrap(Maths.acsch(args[0].scalar)));
        registerMethod(Declarations.ARC_COTH, (ctx, arity, args) -> ctx.wrap(Maths.acoth(args[0].scalar)));
        registerMethod(Declarations.ARC_COTH_ALT, (ctx, arity, args) -> ctx.wrap(Maths.acoth(args[0].scalar)));

        registerMethod(Declarations.CUBE, (ctx, arity, args) -> {
            double x = args[0].scalar;

            ctx.wrap(x * x * x);
            return ctx;
        });
        registerMethod(Declarations.SQUARE, (ctx, arity, args) -> {
            double x = args[0].scalar;

            ctx.wrap(x * x);
            return ctx;
        });
        registerMethod(Declarations.CBRT, (ctx, arity, args) -> ctx.wrap(Math.cbrt(args[0].scalar)));
        registerMethod(Declarations.SQRT, (ctx, arity, args) -> ctx.wrap(Math.sqrt(args[0].scalar)));
        registerMethod(Declarations.POW, (ctx, arity, args) -> ctx.wrap(Math.pow(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.EXP, (ctx, arity, args) -> ctx.wrap(Math.exp(args[0].scalar)));

        registerMethod(Declarations.LG, (ctx, arity, args) -> ctx.wrap(Math.log10(args[0].scalar)));
        registerMethod(Declarations.LG_INV, (ctx, arity, args) -> ctx.wrap(Math.pow(10, args[0].scalar)));
        registerMethod(Declarations.LG_INV_ALT, (ctx, arity, args) -> ctx.wrap(Math.pow(10, args[0].scalar)));
        registerMethod(Declarations.LOG, (ctx, arity, args) -> ctx.wrap(Maths.logToAnyBase(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.LOG_INV, (ctx, arity, args) -> ctx.wrap(Maths.antiLogToAnyBase(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.LOG_INV_ALT, (ctx, arity, args) -> ctx.wrap(Maths.antiLogToAnyBase(args[0].scalar, args[1].scalar)));

        registerMethod(Declarations.LN, (ctx, arity, args) -> ctx.wrap(Math.log(args[0].scalar)));
        registerMethod(Declarations.LN_INV, (ctx, arity, args) -> ctx.wrap(Math.exp(args[0].scalar)));
        registerMethod(Declarations.LN_INV_ALT, (ctx, arity, args) -> ctx.wrap(Math.exp(args[0].scalar)));
        registerMethod(Declarations.INVERSE, (ctx, arity, args) -> ctx.wrap(1.0 / args[0].scalar));

        registerMethod(Declarations.FACT, (ctx, arity, args) -> ctx.wrap(Maths.fact(args[0].scalar)));
        registerMethod(Declarations.COMBINATION, (ctx, arity, args) -> ctx.wrap(Maths.combination(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.PERMUTATION, (ctx, arity, args) -> ctx.wrap(Maths.permutation(args[0].scalar, args[1].scalar)));
        registerMethod(Declarations.DIFFERENTIATION, (ctx, arity, args) -> {
//            System.out.println("Derivatives Action");
//            System.out.println("funcName: " + funcName);
//            System.out.println("arity: " + arity);
//            System.out.println("args: " + Arrays.toString(args));
            int sz = args.length;
            switch (sz) {
                case 1: {
                    String solution = Derivative.eval("diff(" + args[0] + ",1)");//only the function handle was sent...e.g diff(F)
                    return ctx.wrap(solution);
                }
                case 2: {//diff(F,v|n) F = func to be differentiated, v = new func to hold return value of differentiation, n = order of differentiation
                    String anonFunc = args[0].textRes;
                    String solution = Derivative.eval("diff(" + anonFunc + "," + (args[1].textRes != null ? args[1].textRes : args[1].scalar) + ")");
                    return ctx.wrap(solution);
                }
                case 3: {
//diff(F,v|x, n) F = func to be differentiated, v = new func to hold return value of differentiation, 
//x = x point to evaluate final detivative at n = order of differentiation
                    String anonFunc = args[0].textRes;
                    int order = (int) args[2].scalar;
                    /*  NumericalDerivative der = new NumericalDerivative(FunctionManager.lookUp(data.get(0)),Double.parseDouble(data.get(1)));
                return der.findDerivativeByPolynomialExpander();
                     */
                    String solution = Derivative.eval("diff(" + anonFunc + "," + (args[1].textRes != null ? args[1].textRes : args[1].scalar) + "," + args[2] + ")");
                    if (com.github.gbenroscience.parser.Number.isNumber(solution)) {
                        return ctx.wrap(Double.parseDouble(solution));
                    } else {
                        return ctx.wrap(solution);
                    }
                }
                default:
                    return ctx.wrap(Double.NaN);
            }
        });
        registerMethod(Declarations.INTEGRATION, (ctx, arity, args) -> {
            boolean has2NumberArguments = args.length == 2;
            boolean has3NumberArguments = args.length == 3;
            if (has2NumberArguments) {
                NumericalIntegral intg = new NumericalIntegral(args[0].scalar, args[1].scalar, 0, args[2].textRes);
                return ctx.wrap(intg.findHighRangeIntegral());
            }//end if
            else if (has3NumberArguments) {
                NumericalIntegral intg = new NumericalIntegral(args[0].scalar, args[1].scalar, (int) args[2].scalar, args[3].textRes);
                return ctx.wrap(intg.findHighRangeIntegral());
            }//end else if
            return ctx.wrap(Double.NaN);
        });
        registerMethod(Declarations.PLOT, (ctx, arity, args) -> ctx.wrap(-1));

        registerMethod(Declarations.PRINT, (ctx, arity, args) -> {
            for (MathExpression.EvalResult arg : args) {
                System.out.println(arg);
            }
            return ctx.wrap(-1);
        });
        /*registerMethod(Declarations.SUM, (ctx, arity, args) -> {
            double x = 0;
            for (MathExpression.EvalResult i : args) {
                x += i.scalar;
            }
            return ctx.wrap(x);
        });
         */
        registerMethod(Declarations.LIST_SUM, (ctx, arity, args) -> {

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

            ctx.wrap(total);
            return ctx;
        });
        registerMethod(Declarations.PROD, (ctx, arity, args) -> {
            double x = 1;
            for (MathExpression.EvalResult i : args) {
                x *= i.scalar;
            }
            return ctx.wrap(x);
        });
        registerMethod(Declarations.MEAN, (ctx, arity, args) -> {
            double n = args.length;
            return ctx.wrap(getAction(getMethodID(Declarations.SUM)).calc(ctx, arity, args).scalar / n);
        });
        //Registered under BasicNumeral
        //registerMethod(Declarations.AVG, (ctx, arity, args) -> getAction(getMethodID(Declarations.MEAN)).calc(ctx, arity, args));

        registerMethod(Declarations.MEDIAN, (ctx, arity, args) -> {
            if (args == null || args.length == 0) {
                return ctx.wrap(-1);
            }
            int n = args.length;

            if (n % 2 != 0) {
                // Odd length: Find the exact middle
                return ctx.wrap(quickSelect(args, 0, n - 1, n / 2));
            } else {
                // Even length: Average of the two middle elements
                // Note: After the first select, the array is partially partitioned,
                // making the second select significantly faster.
                double m1 = quickSelect(args, 0, n - 1, n / 2 - 1);
                double m2 = quickSelect(args, 0, n - 1, n / 2);
                return ctx.wrap((m1 + m2) / 2.0);
            }
        });
        registerMethod(Declarations.MODE, (ctx, arity, args) -> {
            if (arity == 0) {
                return ctx.wrap(new double[0]);
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

            return ctx.wrap(result);
        });
        registerMethod(Declarations.RANGE, (ctx, arity, args) -> {
            // Handle empty or null datasets
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
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
            ctx.wrap(range);
            return ctx;
        });
        registerMethod(Declarations.MID_RANGE, (ctx, arity, args) -> {
            // Handle empty or null datasets
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
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
            MathExpression.EvalResult res = ctx;
            res.wrap(midRange);
            return res;
        });
        registerMethod(Declarations.RANDOM, (ctx, arity, args) -> {
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
            MathExpression.EvalResult res = ctx;
            res.wrap(result);
            return res;
        });
        registerMethod(Declarations.STD_DEV, (ctx, arity, args) -> {
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx;
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
            MathExpression.EvalResult res = ctx;
            res.wrap(stdDev);
            return res;
        });
        registerMethod(Declarations.STD_ERR, (ctx, arity, args) -> {
            // SEM requires at least 2 data points for a sample deviation
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx;
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
            MathExpression.EvalResult res = ctx;
            res.wrap(stdError);
            return res;
        });
        registerMethod(Declarations.VARIANCE, (ctx, arity, args) -> {
            // Variance requires at least 2 data points for a sample calculation
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx;
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
            MathExpression.EvalResult res = ctx;
            res.wrap(variance);
            return res;
        });

        registerMethod(Declarations.SORT, (ctx, arity, args) -> {
            // 1. Handle edge cases
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
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
            MathExpression.EvalResult res = ctx;
            res.wrap(out);
            return res;
        });

        registerMethod(Declarations.MIN, (ctx, arity, args) -> {
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
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

            MathExpression.EvalResult res = ctx;
            res.wrap(min);
            return res;
        });

        registerMethod(Declarations.MAX, (ctx, arity, args) -> {
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
                res.wrap(Double.NaN);
                return res;
            }

            double max = args[0].scalar;
            for (int i = 1; i < args.length; i++) {
                if (args[i].scalar > max) {
                    max = args[i].scalar;
                }
            }

            MathExpression.EvalResult res = ctx;
            res.wrap(max);
            return res;
        });

        registerMethod(Declarations.ROOT_MEAN_SQUARED, (ctx, arity, args) -> {
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
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

            MathExpression.EvalResult res = ctx;
            res.wrap(rms);
            return res;
        });
        registerMethod(Declarations.COEFFICIENT_OF_VARIATION, (ctx, arity, args) -> {
            if (args == null || args.length < 2) {
                MathExpression.EvalResult res = ctx;
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
                MathExpression.EvalResult res = ctx;
                res.wrap(Double.NaN);
                return res;
            }

            double stdDev = Math.sqrt(s / (n - 1));
            double cv = stdDev / mean;

            MathExpression.EvalResult res = ctx;
            res.wrap(cv);
            return res;
        });

        registerMethod(Declarations.GENERAL_ROOT, (ctx, arity, args) -> {
            RootFinder rf = new RootFinder(FunctionManager.lookUp(args[0].textRes), args[1].scalar);
            String root = rf.findRoots();
            MathExpression.EvalResult res = ctx;
            res.wrap(com.github.gbenroscience.parser.Number.isNumber(root) ? Double.parseDouble(root) : Double.NaN);
            return res;
        });
        registerMethod(Declarations.QUADRATIC, (ctx, arity, args) -> {
            QuadraticSolver qs = new QuadraticSolver(args[0].scalar, args[1].scalar, args[2].scalar);
            MathExpression.EvalResult res = ctx;
            if (qs.isComplex()) {
                // Return a vector [real1, imag1, real2, imag2]
                res.wrap(qs.solutions);
            } else {
                // Return a vector [root1, root2]
                res.wrap(new double[]{qs.solutions[0], qs.solutions[1]});
            }
            return res;
        });
        registerMethod(Declarations.QUADRATIC, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
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
            return ctx.wrap(alg.solutions);
        });
        registerMethod(Declarations.TARTAGLIA_ROOTS, (ctx, arity, args) -> {

            Function f = FunctionManager.lookUp(args[0].textRes);

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
                MathExpression.EvalResult res = ctx;
                res.wrap(-b_coeff / a_coeff);
                return res;
            }

            // Normalize to x^3 + px + q = 0
            double p = a_coeff / c_coeff;
            double q = b_coeff / c_coeff;

            // Cardano's Discriminant for depressed cubic
            double discriminant = (q * q / 4.0) + (p * p * p / 27.0);

            MathExpression.EvalResult res = ctx;

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

        registerMethod(Declarations.HELP, (ctx, arity, args) -> {
            return ctx.wrap(Double.POSITIVE_INFINITY);
        });
        registerMethod(Declarations.DETERMINANT, (ctx, arity, args) -> {

            if (args.length == 1 && args[0].textRes != null) {//det(1,2,-1,3,)
                Function f = FunctionManager.lookUp(args[0].textRes);
                Matrix m = f.getMatrix();
                return ctx.wrap(m.determinant());
            }
            //else --- //det(A) where A is a defined mXm+1 matrix
            double[] arr = new double[args.length];
            for (int i = 0; i < args.length; i++) {
                arr[i] = args[i].scalar;
            }
            boolean isPerfectSquare = Utils.isPerfectSquare(args.length);
            int sqrtLen = (int) Math.sqrt(args.length);
            if (isPerfectSquare) {
                int rows = sqrtLen;
                int cols = sqrtLen;
                Matrix m = new Matrix(arr, rows, cols);
                return ctx.wrap(m.determinant());
            }
            return MathExpression.EvalResult.ERROR;
        });
        registerMethod(Declarations.LINEAR_SYSTEM, (ctx, arity, args) -> {
            //System.out.println("LINEAR_SYSTEM: args-->>" + Arrays.deepToString(args) + ", args[0].type = " + args[0].getTypeName() + ",funcName: " + funcName);
            if (args.length == 1 && args[0].textRes != null) {//linear_sys(1,2,-1,3,4,9,-3,7)
                Function f = FunctionManager.lookUp(args[0].textRes);
                Matrix m = f.getMatrix().solveEquation();
                return ctx.wrap(m);
            }
            //else --- //linear_sys(A) where A is a defined mXm+1 matrix
            double[] arr = new double[args.length];
            for (int i = 0; i < args.length; i++) {
                arr[i] = args[i].scalar;
            }
            int rows = (int) ((int) (-1 + Math.sqrt(1 + 4 * arr.length)) / 2.0);
            int cols = rows + 1;

            Matrix m = new Matrix(arr, rows, cols);
            /* Orig   Soln-Matrix
               2x2       2X3
               3x3       3X4
              nxn       nx(n+1)-M ------n^2+n-M=0----  (-1+sqrt(1+4M))/2  AND (-1-sqrt(1+4M))/2 where M = args.length, the correct soln is the first,as rows and cols cant be < 0
             */

            Matrix n = m.solveEquation();
            return ctx.wrap(n);
        });

        registerMethod(Declarations.MATRIX_ADD, (ctx, arity, args) -> {
            Function fA = FunctionManager.lookUp(args[0].textRes);
            Function fB = FunctionManager.lookUp(args[1].textRes);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.wrap(mA.add(mB));
        });
        registerMethod(Declarations.MATRIX_ADJOINT, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix().adjoint();
            return ctx.wrap(m);
        });
        registerMethod(Declarations.MATRIX_COFACTORS, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix().getCofactorMatrix();
            return ctx.wrap(m);
        });
        registerMethod(Declarations.MATRIX_DIVIDE, (ctx, arity, args) -> {
            Function fA = FunctionManager.lookUp(args[0].textRes);
            Function fB = FunctionManager.lookUp(args[1].textRes);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.wrap(Matrix.multiply(mA, mB.inverse()));
        });
        registerMethod(Declarations.MATRIX_EDIT, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix();
            int row = (int) args[1].scalar;
            int col = (int) args[2].scalar;
            double val = args[3].scalar;
            boolean updated = m.update(val, row, col);
            return ctx.wrap(m);
        });
        registerMethod(Declarations.MATRIX_EIGENPOLY, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            String v = f.getMatrix().getCharacteristicPolynomialForEigenVector();
            return ctx.wrap(v);
        });
        registerMethod(Declarations.MATRIX_EIGENVALUES, (ctx, arity, args) -> {
            //System.out.println("eigValues branch: args-->>" + Arrays.deepToString(args) + ", args[0].type = " + args[0].getTypeName() + ",funcName: " + funcName);
            Matrix m = FunctionManager.lookUp(args[0].textRes).getMatrix();
            double[] evals = m.computeEigenValues();

            // Create a 1xN matrix
            Matrix result = new Matrix(1, evals.length);
            // Directly copy the array into the matrix's internal storage
            double array[] = new double[evals.length];
            System.arraycopy(evals, 0, array, 0, evals.length);
            result.setArray(array, 1, evals.length);
            return ctx.wrap(result);
        });
        registerMethod(Declarations.MATRIX_EIGENVEC, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
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
            return ctx.wrap(eigVectorMatrix);
        });
        registerMethod(Declarations.MATRIX_MULTIPLY, (ctx, arity, args) -> {
            Function fA = FunctionManager.lookUp(args[0].textRes);
            Function fB = FunctionManager.lookUp(args[1].textRes);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.wrap(Matrix.multiply(mA, mB));
        });
        registerMethod(Declarations.MATRIX_POWER, (ctx, arity, args) -> {
            Function fA = FunctionManager.lookUp(args[0].textRes);
            Matrix mA = fA.getMatrix();
            return ctx.wrap(Matrix.pow(mA, (int) args[0].scalar));

        });
        registerMethod(Declarations.MATRIX_SUBTRACT, (ctx, arity, args) -> {
            Function fA = FunctionManager.lookUp(args[0].textRes);
            Function fB = FunctionManager.lookUp(args[1].textRes);
            Matrix mA = fA.getMatrix();
            Matrix mB = fB.getMatrix();
            return ctx.wrap(mA.subtract(mB));
        });
        registerMethod(Declarations.MATRIX_TRANSPOSE, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix().transpose();
            return ctx.wrap(m);
        });
        registerMethod(Declarations.INVERSE_MATRIX, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix().inverse();
            return ctx.wrap(m);
        });
        registerMethod(Declarations.TRIANGULAR_MATRIX, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix().reduceToTriangularMatrix();
            return ctx.wrap(m);
        });
        registerMethod(Declarations.ECHELON_MATRIX, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix().reduceToRowEchelonMatrix();
            return ctx.wrap(m);
        });

        // Users can call MethodRegistry.registerMethod("custom", (ctx, arity, args) -> ...) at runtime!
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
