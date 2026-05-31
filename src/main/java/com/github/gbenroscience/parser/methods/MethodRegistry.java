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

import com.github.gbenroscience.interfaces.Savable;
import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.math.differentialcalculus.Derivative;
import com.github.gbenroscience.math.geom.Direction;
import com.github.gbenroscience.math.geom.Line3D;
import com.github.gbenroscience.math.geom.Point;
import com.github.gbenroscience.math.geom.ROTOR;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.math.numericalmethods.NumericalIntegral;
import com.github.gbenroscience.math.numericalmethods.RootFinder;
import com.github.gbenroscience.math.quadratic.QuadraticSolver;
import com.github.gbenroscience.math.quadratic.Quadratic_Equation;
import com.github.gbenroscience.math.tartaglia.Tartaglia_Equation;
import com.github.gbenroscience.parser.Bracket;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.Variable;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.Utils;
import com.github.gbenroscience.util.io.ByteArrayBuilder;
import com.github.gbenroscience.util.io.TextFileWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;
import java.util.HashSet;

/**
 *
 * @author GBEMIRO
 */
public class MethodRegistry {

    public interface MethodAction extends Savable {

        static final long serialVersionUID = 1L;

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
        MathExpression.EvalResult calc(MathExpression.EvalResult nextResult, int arity, MathExpression.EvalResult[] args);
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
     *
     * @param id
     */
    public static MethodAction getAction(int id) {
        return actions[id];
    }

    public static final String[] expandedTrigAndHypMethodNames = new String[72];

    // --- Pre-register Built-ins ---
    static {
        loadInBuiltMethods();
    }

    private static void loadInBuiltMethods() {

        // Standard Trig
        String sinDeg = expandedTrigAndHypMethodNames[0] = Declarations.getTrigFuncDRGVariant(Declarations.SIN, DRG_MODE.DEG);
        String sinRad = expandedTrigAndHypMethodNames[1] = Declarations.getTrigFuncDRGVariant(Declarations.SIN, DRG_MODE.RAD);
        String sinGrad = expandedTrigAndHypMethodNames[2] = Declarations.getTrigFuncDRGVariant(Declarations.SIN, DRG_MODE.GRAD);
        String cosDeg = expandedTrigAndHypMethodNames[3] = Declarations.getTrigFuncDRGVariant(Declarations.COS, DRG_MODE.DEG);
        String cosRad = expandedTrigAndHypMethodNames[4] = Declarations.getTrigFuncDRGVariant(Declarations.COS, DRG_MODE.RAD);
        String cosGrad = expandedTrigAndHypMethodNames[5] = Declarations.getTrigFuncDRGVariant(Declarations.COS, DRG_MODE.GRAD);
        String tanDeg = expandedTrigAndHypMethodNames[6] = Declarations.getTrigFuncDRGVariant(Declarations.TAN, DRG_MODE.DEG);
        String tanRad = expandedTrigAndHypMethodNames[7] = Declarations.getTrigFuncDRGVariant(Declarations.TAN, DRG_MODE.RAD);
        String tanGrad = expandedTrigAndHypMethodNames[8] = Declarations.getTrigFuncDRGVariant(Declarations.TAN, DRG_MODE.GRAD);

// Inverse Trig (Standard)
        String asinDeg = expandedTrigAndHypMethodNames[9] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN, DRG_MODE.DEG);
        String asinRad = expandedTrigAndHypMethodNames[10] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN, DRG_MODE.RAD);
        String asinGrad = expandedTrigAndHypMethodNames[11] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN, DRG_MODE.GRAD);
        String acosDeg = expandedTrigAndHypMethodNames[12] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS, DRG_MODE.DEG);
        String acosRad = expandedTrigAndHypMethodNames[13] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS, DRG_MODE.RAD);
        String acosGrad = expandedTrigAndHypMethodNames[14] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS, DRG_MODE.GRAD);
        String atanDeg = expandedTrigAndHypMethodNames[15] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN, DRG_MODE.DEG);
        String atanRad = expandedTrigAndHypMethodNames[16] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN, DRG_MODE.RAD);
        String atanGrad = expandedTrigAndHypMethodNames[17] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN, DRG_MODE.GRAD);

// Inverse Trig (Alt)
        String asinAltDeg = expandedTrigAndHypMethodNames[18] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN_ALT, DRG_MODE.DEG);
        String asinAltRad = expandedTrigAndHypMethodNames[19] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN_ALT, DRG_MODE.RAD);
        String asinAltGrad = expandedTrigAndHypMethodNames[20] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SIN_ALT, DRG_MODE.GRAD);
        String acosAltDeg = expandedTrigAndHypMethodNames[21] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS_ALT, DRG_MODE.DEG);
        String acosAltRad = expandedTrigAndHypMethodNames[22] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS_ALT, DRG_MODE.RAD);
        String acosAltGrad = expandedTrigAndHypMethodNames[23] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COS_ALT, DRG_MODE.GRAD);
        String atanAltDeg = expandedTrigAndHypMethodNames[24] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN_ALT, DRG_MODE.DEG);
        String atanAltRad = expandedTrigAndHypMethodNames[25] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN_ALT, DRG_MODE.RAD);
        String atanAltGrad = expandedTrigAndHypMethodNames[26] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_TAN_ALT, DRG_MODE.GRAD);

// Reciprocal Trig
        String secDeg = expandedTrigAndHypMethodNames[27] = Declarations.getTrigFuncDRGVariant(Declarations.SEC, DRG_MODE.DEG);
        String secRad = expandedTrigAndHypMethodNames[28] = Declarations.getTrigFuncDRGVariant(Declarations.SEC, DRG_MODE.RAD);
        String secGrad = expandedTrigAndHypMethodNames[29] = Declarations.getTrigFuncDRGVariant(Declarations.SEC, DRG_MODE.GRAD);
        String cscDeg = expandedTrigAndHypMethodNames[30] = Declarations.getTrigFuncDRGVariant(Declarations.COSEC, DRG_MODE.DEG);
        String cscRad = expandedTrigAndHypMethodNames[31] = Declarations.getTrigFuncDRGVariant(Declarations.COSEC, DRG_MODE.RAD);
        String cscGrad = expandedTrigAndHypMethodNames[32] = Declarations.getTrigFuncDRGVariant(Declarations.COSEC, DRG_MODE.GRAD);
        String cotDeg = expandedTrigAndHypMethodNames[33] = Declarations.getTrigFuncDRGVariant(Declarations.COT, DRG_MODE.DEG);
        String cotRad = expandedTrigAndHypMethodNames[34] = Declarations.getTrigFuncDRGVariant(Declarations.COT, DRG_MODE.RAD);
        String cotGrad = expandedTrigAndHypMethodNames[35] = Declarations.getTrigFuncDRGVariant(Declarations.COT, DRG_MODE.GRAD);

// Inverse Reciprocal (Standard)
        String asecDeg = expandedTrigAndHypMethodNames[36] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.DEG);
        String asecRad = expandedTrigAndHypMethodNames[37] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.RAD);
        String asecGrad = expandedTrigAndHypMethodNames[38] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC, DRG_MODE.GRAD);
        String acscDeg = expandedTrigAndHypMethodNames[39] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.DEG);
        String acscRad = expandedTrigAndHypMethodNames[40] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.RAD);
        String acscGrad = expandedTrigAndHypMethodNames[41] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC, DRG_MODE.GRAD);
        String acotDeg = expandedTrigAndHypMethodNames[42] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.DEG);
        String acotRad = expandedTrigAndHypMethodNames[43] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.RAD);
        String acotGrad = expandedTrigAndHypMethodNames[44] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT, DRG_MODE.GRAD);

// Inverse Reciprocal (Alt)
        String asecAltDeg = expandedTrigAndHypMethodNames[45] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC_ALT, DRG_MODE.DEG);
        String asecAltRad = expandedTrigAndHypMethodNames[46] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC_ALT, DRG_MODE.RAD);
        String asecAltGrad = expandedTrigAndHypMethodNames[47] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_SEC_ALT, DRG_MODE.GRAD);
        String acscAltDeg = expandedTrigAndHypMethodNames[48] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC_ALT, DRG_MODE.DEG);
        String acscAltRad = expandedTrigAndHypMethodNames[49] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC_ALT, DRG_MODE.RAD);
        String acscAltGrad = expandedTrigAndHypMethodNames[50] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COSEC_ALT, DRG_MODE.GRAD);
        String acotAltDeg = expandedTrigAndHypMethodNames[51] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT_ALT, DRG_MODE.DEG);
        String acotAltRad = expandedTrigAndHypMethodNames[52] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT_ALT, DRG_MODE.RAD);
        String acotAltGrad = expandedTrigAndHypMethodNames[53] = Declarations.getTrigFuncDRGVariant(Declarations.ARC_COT_ALT, DRG_MODE.GRAD);

        expandedTrigAndHypMethodNames[54] = Declarations.SINH;
        expandedTrigAndHypMethodNames[55] = Declarations.COSH;
        expandedTrigAndHypMethodNames[56] = Declarations.TANH;
        expandedTrigAndHypMethodNames[57] = Declarations.SECH;
        expandedTrigAndHypMethodNames[58] = Declarations.COSECH;
        expandedTrigAndHypMethodNames[59] = Declarations.COTH;

        expandedTrigAndHypMethodNames[60] = Declarations.ARC_SINH;
        expandedTrigAndHypMethodNames[61] = Declarations.ARC_SINH_ALT;
        expandedTrigAndHypMethodNames[62] = Declarations.ARC_COSH;
        expandedTrigAndHypMethodNames[63] = Declarations.ARC_COSH_ALT;
        expandedTrigAndHypMethodNames[64] = Declarations.ARC_TANH;
        expandedTrigAndHypMethodNames[65] = Declarations.ARC_TANH_ALT;

        expandedTrigAndHypMethodNames[66] = Declarations.ARC_SECH;
        expandedTrigAndHypMethodNames[67] = Declarations.ARC_SECH_ALT;
        expandedTrigAndHypMethodNames[68] = Declarations.ARC_COSECH;
        expandedTrigAndHypMethodNames[69] = Declarations.ARC_COSECH_ALT;
        expandedTrigAndHypMethodNames[70] = Declarations.ARC_COTH;
        expandedTrigAndHypMethodNames[71] = Declarations.ARC_COTH_ALT;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        registerMethod(sinDeg, (ctx, arity, args) -> ctx.wrap(Maths.sinDegToRad(args[0].scalar)));
        registerMethod(sinRad, (ctx, arity, args) -> ctx.wrap(Math.sin(args[0].scalar)));
        //registerMethod(sinGrad, (ctx, arity, args) -> ctx.wrap(Maths.sinGradToRad(args[0].scalar)));
        registerMethod(sinGrad, (ctx, arity, args) -> ctx.wrap(Maths.sinGradToRad(args[0].scalar)));

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
        registerMethod(Declarations.LOG, (ctx, arity, args) -> {
            if (arity == 1) {
                return ctx.wrap(Maths.logToAnyBase(args[0].scalar, 10));
            } else {
                return ctx.wrap(Maths.logToAnyBase(args[0].scalar, args[1].scalar));
            }
        });
        registerMethod(Declarations.LOG_INV, (ctx, arity, args) -> {
            if (arity == 1) {
                return ctx.wrap(Maths.antiLogToAnyBase(args[0].scalar, 10));
            } else {
                return ctx.wrap(Maths.antiLogToAnyBase(args[0].scalar, args[1].scalar));
            }
        });
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

            int sz = args.length;
            switch (sz) {
                case 1: {
                    MathExpression.EvalResult solution = Derivative.eval("diff(" + args[0] + ",1)");//only the function handle was sent...e.g diff(F)
                    return ctx.wrap(solution);
                }
                case 2: {//diff(F,v|n) F = func to be differentiated, v = new func to hold return value of differentiation, n = order of differentiation
                    String anonFunc = args[0].textRes;
                    MathExpression.EvalResult solution = Derivative.eval("diff(" + anonFunc + "," + (args[1].textRes != null ? args[1].textRes : args[1].scalar) + ")");
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
                    MathExpression.EvalResult ev = Derivative.eval("diff(" + anonFunc + "," + (args[1].textRes != null ? args[1].textRes : args[1].scalar) + "," + args[2] + ")");
                    return ctx.wrap(ev);

                }
                default:
                    return ctx.wrap(Double.NaN);
            }
        });
        registerMethod(Declarations.INTEGRATION, (ctx, arity, args) -> {
            boolean hasIterations = args.length == 4;//[F, 2.0, 3.0, 10000] ---rotates any function in 3D including points and planes and generic functions
            boolean hasNoIterations = args.length == 3;//[F, 2.0, 3.0]
            if (hasNoIterations) {
                NumericalIntegral intg = new NumericalIntegral(args[1].scalar, args[2].scalar, 0, args[0].textRes);
                return ctx.wrap(intg.findHighRangeIntegral());
            }//end if
            else if (hasIterations) {
                NumericalIntegral intg = new NumericalIntegral(args[1].scalar, args[2].scalar, (int) args[3].scalar, args[0].textRes);
                return ctx.wrap(intg.findHighRangeIntegral());
            }//end else if
            return ctx.wrap(Double.NaN);
        });
        registerMethod(Declarations.DIFF_EQN, (ctx, arity, args) -> {
//            System.out.println("Derivatives Action");
//            System.out.println("funcName: " + funcName);
//            System.out.println("arity: " + arity);

            int sz = args.length;
            switch (sz) {
                case 1: {
                    MathExpression.EvalResult solution = Derivative.eval("diff(" + args[0] + ",1)");//only the function handle was sent...e.g diff(F)
                    return ctx.wrap(solution);
                }
                case 2: {//diff(F,v|n) F = func to be differentiated, v = new func to hold return value of differentiation, n = order of differentiation
                    String anonFunc = args[0].textRes;
                    MathExpression.EvalResult solution = Derivative.eval("diff(" + anonFunc + "," + (args[1].textRes != null ? args[1].textRes : args[1].scalar) + ")");
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
                    MathExpression.EvalResult ev = Derivative.eval("diff(" + anonFunc + "," + (args[1].textRes != null ? args[1].textRes : args[1].scalar) + "," + args[2] + ")");
                    return ctx.wrap(ev);

                }
                default:
                    return ctx.wrap(Double.NaN);
            }
        });

        registerMethod(Declarations.ROTOR, (ctx, arity, args) -> {

            int sz = args.length;
            if (args.length == 4) {//rot(F,a,O,D) function, angle, origin, direction vector
                //confirm the last 3 other args
                double angle = args[1].scalar;
                String anonFuncOrig = args[2].textRes;
                String anonFuncDir = args[3].textRes;
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

                Function f = FunctionManager.lookUp(args[0].textRes);
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
                double angle = args[2].scalar;
                String anonFuncOrig = args[3].textRes;
                String anonFuncDir = args[4].textRes;
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

                Function p1 = FunctionManager.lookUp(args[0].textRes);
                Function p2 = FunctionManager.lookUp(args[1].textRes);
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
                        Matrix out = new Matrix(new double[]{p11Rot.x, p11Rot.y, p11Rot.z, p22Rot.x, p22Rot.y, p22Rot.z}, 2, 3);//BREAKING CHANGE---CHANGED 2 point rotation to 2x3 Matrix output
                        return ctx.wrap(out);
                    } else {
                        return MathExpression.EvalResult.ERROR;
                    }
                }

            } else {
                return MathExpression.EvalResult.ERROR;
            }
            return MathExpression.EvalResult.ERROR;

        });

        registerMethod(Declarations.GENERAL_ROOT, (ctx, arity, args) -> {
            RootFinder rf;
            switch (args.length) {
                case 1:
                    rf = new RootFinder(FunctionManager.lookUp(args[0].textRes));
                    ctx.wrap(rf.findRoots());
                    break;
                case 2:
                    rf = new RootFinder(FunctionManager.lookUp(args[0].textRes), args[1].scalar);
                    ctx.wrap(rf.findRoots());
                    break;
                case 3:
                    rf = new RootFinder(FunctionManager.lookUp(args[0].textRes), args[1].scalar, args[2].scalar);
                    ctx.wrap(rf.findRoots());
                    break;
                case 4:
                    rf = new RootFinder(FunctionManager.lookUp(args[0].textRes), args[1].scalar, args[2].scalar, (int) args[3].scalar);
                    ctx.wrap(rf.findRoots());
                    break;

                default:
                    throw new AssertionError();
            }

            return ctx;
        });
        registerMethod(Declarations.PLOT, (ctx, arity, args) -> ctx.wrap(-1));

        registerMethod(Declarations.PRINT, (ctx, arity, args) -> {
            for (MathExpression.EvalResult arg : args) {
                switch (arg.type) {
                    case MathExpression.EvalResult.TYPE_STRING:
                        Function f = FunctionManager.lookUp(arg.textRes);
                        if (f != null) {
                            switch (f.getType()) {
                                case ALGEBRAIC_EXPRESSION:
                                    System.out.println(f.toString());
                                    ctx.wrap(f.toString());
                                    break;
                                case MATRIX:
                                    System.out.println(f.getName() + "=" + f.getMatrix().toString());
                                    ctx.wrap(f.getName() + "=" + f.getMatrix().toString());
                                    break;
                                default:
                                    System.out.println(f.toString());
                                    ctx.wrap(f.toString());
                                    break;
                            }
                        } else {
                            System.out.println(arg.textRes);
                            ctx.wrap(arg.textRes);
                        }
                        break;
                    case MathExpression.EvalResult.TYPE_ERROR:
                        ctx.wrap(arg.toString());
                        System.out.println(arg.toString());
                        break;
                    case MathExpression.EvalResult.TYPE_MATRIX:
                        ctx.wrap(arg.matrix.getName() + "=" + arg.matrix.toString());
                        arg.matrix.print();
                        break;
                    case MathExpression.EvalResult.TYPE_VECTOR:
                        ctx.wrap(Arrays.toString(arg.vector));
                        System.out.println(Arrays.toString(arg.vector));
                        break;
                    default:
                        System.out.println(arg.toString());
                        ctx.wrap(arg.toString());
                }

            }
            return ctx;
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
//System.out.println("listsum -> arity = "+arity+", args = "+Arrays.toString(args));
            ByteArrayBuilder bab = new ByteArrayBuilder();
            for (MathExpression.EvalResult e : args) {
                if (e.type == MathExpression.EvalResult.TYPE_SCALAR) {
                    bab.append(e.scalar);
                } else {
                    bab.append(e.vector);
                }
            }
            double[] data = bab.getAsDoubleArray();

            for (double d : data) {
                total += d;
            }
            // System.out.println("IN->"+Arrays.toString(args)+"   total: "+total);

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
        registerMethod(Declarations.NANOS, (ctx, arity, args) -> {
            ctx.wrap(System.nanoTime());
            return ctx;
        });
        registerMethod(Declarations.NOW, (ctx, arity, args) -> {
            ctx.wrap(System.nanoTime());
            return ctx;
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

            // System.out.println("sort -> arity = "+arity+", args = "+Arrays.toString(args));
            // 1. Handle edge cases
            if (args == null || args.length == 0) {
                MathExpression.EvalResult res = ctx;
                res.wrap(new double[0]);
                return res;
            }

            // 2. Clone the array to ensure the original dataset (args) remains 
            // unchanged for other parts of the expression evaluation.
            ByteArrayBuilder bab = new ByteArrayBuilder();
            for (MathExpression.EvalResult e : args) {
                if (e.type == MathExpression.EvalResult.TYPE_SCALAR) {
                    bab.append(e.scalar);
                } else {
                    bab.append(e.vector);
                }
            }
            double[] sortedData = bab.getAsDoubleArray();
            // 3. Perform the high-speed Dual-Pivot Quicksort 
            java.util.Arrays.sort(sortedData);

            //   System.out.println("IN->"+Arrays.toString(sortedData));
            ctx.wrap(sortedData);
            return ctx;
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
            if (alg.isComplex()) {
                return ctx.wrap(alg.solutions);
            } else {
                return ctx.wrap(new double[]{alg.solutions[0], alg.solutions[2]});
            }
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
            solver.solutions();
            return ctx.wrap(solver.getAlgorithm().solutions);
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
            if (args[0].type == MathExpression.EvalResult.TYPE_SCALAR && args[1].type == MathExpression.EvalResult.TYPE_SCALAR) {
                return ctx.wrap(args[0].scalar / args[1].scalar);
            } else if (args[0].type == MathExpression.EvalResult.TYPE_STRING && args[1].type == MathExpression.EvalResult.TYPE_SCALAR) {
                Function fA = FunctionManager.lookUp(args[0].textRes);
                Matrix mA = fA.getMatrix();
                return ctx.wrap(mA.scalarMultiply(1.0 / args[1].scalar));
            } else if (args[0].type == MathExpression.EvalResult.TYPE_SCALAR && args[1].type == MathExpression.EvalResult.TYPE_STRING) {
                Function fA = FunctionManager.lookUp(args[1].textRes);
                Matrix mA = fA.getMatrix();
                Matrix inv = mA.inverse();
                return ctx.wrap(inv.scalarMultiply(args[0].scalar));
            } else if (args[0].type == MathExpression.EvalResult.TYPE_STRING && args[1].type == MathExpression.EvalResult.TYPE_STRING) {
                Function fA = FunctionManager.lookUp(args[0].textRes);
                Function fB = FunctionManager.lookUp(args[1].textRes);
                Matrix mA = fA.getMatrix();
                Matrix mB = fB.getMatrix();
                return ctx.wrap(Matrix.multiply(mA, mB.inverse()));
            }
            return MathExpression.EvalResult.ERROR;
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
            // Wrap the 2n array into an n-row, 2-column Matrix
            Matrix e = new Matrix(evals, m.getRows(), 2);
            return ctx.wrap(e);

        });
        registerMethod(Declarations.MATRIX_EIGENVEC, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = f.getMatrix();
            Matrix eigenVec = m.getEigenVectorMatrix();
            return ctx.wrap(eigenVec);
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
            return ctx.wrap(Matrix.power(mA, (int) args[0].scalar));

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

        registerMethod(Declarations.SUB_MATRIX, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = null;
            if (f != null && (m = f.getMatrix()) != null) {
                int row = (int) args[1].scalar;
                int col = (int) args[2].scalar;

                if (row > 0 & col > 0) {
                    Matrix sm = Matrix.getSubmatrix(m, row, col);
                    return ctx.wrap(sm);
                } else {
                    return ctx.wrap(MathExpression.EvalResult.ERROR);
                }
            } else {
                return ctx.wrap(MathExpression.EvalResult.ERROR);
            }
        });
        registerMethod(Declarations.RANDOM_MATRIX, (ctx, arity, args) -> {
            int n = (int) args[0].scalar;
            int rowSz = (int) args[1].scalar;
            int colSz = (int) args[2].scalar;
            if (n > 0 && rowSz > 0 && colSz > 0) {
                Matrix sm = new Matrix(rowSz, colSz);
                sm.randomFill(n);
                return ctx.wrap(sm);
            }
            return ctx.wrap(MathExpression.EvalResult.ERROR);
        });

        registerMethod(Declarations.MATRIX_MINOR, (ctx, arity, args) -> {
            Function f = FunctionManager.lookUp(args[0].textRes);
            Matrix m = null;
            if (f != null && (m = f.getMatrix()) != null) {
                int row = (int) args[1].scalar;
                int col = (int) args[2].scalar;

                if (row > 0 & col > 0) {
                    Matrix min = m.minor(row, col);
                    return ctx.wrap(min);
                } else {
                    return ctx.wrap(MathExpression.EvalResult.ERROR);
                }
            } else {
                return ctx.wrap(MathExpression.EvalResult.ERROR);
            }
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

    public static void main(String[] args) {

        MathExpression me = new MathExpression("y=@(x)sin(x);rot(y, pi/2, @(1,3)(0,0,0),@(1,3)(0,0,1))");
        System.out.println("scanner = " + me.getScanner() + ",\n anon2 = " + FunctionManager.lookUp("anon2") + ",\n anon3 = " + FunctionManager.lookUp("anon3"));
        System.out.println("vector = " + me.solveGeneric());

        MathExpression m = new MathExpression("p=@(1,3)(4,2,5);q=@(1,3)(12,3,-1);rot(p,q, pi, @(1,3)(1,0,1),@(1,3)(1,1,0))");
        System.out.println("scanner = " + m.getScanner() + ",\n anon5 = " + FunctionManager.lookUp("anon5") + ",\n anon6 = " + FunctionManager.lookUp("anon6"));
        System.out.println("vector = " + m.solveGeneric());

        HashSet<String> data = new HashSet<>(Arrays.asList(expandedTrigAndHypMethodNames));
        StringBuilder sb = new StringBuilder();

        for (String s : methodIds.keySet()) {
            data.add(s);
        }
        int i = 0;
        for (String s : data) {
            if (i % 6 == 0) {
                sb.append("\"").append(s).append("\",\n");
            } else {
                sb.append("\"").append(s).append("\",");
            }
            i++;
        }

        TextFileWriter.writeText(new File(System.getProperty("user.home") + "/tokens.txt"), sb.toString());
        System.out.println("Saved at " + new File(System.getProperty("user.home") + "/tokens.txt").getAbsolutePath());
    }
}
