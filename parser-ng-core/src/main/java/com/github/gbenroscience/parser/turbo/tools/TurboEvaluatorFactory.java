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

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.util.FunctionManager;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;

/**
 *
 * @author GBEMIRO
 */
public class TurboEvaluatorFactory {

    /**
     * Intelligently selects and returns the best Turbo engine for the
     * expression.
     * @param me The {@link MathExpression} 
     * @return A {@link TurboExpressionEvaluator} that can be used to evaluate expressions at turbo speed
     */
    public static TurboExpressionEvaluator getCompiler(MathExpression me) {
        MathExpression.Token[] postfix = me.getCachedPostfix();
        boolean involvesMatrices = false;

        // Scan tokens for Matrix indicators
        for (MathExpression.Token t : postfix) {
            if (isMatrixToken(t)) {
                involvesMatrices = true;
                break;
            }
        }
        return involvesMatrices ? new MatrixTurboEvaluator(me) : new ScalarTurboEvaluator(me);//defaults to an array based approach
    }
    /**
     * 
     * @param me The root MathExpression
     * @param useWideningVarsPassing If true, will create a turbo evaluator that uses widening to pass variables, 
     * else it uses an array based approach. If you are not sure, if your expression has more than 63 unique variables,
     * definitely set this variable to false. Else test with true or false for your particular expression and see which has higher performance.
     * They usually give performances within 1-20ns of each other.
     * @return A {@link TurboExpressionEvaluator} that can be used to evaluate expressions at turbo speed
     */
        public static TurboExpressionEvaluator getCompiler(MathExpression me, boolean useWideningVarsPassing) {
        MathExpression.Token[] postfix = me.getCachedPostfix(); 
        boolean involvesMatrices = false;

        // Scan tokens for Matrix indicators
        for (int i=0;i<postfix.length;i++) {
            MathExpression.Token t = postfix[i];
            if (isMatrixToken(t)) {
                involvesMatrices = true;
                break;
            }
        }
        return involvesMatrices ? new MatrixTurboEvaluator(me) : new ScalarTurboEvaluator(me, useWideningVarsPassing);//defaults to an array based approach
    }

    private static boolean isMatrixToken(MathExpression.Token t) {
        // 1. Check for Matrix-specific functions/methods
        if (t.kind == MathExpression.Token.FUNCTION || t.kind == MathExpression.Token.METHOD) {
            String name = t.name.toLowerCase();
            if (name.contains("matrix") || name.equals("det") || name.equals("inv") || name.equals("transpose") || name.equals(Declarations.LINEAR_SYSTEM)) {
                return true;
            }
        }
        // 2. Check if it's a known Matrix literal or constant
        Function f = FunctionManager.lookUp(t.name);
        return f != null && f.getType() == TYPE.MATRIX;
    }

    public static MethodHandle createMatrixLeaf(String symbol) {
        // Look up the Function object from the manager
        Function f = FunctionManager.lookUp(symbol);

        if (f != null && f.getMatrix() != null) {
            // Bind the specific Matrix object directly into the handle
            // This makes the matrix access O(1) during execution
            return MethodHandles.constant(Matrix.class, f.getMatrix());
        }
        return null;
    }
}
