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
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.MathExpression;

/**
 *
 * @author GBEMIRO
 * Compiled expression that returns any type of EvalResult (scalar, matrix, vector, etc.).
 * 
 * Performance characteristics:
 * - Scalar expressions: 5-10 ns
 * - Small matrices: 50-200 ns
 * - Large matrices: linear to problem size (excellent cache locality)
 */
public interface FastCompositeExpression extends Savable{
    
    
    
    public default MathExpression getRoot(){
      return null;
    };
    
    /**
     * Evaluate expression with given variable values.
     * Returns EvalResult to support all types (scalar, matrix, vector, etc.).
     * 
     * @param variables array of variable values indexed by registry slots
     * @return EvalResult of any type (scalar, matrix, vector, string, error)
     */
    MathExpression.EvalResult apply(double[] variables);
 
     /**
     * Convenience method for scalar extraction.
     * Throws ClassCastException if result is not scalar.
     * 
     * @param variables execution frame variables
     * @return scalar value
     */
     double applyScalar(double[] variables);
     
    /**
     * Convenience method for matrix extraction.
     * Throws ClassCastException if result is not a matrix.
     * 
     * @param variables execution frame variables
     * @return Matrix result
     */
    default Matrix applyMatrix(double[] variables) {
        MathExpression.EvalResult result = apply(variables);
        if (result.type != MathExpression.EvalResult.TYPE_MATRIX) {
            throw new ClassCastException(
                "Expected matrix but got: " + result.getTypeName()
            );
        }
        return result.matrix;
    }
    
    /**
     * Convenience method for vector extraction.
     * Throws ClassCastException if result is not a vector.
     * 
     * @param variables execution frame variables
     * @return double[] vector
     */
    default double[] applyVector(double[] variables) {
        MathExpression.EvalResult result = apply(variables);
        if (result.type != MathExpression.EvalResult.TYPE_VECTOR) {
            throw new ClassCastException(
                "Expected vector but got: " + result.getTypeName()
            );
        }
        return result.vector;
    }
   
       String checkErrorLogs();
       
       default TurboExpressionEvaluator getCompiler(){
          return null;
       }
    
    /**
     * Convenience method for string extraction.
     * Throws ClassCastException if result is not a string.
     * 
     * @param variables execution frame variables
     * @return String result
     */
    default String applyString(double[] variables) {
        MathExpression.EvalResult result = apply(variables);
        if (result.type != MathExpression.EvalResult.TYPE_STRING) {
            throw new ClassCastException(
                "Expected string but got: " + result.getTypeName()
            );
        }
        return result.textRes;
    }
}
