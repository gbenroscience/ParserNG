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
package com.github.gbenroscience.math.numericalmethods;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpressionTreeDepth;

/**
 *
 * @author GBEMIRO
 */
public class ComplexityAnalyst {
    
    public enum Strategy { GAUSSIAN, CHEBYSHEV_FOREST, MACHINE }

    public static Strategy selectStrategy(MathExpression expr) {
        
        MathExpressionTreeDepth.Result r = expr.getTreeStats();
        int heavyOps = r.functions;
        boolean hasPotentialSingularities = r.divOperators > 0;

        // HEURISTIC RULES:
        // 1. If the tree is shallow and simple (e.g., polynomials), use Gaussian.
        // 2. If it contains trig/exp/log and is deep, use Chebyshev.
        // 3. If there is a division, Chebyshev Forest is mandatory for interval splitting.
        
        if (hasPotentialSingularities || r.depth > 8 || heavyOps > 5) {
            return Strategy.CHEBYSHEV_FOREST;
        }

        return Strategy.GAUSSIAN;
    }
}