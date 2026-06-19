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
package com.github.gbenroscience.parser.turbo.examples;

import com.github.gbenroscience.parser.MathExpression;

/**
 *
 * @author GBEMIRO
 */
public class Test {
    
    public static void main(String[] args) {
            //String expr = "diff(@(x)cos(x)+sin(x),2,1)";
        String expr = "sort(12,1,23,5,listsum(13,2,20,30,40,-9,-23),1,1,1,2)"; 
        // Warm up JIT
        MathExpression interpreted = new MathExpression(expr, false);
        System.out.println(interpreted.solveGeneric());
    }
    
}
