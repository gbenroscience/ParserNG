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
package com.github.gbenroscience.parser.benchmarks;


import com.github.gbenroscience.parser.MathExpression;
import java.util.*;


public class ExpressionChecks {

    private static MathExpression runExpr(String expr, boolean strictMode){
        
            System.out.println("Input - "+expr); 
         MathExpression me = null;
        try {
            me = new MathExpression(expr, true, strictMode);
        } catch (InputMismatchException e) {
            e.printStackTrace();
        }
        return me;
    }
    public static void main(String[] args) { 
        MathExpression m = new MathExpression("4!");
        System.out.println("fact-solve: "+m.solve());
        runExpr("2+", true);
        runExpr("4/0", true);
        runExpr("2+≠pi", true);
        runExpr("(((), 2+≠pi, 2+*3", true);
        MathExpression me = runExpr("hello world", true); 
          runExpr("x", true); 
        runExpr("(((), 2+≠pi, 2+*3", true);
        runExpr("0/0", true);
        me=runExpr("sqrt(-1)", false);
         System.out.println("scanner: "+me.getScanner());
        me = runExpr("foo(3)", true);
        System.out.println("scanner: "+me.getScanner());
        
 

    
    }
 
}
