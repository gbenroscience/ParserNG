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
import com.github.gbenroscience.util.FunctionManager;

import java.util.*;

public class ExpressionChecks {

    private static MathExpression runExpr(String expr, boolean strictMode) {

        System.out.println("Input - " + expr);
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
        System.out.println("fact-solve: " + m.solve());
        m = new MathExpression("\"hi\"", true, true);
        System.out.println("\"hi\": " + m.solve());
        
        
        m = new MathExpression("1,2", true, true);
        System.out.println("1,2: " + m.solve()+", scanner: "+m.getScanner());
        
        m = new MathExpression("1,2+3,9", true, true);
        System.out.println("1,2: " + m.solve()+", scanner: "+m.getScanner());
        
        m = new MathExpression("a;b;", true, true);
        System.out.println("a;b: " + m.solve());
        

        m = runExpr("4+tan(3)", true);
        System.out.println("tan : " + m.solve());
        m = runExpr("cos()", true);
        System.out.println("cos() : " + m.solve());
        runExpr("sec(x,y,z)=0", true);
        System.out.println("sec(x,y,z)=0 : " + FunctionManager.FUNCTIONS);
        runExpr("sin(x)=5+x/3", true);
        System.out.println("sin(x) : " + FunctionManager.FUNCTIONS);
        System.out.println("sin(x) : " + new MathExpression("sin(2)").solve());
        runExpr("cosh=@(x)0", true);
        System.out.println("cosh=@(x)0 : " + FunctionManager.FUNCTIONS);
        runExpr("cos()=(4+cos(2))", true);
      
        System.out.println("cos()=(4+cos(2)) : " + FunctionManager.FUNCTIONS);
        System.out.println("cos() : " + new MathExpression("cos()").solve());
        runExpr("2+", true);
        runExpr("4/0", true);
        runExpr("2+≠pi", true);
        runExpr("(((), 2+≠pi, 2+*3", true);
        MathExpression me = runExpr("hello world", true);
        runExpr("x", true);
        runExpr("(((), 2+≠pi, 2+*3", true);
        runExpr("0/0", true);
        me = runExpr("sqrt(-1)", false);
        System.out.println("scanner: " + me.getScanner());
        me = runExpr("foo(3)", true);
        System.out.println("scanner: " + me.getScanner());
        
        
        m = new MathExpression("j", true, true);
        System.out.println("j: " + m.solve());

    }

}
