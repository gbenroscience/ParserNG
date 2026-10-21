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

import com.github.gbenroscience.parser.CustomScanner;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.Scanner;
import java.util.*;


public class ExpressionChecks {

    private static MathExpression runExpr(String expr){
        
            System.out.println("Input - "+expr); 
         MathExpression me = null;
        try {
            me = new MathExpression(expr);
        } catch (InputMismatchException e) {
            e.printStackTrace();
        }
        return me;
    }
    public static void main(String[] args) {
        runExpr("2+");
        runExpr("4/0");
        runExpr("2+≠pi");
        MathExpression me = runExpr("hello world");
        System.out.println("scanner: "+me.getScanner());
        runExpr("(((), 2+≠pi, 2+*3");
        runExpr("0/0");
        me=runExpr("sqrt(-1)");
         System.out.println("scanner: "+me.getScanner());
        me = runExpr("foo(3)");
        System.out.println("scanner: "+me.getScanner());
        
 

    
    }
 
}
