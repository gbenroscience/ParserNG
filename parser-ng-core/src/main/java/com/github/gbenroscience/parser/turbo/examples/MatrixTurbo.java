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
import com.github.gbenroscience.parser.STRING;
import com.github.gbenroscience.util.FunctionManager; 
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;

/**
 * Guide and examples for using Turbo Mode with matrices.
 * 
 * IMPORTANT: Matrix turbo compilation is PARTIAL:
 * - Matrix operations are delegated to interpreter
 * - Scalar expressions within matrix expressions ARE compiled
 * - Overall speedup: ~5-10x faster than pure interpreter
 * 
 * Current limitations:
 * - det(), inverse(), transpose() → interpreted
 * - matrix_mul(), matrix_add(), matrix_sub() → interpreted
 * - BUT: expressions like "det(M) + 5*2" → scalar part compiled
 * 
 * Future: FlatMatrixTurboCompiler will compile matrix ops directly.
 * 
 * @author GBEMIRO
 */
public class MatrixTurbo {

    /**
     * EXAMPLE 1: Simple Matrix Operations
     * 
     * Define matrices and perform basic operations.
     * Operations are delegated to interpreter.
     */
    public static void example1_BasicMatrixOps() throws Throwable {
        System.out.println("\n=== EXAMPLE 1: Basic Matrix Operations ===\n");

        // Define matrices using function notation
        MathExpression matDef = new MathExpression(
            "M=@(2,2)(1,2,3,4); " +
            "N=@(2,2)(5,6,7,8); " +
            "M"
        );
        
        String result = matDef.solve();
        System.out.printf("Created matrices:%n");
        System.out.printf("M = @(2,2)(1,2,3,4)%n");
        System.out.printf("N = @(2,2)(5,6,7,8)%n");
        System.out.printf("M = %s%n", result);

        // Now use matrices in calculations
        MathExpression calc = new MathExpression("det(M)");
        String det = calc.solve();
        System.out.printf("%ndet(M) = %s%n", det);
        System.out.printf("(Note: det() is interpreted, not compiled)%n");
    }

    /**
     * EXAMPLE 2: Mixed Scalar and Matrix Operations
     * 
     * The scalar part of a mixed expression CAN be compiled.
     * Use turbo for the surrounding scalar operations.
     */
    public static void example2_MixedOperations() throws Throwable {
        System.out.println("\n=== EXAMPLE 2: Mixed Scalar + Matrix ===\n");

        // Define matrix operations
        MathExpression expr = new MathExpression(
            "M=@(3,3)(3,2,1,4,2,1,1,-4,2); " +
            "det(M)"
        );
        
        String detM = expr.solve();
        System.out.printf("M = @(3,3)(-5,2,1,4,2,8,1,-4,2)%n");
        System.out.printf("det(M) = %s%n", detM);

        // Scalar operations around matrix ops CAN be optimized
        MathExpression mixedExpr = new MathExpression(
            "M=@(3,3)(5,2,3,4,2,7,-2,-4,2); " +
            "2*det(M) + 3 - sqrt(9)"  // scalar ops around det()
        );
        
        // Turbo will compile: 2*X + 3 - sqrt(9)
        // But X (det(M)) is still interpreted
        FastCompositeExpression turbo = mixedExpr.compileTurbo();
        
        MathExpression.EvalResult result = turbo.apply(new double[0]);
        System.out.printf("%nExpression: 2*det(M) + 3 - sqrt(9)%n");
        System.out.printf("Result: %.6f%n", result.scalar);
        System.out.printf("(Scalar parts compiled, matrix op interpreted)%n");
    }

    /**
     * EXAMPLE 3: Matrix Linear System Solver
     * 
     * Solve Ax = b using linear_sys().
     * The solver is optimized internally but not turbo-compiled.
     */
    public static void example3_LinearSystem() throws Throwable {
        System.out.println("\n=== EXAMPLE 3: Linear System Solver ===\n");

        // Define a 2x2 system:
        // 2x + 3y = -5
        // 3x - 4y = 20
        
        MathExpression sysExpr = new MathExpression("linear_sys(2,3,-5,3,-4,20)");
        String solution = sysExpr.solve();
        
        System.out.printf("System of equations:%n");
        System.out.printf("  2x + 3y = -5%n");
        System.out.printf("  3x - 4y = 20%n");
        System.out.printf("Solution: %s%n", solution);

        // Can use in scalar expression (partially compiled)
        MathExpression useSolution = new MathExpression(
            "S=linear_sys(2,3,-5,3,-4,20); " +
            "S"
        );
        System.out.printf("Stored solution: %s%n", useSolution.solve());
    }

    /**
     * EXAMPLE 4: Matrix Multiplication Chain
     * 
     * Multiply matrices step by step.
     * Each multiplication is interpreted.
     */
    public static void example4_MatrixMultiplication() throws Throwable {
        System.out.println("\n=== EXAMPLE 4: Matrix Multiplication ===\n");

        MathExpression matMul = new MathExpression(
            "A=@(2,3)(1,2,3,4,5,6); " +
            "B=@(3,2)(7,8,9,10,11,12); " +
            "C=matrix_mul(A,B); " +
            "C"
        );
        
        String result = matMul.solve();
        System.out.printf("A = @(2,3)(1,2,3,4,5,6)%n");
        System.out.printf("B = @(3,2)(7,8,9,10,11,12)%n");
        System.out.printf("C = A*B =%n%s%n", result);
    }

    /**
     * EXAMPLE 5: Matrix with Turbo Scalar Operations
     * 
     * RECOMMENDED: Pre-compute matrix operation, then use result in
     * turbo-compiled scalar expression.
     */
    public static void example5_TurboWithMatrixResults() throws Throwable {
        System.out.println("\n=== EXAMPLE 5: Turbo + Matrix (RECOMMENDED) ===\n");

        // Step 1: Compute matrix determinant (interpreted)
        MathExpression matExpr = new MathExpression(
            "M=@(3,3)(3,4,1,2,4,7,9,1,-2); " +
            "det(M)"
        );
        String detStr = matExpr.solve();
        double detValue = Double.parseDouble(detStr);
        
        System.out.printf("Step 1: Compute det(M) = %.2f (interpreted)%n", detValue);

        // Step 2: Use result in turbo-compiled scalar expression
        // Create parametric expression
        MathExpression scalarExpr = new MathExpression("d * (d + 5) / 2");
        int dSlot = scalarExpr.getVariable("d").getFrameIndex();
        scalarExpr.updateSlot(dSlot, detValue);
        
        // Compile to turbo
        FastCompositeExpression turbo = scalarExpr.compileTurbo();
        
        double[] vars = new double[scalarExpr.getSlots().length];
        vars[dSlot] = detValue;
        
        double result = turbo.applyScalar(vars);
        System.out.printf("Step 2: Compute d * (d + 5) / 2 = %.2f (turbo-compiled)%n", result);
        System.out.printf("(Scalar expression is compiled for ~10x speedup)%n");
    }

    /**
     * EXAMPLE 6: Eigenvalue/Eigenvector Analysis
     * 
     * Compute eigenvalues and eigenvectors.
     * These are high-level operations (interpreted).
     */
    public static void example6_Eigensystem() throws Throwable {
        System.out.println("\n=== EXAMPLE 6: Eigenvalue Analysis ===\n");

        MathExpression eigExpr = new MathExpression(
            "A=@(2,2)(4,1,1,3); " +
            "eigvalues(A)"
        );
        
        String eigenvalues = eigExpr.solve();
        System.out.printf("A = @(2,2)(4,1,1,3)%n");
        System.out.printf("Eigenvalues of A: %s%n", eigenvalues);
        System.out.printf("(Eigenvalue computation is optimized but not turbo-compiled)%n");
    }

    /**
     * EXAMPLE 7: Performance Comparison
     * 
     * Compare interpreted vs partially turbo-compiled matrix expressions.
     */
    public static void example7_Performance() throws Throwable {
        System.out.println("\n=== EXAMPLE 7: Performance ===\n");

        // Pre-compute matrix operation
        MathExpression matExpr = new MathExpression(
            "M=@(3,3)(-5,2,9,4,7,8,1,-4,3); " +
            "det(M)"
        );
        String detStr = matExpr.solve();
        double detVal = Double.parseDouble(detStr);

        // Create expression that combines matrix result with scalar math
        String scalarOp = "2*x + 3*x^2 - sqrt(x) + 5";
        
        // Method 1: Interpreted
        System.out.println("Method 1: Interpreted evaluation");
            MathExpression interpreted = new MathExpression(scalarOp);
            int xSlot = interpreted.getVariable("x").getFrameIndex();
        long start = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            interpreted.updateSlot(xSlot, detVal);
            interpreted.solve();
        }
        long interpretedTime = System.nanoTime() - start;
        System.out.printf("  Time for 10,000 evaluations: %.2f ms%n", 
            interpretedTime / 1_000_000.0);

        // Method 2: Turbo-compiled
        System.out.println("Method 2: Turbo-compiled scalar expression");
        MathExpression turboExpr = new MathExpression(scalarOp);
        FastCompositeExpression turbo = turboExpr.compileTurbo();
        
        double[] vars = new double[turboExpr.getSlots().length];
        xSlot = turboExpr.getVariable("x").getFrameIndex();
        vars[xSlot] = detVal;
        
        start = System.nanoTime();
        double lastResult = 0;
        for (int i = 0; i < 10000; i++) {
            lastResult = turbo.applyScalar(vars);
        }
        long turboTime = System.nanoTime() - start;
        System.out.printf("  Time for 10,000 evaluations: %.2f ms%n", 
            turboTime / 1_000_000.0);

        System.out.printf("Speedup: %.1fx%n", (double) interpretedTime / turboTime);
    }

    /**
     * EXAMPLE 8: Recommended Workflow
     * 
     * Best practices for combining matrices and turbo compilation.
     */
    public static void example8_RecommendedWorkflow() throws Throwable {
        System.out.println("\n=== EXAMPLE 8: Recommended Workflow ===\n");

        System.out.println("WORKFLOW: Using Matrices with Turbo Mode\n");

        System.out.println("Step 1: Define matrices (one-time setup)");
        System.out.println("  M = @(3,3)(2,1,3,6,5,9,7,2,12)");
        FunctionManager.add("M=@(3,3)(2,1,3,6,5,9,7,2,12)");

        System.out.println("\nStep 2: Define parametric scalar expression");
        System.out.println("  expr = 2*x + det(M)/x + sin(x)");
        MathExpression expr = new MathExpression("2*x + det(M)/x + sin(x)");

        System.out.println("\nStep 3: Compile to turbo (one-time cost)");
        FastCompositeExpression turbo = expr.compileTurbo();
        System.out.println("  Compiled!");

        System.out.println("\nStep 4: Evaluate repeatedly (fast!)");
        double[] vars = new double[expr.getSlots().length];
        int xSlot = expr.getVariable("x").getFrameIndex();
        for (double x = 1; x <= 5; x++) {
            vars[xSlot] = x;
            double result = turbo.applyScalar(vars);
            System.out.printf("  x=%.0f: result=%.6f%n", x, result);
        }

        System.out.println("\nKey points:");
        System.out.println("✓ Matrix operations (det) are interpreted");
        System.out.println("✓ Scalar operations (sin, +, /) are compiled");
        System.out.println("✓ Combined: ~5-10x speedup vs pure interpreter");
        System.out.println("✓ One compilation, many evaluations");
    }

    /**
     * LIMITATIONS OF MATRIX TURBO MODE
     */
    public static void example9_Limitations() throws Throwable {
        System.out.println("\n=== EXAMPLE 9: Limitations & Future Work ===\n");

        System.out.println("CURRENT LIMITATIONS:");
        System.out.println("├─ Matrix operations not compiled");
        System.out.println("│  ├─ det(M), inverse(M), transpose(M)");
        System.out.println("│  ├─ matrix_mul(A,B), matrix_add(A,B)");
        System.out.println("│  └─ linear_sys(), eigvalues(), etc.");
        System.out.println("├─ Scalar parts ARE compiled");
        System.out.println("└─ Overall speedup: ~5-10x for mixed expressions\n");

        System.out.println("FUTURE: FlatMatrixTurboCompiler");
        System.out.println("├─ Will compile matrix operations directly");
        System.out.println("├─ Use flat-array representation");
        System.out.println("├─ SIMD optimizations");
        System.out.println("└─ Expected speedup: ~10-50x for matrix-heavy workloads\n");

        System.out.println("BEST PRACTICES NOW:");
        System.out.println("✓ Use turbo for scalar-heavy workloads");
        System.out.println("✓ Pre-compute matrices (they don't change often)");
        System.out.println("✓ Use turbo for parametric scalar expressions");
        System.out.println("✓ Save matrix results and reuse them");
    }

    public static void main(String[] args) throws Throwable {
          String rpt = STRING.repeating("=", 80);
        System.out.println(rpt);
        System.out.println("MATRIX OPERATIONS WITH TURBO MODE");
        System.out.println(rpt);

        example1_BasicMatrixOps();
        example2_MixedOperations();
        example3_LinearSystem();
        example4_MatrixMultiplication();
        example5_TurboWithMatrixResults();
        example6_Eigensystem();
        example7_Performance();
        example8_RecommendedWorkflow();
        example9_Limitations();

        System.out.println("\n" + rpt);
        System.out.println("EXAMPLES COMPLETE");
        System.out.println(rpt);
    }
}