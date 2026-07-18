/*
 * Copyright 2026 oluwagbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.numericalmethods;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.MathExpression.Token;

/**
 * Enhanced GC-Free Symbolic-Assisted Interior Singularity Extractor.
 * 
 * JDK 8 compatible version with broader singularity coverage
 * for robust numerical integration.
 */
public class SingularityExtractor {

    private static final int MAX_EXTRACTED_NODES_LIMIT = 64;
    private static final int MAX_ASYMPTOTE_TRACKING_LIMIT = 128;

    private final int[] criticalOpIndices;
    private final int[] argStartIndices;
    private final int[] argEndIndices;
    private int criticalNodesCount;

    private final double[] localEvalBuffer;
    private final int[] parsingArityStack;

    public SingularityExtractor(Token[] rpnTokens) {
        if (rpnTokens == null || rpnTokens.length == 0) {
            throw new IllegalArgumentException("RPN tokens array cannot be empty.");
        }

        this.criticalOpIndices = new int[MAX_EXTRACTED_NODES_LIMIT];
        this.argStartIndices = new int[MAX_EXTRACTED_NODES_LIMIT];
        this.argEndIndices = new int[MAX_EXTRACTED_NODES_LIMIT];
        this.criticalNodesCount = 0;
        this.localEvalBuffer = new double[8];

        int n = rpnTokens.length;
        this.parsingArityStack = new int[n + 4];

        int stackPointer = 0;

        // Allocation-Free AST Tree-Walking Simulation Pass
        for (int i = 0; i < n; i++) {
            Token t = rpnTokens[i];
            boolean isOperand = t.kind == Token.NUMBER ||
                                t.kind == Token.VARIABLE ||
                                t.kind == Token.MATRIX;

            if (isOperand) {
                setIntVal(parsingArityStack, stackPointer++, i);
            } else {
                int arity = t.arity;
                int argumentStartIndex = i;

                if (stackPointer >= arity) {
                    stackPointer -= arity;
                    argumentStartIndex = getIntVal(parsingArityStack, stackPointer);
                }

                boolean isTarget = isSingularityCausingToken(t);
                boolean fitLimit = criticalNodesCount < MAX_ASYMPTOTE_TRACKING_LIMIT;

                if (isTarget && fitLimit) {
                    setVal(criticalOpIndices, criticalNodesCount, i);
                    setVal(argStartIndices, criticalNodesCount, argumentStartIndex);
                    setVal(argEndIndices, criticalNodesCount, i - 1);
                    criticalNodesCount++;
                }

                setIntVal(parsingArityStack, stackPointer++, argumentStartIndex);
            }
        }
    }

    private boolean isSingularityCausingToken(Token t) {
        if (t.kind == Token.OPERATOR) {
            return t.opChar == '/' || t.opChar == '^';
        }

        if (t.kind == Token.FUNCTION || t.kind == Token.METHOD) {
            String name = t.name.toLowerCase();
            if ("ln".equals(name) || "log".equals(name) || "log10".equals(name) ||
                "sqrt".equals(name) ||
                "tan".equals(name) || "sec".equals(name) ||
                "csc".equals(name) || "cot".equals(name) ||
                "asin".equals(name) || "acos".equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static int getIntVal(int[] array, int index) {
        return java.lang.reflect.Array.getInt(array, index);
    }

    private static void setVal(int[] array, int index, int value) {
        java.lang.reflect.Array.setInt(array, index, value);
    }

    private static void setIntVal(int[] array, int index, int value) {
        java.lang.reflect.Array.setInt(array, index, value);
    }

    private static double getVal(double[] array, int index) {
        return java.lang.reflect.Array.getDouble(array, index);
    }

    private static void setDoubleVal(double[] array, int index, double value) {
        java.lang.reflect.Array.setDouble(array, index, value);
    }

  
    /**  
     * Scans for singularities within [a, b] and returns the count.
     * Singularities are stored in sorted order in resultBufferOut.
     * @param originalRpn
     * @param hostExpr
     * @param xSlotId
     * @param a
     * @param b
     * @param resultBufferOut
     * @return 
     */
    public int findSingularities(
            Token[] originalRpn,
            MathExpression hostExpr,
            int xSlotId,
            double a,
            double b,
            double[] resultBufferOut
    ) {
        if (resultBufferOut == null || resultBufferOut.length == 0) {
            return 0;
        }

        int entriesFoundCount = 0;
        int maxOutputsCapacity = resultBufferOut.length;

        double lower = Math.min(a, b);
        double upper = Math.max(a, b);

        // Higher resolution scanning
        int numScanPoints = 800;
        double scanRangeStep = (upper - lower) / numScanPoints;

        for (int i = 0; i < criticalNodesCount; i++) {
            int opIdx = getIntVal(criticalOpIndices, i);
            Token operatorToken = originalRpn[opIdx];

            for (double scanPoint = lower; scanPoint <= upper; scanPoint += scanRangeStep) {
                hostExpr.updateSlot(xSlotId, scanPoint);
                double subValue = hostExpr.solveGeneric(scanPoint).scalar;

                if (detectSingularity(operatorToken, subValue)) {
                    double refined = bisectRefine(
                            hostExpr, xSlotId, operatorToken,
                            scanPoint - scanRangeStep,
                            Math.min(scanPoint + scanRangeStep, upper)
                    );

                    if (!isDuplicate(resultBufferOut, entriesFoundCount, refined)) {
                        if (entriesFoundCount < maxOutputsCapacity) {
                            setDoubleVal(resultBufferOut, entriesFoundCount++, refined);
                        }
                    }
                }
            }
        }

        sortBuffer(resultBufferOut, entriesFoundCount);
        return entriesFoundCount;
    }

    private boolean detectSingularity(Token token, double value) {
        if (token.kind == Token.OPERATOR) {
            if (token.opChar == '/') {
                return Math.abs(value) <= 1e-10 || !Double.isFinite(value);
            }
            if (token.opChar == '^') {
                return !Double.isFinite(value) || value < -1e-8;
            }
        }

        if (token.kind == Token.FUNCTION || token.kind == Token.METHOD) {
            String name = token.name.toLowerCase();
            if ("ln".equals(name) || "log".equals(name) || "log10".equals(name)) {
                return value <= 1e-10 || !Double.isFinite(value);
            }
            if ("sqrt".equals(name)) {
                return value < -1e-10 || !Double.isFinite(value);
            }
            if ("asin".equals(name) || "acos".equals(name)) {
                return Math.abs(value) > 1.000001 || !Double.isFinite(value);
            }
            if ("tan".equals(name) || "sec".equals(name)) {
                return Math.abs(Math.cos(value)) <= 1e-8;
            }
            if ("csc".equals(name) || "cot".equals(name)) {
                return Math.abs(Math.sin(value)) <= 1e-8;
            }
        }
        return false;
    }

    private double bisectRefine(MathExpression hostExpr, int xSlotId, Token token,
                                double left, double right) {
        double refined = (left + right) / 2.0;

        for (int step = 0; step < 20; step++) {
            double mid = 0.5 * (left + right);
            hostExpr.updateSlot(xSlotId, mid);
            double val = hostExpr.solveGeneric(mid).scalar;

            if (!Double.isFinite(val) || Math.abs(val) < 1e-12) {
                refined = mid;
                break;
            }
            left = mid;
        }
        return refined;
    }

    private boolean isDuplicate(double[] buffer, int count, double point) {
        for (int k = 0; k < count; k++) {
            if (Math.abs(getVal(buffer, k) - point) < 1e-5) {
                return true;
            }
        }
        return false;
    }

    private void sortBuffer(double[] buffer, int count) {
        // Simple insertion sort for small number of singularities
        for (int i = 1; i < count; i++) {
            double key = getVal(buffer, i);
            int j = i - 1;
            while (j >= 0 && getVal(buffer, j) > key) {
                setDoubleVal(buffer, j + 1, getVal(buffer, j));
                j--;
            }
            setDoubleVal(buffer, j + 1, key);
        }
    }
}