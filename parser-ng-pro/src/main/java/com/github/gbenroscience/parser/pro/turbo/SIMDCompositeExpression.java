package com.github.gbenroscience.parser.pro.turbo;

import com.github.gbenroscience.parser.pro.turbo.tools.FlatMatrix;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;

/**
 *
 * @author GBEMIRO
 */
public interface SIMDCompositeExpression extends FastCompositeExpression {

    /**
     * Executes the compiled expression over massive arrays using Hardware SIMD
     * Vectors.
     *
     * @param variables A 2D array where variables[slot][i] is the data.
     * @param output The output array to dump the parallel results into.
     */
    void applyBulk(double[][] variables, double[] output);

    /**
     * Executes fused deep learning kernels directly via FlatMatrix.
     *
     * @param inputs Array of input matrices (e.g., [A, B] or [A, B, Bias])
     * @param output The pre-allocated output matrix to store the result
     * @param operation The kernel identifier (e.g., "matmul", "gelu")
     */
    default void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String operation) {
        throw new UnsupportedOperationException("Matrix kernel not implemented for: " + operation);
    }
}
