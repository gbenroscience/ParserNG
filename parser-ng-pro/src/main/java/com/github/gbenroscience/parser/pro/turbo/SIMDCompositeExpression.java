package com.github.gbenroscience.parser.pro.turbo;

import com.github.gbenroscience.parser.pro.turbo.tools.FlatMatrix;
import com.github.gbenroscience.parser.pro.turbo.tools.FlatMatrixF;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import java.util.concurrent.ExecutorService;

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

    public void applyBulk(double[][] variables, double[] outputBuffer, int offset) throws Throwable;

    /**
     * If the double[][] variables passed to this method is a "jagged" array or an array-of-arrays,
     * the performance will suffer compared to the flat 1D array we used earlier. 
     * If the API allows, encourage users to flatten their input data before calling
     * these methods to achieve that 100x+ speedup we saw in your previous benchmark.
     * @param variables
     * @param output
     * @param executor 
     */
    public void applyBulk(double[][] variables, double[] output, ExecutorService executor);
    
    /**
     * We suggest a default of 1024 or 2048. 
     * This is generally large enough to saturate the SIMD pipeline but small enough to fit 
     * within most L1/L2 caches across different CPU architectures.
     * @param variables
     * @param output
     * @param batchSize 
     */
    public void applyBulkBatched(double[][] variables, double[] output, int batchSize);

    /**
     * Executes fused deep learning kernels directly via FlatMatrix.
     *
     * @param inputs Array of input matrices (e.g., [A, B] or [A, B, Bias])
     * @param output The pre-allocated output matrix to store the result
     * @param operation The kernel identifier (e.g., "matmul", "gelu")
     */
    public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String operation);
    public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op);
}
