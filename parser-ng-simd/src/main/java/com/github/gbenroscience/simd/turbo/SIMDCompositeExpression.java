package com.github.gbenroscience.simd.turbo;

import com.github.gbenroscience.simd.turbo.tools.FlatMatrix;
import com.github.gbenroscience.simd.turbo.tools.FlatMatrixF;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;

/**
 * Super-fast, SIMD-aligned vector expression evaluator interface. Exposes a
 * dual-path API optimized for massive parallel time-series calculation, machine
 * learning transformations, and fused token attention operations.
 *
 * * @author GBEMIRO
 */
public interface SIMDCompositeExpression extends FastCompositeExpression {

    /**
     * Executes the compiled expression over a convenient 2D array of variables.
     * Under the hood, this method flattens and groups data into sequential
     * memory blocks before delegating to the core vectorized execution kernel.
     *
     * @param variables A 2D array of variable channels where each outer row
     * represents an entire vector for a specific variable slot, e.g.,
     * {@code [[x1, x2... xn], [y1, y2... yn], [z1, z2... zn]]}.
     * @param output The pre-allocated target array where the final evaluated
     * results will be directly dumped.
     */
    void applyBulk(double[][] variables, double[] output);

    /**
     * Distributes the evaluation of a 2D array dataset evenly across multiple
     * processing threads using a fork-join block chunking methodology.
     *
     * @param variables A 2D array of variable channels where each outer row
     * represents an entire vector for a specific variable slot, e.g.,
     * {@code [[x1, x2... xn], [y1, y2... yn]]}.
     * @param output The pre-allocated target array where the parallel
     * calculations will be written.
     */
    public void applyBulkParallel(double[][] variables, double[] output);

    /**
     * Evaluates a 2D variable structure using an explicit, cache-bounded window
     * chunk size to enforce strict CPU L1/L2 data cache localization.
     *
     * @param variables A 2D array of variable channels where each outer row
     * represents an entire vector for a specific variable slot, e.g.,
     * {@code [[x1, x2... xn], [y1, y2... yn]]}.
     * @param output The pre-allocated target array where the evaluated blocks
     * will be written.
     * @param batchSize The strict memory segment window slice size processed
     * per individual cache loop pass.
     */
    public void applyBulkBatched(double[][] variables, double[] output, int batchSize);

    /**
     * <b>Warp Speed Path (Power Users):</b> Evaluates a single, raw,
     * pre-grouped flat array at maximum hardware throughput with zero memory
     * copies, zero allocations, and direct sequential prefetching.
     *
     * @param flatVariables A single flat array containing variables
     * contiguously grouped back-to-back by variable slot.
     * <b>CRITICAL:</b> Data must use a Grouped structure, e.g.,
     * {@code [x1, x2... xn, y1, y2... yn, z1, z2... zn]}. Interleaved arrays
     * will yield corrupted data.
     * @param output The pre-allocated target array where the evaluated vector
     * stream is directly copied.
     */
    public void applyBulk(double[] flatVariables, double[] output);

    /**
     * <b>Warp Speed Path (Power Users):</b> Concurrently evaluates a
     * pre-grouped flat array by cleanly dividing segments across active CPU
     * cores for optimal multi-threaded memory bandwidth consumption.
     *
     * @param flatVariables A single flat array containing variables
     * contiguously grouped back-to-back by variable slot.
     * <b>CRITICAL:</b> Data must use a Grouped structure, e.g.,
     * {@code [x1, x2... xn, y1, y2... yn, z1, z2... zn]}. Do NOT pass
     * interleaved data.
     * @param output The pre-allocated target array where parallel workers will
     * drop evaluated computations.
     */
    public void applyBulkParallel(double[] flatVariables, double[] output);

    /**
     * <b>Warp Speed Path (Power Users):</b> Evaluates a pre-grouped flat array
     * using custom-defined batch chunk boundaries to maximize custom hardware
     * L1/L2 execution efficiency.
     *
     * @param flatVariables A single flat array containing variables
     * contiguously grouped back-to-back by variable slot.
     * <b>CRITICAL:</b> Data must use a Grouped structure, e.g.,
     * {@code [x1, x2... xn, y1, y2... yn, z1, z2... zn]}. Do NOT pass
     * interleaved data.
     * @param output The pre-allocated target array where the evaluated blocks
     * will be written.
     * @param batchSize The localized processing block window length applied
     * across individual memory loops.
     */
    public void applyBulkBatched(double[] flatVariables, double[] output, int batchSize);

    /**
     * Fuses deep learning and high-performance neural network transformations
     * directly over pre-allocated double-precision structural tensor types.
     *
     * @param inputs An ordered array of operational tensor matrices, e.g.,
     * weights, targets, scales, or vector biases.
     * @param output The destination matrix wrapper initialized to capture
     * transformed weights or spatial dimensions.
     * @param operation The explicit execution identifier targeting underlying
     * matrix optimization layers (e.g., "matmul", "rms_norm", "swiglu",
     * "q8_quantize").
     */
    public void applyMatrixKernel(FlatMatrix[] inputs, FlatMatrix output, String operation);

    /**
     * Fuses deep learning and high-performance neural network transformations
     * directly over pre-allocated single-precision (float) tensor execution
     * spaces.
     *
     * @param inputs An ordered array of float operational matrices, optimized
     * for lightning-fast transformer workloads.
     * @param output The destination single-precision tensor buffer designed to
     * safely lock down mutated states.
     * @param op The explicit execution identifier targeting underlying matrix
     * optimization layers (e.g., "matmul_bias_gelu", "rope_split",
     * "mha_attention").
     */
    public void applyMatrixKernel(FlatMatrixF[] inputs, FlatMatrixF output, String op);
}
