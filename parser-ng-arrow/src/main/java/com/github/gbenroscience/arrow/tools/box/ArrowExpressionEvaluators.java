package com.github.gbenroscience.arrow.tools.box;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.parser.MathExpression;

/**
 * Single entry point for compiling an {@link ArrowExpressionEvaluator}
 * against whichever backend you choose. This is the intended way to switch
 * between the CPU SIMD engine ({@link ArrowBulkEvaluator}) and a GPU backend
 * ({@link ArrowGpuBulkEvaluator}, CUDA or OpenCL) — pick an
 * {@link ArrowExecutionBackend} value rather than constructing either
 * concrete class directly, so changing backends later is a one-line change
 * here instead of a rewrite at every call site.
 *
 * <h2>Fixed backend vs "prefer GPU, fall back to CPU"</h2>
 * {@link #compile(MathExpression, ArrowExecutionBackend)} compiles against
 * exactly the backend you name and throws if that backend can't be
 * bootstrapped — including {@link ArrowExecutionBackend#GPU_AUTO}, which
 * still throws if NO GPU backend (CUDA or OpenCL) is available; it does not
 * fall back to the CPU. {@link #compilePreferGpu} is the method that
 * actually falls back: it tries {@code GPU_AUTO} first and only compiles the
 * CPU SIMD evaluator if every GPU backend fails to bootstrap. Use the fixed
 * form when the caller has already decided (e.g. from configuration, or
 * after checking {@link ArrowGpuBulkEvaluator#isBackendAvailable}); use
 * {@code compilePreferGpu} for a simple "use the GPU if there is one"
 * default.
 *
 * <h2>Device selection</h2>
 * Selecting a specific GPU device (as opposed to just a backend) is
 * backend-specific and happens on {@link ArrowGpuBulkEvaluator} itself,
 * before compiling, since it isn't part of the portable
 * {@link ArrowExpressionEvaluator} surface — see
 * {@link ArrowGpuBulkEvaluator#listOpenClDevices()},
 * {@link ArrowGpuBulkEvaluator#selectOpenClDevice(String)}, and
 * {@link ArrowGpuBulkEvaluator#selectCudaDeviceIndex(int)}. Call the
 * relevant selection method, then compile through this class as usual — the
 * selection is picked up by whichever backend actually runs.
 */
public final class ArrowExpressionEvaluators {

    private ArrowExpressionEvaluators() {
    }

    /**
     * Compiles {@code expr} against {@code backend}.
     * @param expr
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compile(String expr, ArrowExecutionBackend backend) throws Throwable {
        return compile(new MathExpression(expr), backend);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} against
     * {@code backend}.
     * @param expression
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compile(MathExpression expression, ArrowExecutionBackend backend) throws Throwable {
        if (backend == null) {
            throw new NullPointerException("backend must not be null");
        }
        switch (backend) {
            case CPU_SIMD:
                return ArrowBulkEvaluator.compile(expression, 0);
            case GPU_AUTO:
                return ArrowGpuBulkEvaluator.compile(expression);
            case GPU_CUDA:
                return ArrowGpuBulkEvaluator.compile(expression, GpuBackend.CUDA);
            case GPU_OPENCL:
                return ArrowGpuBulkEvaluator.compile(expression, GpuBackend.OPENCL);
            default:
                // Unreachable unless ArrowExecutionBackend grows a new constant
                // without a matching case here.
                throw new IllegalArgumentException("Unhandled ArrowExecutionBackend: " + backend);
        }
    }

    /**
     * Compiles {@code expr} against {@link ArrowExecutionBackend#GPU_AUTO},
     * falling back to {@link ArrowExecutionBackend#CPU_SIMD} if no GPU
     * backend bootstraps on this machine at all. The most common choice for
     * "use the GPU when there is one, otherwise just work".
     *
     * <p>If both the GPU attempt and the CPU fallback fail, the CPU
     * exception is thrown with the original GPU failure attached via
     * {@link Throwable#addSuppressed}, so the real GPU bootstrap error isn't
     * lost — a fallback CPU compile failure almost always means the
     * expression itself is invalid, not that the CPU engine is unavailable,
     * so that's the more informative exception to surface as primary.
     * @param expr
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compilePreferGpu(String expr) throws Throwable {
        return compilePreferGpu(new MathExpression(expr));
    }

    /**
     * Compiles an already-constructed {@link MathExpression} against
     * {@link ArrowExecutionBackend#GPU_AUTO}, falling back to
     * {@link ArrowExecutionBackend#CPU_SIMD} if no GPU backend bootstraps at
     * all. See {@link #compilePreferGpu(String)} for the fallback/exception
     * behavior.
     * @param expression
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowExpressionEvaluator compilePreferGpu(MathExpression expression) throws Throwable {
        try {
            return ArrowGpuBulkEvaluator.compile(expression);
        } catch (Throwable gpuFailure) {
            try {
                return ArrowBulkEvaluator.compile(expression, 0);
            } catch (Throwable cpuFailure) {
                cpuFailure.addSuppressed(gpuFailure);
                throw cpuFailure;
            }
        }
    }
}