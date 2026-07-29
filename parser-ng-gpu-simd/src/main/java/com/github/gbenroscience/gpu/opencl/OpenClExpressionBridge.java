package com.github.gbenroscience.gpu.opencl;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.gpu.GpuCompositeExpression;

/**
 * Links the CPU-side compiled program (opcodes/targetSlots/literalConstants,
 * produced by VectorTurboEvaluator's constructor -- the SAME parse-to-bytecode
 * step the SIMD/scalar paths already use) to the OpenCL interpreter. No
 * separate "GPU compiler" exists or is needed: the bytecode IS the GPU
 * program, which is the entire point of the generic-interpreter-kernel
 * design -- see OpenClKernelSource's javadoc.
 *
 * Returns the shared com.github.gbenroscience.gpu.GpuCompositeExpr
 * interface rather than the concrete OpenClCompositeExpression type, so
 * callers that don't specifically need OpenCL can go through
 * com.github.gbenroscience.gpu.GpuExpressionBridge instead and stay
 * backend-agnostic. Use this bridge directly only when OpenCL specifically
 * (not CUDA) is required.
 */
public final class OpenClExpressionBridge {

    private OpenClExpressionBridge() {
    }

    /**
     * Builds an OpenCL-backed evaluator directly from an already-constructed
     * VectorTurboEvaluator, reusing its compiled program without recompiling
     * or re-parsing anything.
     *
     * @param vte
     * @return 
     * @throws IllegalArgumentException if the expression's stack depth
     *         exceeds OpenClKernelSource.MAX_STACK. The GPU kernel's
     *         per-thread stack is a fixed-size private array with no bounds
     *         checking -- silently letting an over-deep expression through
     *         would corrupt kernel-local memory rather than fail loudly like
     *         the CPU path would.
     */
    public static GpuCompositeExpression from(VectorTurboEvaluator vte) {
        int depth = vte.getStackDepth();
        if (depth > OpenClKernelSource.MAX_STACK) {
            throw new IllegalArgumentException(
                    "Expression stack depth (" + depth + ") exceeds GPU kernel's MAX_STACK ("
                            + OpenClKernelSource.MAX_STACK + "). Raise MAX_STACK in "
                            + "OpenClKernelSource and OpenClCompositeExpression to match, "
                            + "or this expression cannot safely run on the GPU path yet.");
        }

        return new OpenClCompositeExpression(
                vte.getOpcodes(),
                vte.getTargetSlots(),
                vte.getLiteralConstants(),
                vte.getInstructionCount(),
                vte.getVarCount());
    }

    /**
     * Convenience one-shot: parses/compiles straight from a MathExpression,
     * skipping the intermediate VectorTurboEvaluator variable if the caller
     * doesn't need it for anything else (e.g. doesn't also want a CPU
     * fallback evaluator for the same expression).
     */
    public static GpuCompositeExpression compile(MathExpression me) throws Throwable {
        return from(new VectorTurboEvaluator(me));
    }
}