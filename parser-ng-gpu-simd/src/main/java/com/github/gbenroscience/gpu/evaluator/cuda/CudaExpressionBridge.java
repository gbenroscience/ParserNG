package com.github.gbenroscience.gpu.evaluator.cuda;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;

/**
 * CUDA counterpart of com.github.gbenroscience.gpu.opencl.OpenClExpressionBridge --
 * same idea, same usage shape, just wired to CudaCompositeExpression
 * instead of the OpenCL one. No separate "CUDA compiler" step exists here
 * either: VectorTurboEvaluator's bytecode (opcodes/targetSlots/
 * literalConstants) is compiled to PTX and dispatched as-is, same as the
 * OpenCL path compiles the same bytecode's opcode values into a switch
 * statement.
 *
 * Returns the shared com.github.gbenroscience.gpu.GpuCompositeExpr
 * interface rather than the concrete CudaCompositeExpression type, so
 * callers that don't specifically need CUDA can go through
 * com.github.gbenroscience.gpu.GpuExpressionBridge instead and stay
 * backend-agnostic. Use this bridge directly only when CUDA specifically
 * (not OpenCL) is required.
 *
 * Usage matches both the CPU and OpenCL patterns exactly:
 *
 *   MathExpression me = new MathExpression("3*cos(x-2)+ln(3x^3-5x-4*tan(x))");
 *   VectorTurboEvaluator vet = new VectorTurboEvaluator(me);
 *
 *   try (GpuCompositeExpr cuda = CudaExpressionBridge.from(vet)) {
 *       cuda.applyBulk(flatVariables, output);        // double[] overload
 *       cuda.applyBulk(inMemorySegment, outSegment);   // MemorySegment overload
 *   }
 */
public final class CudaExpressionBridge {

    private CudaExpressionBridge() {
    }

    /**
     * Builds a CUDA-backed evaluator directly from an already-constructed
     * VectorTurboEvaluator, reusing its compiled program without recompiling
     * or re-parsing anything.
     *
     * @throws IllegalArgumentException if the expression's stack depth
     *         exceeds CudaKernelSource.MAX_STACK. Same rationale as the
     *         OpenCL bridge: the kernel's per-thread stack is a fixed-size
     *         local array with no bounds checking, so an over-deep
     *         expression must be rejected here rather than silently
     *         corrupting kernel-local memory on-device.
     */
    public static GpuCompositeExpression from(VectorTurboEvaluator vte) {
        int depth = vte.getStackDepth();
        if (depth > CudaKernelSource.MAX_STACK) {
            throw new IllegalArgumentException(
                    "Expression stack depth (" + depth + ") exceeds CUDA kernel's MAX_STACK ("
                            + CudaKernelSource.MAX_STACK + "). Raise MAX_STACK in "
                            + "CudaKernelSource to match, or this expression cannot safely "
                            + "run on the CUDA path yet.");
        }

        return new CudaCompositeExpression(
                vte.getOpcodes(),
                vte.getTargetSlots(),
                vte.getLiteralConstants(),
                vte.getInstructionCount(),
                vte.getVarCount());
    }

    /**
     * Convenience one-shot: parses/compiles straight from a MathExpression,
     * skipping the intermediate VectorTurboEvaluator variable if the caller
     * doesn't need it for anything else.
     */
    public static GpuCompositeExpression compile(MathExpression me) throws Throwable {
        return from(new VectorTurboEvaluator(me));
    }
}