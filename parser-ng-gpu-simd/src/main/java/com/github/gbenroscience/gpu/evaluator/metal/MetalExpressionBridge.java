package com.github.gbenroscience.gpu.evaluator.metal;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;

/**
 * Links the CPU-side compiled program (opcodes/targetSlots/literalConstants,
 * produced by VectorTurboEvaluator's constructor -- the SAME parse-to-bytecode
 * step the SIMD/scalar and OpenCL paths already use) to the Metal interpreter.
 * No separate "GPU compiler" exists here either, for the same reason
 * OpenClExpressionBridge doesn't need one: the bytecode IS the GPU program.
 *
 * Returns the shared com.github.gbenroscience.gpu.GpuCompositeExpr interface
 * rather than the concrete MetalCompositeExpression type, so callers that
 * don't specifically need Metal can go through
 * com.github.gbenroscience.gpu.GpuExpressionBridge instead and stay
 * backend-agnostic. Use this bridge directly only when Metal specifically
 * (not OpenCL/CUDA) is required -- e.g. because the target machine is a Mac
 * where Metal is guaranteed present and Apple's OpenCL is deprecated, or
 * because the discrete float-only throughput profile described in
 * MetalCompositeExpression's javadoc is exactly what's wanted.
 *
 * NOTE ON PRECISION: unlike OpenClExpressionBridge, the evaluator this
 * produces has no double-precision path at all -- see
 * MetalCompositeExpression's class javadoc point 1. Callers whose
 * expressions require exact double precision should route through
 * OpenClExpressionBridge instead; {@link #from} does not attempt to detect
 * or warn about precision-sensitive expressions, since that determination is
 * about the expression's numerical requirements, not something derivable
 * from its bytecode alone.
 */
public final class MetalExpressionBridge {

    private MetalExpressionBridge() {
    }

    /**
     * Builds a Metal-backed evaluator directly from an already-constructed
     * VectorTurboEvaluator, reusing its compiled program without recompiling
     * or re-parsing anything.
     *
     * @param vte
     * @return
     * @throws IllegalArgumentException if the expression's stack depth
     *         exceeds MetalKernelSource.MAX_STACK. The GPU kernel's
     *         per-thread stack is a fixed-size private array with no bounds
     *         checking -- silently letting an over-deep expression through
     *         would corrupt kernel-local (threadgroup-private) memory rather
     *         than fail loudly like the CPU path would, exactly as the
     *         equivalent OpenCL check documents.
     */
    public static GpuCompositeExpression from(VectorTurboEvaluator vte) {
        int depth = vte.getStackDepth();
        if (depth > MetalKernelSource.MAX_STACK) {
            throw new IllegalArgumentException(
                    "Expression stack depth (" + depth + ") exceeds GPU kernel's MAX_STACK ("
                            + MetalKernelSource.MAX_STACK + "). Raise MAX_STACK in "
                            + "MetalKernelSource and MetalCompositeExpression to match, "
                            + "or this expression cannot safely run on the GPU path yet.");
        }
        return new MetalCompositeExpression(
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