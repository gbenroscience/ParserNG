package com.github.gbenroscience.math.differentialcalculus.equations.turbo;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * Fuses N independently-compiled per-equation MethodHandles (one per
 * dy[i]/dt = f_i(t, y)) into one MethodHandle matching the standard
 * (double[], double[])void dy_dt descriptor — the explicit-system
 * counterpart to CompanionSystemHandles' single-top-derivative reduction.
 *
 * Each perEquationFn[i] is expected to already be frame-remapped (via
 * FrameRemapper) against ITS OWN independently-resolved CanonicalFrame —
 * this class doesn't know or care how each component was compiled, only
 * that each matches (double[] canonicalVars, double[] singleResultOut)void,
 * writing its one derivative value to index 0 of its own output array.
 */
public final class SystemFunctionHandles {

    private static final MethodType DY_DT_TYPE
            = MethodType.methodType(void.class, double[].class, double[].class);

    private SystemFunctionHandles() {
    }

    /**
     * @param perEquationFn N handles, each matching (double[], double[])void;
     * perEquationFn[i] computes dy[i]/dt
     * @param systemSize N
     * @return a new MethodHandle matching (double[], double[])void, ready to
     * hand to the vector solvers like any hand-written system dy_dt
     *
     * Note: stateful per instance (small reusable scratch buffer) — one
     * built handle backs exactly one solve call, same contract as every
     * other per-solve adapter in this pipeline.
     */
    public static MethodHandle buildSystem(MethodHandle[] perEquationFn, int systemSize) {
        if (perEquationFn == null || perEquationFn.length != systemSize) {
            throw new IllegalArgumentException(
                    "perEquationFn.length must equal systemSize, got "
                    + (perEquationFn == null ? "null" : perEquationFn.length) + " vs " + systemSize);
        }
        for (int i = 0; i < perEquationFn.length; i++) {
            if (perEquationFn[i] == null) {
                throw new IllegalArgumentException("perEquationFn[" + i + "] must not be null");
            }
            if (!perEquationFn[i].type().equals(DY_DT_TYPE)) {
                throw new IllegalArgumentException(
                        "perEquationFn[" + i + "] has incompatible signature. Expected " + DY_DT_TYPE
                        + " but got " + perEquationFn[i].type());
            }
        }

        Adapter adapter = new Adapter(perEquationFn);
        try {
            return MethodHandles.lookup().bind(adapter, "systemDerivative", DY_DT_TYPE);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("Failed to bind system-function adapter", e);
        }
    }

    private static final class Adapter {

        private final MethodHandle[] perEquationFn;
        // Reusable per-call scratch -- one instance backs exactly one solve call.
        private final double[] componentOut = new double[1];

        Adapter(MethodHandle[] perEquationFn) {
            this.perEquationFn = perEquationFn;
        }

        void systemDerivative(double[] vars, double[] outDerivatives) throws Throwable {
            for (int i = 0; i < perEquationFn.length; i++) {
                perEquationFn[i].invokeExact(vars, componentOut);
                outDerivatives[i] = componentOut[0];
            }
        }
    }
}