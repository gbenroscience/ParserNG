package com.github.gbenroscience.math.differentialcalculus.equations.standard;

/**
 * Fuses N independently-compiled per-equation ODEFunctions (one per
 * dy[i]/dt = f_i(t, y)) into one ODEFunction matching the standard
 * apply(double[], double[]) dy_dt descriptor — the explicit-system
 * counterpart to CompanionSystemHandles' single-top-derivative reduction.
 *
 * Each perEquationFn[i] is expected to already be frame-remapped (via
 * FrameRemapper) against ITS OWN independently-resolved CanonicalFrame —
 * this class doesn't know or care how each component was compiled, only
 * that each writes its one derivative value to index 0 of its own output
 * array when applied.
 */
public final class SystemFunctionHandles {

    private SystemFunctionHandles() {
    }

    /**
     * @param perEquationFn N ODEFunctions; perEquationFn[i] computes dy[i]/dt
     * @param systemSize N
     * @return a new ODEFunction, ready to hand to the vector solvers like any
     * hand-written system dy_dt
     *
     * Note: stateful per instance (small reusable scratch buffer) — one
     * built instance backs exactly one solve call, same contract as every
     * other per-solve adapter in this pipeline.
     */
    public static ODEFunction buildSystem(ODEFunction[] perEquationFn, int systemSize) {
        if (perEquationFn == null || perEquationFn.length != systemSize) {
            throw new IllegalArgumentException(
                    "perEquationFn.length must equal systemSize, got "
                    + (perEquationFn == null ? "null" : perEquationFn.length) + " vs " + systemSize);
        }
        for (int i = 0; i < perEquationFn.length; i++) {
            if (perEquationFn[i] == null) {
                throw new IllegalArgumentException("perEquationFn[" + i + "] must not be null");
            }
        }
        return new SystemFunction(perEquationFn);
    }

    private static final class SystemFunction implements ODEFunction {

        private final ODEFunction[] perEquationFn;
        private final double[] componentOut = new double[1];

        SystemFunction(ODEFunction[] perEquationFn) {
            this.perEquationFn = perEquationFn;
        }

        @Override
        public void apply(double[] vars, double[] outDerivatives) {
            for (int i = 0; i < perEquationFn.length; i++) {
                perEquationFn[i].apply(vars, componentOut);
                outDerivatives[i] = componentOut[0];
            }
        }
    }
}