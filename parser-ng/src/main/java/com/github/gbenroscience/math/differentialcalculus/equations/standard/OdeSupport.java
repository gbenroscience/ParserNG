package com.github.gbenroscience.math.differentialcalculus.equations.standard;

/**
 * Small shared helpers used across the Turbo/Vector/HigherOrder ODE entry points.
 * Package-private: not part of the public ParserNG-facing surface.
 */
final class OdeSupport {

    /** Default output resolution for *Path entry points when the caller doesn't specify one. */
    static final int DEFAULT_PATH_POINTS = 100;

    private OdeSupport() {
    }

    /**
     * Converts a signed interval and a positive step magnitude into a positive
     * fixed step count, so backward integration (tEnd &lt; t0) gets the same
     * resolution as forward integration.
     */
    static int fixedStepCount(double t0, double tEnd, double stepMagnitude) {
        int steps = (int) Math.round(Math.abs(tEnd - t0) / stepMagnitude);
        return Math.max(steps, 1);
    }
}