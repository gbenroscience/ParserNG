package com.github.gbenroscience.math.differentialcalculus.equations.standard;

/**
 * The ParserNG Standard replacement for the Turbo tier's MethodHandle dy_dt
 * convention. A compiled RHS expression — whether it is the plain first-order
 * f(t, y), a higher-order top derivative, or a linear-form coefficient —
 * implements this interface directly instead of being linked as a
 * MethodHandle.
 *
 * There is no invokeExact, no MethodType checking, and no reflective
 * binding anywhere in this package as a result: a call is just a normal
 * virtual method call, and any error from evaluating the expression is a
 * normal unchecked exception (typically ArithmeticException for a domain
 * violation), not a checked Throwable propagated up through every solver
 * signature.
 */
@FunctionalInterface
public interface ODEFunction {

    /**
     * @param vars           execution frame — vars(tSlot) holds t, vars(ySlotStart..)
     *                       holds the current state (or, for a linear-form
     *                       coefficient, vars(tSlot) is all that is read)
     * @param outDerivatives output array to fill; for a system this is one
     *                       entry per state component, for a scalar or a
     *                       single coefficient it is a length-1 array whose
     *                       only entry is written
     */
    void apply(double[] vars, double[] outDerivatives);
}