/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.differentialcalculus.pde;

/**
 * A hyperbolic conservation law {@code U_t + sum_d d/dx_d F_d(U) = S}: the
 * physics the finite-volume solver needs, independent of the grid, the
 * reconstruction and the numerical flux.
 * <p>
 * Implementations must be allocation free in these methods (they are called
 * for every face of every step) and must not retain the arrays passed in.
 */
public interface ConservationLaw {

    /** @return number of conserved components. */
    int components();

    /** @return number of space dimensions (1, 2 or 3). */
    int dimension();

    /** @return display names of the components. */
    String[] names();

    /** Physical flux in direction {@code dir} of the conserved state {@code u}. */
    void flux(int dir, double[] u, double[] f);

    /**
     * Smallest and largest characteristic speed in direction {@code dir}
     * (a safe bound is acceptable): {@code out[0] = min, out[1] = max}.
     */
    void waveSpeeds(int dir, double[] u, double[] out);

    /** @return true if the state is admissible (e.g. positive density and pressure). */
    default boolean isPhysical(double[] u) {
        return true;
    }

    /** @return true if reconstruction should act on primitive variables ({@link #toPrimitive}). */
    default boolean usesPrimitiveReconstruction() {
        return false;
    }

    /** Conserved to primitive variables (only used when {@link #usesPrimitiveReconstruction()}). */
    default void toPrimitive(double[] u, double[] w) {
        System.arraycopy(u, 0, w, 0, components());
    }

    /** Primitive to conserved variables. */
    default void toConserved(double[] w, double[] u) {
        System.arraycopy(w, 0, u, 0, components());
    }

    /** @return true if {@link #hllc} is available. */
    default boolean supportsHllc() {
        return false;
    }

    /** HLLC flux (laws with a contact wave only). */
    default void hllc(int dir, double[] uL, double[] uR, double[] f) {
        throw new PDEConfigurationException("This conservation law has no HLLC flux; use LLF, RUSANOV or HLL");
    }

    /** Mirrors a state about a solid wall normal to {@code dir} (reflecting boundary). */
    default void reflect(int dir, double[] u) {
        throw new PDEConfigurationException("This conservation law does not define a reflecting wall; use OUTFLOW, PERIODIC or INFLOW");
    }

    /** @return true if the law has a source term. */
    default boolean hasSource() {
        return false;
    }

    /** Source term {@code S(u, x, y, z, t)} (only called when {@link #hasSource()}). */
    default void source(double[] u, double x, double y, double z, double t, double[] out) {
        throw new UnsupportedOperationException("no source term");
    }
}
