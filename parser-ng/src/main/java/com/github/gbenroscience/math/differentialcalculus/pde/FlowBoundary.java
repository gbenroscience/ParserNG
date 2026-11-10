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

import java.util.Arrays;

/**
 * Condition on one face of the domain for the incompressible flow solver.
 * Velocity data are ParserNG expressions of {@code x}, {@code y}, {@code z},
 * {@code t} (and any declared parameters), evaluated at the face.
 * <ul>
 * <li>{@link #noSlip()} / {@link #movingWall}: solid wall; fluid velocity equals the wall velocity.</li>
 * <li>{@link #slip()}: solid wall without friction (zero normal velocity, zero shear).</li>
 * <li>{@link #inflow}: prescribed velocity vector.</li>
 * <li>{@link #outflow()}: open boundary; zero normal gradient of velocity, pressure fixed to zero.</li>
 * <li>{@link #periodic()}: wraps to the opposite face (must be given on both faces of the axis).</li>
 * </ul>
 */
public final class FlowBoundary {

    /** Kind of boundary. */
    public enum Type {
        NO_SLIP, SLIP, INFLOW, OUTFLOW, PERIODIC
    }

    private final Type type;
    private final String[] velocity;

    private FlowBoundary(Type type, String[] velocity) {
        this.type = type;
        this.velocity = velocity;
    }

    /** Stationary no-slip wall. */
    public static FlowBoundary noSlip() {
        return new FlowBoundary(Type.NO_SLIP, null);
    }

    /**
     * No-slip wall moving with the given velocity (one expression per component; the
     * component normal to the wall is ignored and taken as zero).
     */
    public static FlowBoundary movingWall(String... velocity) {
        return new FlowBoundary(Type.NO_SLIP, Arrays.copyOf(velocity, velocity.length));
    }

    /** Frictionless wall. */
    public static FlowBoundary slip() {
        return new FlowBoundary(Type.SLIP, null);
    }

    /** Prescribed inflow velocity, one expression per velocity component. */
    public static FlowBoundary inflow(String... velocity) {
        if (velocity.length == 0) {
            throw new PDEConfigurationException("inflow() needs a velocity expression per component");
        }
        return new FlowBoundary(Type.INFLOW, Arrays.copyOf(velocity, velocity.length));
    }

    /** Open (outflow) boundary with zero reference pressure. */
    public static FlowBoundary outflow() {
        return new FlowBoundary(Type.OUTFLOW, null);
    }

    /** Periodic. */
    public static FlowBoundary periodic() {
        return new FlowBoundary(Type.PERIODIC, null);
    }

    public Type getType() {
        return type;
    }

    /** @return the velocity expressions, or null if none were given. */
    public String[] getVelocity() {
        return velocity == null ? null : velocity.clone();
    }

    @Override
    public String toString() {
        return type + (velocity == null ? "" : Arrays.toString(velocity));
    }
}
