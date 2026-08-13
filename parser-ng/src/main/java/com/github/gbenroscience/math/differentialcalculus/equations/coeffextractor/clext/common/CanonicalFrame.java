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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common;

/**
 *
 * @author GBEMIRO
 */ 

/**
 * Fixes the frame-ordering bug found while building this pipeline:
 * MathExpression's VariableRegistry assigns frame slots in first-appearance
 * order in the source text, not in ascending {@code y[k]} order — so for an
 * equation like {@code a(t)*y[4]+b(t)*y[3]+...+e(t)*y[0]}, the real slots
 * for {@code y[0]..y[4]} can land anywhere, in any order, non-contiguous.
 * Every solver in this codebase ({@code CompanionSystemHandles}, the
 * stepping cores in {@code DifferentialEquations}) hard-assumes {@code
 * vars[ySlotStart+k]} is state component k, contiguous and ascending — a
 * direct conflict with the real registry layout.
 *
 * <h2>The fix</h2>
 * Solvers only ever see a CANONICAL frame this class defines: canonical
 * slot 0 = t, canonical slots {@code 1..order} = {@code y[0..order-1]},
 * contiguous by construction (since a caller who builds one gets to pick
 * that layout, independent of whatever the real registry happened to
 * assign). {@link #toReal} translates a canonical-frame array into the real
 * (scattered) frame the compiled {@code ExprNode} expressions actually read
 * from — a compiled expression's own {@code vars[node.frameIndex]} reads
 * are untouched by any of this; only the array they read from gets built
 * correctly first.
 *
 * <h2>Missing real slots — the sparse-equation case</h2>
 * A state index that never appears anywhere in the raw equation text has no
 * real registry slot at all — {@code MathExpression} never scanned that
 * name, so there is nothing to map it to. {@code canonicalToReal[i] ==
 * NO_REAL_SLOT} for such an index means: skip writing that canonical value
 * into the real frame. That is always safe, precisely because a compiled
 * expression built from that same equation text can only read a real frame
 * index for a variable it actually references — if {@code y[k]} has no real
 * slot, the compiled top-derivative expression provably never reads it
 * either, and its true partial derivative w.r.t. that state component is
 * exactly zero (used by {@link EquationRuntime}'s Jacobian construction).
 */
public final class CanonicalFrame {

    public static final int NO_REAL_SLOT = -1;

    /** canonicalToReal[0] = t's real slot; canonicalToReal[1+k] = y[k]'s real slot, or NO_REAL_SLOT. */
    public final int[] canonicalToReal;
    public final int realFrameSize;

    private final double[] scratch;

    public CanonicalFrame(int[] canonicalToReal, int realFrameSize) {
        if (canonicalToReal == null) {
            throw new IllegalArgumentException("canonicalToReal must not be null");
        }
        this.canonicalToReal = canonicalToReal;
        this.realFrameSize = realFrameSize;
        this.scratch = new double[realFrameSize];
    }

    public int order() {
        return canonicalToReal.length - 1;
    }

    /**
     * Translates canonical values into the shared real-frame scratch
     * buffer, returning it. Not thread-safe — the buffer is reused per
     * call, the same contract every other per-solve scratch buffer in this
     * pipeline already uses (one instance backs one sequential solve).
     * @param canonicalVars
     * @return 
     */
    public double[] toReal(double[] canonicalVars) {
        for (int i = 0; i < canonicalToReal.length; i++) {
            int realSlot = canonicalToReal[i];
            if (realSlot != NO_REAL_SLOT) {
                scratch[realSlot] = canonicalVars[i];
            }
        }
        return scratch;
    }
}