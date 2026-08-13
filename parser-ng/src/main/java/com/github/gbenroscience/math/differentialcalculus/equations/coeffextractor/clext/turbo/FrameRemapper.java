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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.turbo;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.CanonicalFrame;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
/**
 *
 * @author GBEMIRO 
 * Turbo-tier counterpart to {@link FrameRemappingODEFunction}: wraps a
 * {@code MethodHandle} compiled against ParserNG's real (possibly
 * scattered) frame indices so the solver only ever sees {@link
 * CanonicalFrame}'s contiguous layout — see that class's javadoc for why
 * this exists.
 * <p>
 * Uses the exact same {@code MethodHandles.lookup().bind(...)} adapter
 * pattern already established by {@code CompanionSystemHandles} and {@code
 * LinearHODifferentialEquations}: a small private instance method matching
 * {@code (double[], double[])void} is bound to a per-call adapter instance,
 * producing a {@code MethodHandle} with that same descriptor.
 */
public final class FrameRemapper {

    private static final MethodType TYPE = MethodType.methodType(void.class, double[].class, double[].class);

    private FrameRemapper() {
    }

    /**
     * @param real  MethodHandle compiled against the real frame, matching
     *              {@code (double[], double[])void}
     * @param frame the canonical/real translation to apply before each call
     * @return a new MethodHandle, same descriptor, that accepts canonical-
     *         frame input and delegates to {@code real} against the real frame
     */
    public static MethodHandle wrap(MethodHandle real, CanonicalFrame frame) {
        if (real == null) {
            throw new IllegalArgumentException("real MethodHandle must not be null");
        }
        if (frame == null) {
            throw new IllegalArgumentException("frame must not be null");
        }
        if (!real.type().equals(TYPE)) {
            throw new IllegalArgumentException(
                    "real MethodHandle has incompatible signature. Expected " + TYPE + " but got " + real.type());
        }

        Adapter adapter = new Adapter(real, frame);
        try {
            return MethodHandles.lookup().bind(adapter, "apply", TYPE);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("Failed to bind frame-remapping adapter", e);
        }
    }

    private static final class Adapter {
        private final MethodHandle real;
        private final CanonicalFrame frame;

        Adapter(MethodHandle real, CanonicalFrame frame) {
            this.real = real;
            this.frame = frame;
        }

        // Signature must exactly match (double[], double[])void for invokeExact/bind.
        void apply(double[] canonicalVars, double[] outDerivatives) throws Throwable {
            real.invokeExact(frame.toReal(canonicalVars), outDerivatives);
        }
    }
}