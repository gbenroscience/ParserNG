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
 * A face of the rectangular computational domain. {@code X_MIN} is the face
 * where the first independent variable takes its smallest value, and so on.
 */
public enum Face {
    X_MIN(0, false), X_MAX(0, true),
    Y_MIN(1, false), Y_MAX(1, true),
    Z_MIN(2, false), Z_MAX(2, true);

    private static final Face[] ALL = values();

    private final int axis;
    private final boolean max;

    Face(int axis, boolean max) {
        this.axis = axis;
        this.max = max;
    }

    /** @return 0, 1 or 2 for the x, y or z axis (the first/second/third declared domain variable). */
    public int axis() {
        return axis;
    }

    /** @return true for the upper face of the axis. */
    public boolean isMax() {
        return max;
    }

    /** @return the face on the other side of the same axis. */
    public Face opposite() {
        return ALL[axis * 2 + (max ? 0 : 1)];
    }

    /** @return the face for the given axis and side. */
    public static Face of(int axis, boolean max) {
        return ALL[axis * 2 + (max ? 1 : 0)];
    }
}
