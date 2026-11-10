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
 * A structured array layout with a one-cell ghost layer around every ACTIVE
 * axis (2 or 3), x-fastest, stored in one flat {@code double[]}. Shared by
 * the multigrid and the flow solver so fields can be handed between them
 * without copying.
 */
final class Box {

    final int dim;
    final int[] n = new int[3];
    final int[] st = new int[3];
    final int g2;       // ghost width of axis 2 (0 in 2-D)
    final int size;

    Box(int dim, int n0, int n1, int n2) {
        this.dim = dim;
        this.n[0] = n0;
        this.n[1] = n1;
        this.n[2] = dim == 3 ? n2 : 1;
        this.g2 = dim == 3 ? 1 : 0;
        this.st[0] = 1;
        this.st[1] = n0 + 2;
        this.st[2] = (n0 + 2) * (n1 + 2);
        this.size = (n0 + 2) * (n1 + 2) * (this.n[2] + 2 * g2);
    }

    /** @return flat index of cell/face (i,j,k); ghost cells use -1 and n. */
    int idx(int i, int j, int k) {
        return (i + 1) + (j + 1) * st[1] + (k + g2) * st[2];
    }
}
