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
package com.github.gbenroscience.simd.turbo.tools.utils;


import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * @author GBEMIRO
 */
public final class VectorConfig {
    // Single source of truth for the entire engine
    public static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    public static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;
    
    
    public static final VectorSpecies<Byte> B_SPECIES = ByteVector.SPECIES_PREFERRED;
    public static final VectorSpecies<Integer> I_SPECIES = IntVector.SPECIES_PREFERRED;

    public static final VectorSpecies<Short> S_SPECIES = ShortVector.SPECIES_PREFERRED;
    
    // Derived constants for quick inline compilation references
    public static final int VLEN = SPECIES.length();
    // Derived constants for quick inline compilation references
    public static final int VF_LEN = F_SPECIES.length();
    
    private VectorConfig() {} // Non-instantiable
}