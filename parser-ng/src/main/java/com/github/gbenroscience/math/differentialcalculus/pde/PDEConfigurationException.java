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
 * Thrown when a PDE problem is mathematically or structurally invalid:
 * missing boundary/initial conditions, unknown symbols, unsupported equation
 * forms, incompatible method selections, and similar.
 */
public class PDEConfigurationException extends PDEException {

    private static final long serialVersionUID = 1L;

    public PDEConfigurationException(String message) {
        super(message);
    }

    public PDEConfigurationException(String message, Throwable cause) {
        super(message, cause);
    }
}
