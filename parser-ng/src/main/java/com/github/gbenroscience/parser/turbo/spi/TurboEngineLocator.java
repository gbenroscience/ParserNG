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
package com.github.gbenroscience.parser.turbo.spi;

import java.util.Iterator;
import java.util.ServiceLoader;

/**
 * (The Auto-Detector): This class quietly scans the classpath. If it finds the
 * Pro module, it caches it. If not, it gracefully does nothing.
 *
 * @author GBEMIRO
 */
public class TurboEngineLocator {

    private static TurboEvaluatorProvider provider = null;
    private static boolean loaded = false;

    public static TurboEvaluatorProvider getProvider() {
        if (!loaded) {
            try {
                ServiceLoader<TurboEvaluatorProvider> loader = ServiceLoader.load(TurboEvaluatorProvider.class);
                Iterator<TurboEvaluatorProvider> it = loader.iterator();
                if (it.hasNext()) {
                    provider = it.next();
                }
            } catch (Throwable t) {
                // Failsafe: If Pro module is malformed or JVM flags are missing, we stay null.
            }
            loaded = true;
        }
        return provider;
    }

    public static boolean isSimdAvailable() {
        TurboEvaluatorProvider p = getProvider();
        return p != null && p.isVectorHardwareSupported();
    }
}
