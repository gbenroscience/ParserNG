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
package com.github.gbenroscience.arrow;

import com.github.gbenroscience.simdext.turbo.tools.command.temp.SIMDCommandF64; 
import java.util.Arrays;
import java.util.Random;
/**
 *
 * @author GBEMIRO
 */
public class ParserNgArrow {

    public static void main(String[] args) throws Throwable {
        SIMDCommandF64.SIMDVectorCompositeExpression see = SIMDCommandF64.getEvaluator("3*x^2+sin(x^3)");
        double[] in = new double[10000];
        double[] out = new double[10000];
 
        Random r = new Random(System.currentTimeMillis());
        for (int i = 0; i < in.length; i++) {
            in[i] = r.nextDouble();
        }
        see.applyBulk(in, out);
        System.out.println("Hello World! - out - \n" + Arrays.toString(out));
    }
}
