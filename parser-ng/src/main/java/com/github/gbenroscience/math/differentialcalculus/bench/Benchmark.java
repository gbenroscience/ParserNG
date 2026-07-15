/*
 * Copyright 2026 oluwagbemirojiboye.
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
package com.github.gbenroscience.math.differentialcalculus.bench;

import com.github.gbenroscience.math.differentialcalculus.Derivative;
import com.github.gbenroscience.math.differentialcalculus.autodiff.AutoDiffNEvaluator;
import com.github.gbenroscience.parser.MathExpression;

/**
 *
 * @author oluwagbemirojiboye
 */
public class Benchmark {

    public static void main(String[] args) {

        String expr = "x^2*cos(x)-2*x*sin(x)-2*cos(x)";
        MathExpression.EvalResult ev = new MathExpression.EvalResult();

        double evalPoint = 4;
        int orderOfDiff = 5;
        double[] resultOut = Derivative.ThreadLocalBufferPool.getOrCreateBuffer(orderOfDiff + 1);
        AutoDiffNEvaluator adne = new AutoDiffNEvaluator(new MathExpression(expr), 20);
        double N = 10000900;
        double t = System.nanoTime();
        for (int i = 0; i < N; i++) {
            adne.evaluateRPN("x", evalPoint, orderOfDiff, resultOut);
            ev.wrap(resultOut);
        }
        t = (System.nanoTime() - t) / N;
        System.out.println("res = " + ev + ", " + t + "ns");

    }

}
