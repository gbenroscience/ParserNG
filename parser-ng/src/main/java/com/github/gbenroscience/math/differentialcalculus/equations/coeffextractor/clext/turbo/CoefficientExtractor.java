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

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.ExprNodeCompiler;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.refactor.EquationDivider;
import com.github.gbenroscience.parser.MathExpression.Token;

import java.lang.invoke.MethodHandle;
/**
 *
 * @author GBEMIRO 
 * ParserNG Turbo tier: the {@link TurboEquationCoefficientResolver}
 * implementation {@link TurboEquationRuntime} was built to receive.
 * <p>
 * Identical algorithm to {@link CoefficientExtractor} — both share {@link
 * EquationDivider} verbatim for the symbolic work (term splitting,
 * linearity checking, the frame-ordering fix), since none of that touches
 * {@code ODEFunction} or {@code MethodHandle} at all. This class differs
 * from {@link CoefficientExtractor} in exactly one line: the divided tree
 * is compiled via {@link ExprNodeCompiler#compileTurbo(ExprNode)} instead
 * of {@code compileStandard}.
 */
public final class CoefficientExtractor {

    private CoefficientExtractor() {
    }

    public static ResolvedEquation resolve(Token[] equationPostfix, int order) {
        EquationDivider.Divided divided = EquationDivider.divide(equationPostfix, order);
        MethodHandle topDerivativeRealFrame = ExprNodeCompiler.compileTurbo(divided.tree);
        return new ResolvedEquation(
                topDerivativeRealFrame, divided.tree, divided.canonicalToReal, divided.realFrameSize); 
    }
}