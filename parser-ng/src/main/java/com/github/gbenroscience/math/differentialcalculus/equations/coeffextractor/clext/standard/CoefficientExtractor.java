package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard;

import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.ExprNodeCompiler;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.standard.ResolvedEquation;
import com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common.EquationDivider; 
import com.github.gbenroscience.math.differentialcalculus.equations.standard.ODEFunction;
import com.github.gbenroscience.parser.MathExpression.Token;

/**
 * ParserNG Standard tier: turns one isolated general-form equation (already
 * {@code LHS-RHS}, implicitly {@code = 0}) into a {@link ResolvedEquation} —
 * the divided top-derivative {@code ODEFunction}, its tree, and the
 * real-frame mapping. This is the {@link EquationCoefficientResolver}
 * implementation {@link EquationRuntime} was built to receive.
 * <p>
 * All the actual symbolic work — term splitting, linearity checking, the
 * frame-ordering fix — lives in {@link EquationDivider}, shared verbatim
 * with the Turbo tier ({@code TurboCoefficientExtractor}); this class only
 * adds the one Standard-specific step: compiling the divided tree via
 * {@link ExprNodeCompiler#compileStandard(ExprNode)}.
 */
public final class CoefficientExtractor {

    private CoefficientExtractor() {
    }

    public static ResolvedEquation resolve(Token[] equationPostfix, int order) {
        EquationDivider.Divided divided = EquationDivider.divide(equationPostfix, order);  
        ODEFunction topDerivativeRealFrame = ExprNodeCompiler.compileStandard(divided.tree); 
        return new ResolvedEquation(
                topDerivativeRealFrame, divided.tree, divided.canonicalToReal, divided.realFrameSize);
    }
}