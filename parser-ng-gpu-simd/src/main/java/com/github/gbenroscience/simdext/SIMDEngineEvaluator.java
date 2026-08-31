package com.github.gbenroscience.simdext;

import com.github.gbenroscience.parser.MathExpression;
 
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;
 
/**
 * This class is maintained for backwards compatibility for existing users.
 * @deprecated Do not use this class again. The sheer size of its methods make
 * escape analysis fail(by the JVM) and its otherwise blazing speeds is lost
 * when this happens. Use the classes of the
 * {@link com.github.gbenroscience.simdext.turbo.tools.command.v2} package. You
 * will find float32 and float64 evaluators and MemorySegment evaluators of the
 * float32 AND float64(double) KIND there 
 */
@Deprecated
public class SIMDEngineEvaluator extends SIMDCommandF64 {

    public SIMDEngineEvaluator(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDEngineEvaluator(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }

}
