package com.github.gbenroscience.simdext;


import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.temp.SIMDCommandF64;
 

/**
 * @deprecated This class's former functionality is being expanded and now exists as separate classes
 * in the {@link com.github.gbenroscience.simdext.turbo.tools.command} package.
 * In that package, we now have command based evaluators for doubles and floats and MemorySegments of doubles and floats
 * ParserNG's direction as regards bulk evaluation has shifted from the SIMDEngineEvaluator whose design 
 * unfortunately causes it to fail the JVM's escape analysis and so cause allocation blowout which 
 * causes GC triggers and low speeds.
 * 
 * This class will be phased out any version from now
 * 
 * The direction now leans towards JIT styled pre-compiled commands and arithmetic ops fusion.
 * The fusion helps ParserNG beat Janino by up to 1.5x on the development box on some arithmetic ops(x+y+z) which was hitherto impossible, 
 * a Dell Inspiron 5759 with Processor: Intel(R) Core(TM) i7-6500U CPU 2.50GHz (4 CPUs), 2.6GHz Memory: 16384MB RAM
 * 
 * High-Performance Vector API & Engine that fuses explicit SIMD vectorization
 * with a zero-allocation primitive stack interpreter. Completely eliminates the
 * scalar parser overhead and task object allocations on the hot path.
 *
 * This version is the second fastest of all the SIMD evaluators.
 * Combines near zero-allocation with parallel operations greatly enhanced with cpu-pinning.
 * Cpu pinning is the reason why this class is a native of this extension and is the main reason
 * why this extension is JDK22+
 * Note that CPU PINNING works best on Linux, so the worker efficiency of these classes
 * is best seen on Linux. Where 2 workers perform at almost 2x the rate of one worker.. usually between 1.88x to 2.02x
 * 
 *
 */ 
@Deprecated
public class SIMDCommandEvaluator extends SIMDCommandF64 {

    public SIMDCommandEvaluator(MathExpression me) throws Throwable {
        super(me);
    }

    public SIMDCommandEvaluator(MathExpression me, int numWorkers) throws Throwable {
        super(me, numWorkers);
    }

}