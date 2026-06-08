package com.github.gbenroscience.parser.pro.spi;


import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.spi.TurboEvaluatorProvider;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import com.github.gbenroscience.parser.pro.turbo.tools.VectorTurboEvaluator; // The SIMD class we wrote

/**
 *
 * @author GBEMIRO
 */ 
public class ProTurboEvaluatorProvider implements TurboEvaluatorProvider {

    @Override
    public boolean isVectorHardwareSupported() {
        try {
            // Verify that the incubating module was actually loaded by the JVM
            Class.forName("jdk.incubator.vector.DoubleVector");
            return true;
        } catch (ClassNotFoundException e) {
            System.err.println("ParserNG Pro Error: Vector API not found. Please add '--add-modules jdk.incubator.vector' to your JVM arguments.");
            return false;
        }
    }

    @Override
    public TurboExpressionEvaluator getVectorEvaluator(MathExpression me) {
        try {
            return new VectorTurboEvaluator(me);
        } catch (Throwable ex) {
            System.getLogger(ProTurboEvaluatorProvider.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
            return null;
        }
    }
}