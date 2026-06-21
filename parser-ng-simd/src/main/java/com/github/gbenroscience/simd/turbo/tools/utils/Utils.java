package com.github.gbenroscience.simd.turbo.tools.utils;

import jdk.incubator.vector.*;

/**
 *
 * @author GBEMIRO
 */
public class Utils {

    public static IntVector toInt(Vector<?> v) {
        return (IntVector) v;
    }

    public static FloatVector toFloat(Vector<?> v) {
        return (FloatVector) v;
    }

    public static ByteVector toByte(Vector<?> v) {
        return (ByteVector) v;
    }
    
    
    
}
