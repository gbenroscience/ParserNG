package com.github.gbenroscience.simd.turbo.tools;

 
/**
 * @author GBEMIRO
 * An internal control-flow signal thrown during AST compilation 
 * to indicate that a function should be routed to a FlatMatrix kernel
 * instead of being compiled into a standard lane-by-lane vector tree.
 */
public class KernelInterceptException extends RuntimeException {
    
    private final String kernelName;
    private final int expectedArity;

    public KernelInterceptException(String kernelName, int expectedArity) {
        super("Intercepted Matrix Kernel: " + kernelName);
        this.kernelName = kernelName;
        this.expectedArity = expectedArity;
    }

    public String getKernelName() {
        return kernelName;
    }

    public int getExpectedArity() {
        return expectedArity;
    }
    
    // Performance Optimization: 
    // Since this exception is used for control-flow and not error logging,
    // we override fillInStackTrace() to avoid the massive performance penalty
    // of generating a stack trace during the compilation phase.
    @Override
    public synchronized Throwable fillInStackTrace() {
        return this;
    }
}
