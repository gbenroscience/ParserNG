package com.github.gbenroscience.parser.logical;

import com.github.gbenroscience.parser.logical.ExpressionLogger;

public class PrintingExpressionLogger implements  ExpressionLogger {

    private final boolean tests_verbose = true;

    @Override
    public void log(String s) {
        if (tests_verbose) {
            System.out.println(s);
        }
    }
}
