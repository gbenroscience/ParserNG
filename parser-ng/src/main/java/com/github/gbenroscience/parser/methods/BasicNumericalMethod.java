package com.github.gbenroscience.parser.methods;

import java.util.List;

public interface BasicNumericalMethod {

    String solve(List<String> tokens);

    String getHelp();

    String getName();

    String getType();

}
