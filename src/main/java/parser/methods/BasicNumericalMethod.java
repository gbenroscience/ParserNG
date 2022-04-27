package parser.methods;

import java.util.List;

public interface BasicNumericalMethod {

    String solve(List<String> tokens);

    String getHelp();

    String getName();

    String getType();

    public static String toLine(String op, String text) {
        return Help.align(op) + " - " + text;
    }

}
