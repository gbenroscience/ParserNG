package parser.methods;

import static  parser.methods.Declarations.*;
import static  parser.methods.BasicNumericalMethod.*;

public class Help {


    public static String getHelp() {
        StringBuilder sb = new StringBuilder();
        sb.append("List of currently known methods:").append("\n");
        for (String op : getInbuiltMethods()) {
            sb.append(getHelp(op)).append("\n");
        }
        sb.append("List of functions is just tip of iceberg, see: https://github.com/gbenroscience/ParserNG for all features").append("\n");
        return sb.toString();
    }

    public static String getHelp(String op) {
        for(BasicNumericalMethod basicNumericalMethod: Declarations.getBasicNumericalMethods()){
            if (op.equals(basicNumericalMethod.getName())){
                return basicNumericalMethod.getHelp();
            }
        }
        switch (op) {
            case PROD:
                return toLine(op, "product of all elements. s. Eg: " + op + "(1,2) will evaluate 1*2 to 2");
            case MEDIAN:
                return toLine(op, "median from set. s. Eg: " + op + "(1,2,3) will evaluate as 2");
            default:
                return toLine(op, "help not yet written. See https://github.com/gbenroscience/ParserNG");
        }
    }


    static String align(String op) {
        for (String s : getInbuiltMethods()) {
            while (s.length() > op.length()) {
                op = op + " ";
            }
        }
        return op;
    }

    public static String toLine(String op, String text) {
        return Help.align(op) + " - " + text;
    }

}
