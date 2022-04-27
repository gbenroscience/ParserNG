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
        sb.append("See: https://github.com/gbenroscience/ParserNG").append("\n");
        return sb.toString();
    }

    public static String getHelp(String op) {
        for(BasicNumericalMethod basicNumericalMethod: Declarations.getBasicNumericalMethods()){
            if (op.equals(basicNumericalMethod.getName())){
                return basicNumericalMethod.getHelp();
            }
        }
        switch (op) {
            case SUM:
                return toLine(op, "Returns sum of all values. Eg: " + op + "(1,2) will evaluate as 1+2 to 3");
            case PROD:
                return toLine(op, "product of all elements. s. Eg: " + op + "(1,2) will evaluate 1*2 to 2");
            case GEOM:
                return toLine(op, "returns geometrical average of elements. s. Eg: " + op + "(1,2,4) will evaluate (1*2*4)^1/3 thus na 8^1/3 thus to 2");
            case GSUM:
                return toLine(op, "product of all elements. s. Eg: " + op + "(1,2) will evaluate 1*2 to 2");
            case COUNT:
                return toLine(op, "number of elements in the function. Eg: " + op + "(1,1.1) will evaluate 3");
            case AVG:
                return toLine(op, "avarage of all elements. s. Eg: " + op + "(1,2) will evaluate (1*2)/2 to 0.5");
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

}
