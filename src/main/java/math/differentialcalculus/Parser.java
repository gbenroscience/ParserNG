/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import parser.Number;
import parser.Bracket;
import parser.DataSetFormatter;
import parser.Function;
import parser.LISTS;
import parser.MathExpression;
import parser.MathScanner;
import parser.Operator;
import parser.Parser_Result;
import parser.Variable;
import java.util.Arrays;
import java.util.List;
import util.FunctionManager;
import util.VariableManager;

/**
 *
 * @author JIBOYE, OLUWAGBEMIRO OLAOLUWA Parses derivative commands of the
 * format:
 *
 *
 * diff(@(x)sin(x),5)... diff(@(x)sin(x),5,2)... diff(y,5)... diff(y,5,2)...
 *
 * The first command means the function, sin(x) is to be differentiated wrt x,
 * and evaluated at x = 5.
 * The second command means the function, sin(x) is to
 * be differentiated wrt x, twice and then evaluated at x = 5.
 * The third command means that a function called y has been pre-defined. The parser will load the
 * function and differentiate it wrt x, and then evaluate it at x = 5.
 * The fourth command means that a function called y has been pre-defined. The
 * parser will load the function and differentiate it wrt x, twice, and then
 * evaluate it at x = 5.
 *
 *
 *
 *
 * Here sin(x) is to be differentiated and evaluated at x=5.
 *
 */
public class Parser {

    /**
     * The x coordinate at which the derivative is to be evaluated.
     */
    private double evalPoint = Double.POSITIVE_INFINITY;
    /**
     * The {@link Function} object created from the command
     */
    private Function function;
    /**
     * The number of times to differentiate the function.
     */
    private int orderOfDifferentiation = Integer.MAX_VALUE;
    public Parser_Result result = Parser_Result.VALID;

    /**
     * This is a very important field as it tells if the function is to be
     * differentiated and the value of its derivative at a point evaluated,
     * (This sets this field to {@link Parser#GRAD_VAL}) or perhaps it should be
     * differentiate and the gradient function itself returned (This sets this
     * field to {@link Parser#GRAD_FUNC}).
     *
     * The first type is for expressions of type diff(func,x1,x2) where x1 is
     * the point at which the derivative is to be evaluated and x2 is the number
     * of times the function should be differentiated before evaluating it at
     * x1.
     *
     * The second type is for expressions of type diff(func,x1) where x1 is the
     * number of times the function should be differentiated.
     *
     *
     */
    private int diffType = GRAD_VAL;

    public static final int GRAD_FUNC = 1;
    public static final int GRAD_VAL = 2;

    /**
     *
     * @param expression The expression to parse. e.g. diff(@(x)cos(x)+5*x,6,1)
     * or diff(F,6,1) where F is a function that has been previously defined in
     * the workspace and so is in the {@link FunctionManager} It may also take
     * the form diff(@(x)cos(x)+5*x,6) or diff(F,6) where it assumes that the
     * function is to be differentiated only once. In the future, we have
     * diff(F) or (diff(@(x)cos(x)+5*x) which will return the gradient function
     * itself.
     */
    public Parser(String expression) {

        DataSetFormatter dsf = new DataSetFormatter(expression);
        List<String> scanner = dsf.getDataset();

        MathScanner.recognizeAnonymousFunctions(scanner);


        this.function = localParseDerivativeCommand(scanner);
    }//end constructor

    public double getEvalPoint() {
        return evalPoint;
    }

    public Function getFunction() {
        return function;
    }

    public int getOrderOfDifferentiation() {
        return orderOfDifferentiation;
    }

    public void setOrderOfDifferentiation(int orderOfDifferentiation) {
        this.orderOfDifferentiation = orderOfDifferentiation;
    }

    public int getDiffType() {
        return diffType;
    }

    public void setDiffType(int diffType) {
        this.diffType = diffType;
    }

    public boolean isGradFunc() {
        return diffType == GRAD_FUNC;
    }

    public boolean isGradEval() {
        return diffType == GRAD_VAL;
    }

    /**
     *
     * @param list A list containing the scanned form of an expression
     * containing information about the function whose derivative is to be
     * evaluated and the point at which the derivative is to be evaluated.
     * Common forms for the expression are: diff(@(x)sin(x),5)...
     * diff(@(x)sin(x),5,2)... diff(y,5)... diff(y,5,2)...
     *
     * The first command means the function, sin(x) is to be differentiated wrt
     * x, and evaluated at x = 5. The second command means the function, sin(x)
     * is to be differentiated wrt x, twice and then evaluated at x = 5. The
     * third command means that a function called y has been pre-defined. The
     * parser will load the function and differentiate it wrt x, and then
     * evaluate it at x = 5. The fourth command means that a function called y
     * has been pre-defined. The parser will load the function and differentiate
     * it wrt x, twice, and then evaluate it at x = 5.
     *
     * Direct examples would be: diff(@(x)sin(x+1),4) diff(F,5.32) where F is a
     * function that has been defined before in the workspace.. and so on.
     * @return the Function object in the expression and in the process also
     * discovers the point at which the derivative is to be evaluated.
     *
     */
    private Function localParseDerivativeCommand(List<String> list) {

        list.removeAll(Arrays.asList(","));
        String args1, args2 = "";
        if (list.get(0).equals("diff") && list.get(1).equals("(") && list.get(list.size() - 1).equals(")")) {

            String functionName = list.get(2);
            if (Variable.isVariableString(functionName)) {

                boolean exists = FunctionManager.contains(functionName);

                if (exists) {

                    for (int i = 3; i < list.size(); i++) {

                        if (Operator.isOpeningBracket(list.get(i))) {
                            int closeBracket = Bracket.getComplementIndex(true, i, list);
                            args1 = new MathExpression(LISTS.createStringFrom(list, i, closeBracket + 1)).solve();
                            List l = list.subList(i, closeBracket + 1);
                            l.clear();
                            l.add(args1);
                        } else if (Variable.isVariableString(list.get(i))) {
                            String val = VariableManager.getVariable(list.get(i)).getValue();
                            list.set(i, val);
                        }

                    }

                }

            }

            if (Bracket.isCloseBracket(args1 = list.get(3))) {
                this.diffType = GRAD_FUNC;
                this.orderOfDifferentiation = 1;
            } else if (Number.validNumber(args1)) {//detect first arg
                if (Number.validNumber(args2 = list.get(4))) {//detect second arg
                    this.diffType = GRAD_VAL;
                    this.evalPoint = Double.parseDouble(args1);
                    this.orderOfDifferentiation = (int) Double.parseDouble(args2);
                } else {//second arg not available
                    this.diffType = GRAD_FUNC;
                    this.orderOfDifferentiation = (int) Double.parseDouble(args1);
                }

            }

            return FunctionManager.lookUp(functionName);
        }
        return null;
    }//end method

    /**
     *
     * @param list A list containing the scanned form of an expression
     * containing information about the function whose derivative is to be
     * evaluated and the point at which the derivative is to be evaluated.
     * Common forms for the expression are: diff(@(x)sin(x),5)...
     * diff(@(x)sin(x),5,2)... diff(y,5)... diff(y,5,2)...
     *
     * The first command means the function, sin(x) is to be differentiated wrt
     * x, and evaluated at x = 5. The second command means the function, sin(x)
     * is to be differentiated wrt x, twice and then evaluated at x = 5. The
     * third command means that a function called y has been pre-defined. The
     * parser will load the function and differentiate it wrt x, and then
     * evaluate it at x = 5. The fourth command means that a function called y
     * has been pre-defined. The parser will load the function and differentiate
     * it wrt x, twice, and then evaluate it at x = 5.
     *
     * Direct examples would be: diff(@(x)sin(x+1),4) diff(F,5.32) where F is a
     * function that has been defined before in the workspace.. and so on.
     *
     */
    public static void parseDerivativeCommand(List<String> list) {

        list.removeAll(Arrays.asList(","));

        String args1, args2 = "";
        if (list.get(0).equals("diff") && list.get(1).equals("(") && list.get(list.size() - 1).equals(")")) {

            String functionName = list.get(2);
            if (Variable.isVariableString(functionName)) {

                boolean exists = FunctionManager.contains(functionName);

                if (exists) {

                    for (int i = 3; i < list.size(); i++) {

                        if (Operator.isOpeningBracket(list.get(i))) {
                            int closeBracket = Bracket.getComplementIndex(true, i, list);
                            args1 = new MathExpression(LISTS.createStringFrom(list, i, closeBracket + 1)).solve();
                            List l = list.subList(i, closeBracket + 1);
                            l.clear();
                            l.add(args1);
                        } else if (Variable.isVariableString(list.get(i))) {
                            String val = VariableManager.getVariable(list.get(i)).getValue();
                            list.set(i, val);
                        }

                    }

                }

            }

            int sz = list.size();

            /**
             * diff,(,f,)--sz = 4 diff,(,f,1,)--sz = 5 diff,(,f,2,3,)--sz = 6
             */
            switch (sz) {

                case 4:

                    break;
                case 5:
                    args1 = list.get(3);
                    break;
                case 6:
                    args1 = list.get(3);
                    args2 = list.get(4);
                    break;
                default:
                    list.clear();
                    break;
            }

        }

    }//end method

    public static void main(String[] args) {
        FunctionManager.add(new Function("F(x)=sin(x)"));
        Parser p = new Parser("diff(F,2,5)");
        System.err.println("-----func: " + p.function);
        System.err.println("-----evalPoint: " + p.evalPoint);
        System.err.println("-----order: " + p.orderOfDifferentiation);
    }

}//end class Parser
