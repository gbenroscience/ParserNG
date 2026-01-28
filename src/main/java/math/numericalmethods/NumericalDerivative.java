/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.numericalmethods;

import parser.Function;
import parser.MathExpression;
import parser.Variable;
import java.util.InputMismatchException;
import java.util.logging.Level;
import java.util.logging.Logger;
import math.differentialcalculus.Derivative;
import util.FunctionManager;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class NumericalDerivative {
    /**
     * The function to differentiate.
     */
    private Function function;
    /**
     * The x coordinate of the point on the function
     * where we need to find the derivative.
     */
    private double xPoint;

    public NumericalDerivative() {
    }


    /**
     *
     * @param expression An expression containing information about
     * the function whose derivative is to be evaluated and the point
     * at which the derivative is to be evaluated.
     * For example:
     * function,4....means find the derivative of the function at the
     * point where x=4 on the curve.
     * Direct examples would be:
     * diff(@(x)sin(x+1),4)
     * diff(F,5.32) and so on.
     */
    public NumericalDerivative(String expression) {
        new Parser(expression);
    }

    public NumericalDerivative(Function function, double xPoint) {
        this.function = function;
        this.xPoint = xPoint;
    }





    public void setFunction(Function function) {
        this.function = function;
    }

    public Function getFunction() {
        return function;
    }




    public void setxPoint(double xPoint) {
        this.xPoint = xPoint;
    }

    public double getxPoint() {
        return xPoint;
    }


    /**
     * @return the numerical value of the
     * derivative very near the given point.
     */
    public String findDerivativeByPolynomialExpander(){
        FunctionExpander expander = new FunctionExpander(xPoint-0.0001, xPoint+0.1, 20,FunctionExpander.DOUBLE_PRECISION, function );
        MathExpression polyDerivative = new MathExpression( expander.getPolynomialDerivative() );

        polyDerivative.setDRG(1);
        String  variable = function.getIndependentVariables().get(0).getName();
        polyDerivative.setValue(variable, String.valueOf(xPoint) );
        return polyDerivative.solve();
    }


    /**
     * @param dx The infinitesimal used to compute the
     * numerical derivative at the given xPoint
     * on the function.
     * @return the numerical value of the
     * derivative very near the given point.
     */
    public String findDerivativeByLimit(double dx){
        MathExpression func = function.getMathExpression();
        func.setDRG(1);
        func.setValue(function.getIndependentVariables().get(0).getName(), String.valueOf(xPoint+dx));
        String upper = func.solve();

        func.setValue(function.getIndependentVariables().get(0).getName(), String.valueOf(xPoint-dx));
        String lower = func.solve();

        double derived = ( Double.parseDouble(upper) -  Double.parseDouble(lower) )/(2.0*dx);
        return String.valueOf(derived);
    }//end method


    /**
     * Analyzes the expression and extracts the Function string from it.
     * @param expression The expression to be analyzed.
     * Direct examples would be:
     * diff(@sin(x+1),4)
     * diff(F,5.32) where F is a    function that has been defined before in the workspace.. and so on.
     *
     * @return an Object array containing at index 0, the function string,which may or may not be anonymous,
     * and in index 1, the horizontal coordinate of type double where the derivative is needed.
     */
    public static Object[] extractFunctionStringFromExpression( String expression ){

        expression = expression.trim();
        if( expression.startsWith("diff(")&&expression.endsWith(")")){
            try{
                expression = expression.substring(0,expression.length()-1);//remove the last bracket.
                expression = expression.substring(expression.indexOf("(")+1);//remove the starting command and the first bracket.

                double xPoint =  Double.parseDouble( new MathExpression(expression.substring(expression.lastIndexOf(",")+1).trim()).solve() ) ;
                String func = expression.substring(0, expression.lastIndexOf(",")).trim();

                try{
                    if(func.startsWith("@")){
                        return new Object[]{func,xPoint};
                    }
                    else if (Variable.isVariableString(func)){
                        return new Object[]{func,xPoint};
                    }//end else if
                    else{
                        throw new InputMismatchException("ALGEBRAIC_EXPRESSION SYNTAX ERROR");
                    }
                }//end try
                catch(IndexOutOfBoundsException boundsException){

                }//end catch
            }//end try
            catch(NumberFormatException formatException){
                throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
            }//end catch
            catch(IndexOutOfBoundsException indException){
                throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
            }//end catch
            catch(NullPointerException exception){
                throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
            }//end catch

        }//end if
        else{
            throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
        }//end else

        throw new InputMismatchException("INVALID ALGEBRAIC_EXPRESSION");
    }



    public class Parser{

        public Parser( String expression ) {
            setFunction(parseDerivativeCommand(expression));
        }//end constructor




        /**
         *
         * @param expression An expression containing information about
         * the function whose derivative is to be evaluated and the point
         * at which the derivative is to be evaluated.
         * For example:
         * function,4....means find the derivative of the function at the
         * point where x=4 on the curve.
         * Direct examples would be:
         * diff(@sin(x+1),4)
         * diff(F,5.32) where F is a    function that has been defined before in the workspace.. and so on.
         *
         *
         */

        public Function parseDerivativeCommand(String expression){
            expression = expression.trim();
            if( expression.startsWith("diff(")&&expression.endsWith(")")){
                try{
                    expression = expression.substring(0,expression.length()-1);//remove the last bracket.
                    expression = expression.substring(expression.indexOf("(")+1);//remove the starting command and the first bracket.

                    double xPoint = Double.parseDouble( expression.substring(expression.lastIndexOf(",")+1).trim() );
                    String func = expression.substring(0, expression.lastIndexOf(",")).trim();

                    setxPoint(xPoint);
                    try{
                        if(func.startsWith("@")){
                            return new Function(func);
                        }
                        else if (Variable.isVariableString(func)){
                            return FunctionManager.lookUp(func);
                        }//end else if
                        else{
                            throw new InputMismatchException("ALGEBRAIC_EXPRESSION SYNTAX ERROR");
                        }
                    }//end try
                    catch(IndexOutOfBoundsException boundsException){

                    }//end catch
                }//end try
                catch(NumberFormatException formatException){
                    throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
                }//end catch
                catch(IndexOutOfBoundsException indException){
                    throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
                }//end catch
                catch(NullPointerException exception){
                    throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
                }//end catch

            }//end if
            else{
                throw new InputMismatchException("SYNTAX ERROR! CHECK THE HELP FILE FOR THE VALID COMMAND TO USE NEAR \"diff\"");
            }//end else

            throw new InputMismatchException("INVALID ALGEBRAIC_EXPRESSION");
        }//end method



    }//end class Parser
    public static void main(String args[]){
        double evalPoint = 3.2;
        FunctionManager.add("F=@(x)sin(x)/(x^3)*sin(x)");
        FunctionManager.add("P=@(x)ln(x)");
        //NumericalDerivative der = new NumericalDerivative("diff(P,5.134)" );
        NumericalDerivative der = new NumericalDerivative(FunctionManager.lookUp("F"),evalPoint);
        System.out.println("expression = "+der.function.getMathExpression().getExpression());
        System.out.println("dependent variable = "+der.function.getDependentVariable());
        System.out.println("independent variable = "+der.function.getIndependentVariables());
        System.out.println("Derivative by polynomial expander approx: "+der.findDerivativeByPolynomialExpander());
        System.out.println("Derivative by limit approx: "+der.findDerivativeByLimit(2.0E-6));
        try {
            String expr = Derivative.eval( "diff(F,"+evalPoint+")");
            System.out.println("Absolute derivative: "+expr);
        } catch (Exception ex) {
            Logger.getLogger(NumericalDerivative.class.getName()).log(Level.SEVERE, null, ex);
        }


    }


    /**
     *
     * @return the independent variable in a 2D function
     * e.g. f(x)=sin(x). If the function is more than 2D
     * then it returns the first independent variable alone.
     */
    public String getVariable() {
        return function.getIndependentVariables().get(0).getName();
    }//end method




}//end class