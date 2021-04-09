/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.numericalmethods;

import parser.Number;
import parser.Bracket;
import parser.Variable;
import java.util.InputMismatchException;
import parser.Function;
import parser.LISTS;
import parser.MathExpression;
import parser.Operator;
import util.FunctionManager;
import static java.lang.Math.*;
import java.util.Arrays;
import java.util.List;
import math.differentialcalculus.Derivative;
import util.VariableManager;


/**
 *
 * Objects of this class are used
 * to solve for the roots of non-implicit
 * equations. They combine a variety of
 * algorithms in an if-fail-switch-algorithm
 * fashion to iteratively deduce the roots.
 *<ol>
 * <li>
 * <b>The Secant Algorithm.</b>
 * <p>
 * The first algorithm is the secant algorithm.
 * It requires 2 values of x between which it
 * seeks out the root.
 * If it does not find it in the specified range however,
 * it may search outside the range. 
 * 
 * If the secant fails, the object automatically
 * switches over to another of its methods...
 * </p>
 * </li>
 * <li>
 * <b> The Bisection Algorithm.</b>
 * <p>
 * This algorithm searches for a root ONLY
 * within the specified range and returns
 * one if it exists.
 * However if it fails, the object again automatically
 * switches over to an highly unpredictable algorithm
 * called here: 
 * </p>
 * </li>
 * <li>
 * <b> The Self-Evaluating Algorithm. </b>
 * <p>
 * Two variants of these are used here and due to its
 * unstable nature,it is the algorithm of last resort.
 * It searches for a root starting at the first limit specified for
 * x but the direction of search is not guaranteed due to
 * its instability.
 * </p>
 * </li>
 *  
 * </ol>
 * If both flavors of this algorithm fail, then an error
 * report is generated.
 *
 * The search for the root is intensive and
 * if no root is found, it is usually because no real
 * root exists for the function in the specified range.
 *
 *<br><br>
 *<b>Usage:</b><br>
 * The input that initializes objects of this class is a String value
 * that contains information about the function whose roots we seek
 * and the range in which we need to search for the function.
 * Always specify 2 values for the range,please.
 * If the variable has been initialized before in the workspace
 * and is visible to the currently evaluating object
 * of this class then an example could be:<br>
 * <b>
 * 2x^3-5x+sin(x)-1=0,-3,5
 * </b><br>
 * This will try to seek out the zeroes of 2x^3-5x+sin(x)-1
 * between x = -3 and x = 5 depending on the algorithm in use.
 *
 * If however, the variable has not been initialized before in the workspace
 * or has been initialized but is not visible to the currently evaluating object
 * of this class then an example could be:<br>
 * <b>
 * x=0;2x^3-5x+sin(x)-1=0,-3,5
 * </b><br>
 *
 * <b>
 * <strong color="red">
 * CAUTION!!!!!
 * Always end your equations with "=0"<br>
 * Objects of this class assume that you do
 * and make calculations based on that.
 * </strong>
 * </b>
 *
 *
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class RootFinder {

    /**
     * The equation whose zeroes are desired
     */
    private Function function;


    private double x1;

    private double x2;
    /**
     *
     * @param expression An input expression containing information
     * about the function whose roots are needed.
     * e.g.
     * var x=0;//initialization
     * root(@(x)3sin(x)-4x+1,10)
     *
     */
    public RootFinder(String expression) {
        setFunction( parseFunction(expression) );
    }//end constructor
    /**
     *
     * @param function A String to be used to initialize
     * the Function attribute of this class.
     * It could be anonymous e.g @(x)sin(x+1)
     * or be the name of a pre-defined function.
     * @param x1 A starting value for iteration...will automatically be set to the smaller value of the boundaries.
     *
     *
     */
    public RootFinder(String function, double x1) {
        try {
            this.function = new Function(function);
            this.x1 = x1;
            this.x2 = x1 + 2;
        }
        catch (Exception ex) {
        }
    }//end constructor
    
    
    /**
     *
     * @param function A String to be used to initialize
     * the Function attribute of this class.
     * It could be anonymous e.g @(x)sin(x+1)
     * or be the name of a pre-defined function.
     * @param x1 A starting value for iteration...will automatically be set to the smaller value of the boundaries.
     *
     *
     */
    public RootFinder(Function function, double x1) {
        try {
            this.function = function;
            this.x1 = x1;
            this.x2 = x1 + 2;
        }
        catch (Exception ex) {
        }
    }//end constructor

    public void setX1(double x1) {
        this.x1 = x1;
    }

    public double getX1() {
        return x1;
    }

    public void setX2(double x2) {
        this.x2 = x2;
    }

    public double getX2() {
        return x2;
    }





    /**
     * Method that processes the format that
     * this software will recognize for
     * user input of an integral expression.
     *
     * The general format is:
     *
     *  expression,lowerLimit,upperLimit,iterations(optional)
     *
     * <br>
     * <b color='red'>
     *  Actually, the lower-limit and upper-limit values specified
     *  does not guarantee that the root returned will be between the values specified.
     *  In fact, the root could be far away from the values specified.
     * <br><br><br><br>
     * </b>
     *
     *
     * Example...
     * Using anonymous functions:
     * @(x)sin(3x-5),2,5.//assuming default number of iterations which will be computed automatically
     * @(x)sin(3x-5),2,5,50.//specifies 50 iterations.
     *
     * Using defined functions:
     * F,2,5.//assuming default number of iterations which will be computed automatically
     * F,2,5,50.//specifies 50 iterations.
     *
     * Please ensure that the function
     * is continuous in the specified range.
     *
     * @param expression  The expression
     * containing the function to integrate and
     * the lower and upper boundaries of
     * integration.
     *
     * @return an array containing::
     *  At index 0.....the expression to expand.
     *  At index 1.....the lower limit of expansion.
     *  At index 2.....the upper limit of expansion.
     *  At index 3(optional)...the degree of the polynomial up to
     * which the function should be expanded.
     *
     */

    public Function parseFunction(String expression){

        expression = expression.trim();
        if(expression.startsWith("root(") && expression.endsWith(")")){
            expression=expression.substring(expression.indexOf("(")+1);
            expression =expression.substring(0,expression.length()-1);//remove the last bracket
            double args[] = new double[2];
//The expression should look like...function,x1,x2,iterations(optional)
            int lastCommaIndex=expression.lastIndexOf(",");
            try{
                args[1] = Double.parseDouble( new MathExpression(expression.substring(lastCommaIndex+1).trim()).solve() );
                expression = expression.substring(0,lastCommaIndex).trim();
            }//end try
            catch(NumberFormatException numErr){
                throw new InputMismatchException("SYNTAX ERROR!");
            }//end catch

            lastCommaIndex=expression.lastIndexOf(",");
            try{
                args[0] = Double.parseDouble( new MathExpression(expression.substring(lastCommaIndex+1).trim()).solve() );
                expression = expression.substring(0,lastCommaIndex).trim();
            }//end try
            catch(NumberFormatException numErr){
                throw new InputMismatchException("SYNTAX ERROR!");
            }//end catch

/**
 * test for a third argument and
 * report an error if one is found,
 * else exit test quietly.
 */
            lastCommaIndex=expression.lastIndexOf(",");
            try{
                args[0] = Double.parseDouble( new MathExpression(expression.substring(lastCommaIndex+1).trim()).solve() );
                expression = expression.substring(0,lastCommaIndex).trim();
                throw new InputMismatchException(" Max of 2 args allowed! ");
            }//end try
            catch(NumberFormatException numErr){

            }//end catch
            catch(IndexOutOfBoundsException indErr){

            }//end catch

            setX1(args[0]);
            setX2(args[1]);
            try{
                if(expression.startsWith("@")){
                    //expression="anon=".concat(expression);
                    return new Function(expression);
                }
                else if (Variable.isVariableString(expression)){
                    return FunctionManager.lookUp(expression);
                }//end else if
                else{
                    throw new InputMismatchException("ALGEBRAIC_EXPRESSION SYNTAX ERROR");
                }
            }//end try
            catch(IndexOutOfBoundsException boundsException){

            }//end catch


        }//end if
        else if( !expression.startsWith("root(")){
            throw new InputMismatchException("Invalid Function Root Expression!");
        }
        else if( !expression.endsWith(")")){
            throw new InputMismatchException("Missing Closing Parenthesis");
        }
        throw new InputMismatchException("INVALID ALGEBRAIC_EXPRESSION");
    }//end method



    /**
     * Analyzes the list and extracts the Function string from it.
     * @param list The list to be analyzed.
    Direct examples would be:
    root(@sin(x+1),4,7)
    root(F,4,7) where F is a function that has been defined before in the workspace.. and so on.
     *
     * Simplifies the list to the form intg(funcName,x1,x2) or intg(funcName,x1,x2,iterations)
     *
     */
    public static void extractFunctionStringFromExpression(List<String> list){
        list.removeAll(Arrays.asList(","));

        String args1,args2,args3;
        if(list.get(0).equals("root") && list.get(1).equals("(") && list.get(list.size()-1).equals(")")){

            String functionName = list.get(2);
            if(Variable.isVariableString(functionName)){

                boolean exists = FunctionManager.contains(functionName);

                if(exists){

                    for(int i=3;i<list.size();i++){

                        if(Operator.isOpeningBracket(list.get(i))){
                            int closeBracket = Bracket.getComplementIndex(true, i, list);
                            args1 = new MathExpression(LISTS.createStringFrom(list, i, closeBracket+1)).solve();
                            List l = list.subList(i, closeBracket+1);
                            l.clear();
                            l.add(args1);
                        }
                        else if(Variable.isVariableString(list.get(i))){
                            String val = VariableManager.getVariable(list.get(i)).getValue();
                            list.set(i, val);
                        }



                    }


                }


            }

            args1 = list.get(3);
            //args2 = list.get(4);
            //args3 = list.get(5);
            if(!Number.validNumber(args1)){
                list.clear();
            }
        }

    }//end method



    /**
     *
     * @return true if the equation is valid.
     */
    public boolean isValid(){
        return function.getMathExpression().getExpression().length() > 0;
    }

    public void setFunction(Function function) {
        this.function = function;
    }

    public Function getFunction() {
        return function;
    }



    public String getVariable() {
        return function.getIndependentVariables().get(0).getName();
    }

    /**
     * This method starts with the secant algorithm and if this fails,
     * switches to the bisection algorithm, which if it does not succeed
     * switches to a form of the self-evaluating algorithm.
     * If this final attempt does not return a result after about 2000 counts,
     * it switches to another form of the same algorithm
     * which will run for another 2000 counts
     * and if it fails, will deliver an error report or
     * switch to another iterative method when this becomes available.
     * @return the root of the equation.
     *
     */
    public String findRoots(){


        String ans = new Newtonian().findRoot();

        double val = Double.valueOf( function.evalArgs(function.getDependentVariable().getName()+"("+ans+")") );
        System.err.println("Using Newton's Method: f("+ans+") = "+val);

        if( approxEqualsZero(val) || lenientApproxEqualsZero(val)  ){
            return ans;
        }
        else{
            return "Range Error!";
        }
    }




    private class Bisection{




        /**
         * Workaholic method that evaluates the roots by bisection.
         * @return the root of the equation.
         */
        private String findRoot(){

            String variable = getVariable();
            double f1 = 0.0;
            double f2 = 0.001;
            double f_middle = 0.0;

            int count = 0;
            double xOne =x1;
            double xTwo =x2;

            double x_middle=0;



            boolean switchToOtherMethod = false;
//(abs(df_x) >= 0.0 && abs(f_x)!=abs(df_x) && count < 2000)
//abs( x2 - x1 ) >= 5.0E-16
            while(abs(f2) >= 0.0 && abs(f1)!=abs(f2)  && count < 2000 ){

                x_middle = 0.50*(xOne+xTwo);

                function.getMathExpression().setValue(variable, String.valueOf(xOne));
                f1 = Double.valueOf( function.getMathExpression().solve() );

                function.getMathExpression().setValue(variable, String.valueOf(xTwo));
                f2 = Double.valueOf( function.getMathExpression().solve() );

                function.getMathExpression().setValue(variable, String.valueOf(x_middle));
                f_middle = Double.valueOf( function.getMathExpression().solve() );


                boolean sign_middle = f_middle<0;

                boolean sign_x1 = f1<0;

                boolean sign_x2 = f2<0;

                if(  sign_x1 != sign_x2    ){

                    if( sign_middle == sign_x1 ){
                        xOne = x_middle;
                    }
                    else if(   sign_middle == sign_x2   ){
                        xTwo = x_middle;
                    }

                }//end if

                else if( sign_x1 == sign_x2  ){
                    switchToOtherMethod = true;
                    break;
                }//end else if



                ++count;

            }//end while

//if rootfinding by bisection fails, switch to an alternative method.
            if(switchToOtherMethod){
                SelfEvaluator selfEvaluator = new SelfEvaluator();
                try{
                    return selfEvaluator.findRoot();
                }//end try
                catch(NumberFormatException numErr){
                    throw new NullPointerException("NO SOLUTION FOUND!");
                }
            }
            else{
                return String.valueOf(x_middle);
            }

        }


    }//end private class



    private class Newtonian{




        public String findRoot(){
            //find the root nearest to x1.
            double xOne = x1;//save the value of x1 in this variable.

            String variable = getVariable();

            //System.err.println(" function to differentiate: "+function.expressionForm());
            String gradFunxn =   Derivative.eval( "diff("+function.expressionForm()+",1)" );

            //System.err.println("gradient function is "+gradFunxn);
            Function gradFunc = new Function("@("+variable+")"+gradFunxn);
            //System.err.println("gradient function is "+gradFunc.expressionForm());

            function.getMathExpression().setDRG(1);
            gradFunc.getMathExpression().setDRG(1);

            function.getMathExpression().setValue(variable, String.valueOf(xOne));
            double f_x = Double.parseDouble(  function.eval() );

            gradFunc.getMathExpression().setValue(variable, String.valueOf(xOne));
            double df_x = Double.parseDouble(  gradFunc.eval() );

            double x = xOne;
            int count = 0;


            boolean iterationfailed = false;
            double ratio = 0;
            //iterate
            while( ( !approxEqualsZero(abs(ratio = f_x/df_x)) && count < 2000 ) ){

                try{
                    x = x - ratio;


                    function.getMathExpression().setValue(variable, String.valueOf(x));
                    f_x = Double.parseDouble(  function.eval() );

                    gradFunc.getMathExpression().setValue(variable, String.valueOf(x));
                    df_x = Double.parseDouble(  gradFunc.eval() );

                    ++count;
                    System.err.println(variable+" = "+x+" at count = "+count);
                }
                catch(Exception nfe){
                    nfe.printStackTrace();
                    iterationfailed = true;
                    break;
                }

            }//end while




// did loop terminate because count >= 2000? if so then the root finder failed again....
            //so switch to something else. Place an alternative method in this loop.
            if (count >= 2000 || iterationfailed ){
                return new Secant().findRoot();
            }

            return String.valueOf(x);
        }


    }//end class Secant


    private class Secant{




        public String findRoot(){
            //find the root nearest to x1.
            double xOne = x1;//save the value of x1 in this variable.
            double xTwo = x2;//save the value of x1 in this variable.

            String variable = getVariable();

            function.getMathExpression().setDRG(1);
            function.getMathExpression().setValue(variable, String.valueOf(xOne));
            System.err.println("xOne = "+xOne+", expression: "+function.getMathExpression().getExpression());
            double f1 = Double.parseDouble(  function.eval() );

            function.getMathExpression().setValue(variable, String.valueOf(xTwo));
            double f2 = Double.parseDouble(  function.eval() );

            double x = 0.5*(xOne+xTwo);
            int count = 0;


            boolean iterationfailed = false;
            //iterate
            while( ( !approxEqualsZero(abs(f2)) && !approxEqualsZero(abs(f1))  && count < 2000 ) ){

                try{
                    x = xTwo - (f2*((xTwo-xOne)/(f2-f1)));

                    xOne = xTwo;
                    xTwo =  x;

                    function.getMathExpression().setValue(variable, String.valueOf(xOne));
                    f1 = Double.parseDouble(  function.getMathExpression().solve() );
                    function.getMathExpression().setValue(variable, String.valueOf(xTwo));
                    f2 = Double.parseDouble( function.getMathExpression().solve()  );
                    ++count;

                }
                catch(NumberFormatException nfe){
                    nfe.printStackTrace();
                    iterationfailed = true;
                    break;
                }

            }//end while




// did loop terminate because count >= 2000? if so then the root finder failed again....
            //so switch to something else. Place an alternative method in this loop.
            if (count >= 2000 || iterationfailed ){
                return new Bisection().findRoot();
            }

            return String.valueOf(x);
        }


    }//end class Secant


    private class SelfEvaluator{


        /**
         *
         * @return the root, if it finds one.
         */
        public String findRoot(){

            double x1backup = x1;//save the value of x1 in this variable.


            String variable = getVariable();
            function.getMathExpression().setValue(variable, String.valueOf(x1));
            double f1 = Double.parseDouble(  function.getMathExpression().solve() );

            int count = 0;
            while( abs(f1) >= 5.0E-16 &&count < 2000 ){

                function.getMathExpression().setValue(variable, String.valueOf(x1));
                f1 = Double.parseDouble(  function.getMathExpression().solve() );
                x1 = x1 - ( f1/ ( f1 - 8 ) );

                ++count;
            }//end while

            // did loop terminate because count >= 2000? if so then the root finder failed
            //so switch to something else. Place an alternative method in this loop.
            if( count >= 2000 && abs(f1) >= 5.0E-16 ){
                count = 0;
                x1 = x1backup;

                function.getMathExpression().setValue(variable, String.valueOf(x1));
                while( abs(f1) >= 5.0E-16 &&count < 2000 ){

                    function.getMathExpression().setValue(variable, String.valueOf(x1));
                    f1 = Double.parseDouble(  function.getMathExpression().solve() );
                    x1 = x1 - ( f1/ ( f1 + 8 ) );

                    ++count;
                }//end while
                // did loop terminate because count >= 2000? if so then the root finder failed again....
                //so switch to something else. Place an alternative method in this loop.
                if( count >= 2000 && abs(f1) >= 5.0E-16 ){

                    return "NO SOLUTION FOUND\n TRY SPECIFYING BOUNDS\n"
                            + "BETWEEN WHICH A ROOT IS KNOWN TO EXIST.";
                }//end if

            }
            return String.valueOf(x1);
        }//end method

    }// end class


    public boolean approxEqualsZero( double number ){
        return abs(number)<=5.0e-16;
    }
    public boolean lenientApproxEqualsZero( double number ){
        return abs(number)<=5.0e-11;
    }



    public static void main( String args[]){

        FunctionManager.add("f=@(p)3*p^3+2*p^2-8*p+1");
        //RootFinder finder = new RootFinder("root(f,2,4)");

        RootFinder finder = new RootFinder("root(f,-0.4,0.11)");

        System.out.println( finder.findRoots() );
    
    }//end method





}//end class