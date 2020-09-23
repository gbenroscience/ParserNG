/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.numericalmethods;
import parser.Number;
import parser.Bracket;
import parser.Variable;
import util.FunctionManager;
import parser.Function;
import parser.LISTS;
import parser.MathExpression;
import parser.Operator;
import java.util.InputMismatchException;
import static java.lang.Math.*;
import java.util.Arrays;
import java.util.List;
import util.VariableManager;

/**
 * Objects of this class are able
 * to perform numerical integration
 * of a curve within a given range
 * given that the function is continuous
 * throughout that range.
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class NumericalIntegral {

    /**
     * Use this to integrate using the integral symbol.
     */
    public static final int SYMBOLIC_INTEGRATION = 1;

    /**
     * Use this to integrate without using the integral symbol.
     * Here, the intg() command containing the function and the bounds are specified.
     */
    public static final int FUNCTIONAL_INTEGRATION = 2;


    /**
     * The upper boundary value of x.
     */
    private double xUpper;
    /**
     * The lower boundary value of x.
     */
    private double xLower;
    /**
     * The function to integrate.
     */
    private Function function;
    /**
     * The number of iterations.
     */
    private int iterations=0;
    /**
     *
     *
     * @param xLower The lower limit of x
     * @param xUpper The upper limit of x
     * @param iterations The number of iterations
     * @param function The name of a Function that has been defined before
     * in the WorkSpace or an anonymous Function just specified.
     */
    public NumericalIntegral(double xLower, double xUpper, int iterations, String function) {
        this.xLower = xLower;
        this.xUpper = xUpper;
        try{
            this.function = FunctionManager.lookUp(function);
        }//end try
        catch(NullPointerException npe){
            npe.printStackTrace();
            this.function=new Function(function);
        }
        if( iterations == 0 ){
            int trialValue = abs( (int) ((xUpper - xLower) / 0.01) );
            setIterations(  (trialValue < 10)?15:( ( trialValue > 30 )? 30:trialValue ) );
        }//end if
        else{
            setIterations(iterations);
        }//end else
    }
    /**
     *
     * @param expression An expression containing information about
     * the function whose integral is to be evaluated and the limits
     * of integration.
     * For example:
     * function,2,4....means integrate function between horizontal
     * coordinates 2 and 3.
     * Direct examples would be:
     * sin(x+1),3,4
     * cos(sinh(x-2/tan9x)),4,4.32 and so on.
     *
     * @param chooseExpressionType Determines if the expression to
     * integrate contains the integral symbol or not.
     *
     * F(x) = sin(x)/2x;
     * intg(F(x),0,2,iterations)
     */
    public NumericalIntegral( String expression , int chooseExpressionType ) {
        if( chooseExpressionType == SYMBOLIC_INTEGRATION ){
            new Parser(expression, chooseExpressionType);
            //no info about the number of interations specified,so set default number of iterations.
            if( iterations==0){
                //int trialValue = abs( (int) ((xUpper - xLower) / 0.01) );
                setIterations( 20 );
            }//end if
        }//end if
        else if( chooseExpressionType == FUNCTIONAL_INTEGRATION ){
            new Parser(expression, chooseExpressionType);
//no info about the number of interations specified,so set default number of iterations.
            if( iterations==0){
                //int trialValue = abs( (int) ((xUpper - xLower) / 0.01) );
                setIterations( 20 );
            }//end if
        }//end else if
        else{
            throw new InputMismatchException("Input Type Error");
        }
    }//end constructor



    /**
     * Set the number of iterations
     * and ensure that it is even.
     * @param iterations
     */
    public void setIterations(int iterations) {
        iterations = abs(iterations);
        this.iterations = (iterations%2==0)?iterations:(iterations+1);
    }
    /**
     *
     * @return the number of iterations.
     */
    public int getIterations() {
        return iterations;
    }


    public Function getFunction() {
        return function;
    }

    public void setFunction(Function function) {
        this.function = function;
    }

    public double getxLower() {
        return xLower;
    }

    public void setxLower(double xLower) {
        this.xLower = xLower;
    }

    public double getxUpper() {
        return xUpper;
    }

    public void setxUpper(double xUpper) {
        this.xUpper = xUpper;
    }
    /**
     *
     * @return the Simpson integral of the function
     *
     * Simpson's rule requires an even number of
     * rectangular strips and so an odd number of
     * ordinates(y-coordinates)
     * The x distances must also be equal.
     * h= (xUpper-xLower)/(2*m)
     */
    public String findSimpsonIntegral(){
        double m = 20000;
        double h = (xUpper - xLower)/( 2.0 * m );
        MathExpression fun = function.getMathExpression();
        fun.setDRG(1);
        String variable = function.getIndependentVariables().get(0).getName();


        fun.setValue(variable, String.valueOf(xLower));
        double first = Double.parseDouble( fun.solve() );

        fun.setValue(variable, String.valueOf(xUpper));
        double last = Double.parseDouble( fun.solve() );

        double count = 1.0;

        double sumFirstAndLast=first+last;
        double sumEven=0.0;
        double sumOdd=0.0;
        double x=0.0;


        for(;x<(xUpper-h);){
            x= xLower+count*h;
            try{
                if( (count%2) == 0 ){
                    fun.setValue(variable, String.valueOf(x));
                    sumEven += Double.parseDouble(fun.solve());
                }//end if
                else if( (count%2) == 1 ){
                    fun.setValue(variable, String.valueOf(x));
                    sumOdd += Double.parseDouble(fun.solve());
                }//end else if
                ++count;
            }//end try
            catch(NumberFormatException numExcep){
            }//end catch
        }//end for


        double sum = (h/3.0)*(sumFirstAndLast+4.0*sumOdd+2.0*sumEven);

        return String.valueOf(sum);
    }//end method

    /**
     *
     * @return the Simpson integral of the function
     *
     * Simpson's rule requires an even number of
     * rectangular strips and so an odd number of
     * ordinates(y-coordinates)
     * The x distances must also be equal.
     * h= (xUpper-xLower)/(2*m)
     */
    public String findSimpsonIntegral( double h ){

        double n = (xUpper-xLower)/(h);


        double xLower = this.xLower;
        double xUpper = xLower+Math.floor(n)*h;


        MathExpression fun = function.getMathExpression();
        fun.setDRG(1);
        String variable = function.getIndependentVariables().get(0).getName();


        fun.setValue(variable, String.valueOf(xLower));
        double first = Double.parseDouble( fun.solve() );

        fun.setValue(variable, String.valueOf(xUpper));
        double last = Double.parseDouble( fun.solve() );

        double count = 1.0;

        double sumFirstAndLast=first+last;
        double sumEven=0.0;
        double sumOdd=0.0;
        double x=0.0;

        for(;x<(xUpper-h);){
            x= xLower+count*h;
            try{
                if( (count%2) == 0 ){
                    fun.setValue(variable, String.valueOf(x));
                    sumEven += Double.parseDouble(fun.solve());
                }//end if
                else if( (count%2) == 1 ){
                    fun.setValue(variable, String.valueOf(x));
                    sumOdd += Double.parseDouble(fun.solve());
                }//end else if
                ++count;
            }//end try
            catch(NumberFormatException numExcep){
            }//end catch
        }//end for


        double mainSum = (h/3.0)*(sumFirstAndLast+4.0*sumOdd+2.0*sumEven);

        if( xUpper == this.xUpper ){
            return String.valueOf(mainSum);
        }

        else{
            xLower = xUpper;
            xUpper = this.xUpper;

            h = (xUpper-xLower)/(10.0);
            fun.setValue(variable, String.valueOf(xLower));
            first = Double.parseDouble( fun.solve() );

            fun.setValue(variable, String.valueOf(xUpper));
            last = Double.parseDouble( fun.solve() );
            count = 1.0;
            sumFirstAndLast=first+last;
            sumEven=0.0;
            sumOdd=0.0;
            x=0.0;



            for(;x<(xUpper-h);){
                x= xLower+count*h;
                try{

                    if( (count%2) == 0 ){
                        fun.setValue(variable, String.valueOf(x));
                        sumEven += Double.parseDouble(fun.solve());
                    }//end if
                    else if( (count%2) == 1 ){
                        fun.setValue(variable, String.valueOf(x));
                        sumOdd += Double.parseDouble(fun.solve());
                    }//end else if
                    ++count;
                }//end try
                catch(NumberFormatException numExcep){
                }//end catch
            }//end for

            double sum=(h/3.0)*(sumFirstAndLast+4.0*sumOdd+2.0*sumEven);

            return String.valueOf(sum+mainSum);
        }
    }//end method



    /**
     *
     * @return the integral of the
     * function using the trapezoidal rule.
     */
    public String findTrapezoidalIntegral(){
        double dx = (xUpper - xLower)/( 10000.0 );
        MathExpression fun = function.getMathExpression();
        fun.setDRG(1);
        String variable = function.getIndependentVariables().get(0).getName();
        fun.setValue(variable, String.valueOf(xLower));


        double first = Double.parseDouble(fun.solve());

        fun.setValue(variable, String.valueOf(xUpper));

        double last = Double.parseDouble(fun.solve());

        fun.setValue(variable, String.valueOf(xLower+dx));

        double y = 0.0;
        double count=0.0;
        double sum=0.0;
        double x=xLower+dx;
        for(;x<(xUpper);){
            try{
                fun.setValue(variable, String.valueOf(x));
                sum += Double.parseDouble(fun.solve());
                ++count;
                x = xLower+count*dx;
            }//end try
            catch(NumberFormatException numErr){

            }//end catch
        }
        sum+=(0.5*(first+last));

        return String.valueOf(sum*(dx));
    }

    /**
     *
     * @return the integral of the
     * function using the trapezoidal rule.
     */
    public String findTrapezoidalIntegral( double h ){

        double n = (xUpper-xLower)/(h);


        double xLower = this.xLower;
        double xUpper = xLower+Math.floor(n)*h;


        MathExpression fun = function.getMathExpression();
        fun.setDRG(1);
        String variable = function.getIndependentVariables().get(0).getName();
        fun.setValue(variable, String.valueOf(xLower));


        double first = Double.parseDouble(fun.solve());

        fun.setValue(variable, String.valueOf(xUpper));

        double last = Double.parseDouble(fun.solve());


        double count=1.0;
        double mainSum=0.0;
        double x=xLower+h;
        for(;x<(xUpper);){
            try{
                fun.setValue(variable, String.valueOf(x));
                mainSum += Double.parseDouble(fun.solve());
                ++count;
                x = xLower+count*h;
            }//end try
            catch(NumberFormatException numErr){

            }//end catch
        }
        mainSum+=(0.5*(first+last));

        if( xUpper == this.xUpper ){
            return String.valueOf(mainSum*(h));
        }//end if
        else{
            xLower=xUpper;
            xUpper=this.xUpper;

            fun.setValue(variable, String.valueOf(xLower));
            first = Double.parseDouble(fun.solve());

            fun.setValue(variable, String.valueOf(xUpper));

            last = Double.parseDouble(fun.solve());
            double sum = 0.50*(xUpper-xLower)*(first+last);


            return String.valueOf( h*mainSum+sum );
        }//end else.

    }






    /**
     *
     * @return the integral of the
     * function using the polynomial rule.
     */
    public double findPolynomialIntegral(){

        FunctionExpander expander = new FunctionExpander( xLower, xUpper,iterations,FunctionExpander.DOUBLE_PRECISION, function);
        //System.out.printf("xLower = %4.2f, xUpper = %4.2f\n",xLower,xUpper);
        MathExpression approxFunction = new MathExpression(expander.getPolynomialIntegral());

        String variable = function.getIndependentVariables().get(0).getName();
        approxFunction.setValue(variable, String.valueOf( xLower ) );

        double lower = Double.parseDouble( approxFunction.solve() );

        approxFunction.setValue(variable, String.valueOf( xUpper ) );

        double upper = Double.parseDouble( approxFunction.solve() );

        return     upper - lower;
    }
//≤≤≤≥


    /**
     *
     * Determines the integral in a given range by
     * splitting the range into sub-ranges of width that are at most
     * 0.1 units along x, and finding the polynomial curve for each sub-range.
     * @return the integral of the
     * function using the trapezoidal rule.
     */
    public double findHighRangeIntegralWithAdvancedPolynomial(){
        double dx = 0.5;

        String fName = function.getName();


        if( Math.abs( xUpper-xLower ) < dx ){
            return findAdvancedPolynomialIntegral();
        }
        else{

            double sum = 0.0;
            if(xLower<=xUpper){
                double x=xLower;
                for(;x<(xUpper-dx);x+=dx){
                    NumericalIntegral integral = new NumericalIntegral(x, x+dx,iterations, fName );
                    sum += integral.findAdvancedPolynomialIntegral();
                }//end for

                if( x < xUpper ){
                    /**
                     * This try-catch block is necessary because sometimes,
                     * x and xUpper are so close and in the case of the polynomial integral, computing it uses matrices
                     * which means that row-reduction will fail if the coefficients of the matrices are  too close
                     * due to the computational values of x and y..which are close.
                     * If such an exception occurs we can safely neglect it since it means that the area we
                     * are considering is almost infinitesimal
                     */
                    try{
                        NumericalIntegral integral = new NumericalIntegral(x, xUpper,iterations, fName );
                        sum += integral.findAdvancedPolynomialIntegral();
                    }
                    catch(Exception e){}
                }
            }

            else if(xUpper < xLower){
                double x=xLower;
                for(;x>(xUpper+dx);x-=dx){
                    NumericalIntegral integral = new NumericalIntegral(x, x-dx,iterations, fName );
                    sum += integral.findAdvancedPolynomialIntegral();
                }//end for

                if( x > xUpper ){
                    /**
                     * This try-catch block is necessary because sometimes,
                     * x and xUpper are so close and in the case of the polynomial integral, computing it uses matrices
                     * which means that row-reduction will fail if the coefficients of the matrices are  too close
                     * due to the computational values of x and y..which are close.
                     * If such an exception occurs we can safely neglect it since it means that the area we
                     * are considering is almost infinitesimal
                     */
                    try{
                        NumericalIntegral integral = new NumericalIntegral(x, xUpper,iterations, fName );
                        sum += integral.findAdvancedPolynomialIntegral();
                    }
                    catch(Exception e){}
                }

            }

            return sum;
        }
    }//end method

    /**
     *
     * Determines the integral in a given range by
     * splitting the range into sub-ranges of width that are at most
     * 0.1 units along x, and finding the polynomial curve for each sub-range.
     * @return the integral of the
     * function using the trapezoidal rule.
     */
    public double findHighRangeIntegral(){
        double dx = 0.2;

        String fName = function.getName();

        try{
            if( Math.abs( xUpper-xLower ) < dx ){
                return findGaussianQuadrature();
            }
            else{

                double sum = 0.0;
                if(xLower<=xUpper){
                    double x=xLower;
                    for(;x<(xUpper-dx);x+=dx){
                        NumericalIntegral integral = new NumericalIntegral(x, x+dx,iterations, fName );
                        sum += integral.findGaussianQuadrature();
                    }//end for

                    if( x < xUpper ){
                        /**
                         * This try-catch block is necessary because sometimes,
                         * x and xUpper are so close and in the case of the polynomial integral, computing it uses matrices
                         * which means that row-reduction will fail if the coefficients of the matrices are  too close
                         * due to the computational values of x and y..which are close.
                         * If such an exception occurs we can safely neglect it since it means that the area we
                         * are considering is almost infinitesimal
                         */
                        try{
                            NumericalIntegral integral = new NumericalIntegral(x, xUpper,iterations, fName );
                            sum += integral.findGaussianQuadrature();
                        }
                        catch(Exception e){}
                    }
                }

                else if(xUpper < xLower){
                    double x=xLower;
                    for(;x>(xUpper+dx);x-=dx){
                        NumericalIntegral integral = new NumericalIntegral(x, x-dx,iterations, fName );
                        sum += integral.findGaussianQuadrature();
                    }//end for

                    if( x > xUpper ){
                        /**
                         * This try-catch block is necessary because sometimes,
                         * x and xUpper are so close and in the case of the polynomial integral, computing it uses matrices
                         * which means that row-reduction will fail if the coefficients of the matrices are  too close
                         * due to the computational values of x and y..which are close.
                         * If such an exception occurs we can safely neglect it since it means that the area we
                         * are considering is almost infinitesimal
                         */
                        try{
                            NumericalIntegral integral = new NumericalIntegral(x, xUpper,iterations, fName );
                            sum += integral.findGaussianQuadrature();
                        }
                        catch(Exception e){}
                    }

                }


                if(sum == Double.POSITIVE_INFINITY || sum == Double.NEGATIVE_INFINITY || sum == Double.NaN){
                    return findHighRangeIntegralWithAdvancedPolynomial();
                }
                return sum;
            }
        }
        catch(Exception e){
            return findHighRangeIntegralWithAdvancedPolynomial();
        }
    }//end method


    /**
     *
     * @return The Gaussian Quadrature Version.
     */
    public double findGaussianQuadrature(){
        return Integration.gaussQuad(function, xLower, xUpper, 8);
    }

    /**
     * Algorithm that combines a variant of the Simpson rule
     * and the polynomial rule to produce higher accuracy integrals.
     */
    public double findAdvancedPolynomialIntegral(){

        double dx = ( xUpper - xLower )/(iterations);

        FunctionExpander expander = new FunctionExpander( xLower, xUpper,iterations,FunctionExpander.DOUBLE_PRECISION ,function);

        MathExpression approxFunction = new MathExpression(expander.getPolynomial());


        MathExpression fun = new MathExpression( function.getMathExpression().getExpression() );
        fun.setDRG(1);
        String variable = function.getIndependentVariables().get(0).getName();



        double sum1=  this.findPolynomialIntegral();
        double sum2=0.0;
        for(double x = xLower;x<xUpper;x+=dx){

            double x1 = ( x+(x+dx) )/2.0;

            fun.setValue( variable, String.valueOf(x1) );
            approxFunction.setValue( variable ,   String.valueOf(x1) );
            try{
                sum2+=( Double.parseDouble(approxFunction.solve()) - Double.parseDouble(fun.solve()) );
            }//end try
            catch(  NumberFormatException numErr){
            }//end catch
        }//end for
        sum1 -= ( (2.0/3.0)*sum2*(dx) );
        return sum1;
    }//end method



    /**
     * Analyzes the list and extracts the Function string from it.
     * @param list The list to be analyzed.
    Direct examples would be:
    intg(@sin(x+1),4,7)
    intg(F,4,7) where F is a function that has been defined before in the workspace.. and so on.
     *
     * Simplifies the list to the form intg(funcName,x1,x2) or intg(funcName,x1,x2,iterations)
     *
     */
    public static void extractFunctionStringFromExpression(List<String> list){
        list.removeAll(Arrays.asList(","));

        if(list.get(0).equals("quad")){
            list.set(0,"intg");
        }
        String args1,args2,args3;
        if(list.get(0).equals("intg") && list.get(1).equals("(") && list.get(list.size()-1).equals(")")){

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
            args2 = list.get(4);
            args3 = list.get(5);

            if(Number.validNumber(args1) && Number.validNumber(args2)){
                if(!Number.validNumber(args3) && !Operator.isClosingBracket(args3)){
                    list.clear();
                }
            }

        }

    }//end method



    class Parser{


        public Parser( String expression, int mode ) {

            if( mode == FUNCTIONAL_INTEGRATION ){
                setFunction(getFunctionFromIntegralCommand(expression));
            }//end if
            else if( mode == SYMBOLIC_INTEGRATION ){
                setFunction(getFunctionFromSymbolicIntegralCommand(expression));
            }//end else if
            else{
                throw new InputMismatchException("SYNTAX ERROR FOUND IN INTEGRAL EXPRESSION");
            }
        }//end constructor










        /**
         * Method that processes the format that
         * this software will recognize for
         * user input of an integral expression.
         *
         * The general format is:
         *
         *  expression,lowerLimit,upperLimit,iterations(optional)
         * e.g...
         * sin(3x-5),2,5.//assuming default number of iterations which will be computed automatically
         * sin(3x-5),2,5,50.//specifies 50 iterations.
         * Please ensure that the function
         * is continuous in the specified range.
         *
         * @param expression  The expression
         * containing the function to integrate and
         * the lower and upper boundaries of
         * integration.
         *
         * @return an array containing::
         * At index 0.....the expression to integrate
         * At index 1.....the lower limit of integration
         * At index 2.....the upper limit of integration.
         * At index 3(optional)...the number of iterations to employ
         * in evaluating this expression.
         *
         * intg( F,0,2,3 )
         * intg(@3x+1,0,2,5)
         * OR
         * intg(F(x),0,2,5)
         * OR
         * intg(F,0,2,5)
         *
         */
        public Function getFunctionFromIntegralCommand(String expression){

            expression = expression.trim();


            if(expression.startsWith("quad")){
                expression="intg"+expression.substring(4);
            }
            if(expression.startsWith("intg(") && expression.endsWith(")")){
                expression=expression.substring(expression.indexOf("(")+1);
                expression =expression.substring(0,expression.length()-1);//remove the last bracket
                double args[] = new double[3];
                args[0]=Double.NaN;
//The expression should look like...function,x1,x2,iterations(optional)
                int lastCommaIndex=expression.lastIndexOf(",");
                try{
                    args[2] = Double.parseDouble( expression.substring(lastCommaIndex+1).trim() );
                    expression = expression.substring(0,lastCommaIndex).trim();
                }//end try
                catch(NumberFormatException numErr){
                    throw new InputMismatchException("SYNTAX ERROR!");
                }//end catch

                lastCommaIndex=expression.lastIndexOf(",");
                try{
                    args[1] = Double.parseDouble( expression.substring(lastCommaIndex+1).trim() );
                    expression = expression.substring(0,lastCommaIndex).trim();
                }//end try
                catch(NumberFormatException numErr){
                    throw new InputMismatchException("SYNTAX ERROR!");
                }//end catch
                lastCommaIndex=expression.lastIndexOf(",");
                try{
                    args[0] = Double.parseDouble( expression.substring(lastCommaIndex+1).trim() );
                    expression = expression.substring(0,lastCommaIndex).trim();
                }//end try
                catch(NumberFormatException numErr){

                }//end catch
                catch(IndexOutOfBoundsException indErr){
                    throw new InputMismatchException("SYNTAX ERROR!");
                }//end catch
/**
 * test for a fourth argument and
 * report an error if one is found,
 * else exit test quietly.
 */
                lastCommaIndex=expression.lastIndexOf(",");
                try{
                    args[0] = Double.parseDouble( expression.substring(lastCommaIndex+1).trim() );
                    expression = expression.substring(0,lastCommaIndex).trim();
                    throw new InputMismatchException(" Max of 3 args allowed! ");
                }//end try
                catch(NumberFormatException numErr){

                }//end catch
                catch(IndexOutOfBoundsException indErr){

                }//end catch

                if( new Double(args[0]).isNaN() ){
                    setxLower(args[1]);
                    setxUpper(args[2]);
//setIterations(5);
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
                else if( !new Double(args[0]).isNaN() ){
                    setxLower(args[0]);
                    setxUpper(args[1]);
                    setIterations((int) args[2]);
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
                }//end else if
                else{
                    throw new InputMismatchException("Invalid Integral Expression!");
                }//end else

            }//end if
            else if( !expression.startsWith("intg(")){
                throw new InputMismatchException("Invalid Integral Expression!");
            }
            else if( !expression.endsWith(")")){
                throw new InputMismatchException("Missing Closing Parenthesis");
            }
            throw new InputMismatchException("INVALID ALGEBRAIC_EXPRESSION");
        }//end method




        /**
         *
         * Parses the format: "∫(funxn,x1,x2)dx"
         *
         * Method that processes the format that
         * this software will recognize for
         * user input of an integral expression.
         *
         * The general format is:
         *
         *  expression,lowerLimit,upperLimit,iterations(optional)
         * e.g...
         * sin(3x-5),2,5.//assuming default number of iterations which will be computed automatically
         * sin(3x-5),2,5,50.//specifies 50 iterations.
         * Please ensure that the function
         * is continuous in the specified range.
         *
         * @param expression  The expression
         * containing the function to integrate and
         * the lower and upper boundaries of
         * integration.
         *
         * @return an array containing::
         *  At index 0.....the expression to integrate
         *  At index 1.....the lower limit of integration
         *  At index 2.....the upper limit of integration.
         *  At index 3(optional)...the number of iterations to employ
         * in evaluating this expression.
         * F(x)=3x+1;
         * ∫( F,0,2,3 )dx
         * ∫(@3x+1,0,2,5)dx
         * OR
         * ∫(F(x),0,2,5)dx
         *
         */

        public Function getFunctionFromSymbolicIntegralCommand(String expression){
            expression = expression.trim();
            double args[] = new double[3];
            args[0]=Double.NaN;
            if(expression.startsWith("∫(")){

                String differential = expression.substring(expression.indexOf(")")+1).trim();

                if(differential.startsWith("d")){
                    String variable = differential.substring(1);
                    expression = expression.replace("∫", "intg");
                    expression=expression.substring(0, expression.lastIndexOf(")")+1 );
                    Function f = getFunctionFromIntegralCommand(expression);
                    if(f.hasIndependentVariable(variable)){
                        return f;
                    }//end if
                    else{
                        throw new InputMismatchException("Function "+f+" has no independent variable like "+variable);
                    }

                }
                else{
                    throw new InputMismatchException("Invalid differential!");
                }

            }//end outermost if
            else{
                throw new InputMismatchException("Invalid Integral Expression!");
            }//end else
        }//end method

    }//end class Parser


    public String getVariable() {
        return function.getIndependentVariables().get(0).getName();
    }
    public static void main(String args[]){

        FunctionManager.add("F=@(x)2*x+3");
        FunctionManager.add("P=@(x)ln(x)");
        FunctionManager.add("G=@(x)x*ln(x)-x");
        FunctionManager.add("H=@(x)(x^2+1)^-1");
        FunctionManager.add("I=@(x)atan(x)");


        System.out.println(FunctionManager.FUNCTIONS);
        //∫(F,2,3)dx
//    NumericalIntegral intg = new NumericalIntegral(2,10,12,"F");
//    System.out.println( intg.findPolynomialIntegral(12));
        double x1 = -10;
        double x2 = 10;

        NumericalIntegral numericalIntegral = new NumericalIntegral("intg(H,"+x1+","+x2+")", FUNCTIONAL_INTEGRATION);
//System.out.println(numericalIntegral.findTrapezoidalIntegral(0.1));
//System.out.println("AdvancedPolynomialIntegral:  "+numericalIntegral.findAdvancedPolynomialIntegral());
//System.out.println("GaussianQuadrature: "+numericalIntegral.findGaussianQuadrature());


        double numericalValue = numericalIntegral.findHighRangeIntegralWithAdvancedPolynomial();
        Function f = FunctionManager.lookUp("I");

        double realValue = f.calc(x2) - f.calc(x1);
        System.out.println("Numerical value: "+ numericalValue);
        System.out.println("Real value: "+realValue);

        System.out.println("%Error: "+( 100 * (realValue - numericalValue)/realValue ));




//
    }//end main












}//end class NumericalIntegral