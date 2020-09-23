/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.tartaglia;
import parser.STRING;
import parser.CustomScanner;
import parser.LISTS;
import parser.MathScanner;
import java.util.ArrayList;

import static parser.STRING.*;
import static parser.Operator.*;
import static parser.Variable.*;
import static parser.Number.*;
import java.util.List;
import math.differentialcalculus.Formula;
/**
 *
 * @author JIBOYE OLuwagbemiro Olaoluwa
 */
public class TartagliaExpressionParser {
    private String expression="";
    private ArrayList<String>scanner=new ArrayList<String>();
    private boolean valid=true;
    private ArrayList<String>vars=new ArrayList<String>();
    //store that contains the coefficients of a,b,c as extracted by the parsing action.
    private final ArrayList<Double>coefficients=new ArrayList<Double>();
    public TartagliaExpressionParser(String expression) {
        expression=purifier(expression);

        String LHS = expression.substring(0, expression.indexOf("="));
        String RHS = expression.substring(1+expression.indexOf("="));


        List<String> scanLHS = new MathScanner(LHS).scanner();
        Formula.simplify( scanLHS );
        List<String> scanRHS = new MathScanner(RHS).scanner();
        Formula.simplify( scanRHS );
        scanLHS.add("=");
        scanLHS.addAll(scanRHS);

        expression = LISTS.createStringFrom(scanLHS, 0, scanLHS.size());




        expression=expression.replace("*", "");
        if(isVariableString(expression.substring(0 , 1))){
            expression="1.0"+expression;
        }

        for(int i=0;i<expression.length();i++){
            String s=expression.substring(i, i+1);
            if(!s.equals(" ")&&!s.equals("\n")){
                this.expression+=s;
            }


        }
        plusAndMinusStringHandler();

        scan();


        /**
         * The scanner of this project does not
         * separate number.concat(variable) tokens but
         * requires them to be separated by the * operator.
         *
         * Our work here however requires this feature so we
         * enable it with this for loop.
         * This loop takes the unscanned number.concat(variable) tokens
         * and separates them into a number token, a * token and a variable token.
         */
        for(int i=0;i<scanner.size();i++){
            try{
                if(!scanner.get(i).equals("+")&&!scanner.get(i).equals("-")&&!scanner.get(i).equals("=")&&
                        !scanner.get(i).equals(";")
                        &&!validNumber(scanner.get(i))&&!isVariableString(scanner.get(i))){
                    MathScanner scan=new MathScanner(scanner.get(i));
                    ArrayList<String> split=scan.splitStringAtFirstNumber(scanner.get(i));
                    if( validNumber( split.get(0) )&&isVariableString(split.get(1)) ){
                        scanner.remove(i);
                        scanner.addAll(i, split);
                        i+=2;
                    }
                }
            }//end try
            catch(IndexOutOfBoundsException indexErr){
                break;
            }

        }//end loop

//validates variable..power..exponent arrangement
        //and selects which numbers can serve as the exponent.
        //The second logic turns say 2x^3 to 2x³,..2x^1 to 2x and 2x^0 to 2.
        for(int i=0;i<scanner.size();i++){

            try{
                if(   !isVariableString(scanner.get(i-1))   && isPower(scanner.get(i))&&  !validNumber(scanner.get(i+1))                   ){
                    setValid(false);
                    break;
                }
                if(   isVariableString(scanner.get(i-1))   && isPower(scanner.get(i))&&  !validNumber(scanner.get(i+1))                   ){
                    setValid(false);
                    break;
                }
                if(   !isVariableString(scanner.get(i-1))   && isPower(scanner.get(i))&&  validNumber(scanner.get(i+1))                   ){
                    setValid(false);
                    break;
                }
                if(   isVariableString(scanner.get(i-1))   && isPower(scanner.get(i))&&    validNumber(scanner.get(i+1))&&
                        Double.valueOf(scanner.get(i+1))!=0.0 &&   Double.valueOf(scanner.get(i+1))!=1.0&&  Double.valueOf(scanner.get(i+1))!=3.0 ){
                    setValid(false);
                    break;
                }





                else if( isVariableString(scanner.get(i-1))   && isPower(scanner.get(i))&&    validNumber(scanner.get(i+1))&&
                        (Double.valueOf(scanner.get(i+1))==0.0 ||  Double.valueOf(scanner.get(i+1))==1.0 ||  Double.valueOf(scanner.get(i+1))==3.0) ){

                    //turns number*x^0 to number
                    if(Double.valueOf(scanner.get(i+1))==0.0){
                        scanner.set(i-1,"" );
                        scanner.set(i,"");
                        scanner.set(i+1,"");
                    }
                    //turns number*x^1 to number*x
                    else if(Double.valueOf(scanner.get(i+1))==1.0){
                        scanner.set(i,"");
                        scanner.set(i+1,"");
                    }
                    else if(Double.valueOf(scanner.get(i+1))==3.0){
                        scanner.set(i,"³");
                        scanner.set(i+1,"");
                    }

                }

            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }//end catch

        }//end for

        freeWhiteSpaces();
        appendOneToStartOfFreeVariables();
        validateAll();
        recognizeNegativesAndPositives();
        recognizeCompoundVariables();
//The string of insertions is to add the complete pattern
//0x^2+0x+0 to the equation.

        try{
            scanner.add(0,"+");
            scanner.add(0,"0");
            scanner.add(0,"+");
            scanner.add(0,vars.get(1));
            scanner.add(0,"0");
            scanner.add(0,"+");
            scanner.add(0,vars.get(0));
            scanner.add(0,"0");
        }
        catch( IndexOutOfBoundsException indexErr ){

        }
        doArithmetic();

    }//end constructor
    /**
     *
     * @return the coefficients ArrayList object.
     */
    public ArrayList<Double> getCoefficients() {
        return coefficients;
    }





    /**
     * removes white space from the ArrayList
     */
    public void freeWhiteSpaces(){

        ArrayList<String>purify=new ArrayList<String>();
        purify.add("");purify.add(" ");
        scanner.removeAll(purify);


    }
    public void setValid(boolean valid) {
        this.valid = valid;
        if(!valid){
            scanner.clear();
        }
    }

    public boolean isValid() {
        return valid;
    }

    public void setScanner(ArrayList<String> scanner) {
        this.scanner = scanner;
    }

    public ArrayList<String> getScanner() {
        return scanner;
    }

    public void setExpression(String expression) {
        this.expression = expression;
    }

    public String getExpression() {
        return expression;
    }


    /**
     * Handles consecutive strings of plus and minus operators
     * and simplifies them, replacing them with their single equivalent
     * in plus and minus terms.e.g +++---+ is equivalent to a -
     * and --++-- is equivalent to a +.
     */
    private void plusAndMinusStringHandler(){

        int size=expression.length();
        for(int i=0;i<size;i++){

            try{
                if(expression.substring(i,i+1).equals("-")&&expression.substring(i+1,i+2).equals("+") ){
                    expression=replace(expression,  " -", i, i+2);
                }
                else if(expression.substring(i,i+1).equals("+")&&expression.substring(i+1,i+2).equals("-") ){
                    expression=replace(expression,  " -", i, i+2);
                }
                else if(expression.substring(i,i+1).equals("+")&&expression.substring(i+1,i+2).equals("+") ){
                    expression=replace(expression,  " +", i, i+2);
                }
                else if(expression.substring(i,i+1).equals("-")&&expression.substring(i+1,i+2).equals("-") ){
                    expression=replace(expression,  " +", i, i+2);
                }

            }
            catch(IndexOutOfBoundsException ind){

            }
        }//end for


    }//end method


    /**
     *
     * @param input  The word to search.
     * @return the first index of occurrence of a letter,
     * and -1 if no letter of the English alphabet is present.
     */
    private int indexOfLetter( String input ){
        int sz=input.length();
        for( int i=0;i<sz;i++){
            if( STRING.isLetter(input.substring(i,i+1)) ){
                return i;
            }//end if

        }//end for
        return -1;
    }//end method

    /**
     *
     * @param input  The word to search.
     * @return true if the input begins with a digit or a point.
     */
    private boolean startsWithDigitOrPoint( String input ){
        String st = input.substring(0,1);
        return STRING.isDigit(st)||st.equals(".");
    }//end method

    /**
     *
     * @param input  The word to search.
     * @return true if the input begins with a letter.
     */
    private boolean startsWithLetter( String input ){
        String st = input.substring(0,1);
        return STRING.isLetter(st);
    }//end method


    /**
     * Scan the system.
     */
    private void scan(){
        ArrayList<String>remover=new ArrayList<String>();

        remover.add("");
        CustomScanner cs = new CustomScanner(expression, true,
                new String[]{"*","+","-","^","=",});
        scanner = (ArrayList<String>) cs.scan();
/*
3x-9y+3.23z=9:
1.3x-0.19y+3.1z=22.91:
-2.3E-2x-2.09y+3.01E-2z=1.39:
*/
/**
 * Separate number.concat(variable) complexes.
 */
        for( int i=0;i<scanner.size();i++){
            String st = scanner.get(i);
            int ind = indexOfLetter(scanner.get(i));
            if( ind != -1 ){
                if(startsWithDigitOrPoint(st)){
                    try{
                        String part1 = st.substring(0, ind);
                        String part2 = st.substring(ind);
                        scanner.set(i,part1);
                        scanner.add(i+1,part2);

                    }//end try
                    catch(IndexOutOfBoundsException indErr){
                    }
                }//end if

                else if( startsWithLetter(st) && st.substring(0,1).equalsIgnoreCase("E") ){
                    try{
                        String part1 = st.substring(0, 1);
                        String part2 = st.substring(1);
                        scanner.set(i,part1);
                        scanner.add(i+1,part2);
                    }//end try
                    catch(IndexOutOfBoundsException indErr){
                    }
                }//end else if

            }//end if


        }//end for

        scanner.removeAll(remover);//removes white spaces.
/**
 * Resolve issues with E the exp operator.
 */
        for( int i=0;i<scanner.size();i++){

            try{
                String st = scanner.get(i);
                if( st.equalsIgnoreCase("E")){
                    String st1 = scanner.get(i-1);
                    String st2 = scanner.get(i+1);
                    String st3 = scanner.get(i+2);
                    String st4 = scanner.get(i+3);
                    String st5 = scanner.get(i+4);
/**
 * 2 scenarios are possible:
 * 1. number,E|e,+|-,number,variable,+|-....e.g 2.089E-5x,or 2.089E+5x
 * OR:
 *    number,E|e,number,variable,+|- 0.089E5x
 *
 */
                    /**
                     * This handles the first scenario
                     */
                    if( validNumber(st1) && isPlusOrMinus(st2) && validNumber(st3)&&isVariableString(st4)&&(isPlusOrMinus(st5)||st5.equals("=")) ){
                        st = st1.concat(st);
                        st = st.concat(st2);
                        st = st.concat(st3);
                        scanner.set(i-1,st);
                        scanner.subList(i, i+3).clear();


                    }//end if
                    else if( validNumber(st1) && validNumber(st2)&&isVariableString(st3)&&(isPlusOrMinus(st4)||st4.equals("=")) ){
                        st = st1.concat(st+"+");
                        st = st.concat(st2);
                        scanner.set(i-1,st);
                        scanner.subList(i, i+2).clear();
                    }//end if



                }//end if
            }//end try
            catch(IndexOutOfBoundsException boundsException){

            }
        }//end for


        /**
         * In dealing with input like 2x or other number.concat(variable) forms,
         * the scanner  turns them into (,2,*,x,)..so that expressions like sin2x
         * can be correctly interpreted as sin(2x) and not sin2 * x i.e x*sin2.
         * Our work here does not require brackets and the multiplication symbol so
         * we remove them.
         */
        remover.add("");
        remover.add("(");
        remover.add(")");
        remover.add("*");
        scanner.removeAll(remover);

    }


    /**
     * When the situation x+2y+z=9; is met, this method will convert
     * it into 1.0x+2y+z=9;
     */
    public void appendOneToStartOfFreeVariables(){

        for(int i=0;i<scanner.size();i++){
            try{
                if( isVariableString(scanner.get(i))&&!validNumber(scanner.get(i-1))){
                    scanner.add(i, "1.0");
                }
            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }
        }

    }//end method

    public void validateEqualsSymbol(){
        int count=0;
        for(int i=0;i<scanner.size();i++){
            if(isAssignmentOperator(scanner.get(i))){
                ++count;
            }
            if(count>1){
                setValid(false);
                break;
            }//end if

        }//end for

        if(count<1){
            setValid(false);
        }


    }//end method


    /**
     * Checks the character set of this parser.
     */
    public void validateChars(){

        for(int i=0;i<scanner.size();i++){
            try{
                if(!isCube(scanner.get(i))&&!validNumber(scanner.get(i))&&!isVariableString(scanner.get(i))
                        &&!isAssignmentOperator(scanner.get(i))&&!scanner.get(i).equals("+")&&!scanner.get(i).equals("-") ){
                    setValid(false);
                }

            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }
        }//end for

    }
    /**
     * Checks if the variables are properly arranged wrt other tokens.
     */
    public void validateVars(){

        for(int i=0;i<scanner.size();i++){
            try{
                if(isVariableString(scanner.get(i))    ){
                    if(!validNumber(scanner.get(i-1))){
                        setValid(false);
                    }//end if
                    if( !isCube(scanner.get(i+1))&&!scanner.get(i+1).equals("+")&&!scanner.get(i+1).equals("-")
                            &&!isAssignmentOperator( scanner.get(i+1)) ){
                        setValid(false);
                    }//end if

                }//end if
            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }


        }//end for

    }//end method

    /**
     * Checks the left and right of a number to see if
     * the appropriate items are the ones there.
     * It is a syntax error if an invalid item is found there
     */
    public void validateNumbers(){
        for(int i=0;i<scanner.size();i++){
            try{
                if(validNumber(scanner.get(i))){
                    if(!scanner.get(i-1).equals("+")&&!scanner.get(i-1).equals("-")&&!scanner.get(i-1).equals("=")){
                        setValid(false);
                    }

                    if(!isVariableString(scanner.get(i+1))&&!isAssignmentOperator(scanner.get(i+1))
                            &&!scanner.get(i+1).equals("+")&&!scanner.get(i+1).equals("-")){
                        setValid(false);
                    }


                }//end if
            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }
        }//end for
    }//end method



    /**
     * The validator of the expression.
     */
    private void validateAll(){
        validateEqualsSymbol();
        validateChars();
        validateVars();
        validateNumbers();


    }


    public void recognizeNegativesAndPositives(){
        for(int i=0;i<scanner.size();i++){
            try{
                if(  scanner.get(i).equals("-")||scanner.get(i).equals("+")  ){
                    if(i==0&&validNumber(scanner.get(i+1))){
                        scanner.set(i+1, scanner.get(i)+scanner.get(i+1));
                        scanner.set(i,"");
                    }
                    else if(isAssignmentOperator(scanner.get(i-1))){
                        scanner.set(i+1, scanner.get(i)+scanner.get(i+1));
                        scanner.set(i,"");
                    }

                }
            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }//end catch



        }//end for
        freeWhiteSpaces();
    }//end method
    /**
     * turns the Xsquare into a single variable.
     */
    public void recognizeCompoundVariables(){
        for(int i=0;i<scanner.size();i++){
            try{
                if(isVariableString(scanner.get(i))&&isCube(scanner.get(i+1))){
                    if(!vars.contains(scanner.get(i))){
                        vars.add(scanner.get(i)+"³");
                        vars.add(scanner.get(i));
                    }
                    scanner.set(i,scanner.get(i)+scanner.get(i+1));scanner.set(i+1,"");
                }
            }//end try
            catch(IndexOutOfBoundsException indexErr){

            }//end catch
        }//end for

        freeWhiteSpaces();

    }//end method

    /**
     *
     * @return the name of the unknown.
     */
    public String getUnknown(){
        return vars.get(1);
    }


    /**
     *
     *This method initializes the coefficients field with
     * the coefficient of Xsquared at index 0
     * the coefficient of X at index 1
     * the coefficient of the constant term at index 2
     *
     */
    public void doArithmetic(){
        ArrayList<Double>coeffs=new ArrayList<Double>();
        coeffs.add(0.0);//coeff of square goes here
        coeffs.add(0.0);//coeff of X goes here
        coeffs.add(0.0);//coeff of constant term goes here.
        for(int j=0;j<2;j++){

            for(int i=0;i<scanner.size();i++){
                //handle the variables.
                try{
                    if( vars.get(j).equals( scanner.get(i) ) ){
                        if(j==0){
                            if(i<scanner.indexOf("=")){
                                if(scanner.get(i-2).equals("-")){
                                    coeffs.set(0, coeffs.get(0)+(- Double.parseDouble(scanner.get(i-1))));
                                }
                                else if(!scanner.get(i-2).equals("-")){
                                    coeffs.set(0, coeffs.get(0)+(+ Double.parseDouble(scanner.get(i-1))));
                                }

                            }
                            else if(i>scanner.indexOf("=")){
                                if(scanner.get(i-2).equals("-")){
                                    coeffs.set(0, coeffs.get(0)-(- Double.parseDouble(scanner.get(i-1))));
                                }
                                else if(!scanner.get(i-2).equals("-")){
                                    coeffs.set(0, coeffs.get(0)-(+ Double.parseDouble(scanner.get(i-1))));
                                }
                            }

                        }//end if


                        else if(j==1){
                            if(i<scanner.indexOf("=")){
                                if(scanner.get(i-2).equals("-")){
                                    coeffs.set(1, coeffs.get(1)+(- Double.parseDouble(scanner.get(i-1))));
                                }
                                else if(!scanner.get(i-2).equals("-")){
                                    coeffs.set(1, coeffs.get(1)+(+ Double.parseDouble(scanner.get(i-1))));
                                }

                            }
                            else if(i>scanner.indexOf("=")){
                                if(scanner.get(i-2).equals("-")){
                                    coeffs.set(1, coeffs.get(1)-(- Double.parseDouble(scanner.get(i-1))));
                                }
                                else if(!scanner.get(i-2).equals("-")){
                                    coeffs.set(1, coeffs.get(1)-(+ Double.parseDouble(scanner.get(i-1))));
                                }
                            }
                        }//end else if
                    }//end if
                    else if(i==scanner.size()-1&&validNumber(scanner.get(i))){//constants located at the end of the scanner

                        if(j==1){
                            if(scanner.get(i-1).equals("-")){
                                coeffs.set(2, coeffs.get(2)-(- Double.parseDouble(scanner.get(i))));
                            }
                            else if(!scanner.get(i-1).equals("-")){
                                coeffs.set(2, coeffs.get(2)-(+ Double.parseDouble(scanner.get(i))));
                            }

                        }


                    }//end if
                    else if(validNumber(scanner.get(i))&&!scanner.get(i+1).equals(vars.get(0))&&!scanner.get(i+1).equals(vars.get(1)) ){
                        if(j==1){

                            if(i<scanner.indexOf("=")){
                                if(scanner.get(i-1).equals("-")){
                                    coeffs.set(2, coeffs.get(2)+(- Double.parseDouble(scanner.get(i))));
                                }
                                else if(!scanner.get(i-1).equals("-")){
                                    coeffs.set(2, coeffs.get(2)+(+ Double.parseDouble(scanner.get(i))));
                                }

                            }
                            else if(i>scanner.indexOf("=")){
                                if(scanner.get(i-1).equals("-")){
                                    coeffs.set(2, coeffs.get(2)-(- Double.parseDouble(scanner.get(i))));
                                }
                                else if(!scanner.get(i-1).equals("-")){
                                    coeffs.set(2, coeffs.get(2)-(+ Double.parseDouble(scanner.get(i))));
                                }
                            }//end else if


                        }//end if


                    }//end else if






                }//end try
                catch(IndexOutOfBoundsException indexErr){


                }

            }//end inner for




        }//end outer for

        coefficients.addAll(coeffs);
    }




    /**
     *
     * @return the reduced form of the system in the form  Ax³+Bx+C=0;
     */
    public String interpretedSystem(){

        String eval = "";
        try{
            eval = coefficients.get(0)+vars.get(0)+"+"+coefficients.get(1)+vars.get(1)+"+"+coefficients.get(2)+" = 0.0";
            eval=eval.replace("+-", "-");
            eval=eval.replace("-+", "-");
            eval=eval.replace("+=", " =");
        }
        catch(IndexOutOfBoundsException indexErr){

        }

        return eval;
    }










    public static void main(String args[]){
        TartagliaExpressionParser parser=new TartagliaExpressionParser("3x-5x^3=-9x-9x^3+2");
        System.out.println(parser.interpretedSystem());
        System.out.println(parser.scanner);

    }




}//end class
