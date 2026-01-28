/**
 *
 * Scanner objects created from this template are operator prioritizing. So they
 * first seek out the operators in the input and tokenize them.
 *
 * To change this template, choose Tools | Templates and open the template in
 * the editor.
 */
package parser;

/**
 * TASK: build regex for matching input expressions...e.g. ^([-+/*]\d+(\.\d+)?)*
 */
//sin2+cos3-9tan(A+3B)
//[sin,2,+,cos,3,-,9,tan (,A,+,3,B,)]
import parser.methods.Method;

import math.numericalmethods.RootFinder;
import java.util.*;
import java.util.ArrayList;

import math.numericalmethods.NumericalIntegral;
import util.*;
import static parser.STRING.*;
import static parser.Operator.*;
import static parser.Variable.*;
import static parser.Number.*;

import math.differentialcalculus.Parser;

/**
 *
 * @author GBEMIRO
 */
public class MathScanner {

    public Parser_Result parser_Result = Parser_Result.VALID;
    /**
     * Returns true if the expression is validated by the scanner as containing
     * all the necessary objects and no foreign object. This does not validate
     * syntax but only validates that the objects used are the valid ones of a
     * mathematical language
     */

    /**
     * Contains a list of Variable objects used that are not declared.
     */
    private ArrayList<String> errorList = new ArrayList<>();
    private boolean runnable = true;

    /**
     * The math expression to be scanned.
     *
     */
    private String scannerInput;

    /**
     * Contains the scanned expression
     */
    private ArrayList<String> scanner = new ArrayList<>();

    /**
     *
     * @param scannerInput the input of this Scanner object
     */
    public MathScanner(String scannerInput) {

        //±–
        /**
         * The loop looks for the occurrence of + and - operators that occur as
         * a part of numbers and not as operators on numbers and replaces them
         * with ± and – respectively. This is needed while it looks for operator
         * tokens which we do not wish to confuse these ones with. Once we have
         * our operator tokens, we can again convert them back into their real
         * forms.
         */
        for (int i = 0; i < scannerInput.length(); i++) {
            try {
                if (scannerInput.substring(i, i + 1).equals("π")) {
                    scannerInput = STRING.replace(scannerInput, "pi", i, i + 1);
                } else if (scannerInput.substring(i, i + 1).equals("×")) {
                    scannerInput = STRING.replace(scannerInput, MULTIPLY, i, i + 1);
                } else if (scannerInput.substring(i, i + 1).equals("÷")) {
                    scannerInput = STRING.replace(scannerInput, DIVIDE, i, i + 1);
                } else if (scannerInput.substring(i, i + 1).equalsIgnoreCase("E")
                        && isDigit(scannerInput.substring(i + 2, i + 3))) {

                    if (scannerInput.substring(i + 1, i + 2).equals(PLUS)) {
                        scannerInput = STRING.replace(scannerInput, "±", i + 1, i + 2);
                    } else if (scannerInput.substring(i + 1, i + 2).equals(MINUS)) {
                        scannerInput = STRING.replace(scannerInput, EN_DASH, i + 1, i + 2);//replacing the minus operator with the En-Dash symbol
                    }
                }

            }//end try
            catch (IndexOutOfBoundsException indexErr) {

            }

        }//end for

        /*
         * Convert the - in negative numbers of a statistical
         * data set into ~.
         */
        for (int i = 0; i < scannerInput.length(); i++) {
            if (scannerInput.substring(i, i + 1).equals(",")
                    && scannerInput.substring(i + 1, i + 2).equals(MINUS)
                    && (scannerInput.substring(i + 2, i + 3).equals(".") || isDigit(scannerInput.substring(i + 2, i + 3)))) {
                scannerInput = replace(scannerInput, "~", i + 1, i + 2);
            }

        }//sort(5,3,2,1,-8,-9,12,34,98,-900,34,23,12,340)

        DataSetFormatter dsf = new DataSetFormatter(scannerInput);
        this.scannerInput = dsf.getFormattedDataSet();

    }

    /**
     *
     * @param scannerInput sets the input of this Scanner object
     */
    public void setScannerInput(String scannerInput) {
        this.scannerInput = scannerInput;
    }

    /**
     *
     * @return the input into the scanner.
     */
    public String getScannerInput() {
        return scannerInput;
    }

    /**
     *
     * @param scanner sets the ArrayList object that holds the scanned output.
     */
    public void setScanner(ArrayList<String> scanner) {
        this.scanner = scanner;
    }

    /**
     *
     * @return the ArrayList object that holds the scanned output.
     */
    public ArrayList<String> getScanner() {
        return scanner;
    }

    /**
     *
     * @param runnable sets whether object of this class are runnable on a
     * calculating device or not.
     */
    public void setRunnable(boolean runnable) {
        this.runnable = runnable;
        if (!this.runnable) {
            scanner.clear();
        }
    }

    /**
     *
     * @return true if the object of this class is runnable on a calculating
     * device.
     */
    public boolean isRunnable() {
        return runnable;
    }

    /**
     *
     * @param errorList sets the list of errors.
     */
    public void setErrorList(ArrayList<String> errorList) {
        this.errorList = errorList;
    }

    /**
     *
     * @return the list of errors.
     */
    public ArrayList<String> getErrorList() {
        return errorList;
    }

    /**
     * method getFirstNumberSubstring takes a String object and returns the
     * first number string in the String.
     *
     * @param val the string to analyze
     * @return the index of the first occurrence of a number character in the
     * sequence.
     * @throws StringIndexOutOfBoundsException
     */
    public static String getFirstNumberSubstring(String val) throws StringIndexOutOfBoundsException {

        /*The number model used to detect the number is....
         * numberstring...a
         * floating point.p
         * numberstring...b
         * exp symbol.....e
         * operator.......s
         * numberstring...c
         * There should be an operator symbol before the first number
         * but in this scenario we do not handle that.
         *
         */
        String num1 = "";
        String num2 = "";
        String point = "";
        String exp = "";
        String sign = "";
        String num3 = "";
        int plusCount = 0;
        int minusCount = 0;
        int expCount = 0;
        int pointCount = 0;

        int ind = getFirstIndexOfDigit(val);
        try {
            if (val.substring(ind - 1, ind).equals(".")) {
                num2 = ".";
                pointCount++;
            }
        } catch (IndexOutOfBoundsException in) {
            num2 = "";
            num1 = "";
        }
//num=a+p+b+e+s+c
        if (num2.equals(".")) {
            try {
                for (int i = ind; i < val.length(); i++) {
                    if (!isNumberComponent(val.substring(i, i + 1))) {
                        break;
                    }
                    //a number cannot contain more than 1 decimal point.
                    if (val.substring(i, i + 1).equals(".")) {
                        break;
                    }

                    //if a + is encountered and no sign operator has been encountered then parse it accordingly
                    if (val.substring(i, i + 1).equals("+")) {
                        if (expCount == 0) {
                            break;
                        }
                        if (plusCount == 1 || minusCount == 1) {
                            break;
                        }
                        plusCount++;

                    }
                    //if a - is encountered and no sign operator has been encountered then parse it accordingly
                    if (val.substring(i, i + 1).equals("-")) {
                        if (expCount == 0) {
                            break;
                        }
                        if (minusCount == 1 || plusCount == 1) {
                            break;
                        }
                        minusCount++;

                    }
                    if (val.substring(i, i + 1).equals("E")) {
                        if (expCount == 1) {
                            break;
                        }
                        exp = "E";
                        expCount++;

                    }

                    //build the decimal part of the number on the condition that it must occur
                    //before the exp symbol is encountered
                    if (isDigit(val.substring(i, i + 1)) && (pointCount == 1 && expCount == 0)) {
                        point += val.substring(i, i + 1);
                    }
                    if (val.substring(i, i + 1).equals("+") && expCount == 1 && num3.equals("") && sign.equals("")) {
                        sign += val.substring(i, i + 1);
                    }
                    if (val.substring(i, i + 1).equals("-") && expCount == 1 && num3.equals("") && sign.equals("")) {
                        sign += val.substring(i, i + 1);
                    }

                    if (isDigit(val.substring(i, i + 1)) && ((pointCount == 0 && expCount == 1) || (pointCount == 1 && expCount == 1))) {
                        num3 += val.substring(i, i + 1);
                    }

                }//end for

            }//end try
            catch (IndexOutOfBoundsException indexerr) {

            }
        }//end if
        //num=a+p+b+e+s+c
        else if (num2.equals("")) {
            try {
                for (int i = ind; i < val.length(); i++) {
                    if (!isNumberComponent(val.substring(i, i + 1))) {
                        break;
                    }
                    if (isDigit(val.substring(i, i + 1)) && (pointCount == 0 && expCount == 0)) {
                        num1 += val.substring(i, i + 1);
                    }
                    if (val.substring(i, i + 1).equals(".")) {
                        //disallow decimal points after an exponent symbol has been parsed
                        if (expCount == 1) {
                            break;
                        }
                        //disallow more than 1 point from being parsed
                        if (pointCount == 1) {
                            break;
                        }
                        num2 = ".";
                        pointCount++;

                    }
                    //if a + is encountered and no sign operator has been encountered then parse it accordingly
                    if (val.substring(i, i + 1).equals("+")) {
                        if (expCount == 0) {
                            break;
                        }
                        if (plusCount == 1 || minusCount == 1) {
                            break;
                        }
                        plusCount++;

                    }
                    //if a - is encountered and no sign operator has been encountered then parse it accordingly
                    if (val.substring(i, i + 1).equals("-")) {
                        if (expCount == 0) {
                            break;
                        }
                        if (minusCount == 1 || plusCount == 1) {
                            break;
                        }
                        minusCount++;

                    }
                    if (val.substring(i, i + 1).equals("E")) {
                        if (expCount == 1) {
                            break;
                        }
                        exp = "E";
                        expCount++;

                    }

                    //build the decimal part of the number on the condition that it must occur
                    //before the exp symbol is encountered
                    if (isDigit(val.substring(i, i + 1)) && (pointCount == 1 && expCount == 0)) {
                        point += val.substring(i, i + 1);
                    }
                    if (val.substring(i, i + 1).equals("+") && expCount == 1 && num3.equals("") && sign.equals("")) {
                        sign += val.substring(i, i + 1);
                    }
                    if (val.substring(i, i + 1).equals("-") && expCount == 1 && num3.equals("") && sign.equals("")) {
                        sign += val.substring(i, i + 1);
                    }

                    if (isDigit(val.substring(i, i + 1)) && ((pointCount == 0 && expCount == 1) || (pointCount == 1 && expCount == 1))) {
                        num3 += val.substring(i, i + 1);
                    }

                }//end for
            }//end try
            catch (IndexOutOfBoundsException indexerr) {

            }

        }//end else if

//klnsdkn23.94E2jwjji
        //klwejh.7856E++43
        String result = num1 + num2 + point + exp + sign + num3;
        try {
            if (lastElement(result).equals("+") || lastElement(result).equals("-")) {
                result = result.substring(0, result.length() - 1);
            }
            if (lastElement(result).equals("+") || lastElement(result).equals("-")) {
                result = result.substring(0, result.length() - 1);
            }
            if (lastElement(result).equals("E")) {
                result = result.substring(0, result.length() - 1);
            }

        } catch (IndexOutOfBoundsException in) {
        }
        return result;
    }//end method getFirstNumberSubstring

    /**
     *
     * @param val the String to analyze
     * @return an ArrayList consisting of 2 or 3 parts. a. If a number substring
     * starts the string,then it returns an ArrayList object containing the
     * number substring and the part of the string after the number. e.g. for
     * 231.62A+98B+sin45C, the method returns [231.62,A+98B+sin45C] b. If a
     * number substring does not start the string but occurs later on in it, the
     * method returns an ArrayList object containing (i.)the substring from
     * index 0 up to, but not including the index where the first number occurs.
     * (ii.)The number substring (iii.)The remaining part of the string after
     * the number.
     *
     * The method will not be used directly by developers but will be used as a
     * tool in math expression scanning.
     *
     *
     */
    public ArrayList<String> splitStringAtFirstNumber(String val) {
        int firstOccDigit = getFirstIndexOfDigitOrPoint(val);//record the index where a digit or point first occurs.

        if (firstOccDigit > -1) {
//CASE 1: i.e a number starts the input string.
            if (firstOccDigit == 0) {
                String getFirstNumber = getFirstNumberSubstring(val);
                int firstNumberLen = getFirstNumber.length();
                scanner.add(getFirstNumber);
                scanner.add(val.substring(firstNumberLen));
            } //CASE 2: a number does not start the input string.
            else if (firstOccDigit > 0) {

                String getFirstNumber = getFirstNumberSubstring(val);
                int firstNumberLen = getFirstNumber.length();
                scanner.add(val.substring(0, firstOccDigit));
                scanner.add(getFirstNumber);
                scanner.add(val.substring(firstOccDigit + firstNumberLen));
            }

        }
        return scanner;
    }

    /**
     * method splitStringOnNumbers takes a String object and returns an
     * ArrayList of substrings in which the original string has been split into
     * different parts based on the amount of number substrings it has.
     *
     * @param val the string to analyze e.g if val="123.234+43.6",the output
     * will be [123.234,+,43.6]
     * @return An ArrayList of substrings consisting of number substrings and
     * the other substrings of the input.
     * @throws StringIndexOutOfBoundsException
     */
    public ArrayList<String> splitStringOnNumbers(String val) {

        ArrayList<String> scan = new ArrayList<>();
        ArrayList<String> split = new ArrayList<>();

        boolean canSplit = getFirstIndexOfDigitOrPoint(val) > -1;
        if (canSplit) {
            int passes = 0;
            while (canSplit) {
                split.clear();
                split = splitStringAtFirstNumber(val);//the splitting
                int sizeAfterSplit = split.size();//the size after the splitting occurs

                val = split.get(sizeAfterSplit - 1);//get the new string to split if it is still splittable on numbers.

                scan.addAll(split.subList(0, sizeAfterSplit - 1));
                canSplit = getFirstIndexOfDigitOrPoint(val) > -1;

                if (!canSplit) {
                    scan.add(val);
                    break;
                }

                ++passes;
            }

        } else {
            scan.add(val);
        }

        return scan;
    }

//
    /**
     * method getNumberStrings takes a String object and returns an ArrayList of
     * substrings of all numbers in the input String
     *
     * @param val the string to analyze e.g if val="123.234+43.6",the output
     * will be [123.234,43.6]
     * @return An ArrayList of substrings consisting of number substrings and
     * the other substrings of the input.
     * @throws StringIndexOutOfBoundsException
     */
    public ArrayList<String> getNumberStrings(String val) {
        ArrayList<String> split = new ArrayList<>();
        int firstOcc = -1;
        int extent = 0;//measures the length of each number substring

        int size = 0;
        val = " " + val;
        while (firstOcc != 0) {
            firstOcc = getFirstIndexOfDigit(val);

            if (firstOcc > 0 && val.substring(firstOcc - 1, firstOcc).equals(".")) {
                firstOcc -= 1;
            }

            split.add(getFirstNumberSubstring(val));//store number token in split.
            //get the number of characters in the last element in split.
            size = split.get(split.size() - 1).length();//get the size of the last element in the ArrayList
            extent = firstOcc + size;//index of the remaining part of the string ro be processed
            val = val.substring(extent);

        }
        ArrayList<String> clearEmptyString = new ArrayList<>();
        clearEmptyString.add("");
        split.removeAll(clearEmptyString);
        return split;
    }

    /**
     *
     * Split the {@link MathScanner#scannerInput} String on the operators.
     */
    public void splitStringOnMethods_Variables_And_Operators() {
        CustomScanner cs = new CustomScanner(scannerInput, true, VariableManager.VARIABLES.keySet().toArray(new String[]{}), Method.getAllFunctions(), operators);
        ArrayList<String> filter = new ArrayList<>();
        filter.add("");
        filter.add(",");
        scanner.addAll(cs.scan());

      
        for (int i = 0; i < scanner.size(); i++) {
            String token = scanner.get(i);

            if (isNumber(token) && !validNumber(token)) {

                for (int j = 0; j < token.length(); j++) {//3.4567A
                    if (!isNumberComponent(token.substring(j, j + 1)) && isVariableString(token.substring(j))) {
                        scanner.set(i, token.substring(0, j));
                        scanner.add(i + 1, "*");
                        scanner.add(i + 2, token.substring(j));
                        i++;
                        break;

                    }
                }

            }

            /**
             * convert 2,sin patterns to 2,*,sin This allows us to use 2sin(5)
             * or 2A + 3B As the parser will interprete them into: 2*sin(5) and
             * 2*A+3*B patterns
             *
             * It also enables Asin(5) patterns
             *
             */

            if (i+1<scanner.size() && (validNumber(token) || isVariableString(token)) && isVariableString(scanner.get(i + 1))) {
                scanner.add(i + 1, "*");
            }
            /**
             * Allow early conversion of )sin(k) into )*sin(k)
             */
            else if (i+1<scanner.size() && isClosingBracket(token) && isVariableString(scanner.get(i + 1)) &&
                    !Method.isListReturningStatsMethod(scanner.get(i + 1))  ) {
                /**
                 * Check if the bracket owner allows for multiplication.
                 */
                int open = Bracket.getComplementIndex(false, i, scanner);
                if(open>0 && !isAtOperator(scanner.get(open-1)) && !Method.isListReturningStatsMethod(scanner.get(open-1))){
                    scanner.add(i + 1, "*");
                }
            }
            else if(i+1<scanner.size() && isNumber(token)&& isOpeningBracket(scanner.get(i+1))){
                scanner.add(i+1,"*");
            }


        }

        /**
         * Distinguish between variables and methods. Consider the object sin5
         * intended to be a variable in the function 3+2*cos(3)-sin5. The just
         * concluded scan will see sin as a valid inbuilt method and so will
         * split the variable into sin and 5.
         *
         * Correct this by analyzing the structure. All method names must have a
         * ( in front of them. So check and re-couple the split variable.
         *
         */
        for (int i = 0; i < scanner.size(); i++) {
            try {
                if (Method.isMethodName(scanner.get(i)) && (isNumber(scanner.get(i + 1)) || Method.isMethodName(scanner.get(i + 1)))) {
                    scanner.set(i, scanner.get(i) + scanner.get(i + 1));
                    scanner.remove(i + 1);
                    --i;
                }//end if
            }//end try
            catch (IndexOutOfBoundsException ind) {

            }
        }//end for loop
  
//sort(5,3,2,1,-8,-9,12,34,98,-900,34,23,12,340)

        for (int i = 0; i < scanner.size(); i++) {//±–
            if (scanner.get(i).contains("±")) {
                int index = scanner.get(i).indexOf("±");
                scanner.set(i, replace(scanner.get(i), "+", index, index + 1));
            } else if (scanner.get(i).contains("–")) {//replace the en-dash symbol with the minus (-) symbol again..
                int index = scanner.get(i).indexOf("–");
                scanner.set(i, replace(scanner.get(i), "-", index, index + 1));
            }
            if (isOpeningBracket(scanner.get(i)) && scanner.get(i + 1).equals("-") && isNumber(scanner.get(i + 2))) {

                int index = scanner.get(i+2).indexOf(EN_DASH);
                if (index != -1) {//In case the number at i+2 contains an EN_DASH, replace it with a minus
                    scanner.set(i+2, replace(scanner.get(i+2), MINUS, index, index + 1));
                }
                scanner.set(i + 1, String.valueOf(-1 * Double.parseDouble(scanner.get(i + 2))));
                scanner.remove(i + 2);
            }
            /**
             * Convert the ~ symbol back to - for negative numbers in stats
             * methods.
             */
            if (scanner.get(i).substring(0, 1).equals("~")) {
                scanner.set(i, "-" + scanner.get(i).substring(1));
            }

        }//end if
        
        removeExcessBrackets(scanner);
        recognizeAnonymousFunctions(scanner);
        for (int i = 0; i < scanner.size(); i++) {

            String token = scanner.get(i);
            if(i+1>=scanner.size()){break;}
            String nextToken = scanner.get(i+1);

            if (token.equals("diff") && nextToken.equals("(")) {
///diff,(,@,(x),log,(,x, , ,2,), , ,4,)

                int close = Bracket.getComplementIndex(true, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);
                Parser.parseDerivativeCommand(list);
                if (list.isEmpty()) {
                    parser_Result = Parser_Result.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }

            }//end if
            else if (token.equals("intg") && nextToken.equals("(")) {
// intg,(,@,(x),log(x,2),4,2,5)

                int close = Bracket.getComplementIndex(true, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);
                NumericalIntegral.extractFunctionStringFromExpression(list);
                if (list.isEmpty()) {
                    parser_Result = Parser_Result.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }

            }//end else if
            else if (Method.isMatrixMethod(token) && nextToken.equals("(")) {
// matrix_mul,(,@,(x),log(x,2),4,8)
                int close = Bracket.getComplementIndex(true, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);
                extractFunctionStringFromExpressionForMatrixMethods(list);
                if (list.isEmpty()) {
                    parser_Result = Parser_Result.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }

            }//end else if
            else if ((token.equals("root")) && nextToken.equals("(")) {
// intg,(,@,(x),log(x,2),4,2,5)
                int close = Bracket.getComplementIndex(true, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);

                RootFinder.extractFunctionStringFromExpression(list);

                if (list.isEmpty()) {
                    parser_Result = Parser_Result.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }
            }//end else if
        }//end for.

        scanner.removeAll(filter);
//23sin2-7cosh55-90+223-23sin2-7cosh55-90+223
    }

    /**
     * Check that variables and methods do not conflict..if that can happen. Now
     * method names and variable names are similar. The only difference between
     * them is that methods use parentheses while variables don't. Enforce
     * especially then that variable-parenthesis combinations that may be
     * mistaken for methods are not allowed.
     *
     * e.g sin(..) is a method. A user may name a variable sin, if he/she so
     * wishes. make sure that the user does not do things like..
     * varname(expr..). Rather, he/she must separate variables from parentheses
     * using the multiplication operator.
     *
     */
    public void validateInputAfterSplitOnMethodsAndOps() {
        for (int i = 0; i < scanner.size(); i++) {
            try {

                /**
                 * Skip over the contents of stats methods. Move the cursor (i)
                 * to the end of the closing bracket. This will allow stats
                 * methods to be able to handle the production:
                 * sum(2,3,4,sin(3),5..) and not be limited to sum(3,4,5,6,7,..)
                 * ALSO Skip the function operating methods...their syntax is
                 * too complex to be analyzed here.
                 */
                if ((Method.isFunctionOperatingMethod(scanner.get(i)) || Method.isStatsMethod(scanner.get(i))) && isOpeningBracket(scanner.get(i + 1))) {
                    i = Bracket.getComplementIndex(true, i + 1, scanner);
                }//end if
                /**
                 * Enable the use of number(expr..)...and var(expr...) but avoid
                 * func_name(expr....) Users dont have to enter products of
                 * numbers and bracketed expressions as number*(expr)
                 */
                else if ((isNumber(scanner.get(i)) || (isVariableString(scanner.get(i)) && !Method.isDefinedMethod(scanner.get(i))))
                        && scanner.get(i + 1).equals("(")) {
                    scanner.add(i + 1, "*");
                    i++;
                }//end if
                /**
                 * Enable the use of number-concat-funcName(...)...e.g.
                 * 2sin(3)...becomes 2*sin(3)
                 */
                else if (isNumber(scanner.get(i)) && Method.isDefinedMethod(scanner.get(i + 1))) {
                    scanner.add(i + 1, "*");
                    i++;
                }//end if
                /**
                 * Allow things like: (expr)3+1 or (expr)A+... where A is a
                 * variable or method. Force the user to enter it as
                 * (expr)*number or (expr)*A..
                 */
                else if (isClosingBracket(scanner.get(i)) && (isVariableString(scanner.get(i + 1)) || isNumber(scanner.get(i + 1)))) {
                    scanner.add(i + 1, "*");
                    i++;
                }//end else if
                /**
                 * convert ),( into ),*,( e.g (3+4)(5+6) becomes (3+4)*(5+6)
                 * which both the parser and math understand
                 */
                else if (isClosingBracket(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                    scanner.add(i + 1, "*");
                    i++;
                }

            }//end try
            catch (IndexOutOfBoundsException boundsException) {

            }//end catch

        }//end for loop

    }//end method validateInput...

    /**
     * Identifies that the input is a valid one by checking that all tokens are
     * either Number objects, Variable objects or Operator objects. Then it
     * registers any Variable object found and initializes it to zero.
     */
    public void validateTokens() {
        for (int i = 0; i < scanner.size(); i++) {
            try {

                if (!isOperatorString(scanner.get(i)) && !isVariableString(scanner.get(i))
                        && !validNumber(scanner.get(i)) && !Method.isMethodName(scanner.get(i))) {
                    parser_Result = Parser_Result.STRANGE_INPUT;
                    setRunnable(false);
                    errorList.add(scanner.get(i) + " is a strange math object!");
                    return;
                }//end if
            }//end try
            catch (IndexOutOfBoundsException boundsException) {
                return;
            }//end catch
        }//end for
    }//end validateTokens

    /**
     * Handles repeated concatenations of plus and minus operators.
     */
    public void plusAndMinusStringHandler() {

        for (int i = 0; i < scanner.size(); i++) {

            if (scanner.get(i).equals("-") && scanner.get(i + 1).equals("-")) {
                scanner.set(i, "+");
                scanner.subList(i + 1, i + 2).clear();
                i -= 1;
            }//end if
            if (scanner.get(i).equals("-") && scanner.get(i + 1).equals("+")) {
                scanner.set(i, "-");
                scanner.subList(i + 1, i + 2).clear();
                i -= 1;
            }//end else if
            if (scanner.get(i).equals("+") && scanner.get(i + 1).equals("-")) {
                scanner.set(i, "-");
                scanner.subList(i + 1, i + 2).clear();
                i -= 1;
            }//end else if
            if (scanner.get(i).equals("+") && scanner.get(i + 1).equals("+")) {
                scanner.set(i, "+");
                scanner.subList(i + 1, i + 2).clear();
                i -= 1;
            }//end else if

        }//end for loop

    }// end method

    /**
     * Utility method,more popularly used as a scanner into mathematical tokens
     * of mathematical expressions.
     *
     * This method does not check expression variables against declared
     * variables and is only useful in breaking down the input into scanned
     * tokens of numbers,variables and operators.
     *
     * The overloaded one takes a VariableManager object that models the store
     * of Variable objects in a computing application and checks if all variable
     * names entered by the user have been declared.
     *
     * @return an ArrayList containing a properly scanned version of the string
     * such that all recognized number substrings,variable substrings and
     * operator substrings have been resolved. To use this method as a scanner
     * for math expressions all the programmer has to do is to take care of the
     * signs where need be(e.g in -2.873*5R+sinh34.2, the scanned view will be
     * [-,2.873,*,5,R,+,sinh,34.2]) To make sense of this, the programmer has to
     * write minor code to concatenate the - and the 2.873 and so on.
     *
     */
    public ArrayList<String> scanner() {
        splitStringOnMethods_Variables_And_Operators();
        if (parser_Result != Parser_Result.VALID) {
            scanner.clear();
            return scanner;
        }
        validateInputAfterSplitOnMethodsAndOps();
        if (parser_Result != Parser_Result.VALID) {
            scanner.clear();
            return scanner;
        }

        /*
         * The for loop above does not properly handle
         * negative exponents of 10, e.g -3E-10
         * It splits it into -3,E, -10
         * So we need to manually couple the split objects together
         *
         */
        for (int i = 0; i < scanner.size(); i++) {

            if (Number.isNegative(scanner.get(i)) && scanner.get(i + 1).equals("E")
                    && Number.isNumber(scanner.get(i + 2))) {
                try {
                    scanner.set(i, scanner.get(i) + scanner.get(i + 1) + scanner.get(i + 2));
                    scanner.remove(i + 1);
                    scanner.remove(i + 1);
                }//end try
                catch (IndexOutOfBoundsException indexErr) {
                    break;
                }//end catch
            }//end if
        }//end for

//enable interpretation of things like 3^-4 or 3^+4 i.e ^- or ^+ patterns
        for (int i = 0; i < scanner.size(); i++) {
            try {
                if (isPower(scanner.get(i))) {
                    if (scanner.get(i + 1).equals("-") && validNumber(scanner.get(i + 2))) {
                        scanner.set(i + 1, String.valueOf(-1 * Double.parseDouble(scanner.get(i + 2))));
                        scanner.remove(i + 2);
                    } else if (scanner.get(i + 1).equals("+") && validNumber(scanner.get(i + 2))) {
                        scanner.set(i + 1, String.valueOf(Double.parseDouble(scanner.get(i + 2))));
                        scanner.remove(i + 2);
                    }
                }//end if

            }//end try
            catch (IndexOutOfBoundsException indexErr) {

            } catch (NumberFormatException numErr) {

            }
        }//end for

        validateTokens();

        /**
         * Automatically initialize and store undeclared variables to 0 in the
         * first if block. To enforce variable declaration and initialization,
         * always come here.
         */
        for (int i = 0; i < scanner.size(); i++) {
            int sz = scanner.size();
            /**
             * Skip the methods that deal with Function objects, to allow parser
             * to be able to deal with anonymous functions.
             */
            if (i+1<sz && Method.isFunctionOperatingMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                int close = Bracket.getComplementIndex(true, i + 1, scanner);
                i = close;
                continue;
            }//end if

            if (!Variable.isVariableString(scanner.get(i)) && !Operator.isOperatorString(scanner.get(i)) && !validNumber(scanner.get(i))
                    && !Method.isMethodName(scanner.get(i))) {
                errorList.add("Syntax Error! Strange Object Found: " + scanner.get(i));
                setRunnable(false);
                parser_Result = Parser_Result.STRANGE_INPUT;
                System.err.println(errorList.get(errorList.size() - 1));
            }
        }

        if (!runnable) {
            errorList.add("\n"
                    + "Sorry, Errors Were Found In Your Expression."
                    + "Please Consult The Help File For Valid Mathematical Syntax.");
            scanner.clear();
        } else {
            errorList.add("Scan SuccessFul.No Illegal Object Found.\n"
                    + "Putting Scanner On StandBy");
        }

        plusAndMinusStringHandler();
        return scanner;
    }

    /**
     * Utility method,more popularly used as a scanner into mathematical tokens
     * of mathematical expressions
     *
     * @param varMan The store of all user variables already declared. The
     * scanner will not allow the user to use any variables he/she/it has not
     * declared
     * @return an ArrayList containing a properly scanned version of the string
     * such that all recognized number substrings,variable substrings and
     * operator substrings have been resolved. To use this method as a scanner
     * for math expressions all the programmer has to do is to take care of the
     * signs where need be(e.g in -2.873*5R+sinh34.2, the scanned view will be
     * [-,2.873,*,5,R,+,sinh,34.2]) To make sense of this, the programmer has to
     * write minor code to concatenate the - and the 2.873 and so on.
     *
     */
    public ArrayList<String> scanner(VariableManager varMan) {
        splitStringOnMethods_Variables_And_Operators();
        validateInputAfterSplitOnMethodsAndOps();

        /*
         * Re-build the negative numbers in a statistical
         * data set. e.g 2,-3,4... is translated into 2, ~3, 4
         * At the end convert the ~3 back to -3

         for(int i=0;i<scanner.size();i++){
         if(scanner.get(i).substring(0,1).equals("~")){
         scanner.set(i, "-"+scanner.get(i).substring(1));
         }

         }//end for
         */

        /*
         * The for loop above does not properly handle
         * negative exponents of 10, e.g -3E-10
         * It splits it into -3,E, -10
         * So we need to manually couple the split objects together
         *
         */
        for (int i = 0; i < scanner.size(); i++) {
            try {

                if (Number.isNegative(scanner.get(i)) && scanner.get(i + 1).equals("E")
                        && Number.isNumber(scanner.get(i + 2))) {
                    scanner.set(i, scanner.get(i) + scanner.get(i + 1) + scanner.get(i + 2));
                    scanner.remove(i + 1);
                    scanner.remove(i + 1);

                }//end if
            }//end try
            catch (IndexOutOfBoundsException indexErr) {
                break;
            }//end catch
        }//end for

//enable interpretation of things like 3^-4 or 3^+4 i.e ^- or ^+ patterns
        for (int i = 0; i < scanner.size(); i++) {
            try {
                if (isPower(scanner.get(i))) {
                    if (scanner.get(i + 1).equals("-") && validNumber(scanner.get(i + 2))) {
                        scanner.set(i + 1, String.valueOf(-1 * Double.parseDouble(scanner.get(i + 2))));
                        scanner.remove(i + 2);
                    } else if (scanner.get(i + 1).equals("+") && validNumber(scanner.get(i + 2))) {
                        scanner.set(i + 1, String.valueOf(Double.parseDouble(scanner.get(i + 2))));
                        scanner.remove(i + 2);
                    }
                }//end if

            }//end try
            catch (IndexOutOfBoundsException indexErr) {

            } catch (NumberFormatException numErr) {

            }
        }//end for
        validateTokens();

        /**
         * Automatically initialize and store undeclared variables to 0 in the
         * first if block. To enforce variable declaration and initialization,
         * always come here.
         */
        for (int i = 0; i < scanner.size(); i++) {
            int sz = scanner.size();
            /**
             * Skip the methods that deal with Function objects, to allow parser
             * to be able to deal with anonymous functions.
             */
            if (i+1<sz && Method.isFunctionOperatingMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                int close = Bracket.getComplementIndex(true, i + 1, scanner);
                i = close;
                continue;
            }//end if

            if (!Variable.isVariableString(scanner.get(i)) && !Operator.isOperatorString(scanner.get(i)) && !validNumber(scanner.get(i)) &&
                    !Method.isMethodName(scanner.get(i))) {
                errorList.add("Syntax Error! Strange Object Found: " + scanner.get(i));
                parser_Result = Parser_Result.STRANGE_INPUT;
                setRunnable(false);
                System.err.println(errorList.get(errorList.size() - 1));
            }
            if (MathExpression.isAutoInitOn()) {
                if (i+1<sz && Variable.isVariableString(scanner.get(i)) && !isOpeningBracket(scanner.get(i + 1)) && !varMan.contains(scanner.get(i))
                        && !FunctionManager.contains(scanner.get(i))) {
                    varMan.parseCommand(scanner.get(i) + "=0.0;");
                }//end if
            }//end if
            else {
                if (i+1<sz && Variable.isVariableString(scanner.get(i)) && !isOpeningBracket(scanner.get(i + 1)) && !varMan.contains(scanner.get(i))
                        && !FunctionManager.contains(scanner.get(i))) {
                    errorList.add(" Unknown Variable: " + scanner.get(i) + "\n Please Declare And Initialize This Variable Before Using It.\n"
                            + "Use The Command, \'variableName=value\' To Accomplish This.");

                    System.err.println("DETECTED ERROR! IN " + scanner);
                    parser_Result = Parser_Result.STRANGE_INPUT;
                    setRunnable(false);

                }//end if
            }//end else
        }//end for loop

        if (!runnable) {
            errorList.add("\n"
                    + "Sorry, Errors Were Found In Your Expression."
                    + "Please Consult The Help File For Valid Mathematical Syntax.");
            scanner.clear();
        } else {
            errorList.add("Scan SuccessFul.No Illegal Object Found.\n"
                    + "Putting Scanner On StandBy");
        }

        plusAndMinusStringHandler();

        return scanner;
    }

    public static void recognizeAnonymousFunctions(List<String> scanner) {
        int indexOfAt = -1;
 
        while ((indexOfAt = scanner.indexOf("@")) != -1) {

            /**
             * Process Anonymous functions After this loop, the anonymous
             * function would have been initialized and its anonymous expression
             * replaced with a function name.
             */
            if (isOpeningBracket(scanner.get(indexOfAt + 1))) {
                //root(@(x)3*x^2+sin(x)+log(x,2),3,4)

                for (int i = indexOfAt; i < scanner.size(); i++) {
                    String token = scanner.get(i);
                    if (isOpeningBracket(token)) {
                        i = Bracket.getComplementIndex(true, i, scanner);
                    } else if (isComma(token)) {
                        Function f = new Function(LISTS.createStringFrom(scanner, indexOfAt, i));
                        FunctionManager.add(f);
                        List<String> sub = scanner.subList(indexOfAt, i);
                        sub.clear();
                        sub.add(f.getName());

                        break;
                    } else if (isClosingBracket(token)) {
                        int open = Bracket.getComplementIndex(false, i, scanner);
                        if (open < indexOfAt) {//you have gotten to the end of the enclosing brackets of the function and no comma yet
                            Function f = new Function(LISTS.createStringFrom(scanner, indexOfAt, i));
                            FunctionManager.add(f);
                            List<String> sub = scanner.subList(indexOfAt, i);
                            sub.clear();
                            sub.add(f.getName());

                            break;
                        }

                    }

                }

            }else{
                throw new InputMismatchException("Syntax Error occurred while scanning math expression.\nReason: The @ symbol is used exclusively to create functions. Expected: `(`, found: `"+scanner.get(indexOfAt + 1)+"`");
            }

        }
        //System.out.println("scanner-debug: "+scanner);
    }

    /**
     * This technique conserves the last single bracket as it is
     * unsure of the rules that allow it to unwrap the last bracket.
     * @param scanner The tokens list
     */
    public static void $removeExcessBrackets(List<String> scanner) {

        for (int i = 0; i < scanner.size(); i++) {

            if (i + 1 < scanner.size() && isClosingBracket(scanner.get(i))) {

                if (isClosingBracket(scanner.get(i + 1))) {//you have a match for an unnecessary wasted bracket
                    int inner_open = Bracket.getComplementIndex(false, i, scanner);
                    int outer_open = Bracket.getComplementIndex(false, i + 1, scanner);

                    if (inner_open != -1 && outer_open != -1 && inner_open - 1 == outer_open) {
                        scanner.remove(i);
                        scanner.remove(inner_open);
                        i -= 2;
                    }
                }

                else {//the bracket is not enclosed by another bracket..check if it is unnecessarily enclosing a positive or negative number
                    int open = Bracket.getComplementIndex(false, i, scanner);
                    if (i - open == 2) {//confirm that the bracket has only 1 token inside it
                        if(open >  0){//open bracket is beyond 0
                            String token = scanner.get(open - 1);
                            if (!isVariableString(token) && !isAtOperator(token) && !Method.isMethodName(token) && !isUnaryPreOperator(token) ) {
                                scanner.remove(i);
                                scanner.remove(open);
                                i -= 2;
                                continue;
                            }
                        }
                        else if(open == 0){//open bracket is at 0
                            scanner.remove(i);
                            scanner.remove(open);
                            i -= 2;
                            continue;
                        }

                    }//end if

                }

            }

        }

    }


    /**
     * This technique will rid tokens of offending brackets
     * up to the last bracket. It assumes that it knows the rules
     * that allow one to remove all brackets from a token.
     * One however needs check the ruls to be sure that there is no
     * error. If this one messes up, please switch to the {@link MathScanner#$removeExcessBrackets(List) } method.
     * @param scanner The list of scanned tokens.
     *
     */
    public static void removeExcessBrackets(List<String> scanner) {

        for (int i = 0; i < scanner.size(); i++) {

            if(isClosingBracket(scanner.get(i))){

                if(i+1<scanner.size() && isOpeningBracket(scanner.get(i+1)) ){//if the pattern is )(...ignore the closing bracket.
                    continue;
                }

                int open = Bracket.getComplementIndex(false, i, scanner);
                if (i - open == 2) {//confirm that the bracket has only 1 token inside it
                    if(open >  0){//open bracket is beyond 0
                        String token = scanner.get(open - 1);

                        if (!isVariableString(token) && !isAtOperator(token) && !Method.isMethodName(token) && !isUnaryPreOperator(token) ) {
                            scanner.remove(i);
                            scanner.remove(open);
                            i -= 2;
                            continue;
                        }
                    }
                    else if(open == 0){//open bracket is at 0
                        scanner.remove(i);
                        scanner.remove(open);
                        i -= 2;
                        continue;
                    }

                }//end if

                if (i + 1 < scanner.size() ) {

                    if (isClosingBracket(scanner.get(i + 1))) {//you have a match for an unnecessary wasted bracket
                        int inner_open = Bracket.getComplementIndex(false, i, scanner);
                        int outer_open = Bracket.getComplementIndex(false, i + 1, scanner);

                        if (inner_open != -1 && outer_open != -1 && inner_open - 1 == outer_open) {
                            //Skip for log tokens...the bracket is needed to evaluate content.
                            if(outer_open>1 && Method.isLogToAnyBase(scanner.get(outer_open-1))){
                                continue;
                            }
                            scanner.remove(i);
                            scanner.remove(inner_open);
                            i -= 2;
                        }
                    }



                }




            }
        }

    }

    /**
     * Analyzes the list and extracts the Function string from it.
     * @param list The list to be analyzed.
    Direct examples would be:
    intg(@sin(x+1),4,7)
    intg(F,4,7) where F is a function that has been defined before in the workspace.. and so on.
     *
     * Simplifies the list to the form matrix_method(funcName,params)
     *
     */
    public static void extractFunctionStringFromExpressionForMatrixMethods(List<String> list){
        list.removeAll(Arrays.asList(","));
      
        int sz = list.size();
        /**
         * Confirm that there remains only one open and one close bracket in the tokens list.
         */
        if(list.indexOf("(") == list.lastIndexOf("(") && list.indexOf(")") == list.lastIndexOf(")")){
            //det,(,A,) or matrix_mul,(,A,B,)
            if(sz == 4 || sz == 5 ){
                if(Method.isMatrixMethod(list.get(0)) && isOpeningBracket(list.get(1)) && Method.isUserDefinedFunction(list.get(2))  ){
                    if(sz==4 && isClosingBracket(list.get(3))){
                        Method.run(list, 1);
                    }
                    else if(sz==5 && (Method.isUserDefinedFunction(list.get(3))||isNumber(list.get(3))||isVariableString(list.get(3))) && isClosingBracket(list.get(4))){System.out.println("Debug--4");
                        Method.run(list, 1);
                    }

                }

            }
            /**
             * There remains only one open and close bracket, but the parameters have not yet been properly ordered!
             * Most of the matrix methods take one or at most 2 parameters, so if you have more than 2 parameters,
             * then check that the first parameter is in the proper format, then run the parser on everything in
             * the second part of the parameters list to get the second parameter.
             */
            else{

            }
        }


        else{
            for(int i=0; i<list.size();i++){

                //System.out.println("looping it! i = "+i+" of "+list.size()+"--now on: '"+list.get(i)+"' out of "+list);
                if( isClosingBracket(list.get(i)) ){
                    int open = Bracket.getComplementIndex(false, i, list);
                    if(open>0){
                        String token = list.get(open-1);
                        if( Method.isMatrixMethod(token) ){
                            List l = list.subList(open-1, i+1); int siz = l.size();
                            System.err.println("list: "+list);
                            extractFunctionStringFromExpressionForMatrixMethods(l);

                            i = i - (siz - l.size());
                        }
                        //Most likely you have gotten to the first parameter...ignore it and process the bracket
                        else if(FunctionManager.contains(token)){
                            List l = list.subList(open, i+1); int siz = l.size();

                            MathExpression me = new MathExpression(LISTS.createStringFrom(list, open, i+1));
                            String val = me.solve();
                            l.clear();
                            switch(me.getReturnType()){
                                case MATRIX:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case ALGEBRAIC_EXPRESSION:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case LIST:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case NUMBER:
                                    l.add(val);
                                    break;
                                default:
                                    break;
                            }//end switch
                            i = i - (siz - l.size());
                        }
                        else if(Method.isMethodName(token)||isUnaryPreOperator(token)||isNumber(token)){
                            List<String> l = list.subList(open-1, i+1); int siz = l.size();
                            String input;
                            if(Method.isStatsMethod(token)){
                                /**
                                 * Pattern is [sum,(, 2, 3, 4, 6, )]
                                 * We need to convert this into: sum(2,3,4,6)
                                 * Our job is to remove the starting, second and final commas,
                                 * the white spaces,
                                 * and the opening and closing braces[]
                                 */
                                StringBuilder builder = new StringBuilder(token);
                                builder.append("(");
                                for(int j=2;j<l.size()-1;j++){
                                    builder.append(l.get(j)).append(",");
                                }
                                input = builder.substring(0,builder.length()-1);
                                input = input.concat(")");
                            }
                            else{
                                input =LISTS.createStringFrom(list, open-1, i+1);
                            }

                            MathExpression me = new MathExpression(input);
                            String val = me.solve();
                            l.clear();
                            switch(me.getReturnType()){
                                case MATRIX:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case ALGEBRAIC_EXPRESSION:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case LIST:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case NUMBER:
                                    l.add(val);
                                    break;
                                default:
                                    break;
                            }//end switch
                            i = i - (siz - l.size());
                        }

                    }


                }

            }//end for loop

        }




 
    }//end method
    /**
     *
     * @param args Command line args (((2+3)^2))!-------((25))!-------
     */
    public static void main(String args[]) {//tester method for STRING methods

        
        //A*B*C*D*E+3*A+4*B-22*A^2*det(A)
        //String expr = "matrix_mul(M,((((M)))))";
        String expr = "matrix_mul(M,3,((det(M))))";
        CustomScanner cs = new CustomScanner(expr, true,"matrix_mul", "sum", ",", "(", ")", "^","a","b","x");
        List<String> scan = cs.scan();
        System.err.println("----" + scan);
        removeExcessBrackets(scan);
        System.err.println("----" + scan);
        /*
        MathScanner scanner = new MathScanner("matrix_mul(@(2,2)(3,1,4,2),@(2,2)(2,9,-4,3))");
        System.out.println(scanner.scanner);
        scanner.scanner();
        System.out.println(scanner.scanner);
        System.out.println(scanner.parser_Result);*/

    }//end method main
}
