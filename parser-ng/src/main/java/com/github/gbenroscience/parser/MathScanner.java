/**
 *
 * Scanner objects created from this template are operator prioritizing. So they
 * first seek out the operators in the input and tokenize them.
 *
 * To change this template, choose Tools | Templates and open the template in
 * the editor.
 */
package com.github.gbenroscience.parser;

/**
 * TASK: build regex for matching input expressions...e.g. ^([-+/*]\d+(\.\d+)?)*
 */
//sin2+cos3-9tan(A+3B)
//[sin,2,+,cos,3,-,9,tan (,A,+,3,B,)]
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.VariableManager;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.Method;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import static com.github.gbenroscience.parser.STRING.*;
import static com.github.gbenroscience.parser.Operator.*;
import static com.github.gbenroscience.parser.Variable.*;
import static com.github.gbenroscience.parser.Number.*;

import com.github.gbenroscience.math.differentialcalculus.Parser;
import static com.github.gbenroscience.parser.TYPE.ALGEBRAIC_EXPRESSION;
import static com.github.gbenroscience.parser.TYPE.MATRIX;
import static com.github.gbenroscience.parser.TYPE.NUMBER;
import static com.github.gbenroscience.parser.TYPE.VECTOR;
import com.github.gbenroscience.util.ErrorLog;
import com.github.gbenroscience.util.StringManager;
import java.util.function.Predicate;

/**
 * Tokenizes a math expression and normalises the token stream (implicit
 * multiplication, unary signs, anonymous functions, variable registration...).
 * <p>
 * Performance notes (behaviour-preserving):
 * <ul>
 * <li>The token vocabulary and the configured {@link Scanner} are cached in
 * {@link TokenTables} instead of being rebuilt for every expression.</li>
 * <li>Input pre-processing is a single pass over the characters (the old code
 * made two passes, each calling {@code substring} per character and rebuilding
 * the whole string on every replacement).</li>
 * <li>Cheap character/length tests run before any map lookup, regex-free number
 * validation avoids throwing {@code NumberFormatException} for non numbers, and
 * every token is classified once per pass instead of up to four times.</li>
 * <li>Exceptions are no longer used (and logged with a stack trace) as ordinary
 * control flow at the end of the token list.</li>
 * </ul>
 *
 * @author GBEMIRO
 */
public class MathScanner {

    public String commaAlias = newCommaAlias();
    public ParserResult parser_Result = ParserResult.VALID;
    /**
     * Returns true if the expression is validated by the scanner as containing
     * all the necessary objects and no foreign object. This does not validate
     * syntax but only validates that the objects used are the valid ones of a
     * mathematical language
     */

    /**
     * Contains a list of Variable objects used that are not declared.
     */
    ErrorLog errorLog = new ErrorLog();
    private boolean runnable = true;

    /**
     * The math expression to be scanned.
     *
     */
    private String scannerInput;

    /**
     * Contains the scanned expression
     */
    private List<String> scanner = new ArrayList<>();

    /**
     * Decides whether an identifier-shaped run of characters is a variable (or
     * anonymous function) name. Stateless, so one shared instance serves every
     * scan instead of a fresh lambda per call.
     */
    static final java.util.function.Predicate<String> DYNAMIC_RULES
            = word -> FunctionManager.isAnonymousFormat(word) || word.startsWith("_") || isVariableString(word);

    private static String newCommaAlias() {
        // ThreadLocalRandom: no new Random() (and its seed bookkeeping) per scan.
        return String.valueOf(ThreadLocalRandom.current().nextDouble());
    }

    /**
     *
     * @param scannerInput the input of this Scanner object
     * @param me
     */
    public MathScanner(String scannerInput, MathExpression me) throws InputMismatchException {

        scannerInput = normalise(scannerInput.replace("-¹", "^-1"));

        DataSetFormatter dsf = new DataSetFormatter(scannerInput);
        this.scannerInput = dsf.getFormattedDataSet();
        scan(me);
    }

    private static boolean isAsciiDigit(char c) {
        return c >= '0' && c <= '9';
    }

    /**
     * Single-pass replacement of what used to be two substring-per-character
     * loops:
     * <ol>
     * <li>{@code π} to {@code pi}, {@code ×} to {@code *}, {@code ÷} to
     * {@code /}, and the sign of an exponent ({@code 2E+5}, {@code 2e-5}) to
     * {@code ±} / the en-dash so it is not mistaken for an operator while
     * tokenizing.</li>
     * <li>The {@code -} of a negative number following a comma
     * ({@code 3,-2}) to {@code ~} so it is not read as subtraction in data
     * sets.</li>
     * </ol>
     * Returns the very same String instance (no allocation at all) when there is
     * nothing to change.
     */
    private static String normalise(String s) {
        final int n = s.length();
        int first = -1;
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c == 'π' || c == '×' || c == '÷') {
                first = i;
                break;
            }
            if (c == ',' && i + 2 < n && s.charAt(i + 1) == '-') {
                first = i; // only ",-" followed by something can need the '~' marker
                break;
            }
            if ((c == 'e' || c == 'E') && i + 2 < n && isAsciiDigit(s.charAt(i + 2))) {
                char sg = s.charAt(i + 1);
                if (sg == '+' || sg == '-') {
                    first = i;
                    break;
                }
            }
        }
        if (first < 0) {
            return s;
        }

        final char enDash = EN_DASH.charAt(0);
        StringBuilder sb = new StringBuilder(n + 8);
        sb.append(s, 0, first);
        for (int i = first; i < n; i++) {
            char c = s.charAt(i);
            switch (c) {
                case 'π':
                    sb.append("pi");
                    break;
                case '×':
                    sb.append(MULTIPLY);
                    break;
                case '÷':
                    sb.append(DIVIDE);
                    break;
                case 'e':
                case 'E':
                    sb.append(c);
                    if (i + 2 < n && isAsciiDigit(s.charAt(i + 2))) {
                        char sg = s.charAt(i + 1);
                        if (sg == '+') {
                            sb.append('±');
                            i++;
                        } else if (sg == '-') {
                            sb.append(enDash);
                            i++;
                        }
                    }
                    break;
                case ',':
                    sb.append(c);
                    // A comma at (or one char from) the end has nothing after it to
                    // mark. This used to throw StringIndexOutOfBoundsException from the
                    // constructor for inputs such as "3," and "3,-"; now it is left to
                    // the later syntax checks like any other malformed input.
                    if (i + 2 < n && s.charAt(i + 1) == '-') {
                        char d = s.charAt(i + 2);
                        if (d == '.' || isAsciiDigit(d)) {
                            sb.append('~');
                            i++;
                        }
                    }
                    break;
                default:
                    sb.append(c);
            }
        }
        return sb.toString();
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
     * @return the input into the scan.
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
    public List<String> getScanner() {
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

    /** Same test as {@link STRING#isNumberComponent(String)}, on a char. */
    private static boolean isNumComp(char c) {
        return Character.isDigit(c) || c == '.' || c == 'E' || c == '-' || c == '+';
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
         * The six parts were previously kept in six Strings, extended with
         * `part += val.substring(i, i + 1)` and concatenated at the end. The parts
         * can only grow in the order a+p+b+e+s+c as the cursor moves right, so
         * appending to one buffer in encounter order yields the identical result.
         */
        final int ind = getFirstIndexOfDigit(val);
        if (ind < 0) {
            return "";
        }

        int plusCount = 0;
        int minusCount = 0;
        int expCount = 0;
        int pointCount = 0;
        boolean hasSign = false;
        boolean hasExpDigits = false;
        StringBuilder sb = new StringBuilder(val.length() - ind + 1);

        final boolean leadingPoint = ind > 0 && val.charAt(ind - 1) == '.';
        if (leadingPoint) {
            sb.append('.');
            pointCount++;
        }

        for (int i = ind, n = val.length(); i < n; i++) {
            char c = val.charAt(i);
            if (!isNumComp(c)) {
                break;
            }
            if (c == '.') {
                //a number cannot contain more than 1 decimal point, nor one after the exp symbol
                if (leadingPoint || expCount == 1 || pointCount == 1) {
                    break;
                }
                pointCount++;
                sb.append('.');
            } else if (c == '+' || c == '-') {
                //an operator not following an exp symbol ends the number
                if (expCount == 0) {
                    break;
                }
                if (plusCount == 1 || minusCount == 1) {
                    break;
                }
                if (c == '+') {
                    plusCount++;
                } else {
                    minusCount++;
                }
                if (!hasExpDigits && !hasSign) {
                    sb.append(c);
                    hasSign = true;
                }
            } else if (c == 'E') {
                if (expCount == 1) {
                    break;
                }
                sb.append('E');
                expCount++;
            } else if (c >= '0' && c <= '9') {
                sb.append(c);
                if (expCount == 1) {
                    hasExpDigits = true;
                }
            }
            // any other Unicode digit is a number component but contributes nothing
        }//end for

//klnsdkn23.94E2jwjji
        //klwejh.7856E++43
        int len = sb.length();
        if (len > 0 && (sb.charAt(len - 1) == '+' || sb.charAt(len - 1) == '-')) {
            len--;
        }
        if (len > 0 && (sb.charAt(len - 1) == '+' || sb.charAt(len - 1) == '-')) {
            len--;
        }
        if (len > 0 && sb.charAt(len - 1) == 'E') {
            len--;
        }
        return sb.substring(0, len);
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
    public List<String> splitStringAtFirstNumber(String val) {
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
    public List<String> splitStringOnNumbers(String val) {

        List<String> scan = new ArrayList<>();
        List<String> split;

        boolean canSplit = getFirstIndexOfDigitOrPoint(val) > -1;
        if (canSplit) {
            while (canSplit) {
                split = splitStringAtFirstNumber(val);//the splitting
                int sizeAfterSplit = split.size();//the size after the splitting occurs

                val = split.get(sizeAfterSplit - 1);//get the new string to split if it is still splittable on numbers.

                scan.addAll(split.subList(0, sizeAfterSplit - 1));
                canSplit = getFirstIndexOfDigitOrPoint(val) > -1;

                if (!canSplit) {
                    scan.add(val);
                    break;
                }
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
        int extent;//measures the length of each number substring
        int size;
        val = " " + val;
        while (firstOcc != 0) {
            firstOcc = getFirstIndexOfDigit(val);

            if (firstOcc > 0 && val.charAt(firstOcc - 1) == '.') {
                firstOcc -= 1;
            }

            String num = getFirstNumberSubstring(val);
            split.add(num);//store number token in split.
            size = num.length();//get the size of the last element in the ArrayList
            extent = firstOcc + size;//index of the remaining part of the string ro be processed
            val = val.substring(extent);

        }
        split.removeIf(String::isEmpty);
        return split;
    }

    /**
     *
     * Split the {@link MathScanner#scannerInput} String on the operators.
     */
    public void splitStringOnMethods_Variables_And_Operators() throws InputMismatchException {

        // A pre-configured scanner (cached vocabulary, delimiters, dynamic matcher)
        // re-targeted at this input: no token arrays or hash tables are rebuilt.
        scanner = TokenTables.mathScanner(scannerInput).scan();

        for (int i = 0; i < scanner.size(); i++) {
            String token = scanner.get(i);
            int len = token.length();
            char first = token.charAt(0);
            if ((first == '"' && token.charAt(len - 1) == '"') || (first == '\'' && token.charAt(len - 1) == '\'')) {
                token = token.substring(1, len - 1);
                StringManager.add(token);
                scanner.set(i, token);
            }

            if (isNumber(token) && !validNumber(token)) {

                for (int j = 0; j < token.length(); j++) {//3.4567A
                    if (!isNumComp(token.charAt(j)) && isVariableString(token.substring(j))) {
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
            if (i + 1 < scanner.size()) {
                String next = scanner.get(i + 1);
                // Both of the first two branches need `next` to be a variable name, and
                // the third needs it to be "(" (which never is), so decide that once.
                if (isVariableString(next)) {
                    if (validNumber(token) || isVariableString(token)) {
                        scanner.add(i + 1, "*");
                    } /**
                     * Allow early conversion of )sin(k) into )*sin(k)
                     */
                    else if (isClosingCircBracket(token) && !Method.isListReturningStatsMethod(next)) {
                        /**
                         * Check if the bracket owner allows for multiplication.
                         */
                        int open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, scanner);
                        if (open > 0 && !isAtOperator(scanner.get(open - 1)) && !Method.isListReturningStatsMethod(scanner.get(open - 1))) {
                            scanner.add(i + 1, "*");
                        }
                    }
                } else if (isOpeningCircBracket(next) && isNumber(token)) {
                    scanner.add(i + 1, "*");
                }
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
         * (The last token has no successor, so it can never be re-coupled; the
         * old code reached that conclusion by throwing and logging an
         * IndexOutOfBoundsException with a full stack trace. We simply stop one
         * token earlier.)
         */
        for (int i = 0; i + 1 < scanner.size(); i++) {
            String cur = scanner.get(i);
            if (Method.isMethodName(cur)) {
                String nxt = scanner.get(i + 1);
                if (isNumber(nxt) || Method.isMethodName(nxt)) {
                    scanner.set(i, cur.concat(nxt));
                    scanner.remove(i + 1);
                    --i;
                }//end if
            }
        }//end for loop

//sort(5,3,2,1,-8,-9,12,34,98,-900,34,23,12,340)
        final String enDash = EN_DASH;
        for (int i = 0; i < scanner.size(); i++) {//±–
            String cur = scanner.get(i);
            int pm = cur.indexOf('±');
            if (pm >= 0) {
                scanner.set(i, cur.substring(0, pm) + '+' + cur.substring(pm + 1));
            } else {
                int dash = cur.indexOf(enDash);//replace the en-dash symbol with the minus (-) symbol again..
                if (dash >= 0) {
                    scanner.set(i, cur.substring(0, dash) + '-' + cur.substring(dash + 1));
                }
            }
            // i + 2 must exist: a "(" in the last two positions (an unclosed bracket such
            // as "3+(" or "x+sin(") used to throw IndexOutOfBoundsException here. It is
            // now left for the bracket-mapping stage, which reports PARENTHESES_ERROR.
            if (i + 2 < scanner.size() && isOpeningCircBracket(scanner.get(i)) && scanner.get(i + 1).equals("-") && isNumber(scanner.get(i + 2))) {

                String num = scanner.get(i + 2);
                // The number at i+2 has not been visited by this loop yet, so its exponent
                // sign may still be masked: EN_DASH for "E-" and '±' for "E+". Restore
                // both before parsing ("(-2E+5)" used to throw NumberFormatException
                // because only the EN_DASH was handled).
                int index = num.indexOf(EN_DASH);
                if (index != -1) {
                    num = replace(num, MINUS, index, index + 1);
                }
                int plusMarker = num.indexOf('±');
                if (plusMarker != -1) {
                    num = num.substring(0, plusMarker) + '+' + num.substring(plusMarker + 1);
                }
                scanner.set(i + 2, num);
                scanner.set(i + 1, String.valueOf(-1 * Double.parseDouble(num)));
                scanner.remove(i + 2);
            }
            /**
             * Convert the ~ symbol back to - for negative numbers in stats
             * methods.
             */
            String now = scanner.get(i);
            if (now.charAt(0) == '~') {
                scanner.set(i, "-" + now.substring(1));
            }

        }//end if

        removeExcessBrackets(scanner);
        recognizeAnonymousFunctions(scanner);
        for (int i = 0; i + 1 < scanner.size(); i++) {
            String nextToken = scanner.get(i + 1);
            // Every branch below requires the next token to be "(".
            if (!isOpeningCircBracket(nextToken)) {
                continue;
            }
            String token = scanner.get(i);
            if (token.equals("diff")) {
                ///diff,(,@,(x),log,(,x, , ,2,), , ,4,)
                int close = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);
                Parser.parseDerivativeCommand(list);
                if (list.isEmpty()) {
                    parser_Result = ParserResult.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }

            }//end if
            else if (token.equals("intg")) {
// intg,(,@,(x),log(x,2),4,2,5)

                int close = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);
                //IF THINGS GO BAD, UNCOMMENT HERE---1 NumericalIntegral.extractFunctionStringFromExpression(list);
                if (list.isEmpty()) {
                    parser_Result = ParserResult.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }

            }//end else if
            else if (Method.isMatrixMethod(token)) {
// matrix_mul,(,@,(x),log(x,2),4,8)
                int close = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);
                if (list.isEmpty()) {
                    parser_Result = ParserResult.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }

            }//end else if
            else if (token.equals("root")) {
// root,(,@,(x),log(x,2),4,2,5)
                int close = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                List<String> list = scanner.subList(i, close + 1);

                if (list.isEmpty()) {
                    parser_Result = ParserResult.INCOMPLETE_PARAMS;
                    setRunnable(false);
                }
            }//end else if
        }//end for.

        //Catch errors like: sin(2,3+4) or cosh(4,9-2) etc
        for (int i = 0; i < scanner.size(); i++) {
            String token = scanner.get(i);
            if (Method.isDefinedMethod(token) && !Method.isMultiArgsMethod(token)) {
                int op = i + 1;
                int cl = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, op, scanner);

                for (int cursor = op + 1; cursor < cl; cursor++) {
                    String tkn = scanner.get(cursor);
                    if (isOpeningCircBracket(tkn)) {
                        //skip to bracket close and continue searching for commas
                        cursor = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, cursor, scanner);
                    } else if (isComma(tkn)) {
                        parser_Result = ParserResult.SYNTAX_ERROR;
                        setRunnable(false);
                        break;
                    }
                }

            }
        }

        // Mask the commas with a number-like alias that cannot clash with a real
        // token. One pass finds both "does any token equal the alias" and "are
        // there any commas at all"; the second pass runs only if there are commas.
        String alias = commaAlias;
        boolean hasComma;
        while (true) {
            hasComma = false;
            boolean clash = false;
            for (int i = 0, n = scanner.size(); i < n; i++) {
                String t = scanner.get(i);
                if (t.equals(alias)) {
                    clash = true;
                    break;
                }
                if (!hasComma && isComma(t)) {
                    hasComma = true;
                }
            }
            if (!clash) {
                break;
            }
            alias = newCommaAlias();
        }
        this.commaAlias = alias;
        if (hasComma) {
            for (int i = 0, n = scanner.size(); i < n; i++) {
                if (isComma(scanner.get(i))) {
                    scanner.set(i, alias);
                }
            }
        }
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
     * @param me
     *
     */
    public void validateInputAfterSplitOnMethodsAndOps(MathExpression me) {
        for (int i = 0; i < scanner.size(); i++) {
            try {
                final String tok = scanner.get(i);
                final boolean canCheckNextToken = i + 1 < scanner.size();
                final String next = canCheckNextToken ? scanner.get(i + 1) : null;
                final boolean nextIsOpen = canCheckNextToken && isOpeningCircBracket(next);

                /**
                 * Skip over the contents of stats methods. Move the cursor (i)
                 * to the end of the closing bracket. This will allow stats
                 * methods to be able to handle the production:
                 * sum(2,3,4,sin(3),5..) and not be limited to sum(3,4,5,6,7,..)
                 * ALSO Skip the function operating methods...their syntax is
                 * too complex to be analyzed here.
                 */
                if (nextIsOpen && (Method.isFunctionOperatingMethod(tok) || Method.isMultiArgsMethod(tok))) {
                    i = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                }//end if

                /**
                 * Enable the use of number(expr..)...and var(expr...) but avoid
                 * func_name(expr....) Users dont have to enter products of
                 * numbers and bracketed expressions as number*(expr) If
                 * autoInitOn is true, then var(expr) is such that, if var is
                 * not a defined method, and is a valid variable name, put a *
                 * between the var and the bracket.
                 */
                else if (nextIsOpen && isNumber(tok)) {
                    scanner.add(i + 1, "*");
                }//end if
                else if (nextIsOpen && isVariableString(tok) && !Method.isDefinedMethod(tok)) {
                    if (me != null || VariableManager.lookUp(tok) != null) {
                        scanner.add(i + 1, "*");
                        i++;
                    } else {
                        parser_Result = ParserResult.UNDEFINED_ARG;
                        setRunnable(false);
                        errorLog.info(tok + " is an undefined variable. Set MathExpression.setAutoInitOn to true to use a variable without defining it");
                    }
                }//end if
                /**
                 * Enable the use of number-concat-funcName(...)...e.g.
                 * 2sin(3)...becomes 2*sin(3)
                 */
                else if (canCheckNextToken && isNumber(tok) && Method.isDefinedMethod(next)) {
                    scanner.add(i + 1, "*");
                    i++;
                }//end if
                /**
                 * Allow things like: (expr)3+1 or (expr)A+... where A is a
                 * variable or method. Force the user to enter it as
                 * (expr)*number or (expr)*A..
                 */
                else if (canCheckNextToken && isClosingCircBracket(tok) && (isVariableString(next) || isNumber(next))) {
                    scanner.add(i + 1, "*");
                    i++;
                }//end else if
                /**
                 * convert ),( into ),*,( e.g (3+4)(5+6) becomes (3+4)*(5+6)
                 * which both the parser and math understand
                 */
                else if (nextIsOpen && isClosingCircBracket(tok)) {
                    scanner.add(i + 1, "*");
                    i++;
                }

            }//end try
            catch (IndexOutOfBoundsException boundsException) {
                errorLog.error(boundsException);
                errorLog.log(ErrorLog.Level.ERROR, "More information on error... Expression was:\n" + scannerInput + ", scanner-state was:\n" + scanner);
            }//end catch

        }//end for loop

    }//end method validateInput...

    private boolean isConstExpr(int startIdx, int endIdx) {
        for (int i = startIdx; i <= endIdx; i++) {
            String token = scanner.get(i);
            if (FunctionManager.lookUp(token) != null) {
                return false;
            }
            if (VariableManager.lookUp(token) != null) {
                if (!Variable.isSystemConstant(token)) {
                    return false;
                }
            }
            if (isVariableString(token) && !Method.isDefinedMethod(token)) {
                return false;
            }
        }
        return true;
    }

    private void processNestedExpressions() {
        for (int i = 0; i + 1 < scanner.size(); i++) {
            // Cheapest test first: a function call needs "(" right after the name.
            if (!isOpeningCircBracket(scanner.get(i + 1))) {
                continue;
            }
            String token = scanner.get(i);
            {
                Function f = FunctionManager.lookUp(token);
                if (f != null && f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {

                    int arity = f.getArity();
                    /**
                     * Detect if it is a function call...e.g. f(3,2) or the
                     * function is being used as part of its host
                     * expression..e.g. new
                     * MathExpression("3+sin(x)+f(x,5*y,2*z-5)")
                     */
                    int closeBracOfFunctionCall = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                    /**
                     * The function is being used in another expression if all
                     * the args passed to it are expressions themselves or
                     * variables. If any of its args are constants, treat the
                     * function as a function call and evaluate it based on the
                     * values of the variables/functions passed to it .
                     */
                    int argStart = -1;//the start of the first argument

                    int argCount = 0;
                    boolean allArgsAreVariableExpressions = true;
                    boolean allArgsAreConstExpressions = true;

                    List<List<String>> argVectors = new ArrayList<>();
                    for (int j = i + 1; j <= closeBracOfFunctionCall; j++) {//j=i+2---start after the bracket that comes after the function name
                        String cToken = scanner.get(j);

                        if (cToken.equals("(")) {
                            if (j == i + 1) {
                                argStart = j;
                            } else {//skip the open bracket ...jump to its close to avoid any complexities inside...e.g. nested commas, more brackets etc
                                j = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, j, scanner);
                            }
                        } else if (cToken.equals(")")) {
                            if (j == closeBracOfFunctionCall) {
                                //capture arg here
                                List<String> l = new ArrayList<>(scanner.subList(argStart + 1, j));
                                argVectors.add(l);
                                if (isConstExpr(argStart, j)) {
                                    allArgsAreVariableExpressions = false;
                                } else {
                                    allArgsAreConstExpressions = false;
                                }
                                argCount++;
                            }
                        } else if (cToken.equals(commaAlias)) {
                            //capture arg here
                            List<String> l = new ArrayList<>(scanner.subList(argStart + 1, j));
                            argVectors.add(l);
                            if (isConstExpr(argStart + 1, j)) {
                                allArgsAreVariableExpressions = false;
                            } else {
                                allArgsAreConstExpressions = false;
                            }
                            argCount++;
                            argStart = j;
                        }
                    }

                    if (argCount != arity) {
                        throw new RuntimeException("Function(" + f.getName() + " requires `" + arity + "` parameters, but " + argCount + " parameters were supplied!");
                    }
                    //f(x,y,z)=sin(x)+cos(y-3*z)+5*z
                    if (allArgsAreConstExpressions) {
                        //g(x,y,z)=3*x+f(3,5,2) --- the f(3,5,2) portion will be treated as a function call with default values being assigned to x and y.
                        MathExpression transform = Function.transformExpr(f, argVectors, commaAlias);
                        MathExpression.EvalResult ev = transform.solveGeneric();
                        scanner.subList(i, closeBracOfFunctionCall + 1).clear();

                        switch (transform.getReturnType()) {
                            case MATRIX:
                                scanner.add(i, transform.getReturnObjectName());
                                break;
                            case ALGEBRAIC_EXPRESSION:
                                scanner.add(i, transform.getReturnObjectName());
                                break;
                            case VECTOR:
                                scanner.add(i, transform.getReturnObjectName());
                                break;
                            case NUMBER:
                                scanner.add(i, String.valueOf(ev.scalar));
                                break;
                            default:
                                break;
                        }//end switch

                    } else {//at least one of the args is a variable expression
                        //g(x,y,z)=3*x+f(x,y,z) --- the f(x,y,z) portion will be treated as an expression
                        //g(x,y,z)=3*x+f(x,y,2) --- the f(x,y,2) portion will be treated as a function call with default values being assigned to x and y.
                        //g(x,y,z)=3*x+f(1,y,2) --- the f(1,y,2) portion will be treated as a function call with default values being assigned to x and y.
                        //g(x,y,z)=3*x+f(3,5,2) --- the f(3,5,2) portion will be treated as a function call with default values being assigned to x and y.
                        List<String> scan = Function.transformScan(f, argVectors);
                        scanner.subList(i, closeBracOfFunctionCall + 1).clear();
                        scanner.addAll(i, scan);
                    }

                }
            }
        }
    }

    /**
     * Identifies that the input is a valid one by checking that all tokens are
     * either Number objects, Variable objects or Operator objects. Then it
     * registers any Variable object found and initializes it to zero.
     */
    private void validateTokens() {
        for (int i = 0; i < scanner.size(); i++) {
            String tok = scanner.get(i);
            if (!isRecognisedToken(tok)) {
                parser_Result = ParserResult.STRANGE_INPUT;
                setRunnable(false);
                errorLog.info(tok + " is a strange math object!");
                return;
            }//end if
        }//end for
    }//end validateTokens

    /**
     * True if the token is an operator, a variable name, a number or a method
     * name. The four tests are pure, so the order is free: numbers (the most
     * common token after operators) are recognised by their first character
     * before any map lookup is made.
     */
    private static boolean isRecognisedToken(String tok) {
        char c = tok.charAt(0);
        if ((c >= '0' && c <= '9') || c == '.') {
            if (validNumber(tok)) {
                return true;
            }
        }
        return isOperatorString(tok) || isVariableString(tok) || validNumber(tok) || Method.isMethodName(tok);
    }

    private void plusAndMinusStringHandler() {
        scanner = plusAndMinusStringHandlerHelper(scanner);
    }

    /**
     * Handles unary plus and minus operations on the numbers that come after
     * them Also handles repeated concatenations of plus and minus operators.
     */
    public static final List<String> plusAndMinusStringHandlerHelper1(List<String> scanner) {
        List<String> result = new ArrayList<>();

        for (int i = 0; i < scanner.size(); i++) {
            String tk = scanner.get(i);

            if (isSign(tk)) {
                // 1. Squash consecutive signs: e.g., --+- becomes -
                int signMultiplier = 1;
                int j = i;
                while (j < scanner.size() && isSign(scanner.get(j))) {
                    if (scanner.get(j).equals("-")) {
                        signMultiplier *= -1;
                    }
                    j++;
                }

                String squashedSign = (signMultiplier == 1) ? "+" : "-";

                // 2. Check Context: Is there a number immediately after the sign chain?
                boolean hasNextNumber = (j < scanner.size() && isNumber(scanner.get(j)));

                // 3. Check if we are in a 'unary' position 
                // (Start of list, after '(', or after an operator)
                boolean isUnaryPos = (result.isEmpty()
                        || result.get(result.size() - 1).equals("(")
                        || isOperator(result.get(result.size() - 1)));

                if (isUnaryPos && hasNextNumber) {
                    // MERGE: Turn ["-", "-", "12"] into ["12.0"] or ["-", "-", "3"] into ["-3.0"]
                    double val = Double.parseDouble(scanner.get(j));
                    if (signMultiplier == -1) {
                        val *= -1;
                    }

                    result.add(String.valueOf(val));
                    i = j; // Advance main loop past the signs and the number
                } else {
                    // LEAVE AS OPERATOR: It's either binary (5 - 3) 
                    // or unary before a bracket -(... ) which can't be merged yet
                    result.add(squashedSign);
                    i = j - 1; // Advance main loop past the extra signs
                }
                continue;
            }

            // Add non-sign tokens (numbers, brackets, special operators) as they are
            result.add(tk);
        }
        return result;
    }

    public static final List<String> plusAndMinusStringHandlerHelper(List<String> scanner) {
        final int size = scanner.size();
        List<String> result = new ArrayList<>(size);

        for (int i = 0; i < size; i++) {
            String tk = scanner.get(i);

            // --- 1. HANDLE SIGN CHAINS (+, -, ---, etc) ---
            if (isSign(tk)) {
                int signMultiplier = 1;
                int j = i;
                // Squash: --+- -> -
                while (j < size && isSign(scanner.get(j))) {
                    if (scanner.get(j).equals("-")) {
                        signMultiplier *= -1;
                    }
                    j++;
                }

                // CONTEXT CHECK: Is this sign at the start, after a bracket, or after an operator?
                boolean isUnaryPos = result.isEmpty() || isOperator(result.get(result.size() - 1));

                if (isUnaryPos) {
                    // Peek at what follows the sign chain
                    if (j < size && isNumber(scanner.get(j))) {
                        // Case: -5 -> Merge into one token "-5.0"
                        double val = Double.parseDouble(scanner.get(j));
                        result.add(String.valueOf(val * signMultiplier));
                        i = j;
                    } else {
                        // Case: -sin(x) or -(5+2) -> Convert to -1 * ...
                        if (signMultiplier == -1) {
                            result.add("-1");
                            result.add("*");
                        }
                        // If it's a unary +, we just discard it as it's mathematically neutral
                        i = j - 1;
                    }
                } else {
                    // Binary case: It's an addition or subtraction operator (e.g., 5 - 3)
                    result.add(signMultiplier == 1 ? "+" : "-");
                    i = j - 1;
                }
                continue;//2²+3³+√9
            }

            // --- 2. REDUNDANT "* 1" or "/ 1" REMOVAL ---
            if (tk.length() == 1 && (tk.charAt(0) == '*' || tk.charAt(0) == '/') && i + 1 < size && isExactlyOne(scanner.get(i + 1))) {
                // Only remove if the previous token is "operand-like" (number, variable, or closing bracket/factorial)
                if (!result.isEmpty() && isOperandLike(result.get(result.size() - 1))) {
                    i++; // Skip the operator and the '1'
                    continue;
                }
            }

            result.add(tk);
        }
        return result;
    }

// --- HELPER METHODS ---
    private static boolean isOperator(String s) {
        // These are tokens that, if they appear BEFORE a sign, make that sign UNARY
        // (A switch on the string hashes once instead of up to 18 equals() calls.)
        switch (s) {
            case "+":
            case "-":
            case "*":
            case "/":
            case "<":
            case ">":
            case "<=":
            case ">=":
            case "==":
            case "!=":
            case "&&":
            case "||":
            case "^":
            case "%":
            case "Р":
            case "Č":
            case "(":
            case "√":
                return true;
            default:
                return false;
        }
    }

    private static boolean isOperandLike(String s) {
        // These are tokens that can be followed by a redundant * 1
        // Includes numbers, variables (x), closing brackets, and factorials
        return isNumber(s) || s.equals(")") || s.equals("!") || isVariableString(s) || s.equals("²") || s.equals("³");
    }

    private static boolean isSign(String s) {
        return s.length() == 1 && (s.charAt(0) == '+' || s.charAt(0) == '-');
    }

    private static boolean isExactlyOne(String s) {
        // Same result as "Double.parseDouble(s) == 1.0, or false if it does not
        // parse", but a non-number (a variable name, say) no longer costs a
        // thrown-and-caught NumberFormatException.
        return validNumber(s) && Double.parseDouble(s) == 1.0;
    }

    /**
     * Utility method,more popularly used as a scan into mathematical tokens of
     * mathematical expressions
     *
     * @return an ArrayList containing a properly scanned version of the string
     * such that all recognized number substrings,variable substrings and
     * operator substrings have been resolved. To use this method as a scan for
     * math expressions all the programmer has to do is to take care of the
     * signs where need be(e.g in -2.873*5R+sinh34.2, the scanned view will be
     * [-,2.873,*,5,R,+,sinh,34.2]) To make sense of this, the programmer has to
     * error minor code to concatenate the - and the 2.873 and so on. 
     * @param me
     * @return 
     */
    private List<String> scan(MathExpression me) throws InputMismatchException {
        splitStringOnMethods_Variables_And_Operators();
        validateInputAfterSplitOnMethodsAndOps(me);

        /*
         * The for loop above does not properly handle
         * negative exponents of 10, e.g -3E-10
         * It splits it into -3,E, -10
         * So we need to manually couple the split objects together
         *
         */
        for (int i = 0; i < scanner.size(); i++) {
            try {
                String tok = scanner.get(i);
                if (isNegativeNumberToken(tok) && scanner.get(i + 1).equals("E")
                        && Number.isNumber(scanner.get(i + 2))) {
                    scanner.set(i, tok + scanner.get(i + 1) + scanner.get(i + 2));
                    scanner.remove(i + 1);
                    scanner.remove(i + 1);

                }//end if
            }//end try
            catch (IndexOutOfBoundsException indexErr) {
                errorLog.error(indexErr);
                break;
            }//end catch//end catch
        }//end for

//enable interpretation of things like 3^-4 or 3^+4 i.e ^- or ^+ patterns
        for (int i = 0; i < scanner.size(); i++) {
            try {
                if (isPower(scanner.get(i))) {
                    String sign = scanner.get(i + 1);
                    if (sign.equals("-") && validNumber(scanner.get(i + 2))) {
                        scanner.set(i + 1, String.valueOf(-1 * Double.parseDouble(scanner.get(i + 2))));
                        scanner.remove(i + 2);
                    } else if (sign.equals("+") && validNumber(scanner.get(i + 2))) {
                        scanner.set(i + 1, String.valueOf(Double.parseDouble(scanner.get(i + 2))));
                        scanner.remove(i + 2);
                    }
                }//end if
            }//end try
            catch (IndexOutOfBoundsException indexErr) {
                errorLog.error(indexErr);
            } catch (NumberFormatException numErr) {
                errorLog.error(numErr);
            }
        }//end for

        validateTokens();
        processNestedExpressions();

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
            if (i + 1 < sz && isOpeningCircBracket(scanner.get(i + 1)) && Method.isFunctionOperatingMethod(scanner.get(i))) {
                int close = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i + 1, scanner);
                i = close;
                continue;
            }//end if

            // Classify the token once instead of re-running isVariableString up to
            // five times per iteration.
            final String cur = scanner.get(i);
            final boolean isVar = isVariableString(cur);
            if (!isVar && !isOperatorString(cur) && !validNumber(cur) && !Method.isMethodName(cur)) {
                errorLog.info("Syntax Error! Strange Object Found: " + cur);
                parser_Result = ParserResult.STRANGE_INPUT;
                setRunnable(false);
            }

            // (Re-read on purpose: if the block above cleared the scanner this throws,
            // exactly as it always did.)
            final String tk = scanner.get(i);
            final boolean hasNext = i + 1 < sz;
            final boolean nextIsOpen = hasNext && isOpeningCircBracket(scanner.get(i + 1));
            if (hasNext && isVar && !nextIsOpen
                    && !FunctionManager.containsAny(tk) && !Method.isDefinedMethod(tk)) {
                declareVariable(tk, me);
            }//end if
            else if (i == sz - 1 && isVar && !FunctionManager.containsAny(tk) && !Method.isDefinedMethod(tk)) {
                declareVariable(tk, me);
            } else if (i == 0 && sz == 1 && isVar) {
                if (!FunctionManager.containsAny(tk)) {
                    declareVariable(tk, me);
                }
            } else {
                if (hasNext && isVar && !nextIsOpen && !VariableManager.contains(tk)
                        && !FunctionManager.containsAny(tk)) {
                    errorLog.info(" Unknown Variable: " + tk + "\n Please Declare And Initialize This Variable Before Using It.\n"
                            + "Use The Command, \'variableName=value\' To Accomplish This.");
                    parser_Result = ParserResult.STRANGE_INPUT;
                    setRunnable(false);
                }//end if
            }//end else
        }//end for loop

        if (!runnable) {
            errorLog.info("\n"
                    + "Sorry, Errors Were Found In Your Expression."
                    + "Please Consult The Help File For Valid Mathematical Syntax.");
            scanner.clear();
        } else {
            errorLog.info("Scan SuccessFul.No Illegal Object Found.\n"
                    + "Putting Scanner On StandBy");
        }

        plusAndMinusStringHandler();

        if (me != null) {
            me.getErrorLog().copyFrom(errorLog);
        }
        return scanner;
    }

    /**
     * Same test as {@code Number.isNegative(tok)} for a non-empty scanner token.
     */
    private static boolean isNegativeNumberToken(String tok) {
        if (tok.charAt(0) != '-' || tok.length() < 2) {
            return false;
        }
        char c = tok.charAt(1);
        return (c >= '0' && c <= '9') || c == '.';
    }

    /**
     * Registers an undeclared variable (value 0, or its global value if it has
     * one) with the expression's registry, or globally when there is no
     * expression. Existing registrations are never overwritten, so the work (and
     * the Variable object that used to be built and discarded on every
     * occurrence of a known variable) is skipped when it is already registered.
     * <p>
     * Do not use saveOrUpdate, so as not to overwrite the gains of assignment
     * statements of same variables in same MathExpression. For example.. MathExpression
     * m= new MathExpression("r=4;3*r"); The assignment phase, runs first via
     * Function.assignObject(assignmentStatement) and saves r as 4 in the
     * registry of this MathExpression. The scanner stage detects `r` in the
     * statement: 3*r, and must not reset it to 0.
     */
    private static void declareVariable(String name, MathExpression me) {
        if (me != null) {
            if (me.registry.lookUp(name) == null) {
                Variable v = VariableManager.lookUp(name);
                me.registry.saveIfNotExists(new Variable(name, v == null ? 0 : v.getValue()));
            }
        } else if (VariableManager.lookUp(name) == null) {
            VariableManager.VARIABLES.put(name, new Variable(name, 0));
        }
    }

    public static void recognizeAnonymousFunctions1(List<String> scanner) {
        int indexOfAt = -1;

        while ((indexOfAt = scanner.indexOf("@")) != -1) {

            /**
             * Process Anonymous functions After this loop, the anonymous
             * function would have been initialized and its anonymous expression
             * replaced with a function name.
             */
            if (isOpeningCircBracket(scanner.get(indexOfAt + 1))) {
                //root(@(x)3*x^2+sin(x)+log(x,2),3,4)

                for (int i = indexOfAt; i < scanner.size(); i++) {
                    String token = scanner.get(i);
                    if (isOpeningCircBracket(token)) {
                        i = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i, scanner);
                    } else if (isComma(token)) {
                        Function f = new Function(LISTS.createStringFrom(scanner, indexOfAt, i));
                        FunctionManager.add(f);
                        List<String> sub = scanner.subList(indexOfAt, i);
                        sub.clear();
                        sub.add(f.getName());
                        break;
                    } else if (isClosingCircBracket(token)) {
                        int open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, scanner);
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
            } else {
                throw new InputMismatchException("Syntax Error occurred while scanning math expression.\nReason: The @ symbol is used exclusively to create functions. Expected: `(`, found: `" + scanner.get(indexOfAt + 1) + "`");
            }
        }
        //System.print.println("scan-debug: "+scan);
    }

    /**
     * Optimized & more correct 2025 version of recognizeAnonymousFunctions
     *
     * Changes & improvements:  
     * Single linear pass (O(n) instead of repeated
     * indexOf → much faster) 
     * Cleaner structure with helper methods  
     * Safer in-place replacement (no risky mutations during iteration) 
     * Exact same replacement logic & output as your original code 
     * Better readability and maintainability 
     * Preserves your original bracket/comma detection rules
     *
     * @param scanner
     */
    public static void recognizeAnonymousFunctions(List<String> scanner) throws InputMismatchException{
        if (scanner == null || scanner.isEmpty()) {
            return;
        }

        int i = 0;
        while (i < scanner.size()) {
            if ("@".equals(scanner.get(i))) {
                if (i + 1 >= scanner.size() || !isOpeningCircBracket(scanner.get(i + 1))) {
                    String found = i + 1 < scanner.size() ? scanner.get(i + 1) : "end of expression";
                    throw new InputMismatchException("Syntax Error occurred while scanning math expression.\n"
                            + "Reason: The @ symbol is used exclusively to create functions. Expected: `(`, found: `" + found + "`");
                }
                i = processOneAnonymousFunction(scanner, i);
            } else {
                i++;
            }
        }
    }

    /**
     * Processes one anonymous function starting at indexOfAt and replaces it
     * with the generated function name (e.g. anon1, anon2...). Returns the
     * index to continue scanning from after replacement.
     */
    private static int processOneAnonymousFunction(List<String> scanner, int indexOfAt) throws InputMismatchException{

        for (int i = indexOfAt; i < scanner.size(); i++) {
            String token = scanner.get(i);
            if (isOpeningCircBracket(token)) {
                i = Bracket.getComplementIndex(true, Bracket.BracketMode.CIRCULAR_OPEN, i, scanner);
            } else if (isComma(token)) {
                replaceWithFunctionName(scanner, indexOfAt, i);
                return indexOfAt + 1;
            } else if (isClosingCircBracket(token)) {
                int open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, scanner);
                if (open < indexOfAt) {
                    replaceWithFunctionName(scanner, indexOfAt, i);
                    return indexOfAt + 1;
                }
            }
        }
        // Reached end of list without comma or parent bracket
        replaceWithFunctionName(scanner, indexOfAt, scanner.size());
        return indexOfAt + 1;
    }

    /**
     * Creates the Function, registers it, and replaces the range with
     * f.getName()
     */
    private static void replaceWithFunctionName(List<String> scanner, int start, int end) throws InputMismatchException{
        String functionText = LISTS.createStringFrom(scanner, start, end);
        Function f = new Function(functionText);
        FunctionManager.add(f);

        // Safe replacement
        scanner.subList(start, end).clear();
        scanner.add(start, f.getName());
    }

    /**
     * This technique will rid tokens of offending brackets up to the last
     * bracket. It assumes that it knows the rules that allow one to remove all
     * brackets from a token. One however needs check the ruls to be sure that
     * there is no error. If this one messes up, please switch to the {@link MathScanner#$removeExcessBrackets(List)
     * } method.
     *
     * @param scanner The list of scanned tokens.
     *
     */
    public static void removeExcessBrackets(List<String> scanner) {
        BracketCleaner.removeExcessBrackets(scanner);
    }

    /**
     * Old version of {@linkplain MathScanner#removeExcessBrackets(java.util.List)
     * }
     *
     * @deprecated
     * @param scanner
     */
    public static void removeExcessBrackets1(List<String> scanner) {

        for (int i = 0; i < scanner.size(); i++) {

            if (isClosingCircBracket(scanner.get(i))) {

                if (i + 1 < scanner.size() && isOpeningCircBracket(scanner.get(i + 1))) {//if the pattern is )(...ignore the closing bracket.
                    continue;
                }

                int open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, scanner);
                if (i - open == 2) {//confirm that the bracket has only 1 token inside it
                    if (open > 0) {//open bracket is beyond 0
                        String token = scanner.get(open - 1);

                        if (!isVariableString(token) && !isAtOperator(token) && !Method.isMethodName(token) && !isUnaryPreOperator(token)) {
                            scanner.remove(i);
                            scanner.remove(open);
                            i -= 2;
                            continue;
                        }
                    } else if (open == 0) {//open bracket is at 0
                        scanner.remove(i);
                        scanner.remove(open);
                        i -= 2;
                        continue;
                    }

                }//end if

                if (i + 1 < scanner.size()) {

                    if (isClosingCircBracket(scanner.get(i + 1))) {//you have a match for an unnecessary wasted bracket
                        int inner_open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, scanner);
                        int outer_open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i + 1, scanner);

                        if (inner_open != -1 && outer_open != -1 && inner_open - 1 == outer_open) {
                            //Skip for log tokens...the bracket is needed to evaluate content.
                            if (outer_open > 1 && Method.isLogToAnyBase(scanner.get(outer_open - 1))) {
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
     * @deprecated 
     * Analyzes the list and extracts the Function string from it.
     *
     * @param list The list to be analyzed. Direct examples would be:
     * intg(@(x)sin(x+1),4,7) intg(F,4,7) where F is a function that has been
     * defined before in the workspace.. and so on.
     *
     * Simplifies the list to the form matrix_method(funcName,params)
     *
     */
    public static void extractFunctionStringFromExpressionForMatrixMethods(List<String> list) {

        int sz = list.size();
        /**
         * Confirm that there remains only one open and one close bracket in the
         * tokens list.
         */
        if (list.indexOf("(") == list.lastIndexOf("(") && list.indexOf(")") == list.lastIndexOf(")")) {
            //det,(,A,) or matrix_mul,(,A, , ,B,)
//System.print.println("list: "+list);
            if (sz == 4 || sz == 6) {
                if (Method.isMatrixMethod(list.get(0)) && isOpeningCircBracket(list.get(1)) && Method.isUserDefinedFunction(list.get(2))) {
                    if (sz == 4 && isClosingCircBracket(list.get(3))) {
                        // Method.run(list, Declarations.degGradRadFromVariable());
                    } else if (sz == 6 && (isComma(list.get(3)) && Method.isUserDefinedFunction(list.get(4)) || isNumber(list.get(4)) || isVariableString(list.get(4))) && isClosingCircBracket(list.get(5))) {
                        //   Method.run(list, Declarations.degGradRadFromVariable());
                    }
                }
            } /**
             * There remains only one open and close bracket, but the parameters
             * have not yet been properly ordered! Most of the matrix methods
             * take one or at most 2 parameters, so if you have more than 2
             * parameters, then check that the first parameter is in the proper
             * format, then run the parser on everything in the second part of
             * the parameters list to get the second parameter.
             */
            else {

            }
        } else {
            for (int i = 0; i < list.size(); i++) {
                if (isClosingCircBracket(list.get(i))) {
                    int open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, list);
                    if (open > 0) {
                        String token = list.get(open - 1);
                        if (Method.isMatrixMethod(token)) {
                            List l = list.subList(open - 1, i + 1);
                            int siz = l.size();
                            extractFunctionStringFromExpressionForMatrixMethods(l);
                            i = i - (siz - l.size());
                        } //Most likely you have gotten to the first parameter...ignore it and process the bracket
                        else if (FunctionManager.containsAlgebraicFunction(token)) {
                            List l = list.subList(open, i + 1);
                            int siz = l.size();
                            MathExpression me = new MathExpression(LISTS.createStringFrom(list, open, i + 1));
                            String val = me.solve();
                            l.clear();
                            switch (me.getReturnType()) {
                                case MATRIX:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case ALGEBRAIC_EXPRESSION:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case VECTOR:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case NUMBER:
                                    l.add(val);
                                    break;
                                default:
                                    break;
                            }//end switch

                            i = i - (siz - l.size());
                        } else if (Method.isMethodName(token) || isUnaryPreOperator(token) || isNumber(token)) {
                            List<String> l = list.subList(open - 1, i + 1);
                            int siz = l.size();
                            String input;
                            if (Method.isMultiArgsMethod(token)) {
                                /**
                                 * Pattern is [sum,(, 2, 3, 4, 6, )] We need to
                                 * convert this into: sum(2,3,4,6) Our job is to
                                 * remove the starting, second and final commas,
                                 * the white spaces, and the opening and closing
                                 * braces[]
                                 */
                                StringBuilder builder = new StringBuilder(token);
                                builder.append("(");
                                for (int j = 2; j < l.size() - 1; j++) {
                                    builder.append(l.get(j)).append(",");
                                }
                                input = builder.substring(0, builder.length() - 1);
                                input = input.concat(")");
                            } else {
                                input = LISTS.createStringFrom(list, open - 1, i + 1);
                            }
                            MathExpression me = new MathExpression(input);
                            String val = me.solve();
                            l.clear();
                            switch (me.getReturnType()) {
                                case MATRIX:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case ALGEBRAIC_EXPRESSION:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case VECTOR:
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
     * @deprecated 
     * Analyzes the list and extracts the Function string from it.
     *
     * @param list The list to be analyzed. Direct examples would be:
     * intg(@(x)sin(x+1),4,7) intg(F,4,7) where F is a function that has been
     * defined before in the workspace.. and so on.
     *
     * Simplifies the list to the form matrix_method(funcName,params)
     *
     */
    public static void extractFunctionStringFromExpressionForMatrixMethods1(List<String> list) {
        list.removeAll(Arrays.asList(","));

        int sz = list.size();
        /**
         * Confirm that there remains only one open and one close bracket in the
         * tokens list.
         */
        if (list.indexOf("(") == list.lastIndexOf("(") && list.indexOf(")") == list.lastIndexOf(")")) {
            //det,(,A,) or matrix_mul,(,A,B,)
            if (sz == 4 || sz == 5) {
                if (Method.isMatrixMethod(list.get(0)) && isOpeningCircBracket(list.get(1)) && Method.isUserDefinedFunction(list.get(2))) {
                    if (sz == 4 && isClosingCircBracket(list.get(3))) {
                        Method.run(list, Declarations.degGradRadFromVariable());
                    } else if (sz == 5 && (Method.isUserDefinedFunction(list.get(3)) || isNumber(list.get(3)) || isVariableString(list.get(3))) && isClosingCircBracket(list.get(4))) {
                        Method.run(list, Declarations.degGradRadFromVariable());
                    }
                }
            } /**
             * There remains only one open and close bracket, but the parameters
             * have not yet been properly ordered! Most of the matrix methods
             * take one or at most 2 parameters, so if you have more than 2
             * parameters, then check that the first parameter is in the proper
             * format, then run the parser on everything in the second part of
             * the parameters list to get the second parameter.
             */
            else {

            }
        } else {
            for (int i = 0; i < list.size(); i++) {
                if (isClosingCircBracket(list.get(i))) {
                    int open = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, i, list);
                    if (open > 0) {
                        String token = list.get(open - 1);
                        if (Method.isMatrixMethod(token)) {
                            List l = list.subList(open - 1, i + 1);
                            int siz = l.size();
                            extractFunctionStringFromExpressionForMatrixMethods(l);

                            i = i - (siz - l.size());
                        } //Most likely you have gotten to the first parameter...ignore it and process the bracket
                        else if (FunctionManager.containsAlgebraicFunction(token)) {
                            List l = list.subList(open, i + 1);
                            int siz = l.size();

                            MathExpression me = new MathExpression(LISTS.createStringFrom(list, open, i + 1));
                            String val = me.solve();
                            l.clear();
                            switch (me.getReturnType()) {
                                case MATRIX:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case ALGEBRAIC_EXPRESSION:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case VECTOR:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case NUMBER:
                                    l.add(val);
                                    break;
                                default:
                                    break;
                            }//end switch
                            i = i - (siz - l.size());
                        } else if (Method.isMethodName(token) || isUnaryPreOperator(token) || isNumber(token)) {
                            List<String> l = list.subList(open - 1, i + 1);
                            int siz = l.size();
                            String input;
                            if (Method.isMultiArgsMethod(token)) {
                                /**
                                 * Pattern is [sum,(, 2, 3, 4, 6, )] We need to
                                 * convert this into: sum(2,3,4,6) Our job is to
                                 * remove the starting, second and final commas,
                                 * the white spaces, and the opening and closing
                                 * braces[]
                                 */
                                StringBuilder builder = new StringBuilder(token);
                                builder.append("(");
                                for (int j = 2; j < l.size() - 1; j++) {
                                    builder.append(l.get(j)).append(",");
                                }
                                input = builder.substring(0, builder.length() - 1);
                                input = input.concat(")");
                            } else {
                                input = LISTS.createStringFrom(list, open - 1, i + 1);
                            }

                            MathExpression me = new MathExpression(input);
                            String val = me.solve();
                            l.clear();
                            switch (me.getReturnType()) {
                                case MATRIX:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case ALGEBRAIC_EXPRESSION:
                                    l.add(me.getReturnObjectName());
                                    break;
                                case VECTOR:
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
     * Hides the commas in the scan output as a double number which is not part
     * of the output
     */
    public void maskCommas() {
        scanner.replaceAll((String t) -> isComma(t) ? this.commaAlias : t);
    }

    /**
     * Brings back the original comma tokens from their masked state
     */
    public void unmaskCommas() {
        scanner.replaceAll((String t) -> t.equals(commaAlias) ? "," : t);
    }

    /**
     *
     * @param args Command line args (((2+3)^2))!-------((25))!-------
     */
    public static void main(String args[]) {//tester method for STRING methods
     
        MathExpression m = new MathExpression("arr = @(10)(4,1,5,8,yy, 32, 22, 2.2, ox, \"1.xxx\")");
        System.out.println(Arrays.toString(FunctionManager.lookUp("arr").getArray()));

        MathExpression m1 = new MathExpression("@(10)(4,1,5,8,yy, 32, 22, 2.2, ox, \"1.xxx\")");
        System.out.println(FunctionManager.FUNCTIONS);
        MathExpression msc = new MathExpression("diffeqn((3*x^2)*y[4]+(5*sin(x))*sin(y[3])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0], 1, 0, @(1,3)(2,3,4))");
         System.out.println(msc.getExpression()+" -> "+msc.scanner);
         System.out.println(msc.getExpression()+" -> "+Arrays.toString(msc.getCachedPostfix()));
        
        MathExpression orig = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,2)");
        System.out.println("f(3,4,2) = " + orig.solve());

        MathExpression finalOrig = new MathExpression("3 + 2*x + f(2, 3*x + sum(4,2,5,sort(9,2,12,25)), 5)");
        System.out.println("scan-output = " + finalOrig.scanner);
        System.out.println("3 + 2*x + f(2, 3*x + sum(4,2,5,sort(9,2,12,25)), 5) = " + finalOrig.solve());

        MathExpression anotherOrig = new MathExpression("3 + 2*x + f(3*x, 4-sin(2*y), 3*x+sum(4,2,5,sort(9,2,12,25)) )");
        System.out.println("scan-output = " + anotherOrig.scanner);
        System.out.println("3 + 2*x + f(3*x, 4-sin(2*y), 3*x + sum(4,2,5,sort(9,2,12,25)) ) = " + anotherOrig.solve());

    }//end method main
}
