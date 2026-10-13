/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.parser.methods.Method;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author GBENRO
 */
public class Number {

    /**
     * The string of digits that represent this Number object.
     */
    private String num;
    /**
     * The location of the Number in the scanner output of the parent
MathExpression object that contains this Number object.
     */
    private int index;

    /**
     * @param num The string of digits that represent this Number object.
     *
     */
    public Number(String num) {
        this.num = num;
    }

    /**
     *
     * @param num The string of digits that represent this Number object.
     * @param index the location of this Operator object in its parent
MathExpression object's scanned ArrayList object.
     */
    public Number(String num, int index, ArrayList<String> scan) {
        this.num = isNumber(num) ? num : "";
        this.index = index;
        if (this.num.equals("")) {
            throw new IndexOutOfBoundsException("Invalid Name For Log or antilog to any base type Operator.");
        }//end if
        else {
            this.index = (index >= 0 && scan.get(index).equals(num)) ? index : -1;
        }//end else
        if (this.index == -1) {
            throw new IndexOutOfBoundsException("Invalid Index");
        }//end if
    }//end constructor

    /**
     *
     * @param index sets the location of this Operator object in its parent
MathExpression
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     *
     * @return the location of this Operator object in its parent MathExpression
     */
    public int getIndex() {
        return index;
    }

    /**
     *
     * @param num sets the string of digits that represent this Number object.
     */
    public void setNum(String num) {
        this.num = num;
    }

    /**
     *
     * @return the string of digits that represent this Number object.
     */
    public String getNum() {
        return num;
    }

    /**
     * This method may be used to test strings to see if or not they represent
     * valid numbers.
     *
     * @param num The string to test.
     * @return true if the string is a valid number
     */
    public static boolean validNumber(String num) {
        // Same answer as "Double.parseDouble(num) does not throw", without paying
        // for a NumberFormatException (stack trace and all) on every token that is
        // not a number -- which, in a scanner, is most of them.
        final int len = num.length(); // NPE on null, like parseDouble
        if (len == 0) {
            return false; // parseDouble("") throws
        }
        final char c = num.charAt(0);
        // parseDouble trims whitespace/control chars (<= ' ') and then needs one of
        // + - . 0-9 or the words NaN / Infinity. Anything else cannot parse.
        if (c > ' ' && c != '+' && c != '-' && c != '.' && c != 'N' && c != 'I' && (c < '0' || c > '9')) {
            return false;
        }
        if (isPlainDecimal(num, len)) {
            return true;
        }
        if (c > ' ') {
            // Not padded with whitespace. Skip the optional sign, then the next char
            // decides whether parseDouble could possibly accept this at all. The
            // overwhelmingly common non-numbers here are the operators "+", "-" and
            // "." (and "-1", "+-" style fragments), which would otherwise pay for a
            // NumberFormatException with a full stack trace.
            final int i = (c == '+' || c == '-') ? 1 : 0;
            if (i >= len) {
                return false; // a bare sign
            }
            final char d = num.charAt(i);
            final boolean maybe;
            if (d >= '0' && d <= '9') {
                maybe = true; // 1d, 2f, 0x1p3, trailing blanks ...
            } else if (d == '.') {
                maybe = len - i > 1; // "." alone is never a number
            } else if (d == 'N') {
                maybe = len - i >= 3; // NaN
            } else if (d == 'I') {
                maybe = len - i >= 8; // Infinity
            } else {
                maybe = false;
            }
            if (!maybe) {
                return false;
            }
            // parseDouble only ever accepts ASCII letters/digits, '.', '+', '-' and
            // padding whitespace/control chars; an expression such as "3+4*2" (which
            // the scanner asks about as a whole) can be rejected without throwing.
            for (int k = 0; k < len; k++) {
                final char ch = num.charAt(k);
                if (ch > ' ' && ch != '.' && ch != '+' && ch != '-'
                        && (ch < '0' || ch > '9') && (ch < 'A' || ch > 'Z') && (ch < 'a' || ch > 'z')) {
                    return false;
                }
            }
        }
        // Rare shapes (whitespace padding, NaN, Infinity, hex floats, 1d / 2f
        // suffixes, malformed input): defer to the JDK so the answer is identical.
        try {
            Double.parseDouble(num);
            return true;
        } catch (NumberFormatException numErr) {
            return false;
        }
    }

    /**
     * Recognises [+-]? (digits [. digits*] | . digits) ([eE] [+-]? digits)? -- a
     * strict subset of what Double.parseDouble accepts.
     */
    private static boolean isPlainDecimal(String s, int len) {
        int i = 0;
        char c = s.charAt(0);
        if (c == '+' || c == '-') {
            i = 1;
        }
        int intDigits = 0;
        while (i < len && (c = s.charAt(i)) >= '0' && c <= '9') {
            i++;
            intDigits++;
        }
        int fracDigits = 0;
        if (i < len && s.charAt(i) == '.') {
            i++;
            while (i < len && (c = s.charAt(i)) >= '0' && c <= '9') {
                i++;
                fracDigits++;
            }
        }
        if (intDigits == 0 && fracDigits == 0) {
            return false;
        }
        if (i < len && ((c = s.charAt(i)) == 'e' || c == 'E')) {
            i++;
            if (i < len && ((c = s.charAt(i)) == '+' || c == '-')) {
                i++;
            }
            int expDigits = 0;
            while (i < len && (c = s.charAt(i)) >= '0' && c <= '9') {
                i++;
                expDigits++;
            }
            if (expDigits == 0) {
                return false;
            }
        }
        return i == len;
    }

    /**
     * Built for use in my parser only.Serves to detect already and properly
     * scanned numbers in the list where string models of other objects reside,
     * e.g Variable objects,Operator objects e.t.c
     *
     * @param num the string to be checked if it is a number or not
     * @return true if the item is a number
     */
    public static boolean isNumber(String num) {
        //sign_num_._num_E_sign_num
        if (num == null) {
            return false;
        }
        // charAt(0) on "" throws StringIndexOutOfBoundsException, exactly as the
        // old STRING.firstChar(num) did for the empty string.
        final char c = num.charAt(0);
        if (c >= '0' && c <= '9') {
            return true;
        }
        if (c == '+' || c == '-') {
            return num.length() > 1 && (STRING.isDigitChar(num.charAt(1)) || num.charAt(1) == '.');
        }
        return c == '.';
    }
    public static boolean isOnlyDigits(String num){
        return STRING.isOnlyDigits(num);
    }

    public static boolean isNegative(String num) {
        return isNumber(num) && num.charAt(0) == '-';
    }
    
   public static boolean isPositive(String num) {
        return isNumber(num) && num.charAt(0) != '-';
    }

    public Number getNumber() {
        return new Number(num);
    }

    public void validateNumber(MathExpression function) {
        List<String> scan = function.getScanner();
        //Numbers

        try {
            //specify valid tokens that can come before a number
            if (!Operator.isCircBracket(scan.get(index - 1))
                    && !Operator.isLogicOperator(scan.get(index - 1)) && !Operator.isUnaryPreOperator(scan.get(index - 1))
                    && !Operator.isBinaryOperator(scan.get(index - 1)) && !Operator.isAssignmentOperator(scan.get(index - 1))
                    && !Method.isMultiArgsMethod(scan.get(index - 1)) && !isNumber(scan.get(index - 1)) && !Variable.isVariableString(scan.get(index - 1))) {
                com.github.gbenroscience.util.Utils.logError(
                        "ParserNG Does Not Allow " + num + " To Combine The Function Members \"" + scan.get(index - 1) + "\" And \"" + scan.get(index) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Numbers!");
                function.setCorrectFunction(false);
                scan.clear();
            }//end if
            //specify valid tokens that can come after a number
            if (!Operator.isCircBracket(scan.get(index + 1)) && !Operator.isBinaryOperator(scan.get(index + 1))
                    && !Operator.isUnaryPostOperator(scan.get(index + 1))
                    && !Operator.isLogicOperator(scan.get(index + 1))
                    && !Operator.isUnaryPreOperator(scan.get(index + 1)) && !Method.isNumberReturningStatsMethod(scan.get(index + 1))
                    && !Method.isLogToAnyBase(scan.get(index + 1)) && !Method.isAntiLogToAnyBase(scan.get(index + 1)) && !isNumber(scan.get(index + 1)) && !Variable.isVariableString(scan.get(index + 1))) {
                com.github.gbenroscience.util.Utils.logError(
                        "ParserNG Does Not Allow " + num + " To Combine The Function Members \"" + scan.get(index) + "\" And \"" + scan.get(index + 1) + "\""
                        + " As You Have Done."
                        + "ParserNG Error Detector For Numbers!");
                function.setCorrectFunction(false);
                scan.clear();
            }//end if
        }//end try
        catch (IndexOutOfBoundsException ind) {

        }//end catch

    }
 
      public static double fastParseDouble(String s) {
        int len = s.length();
        if (len == 0) {
            return 0.0; // or throw if empty not allowed
        }
        int idx = 0;
        boolean negative = false;
        char first = s.charAt(0);
 
        // Handle sign
        if (first == '-') {
            negative = true;
            idx = 1;
        } else if (first == '+') {
            idx = 1;
        }

        double value = 0.0;

        // Integer part
        boolean hasIntPart = false;
        while (idx < len) {
            char c = s.charAt(idx);
            if (c >= '0' && c <= '9') {
                value = value * 10.0 + (c - '0');
                hasIntPart = true;
                idx++;
            } else {
                break;
            }
        }

        // Fractional part
        boolean hasFracPart = false;
        if (idx < len && s.charAt(idx) == '.') {
            idx++;
            double frac = 0.0;
            double div = 1.0;
            while (idx < len) {
                char c = s.charAt(idx);
                if (c >= '0' && c <= '9') {
                    div *= 10.0;
                    frac = frac * 10.0 + (c - '0');
                    hasFracPart = true;
                    idx++;
                } else {
                    break;
                }
            }
            value += frac / div;
        }

        // Scientific notation
        if (idx < len && (s.charAt(idx) == 'e' || s.charAt(idx) == 'E')) {
            idx++;
            boolean negExp = false;
            if (idx < len && s.charAt(idx) == '-') {
                negExp = true;
                idx++;
            } else if (idx < len && s.charAt(idx) == '+') {
                idx++;
            }

            long exp = 0;
            boolean hasExpDigits = false;
            while (idx < len) {
                char c = s.charAt(idx);
                if (c >= '0' && c <= '9') {
                    exp = exp * 10 + (c - '0');
                    hasExpDigits = true;
                    if (exp > 1000000000L) { // Prevent insane overflow
                        value = negative ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
                        break;
                    }
                    idx++;
                } else {
                    break;
                }
            }
            if (hasExpDigits) {
                value *= Math.pow(10.0, negExp ? -exp : exp);
            }
        }

        // Basic validation (since scanner is trusted, but good to have)
        if (idx != len || (!hasIntPart && !hasFracPart)) {
            return 0.0; // or throw if strict
        }

        return negative ? -value : value;
    }

 
    public static void main(String args[]) {
        System.out.println(isNumber("-"));
        System.out.println(validNumber("-90g"));
    }
}
