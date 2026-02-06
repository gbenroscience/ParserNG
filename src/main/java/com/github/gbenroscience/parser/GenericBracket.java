/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.List;


/**
 *
 * @author JIBOYE, OLUWAGBEMIRO OLAOLUWA
 *
 */
public class GenericBracket {

    private int bracketType = CIRCULAR_BRACE;
    /**
     * For {} brackets
     */
    public static final int CURLY_BRACE = 1;//{}
    /**
     * For [] brackets
     */
    public static final int SQUARE_BRACE = 2;
    /**
     * For () brackets
     */
    public static final int CIRCULAR_BRACE = 3;//()
    /**
     * For &lt; and &gt; brackets
     */
    public static final int ANGULAR_BRACE = 4;//<>

    /**
     * If the particular bracket is an open bracket. If false, then it is a
     * close bracket.
     */
    private boolean open;

    /**
     * The index of the bracket in the ArrayList containing the scanned function
     */
    private int index;
    /**
     * objects of this class keep a record of their counterpart or complementing
     * bracket.
     *
     */
    private GenericBracket complement;
    /**
     * Return true if the contents of the bracket have been evaluated
     */
    private boolean evaluated = false;

    /**
     * Constructor of this class for creating its objects and initializing their
     * names with either a ( or a ) and initial
     *
     * @param bracketType One of:
     * <ol>
     * <li>{@link GenericBracket#CURLY_BRACE}</li>
     * <li>{@link GenericBracket#SQUARE_BRACE}</li>
     * <li>{@link GenericBracket#CIRCULAR_BRACE}</li>
     * <li>{@link GenericBracket#ANGULAR_BRACE}</li>
     * </ol>
     *
     *
     * @param open If true, this Bracket is an opening brace.
     */
    public GenericBracket(int bracketType, boolean open) {
        this.bracketType = bracketType;
        this.open = open;
    }
    /**
     *
     * @param name The bracket string.
     * One of {,},(,),[,],&lt;,&gt;
     */
    public GenericBracket(String name) {
        switch(name){
            case "{":
                this.bracketType = CURLY_BRACE;
                this.open = true;
                break;
            case "[":
                this.bracketType = SQUARE_BRACE;
                this.open = true;
                break;
            case "(":
                this.bracketType = CIRCULAR_BRACE;
                this.open = true;
                break;
            case "<":
                this.bracketType = ANGULAR_BRACE;
                this.open = true;
                break;

            case "}":
                this.bracketType = CURLY_BRACE;
                this.open = false;
                break;
            case "]":
                this.bracketType = SQUARE_BRACE;
                this.open = false;
                break;
            case ")":
                this.bracketType = CIRCULAR_BRACE;
                this.open = false;
                break;
            case ">":
                this.bracketType = ANGULAR_BRACE;
                this.open = false;
                break;

            default:
                throw new InputMismatchException("Bad bracket");

        }

    }
    /**
     *
     * @param evaluated set whether or not this bracket's contents have been
     * evaluated
     */
    public void setEvaluated(boolean evaluated) {
        this.evaluated = evaluated;
    }

    /**
     *
     * @return true if this bracket's contents have been evaluated
     */
    public boolean isEvaluated() {
        return evaluated;
    }

    /**
     *
     * @return the index of this Bracket object in a scanned function
     */
    public int getIndex() {
        return index;
    }

    /**
     *
     * @param index the ne w index to assign to this Bracket object in a scanned
     * Function
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     *
     * @return the string representation of this Bracket half.
     */
    public String getBracketString() {
        switch (bracketType) {
            case CURLY_BRACE:
                return open ? "{" : "}";
            case SQUARE_BRACE:
                return open ? "[" : "]";
            case CIRCULAR_BRACE:
                return open ? "(" : ")";
            case ANGULAR_BRACE:
                return open ? "<" : ">";

            default:
                return null;
        }
    }

    /**
     *
     * @return the string representation of this Bracket half's complement.
     */
    public String getComplementBracketString() {
        switch (bracketType) {
            case CURLY_BRACE:
                return open ? "}" : "{";
            case SQUARE_BRACE:
                return open ? "]" : "[";
            case CIRCULAR_BRACE:
                return open ? ")" : "(";
            case ANGULAR_BRACE:
                return open ? ">" : "<";
            default:
                return null;
        }
    }

    /**
     *
     * @return the Bracket object which is the complement of this Bracket object
     */
    public GenericBracket getComplement() {
        return complement;
    }

    /**
     *
     * @param complement sets the Bracket object which is to be the complement
     * to this one in the scanned Function
     */
    public void setComplement(GenericBracket complement) {
        this.complement = complement;
    }

    /**
     * checks if the Bracket object argument below is the sane as the complement
     * to this Bracket object.
     *
     * @param brac The GenericBracket object whose identity is to be checked whether or
     * not it complements this GenericBracket object.
     * @return true if the parameter is the complement to this one.
     */
    public boolean isComplement(GenericBracket brac) {
        return brac == getComplement();
    }

    public void setBracketType(int bracketType) {
        this.bracketType = bracketType;
    }

    public int getBracketType() {
        return bracketType;
    }

    public boolean isOpen() {
        return open;
    }

    public void setOpen(boolean open) {
        this.open = open;
    }

    /**
     *
     * @param brac the bracket to be checked if or not it is enclosed by this
     * bracket and its complement.
     * @return true if the bracket is enclosed by this bracket and its
     * counterpart.
     */
    public boolean encloses(Bracket brac) {
        boolean truth = false;

        if (this.getIndex() < brac.getIndex() && this.getComplement().getIndex() > brac.getIndex()) {
            truth = true;
        } else if (this.getIndex() > brac.getIndex() && this.getComplement().getIndex() < brac.getIndex()) {
            truth = true;
        }

        return truth;
    }

    /**
     *
     * @param brac an ArrayList object containing all brackets found in a
     * function
     * @return the number of bracket pairs contained between this Bracket object
     * and its complement
     */
    public int getNumberOfInternalBrackets(ArrayList<Bracket> brac) {
        int num = 0;
        int i = 0;
        while (i < brac.size()) {
            if (encloses(brac.get(i))) {
                num++;
            }

            i++;
        }
        return (num / 2);
    }

    /**
     * @param scan The ArrayList object containing the scanned function.
     * @return true if this Bracket object forms with its complement, a single
     * bracket pair that is a bracket pair containing no other bracket pairs.
     */
    public boolean isSBP(ArrayList<String> scan) {
        int i = this.index;
        int j = this.complement.index;

        if (i < j) {
            ++i;//step away from current bracket and start searching for other brackets.
            for (; i < j; i++) {
                if (isBracket(scan.get(i))) {
                    return false;
                }
            }//end for
        } else if (i > j) {
            ++j;//step away from current bracket and start searching for other brackets.
            for (; j < i; j++) {
                if (isBracket(scan.get(j))) {
                    return false;
                }
            }//end for
        } else if (i == j) {
            throw new InputMismatchException("Open MBracket Cannot Be On The Same Index As Closing MBracket");
        }
        return true;
    }//end method

    /**
     * @param brac The {@link GenericBracket} object whose complement we seek is a closing
     * bracket i.e )
     * @param start the index of the given bracket.
     * @param scan the ArrayList containing the scanned function.
     * @return the index of the enclosing or complement bracket of this bracket
     * object
     */
    public static int getComplementIndex(GenericBracket brac, int start, ArrayList<String> scan) {

        int open = 0;
        int close = 0;
        int stop = 0;
        if (brac.isOpen()) {
            try {
                for (int i = start; i < scan.size(); i++) {

                    if (scan.get(i).equals(brac.getBracketString())) {
                        open++;
                    } else if (scan.get(i).equals(brac.getComplementBracketString())) {
                        close++;
                    }
                    if (open == close) {
                        stop = i;
                        break;
                    }

                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }
        }//end if
        else {
            try {
                for (int i = start; i >= 0; i--) {
                    try {
                        if (scan.get(i).equals(brac.getBracketString())) {
                            open++;
                        } else if (scan.get(i).equals(brac.getComplementBracketString())) {
                            close++;
                        }
                        if (open == close) {
                            stop = i;
                            break;
                        }
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                    }
                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }

        }
        return stop;
    }//end method

    /**
     * @param brac The {@link GenericBracket} object whose complement we seek.
     * @param start the index of the given bracket.
     * @param scan the ArrayList containing the scanned function.
     * @return the index of the enclosing or complement bracket of this bracket
     * object
     */
    public static int getComplementIndex(GenericBracket brac, int start, List<String> scan) {
        int open = 0;
        int close = 0;
        int stop = 0;
        if (brac.isOpen()) {
            try {
                for (int i = start; i < scan.size(); i++) {

                    if (scan.get(i).equals(brac.getBracketString())) {
                        open++;
                    } else if (scan.get(i).equals(brac.getComplementBracketString())) {
                        close++;
                    }
                    if (open == close) {
                        stop = i;
                        break;
                    }

                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }
        }//end if
        else {
            try {
                for (int i = start; i >= 0; i--) {
                    try {
                        if (scan.get(i).equals(brac.getBracketString())) {
                            open++;
                        } else if (scan.get(i).equals(brac.getComplementBracketString())) {
                            close++;
                        }
                        if (open == close) {
                            stop = i;
                            break;
                        }
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                    }
                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }

        }
        return stop;
    }


    /**
     * @param bracket The {@link GenericBracket} object whose complement we seek.
     * @param isOpenBracket boolean variable that should be true if this bracket
     * object whose complement we seek is an opening bracket i.e (, and should
     * be set to false if this bracket object whose complement we seek is a
     * closing bracket i.e )
     * @param start the index of the given bracket.
     * @param expr the function string containing the brackets.
     * @return the index of the enclosing or complement bracket of this bracket
     * object
     */
    public static int getComplementIndex(GenericBracket bracket,boolean isOpenBracket, int start, String expr) {

        int open = 0;
        int close = 0;
        int stop = 0;
        int size = expr.length();
        if (bracket.isOpen()) {
            try {
                for (int i = start; i < size; i++) {
                    if (expr.substring(i, i + 1).equals(bracket.getBracketString())) {
                        open++;
                    } else if (expr.substring(i, i + 1).equals(bracket.getComplementBracketString())) {
                        close++;
                    }
                    if (open == close) {
                        stop = i;
                        break;
                    }

                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }
        }//end if
        else if (!isOpenBracket) {
            try {
                for (int i = start; i >= 0; i--) {
                    try {
                        if (expr.substring(i, i + 1).equals(bracket.getBracketString())) {
                            open++;
                        } else if (expr.substring(i, i + 1).equals(bracket.getComplementBracketString())) {
                            close++;
                        }
                        if (open == close) {
                            stop = i;
                            break;
                        }
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                    }
                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }

        }
        return stop;
    }

    /**
     *
     * @param list The list containing the scanned math expression.
     * @param start The point in the list where this algorithm should start
     * checking the bracket syntax.(inclusive)
     * @param end The point in the list where this algorithm should stop
     * checking the bracket syntax.(inclusive)
     * @return true if the bracket syntax of the scanned expression in the given
     * range is valid or the expression in the given range is devoid of
     * brackets.
     */
    public static boolean checkBracketStructure(List<String> list, int start, int end) {
        return validateBracketStructure(list.subList(start, end + 1));
    }//end method

    /**
     *
     * @param list The list containing the scanned math expression.
     * @param start The point in the list where this algorithm should start
     * checking for brackets.(inclusive)
     * @param end The point in the list where this algorithm should stop
     * checking for brackets.(inclusive)
     * @return true if the scanned expression contains brackets in the given
     * range.
     */
    public static boolean hasBracketsInRange(List<String> list, int start, int end) {
        int sz = list.size();
        if (start >= 0 && end < sz) {
            for (int i = start; i <= end; i++) {
                if (isBracket(list.get(i))) {
                    return true;
                }
            }
        }//end if

        return false;
    }//end method

    /**
     *
     * @param bracket The String object.
     * @return true if the String object represents an open bracket
     */
    public static boolean isOpenBracket(String bracket) {
        return bracket.equals("{") || bracket.equals("[") || bracket.equals("(") || bracket.equals("<");
    }

    /**
     *
     * @param bracket The String object.
     * @return true if the String object represents a close bracket
     */
    public static boolean isCloseBracket(String bracket) {
        return bracket.equals("}") || bracket.equals("]") || bracket.equals(")") || bracket.equals(">");
    }

    /**
     *
     * @param scan The ArrayList object containing the scanned function.
     * @return The contents of this bracket and its complement as a string, the
     * bracket and its complement are also returned. e.g in 5+(2+3-sin2).. This
     * method will return (2+3-sin2).
     */
    public String getDomainContents(ArrayList<String> scan) {

        String contents = "";
        if (open) {
            int i = this.index;
            int j = this.complement.index;

            for (; i <= j; i++) {
                contents += scan.get(i);
            }//end for
        }//end if
        else {
            int i = this.index;
            int j = this.complement.index;
            for (; j <= i; j++) {
                contents = contents.concat(scan.get(j));
            }//end for
        }//end else
        return contents;
    }//end method

    /**
     * returns a List containing the contents of a bracket pair,including the
     * bracket pair itself.
     *
     * @param scan the ArrayList containing the scanner output for a Function
     * @return the bracket pair and its contents.
     */
    public ArrayList<String> getBracketDomainContents(ArrayList<String> scan) {
        if (open) {
            return (ArrayList<String>) scan.subList(this.getIndex(), this.getComplement().getIndex() + 1);
        } else {
            return (ArrayList<String>) scan.subList(this.getComplement().getIndex(), this.getIndex() + 1);
        }

    }

    /**
     * method mapBrackets goes over an input equation and maps all positions
     * that have corresponding brackets
     *
     * @param scanner The ArrayList object that contains the scanned math
     * function.
     * @return true if the structure of the bracket is valid.
     */
    private static boolean validateBracketStructure(List<String> scanner) {

        ArrayList<String> scan = new ArrayList<String>();

        scan.addAll(scanner);
        int open = 0;//tracks the index of an opening bracket
        int close = -1;//tracks the index of a closing bracket
        int i = 0;
        while ( (close = scan.indexOf("}")) != -1 || (close = scan.indexOf("]")) != -1  || (close = scan.indexOf(")")) != -1
                || (close = scan.indexOf(">")) != -1 ) {
            try {
                String val = scan.get(close);

                GenericBracket b = new GenericBracket(val);
                GenericBracket bComp = new GenericBracket(b.getComplementBracketString());


                GenericBracket openBrac = b.isOpen() ? b : bComp;
                GenericBracket closeBrac = !b.isOpen() ? b : bComp;

                open = prevIndexOf(scan, close, openBrac.getBracketString());
                openBrac.setIndex(open);
                closeBrac.setIndex(close);
                openBrac.setComplement(closeBrac);
                closeBrac.setComplement(openBrac);

                scan.set(open, "");
                scan.set(close, "");

                ++i;
            }//end try//end try
            catch (IndexOutOfBoundsException ind) {
                break;
            }
        }//end while

//after the mapping, the algorithm demands that all ( and ) should have been used up in the function
        if (scan.indexOf("{") == -1 && scan.indexOf("}") == -1 &&
                scan.indexOf("[") == -1 && scan.indexOf("]") == -1 &&
                scan.indexOf("(") == -1 && scan.indexOf(")") == -1 &&
                scan.indexOf("<") == -1 && scan.indexOf(">") == -1  ) {
            return true;
        } else {
            return false;
        }

    }

    /**
     *
     * @param list the collection of objects that the search is to be carried
     * out on
     * @param start the starting index of the search from where we search
     * backwards for the object that index itself been not included
     * @param sought the object that we seek
     * @return the index of the first occurrence of the object behind index
     * start or -1 if the object is not found.
     */
    public static int prevIndexOf(List list, int start, Object sought) {
        int index = -1;
        if (sought.getClass() == list.get(0).getClass()) {
            if (start >= 0) {
                for (int i = start - 1; i >= 0; i--) {
                    if (list.get(i).equals(sought)) {
                        index = i;
                        break;
                    }//end if
                }//end for

            }//end if
            else {
                throw new IndexOutOfBoundsException("Attempt to access index less than 0");
            }
        }//end if
        else {
            throw new ClassCastException("The object types must be the same for the list"
                    + " and the object been searched for in the list");
        }

        return index;
    }//end method prevIndexOf

    private static boolean isBracket(String get) {
        return get.equals("{") || get.equals("}") || get.equals("[") || get.equals("]") || get.equals("(") || get.equals(")")
                || get.equals("<") || get.equals(">");
    }



    public static void main(String[] args) {
        String str = "(3+(2-90/4)*{5*{8+2(9)-29<2+4>}})";
        System.err.println("valid: "+validateBracketStructure(Arrays.asList(str.split(""))));
    }
}//end class Bracket
