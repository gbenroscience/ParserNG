/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import util.SimplePoint;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import util.VariableManager;

/**
 *
 * @author GBENRO
 */
public class Bracket extends Operator {

    /**
     * the name of the bracket i.e "(" or ")"
     */
    private transient String name = "";
    /**
     * The index of the bracket in the ArrayList containing the scanned function
     */
    private int index;
    /**
     * objects of this class keep a record of their counterpart or complementing
     * bracket.
     *
     */
    private transient Bracket complement;
    /**
     * Return true if the contents of the bracket have been evaluated
     */
    private boolean evaluated = false;

    /**
     * Constructor of this class for creating its objects and initializing their
     * names with either a ( or a ) and initial
     *
     * @param op
     */
    public Bracket(String op) {
        super(op);
        this.name = op;

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
     * Used to create similar objects that are not equal The object created by
     * this class is similar to the parameter because it contains the same data
     * as the parameter. However,its address in memory is different because it
     * refers to an entirely different object of the same class,but having
     * similar attributes.
     *
     * How can this method be of any use? Imagine an Array of Brackets say array
     * bracs filled with Bracket objects.
     *
     * If we create another Bracket array, say array moreBracs and copy the
     * objects in bracs into moreBracs.Now, both bracs and moreBracs will hold
     * references to these Bracket objects in memory.Java will not create new,
     * similar objects at another address in memory and store in the new array.
     * The command was most likely moreBracs=bracs; or in a loop, it would look
     * like:
     * 
     * for(int i=0;i&lt;bracs.length;i++){ moreBracs=bracs[i]; }
     *
     * These statements will only ensure that both arrays will hold a reference
     * to the same objects in memory,i.e RAM.
     * 
     * Hence whenever an unsuspecting coder modifies the contents of bracs,
     * thinking He/She has a backup in moreBracs,Java is effecting the
     * modification on the objects referred to by moreBracs, too.This can cause
     * a serious logical error in applications. To stop this, we use this method
     * in this way:
     *
     * for(int i=0;i&lt;bracs.length;i++){
     * moreBracs[i]=createTwinBracket(bracs[i]);
     * }
     *
     *
     *
     *
     * Note that this can be applied to all storage objects too e.g Collection
     * objects and so on.
     *
     * @param brac The object whose twin we wish to create.
     * @return a Bracket object that manifests exactly the same attributes as
     * brac but is a distinct object from brac.
     */
    public static Bracket createTwinBracket(Bracket brac) {
        Bracket newBrac = new Bracket(brac.getName());
        newBrac.setComplement(brac.getComplement());
        newBrac.setIndex(brac.getIndex());
        return newBrac;
    }

    /**
     * non-static version of the above method. This one creates a twin for this
     * Bracket object. The one above creates a twin for the specified bracket
     * object.
     *
     * @return a Bracket object that manifests exactly the same attributes as
     * brac but is a distinct object from brac.
     */
    public Bracket createTwinBracket() {
        Bracket newBrac = new Bracket(getName());
        newBrac.setComplement(getComplement());
        newBrac.setIndex(getIndex());
        return newBrac;
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
     * @return the name of this Bracket either ( or )
     */
    public String getName() {
        return name;
    }

    /**
     *
     * @param name sets the name of this bracket to either ( or )
     */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /**
     *
     * @return the Bracket object which is the complement of this Bracket object
     */
    public Bracket getComplement() {
        return complement;
    }

    /**
     *
     * @param complement sets the Bracket object which is to be the complement
     * to this one in the scanned Function
     */
    public void setComplement(Bracket complement) {
        this.complement = complement;
    }

    /**
     * checks if the Bracket object argument below is the sane as the complement
     * to this Bracket object.
     *
     * @param brac The Bracket object whose identity is to be checked whether or
     * not it complements this Bracket object.
     * @return true if the parameter is the complement to this one.
     */
    public boolean isComplement(Bracket brac) {
        return brac == getComplement();
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
     * @param isOpenBracket boolean variable that should be true if this bracket
     * object whose complement we seek is an opening bracket i.e (, and should
     * be set to false if this bracket object whose complement we seek is a
     * closing bracket i.e )
     * @param start the index of the given bracket.
     * @param scan the ArrayList containing the scanned function.
     * @return the index of the enclosing or complement bracket of this bracket
     * object
     */
    public static int getComplementIndex(boolean isOpenBracket, int start, ArrayList<String> scan) {

        int open = 0;
        int close = 0;
        int stop = 0;
        if (isOpenBracket) {
            try {
                for (int i = start; i < scan.size(); i++) {

                    if (scan.get(i).equals("(")) {
                        open++;
                    } else if (scan.get(i).equals(")")) {
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
                        if (scan.get(i).equals("(")) {
                            open++;
                        } else if (scan.get(i).equals(")")) {
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
     * @param isOpenBracket boolean variable that should be true if this bracket
     * object whose complement we seek is an opening bracket i.e (, and should
     * be set to false if this bracket object whose complement we seek is a
     * closing bracket i.e )
     * @param start the index of the given bracket.
     * @param scan the ArrayList containing the scanned function.
     * @return the index of the enclosing or complement bracket of this bracket
     * object
     */
    public static int getComplementIndex(boolean isOpenBracket, int start, List<String> scan) {

        int open = 0;
        int close = 0;
        int stop = 0;
        if (isOpenBracket) {
            try {
                for (int i = start; i < scan.size(); i++) {

                    if (scan.get(i).equals("(")) {
                        open++;
                    } else if (scan.get(i).equals(")")) {
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
                        if (scan.get(i).equals("(")) {
                            open++;
                        } else if (scan.get(i).equals(")")) {
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
     * @param isOpenBracket boolean variable that should be true if this bracket
     * object whose complement we seek is an opening bracket i.e (, and should
     * be set to false if this bracket object whose complement we seek is a
     * closing bracket i.e )
     * @param start the index of the given bracket.
     * @param expr the function string containing the brackets.
     * @return the index of the enclosing or complement bracket of this bracket
     * object
     */
    public static int getComplementIndex(boolean isOpenBracket, int start, String expr) {

        int open = 0;
        int close = 0;
        int stop = 0;
        int size = expr.length();
        if (isOpenBracket) {
            try {
                for (int i = start; i < size; i++) {
                    if (expr.substring(i, i + 1).equals("(")) {
                        open++;
                    } else if (expr.substring(i, i + 1).equals(")")) {
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
                        if (expr.substring(i, i + 1).equals("(")) {
                            open++;
                        } else if (expr.substring(i, i + 1).equals(")")) {
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
     * @return true if the scanned expression contains no brackets in the given
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
        return bracket.equals("(");
    }

    /**
     *
     * @param bracket The String object.
     * @return true if the String object represents a close bracket
     */
    public static boolean isCloseBracket(String bracket) {
        return bracket.equals(")");
    }

    /**
     *
     * @param scan The ArrayList containing the scanned function inside which
     * this Bracket exists.
     * @return true if between this Bracket and its complement, a Variable
     * object is found.
     */
    public boolean simpleBracketPairHasVariables(ArrayList<String> scan) {

        if (isOpenBracket(name)) {
            int i = this.index;
            int j = this.complement.index;

            for (; i <= j; i++) {
                String var = scan.get(i);
                if (Variable.isVariableString(var)) {
                    try {
                        Variable v = VariableManager.lookUp(var);
                        return true;
                    }//end try
                    catch (NullPointerException exception) {
                    }//end catch
                }//end if
            }//end for
        }//end if
        else {
            int i = this.index;
            int j = this.complement.index;
            for (; j <= i; j++) {
                String var = scan.get(j);
                if (Variable.isVariableString(var)) {
                    try {
                        Variable v = VariableManager.lookUp(var);
                        return true;
                    }//end try
                    catch (NullPointerException exception) {
                    }//end catch
                }//end if

            }//end for
        }//end else

        return false;
    }//end method

    /**
     *
     * @param scan The ArrayList object containing the scanned function.
     * @return The contents of this bracket and its complement as a string, the
     * bracket and its complement are also returned. e.g in 5+(2+3-sin2).. This
     * method will return (2+3-sin2).
     */
    public String getDomainContents(ArrayList<String> scan) {

        String contents = "";
        if (isOpenBracket(name)) {
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
    public List<String> getBracketDomainContents(ArrayList<String> scan) {
        if (isOpeningBracket(this.getName())) {
            return scan.subList(this.getIndex(), this.getComplement().getIndex() + 1);
        } else if (isClosingBracket(this.getName())) {
            return scan.subList(this.getComplement().getIndex(), this.getIndex() + 1);
        }
        return null;
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
        boolean correctFunction = false;


        ArrayList<String> scan = new ArrayList<String>();
        ArrayList<SimplePoint> map = new ArrayList<SimplePoint>();
        ArrayList<Bracket> bracs = new ArrayList<Bracket>();



        scan.addAll(scanner);
        int open = 0;//tracks the index of an opening bracket
        int close = scan.indexOf(")");//tracks the index of a closing bracket
        int i = 0;
        while (close != -1) {
            try {
                open = LISTS.prevIndexOf(scan, close, "(");
                map.add(new SimplePoint(open, close));
                Bracket openBrac = new Bracket("(");
                Bracket closeBrac = new Bracket(")");
                openBrac.setIndex(open);
                closeBrac.setIndex(close);
                openBrac.setComplement(closeBrac);
                closeBrac.setComplement(openBrac);


                bracs.add(openBrac);
                bracs.add(closeBrac);

                scan.set(open, "");
                scan.set(close, "");

                close = scan.indexOf(")");
                ++i;
            }//end try
            catch (IndexOutOfBoundsException ind) {
                break;
            }
        }//end while

//after the mapping, the algorithm demands that all ( and ) should have been used up in the function
        if (scan.indexOf("(") == -1 && scan.indexOf(")") == -1) {
            correctFunction = true;
        } else {
            correctFunction = false;
        }



        return correctFunction;
    }

    /**
     *@param scanner  The ArrayList containing the scanner output for a Function
     * Multiplies the contents of this List by -1.
     */
    public void multiplyContentsByMinusOne(List<String> scanner){
        List<String>domain = getBracketDomainContents((ArrayList<String>) scanner);


        for(int i=0;i<domain.size();i++){
            if(domain.get(i).equals("+")){
                domain.set(i,"-");
            }//end if
            else if(domain.get(i).equals("-")) {
                domain.set(i,"+");
            }//end if

        }//end for loop

        if(Number.isNumber(domain.get(1))) {
            domain.set(1,""+(-1*Double.parseDouble(domain.get(1))) );
        }
        else if(Variable.isVariableString(domain.get(1))){
            domain.add(1,"*");
            domain.add(1,"-1");
            complement.setIndex(complement.index+2);
        }

    }//end method

    /**
     * @param scanner  The ArrayList containing the scanner output for a Function
     * @param index The index at which the token is to be retrieved.
     * The first and elements are compulsorily always an open bracket and a close bracket
     * respectively.
     */
    public String domainTokenAt(List<String> scanner,int index){
        List<String>domain = getBracketDomainContents((ArrayList<String>) scanner);
        return domain.get(index);
    }
}//end class Bracket