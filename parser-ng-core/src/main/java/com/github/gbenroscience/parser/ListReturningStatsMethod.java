/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import static com.github.gbenroscience.parser.Operator.*;
import com.github.gbenroscience.parser.methods.Method;
import com.github.gbenroscience.util.ErrorLog;

import java.util.ArrayList;
import java.util.List;

/**
 * Examples are sort function, mode function and other methods that will return
 * a list of numbers after acting on a number or a list of numbers
 *
 * @author GBEMIRO
 */
public final class ListReturningStatsMethod implements Validatable {

    /**
     * The method Name
     */
    private String name;

    /**
     * The index of this Operator in its parent scanned function.
     */
    private int index;

    /**
     * The opening bracket operator that forms one of the bracket pair used by
     * this Operator to bound its data to the left
     */
    private Bracket openBracket;
    /**
     * The closing bracket operator that forms one of the bracket pair used by
     * this Operator to bound its data to the right
     */
    private Bracket closeBracket;

    /**
     * The ListReturningStatsMethod object that immediately envelopes this one.
     */
    private ListReturningStatsMethod parent;

    private String errorMessage = "";

    private boolean rootElement;

    /**
     *
     * @param name the name of the method
     * @param index the index of this Operator in its parent scanned function.
     * @param scan the ArrayList object that contains the scanned function that
     * has this ListReturningStatsOperator object
     */
    public ListReturningStatsMethod(String name, int index, List<String> scan) {
        this.name = name;
        this.index = (scan.get(index).equals(name)) ? index : -1;
        this.openBracket = new Bracket("(");
        this.openBracket.setIndex(index + 1);
        int compIndex = Bracket.getComplementIndex(true, index + 1, scan);
        closeBracket = new Bracket(")");
        closeBracket.setIndex(compIndex);
        openBracket.setComplement(closeBracket);
        closeBracket.setComplement(openBracket);
        determineSuperParentStatus(scan);
        hasParent(scan);
        if (this.index == -1) {
            throw new ArrayIndexOutOfBoundsException("PARSER COULD NOT FIND\n \'" + name + "\' AT INDEX " + index
                    + ".\nFOUND \'" + scan.get(index) + "\' INSTEAD OF \'" + name + "\'");
        }

    }

    /**
     *
     * @return true if this object is the superParent i.e the container for all
     * data in the function.
     */
    private void determineSuperParentStatus(List<String> scan) {
        boolean isSuper = true;

        for (int i = index - 1; i > 0; i--) {
            String token = scan.get(i);
            if (isOpeningBracket(token) && Method.isStatsMethod(scan.get(i - 1))) {

            }
        }

        int scanStartIndex = getCloseBracket().getIndex();
        loopForwards:
        {
            for (int i = scanStartIndex + 1; i < scan.size(); i++) {
                if (!isClosingBracket(scan.get(i))) {
                    rootElement = false;
                    break loopForwards;
                }

            }//end for loop

        }//end loop

        loopBackwards:
        {
            for (int i = index - 1; i > 0; i--) {
                if (!isOpeningBracket(scan.get(i))) {
                    rootElement = false;
                    break loopBackwards;
                }

            }//end for loop

        }//end loop

    }

    /**
     *
     * @param openBracket sets the opening bracket for this Operator
     */
    public void setOpenBracket(Bracket openBracket) {
        this.openBracket = openBracket;
    }

    /**
     *
     * @return the opening bracket for this Operator
     */
    public Bracket getOpenBracket() {
        return openBracket;
    }

    /**
     *
     * @param closeBracket sets the closing bracket for this Operator
     */
    public void setCloseBracket(Bracket closeBracket) {
        this.closeBracket = closeBracket;
    }

    /**
     *
     * @return the closing bracket for this Operator
     */
    public Bracket getCloseBracket() {
        return closeBracket;
    }

    /**
     *
     * @param index sets the location of this Operator object in its parent
     * scanned function.
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     *
     * @return the index of its parent scanned function.
     */
    public int getIndex() {
        return index;
    }

    /**
     * The concept of a parent here is that the first ListReturningStatsOperator
     * object to this Operator's left is one that encloses it. i.e
     * sort(2,3,mode(4,1,1,3)) Here sort is a parent to mode.
     *
     * But in sort(2,3,sort(4,1,3,4),mode(4,5,1)),mode will have no parent here
     * by our definition. The outermost sort which would have been its parent
     * does not immediately enclose it.The outermost sort however is a parent to
     * the second sort(4,1,3,4). We use this special definition for parent
     * because we wish to ensure that each object of this class will validate
     * its own surroundings, not even its contents. It will not validate beyond
     * the next ListReturningStatsOperator to it.
     *
     * @param parent sets the ListReturningStatsMethod object that immediately
     * envelopes this one.
     */
    public void setParent(ListReturningStatsMethod parent) {
        this.parent = parent;
    }

    /**
     *
     * @return the ListReturningStatsMethod object that immediately envelopes
     * this one.
     */
    public ListReturningStatsMethod getParent() {
        return parent;
    }

    /**
     *
     * @param errorMessage sets the error message generated by objects of this
     * class.
     */
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    /**
     *
     * @return the error message generated by objects of this class.
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     *
     * @param scan the ArrayList containing the scanned function
     * @return true if the first ListReturningStatsMethod object to its left is
     * its parent. So it can have a parent and yet return false here.
     */
    public boolean hasParent(List<String> scan) {
        boolean isEnveloped = false;
        int i = index;

        boolean foundLikelyEncloser = false;
        //search for an enclosing ListReturningStatsMethod object to the left of this
        //Operator's location.If one is found set the boolean to true and exit the search loop.
        for (i = index - 1; i >= 0; i--) {
            if (Method.isListReturningStatsMethod(scan.get(i))) {
                foundLikelyEncloser = true;
                break;
            }//end if

        }//end for

        //Try to get the index of the enclosing bracket of the enclosing operator if one exists.
        if (foundLikelyEncloser) {
            int compIndex = Bracket.getComplementIndex(true, i + 1, scan);

            if (compIndex > closeBracket.getIndex()) {
                isEnveloped = true;
                ListReturningStatsMethod listType = new ListReturningStatsMethod(scan.get(i), i, scan);
                setParent(listType);
            }// end if

        }//end if

        return isEnveloped;
    }

    /**
     *
     * @param scan the ArrayList of the scanned function that contains this
     * object
     * @param errorLog
     * @return true if this object scans through its surroundings to the left
     * and to the right and sees a valid bracket structure around itself
     */
    @Override
    public boolean validate(List<String> scan, ErrorLog errorLog) {

        if (rootElement) {
            return true;
        }
        boolean valid = true;

        try {
            if (!Method.isMatrixReturningMethod(name)
                    && (isBinaryOperator(scan.get(index - 1)) || Method.isUnaryPreOperatorORDefinedMethod(scan.get(index - 1)))) {
                errorLog.error("Bad Syntax! Do Not Concatenate operator " + scan.get(index - 1) + " With " + name);
                ///e.g 2+sort(list) is bad! sin(sort(list)) is bad!
                    return false;
            }//end if
        }//end try
        catch (IndexOutOfBoundsException indexErr) {

        }//end catch
        try {
            if (!Method.isMatrixReturningMethod(name)
                    && (isBinaryOperator(scan.get(closeBracket.getIndex() + 1)) || isUnaryPostOperator(scan.get(closeBracket.getIndex() + 1)))) {
                errorLog.error("Bad Syntax! Do Not Append operator " + scan.get(index - 1) + " To " + name);//sort(list)+expr is bad...sort(list)! etc is bad!
                return false;
            }//end else if
        }//end try 
        catch (IndexOutOfBoundsException indexErr) {

        }//end catch
        //validate further
        if (valid) {
            loopBackwards:
            {
                boolean openBracsOnly = true;
                for (int i = index - 1; i > 0; i--) {

                    //this logic recognizes the end point of backwards bracket
                    //validation for a given object of this class. A close bracket
                    //indicates the end of data that can affect this object
                    if (isClosingBracket(scan.get(i)) || Method.isStatsMethod(scan.get(i))) {
                        errorLog.error("Ending Backwards Validation For " + name);
                        break loopBackwards;
                    }//end if

                    //this logic (the next two ifs)will disallow binary operations
                    //on encapsulations of objects of this class
                    // and their operands within brackets e.g sort(3,2,4)+((sort(1,3,1,-9,3))
                    if (!isOpeningBracket(scan.get(i))) {
                        openBracsOnly = false;
                        errorLog.error("MBracket Sequence Established. Trend Finished. Applying Other Validation Techniques.");
                    }
                    
                    String prevToken = scan.get(i - 1);
                    if (openBracsOnly && isBinaryOperator(prevToken) || (Method.isUnaryPreOperatorORDefinedMethod(prevToken) && !Method.isStatsMethod(prevToken))) {
                        // disallows this: +((sort(1,2,3,-9...)) AND this: sin(sort(1,23,-9...))
                        valid = false;
                        errorLog.error("Bad Syntax For Data Set Returning Operator " + name
                                + "REASON:::"
                                + "Cannot Perform Binary Operations On Data Set.");
                        break loopBackwards;
                    }

                }//end for
            }//end loopBackwards

             
                    String tkn=scan.get(closeBracket.getIndex()+1);
                        if ( (isBinaryOperator(tkn) || isUnaryPostOperator(tkn)) ) {
                            //stop sort(list)+expr
                            valid = false;
                        }
 
        }//end if

        return valid;
    }

    /**
     * Takes an object of class Function and validates its
     * ListReturningStatsOperators objects.
     *
     * @param scan the scanner output
     * @param errorLog
     * @return true if valid
     */
    public static boolean validateFunction(List<String> scan, ErrorLog errorLog) {

        ListReturningStatsMethod list;
        boolean validity = true;

        for (int i = 0; i < scan.size(); i++) {
            if (Method.isListReturningStatsMethod(scan.get(i))) {
                list = new ListReturningStatsMethod(scan.get(i), i, scan);
                validity = list.validate(scan, errorLog);
                i = list.getCloseBracket().getIndex();
                if (!validity) {
                    break;
                }//end if
            }//end if
        }//end for
        return validity;
    }//end method validateFunction(args)

    @Override
    public String toString() {
        String objectName = "ListReturningStatsOperator \"" + name + "\" located @ index " + index + " has its opening bracket @ " + openBracket.getIndex()
                + " and its closing bracket at " + closeBracket.getIndex()
                + ((rootElement) ? " and is the root parent of this data set " : " and is not the root parent of this data set ");

        return objectName;
    }


    /*
public static void main(String args[]){

  ArrayList<String>scan= new ArrayList<String>();

  scan = new MathScanner("sum(sort(2,3),sum(3,2))").scanner(null);

ListReturningStatsMethod list = new ListReturningStatsMethod("sum", 0, scan);

 util.Utils.logError(list.toString());

 util.Utils.logError(list.validate(scan));

}//end main

     */
}//end class
