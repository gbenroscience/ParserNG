/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import parser.methods.Method;

import java.util.ArrayList;
import java.util.List;
import static parser.Variable.*;
import static parser.Number.*;
import static parser.Bracket.*;

/**
 * Objects of this class take a data-set and simplify/reduce its complexity so
 * that class MathExpression can easily work with it.
 *
 * The ultimate goal is to convert every data in the data-set into a number or a
 * simple variable.
 *
 * @since Sunday August 07 2011
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class DataSetFormatter {

    private List<String> dataset = new ArrayList<String>();
    /**
     * used to mask the commas after they have been processed.
     */
    public static final String COMMA_MASK = "?";
    /**
     * used to mask the commas after they have been processed.
     */
    public static final String OPEN_BRACKET_MASK = "<<<<";
    /**
     * used to mask the commas after they have been processed.
     */
    public static final String CLOSE_BRACKET_MASK = ">>>>";


    /**
     * Takes a comma separated string of data values and scans them into its
     * dataset attribute.
     *
     * @param datastring A comma separated data set. e.g 2,3,......or
     * sort(2,3,1,5,...) or mode(1,8,...)
     */
    public DataSetFormatter(String datastring) {
//scan(datastring);
        scanCommaSeparatedData(datastring);
    }

    public void setDataset(ArrayList<String> dataset) {
        this.dataset = dataset;
    }

    public List<String> getDataset() {
        return dataset;
    }

    /**
     *
     * @return a formatted data-set.
     */
    public String getFormattedDataSet() {
        StringBuilder formatted = new StringBuilder();
        for (String c : dataset) {
            formatted.append(c);
        }
        return formatted.toString();//change back
    }

    /**
     * @param sbpList A list containing the contents of a single bracket pair.
     * The starting and ending brackets are not included.//
     * 2,3,sin(x),cos,(,x1,),8
     */
    private void processCommasInSingleBracketPair(List<String> sbpList) {
        // System.out.println("START________________________________________________________________________START: "+sbpList);
        int indexOfComma = 0;
        int lastIndexCursor = 0;
        while ((indexOfComma = LISTS.nextIndexOf(sbpList, indexOfComma, ",")) != -1) {

            int spaceWidth = indexOfComma - lastIndexCursor;

            sbpList.set(indexOfComma, COMMA_MASK);//mask the comma.
            String token = sbpList.get(lastIndexCursor);
            /**
             * The logic of this if statement.
             * This list is a comma-separated list
             * We intend to place brackets about every group of tokens that occurs either between 2 commas, or between the beginning of the list and
             * the first comma, or between the last comma and the end of the list.
             * RULES:
             * If only 1 token occurs in the said space, check if it is a compound number-variable token..e.g 3A or 4B
             * If it is, bracket it, else ignore it; we cannot waste parentheses like that.
             *
             * If the group of tokens starts with the @ symbol, a function definition is there, DO NOT BRACKET IT.
             *
             * ELSE, BRACKET IT ALL.
             *
             */
            if ( !isAtOperator(token) && !Method.isListReturningStatsMethod(token) && !Method.isFunctionOperatingMethod(token)
                    && (spaceWidth > 2 || (!validNumber(token) && !isVariableString(token)) )    ) {//Do not bracket anonymous functions.
// System.out.printf("lastIndexOfCursor = %d, indexOfComma = %d\n ",lastIndexCursor,indexOfComma);
                //               System.out.println("HotSpot-Begin: "+sbpList+" targeting entries: "+sbpList.get(lastIndexCursor)+" AND "+sbpList.get(indexOfComma));
                sbpList.add(indexOfComma, CLOSE_BRACKET_MASK);//add an encoded close bracket here
                sbpList.add(lastIndexCursor, OPEN_BRACKET_MASK);//add an encoded open bracket here
                //               System.out.println("HotSpot-End: "+sbpList);
                lastIndexCursor = indexOfComma + 3;
            }
            else{
                lastIndexCursor = indexOfComma+1;
            }

            //sum(@(x)3*x-sin(x),4

        }//end while loop
        int sz = sbpList.size();
        int spaceWidth = sz - lastIndexCursor;
        /**
         * At the end of the loop, quantities in the comma separated list
         * end up not been bracketed, so check if the quantity is a group of
         * tokens. If so, bracket it. Else (a number or a variable, leave it unbracketed.
         *
         *
         */
        String token = sbpList.get(lastIndexCursor);
        if (  !isAtOperator(token) && !Method.isListReturningStatsMethod(token)  && !Method.isFunctionOperatingMethod(token)
                && (spaceWidth > 2 || (!validNumber(token) && !isVariableString(token)) )  ) {
            //     System.out.println("HotSpot-Begin-1: "+sbpList);
            sbpList.add(sz, CLOSE_BRACKET_MASK);//add an encoded close bracket here
            sbpList.add(lastIndexCursor, OPEN_BRACKET_MASK);//add an encoded open bracket here
            //         System.out.println("HotSpot-End-1: "+sbpList);
        }

        // System.out.println("END________________________________________________________________________END:  "+sbpList);

    }

    /**
     * root(@(x)sin(x)-x,2,3)
     *
     * @param myStr A String containing a data that uses commas to separate
     * values. It attacks the commas and encloses the values and other data
     * between commas with brackets. e.g
     * sort(2,sin(3),4,log,(,3*a+12,sum,(,2,3,),),log,(,3,10,)....) becomes
     * sort( (2),(sin(3)),(4)....)
     */
    public final void scanCommaSeparatedData(String myStr) {

        // data = data.replace("@(", "@?");
        //CustomScanner csc = new CustomScanner(data, true, getAllFunctions(), operators);
        CustomScanner csc = new CustomScanner(myStr, true, Method.getAllFunctions(), Operator.COMMA, Operator.OPEN_CIRC_BRAC, Operator.CLOSE_CIRC_BRAC);

        this.dataset = csc.scan();

        //  System.out.println("In the beginning: "+dataset);

        //root(5,@(x)sin(x)-x,2,4)
        //sum(4,3,5,6,7,1
        // (, and , ), and variable names

        for (int i = 0; i < dataset.size(); i++) {

            if (isCloseBracket(dataset.get(i))) {
                int close = i;
                int open = Bracket.getComplementIndex(false, close, dataset);
                dataset.set(open,OPEN_BRACKET_MASK);
                dataset.set(close,CLOSE_BRACKET_MASK);

                // List<String> subData = dataset.subList(open+1, close);
                processCommasInSingleBracketPair(dataset.subList(open+1, close));
                //  System.out.println("dataset is now: "+dataset);
                //  System.out.println("_______________________________________________________________________________________");
            }

        }

        for (int i = 0; i < dataset.size(); i++) {
            String token = dataset.get(i);

            switch (token) {
                case COMMA_MASK:
                    dataset.set(i, ",");
                    break;
                case OPEN_BRACKET_MASK:
                    dataset.set(i, "(");
                    break;
                case CLOSE_BRACKET_MASK:
                    dataset.set(i, ")");
                    break;
                default:
                    break;
            }


        }

        //System.out.println("At the end: "+dataset);

    }

    public static void main(String args[]) {
        String func = "sum(@(x)x^3+2*x+1,3,@(x)cos(x+4),5,-6,7,-8,8)";//"sort(sin(2),cosh(4),-6,log(12,10),-12,2*sin(3),cos(4),34)";
//String func = "(root(f,2,4))";
        DataSetFormatter f = new DataSetFormatter(func);
        System.out.println("Before scan-processing, the data = " + func);
        System.out.println("After scan-processing, the dataset = " + f.dataset);
        System.out.println("After scan-processing, the processed string  = " + f.getFormattedDataSet());
    }

}
