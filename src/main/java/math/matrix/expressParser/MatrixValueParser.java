/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.matrix.expressParser;

import parser.STRING;

import util.Dimension;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import util.MatrixFormatException;
import static parser.Number.*;
import static parser.Operator.*;

/**
 *
 * Objects of this class extract a matrix from an input expression of the
 * format:
 *
 * [num1,num2,...:num_i,num_i+1,....:num_j,num_j+1...: ] e.g [2,3,4:4,1,2:...]
 * The colons represent the end of one row of values and the beginning of
 * another.
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class MatrixValueParser {

    private boolean valid = true;
    ;
    private String values = "";
    private ArrayList<String> scan = new ArrayList<String>();
    /**
     * The size of the matrix
     */
    private Dimension size = new Dimension();

    public MatrixValueParser() {

    }

    /**
     *
     * @param values A matrix value string e.g [2,4,5:3,9.939,45.2:1,4,2:]
     */
    public MatrixValueParser(String values) {
        this.values = STRING.purifier(values);

        try {
            scanner();
        }//end try
        catch (MatrixFormatException matExcep) {

        }//end catch

    }

    /**
     *
     * Scans a matrix value string into tokens, e.g [2,4,5:3,9.939,45.2:1,4,2]
     * and produces an output containing the numbers and the colons in the
     * ArrayList attribute scan. Method getMatrix then extracts the matrix from
     * scan.
     *
     * @throws MatrixFormatException if the format of the input does not conform
     * to that of a matrix.
     */
    public void scanner() throws MatrixFormatException {
        try {
            if (values.substring(0, 1).equals("[") && values.lastIndexOf("]") == values.length() - 1) {
                //validate number of [ and ]...the expression must have only one of these each.
                values = values.substring(1, values.length() - 1);//extract the instructions from between the enclosing [ and ].
                //Now to validate the number of [ and ], check that after the extracted string does not contain any [ and ].

                if (values.contains("[") || values.contains("]") || !values.substring(values.length() - 1, values.length()).equals(":")) {
                    valid = false;
                    throw new MatrixFormatException("Invalid Matrix Input: " + values);
                }//end if
                else {
                    MmathScanner mathScanner = new MmathScanner(values);
                    scan = mathScanner.scanner();
//recognize negatives and positives.
                    for (int i = 0; i < scan.size(); i++) {
                        try {
                            if (isPlusOrMinus(scan.get(i)) && validNumber(scan.get(i + 1))) {
                                scan.set(i, scan.get(i) + scan.get(i + 1));
                                scan.set(i + 1, "");
                            }
                        } catch (IndexOutOfBoundsException indexErr) {

                        }
                    }
                    ArrayList<String> purify = new ArrayList<String>();
                    purify.add("");
                    scan.removeAll(purify);
                    int colonCount = 0;
//count the number of : in the input.
                    for (int i = 0; i < scan.size(); i++) {
                        if (isColon(scan.get(i))) {
                            colonCount++;
                        }
                    }//end for
                    size.width = colonCount;//set the number of rows. to be the colon count.
                    List<String> val = new ArrayList<String>(scan);
                    int lastColSize = 0;
                    int colSize = val.subList(0, val.indexOf(":")).size();

                    while (val.size() > 0) {
                        try {
                            lastColSize = colSize;
                            val.subList(0, val.indexOf(":") + 1).clear();
                            colSize = val.subList(0, val.indexOf(":")).size();

                            if (lastColSize != colSize) {
                                valid = false;
                                break;
                            }

                        } catch (IllegalArgumentException ill) {
                            break;
                        } catch (IndexOutOfBoundsException ind) {
                            break;
                        }

                    }//end while
                    if (valid) {

                        size.height = scan.indexOf(":");// set the number of columns to the index of the
//first colon as it gives the number of items per equal row...and we have already verified that the rows are equal.
                        for (int i = 0; i < scan.size(); i++) {
                            if (!isColon(scan.get(i))) {
                                valid = validNumber(scan.get(i));
                            }
                            if (!valid) {
                                break;
                            }

                        }//end for
                    }//end if
                }//end else

            }//end if
        }//end try
        catch (IndexOutOfBoundsException boundsException) {

        }
    }//end method

    public void setValues(String values) {
        this.values = values;
        try {
            scanner();
        } catch (MatrixFormatException ex) {

        }
    }

    public String getValues() {
        return values;
    }

    public void setValid(boolean valid) {
        this.valid = valid;
        if (!valid) {
            scan.clear();

        }
    }

    public boolean isValid() {
        return valid;
    }

    public Dimension getSize() {
        return size;
    }

    public void setScan(ArrayList<String> scan) {
        this.scan = scan;
    }

    public ArrayList<String> getScan() {
        return scan;
    }

    /**
     *
     * @return a Matrix object from the scanned values.
     */
    public Matrix getMatrix() throws NullPointerException {

        ArrayList<String> purify = new ArrayList<String>();
        purify.add(":");
        scan.removeAll(purify);
        double[][] arr = new double[size.width][size.height];
        if (isValid()) {

            int count = 0;
            for (int rows = 0; rows < size.width; rows++) {
                for (int cols = 0; cols < size.height; cols++) {
                    try {
                        arr[rows][cols] = Double.valueOf(scan.get(count));
                        count++;
                    }//end try
                    catch (NumberFormatException numException) {

                    } catch (IndexOutOfBoundsException index) {

                    }

                }//end inner for
            }//end outer for

            Matrix output = new Matrix(arr);
            return output;
        } else {
            return null;
        }
    }//end method

    /**
     *
     * @return a PrecisionMatrix object from the scanned values.
     */
    public PrecisionMatrix getPrecisionMatrix() throws NullPointerException {

        ArrayList<String> purify = new ArrayList<String>();
        purify.add(":");
        scan.removeAll(purify);
        BigDecimal[][] arr = new BigDecimal[size.width][size.height];
        if (isValid()) {

            int count = 0;
            for (int rows = 0; rows < size.width; rows++) {
                for (int cols = 0; cols < size.height; cols++) {
                    try {
                        arr[rows][cols] = new BigDecimal(scan.get(count));
                        count++;
                    }//end try
                    catch (NumberFormatException numException) {
                    }//end catch
                    catch (IndexOutOfBoundsException index) {
                    }//end catch

                }//end inner for
            }//end outer for

            PrecisionMatrix output = new PrecisionMatrix(arr);
            return output;
        } else {
            return null;
        }
    }//end method

}//end class MatrixValueParser
