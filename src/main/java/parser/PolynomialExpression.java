/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;

import static parser.Number.*;

/**
 *
 * @JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class PolynomialExpression extends MathExpression {

    /**
     * Solves the Polynomial with normal double precision....about 16d.p.
     */
    public static final int DOUBLE_PRECISION = 1;
    /**
     * Solves the Polynomial with BigDecimal precision....about 16d.p.
     */
    public static final int BIGDECIMAL_PRECISION = 2;

    /**
     * Used to select what mode to operate objects of this class...whether
     * DOUBLE_PRECISION or BIGDECIMAL_PRECISION.
     */
    private int precision = DOUBLE_PRECISION;

    /**
     *
     * @param expression A valid polynomial expression, having powers of the
     * variable as only non-negative integers.
     */
    public PolynomialExpression(String expression, int precision) {
        super(expression);
        setPrecision(precision);
        if (isHasLogicOperators() || isHasListReturningOperators() || isHasPreNumberOperators() || isHasNumberReturningStatsOperators()
                || isHasPermOrCombOperators() || isHasRemainderOperators() || isHasPostNumberOperators()) {
            setCorrectFunction(false);
            util.Utils.logError("Only Polynomial Expressions Treated Here!");
        }//end if
    }//end constructor

    /**
     *
     * @param precision The precision to use. If set to any value other than 1
     * (DOUBLE_PRECISION) or 2 (BIGDECIMAL_PRECISION), it defaults to
     * DOUBLE_PRECISION
     */
    public void setPrecision(int precision) {
        this.precision = precision == 1 ? DOUBLE_PRECISION : (precision == 2 ? BIGDECIMAL_PRECISION : DOUBLE_PRECISION);
    }

    public int getPrecision() {
        return precision;
    }

    @Override
    public List<String> solve(List<String> list) {
        if (precision == DOUBLE_PRECISION) {
            return doublePrecisionSolve(list);
        }//end if
        else if (precision == BIGDECIMAL_PRECISION) {
            return bigDecimalPrecisionSolve(list);
        }//end else if
        else {
            throw new InputMismatchException(" Select A Valid Precision");
        }//end else.
    }//end method

    /**
     * Used by the main parser solve to figure out SBP portions of a
     * multi-bracketed expression (MBP)
     *
     * @param list a list of scanner tokens of a maths expression
     * @return the solution to a SBP maths expression. The precision returned
     * here is that of double numbers, namely about 16d.p
     */
    public List<String> doublePrecisionSolve(List<String> list) {

//correct the anomaly: [ (,-,number....,)  ]
        //   turn it into: [ (,,-number........,)     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("(") && list.get(1).equals(Operator.MINUS) && isNumber(list.get(2))) {
            list.set(1, "");

            //if the number is negative,make it positive
            if (list.get(2).substring(0, 1).equals(Operator.MINUS)) {
                list.set(2, list.get(2).substring(1));
            } //if the number is positive,make it negative
            else {
                list.set(2, Operator.MINUS + list.get(2));
            }
        }
//Create a collection to serve as a garbage collector for the empty memory
//locations and other unwanted locations created in the processing collection
        ArrayList<String> real = new ArrayList<String>();
//insert an empty string in it so that we can use it to remove empty spaces from the processing collection.
        real.add("");
        real.add("(");
        real.add(")");

        list.removeAll(real);

        if (isHasPowerOperators()) {

            /*Deals with powers.Handles the  primary power operator e.g in 3^sin3^4.This is necessary at this stage to dis-allow operations like sinA^Bfrom giving the result:(sinA)^B
instead of sin(A^B).
Also instructs the software to multiply any 2 numbers in consecutive positions in the vector.
This is important in distinguishing between functions such as sinAB and sinA*B.Note:sinAB=sin(A*B),while sinA*B=B*sinA.
             */
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (list.get(i).equals("^") && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i - 1)), Double.valueOf(list.get(i + 1)))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                    }//end if
                    else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i + 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) == 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) > 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) == 0) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) < 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i - 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) == 1) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) > 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) == 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) < 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end else if
                    }//end else if
                    else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        list.set(i + 1, "Infinity");
                        list.set(i - 1, "");
                        list.set(i, "");
                    }

                }//end try
                catch (NumberFormatException numerror) {

                } catch (NullPointerException nullerror) {

                } catch (IndexOutOfBoundsException inderror) {

                }
            }//end for

            list.removeAll(real);

        }//end if

        list.removeAll(real);

        boolean skip = false;
        if (isHasMulOrDivOperators()) {
            for (int i = 0; i < list.size(); i++) {

                try {

                    if (list.get(i).equals(Operator.MULTIPLY)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) * Double.valueOf(list.get(i + 1))));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end if
                    else if (list.get(i).equals(Operator.DIVIDE)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) / Double.valueOf(list.get(i + 1))));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end else if

                }//end try
                catch (NullPointerException nolan) {

                }//end catch
                catch (NumberFormatException numerr) {

                }//end catch
                catch (IndexOutOfBoundsException inderr) {

                }//end catch

            }//end for
            list.removeAll(real);

        }//end if
        if (isHasPlusOrMinusOperators()) {
            //Handles the subtraction and addition operators
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (list.get(i).equals(Operator.PLUS) || list.get(i).equals(Operator.MINUS)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            if (list.get(i).equals(Operator.PLUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) + Double.valueOf(list.get(i + 1))));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else
                            else if (list.get(i).equals(Operator.MINUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else if
                        }//end if
                        else {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    }

                }//end try
                catch (NullPointerException nolerr) {

                }//end catch
                catch (NumberFormatException numerr) {

                }//end catch
                catch (IndexOutOfBoundsException inderr) {
                }//end catch
            }//end for
        }//end if

        real.add("(");
        real.add(")");

        list.removeAll(real);
        if (list.size() != 1) {
            this.correctFunction = false;
        }//end if
//Now de-list or un-package the input.If all goes well the list should have only its first memory location occupied.
        return list;

    }//end method solve

    /**
     * used by the main parser solve to figure out SBP portions of a
     * multi-bracketed expression (MBP)
     *
     * @param list a list of scanner tokens of a maths expression
     * @return the solution to a SBP maths expression
     */
    public List<String> bigDecimalPrecisionSolve(List<String> list) {

//correct the anomaly: [ (,-,number....,)  ]
        //   turn it into: [ (,,-number........,)     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("(") && list.get(1).equals(Operator.MINUS) && isNumber(list.get(2))) {
            list.set(1, "");

            //if the number is negative,make it positive
            if (list.get(2).substring(0, 1).equals(Operator.MINUS)) {
                list.set(2, list.get(2).substring(1));
            } //if the number is positive,make it negative
            else {
                list.set(2, Operator.MINUS + list.get(2));
            }
        }
//Create a collection to serve as a garbage collector for the empty memory
//locations and other unwanted locations created in the processing collection
        ArrayList<String> real = new ArrayList<String>();
//insert an empty string in it so that we can use it to remove empty spaces from the processing collection.
        real.add("");
        real.add("(");
        real.add(")");

        list.removeAll(real);

        if (isHasPowerOperators()) {

            /*Deals with powers.Handles the  primary power operator e.g in 3^sin3^4.This is necessary at this stage to dis-allow operations like sinA^Bfrom giving the result:(sinA)^B
instead of sin(A^B).
Also instructs the software to multiply any 2 numbers in consecutive positions in the vector.
This is important in distinguishing between functions such as sinAB and sinA*B.Note:sinAB=sin(A*B),while sinA*B=B*sinA.
             */
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (list.get(i).equals("^") && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                            BigDecimal lhs = new BigDecimal(list.get(i - 1));
                            String rhs = list.get(i + 1);
                            list.set(i + 1, String.valueOf(lhs.pow(Integer.valueOf(rhs), MathContext.DECIMAL128)));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                    }//end if
                    else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i + 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) == 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) > 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) == 0) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) < 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i - 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) == 1) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) > 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) == 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) < 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end else if
                    }//end else if
                    else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        list.set(i + 1, "Infinity");
                        list.set(i - 1, "");
                        list.set(i, "");
                    }

                }//end try
                catch (NumberFormatException numerror) {

                } catch (NullPointerException nullerror) {

                } catch (IndexOutOfBoundsException inderror) {

                }
            }//end for

            list.removeAll(real);

        }//end if

        list.removeAll(real);

        boolean skip = false;
        if (isHasMulOrDivOperators()) {
            for (int i = 0; i < list.size(); i++) {

                try {

                    if (list.get(i).equals(Operator.MULTIPLY)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            BigDecimal lhs = new BigDecimal(list.get(i - 1));
                            BigDecimal rhs = new BigDecimal(list.get(i + 1));
                            list.set(i + 1, String.valueOf(lhs.multiply(rhs, MathContext.DECIMAL128)));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end if
                    else if (list.get(i).equals(Operator.DIVIDE)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            BigDecimal lhs = new BigDecimal(list.get(i - 1));
                            BigDecimal rhs = new BigDecimal(list.get(i + 1));

                            list.set(i + 1, String.valueOf(lhs.divide(rhs, MathContext.DECIMAL128)));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end else if

                }//end try
                catch (NullPointerException nolan) {

                }//end catch
                catch (NumberFormatException numerr) {

                }//end catch
                catch (IndexOutOfBoundsException inderr) {

                }//end catch

            }//end for
            list.removeAll(real);

        }//end if
        if (isHasPlusOrMinusOperators()) {
            //Handles the subtraction and addition operators
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (list.get(i).equals(Operator.PLUS) || list.get(i).equals(Operator.MINUS)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            if (list.get(i).equals(Operator.PLUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                BigDecimal lhs = new BigDecimal(list.get(i - 1));
                                BigDecimal rhs = new BigDecimal(list.get(i + 1));

                                list.set(i + 1, String.valueOf(lhs.add(rhs, MathContext.DECIMAL128)));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else
                            else if (list.get(i).equals(Operator.MINUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                BigDecimal lhs = new BigDecimal(list.get(i - 1));
                                BigDecimal rhs = new BigDecimal(list.get(i + 1));

                                list.set(i + 1, String.valueOf(lhs.subtract(rhs, MathContext.DECIMAL128)));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else if
                        }//end if
                        else {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    }

                }//end try
                catch (NullPointerException nolerr) {

                }//end catch
                catch (NumberFormatException numerr) {

                }//end catch
                catch (IndexOutOfBoundsException inderr) {
                }//end catch
            }//end for
        }//end if

        real.add("(");
        real.add(")");

        list.removeAll(real);
        if (list.size() != 1) {
            this.correctFunction = false;
        }//end if
//Now de-list or un-package the input.If all goes well the list should have only its first memory location occupied.
        return list;

    }//end method solve

    public static void main(String args[]) {
        String expr = "x=2;8x*sin(x^2)/3";
        PolynomialExpression polyExpression = new PolynomialExpression(expr, DOUBLE_PRECISION);
        System.out.println(polyExpression.solve());
    }//end method

}//end class
