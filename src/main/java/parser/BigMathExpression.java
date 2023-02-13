/*
 * Copyright 2023 gbemirojiboye.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package parser;

import java.util.*;
import java.math.*;
import static parser.Variable.*;
import static parser.Number.*;
import static parser.Operator.*;
import static parser.methods.Method.*;
import util.FunctionManager;

/**
 *
 * @author gbemirojiboye
 */
public class BigMathExpression extends MathExpression {

    public BigMathExpression(String expression) throws InputMismatchException {
        super(expression);
        util.Utils.loggingEnabled = true;
        
        if (isHasInbuiltFunctions() || isHasLogicOperators() || isHasListReturningOperators()
                || isHasPreNumberOperators() || isHasNumberReturningNonUserDefinedFunctions()
                || isHasPermOrCombOperators() || isHasRemainderOperators() || isHasPostNumberOperators()) {
            setCorrectFunction(false);
            util.Utils.logError("Functions not supported yet!");
        } // end if

        // Check if the user defined functions do not contain any function whose bigmath
        // operation is yet undefined by ParserNG
        if(recursiveHasContrabandFunction(this)){
            setCorrectFunction(false);
           throw new InputMismatchException("Cannot support functions for now!");
        }
    }
    /**
     * This function scans a MathExpression for functions that ParserNG cannot
     * evaluate using big math yet.
     * At the moment, this includes all functions that are not simple algebraic
     * functions.
     * 
     * @return true if it finds any contraband function
     */
    private boolean recursiveHasContrabandFunction(MathExpression expression) {
        // Check if the user defined functions do not contain any function whose big math
        // operation is yet undefined by ParserNG
        if (expression.hasFunctions) {
            int sz = scanner.size();
            for (int i = 0; i < sz; i++) {
                String token = scanner.get(i);
                if (i + 1 < sz) {
                    String nextToken = scanner.get(i + 1);
                    if (isOpeningBracket(nextToken)) {
                        if(isInBuiltMethod(token)){
                            return true;//contraband
                        }
                        Function f = FunctionManager.getFunction(token);
                            if (f != null) {
                                if(f.getType() != Function.ALGEBRAIC){
                                    return true;//contraband
                                }
                                MathExpression me = f.getMathExpression();
                                if(me.hasInbuiltFunctions){
                                    return true;
                                }else if(me.hasUserDefinedFunctions){
                                    return recursiveHasContrabandFunction(me);
                                }
                            }
                    }
                }
            } // end for
        }

        return false;
    }


    /**
     * used by the main parser solve to figure out SBP portions of a
     * multi-bracketed expression (MBP)
     *
     * @param list a list of scanner tokens of a maths expression
     * @return the solution to a SBP maths expression
     */
    @Override
    protected List<String> solve(List<String> list) {
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
        




    public static void main(String[] args) {

        MathExpression me = new MathExpression("a,v,d=2/3;b=3;f=3ab;p(x)=x^3+5*x^2-4*x+1;p(9);");
        System.out.println(me.solve());
        BigMathExpression bme = new BigMathExpression("x=2;h(x)=5*x^2+x+2*x-cos(x);h(3);");
        System.out.println(bme.solve());

        MathExpression bm = new MathExpression("x=9;f(x)=3*x^2+sin(x^2);f(6)<3");
        System.out.println(bm.solve());
    }

}
