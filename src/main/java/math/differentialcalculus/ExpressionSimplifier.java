/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import parser.Bracket;
import static parser.Number.*;
import static parser.Operator.*;
import static parser.Variable.*;

import java.util.ArrayList;
import java.util.List;
import math.Maths;
import static math.differentialcalculus.Utilities.*;



/**
 * Objects of this class seek to 
 * simplify a math expression, by taking it
 * through a series of transformative processes.
 * @author GBEMIRO
 */
public class ExpressionSimplifier {
/**
 * An ArrayList containing the tokens of an algebraic expression.
 */
    ArrayList<String>scanned;
    
    FormulaManager manager = new FormulaManager();

    
    public ExpressionSimplifier(ArrayList<String>scanned) {
        this.scanned = scanned;
        ArrayList<String> duplicate = new ArrayList<>(scanned);
        
        
        /**
         * Read in and simplify the whole expression.
         */
        for(int i=0;i<duplicate.size();i++){
            
            if(duplicate.get(i).equals(")")){
               int open = Bracket.getComplementIndex(false, i, duplicate);
               List<String>subList = duplicate.subList(open, i+1);
               subList.remove(0);//remove the open bracket.
               subList.remove(subList.size()-1);//remove the close bracket.
               
               /**
                * The sublist has only 1 element apart
                * from its brackets
                */
               //if(subList.size()==1){
               //    continue;
               //}
               
              Object[]info = manager.comparisonData(subList);
              /**4+(2*8*x-2)^2/x-3 
               * 
               * 
               * Check for a comparable Formula in the store..A comparable formula is one which is mathematically
               * equal in value, or which is a constant ratio to the Formula object that encapsulates this sublist.
               */ 
              if(Double.parseDouble(info[0].toString())==0){
                    int indexOfFormula = (int) Double.parseDouble(info[1].toString());//ty
                  //Formula found is equivalent to incoming Formula.
                  if(Double.parseDouble(info[2].toString())==1){
                      try {
                          Formula stored = manager.getFormula(indexOfFormula);
                          subList.clear();
                          subList.add(stored.getName());
                      }//end try
                      catch (ClassNotFoundException ex) {
                      }//end catch
                  }//end if
                  //Formula found is factor of incoming Formula.----where: factor=data/stored_formula.data
                  else{
                      try {
                          double factor = Double.parseDouble(info[2].toString());
                          Formula stored = manager.getFormula(indexOfFormula);
                             subList.clear();
                             subList.add("(");
                             subList.add(""+factor);
                             subList.add("*");
                             subList.add(stored.getName());
                             subList.add(")");
                             /**
                              * Note the enclosing brackets? they are necessary in case
                              * the expression we are substituting for has a power operator
                              * or some similar operator.This allows the operator to affect the
                              * whole expression under the bracket.
                              * e.g. ...,+,(x,+,1,),^,2...If the x+1
                              * is being replaced by 0.4*myForm_3, then
                              * we have ...,+,(,0.4,*,myForm_3,),^,2..If we ignore
                              * the brackets then we have the error expression:
                              * ...,+,0.4,*,myForm_3,^,2.
                              */
                      }//end try
                      catch (ClassNotFoundException ex) {
                      }//end catch
                  }//end else
                  
              }//end if
              /**
               * 
               * 
               * Comparable Formula not found in store so create a new Formula,
               * substitute its name for the sublist data in the expression
               * and then store the Formula.
               */ 
              
              else{
                  String name = generateName();
                  Formula.simplify(subList);
                  Formula f = new Formula(name, new ArrayList<String>(subList));
                  subList.clear();
                  subList.add(f.getName());
                  manager.add(f);
                  
              }
              
              
              
                
               
            }//end if
            print(duplicate);
        }//end for loop
        
       
        
     
        
        
    }//end constructor.
    
    
    /**
     * 
     * @return an ArrayList containing the simplified 
     * expression.
     */
    public ArrayList<String> getSimplifiedExpression(){
       // scanned.clear();
     return scanned = translateToBaseTerms(manager.lastFormula());  
    }
    
/**
 * 
 * @return an ArrayList containing a Formula object's data in terms
 * of the base variable.
 */    
    private ArrayList<String>translateToBaseTerms(Formula f){
        ArrayList<String>data=new ArrayList<String>(f.getData());
        for(int i=0;i<data.size();i++){
            
            if(isFormula(data.get(i))){
                String name = data.get(i);
              List<String>temp = data.subList(i, i+1);
              temp.clear();
              temp.addAll(manager.lookUp(name).getData());
              if(temp.size()>1){
              temp.add(0,"(");
              temp.add(")");
              }
            }//end if
        }//end for
        

        return data;
    } 
    
    
    /**
     * Simplifies tokens in a List that have no + or - operator,
     * but may have brackets.
     * e.g. ((x^2)*(2/x)*...)
     * @param list The List containing the tokens.
     */
    public static void simplifyCompoundBrackets(List<String>list){
        
       /**
        * Identify the variables.
        */
                ArrayList<String>vars = new ArrayList<String>();
        /*
         * This first index will contain the constant value.
         * Following indices will have variable,power patterns.
         * e.g x,2 means x^2 and so on.
         * x,2,x 
         */ 
        vars.add("1");
        for(int i=0;i<list.size();i++){
        if(isVariableString(list.get(i))&&!vars.contains(list.get(i))){
            vars.add(list.get(i));
            vars.add("0");
        }//end if

        }//end for loop. 
        
        
        
     
        
        
        
        
        
        
        
    }
    
    
    /**
     * Used by the main parser solve to figure out SBP portions of a multi-bracketed expression (MBP)
     * @param list a list of scanner tokens of a maths expression
     * @return the solution to a SBP maths expression
     */
    public static List<String> solve(List<String> list) {

        
     boolean isHasPostNumberOperators=false;
     boolean isHasPowerOperators=false;
     boolean isHasPreNumberOperators=false;
     boolean isHasPermOrCombOperators=false;
     boolean isHasMulOrDivOperators=false;
     boolean isHasRemainderOperators=false;
     boolean isHasLogicOperators=false;
     boolean isHasPlusOrMinusOperators=false;
     
        for (int i = 0; i < list.size(); i++) {
            if (isPlusOrMinus(list.get(i))) {
                isHasPlusOrMinusOperators = true;
            } else if (isUnaryPreOperator(list.get(i))) {
                isHasPreNumberOperators = true;
            } else if (isUnaryPostOperator(list.get(i))) {
                isHasPostNumberOperators = true;
            } else if (isMulOrDiv(list.get(i))) {
                isHasMulOrDivOperators = true;
            } else if (isPermOrComb(list.get(i))) {
                isHasPermOrCombOperators = true;
            } else if (isRemainder(list.get(i))) {
                isHasRemainderOperators = true;
            } else if (isPower(list.get(i))) {
                isHasPowerOperators = true;
            } else if (isLogicOperator(list.get(i))) {
                isHasLogicOperators = true;
            }
        }//end for     
     
     
    
//correct the anomaly: [ (,-,number....,)  ]
        //   turn it into: [ (,,-number........,)     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("(") && list.get(1).equals("-") && isNumber(list.get(2))) {
            list.set(1, "");
            //if the number is negative,make it positive
            if (list.get(2).substring(0, 1).equals("-")) {
                list.set(2, list.get(2).substring(1));
            } //if the number is positive,make it negative
            else {
                list.set(2, "-" + list.get(2));
            }
        }
        
//correct the anomaly: [ ,-,number....,  ]
        //   turn it into: [ -number........     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("-") && isNumber(list.get(1))) {
            list.remove(0);
                list.set(0, ""+(-1*Double.parseDouble(list.get(0))));
        }
        
//Create a collection to serve as a garbage collector for the empty memory
//locations and other unwanted locations created in the processing collection
        ArrayList<String> real = new ArrayList<String>();
//insert an empty string in it so that we can use it to remove empty spaces from the processing collection.
        real.add("");
        real.add("(");
        real.add(")");

        list.removeAll(real);

//solves the factorial component of the input|[²]|[³]|[-¹]²³-¹
        if (isHasPostNumberOperators) {
            for (int i = 0; i < list.size(); i++) {
                try {

                    if (isFactorial(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(Double.valueOf(Maths.fact(list.get(i)))));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else

                    } else if (isSquare(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i)), 2)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else
                    } else if (isCube(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i)), 3)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else
                    } else if (isInverse(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(1 / Double.valueOf(list.get(i))));
                            list.set(i, "");
                        } else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "0.0");
                            list.set(i, "");
                        }//end else
                    }
                }//end try
                catch (NumberFormatException numerror) {
                } catch (NullPointerException nullerror) {
                } catch (IndexOutOfBoundsException inderror) {
                }
            }//end for
            list.removeAll(real);
        }//end if

        if (isHasPowerOperators) {

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
        //Handles the pre-number operators.


        if (isHasPreNumberOperators) {
            for (int i = list.size() - 1; i >= 0; i--) {
                try {
                    if (!list.get(i + 1).equals("Infinity") && isNumber(list.get(i + 1))) {
                        if (list.get(i).equals("√")) {
                            list.set(i, String.valueOf(Math.sqrt(Double.valueOf(list.get(i + 1)))));
                            list.set(i + 1, "");
                        }//end if
//add more pre-number functions here...
                    }//end if
                    else if (list.get(i + 1).equals("Infinity")) {
                        list.set(i, "Infinity");
                        list.set(i + 1, "");
                    }//end else if
                }//end try
                catch (NumberFormatException numerror) {
                } catch (NullPointerException nullerror) {
                } catch (IndexOutOfBoundsException inderror) {
                }
            }//end for
        }//end if

        list.removeAll(real);





        if (isHasPowerOperators) {
            //do the in between operators

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

        if (isHasPermOrCombOperators) {
            //do the lower precedence in between operators

            for (int i = 0; i < list.size(); i++) {
                try {
                    if (list.get(i).equals("Р")) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.valueOf(Maths.fact(list.get(i - 1))) / (Double.valueOf(Maths.fact(String.valueOf(Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))))))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                        else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }

                    }//end if
                    else if (list.get(i).equals("Č")) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.valueOf(Maths.fact(list.get(i - 1))) / (Double.valueOf(Maths.fact(String.valueOf(Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))))) * Double.valueOf(Maths.fact(list.get(i + 1))))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                        else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
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
        boolean skip = false;
        if (isHasMulOrDivOperators || isHasRemainderOperators || isHasLogicOperators ) {
            for (int i = 0; i < list.size(); i++) {

                try {

                    if (list.get(i).equals("*")) {
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
                    else if (list.get(i).equals("/")) {
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
                    else if (list.get(i).equals("%")) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) % Double.valueOf(list.get(i + 1))));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, list.get(i - 1));
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

                if (!skip) {
                    try {
                        if (list.get(i).equals("==")) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) == 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
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



                    try {
                        if (list.get(i).equals(">")) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) > 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
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


                    try {
                        if (list.get(i).equals("≥")) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) >= 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
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


                    try {
                        if (list.get(i).equals("<")) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) < 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }

                        }//end if
                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch


                    try {
                        if (list.get(i).equals("≤")) {
                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {

                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) <= 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }
                        }//end if


                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch

                }//end if (skip)
            }//end for
            
            list.removeAll(real);

        }//end if
        if (isHasPlusOrMinusOperators) {
            //Handles the subtraction and addition operators
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (list.get(i).equals("+") || list.get(i).equals("-")) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            if (list.get(i).equals("+") && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) + Double.valueOf(list.get(i + 1))));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else
                            else if (list.get(i).equals("-") && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
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
           // this.correctFunction = false;
        }//end if
//Now de-list or un-package the input.If all goes well the list should have only its first memory location occupied.
        return list;


    }//end method solve
        
    

    
    
    

    

      /**
     * Automatically generates a name for 
     * a given Formula object..especially
     * since these are automatically created on the
     * fly during expression simplification.
     * @return a unique name for the object.
     */
    private String generateName(){
        int count = manager.count();
        return "myForm_"+count;
    }//end method
    
    
        
}//end class