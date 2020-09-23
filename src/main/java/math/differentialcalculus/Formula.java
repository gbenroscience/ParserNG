/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import java.util.ArrayList;
import java.util.Random;

import static math.differentialcalculus.Utilities.*;
import static parser.Number.*;
import static parser.Operator.*;
import static parser.Variable.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author GBEMIRO Objects of this class represent name-value pairs useful in
 * the simplification of variables on the expression. Portions of a multi
 * bracketed expression are the values and are assigned names. Any 2 different
 * portions having the same content must be assigned the same name.
 *
 * The class must employ techniques such that any 2 or more objects of this
 * class that have the same tokens stored in the <code>expression</code> must
 * have the same <code>name</code>
 *
 *
 * When employing objects of this class, strip the expression of the opening and
 * closing brackets, in fact, no bracket must be found in the expression.
 *
 */
public class Formula {

    /**
     * The name assigned to the object.
     */
    private String name;
    /**
     * The portion of the main expression stored by the object.. usually an SBP.
     */
    private List<String> data;

    /**
     * Stores all the exponents(which are usually lists of compound tokens as
     * the values in this map, the keys are the unique names used to replace
     * them as the power of the token.
     *
     * e.g if x^(2+3*x+4*y) is scanned, we get x,^,(,2,+,3,*,x,+,4,*,y,) the
     * sublist in brackets is stored as a list in the
     * {@link Formula#tokenExponentMap} and its key is the name used to store
     * it. This name is auto-generated and used to replace the sub-list that is
     * in brackets, the brackets included. It is also used as the key in this
     * map. E.g. if the name chosen is formula_1 then the key to the map is
     * formula_1, and its value is that sub-list.
     */
    HashMap<String, List<String>> tokenExponentMap = new HashMap<>();

    public Formula(String name, List<String> data) {
        this.name = name;
        this.data = data;
    }

    public void setData(List<String> data) {
        this.data = data;
    }

    public List<String> getData() {
        return data;
    }

    public static boolean approxEquals(double num1, double num2) {
        return Math.abs((num1 - num2) / num2) <= 1.0E-11;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    /**
     *
     * @return an ArrayList containing all the variable names found in this
     * Formula object.
     */
    public ArrayList<String> getVariables() {
        ArrayList<String> vars = new ArrayList<>();

        for (String s : data) {
            if ((isVariableString(s)) && !vars.contains(s)) {
                vars.add(s);
            }//end if

        }//end for loop.

        return vars;
    }//end method

    /**
     *
     * @param formula The Formula object that we are comparing this Formula
     * object with.
     * @return true if both objects have no variables at all, or if both objects
     * have exactly the same number and the same set of variable names.
     */
    public boolean hasSameVariables(Formula formula) {
        ArrayList<String> vars = new ArrayList<>(this.getVariables());
        ArrayList<String> vars1 = new ArrayList<>(formula.getVariables());

        vars.removeAll(vars1);
        return vars.isEmpty();
    }//end method

    /**
     *
     * @param formula The Formula to express as a scalar factor of this
     * Formula...Evaluates <code>this</code>/<code>formula</code> and returns
     * true if the result will always be a constant and Nan if otherwise.
     * @return a constant which is the ratio of this Formula and the parameter
     * Formula and returns Nan if not possible.
     */
    public double getFactor(Formula formula) {
        boolean same = true;
        /**
         * Try a straightforward token by token comparison
         */
        if (data.size() == formula.data.size()) {
            for (int i = 0; i < data.size(); i++) {
                if (!data.get(i).equals(formula.data.get(i))) {
                    same = false;
                    break;
                }//end if
            }//end for
            if (same) {
                return 1.0;
            }
        }

        /**
         * The two lists are not similar on a straightforward token by token
         * basis. So an intelligent scan needs to be done.
         *
         * Check if the 2 data lists have exactly the same kind of
         * variables.This should be true for situations where both lists have
         * similar variables or where both lists have no variable at all. Valid
         * variables are formula names and the base variable..usually x.
         *
         */
        boolean sameVariables = this.hasSameVariables(formula);
        /**
         * They can't be compared... So comparisons end here.
         */
        if (!sameVariables) {
            return Double.NaN;
        }
        /**
         * Everything done below this point assumes that the 2 Formulae either
         * have no variables or have the same number and the same set of
         * variables.
         */
        ArrayList<String> vars = getVariables();
        ArrayList<String> otherVars = formula.getVariables();
        MultivaluedVariable[] variables = new MultivaluedVariable[vars.size()];
        /**
         * 2 Formula cannot be constant factors of each other if one has
         * variables and one does not.
         */
        if ((vars.isEmpty() != otherVars.isEmpty())) {
            return Double.NaN;
        }
        /**
         * The formulae have no variables. They have only constant values.
         */
        if (vars.isEmpty() && otherVars.isEmpty()) {
            /**
             * Create 2 fresh images of the lists and evaluate.
             */
            ArrayList<String> thisImage = new ArrayList<>(data);
            ArrayList<String> formulaImage = new ArrayList<>(formula.data);
            String v1 = ExpressionSimplifier.solve(thisImage).get(0);
            String v2 = ExpressionSimplifier.solve(formulaImage).get(0);
            return Double.parseDouble(v1) / Double.parseDouble(v2);

        }//end if

        /**
         * The two lists are not similar on a straightforward token by token
         * basis. But the scan shows that they have exactly the same set of
         * variables, so they could still be equal.
         *
         * The actions below will be based on a range-of-values test comparison.
         * Here we evaluate the 2 Formula objects 4 times, all the while
         * assigning 4 different random values to each peculiar variable and
         * then we check if the 2 Formulae always have the same value.
         *
         */
        double[][] values = new double[vars.size()][4];

        //Choose variable test values.
        Random rnd = new Random();

        for (int rows = 0; rows < values.length; rows++) {
            for (int cols = 0; cols < values[rows].length; cols++) {
                values[rows][cols] = 1 + rnd.nextInt(20) + rnd.nextDouble();
            }
            variables[rows] = new MultivaluedVariable(rows, vars.get(rows), values[rows]);
        }//end for loop
        /**
         *
         * Armed with different values for each variable, get 4 different values
         * for the Formulae and compare them. If the 4 values are the same all
         * the time, then the Formulae are arithmetic equivalents.And so can be
         * expanded or reduced into each other.
         *
         */
        boolean sameValue = true;
        double[] ratios = new double[4];
        for (int j = 0; j < variables[0].values.length && sameValue; j++) {
            /**
             * Create 2 fresh images of the lists and substitute the values in
             * them on each iteration.
             */
            ArrayList<String> thisImage = new ArrayList<>(data);
            ArrayList<String> formulaImage = new ArrayList<>(formula.data);

            for (int i = 0; i < thisImage.size(); i++) {
                if (isVariableString(thisImage.get(i))) {
                    /**
                     * Find the index of this variable in the variables's list.
                     */
                    int ind = 0;
                    for (; ind < variables.length; ind++) {
                        if (variables[ind].name.equals(thisImage.get(i))) {
                            break;
                        }
                    }

                    thisImage.set(i, "" + variables[ind].currentValue());

                }//end if
            }//end for loop
            for (int i = 0; i < formulaImage.size(); i++) {
                if (isVariableString(formulaImage.get(i))) {
                    /**
                     * Find the index of this variable in the variables's list.
                     */
                    int ind = 0;
                    for (; ind < variables.length; ind++) {
                        if (variables[ind].name.equals(formulaImage.get(i))) {
                            break;
                        }
                    }

                    formulaImage.set(i, "" + variables[ind].currentValue());

                }//end if
            }//end for loop

            String v1 = ExpressionSimplifier.solve(thisImage).get(0);
            String v2 = ExpressionSimplifier.solve(formulaImage).get(0);

            try {
                double thisresult = Double.parseDouble(v1);
                double formularesult = Double.parseDouble(v2);
                ratios[j] = thisresult / formularesult;

            }//end try
            catch (Exception e) {

            }

            for (MultivaluedVariable mv : variables) {
                mv.nextValue();
            }

        }//end external for loop

        for (int i = 1; i < ratios.length; i++) {
            if (!approxEquals(ratios[i - 1], ratios[i])) {
                return Double.NaN;
            }
        }

        return ratios[0];
    }

    /**
     * This method compares two Formula objects to see if they are
     * mathematically equivalent.
     *
     * @param formula The Formula object to be compared with this Formula.
     * @return true if both objects will always evaluate to a common value.
     */
    public boolean isEquivalentTo(Formula formula) {
        boolean same = true;
        /**
         * Try a straightforward token by token comparison
         */
        if (data.size() == formula.data.size()) {
            for (int i = 0; i < data.size(); i++) {
                if (!data.get(i).equals(formula.data.get(i))) {
                    same = false;
                    break;
                }//end if
            }//end for
            if (same) {
                return same;
            }
        }

        /**
         * The two lists are not similar on a straightforward token by token
         * basis. So an intelligent scan needs to be done.
         *
         * Check if the 2 data lists have exactly the same kind of
         * variables.This should be true for situations where both lists have
         * similar variables or where both lists have no variable at all. Valid
         * variables are formula names and the base variable..usually x.
         *
         */
        boolean sameVariables = this.hasSameVariables(formula);
        /**
         * They can't be compared... So comparisons end here.
         */
        if (!sameVariables) {
            return sameVariables;
        }
        /**
         * Everything done below this point assumes that the 2 Formulae either
         * have no variables or have the same number and the same set of
         * variables.
         */
        ArrayList<String> vars = getVariables();
        MultivaluedVariable[] variables = new MultivaluedVariable[vars.size()];
        /**
         * The formulae have no variables. They have only constant values.
         */
        if (vars.isEmpty()) {
            /**
             * Create 2 fresh images of the lists and evaluate.
             */
            ArrayList<String> thisImage = new ArrayList<>(data);
            ArrayList<String> formulaImage = new ArrayList<>(formula.data);

            String v1 = ExpressionSimplifier.solve(thisImage).get(0);
            String v2 = ExpressionSimplifier.solve(formulaImage).get(0);

            if (v1.equals(v2)) {
                return true;
            }//end if
            //Use an approximate comparator to account for truncation errors.
            else {
                try {
                    double thisresult = Double.parseDouble(v1);
                    double formularesult = Double.parseDouble(v2);

                    if (approxEquals(thisresult, formularesult)) {
                        return true;
                    } else {
                        return false;
                    }

                }//end try
                catch (Exception e) {
                    return false;
                }

            }//end else

        }//end if

        /**
         * The two lists are not similar on a straightforward token by token
         * basis. But the scan shows that they have exactly the same set of
         * variables, so they could still be equal.
         *
         * The actions below will be based on a range-of-values test comparison.
         * Here we evaluate the 2 Formula objects 4 times, all the while
         * assigning 4 different random values to each peculiar variable and
         * then we check if then 2 Formulae always have the same value.
         *
         */
        double[][] values = new double[vars.size()][4];

        //Choose variable test values.
        Random rnd = new Random();

        for (int rows = 0; rows < values.length; rows++) {
            for (int cols = 0; cols < values[rows].length; cols++) {
                values[rows][cols] = 1 + rnd.nextInt(20) + rnd.nextDouble();
            }
            variables[rows] = new MultivaluedVariable(rows, vars.get(rows), values[rows]);
        }//end for loop
        /**
         *
         * Armed with different values for each variable, get 4 different values
         * for the Formulae and compare them. If the 4 values are the same all
         * the time, then the Formulae are arithmetic equivalents.And so can be
         * expanded or reduced into each other.
         *
         */
        boolean sameValue = true;
        for (int j = 0; j < variables[0].values.length && sameValue; j++) {
            /**
             * Create 2 fresh images of the lists and substitute the values in
             * them on each iteration.
             */
            ArrayList<String> thisImage = new ArrayList<>(data);
            ArrayList<String> formulaImage = new ArrayList<>(formula.data);

            for (int i = 0; i < thisImage.size(); i++) {
                if (isVariableString(thisImage.get(i))) {
                    /**
                     * Find the index of this variable in the variables's list.
                     */
                    int ind = 0;
                    for (; ind < variables.length; ind++) {
                        if (variables[ind].name.equals(thisImage.get(i))) {
                            break;
                        }
                    }

                    thisImage.set(i, "" + variables[ind].currentValue());

                }//end if
            }//end for loop
            for (int i = 0; i < formulaImage.size(); i++) {
                if (isVariableString(formulaImage.get(i))) {
                    /**
                     * Find the index of this variable in the variables's list.
                     */
                    int ind = 0;
                    for (; ind < variables.length; ind++) {
                        if (variables[ind].name.equals(formulaImage.get(i))) {
                            break;
                        }
                    }

                    formulaImage.set(i, "" + variables[ind].currentValue());

                }//end if
            }//end for loop

            String v1 = ExpressionSimplifier.solve(thisImage).get(0);
            String v2 = ExpressionSimplifier.solve(formulaImage).get(0);

            if (v1.equals(v2)) {
                sameValue = true;
            }//end if
            //Use an approximate comparator to account for truncation errors.
            else {
                try {
                    double thisresult = Double.parseDouble(v1);
                    double formularesult = Double.parseDouble(v2);

                    if (approxEquals(thisresult, formularesult)) {
                        sameValue = true;
                    } else {
                        sameValue = false;
                    }

                }//end try
                catch (Exception e) {
                    sameValue = false;
                }

            }//end else
            for (MultivaluedVariable mv : variables) {
                mv.nextValue();
            }

        }//end external for loop

        return sameValue;
    }//end method

    /**
     * A Compound token is simply used to describe a group of tokens found
     * between 2 +|- and +|- symbols or before a +|- symbol(at the beginning of
     * the parent list before any +|- has been encountered) or at the end of the
     * parent list after all +|- have been passed. They typically never include
     * any + or - symbol
     *
     * @param list A group of tokens that do not contain any +|- but represent a
     * mathematically correct expression.. e.g. 2,*,x,*,x e.t.c.
     */
    static void simplifyCompoundTokens(List<String> list) {

        if (list.contains("^")) {
            if (list.size() == 3 && isNumber(list.get(0)) && isNumber(list.get(2))) {
                String eval = "" + Math.pow(Double.parseDouble(list.get(0)), Double.parseDouble(list.get(2)));
                list.clear();
                list.add(eval);
                return;
            } else if (list.size() == 3
                    && ((isVariableString(list.get(0)) || isNumber(list.get(0))) && (isNumber(list.get(2)) || isVariableString(list.get(2))))) {
                return;
            }

        } else if (list.size() <= 2) {
            return;
        }
        /**
         *
         * Simplify ^,0 and *,0 and /,0 patterns.
         */
        for (int i = 0; i < list.size(); i++) {
            try {
                if (isNumber(list.get(i)) && Double.parseDouble(list.get(i)) == 0) {
                    switch (list.get(i - 1)) {
                        //end if
                        case "*":
                            list.clear();
                            list.add("0");
                            return;
                        //end else if ty
                        case "/":
                            list.clear();
                            list.add("Infinity");
                            return;
                        //end else if
                        case "^":
                            List sub = list.subList(i - 2, i + 1);
                            sub.clear();
                            sub.add("1");
                            break;
                    }
                }//end if

            }//end try
            catch (IndexOutOfBoundsException in) {
            }//end catch

        }//end for loop.

        /**
         * Handle number,^,number patterns.
         */
        for (int i = 0; i < list.size(); i++) {
            try {
                if (isNumber(list.get(i))) {
                    if (isPower(list.get(i + 1)) && isNumber(list.get(i + 2))) {
                        String eval = "" + Math.pow(Double.parseDouble(list.get(i)), Double.parseDouble(list.get(i + 2)));
                        List sub = list.subList(i, i + 3);
                        sub.clear();
                        sub.add(eval);
                    }
                }//end else if
            }//end try
            catch (IndexOutOfBoundsException in) {

            }//end catch
        }//end for loop.

        /**
         *
         * Simplify multiple power patterns to a simple power pattern..e.g
         * aliasing. For example: If the pattern is x^v where v may be a
         * constant or a variable, then leave unchanged. But if the pattern is
         * of the form: x^a^b^c.... then convert a^b^c...to a single variable,
         * say v. Then set the expression to x^v. Provide a way to map v to
         * a^b^c, so that at the end of all processing, the values can be
         * replaced.
         *
         */
        HashMap<String, List<String>> powerChainPatterns = new HashMap<>();
        boolean powerFound = false;
        String recordPatternVariable = "";
        int count = 0;//Count the number of power operators found in the chain. Once it is more than 1, then the production needs to be aliased.
        int indexOfProductionStart = 0;
        for (int i = 0; i < list.size(); i++) {
            String operator = list.get(i);
            if (isPower(operator) && !powerFound) {
                powerFound = true;
                recordPatternVariable = "temp_var_" + powerChainPatterns.size();
                indexOfProductionStart = i + 1;//skip the first power operator
                ++count;
                continue;
            } else if (isOperatorString(operator) && !isPower(operator) || i == list.size() - 1) {
                powerFound = false;
                List<String> portionOfPattern = list.subList(indexOfProductionStart, i == list.size() - 1 ? i + 1 : i);
                if (count >= 2) {
                    powerChainPatterns.put(recordPatternVariable, new ArrayList<>(portionOfPattern));
                    portionOfPattern.clear();
                    portionOfPattern.add(recordPatternVariable);
                    i = indexOfProductionStart + 1;
                }

                recordPatternVariable = "";
                count = 0;
                continue;
            }
            if (powerFound) {
                if (isPower(operator)) {
                    ++count;
                }
            }

        }

        //At the end of the last loop, all compound power productions would have become simplified to simple power productions.

        /**
         * x^2*x^y/x^-1*x^3*x^x*x^x-----x---------:2- -1+3, y,1, x,2 x^(6+y+2x)
         *
         *
         * The unique thing about a compound token is that it contains no + or - symbol.
         * It consists only of variables, numbers, multiplication, division and power operators.
         *
         * To model this effectively, a loop above reduces all compound operators to a single power operator.
         * What this means is that: x^y^x is reduced to x^a, where a=y^x is recorded somewhere.
         *
         * Now we then create a Map, whose key is a string and whose value is a List of Strings.
         *
         * The coefficient of multiplication of the compound token is entered as key-value pair
         * where the key is "coeff" and the value is whatever the coefficient is.
         *
         * Then we look for variables and numbers and enter them and their powers also in this wise:
         *
         * Say we have: [3, *, x, ^, 4, *, x, ^, x, *, x, ^, x, *, x, ^, y, *, y, ^, x, *, y, ^, 2, *, x, *, 1, /, x]
         * Note that this has not + or - so it is a compound token.
         *
         * The map will store this as:
         *
         * {x=[4.0, x, 2.0, y, 1], y=[2, x, 1], coeff=[3.0]}
         *
         * for the first entry, i.e: x=[4.0, x, 2.0, y, 1], the key is x, which means x, ^ something.
         * That something is deduced from the list [4.0, x, 2.0, y, 1]
         * This list is interpreted as 4+2*x+1*y In other words,
         * if the x has a constant power, that power must be the first entry in the list.
         * So we have x, ^,(,4,+
         * Next, after the 4 comes x which is a power of the base(key), x itself.
         * So we have x, ^,(, 4,+, x,
         * The next token is 2 which serves as a multiplier for x
         * So we have x, ^,(,4,+, 2, *, x,
         * Next we have y,1.
         * So we have x, ^,(,4,+, 2, *, x, +, 1, *, y)
         *
         * Applying this to the remainder:
         * So we have x, ^,(,4,+, 2, *, x, +, 1, *, y), *, y, ^, (, 2, +, x, ), * 3
         */

        HashMap<String, List<String>> varMap = new HashMap<>();
        varMap.put("coeff", new ArrayList<>(Arrays.asList(new String[]{"1"})));

        for (int i = 0; i < list.size(); i++) {
            List<String> coeff = varMap.get("coeff");
            /**
             * x*.... x/.... x^.... 2*... 2/.... 2^.....
             *
             *
             *
             */
            if (i == 0) {//  x/x^-1 OR x*x^-1

                if (isPower(list.get(1))) {

                    //x^3*2/x*5^x
                    // Get the operator before this a^b pattern..It could be a * or / or % or nothing if it is the beginning of the production.
                    String var = list.get(0);
                    String exponent = list.get(2);//3^x/x^2
                    if (!varMap.containsKey(var)) {
                        varMap.put(var, new ArrayList<String>());
                    }

                    List<String> value = varMap.get(var);

                    if (value.isEmpty()) {
                        if (isNumber(exponent)) {
                            value.add(exponent);
                        } else {
                            value.add(exponent);
                            value.add("1");
                        }
                    } else {
                        if (isNumber(exponent)) {
                            if (isNumber(value.get(0))) {
                                value.set(0, String.valueOf(Double.parseDouble(value.get(0)) + Double.parseDouble(exponent)));
                            } else {
                                value.add(0, exponent);
                            }
                        } else {
                            int index;
                            if ((index = value.indexOf(exponent)) != -1) {
                                String coeffOfExponent = value.get(index + 1);
                                value.set(index + 1, String.valueOf(Double.parseDouble(coeffOfExponent) + 1));
                            } else {
                                value.add(exponent);
                                value.add("1");
                            }
                        }
                    }

                } else if (isMulOrDiv(list.get(1))) {
                    if (1 < list.size()) {
                        if (isNumber(list.get(0))) {
                            coeff.set(0, String.valueOf(Double.parseDouble(coeff.get(0)) * Double.parseDouble(list.get(0))));
                        } else if (isVariableString(list.get(0))) {

                            String var = list.get(0);
                            String exponent = "1";

                            if (!varMap.containsKey(var)) {
                                varMap.put(var, new ArrayList<String>());
                            }

                            List<String> value = varMap.get(var);
                            if (value.isEmpty()) {
                                value.add(exponent);
                            } else {
                                if (isNumber(value.get(0))) {
                                    value.set(0, String.valueOf(Double.parseDouble(value.get(0)) + Double.parseDouble(exponent)));
                                } else {
                                    value.add(0, exponent);
                                }

                            }//end else


                        }//end else if
                    }//end if
                }//end else if

            }//end if i==0
            else if(i!=0){
                if (isPower(list.get(i))) {//5,/,x,^,8    //x^3*2/x*5^x
                    // Get the operator before this a^b pattern..It could be a * or / or % or nothing if it is the beginning of the production.
                    String operator = i - 2 < 0 ? "" : list.get(i - 2);
                    String var = list.get(i - 1);
                    String exponent = list.get(i + 1);//3^x/x^2   5,/,x,^,8
                    if (!varMap.containsKey(var)) {
                        varMap.put(var, new ArrayList<String>());
                    }

                    List<String> value = varMap.get(var);
                    switch (operator) {
                        case "":

                            break;
                        case "*":
                            if (value.isEmpty()) {
                                if (isNumber(exponent)) {
                                    value.add(exponent);
                                } else {
                                    value.add(exponent);
                                    value.add("1");
                                }
                            } else {
                                if (isNumber(exponent)) {
                                    if (isNumber(value.get(0))) {
                                        value.set(0, String.valueOf(Double.parseDouble(value.get(0)) + Double.parseDouble(exponent)));
                                    } else {
                                        value.add(0, exponent);
                                    }
                                } else {
                                    int index;
                                    if ((index = value.indexOf(exponent)) != -1) {
                                        String coeffOfExponent = value.get(index + 1);
                                        value.set(index + 1, String.valueOf(Double.parseDouble(coeffOfExponent) + 1));
                                    } else {
                                        value.add(exponent);
                                        value.add("1");
                                    }
                                }
                            }
                            break;
                        case "/":

                            if (value.isEmpty()) {
                                if (isNumber(exponent)) {
                                    value.add( (-1*Double.parseDouble(exponent))+"");
                                } else {
                                    value.add(exponent);
                                    value.add("-1");
                                }
                            } else {
                                if (isNumber(exponent)) {
                                    if (isNumber(value.get(0))) {
                                        value.set(0, String.valueOf(Double.parseDouble(value.get(0)) - Double.parseDouble(exponent)));
                                    } else {
                                        value.add(0, String.valueOf(0 - Double.parseDouble(exponent)));
                                    }
                                } else {
                                    int index;
                                    if ((index = value.indexOf(exponent)) != -1) {
                                        String coeffOfExponent = value.get(index + 1);
                                        value.set(index + 1, String.valueOf(Double.parseDouble(coeffOfExponent) - 1));
                                    } else {
                                        value.add(exponent);
                                        value.add("-1");
                                    }
                                }
                            }
                            break;
                        default:

                            break;

                    }//end switch

                }//end if
                else if ( isMulOrDiv(list.get(i)) ) {//9*x^3*y^x/3*4*2/6...//5,/,x,^,8

                    if(isNumber(list.get(i + 1))){

                        if ((i + 2 < list.size() && isMulOrDiv(list.get(i + 2))) || i + 1 == list.size() - 1) {
                            switch (list.get(i)) {
                                case "*":
                                    coeff.set(0, String.valueOf(Double.parseDouble(coeff.get(0)) * Double.parseDouble(list.get(i + 1))));
                                    break;
                                case "/":
                                    coeff.set(0, String.valueOf(Double.parseDouble(coeff.get(0)) / Double.parseDouble(list.get(i + 1))));
                                    break;
                                default:
                                    break;
                            }
                        }
                    }

                    else if (isVariableString(list.get(i + 1))) {//...*|/ x *|/ ...
                        if (i + 2 < list.size() && isMulOrDiv(list.get(i + 2)) || i + 1 == list.size() - 1) {
                            String var = list.get(i + 1);
                            String exponent = list.get(i).equals("*") ? "1" : "-1";

                            if (!varMap.containsKey(var)) {
                                varMap.put(var, new ArrayList<String>());
                            }

                            List<String> value = varMap.get(var);
                            if (value.isEmpty()) {
                                value.add(exponent);
                            } else {
                                if (isNumber(value.get(0))) {
                                    value.set(0, String.valueOf(Double.parseDouble(value.get(0)) + Double.parseDouble(exponent)));
                                } else {
                                    value.add(0, exponent);
                                }
                            }
                        }

                    }

                }//end else if
            }//end else if i > 0

        }//end for loop

        print("varMap: "+varMap);
        list.clear();

        if (Double.parseDouble(varMap.get("coeff").get(0)) == 0) {
            list.add("0");
            return;
        } else if (Double.parseDouble(varMap.get("coeff").get(0)) == 1) {
            list.add("1");
            list.add("*");
        } else {
            list.add(varMap.get("coeff").get(0));
            list.add("*");
        }


        for (Map.Entry<String, List<String>> entrySet : varMap.entrySet()) {
            String key = entrySet.getKey();
            List<String> value = entrySet.getValue();

            //Remove zero powers. e.g. x^0 is already 1
            if(value.size()==1){
                if( Double.parseDouble(value.get(0)) == 0){
                    continue;
                }
            }

            if (!key.equals("coeff")) {

                list.add(key);
                list.add("^");
                list.add("(");
                boolean hasConstantExp = isNumber(value.get(0));//set to true if one of the powers of the variable is a constant.
                if (hasConstantExp) {
                    for (int i = 0; i < value.size();) {
                        if (i == 0) {
                            if (Double.parseDouble(value.get(0)) != 0) {
                                list.add(value.get(0));
                            }
                            i++;
                            continue;
                        }
                        if (i % 2 == 1) {
                            double multiplier = Double.parseDouble(value.get(i + 1));
                            if (multiplier < 0) {
                                list.add("-");
                                list.add(String.valueOf(Math.abs(multiplier)));
                                list.add("*");
                                list.add(value.get(i));
                            } else if (multiplier > 0) {
                                if (!list.get(list.size() - 1).equals("(")) {

                                    list.add("+");
                                }
                                if (multiplier == 1) {
                                    list.add(value.get(i));
                                } else {
                                    list.add(String.valueOf(Math.abs(multiplier)));
                                    list.add("*");
                                    list.add(value.get(i));
                                }
                            }
                            i += 2;
                        }

                    }//end for loop

                }//end if
                else {
                    for (int i = 0; i < value.size(); i += 2) {
                        double multiplier = Double.parseDouble(value.get(i + 1));
                        if (multiplier < 0) {
                            list.add("-");
                            list.add(String.valueOf(Math.abs(multiplier)));
                            list.add("*");
                            list.add(value.get(i));
                        } else if (multiplier > 0) {
                            if (!list.get(list.size() - 1).equals("(")) {

                                list.add("+");
                            }

                            if (multiplier == 1) {
                                list.add(value.get(i));
                            } else {
                                list.add(String.valueOf(Math.abs(multiplier)));
                                list.add("*");
                                list.add(value.get(i));
                            }
                        }

                    }//end for loop

                }
                list.add(")");

                list.add("*");

            }
        }

        list.remove(list.size() - 1);

        for (int i = 0; i < list.size(); i++) {
            String var = list.get(i);
            if (isPower(var) && isNumber(list.get(i+1)) && Double.parseDouble(list.get(i+1))==1) {
                list.remove(i + 1);
                list.remove(i);
                i--;
            }

            if (isVariableString(var)) {
                List<String> powerPattern = powerChainPatterns.get(var);
                if (powerPattern != null) {
                    List sub = list.subList(i, i + 1);
                    sub.clear();//remove the dummy variable.
                    sub.addAll(powerPattern);//replace the variable with the original pattern
                    i += powerPattern.size() - 1;
                }
            } else if (i + 2 < list.size() && isOpeningBracket(var) && isClosingBracket(list.get(i + 2))) {
                list.remove(i + 2);
                list.remove(i);
                i-=2;
            }


        }

    }//end method.

    /**
     * Simplifies a Single-Bracket-Pair algebraic expressions's tokens.
     *
     * @param list An ArrayList of tokens of an algebraic expressions.
     */
    public static void simplify(List<String> list) {

        for (int i = 0; i < list.size(); i++) {

            switch (list.get(i)) {
                case "-¹":
                    List<String> sub = list.subList(i, i + 1);
                    sub.clear();
                    sub.add("^");
                    sub.add("-1");
                    break;
                case "²":
                    sub = list.subList(i, i + 1);
                    sub.clear();
                    sub.add("^");
                    sub.add("2");
                    break;
                case "³":
                    sub = list.subList(i, i + 1);
                    sub.clear();
                    sub.add("^");
                    sub.add("3");
                    break;

            }
        }//end for loop

        //correct the anomaly: [ ,-,number....,  ]
        //   turn it into: [ -number........     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("-") && isNumber(list.get(1))) {
            list.remove(0);
            list.set(0, "" + (-1 * Double.parseDouble(list.get(0))));
        }//end if

        if (!list.contains("+") && !list.contains("-")) {
            simplifyCompoundTokens(list);
            return;
        }

        ArrayList<List<String>> compoundTokens = new ArrayList<>();

        String lastOperator = "";
        for (int i = 0; i < list.size(); i++) {
            String operator = list.get(i);
            if (isPlusOrMinus(operator)) {

                if (lastOperator.equals("")) {//detect the first compound token
                    List<String> subList = list.subList(0, i);
                    compoundTokens.add(new ArrayList(subList));
                    subList.clear();
                    i = 0;
                } else {
                    List<String> subList = list.subList(1, i);
                    compoundTokens.add(new ArrayList(subList));//avoid the starting +|- operator
                    if (lastOperator.equals("-")) {
                        compoundTokens.get(compoundTokens.size() - 1).addAll(0, Arrays.asList("-1", "*"));
                    }
                    list.subList(0, i).clear();//delete
                    i = 0;
                }
                lastOperator = operator;
            } else if (i == list.size() - 1) {
                if (lastOperator.equals("")) {
                    compoundTokens.add(new ArrayList(list));

                    list.clear();
                } else {
                    List<String> subList = list.subList(1, list.size());
                    compoundTokens.add(new ArrayList(subList));//avoid the starting +|- operator
                    if (lastOperator.equals("-")) {
                        compoundTokens.get(compoundTokens.size() - 1).addAll(0, Arrays.asList("-1", "*"));
                    }
                    list.clear();//delete
                }
                break;
            }

        }


        /*
         for (int i = 0; i < compoundTokens.size(); i++) {
         simplifyCompoundTokens(compoundTokens.get(i));
         }//end for loop
         System.err.println("After simplifying all compound tokens: compound tokens list: "+compoundTokens);


         */
        ArrayList<List<String>> simplifiedCompoundTokens = new ArrayList<>();

        for (int i = 0; i < compoundTokens.size(); i++) {
            List<String> compoundToken = compoundTokens.get(i);//ty
            if (compoundToken.isEmpty()) {
                continue;
            }
            Formula f = new Formula("myForm_1", compoundToken);//Avoid the +|- sign at the beginning of the List.
            double coeff = 1;
            for (int j = i + 1; j < compoundTokens.size(); j++) {
                try {

                    if (compoundTokens.get(j).isEmpty()) {
                        continue;
                    }
                    Formula fj = new Formula("myForm_2", compoundTokens.get(j));
                    if (f.isEquivalentTo(fj)) {
                        coeff += 1;
                        compoundTokens.remove(j);
                        j--;
                        continue;
                    }
                    double factor = fj.getFactor(f);

                    if (!Double.isNaN(factor)) {
                        coeff += factor;
                        compoundTokens.remove(j);
                        j--;
                    }//end if
                }//end try
                catch (IndexOutOfBoundsException in) {

                }
            }//end inner for loop.
            //If problematic,remove the if surrounding the 2 lines of code below.
            if (coeff != 1) {
                f.data.add(0, "*");
                f.data.add(0, "" + coeff);
            }
            simplifyCompoundTokens(f.data);
            simplifiedCompoundTokens.add(f.data);

        }//end outer for loop.

        list.clear();
        for (int i = 0; i < simplifiedCompoundTokens.size(); i++) {
            if (simplifiedCompoundTokens.get(i).size() == 1 && isNumber(simplifiedCompoundTokens.get(i).get(0))
                    && Double.parseDouble(simplifiedCompoundTokens.get(i).get(0)) == 0) {

            }//end if
            else {
                List<String> lst = simplifiedCompoundTokens.get(i);

                if (!lst.isEmpty() && (lst.get(0).equals("-1") || lst.get(0).equals("-1.0"))) {
                    if (lst.size() == 1) {
                        lst.add(0, "+");
                        list.addAll(lst);
                    } else if (lst.get(1).equals("*")) {
                        lst.subList(0, 2).clear();
                        lst.add(0, "-");
                        list.addAll(lst);
                    }

                }//end if
                else {
                    if (i != 0) {
                        lst.add(0, "+");
                    }
                    list.addAll(lst);
                }
            }//end else

        }//end for loop.

        for (int i = 0; i < list.size(); i++) {
            if (list.get(i).equals("+") && list.get(i + 1).substring(0, 1).equals("-")) {
                list.set(i, "-");
                list.set(i + 1, list.get(i + 1).substring(1));
            } else if (list.get(i).equals("-") && list.get(i + 1).substring(0, 1).equals("-")) {
                list.set(i, "+");
                list.set(i + 1, list.get(i + 1).substring(1));
            }

        }

    }//end method

    private class MultivaluedVariable {

        /**
         * In cases where this object has to be referenced from an array,this
         * attribute tells exactly where to find the object in the array.
         */
        int index;
        /**
         * The name of the object.
         */
        String name;
        /**
         * Navigates through the <code>values</code> array and so determines the
         * current value of the variable.
         */
        int cursor = 0;
        /**
         * The various values which this object can hold.
         */
        double[] values;

        /**
         *
         * @param index In cases where this object has to be referenced from an
         * array,this attribute tells exactly where to find the object in the
         * array.
         * @param name The name of the object.
         * @param values The various values which this object can hold.
         */
        public MultivaluedVariable(int index, String name, double[] values) {
            this.index = index;
            this.name = name;
            this.values = values;
        }

        public void setCursor(int cursor) {
            this.cursor = cursor;
        }

        /**
         *
         * @return the current value of the object.
         */
        public double currentValue() {
            return values[cursor];
        }

        /**
         *
         * @return Changes the value of the object to the next one and returns
         * that value.
         */
        public double nextValue() {
            if (cursor < values.length - 1) {
                ++cursor;
            } else {
                cursor = 0;
            }
            return values[cursor];
        }

    }//end inner class

    @Override
    public String toString() {
        return "(name:" + name + ",data:" + data + ")";
    }



}
