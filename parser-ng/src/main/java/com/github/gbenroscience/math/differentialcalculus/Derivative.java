/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.differentialcalculus;

import com.github.gbenroscience.parser.MathExpression;
import static com.github.gbenroscience.parser.Number.*;
import com.github.gbenroscience.parser.ParserResult;
import static com.github.gbenroscience.parser.Variable.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import static com.github.gbenroscience.math.differentialcalculus.Utilities.*;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.util.FunctionManager;

/**
 *
 * @author GBEMIRO
 */
public class Derivative {

    public DerivativeStructureBuilder builder;
    /**
     * The base variable that the top level expression is to be differentiated
     * with respect to.
     */
    protected String baseVariable;

    /**
     * Accepted format...diff(expr);or diff(diffName)...where diffName is the
     * name of a stored Differentiable. e.g. diff(sin(3*x+1))
     *
     * @param expression The expression to be parsed into <code>mathExpr</code>
     * and <code>variable</code>.
     *
     *
     */
    Derivative(String expression) throws Exception {
        String mathExpr = "";
        if (isAutoGenNameFormat(expression)) {
            mathExpr = builder.getManager().lookUp(expression).getExpression();
        } else {
            mathExpr = expression;
        }
        this.builder = new DerivativeStructureBuilder(mathExpr);

    }

    public String getMathExpr(String expression) throws Exception {

        if (isAutoGenNameFormat(expression)) {
            return builder.getManager().lookUp(expression).getExpression();
        } else {
            return expression;
        }

    }

    /**
     * @param d The Differentiable item
     * @return an ArrayList containing this object's data in terms of the base
     * variable.
     */
    public ArrayList<String> translateToBaseTerms(Differentiable d) {
        ArrayList<String> data = new ArrayList<>(d.getData());
        print("Expanding " + d.getName() + "---expression list is---" + d.getData());
        for (int i = 0; i < data.size(); i++) {

            if (isAutoGenNameFormat(data.get(i))) {
                String name = data.get(i);
                List<String> temp = builder.getManager().lookUp(name).getData();
                print("found----diff variable---" + data.get(i) + "--internal data is---" + temp);
                List<String> subList = data.subList(i, i + 1);
                subList.clear();
                subList.addAll(temp);
                i -= 1;
                print("Expansion gives------" + d.getData());
            }//end if
        }//end for

        return data;
    }

    /**
     *
     * @return an ArrayList containing this object's data in terms of the base
     * variable.
     */
    public ArrayList<String> translateToBaseTerms_1(Differentiable d) {
        ArrayList<String> data = new ArrayList<>(d.getData());
        for (int i = 0; i < data.size(); i++) {

            if (isAutoGenNameFormat(data.get(i))) {
                String name = data.get(i);
                List<String> temp = builder.getManager().lookUp(name).getData();
                data.remove(i);
                data.addAll(i, temp);
                print("temp---" + temp);
            }//end if
        }//end for

        return data;
    }

    /**
     * Differentiates the expression.
     *
     * @return the derivative as a string of characters.
     */
    public String differentiate() {
        ArrayList<String> array = differentiateAsList();
        StringBuilder build = new StringBuilder("");

        for (String str : array) {
            build.append(str);
        }
        return build.toString();
    }//end method

    /**
     * Differentiates the expression.
     *
     * @return the derivative as a list of scanned tokens.
     */
    public ArrayList<String> differentiateAsList() {
        Differentiable root = builder.getManager().lastDifferentiable();
        print("Here is the chain---" + builder.getManager().getDIFFERENTIABLES());
        print("root---" + root);
        ArrayList<String> array = root.differentiate(this);
        print("diff array---" + array);
        /**
         * Cleaning up the output. Calculus ended. Algebra begins here.
         */
        for (int i = 0; i < array.size(); i++) {
            try {
                if (isAutoGenNameFormat(array.get(i))) {

                    Differentiable d = builder.getManager().lookUp(array.get(i));
                    ArrayList<String> data = translateToBaseTerms(d);
                    array.remove(i);
                    array.addAll(i, data);
                    i -= 2;
                }//end if
            }//end try
            catch (IndexOutOfBoundsException boundsException) {

            }
        }//end for loop
        print("list size before optimizing = " + array.size());
        root.simplifyDerivedData(array);
        print("list size after optimizing = " + array.size());

        for (int i = 0; i < array.size(); i++) {
            try {
                //Simplify x^1 patterns.
                if (isVariableString(array.get(i)) && array.get(i + 1).equals("^") && isNumber(array.get(i + 2)) && Double.parseDouble(array.get(i + 2)) == 1.0) {
                    List temp = array.subList(i + 1, i + 3);
                    temp.clear();
                    i--;
                }

                CodeGenerator.openUpUnnecessaryBrackets(array);

            }//end try
            catch (IndexOutOfBoundsException boundsException) {

            }
        }//end for loop

        return array;
    }//end method

    /**
     *
     * @param name The name to check.
     * @return true if the name is automatically generated and so, most likely
     * refers to a stored Differentiable.
     */
    public boolean isBaseVariable(String name) {
        return name.equals(this.baseVariable);
    }//end method

    /**
     *
     * @param expr The expression to differentiate. It must have the form:
     * diff(@(x)sin(x),2) or diff(@(x)sin(x),2,3)
     *
     * If the diff(@(x)sin(x),2) form is specified, then the function is
     * differentiated 2 times and the result is returned.
     *
     * If the diff(@(x)sin(x),2,3) form is specified, then the function is
     * differentiated 3 times and the result is evaluated at x = 2 and then the
     * value is returned.
     *
     */
    public static MathExpression.EvalResult eval(String expr) {
//the anonymous function to be differentiated: e.g.diff(@(p)(3*p^3+2*p^2-8*p+1),1)
        try {
            Parser p = new Parser(expr);

            if (p.result == ParserResult.VALID) {
                expr = "diff(" + p.getFunction().getMathExpression().getExpression() + ")";
                String baseVariable = p.getFunction().getIndependentVariables().get(0).getName();
                int orderOfDiff = p.getOrderOfDifferentiation();
                if (p.isNotSetOrderOfDiff()) {
                    orderOfDiff = 1;
                }
                if (p.isGradEval()) {
                    double evalPoint = p.getEvalPoint();
                    for (int i = 1; i <= orderOfDiff; i++) {
                        Derivative derivative = new Derivative(expr);
                        derivative.baseVariable = baseVariable;
                        expr = "diff(" + derivative.differentiate() + ")";
                    }//end for loop
                    expr = expr.substring(5, expr.length() - 1);
                    MathExpression me = new MathExpression(baseVariable + "=" + evalPoint + ";" + expr);
                    //System.out.println(baseVariable + "=" + evalPoint + ";" + expr);
                    me.updateArgs(evalPoint);
                    return me.solveGeneric();
                } else {
                    for (int i = 1; i <= orderOfDiff; i++) {
                        Derivative derivative = new Derivative(expr);
                        derivative.baseVariable = baseVariable;
                        expr = "diff(" + derivative.differentiate() + ")";
                    }//end for loop
                    expr = expr.substring(5, expr.length() - 1);
                    String funcExpr = (p.getReturnHandle() == null ? "@(" : p.getReturnHandle() + "=@(") + baseVariable + ")" + expr;
                    Function f = FunctionManager.add(funcExpr);
                    return f.getMathExpression().getNextResult().wrap(f.getName());
                }
            }
            return new MathExpression.EvalResult().wrap(ParserResult.STRANGE_INPUT);
        } catch (Exception e) {
            e.printStackTrace();
            return new MathExpression.EvalResult().wrap(ParserResult.STRANGE_INPUT);
        }
    }

    /**
     *
     * @param f The Function
     * @param orderOfDiff The number of timesthe function should be differentiated
     * @return
     */
    public static MathExpression.EvalResult eval(Function f, int orderOfDiff) {
        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
            if (orderOfDiff == 0) {
                return f.getMathExpression().getNextResult().wrap(f.getName());
            }
            String expr = "diff(" + f.getMathExpression().getExpression() + ")";
            String baseVariable = f.getIndependentVariables().get(0).getName();
            try {
                for (int i = 1; i <= orderOfDiff; i++) {
                    Derivative derivative = new Derivative(expr);
                    derivative.baseVariable = baseVariable;
                    expr = "diff(" + derivative.differentiate() + ")";
                }//end for loop
                expr = expr.substring(5, expr.length() - 1);
                String funcExpr = "@(" + baseVariable + ")" + expr;
                Function ff = FunctionManager.add(funcExpr);
                return ff.getMathExpression().getNextResult().wrap(ff.getName());
            } catch (Exception e) {
                e.printStackTrace();
                return new MathExpression.EvalResult().wrap(ParserResult.STRANGE_INPUT);
            }
        }
        return MathExpression.EvalResult.ERROR;
    }

    /**
     *
     * @param fn A valid function string..e.g. f(x)=sin(x) or f=@(x)cos(x)
     * @param orderOfDiff The order of differentiation
     * @return
     */
    public static MathExpression.EvalResult eval(String fn, int orderOfDiff) {
        Function f = new Function(fn);
        return eval(f, orderOfDiff);
    }
/**
 * 
 * @param f The Function
 * @param returnHandle A function handle/pointer to return the derivative through
 * @param orderOfDiff The number of timesthe function should be differentiated
 * @return 
 */
    public static MathExpression.EvalResult eval(Function f, String returnHandle, int orderOfDiff) {
        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
            if (orderOfDiff == 0) {
                return f.getMathExpression().getNextResult().wrap(f.getName());
            }
            String expr = "diff(" + f.getMathExpression().getExpression() + ")";
            String baseVariable = f.getIndependentVariables().get(0).getName();
            try {
                for (int i = 1; i <= orderOfDiff; i++) {
                    Derivative derivative = new Derivative(expr);
                    derivative.baseVariable = baseVariable;
                    expr = "diff(" + derivative.differentiate() + ")";
                }//end for loop
                expr = expr.substring(5, expr.length() - 1);
                String funcExpr = (returnHandle == null ? "@(" : returnHandle + "=@(") + baseVariable + ")" + expr;
                Function ff = FunctionManager.add(funcExpr);
                return ff.getMathExpression().getNextResult().wrap(ff.getName());
            } catch (Exception e) {
                e.printStackTrace();
                return new MathExpression.EvalResult().wrap(ParserResult.STRANGE_INPUT);
            }

        }
        return MathExpression.EvalResult.ERROR;
    }

    /**
     *
     * @param fn A valid function string..e.g. f(x)=sin(x) or f=@(x)cos(x)
     * @param returnHandle A function handle/pointer to return the derivative through
     * @param orderOfDiff The number of times the function should be differentiated
     * @return
     */
    public static MathExpression.EvalResult eval(String fn, String returnHandle, int orderOfDiff) {
        Function f = new Function(fn);
        return eval(f, returnHandle, orderOfDiff);
    }

    /**
     *
     * @param f The Function
     * @param xVal The value at which the derivative should be evaluated
     * @param orderOfDiff The number of timesthe function should be differentiated
     * @return
     */
    public static double eval(Function f, double xVal, int orderOfDiff) {
        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
            if (orderOfDiff == 0) {
                MathExpression me = f.getMathExpression();
                me.updateArgs(xVal);
                me.solve();
            }
            String expr = "diff(" + f.getMathExpression().getExpression() + ")";
            String baseVariable = f.getIndependentVariables().get(0).getName();
            double evalPoint = xVal;
            try {
                for (int i = 1; i <= orderOfDiff; i++) {
                    Derivative derivative = new Derivative(expr);
                    derivative.baseVariable = baseVariable;
                    expr = "diff(" + derivative.differentiate() + ")";
                }//end for loop
                expr = expr.substring(5, expr.length() - 1);
                MathExpression me = new MathExpression(baseVariable + "=" + evalPoint + ";" + expr);
                //System.out.println(baseVariable + "=" + evalPoint + ";" + expr);
                me.updateArgs(evalPoint);
                return me.solveGeneric().scalar;
            } catch (Exception e) {
                return Double.NaN;
            }

        }
        return Double.NaN;
    }

    /**
     *
     * @param fn
     * @param xVal The value at which the derivative should be evaluated
     * @param orderOfDiff The number of timesthe function should be differentiated
     * @return
     */
    public static double eval(String fn, double xVal, int orderOfDiff) {
        Function f = new Function(fn);
        return eval(f, xVal, orderOfDiff);
    }

    /**
     * @param args
     */
    public static void main(String args[]) {
        String f = "10*x^5";
        Derivative d;
        try {
            d = new Derivative(f);
            System.out.println("diff(" + f + ") = " + d.differentiate());
        } catch (Exception ex) {
            Logger.getLogger(Derivative.class.getName()).log(Level.SEVERE, null, ex);
        }

        f = "diff(@(x)10*x^5,3)";
        try {
            MathExpression.EvalResult ev = Derivative.eval(f);
            String res = ev.textRes;
            System.out.println("diff(" + f + ") = " + res);
            System.out.println("Grad Function = " + FunctionManager.lookUp(res));

        } catch (Exception ex) {
            Logger.getLogger(Derivative.class.getName()).log(Level.SEVERE, null, ex);
        }

        try {
            //MAKE DECISION ON WHETHER TO ENABLE (-x+...
            //OR TO DISABLE IT. ENABLING IT WILL MEAN CONVERTING -x patterns to -1*x....
// String expression = "diff(20-x-sin(sqrt(x-x+3^x+4*x))-x)";
            //"diff(3*x*(x^2)^2*(x^3)*6*5*4/(5-sin(4*x^2-7))^2/5^x+7*x^2)";
            //String expression = "2*x*sin(x+1)";//sin(x)/exp(-3*x^6)   √ ³√

            //String expression = "diff(@(x)cbrt(x)+sqrt(x),2)";///√
            //String expression = "diff(@(x)(3*x+√√x)^2,2,1)";
//8x.sin(x^2)+8x^3.cos(x^2)-1
            String expression = "diff(@(x)4*x*x*sin(x^2)-x,3,1)";
            //System.out.println(Derivative.eval(expression));
            System.out.println(Derivative.eval(expression));

            String expr = "diff(@(x)x^2*cos(x)-2*x*sin(x)-2*cos(x) , 2,1)";
            System.out.println(Derivative.eval(expr));

        } catch (Exception ex) {
            Logger.getLogger(DerivativeStructureBuilder.class.getName()).log(Level.SEVERE, null, ex);
        }

    }//end method main

}//end class Derivative.
