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
import com.github.gbenroscience.math.differentialcalculus.symbolic.SymbolicDifferentiator;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.util.FunctionManager;

/**
 *
 * @author GBEMIRO
 */
public class Derivative1 {

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
                MathExpression.Token[] rpnTokens = p.getFunction().getMathExpression().getCachedPostfix();
                String baseVariable = p.getFunction().getIndependentVariables().get(0).getName();
                int orderOfDiff = p.getOrderOfDifferentiation();
                if (p.isNotSetOrderOfDiff()) {
                    orderOfDiff = 1;
                }
                if (p.isGradEval()) {
                    double evalPoint = p.getEvalPoint();
                    SymbolicDifferentiator sd = new SymbolicDifferentiator(rpnTokens);
                    expr = sd.nthDerivative(baseVariable, orderOfDiff);
                    MathExpression me = new MathExpression(baseVariable + "=" + evalPoint + ";" + expr);
                    me.updateArgs(evalPoint);
                    return me.solveGeneric();
                } else {

                    SymbolicDifferentiator sd = new SymbolicDifferentiator(rpnTokens);
                    expr = sd.nthDerivative(baseVariable, orderOfDiff);
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
     * @param orderOfDiff The number of times the function should be
     * differentiated
     * @return
     */
    public static MathExpression.EvalResult eval(Function f, int orderOfDiff) {
        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
            if (orderOfDiff == 0) {
                return f.getMathExpression().getNextResult().wrap(f.getName());
            }
            String baseVariable = f.getIndependentVariables().get(0).getName();

            MathExpression.Token[] rpnTokens = f.getMathExpression().getCachedPostfix();
            try {

                SymbolicDifferentiator sd = new SymbolicDifferentiator(rpnTokens);
                String expr = sd.nthDerivative(baseVariable, orderOfDiff);
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
     * @param returnHandle A function handle/pointer to return the derivative
     * through
     * @param orderOfDiff The number of times the function should be
     * differentiated
     * @return
     */
    public static MathExpression.EvalResult eval(Function f, String returnHandle, int orderOfDiff) {
        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
            if (orderOfDiff == 0) {
                return f.getMathExpression().getNextResult().wrap(f.getName());
            }

            MathExpression.Token[] rpnTokens = f.getMathExpression().getCachedPostfix();
            String baseVariable = f.getIndependentVariables().get(0).getName();
            try {
                SymbolicDifferentiator sd = new SymbolicDifferentiator(rpnTokens);
                String expr = sd.nthDerivative(baseVariable, orderOfDiff);
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
     * @param returnHandle A function handle/pointer to return the derivative
     * through
     * @param orderOfDiff The number of times the function should be
     * differentiated
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
     * @param orderOfDiff The number of timesthe function should be
     * differentiated
     * @return
     */
    public static double eval(Function f, double xVal, int orderOfDiff) {
        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
            if (orderOfDiff == 0) {
                MathExpression me = f.getMathExpression();
                me.updateArgs(xVal);
                me.solve();
            }

            MathExpression.Token[] rpnTokens = f.getMathExpression().getCachedPostfix();
            String baseVariable = f.getIndependentVariables().get(0).getName();
            double evalPoint = xVal;
            try {
                SymbolicDifferentiator sd = new SymbolicDifferentiator(rpnTokens);
                String expr = sd.nthDerivative(baseVariable, orderOfDiff);
                MathExpression me = new MathExpression(baseVariable + "=" + evalPoint + ";" + expr);
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
     * @param orderOfDiff The number of timesthe function should be
     * differentiated
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

        f = "diff(@(x)10*x^5,3)";
        try {
            MathExpression.EvalResult ev = Derivative1.eval(f);
            String res = ev.textRes;
            System.out.println("diff(" + f + ") = " + res);
            System.out.println("Grad Function = " + FunctionManager.lookUp(res));

        } catch (Exception ex) {
            Logger.getLogger(Derivative1.class.getName()).log(Level.SEVERE, null, ex);
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
            System.out.println(Derivative1.eval(expression));

            String expr = "diff(@(x)x^2*cos(x)-2*x*sin(x)-2*cos(x) , 2,1)";
            System.out.println(Derivative1.eval(expr));

        } catch (Exception ex) {
            Logger.getLogger(DerivativeStructureBuilder.class.getName()).log(Level.SEVERE, null, ex);
        }

    }//end method main

}//end class Derivative.
