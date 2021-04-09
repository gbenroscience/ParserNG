/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import static math.differentialcalculus.Utilities.getText;
import static math.differentialcalculus.Utilities.print;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author GBEMIRO
 */
public class Methods {

	public static final String[] inbuiltOperators = new String[] { "+", "-",
			"*", "/", "^", "(", ")", "-¹", "²", "³" ,"√","³√"};

	/**
	 * A list of all inbuilt methods of the parser of this software.The user is
	 * free to define his own functions.
	 *
	 */
	public static final String[] inbuiltMethods = new String[] { "abs", "sin",
			"cos", "tan", "sinh", "cosh", "tanh", "sin-¹", "cos-¹", "tan-¹",
			"sinh-¹", "cosh-¹", "tanh-¹", "sec", "csc", "cot", "sech", "csch",
			"coth", "sec-¹", "csc-¹", "cot-¹", "sech-¹", "csch-¹", "coth-¹",
			"exp", "ln", "lg", "log", "ln-¹", "lg-¹", "log-¹", "asin", "acos",
			"atan", "asinh", "acosh", "atanh", "asec", "acsc", "acot", "asech",
			"acsch", "acoth", "aln", "alg", "alog", "sqrt","cbrt", "inverse",
			"square", "cube", "pow", "floor", "ceil", "diff", "quad", "intg"
                        };

	/**
	 *
	 * @param name
	 *            The name of the token we wish to verify as an inbuilt method
	 * @return true if the name connotes an inbuilt method's name.
	 */
	public static boolean isInbuiltMethodName(String name) {
		for (String nm : inbuiltMethods) {
			if (nm.equals(name)) {
				return true;
			}
		}
		return false;
	}

	/**
	 *
	 * @param method
	 *            The method name..e.g sin, cos, tan e.t.c.
	 * @param var
	 *            The name of the Differentiable.
	 * @param d
	 *            The invoking Derivative object.
	 * @return the differential coefficient.
	 */
	public static ArrayList<String> getMethodDifferential(String method,String var, Derivative d) {
		String der = "";
		String append = var.equals(d.baseVariable) ? "" : "*diff("
				+ var + ")";

		if (method.equals("sin")) {
			der = "cos(" + var + ")" + append;
		} else if (method.equals("cos")) {
			der = "-sin(" + var + ")" + append;
		} else if (method.equals("tan")) {
			der = "sec(" + var + ")^2" + append;
		} else if (method.equals("sinh")) {
			der = "cosh(" + var + ")" + append;
		} else if (method.equals("cosh")) {
			der = "sinh(" + var + ")" + append;
		} else if (method.equals("tanh")) {
			der = "sech(" + var + ")^2" + append;
		} else if (method.equals("sin-¹") || method.equals("asin")) {
			der = "1/sqrt(1-" + var + "^2)" + append;
		} else if (method.equals("cos-¹") || method.equals("acos")) {
			der = "-1/sqrt(1-" + var + "^2)" + append;
		} else if (method.equals("tan-¹") || method.equals("atan")) {
			der = "1/sqrt(1+" + var + "^2)" + append;
		} else if (method.equals("sinh-¹") || method.equals("asinh")) {
			der = "1/sqrt(" + var + "^2+1)" + append;
		} else if (method.equals("cosh-¹") || method.equals("acosh")) {
			der = "1/sqrt(" + var + "^2-1)" + append;
		} else if (method.equals("tanh-¹") || method.equals("atanh")) {
			der = "1/(1-" + var + "^2)" + append;
		} else if (method.equals("sec")) {
			der = "sec(" + var + ")*tan(" + var + ")" + append;
		} else if (method.equals("csc")) {
			der = "-csc(" + var + ")*cot(" + var + ")" + append;
		} else if (method.equals("cot")) {
			der = "-csc(" + var + ")^2" + append;
		} else if (method.equals("sech")) {
			der = "sech(" + var + ")*tanh(" + var + ")" + append;
		} else if (method.equals("csch")) {
			der = "csch(" + var + ")*coth(" + var + ")" + append;
		} else if (method.equals("coth")) {
			der = "-csch(" + var + ")^2" + append;
		} else if (method.equals("sec-¹") || method.equals("asec")) {
			der = "1/(abs(" + var + ")*sqrt(" + var + "^2-1))" + append;
		} else if (method.equals("csc-¹") || method.equals("acsc")) {
			der = "-1/(abs(" + var + ")*sqrt(" + var + "^2-1))" + append;
		} else if (method.equals("cot-¹") || method.equals("acot")) {
			der = "-1/(1+" + var + "^2)" + append;
		} else if (method.equals("sech-¹") || method.equals("asech")) {
			der = "-1/(" + var + "*sqrt(1-" + var + "^2))" + append;
		} else if (method.equals("csch-¹") || method.equals("acsch")) {
			der = "-1/(" + var + "*sqrt(1+" + var + "^2))" + append;
		} else if (method.equals("coth-¹") || method.equals("acoth")) {
			der = "1/(1-" + var + "^2)" + append;
		} else if (method.equals("exp")) {
			der = "exp(" + var + ")" + append;
		} else if (method.equals("ln")) {
			der = "1/(" + var + ")" + append;
		} else if (method.equals("lg")) {
			der = "1/(" + var + "*ln(10))" + append;
		} else if (method.equals("abs")) {
			der = var + "/abs(" + var + ")" + append;
		} else if (method.equals("log")) {

		} else if (method.equals("ln-¹") || method.equals("aln")) {
			der = "exp(" + var + ")" + append;
		} else if (method.equals("alog") || method.equals("log-¹")) {

		} else if (method.equals("lg-¹") || method.equals("alg")) {
			der = "10^(" + var + ")*ln(10)" + append;
		} else if (method.equals("sqrt")) {
			der = "1/(2*sqrt(" + var + "))" + append;
		} else if (method.equals("cbrt")) {
			der = "1/(3*cbrt(" + var + "^2))" + append;
		} else if (method.equals("inverse")) {
			der = "-" + var + "^-2" + append;
		} else if (method.equals("square")) {
			der = "2*" + var + "" + append;
		} else if (method.equals("cube")) {
			der = "3*" + var + "^2" + append;
		} else if (method.equals("pow")) {

		} else if (method.equals("floor")) {

		} else if (method.equals("ceil")) {

		} else if (method.equals("diff")) {
			try {
				Differentiable diff = d.builder.getManager().lookUp(var);
				String expr = getText(d.translateToBaseTerms(diff));
				print("Expression: " + expr+", var = "+var);
				Derivative drv = new Derivative(expr);
                                drv.baseVariable = d.baseVariable;
                                
				der = drv.differentiate();
			}// end try
			catch (Exception ex) {
				Logger.getLogger(Methods.class.getName()).log(Level.SEVERE,
						null, ex);
			}// end catch
		}// end else if
		else if (method.equals("quad") || method.equals("intg")) {
			der = var;
		}
		try {
			ArrayList<String> scan = new DerivativeScanner("(" + der + ")")
					.getScanner();
			return scan;
		} catch (Exception ex) {
			Logger.getLogger(Methods.class.getName()).log(Level.SEVERE, null,
					ex);
			return null;
		}
	}// end method

}// end class