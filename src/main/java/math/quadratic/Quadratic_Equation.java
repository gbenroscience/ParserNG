/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.quadratic;

import java.util.ArrayList;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class Quadratic_Equation {
private QuadraticExpressionParser parser;//The equation parser
private QuadraticSolver algorithm;//The equation solver.

    public Quadratic_Equation(String equation) {
        this.parser = new QuadraticExpressionParser(equation);
        ArrayList<Double>coeffs=parser.getCoefficients();
        this.algorithm = new QuadraticSolver(coeffs.get(0),coeffs.get(1),coeffs.get(2) );
    }

    public void setParser(QuadraticExpressionParser parser) {
        this.parser = parser;
    }

    public QuadraticExpressionParser getParser() {
        return parser;
    }


    public void setAlgorithm(QuadraticSolver algorithm) {
        this.algorithm = algorithm;
    }

    public QuadraticSolver getAlgorithm() {
        return algorithm;
    }


/**
 *
 * @return the reduced form of the system in the form  Ax²+Bx+C=0;
 */
    public String interpretedSystem(){
        return parser.interpretedSystem();
    }

/**
 *
 * @return The solutions to the equation as
 * a string of values
 *
 */
public String solutions(){
   return algorithm.solve();
}



public static void main(String args[]){
    Quadratic_Equation eqn=new Quadratic_Equation("-2*X^2+3*X+1+X=-9-9X^2");
    System.out.println(eqn.solutions());
}

}
