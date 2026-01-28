/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.tartaglia;

import java.util.ArrayList;

/**
 *
 * @author JIBOYE OLuwagbemiro Olaoluwa
 */
public class Tartaglia_Equation {
private TartagliaExpressionParser parser;//The equation parser
private TartagliaSolver algorithm;//The equation solver.

    public Tartaglia_Equation(String equation) {
        this.parser = new TartagliaExpressionParser(equation);
        ArrayList<Double>coeffs=parser.getCoefficients();
        this.algorithm = new TartagliaSolver(coeffs.get(0),coeffs.get(1),coeffs.get(2) );
    }

    public void setParser(TartagliaExpressionParser parser) {
        this.parser = parser;
    }

    public TartagliaExpressionParser getParser() {
        return parser;
    }


    public void setAlgorithm(TartagliaSolver algorithm) {
        this.algorithm = algorithm;
    }

    public TartagliaSolver getAlgorithm() {
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
    Tartaglia_Equation eqn=new Tartaglia_Equation("-X^3-5X-7=0");
    System.out.println(eqn.solutions());
}

}
