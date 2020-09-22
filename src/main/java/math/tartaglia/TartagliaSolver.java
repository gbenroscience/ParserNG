/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.tartaglia;
 
import math.quadratic.QuadraticSolver;
import static java.lang.Math.*;

/**
 *
 * Objects of this class are
 * real value solvers of the system:
 * cx^3+ax+b=0.
 * @author JIBOYE OLuwagbemiro Olaoluwa
 */
public class TartagliaSolver {
    private double a;//coefficient of x.
    private double b;//coefficient of the constant.
    private double c;//coefficient of x^3.

    public TartagliaSolver(double c, double a, double b) {
            this.a=a;
            this.b=b;
            this.c=c;
normalizeCofficients();

    }




    public double getA() {
        return a;
    }

    public void setA(double a) {
        this.a = a;
    }

    public double getB() {
        return b;
    }

    public void setB(double b) {
        this.b = b;
    }

    public double getC() {
        return c;
    }

    public void setC(double c) {
        this.c = c;
    }




/**
 * Make the value of c to be equal to one by
 * dividing through a,b,c by its value
 *
 */
public void normalizeCofficients(){
    try{
    setA(a/c);
    setB(b/c);
    setC(1.0);

    }
    catch(ArithmeticException arit){
        util.Utils.logError( "THE COEFFICIENT OF x^3 MUST NOT BE ZERO".toUpperCase());
    }
}
/**
 * Finds the real roots
 * of the equation.
 */
public String solve(){

String x1="0.0";

if(a>0){

try{
   double lhs=(b/2.0);
   double rhs= sqrt( ( pow(a,3)/27.0 ) +   ( pow(b,2)/4.0 ) );
 x1=String.valueOf( cbrt( -lhs+rhs)+cbrt( -lhs-rhs) );

 QuadraticSolver solver = new QuadraticSolver(1.0,Double.valueOf(x1), a+ pow( Double.valueOf(x1),2) );

String allSolutions=x1+",\n"+solver.solve();

return allSolutions;
}//end try
catch(NumberFormatException numErr){
    return "SYNTAX ERROR";
}//end catch

}//end if
else{
    return "THE COEFFICIENT OF X CANNOT BE LESS THAN OR EQUAL TO ZERO.";
}//end else



}//end method







}//end class
