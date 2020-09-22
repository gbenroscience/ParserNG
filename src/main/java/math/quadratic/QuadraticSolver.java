package math.quadratic;
import static java.lang.Math.*;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author GBEMIRO
 */


public class QuadraticSolver {
  
    private double a;
    private double b;
    private double c;


/**
 *
 * @param a the co-efficient of the squared variable
 * @param b the co-efficient of the variable
 * @param c the constant term
 */
    public QuadraticSolver(double a, double b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
    }
/**
 * 
 * @param a the co-efficient of the squared variable
 */
    public void setA(double a) {
        this.a = a;
    }
/**
 *
 * @return the co-efficient of the squared variable
 */
    public double getA() {
        return a;
    }
/**
 *
 * @param b the co-efficient of the variable
 */
    public void setB(double b) {
        this.b = b;
    }
/**
 *
 * @return the co-efficient of the squared variable
 */
    public double getB() {
        return b;
    }
/**
 *
 * @param c the constant term
 */
    public void setC(double c) {
        this.c = c;
    }
/**
 *
 * @return the constant term
 */
    public double getC() {
        return c;
    }



/**
 *
 * @return the solutions as a string of values.
 */
public String solve(){
String result="";

if(  (pow(b,2)-4*a*c)>=0  ){
    result=String.valueOf ( (-b/(2*a))+ ( sqrt(  pow(b, 2) - 4*a*c)/(2*a) ) )+",  "+String.valueOf ( (-b/(2*a))- ( sqrt(  pow(b, 2) - 4*a*c)/(2*a) ) );
}
else if(  (pow(b,2)-4*a*c)<0  ){
    result=String.valueOf (-b/(2*a))+"+"+String.valueOf(  ( sqrt( 4*a*c - pow(b, 2) )/(2*a) )   )+" i  ,\n"+
            String.valueOf (-b/(2*a))+"-"+String.valueOf(  ( sqrt( 4*a*c - pow(b, 2) )/(2*a) )   )+" i ";
}
//2p^2-3p-4.09=0
result=result.replace("+-", "-");
result=result.replace("-+", "-");
result=result.replace("--", "+");
return result;
}
/**
 *
 * @return the solutions as an array of values
 */
public String[] soln(){

String result[]=new String[2];

if(  (pow(b,2)-4*a*c)>=0  ){
result[0]=String.valueOf ( (-b/(2*a))+ ( sqrt(  pow(b, 2) - 4*a*c)/(2*a) ) );
result[1]=String.valueOf ( (-b/(2*a))- ( sqrt(  pow(b, 2) - 4*a*c)/(2*a) ) );
}
else if(  (pow(b,2)-4*a*c)<0  ){
result[0]=   String.valueOf (-b/(2*a))+"+"+String.valueOf(  ( sqrt( 4*a*c - pow(b, 2) )/(2*a) )   )+" i";
result[1]=   String.valueOf (-b/(2*a))+"-"+String.valueOf(  ( sqrt( 4*a*c - pow(b, 2) )/(2*a) )   )+" i ";
}
//2p^2-3p-4.09=0

result[0]=result[0].replace("+-", "-");
result[0]=result[0].replace("-+", "-");
result[0]=result[0].replace("--", "+");

result[1]=result[1].replace("+-", "-");
result[1]=result[1].replace("-+", "-");
result[1]=result[1].replace("--", "+");

return result;
}




    @Override
public String toString(){
    return a+"x²+"+b+"x+"+c+"=0";
}

public static void main(String args[]){
    QuadraticSolver ohMy = new QuadraticSolver(3,-9,6);
  System.out.println(ohMy.soln()[0]+", "+ohMy.soln()[1]);

    System.out.println(ohMy );
}


}//end class QuadraticSolver
