/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.fractions;

import java.util.InputMismatchException;

/**
 *
 * @author GBENRO
 */
public class PrimeNumber {
/**
 * The number attribute of objects of this class.
 */
long number;

    public PrimeNumber(long number) {
        this.number = number;
     if(!isPrime()){
         throw new InputMismatchException("Non-Prime Number!");
     }//end if

     else{
            System.out.println("Prime Number "+number+" Created Successfully!!!!!");
     }
    }//end constructor.

    public void setNumber(long number) {
        this.number = number;
    }

    public long getNumber() {
        return number;
    }

/**
 *
 * @return true if the number is a prime one.
 */
    public boolean isPrime(){
long x =2L;
        for(; x*(x+1) <= number; x++){
        if( number%x == 0  ){
            return false;
        }//end if
        }//end for
return true;
    }//end method






public static void main(String args[]){
    double time = System.nanoTime();
    PrimeNumber num = new PrimeNumber(3777172998448777375L);
    System.out.println("\n\n"+ ( ( System.nanoTime()-time )/1.0E6 )+" ms."   );
}//end method



























}
