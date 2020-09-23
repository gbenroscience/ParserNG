/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.util;

import parser.STRING;
import java.util.ArrayList;
import java.util.Random;

/**
 * Objects of this class have the ability to generate a 
 * system of linear equations, randomly.
 *
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class LinearSystemGenerator {

    /**
     * The absolute value of the largest possible number
     * that may be found in the system.
     */
    private int maximumPossibleNumberInSystem;
    /**
     * The number of unknowns in the system.
     */
     private int size;
/**
 * The letter of alphabet that will
 * start the name of the variables
 * that will be used to build the
 * linear system.
 * The system will geerate the variables as x1,x2......
 * where x is any uppercase or lowercase letter of the alphabet
 */
     private String startingLetterOfUnknownName;
/**
 *
 * @param size The number of unknowns that the system will have.
 * @param maximumPossibleNumberInSystem  The absolute value of the largest possible number
 * that may be found in the system.
 * @param startingLetterOfUnknownName The letter of the alphabet that will start the name of the unknown.
 */
    public LinearSystemGenerator(int size, int maximumPossibleNumberInSystem,String startingLetterOfUnknownName) {
        this.size = size;
        this.maximumPossibleNumberInSystem = maximumPossibleNumberInSystem;
        this.startingLetterOfUnknownName = STRING.isWord(startingLetterOfUnknownName)?startingLetterOfUnknownName:"x";
    }//end constructor

    public void setStartingLetterOfUnknownName(String startingLetterOfUnknownName) {
        this.startingLetterOfUnknownName = startingLetterOfUnknownName;
    }

    public String getStartingLetterOfUnknownName() {
        return startingLetterOfUnknownName;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public int getSize() {
        return size;
    }

    public void setMaximumPossibleNumberInSystem(int maximumPossibleNumberInSystem) {
        this.maximumPossibleNumberInSystem = maximumPossibleNumberInSystem;
    }

    public int getMaximumPossibleNumberInSystem() {
        return maximumPossibleNumberInSystem;
    }

    

/**
 *
 * @return an ArrayList containing the unknowns
 */
private ArrayList<String> generateUnknowns(){
    ArrayList<String>vars=new ArrayList<String>();
for(int i=1;i<=size;i++){
    vars.add(startingLetterOfUnknownName+i);
}//end for
return vars;
}
/**
 * @return a double number randomly generated.
 */
private double generateCoefficient(){

Random ran = new Random();
double number = 1.0;
int choiceOfSign=ran.nextInt(2);

/**
 * Generate a number greater than zero or equal to zero
 */
if(choiceOfSign==0){
number = 1+ran.nextInt(maximumPossibleNumberInSystem);

    /**
     * Generate a divisor randomly between zero and the value generated for number.
     */
int val=(int) number;
    double divisor = 1+ran.nextInt(val);
    number/=divisor;



}
/**
 * Generate a number less than zero
 */
else if(choiceOfSign==1){
number = 1+ran.nextInt(maximumPossibleNumberInSystem);
    /**
     * Generate a divisor randomly between zero and the value generated for number.
     */
int val=(int) number;
    double divisor = 1+ran.nextInt(val);
    number/=(-1.0*divisor);
}

return number;
}//end method


/**
 * Builds the liinear system.
 * @return the system after build
 */
private String buildSystem(){
String myLinearSystem="";
ArrayList<String>unknowns =generateUnknowns();

for(int rows=0;rows<size;rows++){
    for(int cols=0;cols<=size;cols++){
        if( cols<(size-1)    ){
myLinearSystem+=(generateCoefficient()+unknowns.get(cols)+"+");
        }//end if
        else if( cols==(size-1) ){
     myLinearSystem+=(generateCoefficient()+unknowns.get(cols));
        }//end else if
        else if(cols == size ){
myLinearSystem+=(" = "+generateCoefficient()+":\n");
        }//end else if


    }//end inner for
}//end outer for

myLinearSystem = myLinearSystem.replace("+-", "-");
myLinearSystem = myLinearSystem.replace("-+", "-");
myLinearSystem = myLinearSystem.replace("--", "+");
myLinearSystem = myLinearSystem.replace("++", "+");
return myLinearSystem;
}


public static void main(String args[]){
    LinearSystemGenerator gen = new LinearSystemGenerator(70 , 100, "x");
System.out.println(gen.buildSystem());
}









}//end class