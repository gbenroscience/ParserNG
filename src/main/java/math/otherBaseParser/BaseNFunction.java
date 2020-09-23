/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.otherBaseParser;

import parser.MathExpression;
import parser.Number;
import static parser.Number.*;
import static parser.methods.Method.*;
import static parser.Variable.*;
import java.util.ArrayList;
import math.Maths;

/**
 *
 *@author JIBOYE Oluwagbemiro Olaoluwa
 */
public class BaseNFunction extends MathExpression {
/**
 * The number base
 * that objects of this class
 * will use in computations.
 */
    private int baseOfOperation;
    /**
     *
     * @param input The function to be evaluated.
     * The general format contains variable and
     * constant declarations for variables and constants
     * that are not yet initialized,assignment expressions
     * for those that have been initialized and then an
     * expression to evaluate.
     * e.g.
     * var x =-12; var y =x+1/12; const x1,x2,x3=10;
     * var z =sin(3x-1)+2.98cos(4x);cos(3x+12);
     * The last expression is function to be evaluated and it is always
     * without any equals sign and may or may not end with a semicolon.
     * @param baseOfOperation The number base
     * that objects of this class
     * will use in computations.
     */
    public BaseNFunction( String input,int baseOfOperation ) {
        super(input);
        this.baseOfOperation=baseOfOperation;
        setDRG(1);

        convertNumbersToDecimal();
    }

    public void setBaseOfOperation(int baseOfOperation) {
        this.baseOfOperation = (baseOfOperation>1 && baseOfOperation <= 10)?baseOfOperation:10;
    }

    public int getBaseOfOperation() {
        return baseOfOperation;
    }



    
/**
 * Checks if the numbers in the input
 * are only of the specified base.
 * @return An {@link ArrayList} containing the indices
 * of all numbers in the scanner.
 */
private ArrayList<Integer> isBaseCompatible(){
ArrayList<String>scanner = getScanner();
ArrayList<Integer>numIndices= new ArrayList<Integer>();//stores the indices of all numbers.

for(int j=0;j<scanner.size();j++) {
    String mathObject = scanner.get(j);
    String num = "";
    if (!isDefinedMethod(mathObject) && isVariableString(mathObject)) {
        num = getValue(mathObject);
        numIndices.add(j);
    } else if (validNumber(mathObject)) {
        num = mathObject;
        numIndices.add(j);

    for (int i = 0; i < num.length(); i++) {
        String val = num.substring(i, i + 1);
        if (Character.isDigit(val.toCharArray()[0]) && Integer.valueOf(val) >= baseOfOperation) {
            setCorrectFunction(false);
            break;
        }//end if
    }//end inner for loop
}
if(!isCorrectFunction()){
    break;
}//end if


    }//end outer for loop
return numIndices;
    }//end method


/**
 * Converts all numbers to base 10 system.
 */
private void convertNumbersToDecimal() {
ArrayList<String>scanner = getScanner();
ArrayList<Integer>numIndices = isBaseCompatible();

ArrayList<String>handledVariables = new ArrayList<String>();
if(isCorrectFunction()){

for(int i=0;i<numIndices.size();i++){
    int indexInScanner = numIndices.get(i);
    String mathObject = scanner.get(indexInScanner);

    if(isVariableString(mathObject) && !handledVariables.contains(mathObject)) {
          String value = Maths.changeBase(getValue(mathObject),"10",String.valueOf(baseOfOperation) );
        setValue(mathObject,value);
        handledVariables.add(mathObject);//to avoid changing the value of a variable more than once.
    }
 else if(validNumber(mathObject)){
    scanner.set( indexInScanner, Maths.num_to_base_10(scanner.get(indexInScanner), String.valueOf(baseOfOperation)));
 }

}//end for loop.

}//end if

}//end method.




    @Override
    public String solve(){
        if(isCorrectFunction()){
            String val = super.solve();
            if(Number.isNumber(val)){
                return Maths.changeBase(val,"10", String.valueOf(baseOfOperation));
            }
            else{
                return val;
            }
        }//end if
        else{
            throw new NumberFormatException("Invalid math expression!");
        }//end else
    }//end method.

public static void main( String args[] ){
    BaseNFunction bNFunction = new BaseNFunction("a=10;b=01010;ln(a+b)/cos(a-b)",2);
     System.out.println( bNFunction.getVariableManager() );
    System.out.println(bNFunction.solve());
}


}//end class