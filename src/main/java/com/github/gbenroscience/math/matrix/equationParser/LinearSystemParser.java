/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.equationParser;

import math.matrix.expressParser.Matrix;
import parser.CustomScanner;
import parser.STRING;
import java.util.ArrayList;
import java.util.List;
import util.ErrorLog;
import static parser.STRING.*;
import static parser.Operator.*;
import static parser.Variable.*;
import static parser.Number.*;
/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class LinearSystemParser {
    private ErrorLog logger=new ErrorLog();
private String systemOfEquations="";
private ArrayList<String>scanner=new ArrayList<String>();
private boolean valid=true;
private int dimension;//the dimension of the system....i.e the number of equations it has.
private ArrayList<String>unknowns=new ArrayList<String>();//A list of the unknowns
private Matrix equationMatrix;
//This is the sngle character string that must end each line of equation.
private String endOfLine=":";
    public LinearSystemParser(String systemOfEquations) {
        systemOfEquations = systemOfEquations.replace(" ", "");
        /*
         * remove all white spaces
         */
systemOfEquations=purifier(systemOfEquations);
      if(isVariableString(  systemOfEquations.substring(0,1) ) ){
                systemOfEquations="1.0"+systemOfEquations;
}
this.systemOfEquations=systemOfEquations;
        plusAndMinusStringHandler();

        scan();

/**
 * The scanner of this project does not
 * separate number.concat(variable) tokens but
 * requires them to be separated by the * operator.
 *
 * Our work here however requires this feature so we
 * enable it with this for loop.
 * This loop takes the unscanned number.concat(variable) tokens
 * and separates them into a number token, a * token and a variable token.

 for(int i=0;i<scanner.size();i++){
try{
     if(!scanner.get(i).equals("+")&&!scanner.get(i).equals("-")&&!scanner.get(i).equals("=")&&
             !scanner.get(i).equals(endOfLine)
             &&!validNumber(scanner.get(i))&&!isVariableString(scanner.get(i))){
        MmathScanner scan=new MmathScanner(scanner.get(i));
        ArrayList<String> split=scan.splitStringAtFirstNumber(scanner.get(i));
        if( validNumber( split.get(0) )&&isVariableString(split.get(1)) ){
            scanner.remove(i);
      scanner.addAll(i, split);
      i+=2;
        }
     }

}
catch(IndexOutOfBoundsException indexErr){

}
 }//end for
*/



        appendOneToStartOfFreeVariables();
        countVariablesAndValidateTheNumberOfEqualsAndSemiColons();
        validateChars();
        validateVars();
        validateNumbers();
        validateOperators();
        doAritmetic();
        buildMatrix();

    }

    public void setEndOfLine(String endOfLine) {
        this.endOfLine = (isValidEndOfLineChar(endOfLine))?endOfLine:",";
    }

    public String getEndOfLine() {
        return endOfLine;
    }

private boolean isValidEndOfLineChar(String eof){
    return (eof.equals(",")||eof.equals(":"));
}

/**
 * Handles consecutive strings of plus and minus operators
 * and simplifies them, replacing them with their single equivalent
 * in plus and minus terms.e.g +++---+ is equivalent to a -
 * and --++-- is equivalent to a +.
 */
private void plusAndMinusStringHandler(){

int size=systemOfEquations.length();
    for(int i=0;i<size;i++){

try{
     if(systemOfEquations.substring(i,i+1).equals("-")&&systemOfEquations.substring(i+1,i+2).equals("+") ){
         systemOfEquations=replace(systemOfEquations,  " -", i, i+2);
     }
     else if(systemOfEquations.substring(i,i+1).equals("+")&&systemOfEquations.substring(i+1,i+2).equals("-") ){
         systemOfEquations=replace(systemOfEquations,  " -", i, i+2);
     }
     else if(systemOfEquations.substring(i,i+1).equals("+")&&systemOfEquations.substring(i+1,i+2).equals("+") ){
         systemOfEquations=replace(systemOfEquations,  " +", i, i+2);
     }
     else if(systemOfEquations.substring(i,i+1).equals("-")&&systemOfEquations.substring(i+1,i+2).equals("-") ){
         systemOfEquations=replace(systemOfEquations,  " +", i, i+2);
     }

}
catch(IndexOutOfBoundsException ind){

}
    }//end for


}//end method

    public ArrayList<String> getUnknowns() {
        return unknowns;
    }

    public Matrix getEquationMatrix() {
        return equationMatrix;
    }

    public void setValid(boolean valid) {
        this.valid = valid;
    }

    public boolean isValid() {
        return valid;
    }

    public void setDimension(int dimension) {
        this.dimension = dimension;
    }

    public int getDimension() {
        return dimension;
    }


    public void setSystemOfEquations(String systemOfEquations) {
        this.systemOfEquations = systemOfEquations;
    }

    public String getSystemOfEquations() {
        return systemOfEquations;
    }

    public void setScanner(ArrayList<String> scanner) {
        this.scanner = scanner;
    }

    public ArrayList<String> getScanner() {
        return scanner;
    }
/**
 *
 * @param input  The word to search.
 * @return the first index of occurrence of a letter,
 * and -1 if no letter of the English alphabet is present.
 */
    private int indexOfLetter( String input ){
    int sz=input.length();
        for( int i=0;i<sz;i++){
            if( Character.isLetter(input.substring(i,i+1).toCharArray()[0]) ){
                return i;
            }//end if

        }//end for
return -1;
    }//end method

/**
 *
 * @param input  The word to search.
 * @return true if the input begins with a digit or a point.
 */
    private boolean startsWithDigitOrPoint( String input ){
        String st = input.substring(0,1);
return STRING.isDigit(st)||st.equals(".");
    }//end method

/**
 *
 * @param input  The word to search.
 * @return true if the input begins with a letter.
 */
    private boolean startsWithLetter( String input ){
        String st = input.substring(0,1);
return Character.isLetter(st.toCharArray()[0]);
    }//end method

/**
 * Scan the system.
 */
private void scan(){
  ArrayList<String>remover=new ArrayList<String>();

remover.add("");
CustomScanner cs = new CustomScanner(systemOfEquations, true,
        new String[]{"*","+","-",endOfLine,"=",});
scanner = (ArrayList<String>) cs.scan();
/*
3x-9y+3.23z=9:
1.3x-0.19y+3.1z=22.91:
-2.3E-2x-2.09y+3.01E-2z=1.39:
*/
/**
 * Separate number.concat(variable) complexes.
 */
for( int i=0;i<scanner.size();i++){
    String st = scanner.get(i);
int ind = indexOfLetter(scanner.get(i));
    if( ind != -1 ){
if(startsWithDigitOrPoint(st)){
    try{
    String part1 = st.substring(0, ind);
    String part2 = st.substring(ind);
     scanner.set(i,part1);
     scanner.add(i+1,part2);

    }//end try
    catch(IndexOutOfBoundsException indErr){
    }
}//end if

else if( startsWithLetter(st) && st.substring(0,1).equalsIgnoreCase("E") ){
    try{
    String part1 = st.substring(0, 1);
    String part2 = st.substring(1);
     scanner.set(i,part1);
     scanner.add(i+1,part2);
    }//end try
    catch(IndexOutOfBoundsException indErr){
    }
}//end else if

    }//end if


}//end for

scanner.removeAll(remover);//removes white spaces.
/**
 * Resolve issues with E the exp operator.
 */
for( int i=0;i<scanner.size();i++){

    try{
String st = scanner.get(i);
if( st.equalsIgnoreCase("E")){
    String st1 = scanner.get(i-1);
    String st2 = scanner.get(i+1);
    String st3 = scanner.get(i+2);
    String st4 = scanner.get(i+3);
    String st5 = scanner.get(i+4);
/**
 * 2 scenarios are possible:
 * 1. number,E|e,+|-,number,variable,+|-....e.g 2.089E-5x,or 2.089E+5x
 * OR:
 *    number,E|e,number,variable,+|- 0.089E5x
 *
 */
    /**
     * This handles the first scenario
     */
        if( validNumber(st1) && isPlusOrMinus(st2) && validNumber(st3)&&isVariableString(st4)&&(isPlusOrMinus(st5)||st5.equals("=")) ){
st = st1.concat(st);
st = st.concat(st2);
st = st.concat(st3);
scanner.set(i-1,st);
scanner.subList(i, i+3).clear();


        }//end if
        else if( validNumber(st1) && validNumber(st2)&&isVariableString(st3)&&(isPlusOrMinus(st4)||st4.equals("=")) ){
st = st1.concat(st+"+");
st = st.concat(st2);
scanner.set(i-1,st);
scanner.subList(i, i+2).clear();
        }//end if



            }//end if
    }//end try
    catch(IndexOutOfBoundsException boundsException){

    }
}//end for


     /**
         * In dealing with input like 2x or other number.concat(variable) forms,
         * the scanner  turns them into (,2,*,x,)..so that expressions like sin2x
         * can be correctly interpreted as sin(2x) and not sin2 * x i.e x*sin2.
         * Our work here does not require brackets and the multiplication symbol so
         * we remove them.
         */
remover.add("");
        remover.add("(");
        remover.add(")");
        remover.add("*");
        scanner.removeAll(remover);
}

private void validateChars(){
    for(int i=0;i<scanner.size();i++){
        if(!scanner.get(i).equals("+")&&!scanner.get(i).equals("-")&&!scanner.get(i).equals("=")&&
           !scanner.get(i).equals(endOfLine)&&!validNumber(scanner.get(i))&&
           !isVariableString( scanner.get(i) )){
            setValid(false);
            scanner.clear();
            break;
        }

    }//end for loop
}


/**
 * When the situation x+2y+z=9; is met, this method will convert
 * it into 1.0x+2y+z=9;
 */
public void appendOneToStartOfFreeVariables(){

    for(int i=0;i<scanner.size();i++){
        try{
    if( isVariableString(scanner.get(i))&&!validNumber(scanner.get(i-1))){
     scanner.add(i, "1.0");
    }
        }
        catch(IndexOutOfBoundsException indexErr){

        }
    }

}//end method


/**
 * Counts the number of unknowns in the system and
 * compares it with the number of equations in the
 * system. They must be equal and so it carries out this validation also.
 *
 * This method also fills the ArrayList object that records the unknowns in the system
 * with one copy of each unknown found in the system.
 */

public void countVariablesAndValidateTheNumberOfEqualsAndSemiColons(){
    int numOfVars=0;
    int numOfSemiColons=0;
    int numOfEquals=0;
    ArrayList<String>counter=new ArrayList<String>();
    for(int i=0;i<scanner.size();i++){
        if(isVariableString(scanner.get(i))&&!counter.contains(scanner.get(i))){
            counter.add(scanner.get(i));
         ++numOfVars;
        }
        else if(scanner.get(i).equals("=")){
            ++numOfEquals;
        }
        else if(scanner.get(i).equals(endOfLine)){
            ++numOfSemiColons;
        }
    }//end for

//validate
 setValid(numOfVars==numOfEquals&&numOfEquals==numOfSemiColons);

if(!valid){
    scanner.clear();
}//end if
else{
     dimension=numOfEquals;
}//end else
 unknowns.addAll(counter);
}//end method



/**
 * Checks the left and right of all variables to see if
 * the appropriate items are the ones there.
 * It is a syntax error if an invalid item is found there
 */
public void validateVars(){

 for(int i=0;i<scanner.size();i++){
     try{
if(isVariableString(scanner.get(i))    ){
    if(!validNumber(scanner.get(i-1))){
        setValid(false);break;
    }//end if
if( !scanner.get(i+1).equals(endOfLine)&&!scanner.get(i+1).equals("+")&&!scanner.get(i+1).equals("-")
            &&!isAssignmentOperator( scanner.get(i+1)) ){
    setValid(false);break;
}//end if

}//end if
}//end try
         catch(IndexOutOfBoundsException indexErr){

         }


}//end for

}//end method

/**
 * Checks the left and right of a number to see if
 * the appropriate items are the ones there.
 * It is a syntax error if an invalid item is found there
 */
public void validateNumbers(){
     for(int i=0;i<scanner.size();i++){
         try{
         if(validNumber(scanner.get(i))){
             if(!scanner.get(i-1).equals("+")&&!scanner.get(i-1).equals("-")&&!scanner.get(i-1).equals("=")&&!scanner.get(i-1).equals(endOfLine)){
setValid(false);break;
             }

             if(!isVariableString(scanner.get(i+1))&&!isAssignmentOperator(scanner.get(i+1))
                     &&!scanner.get(i+1).equals("+")&&!scanner.get(i+1).equals("-")&&!scanner.get(i+1).equals(endOfLine)){
setValid(false);break;
             }


         }//end if
         }//end try
         catch(IndexOutOfBoundsException indexErr){

         }
     }//end for
}//end method
/**
 * Checks the left and right of an operator to see if
 * the appropriate items are the ones there.
 * It is a syntax error if an invalid item is found there
 */
public void validateOperators(){

     for(int i=0;i<scanner.size();i++){
         try{
         if(isPlusOrMinus(scanner.get(i))){
             if(!validNumber(scanner.get(i-1))&&!isVariableString( scanner.get(i-1) )
                     &&!isAssignmentOperator( scanner.get(i-1) )&&!scanner.get(i-1).equals(endOfLine)){
setValid(false);break;
             }

             if(!validNumber(scanner.get(i+1))){
setValid(false);break;
             }


         }//end if

         else if(scanner.get(i).equals(endOfLine)){
             if(!validNumber(scanner.get(i-1))&&!isVariableString( scanner.get(i-1) )){
setValid(false);break;
             }

             if(!validNumber(scanner.get(i+1))&&!scanner.get(i+1).equals("+")&&!scanner.get(i+1).equals("-")){
setValid(false);break;
             }
         }//end else if



         }//end try
         catch(IndexOutOfBoundsException indexErr){

         }
     }//end for
}


/**
 * Handles split variable arithmetic in + and - alone.
 *
 * e.g
 * 2x+3y+4z+7x-9=3x-8y+2;
 * 4z-2x-9=3x-8y+2z+9;
 * -9x+5y+x=9x-2y+2;
 *
 * is converted into
 *
 * 6x+11y+4z=11;
 * -5x+8y+2z=18;
 * -17x+7y=2;
 *
 */
public void doAritmetic(){

  ArrayList<String> scan=new ArrayList<String>();

  int size=unknowns.size()*(unknowns.size()+1);



for(int i=0;i<size;i++){
scan.add("0");
}
int indexer=0;



    for(int j=0;j<unknowns.size();j++){
        int toIndex=-1;
        try{
            toIndex=scanner.indexOf(endOfLine);
            if(toIndex==0){
                break;
            }
        }
        catch(IndexOutOfBoundsException indexErr){

        }

   List<String>sublist=scanner.subList(0, toIndex+1);

        for(int i=0;i<sublist.size();i++){
//discover coefficients
            try{
                if(i>0){
if(isNumber(sublist.get(i))&&unknowns.contains(sublist.get(i+1))){
int index=unknowns.indexOf(sublist.get(i+1))+indexer;
double entry=Double.valueOf( sublist.get(i) );
double getValueAtIndex=Double.valueOf(scan.get(index) );
if( i<sublist.indexOf("=") ){
    if(sublist.get(i-1).equals("-") ){
scan.set(index, String.valueOf( getValueAtIndex+ (-entry )  )   );
    }
    else if( !sublist.get(i-1).equals("-") ){
scan.set(index, String.valueOf( getValueAtIndex+ (entry )  )   );
    }
}//end if

else if( i>sublist.indexOf("=") ){
    if( sublist.get(i-1).equals("-")  ){
scan.set(index, String.valueOf( getValueAtIndex- (-entry )  )   );
    }
    else if( !sublist.get(i-1).equals("-")  ){
scan.set(index, String.valueOf( getValueAtIndex- (entry )  )   );
    }
}//end else if



}//end if
     //discover constants
else if(isNumber(sublist.get(i))&&!unknowns.contains(sublist.get(i+1))){
int index=unknowns.size()+indexer;
double entry=Double.valueOf( sublist.get(i) );
double getValueAtIndex=Double.valueOf(scan.get(index) );
if( i<sublist.indexOf("=") ){
    if( sublist.get(i-1).equals("-")  ){
scan.set(index, String.valueOf( getValueAtIndex- (-entry )  )   );
    }
    else if( !sublist.get(i-1).equals("-")  ){
scan.set(index, String.valueOf( getValueAtIndex- (entry )  )   );
    }
}//end if

else if( i>sublist.indexOf("=") ){
    if( sublist.get(i-1).equals("-")  ){
scan.set(index, String.valueOf( getValueAtIndex+ (-entry )  )   );
    }
    else if( !sublist.get(i-1).equals("-")  ){
scan.set(index, String.valueOf( getValueAtIndex+ (entry )  )   );
    }
}//end else if




}//end else if

                }//end if
                else if(i==0){

//detect the coefficients that occur at the beginning of each line.
if(isNumber(sublist.get(i))&&unknowns.contains(sublist.get(i+1))){
int index=unknowns.indexOf(sublist.get(i+1))+indexer;
double entry=Double.valueOf( sublist.get(i) );
double getValueAtIndex=Double.valueOf(scan.get(index) );
scan.set(index, String.valueOf( getValueAtIndex+ (entry )  )   );
}//end if
//detect the constants that occur at the beginning of each line.
else if(isNumber(sublist.get(i))&&!unknowns.contains(sublist.get(i+1))){
int index=unknowns.size()+indexer;
double entry=Double.valueOf( sublist.get(i) );
double getValueAtIndex=Double.valueOf(scan.get(index) );
scan.set(index, String.valueOf( getValueAtIndex-(entry )  )   );
}

           }//end else if
            }//end try
catch(IndexOutOfBoundsException indexErr){

}


 }//end inner for loop.
indexer+=unknowns.size()+1;
sublist.clear();

}//end outer for loop.
scanner.clear();

scanner.addAll(scan);
}//end method


/**
 * Builds the Matrix object that will contain the executable
 * generated by this LinearSystemParser object.
 */
public void buildMatrix(){
    double values[][]=new double[dimension][dimension+1];
    int index=0;
         for(int rows=0;rows<dimension;rows++){
          for(int cols=0;cols<dimension+1;cols++,index++){
             values[rows][cols]=Double.valueOf( scanner.get(index) );
         }
    }

    equationMatrix=new Matrix(values);
}
/**
 * This builds the ideal form
 * of the system with 0 coefficients assigned to
 * parts of the system where an unknown is missing...e.g in
 * 2x+3y+4z=9;
 * 7x+5z=8;
 * 6y=9;
 *
 * This system will eventually be converted into
 * 2x+3y+4z=9;
 * 7x+0y+5z=8;
 * 0x+6y+0z=9;
 *
 */
public String interpretedSystem(){
String system="";
int pass=0;
    for(int i=0;i<scanner.size();i++,pass++){

 if(pass<unknowns.size()){
     system+=(scanner.get(i)+unknowns.get(pass)+"+");
 }//end if.
 else if(pass==unknowns.size()){
     system+=("= "+scanner.get(i)+":\n");
     pass=-1;
 }//end else if.

    }//end for loop.
system=system.replace("+-", "-");
system=system.replace("-+", "-");
system=system.replace("+=", " =");

return system;
}

public static void main(String args[]){
    LinearSystemParser parser = new LinearSystemParser("x-9y-7w-4d+9=9w-4d+9:" +
                                                       "4x-9y+12d-13w=9:" +
                                                       "-9.0123E-5x+6.02E-8w=12-d-d:" +
                                                       "13-123x-7y=9:");
     util.Utils.logError(  parser.getEquationMatrix().toString() );
  util.Utils.logError(  parser.interpretedSystem() );
  util.Utils.logError(  parser.valid+"" );

                                                      /*"2a+3b-9c-3u+6d+6e=3-4u-9d:" +
                                                       "-5a+b+9c-3u+6e+d-10=3-4u+7d:" +
                                                       "a+b+9c-3u-9+6e+5=13-4u:" +
                                                       "5b+b+9c-3u+6e+12=3-4u+13:"+
                                                       "22e-u-10=3a-4u+14:" +
                                                       "-22e+1+u=3a-4u+11:"*/
}
}//end class