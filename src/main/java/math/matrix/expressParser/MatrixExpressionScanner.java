/**
 *
 *
 * Scanner objects created from this template
 * are operator prioritizing.
 * So they first seek out the operators
 * in the input and tokenize them.
 *
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;


import parser.Number;
import parser.STRING;
import parser.Variable;


import java.util.*;
import java.util.ArrayList;
 
import math.matrix.MatrixVariableManager;
import util.*;
import static parser.STRING.*;
import static parser.Operator.*;
import static parser.Variable.*;
import static parser.Number.*;
import static math.matrix.expressParser.MatrixVariable.*;
/**
 *
 * @author GBEMIRO
 */
public class MatrixExpressionScanner{

    /**
     * Returns true if the expression is validated by the scanner
     * as containing all the necessary objects and no foreign object.
     * This does not validate syntax but only validates that the objects
     * used are the valid ones of a mathematical language
     */

/**
 * Contains a list of Variable objects
 * used that are not declared.
 */
    private ArrayList<String>errorList=new ArrayList<String>();
private boolean runnable=true;

/**
 * The math expression to be scanned.
 *
 */
private String scannerInput;
/**
 * The character used to identify that an expression like sin 1
 * means see sin and 1 as separate tokens.
 *
 */
private String delimiter=" ";

/**
 * Sets the first occurence of a Variable object in this Scanner object.
 */
private int firstVarIndex;

/**
 * Sets the first occurence of a Variable object in this Scanner object.
 */
private int firstOpIndex;


/**
 * Contains the scanned expression
 */
private ArrayList<String> scanner=new ArrayList<String>();

/**
 *
 * @param scannerInput the input of this Scanner object
 */
    public MatrixExpressionScanner(String scannerInput) {

        //±–


/**
 * The loop looks for the occurence of
 * + and - operators that
 * occur as a part of numbers and not as
 * operators on numbers and replaces them with ± and –
 * respectively.
 * This is needed while it looks for operator tokens which
 * we do not wish to confuse these ones with.
 * Once we have our operator tokens, we can again convert
 * them back into their real forms.
 */
  for(int i=0;i<scannerInput.length();i++){
      try{
   if ( ( isDigit(scannerInput.substring(i,i+1))||scannerInput.substring(i,i+1).equals("."))&&
           scannerInput.substring(i+1,i+2).equals("E")&&
           ( scannerInput.substring(i+2,i+3).equals("+") )
           &&isDigit( scannerInput.substring(i+3,i+4) ) ){
        scannerInput= STRING.replace(scannerInput,"±", i+2, i+3);
   }
   else if ( ( isDigit(scannerInput.substring(i,i+1))||scannerInput.substring(i,i+1).equals("."))&&
           scannerInput.substring(i+1,i+2).equals("E")&&
           ( scannerInput.substring(i+2,i+3).equals("-") )
           &&isDigit( scannerInput.substring(i+3,i+4) ) ){
            scannerInput=STRING.replace(scannerInput,"–", i+2, i+3);//replacing the minus operator with the En-Dash symbol
   }

      }//end try
      catch(IndexOutOfBoundsException indexErr){

      }

  }//end for
/*
 * Convert the - in negative numbers of a statistical
 * data set into ~.
 */
    for(int i=0;i<scannerInput.length();i++){
      if(scannerInput.substring(i, i+1).equals(",")&&
         scannerInput.substring(i+1, i+2).equals("-")&&
              (scannerInput.substring(i+2,i+3).equals(".")||isDigit(scannerInput.substring(i+2,i+3)) ) ){
         scannerInput=replace(scannerInput,"~", i+1, i+2);
      }



    }//sort(5,3,2,1,-8,-9,12,34,98,-900,34,23,12,340)



        this.scannerInput =scannerInput;

    }
/**
 *
 * @param scannerInput sets the input of this Scanner object
 */
    public void setScannerInput(String scannerInput) {
        this.scannerInput = scannerInput;
    }
/**
 *
 * @return the input into the scanner.
 */
    public String getScannerInput() {
        return scannerInput;
    }
/**
 *
 * @param delimiter sets the delimiter to be used by this Scanner object
 */
    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }
/**
 *
 * @return  the delimiter to be used by this Scanner object
 */
    public String getDelimiter() {
        return delimiter;
    }
/**
 *
 * @param firstVarIndex the index of occurence of the very first
 * instance of a given token type.
 */
    public void setFirstVarIndex(int firstVarIndex) {
        this.firstVarIndex = firstVarIndex;
    }
/**
 *
 * @return the index of occurence of the very first
 * instance of a given token type.
 */
    public int getFirstVarIndex() {
        return firstVarIndex;
    }
/**
 *
 * @param scanner sets the ArrayList object that holds the scanned output.
 */
    public void setScanner(ArrayList<String> scanner) {
        this.scanner = scanner;
    }
/**
 *
 * @return the ArrayList object that holds the scanned output.
 */
    public ArrayList<String> getScanner() {
        return scanner;
    }
/**
 *
 * @param firstOpIndex sets the index of the first
 * MOperator object in the input
 */
    public void setFirstOpIndex(int firstOpIndex) {
        this.firstOpIndex = firstOpIndex;
    }
/**
 *
 * @return the index of the first
 * MOperator object in the input
 */
    public int getFirstOpIndex() {
        return firstOpIndex;
    }
/**
 *
 * @param runnable sets whether object of this class are runnable on a calculating
 * device or not.
 */
    public void setRunnable(boolean runnable) {
        this.runnable = runnable;
    }
/**
 *
 * @return true if the object of this class is runnable on a calculating
 * device.
 */
    public boolean isRunnable() {
        return runnable;
    }
/**
 *
 * @param errorList sets the list of errors.
 */
    public void setErrorList(ArrayList<String> errorList) {
        this.errorList = errorList;
    }
/**
 *
 * @return the list of errors.
 */
    public ArrayList<String> getErrorList() {
        return errorList;
    }




/**
 * Removes commas from the ArrayList object that contains
 * the scanned function.
 */

private void removeCommas(){
    List<String>commaList=new ArrayList<String>();
    commaList.add(",");
    scanner.removeAll(commaList);

}

/**
 *  Will retrieve the first occurence of a full Variable
 * in a number String. e.g in 3ABC it will return ABC
 * But in A,B,C it will return A
 *
 * Also it will store information about the index of discovery of the first variable
 * in field firstIndex, and -1 in it if no variable is found
 * @param val the String from which the first Variable object
 * name is to be retrieved.
 * @return the first Variable object in
 * the input string.
 */

public  String getFirstVariableInString( String val)
{
String varString="";
int size =val.length();

boolean found =false;


for(int i=0;i<size;i++){
    if(Variable.isVariableBeginner(val.substring(i, i+1))&&!found){
        varString=val.substring(i, i+1);
        found = true;
        firstVarIndex=i;
        try{
            //look ahead
        if(!Variable.isVariableBuilder(val.substring(i+1, i+2))){
            break;
        }//end inner if
        }//end try
        catch(IndexOutOfBoundsException indErr){

        }//end catch
    }//end if
    else if(Variable.isVariableBuilder(val.substring(i, i+1))&&found){
        varString+=val.substring(i, i+1);
try{
        if(!Variable.isVariableBuilder(val.substring(i+1, i+2))){
            break;
        }//end if
}//end try
catch(IndexOutOfBoundsException indErr){
    break;
}//end catch
    }//end else if


}//end for

if(varString.equals("")){
    firstVarIndex=-1;
}

//3AC-4+7sin5

return varString;
}//end method getFirstVariableInString(args)

/**
 *
 * @return the index where the first Variable was found
 * and -1 if none is found
 */

public int getIndexOfFirstVariable(){
return firstVarIndex;
}

/**
 * method splitStringOnVariables takes a String object and returns an ArrayList of substrings
 * in which the original string has been split into different parts based on the amount of
 * Variable substrings it has.
 * e.g if val="2ABC+D+5.21q-$34"
 * it will return [2,ABC,+,D,+,5.21,q,-,$34]
 * @param val
 * @return An ArrayList of substrings consisting of Variable substrings and the other substrings
 * of the input.
 */

public  ArrayList<String> splitStringOnVariables(String val)
{//3AC-4+7sin5

      ArrayList<String>split=new ArrayList<String>();
   ArrayList<String>filter=new ArrayList<String>();

    int varIndex=2;
    String vars="A";
while(val.length()>0){

  try{
   vars=getFirstVariableInString(val);
   varIndex=getIndexOfFirstVariable();

   if(varIndex==0){
    split.add(vars);
    val=val.substring(varIndex+vars.length());
}
   //no variable found
   else if(varIndex==-1){
       split.add(val);
       val= "";
   break;
   }//else if
else{
     split.add(val.substring(0,varIndex));
     split.add(vars);
     val=val.substring(varIndex+vars.length());
}
 //vars = getFirstVariableInString();
 if(vars.equals("")){
     split.add(val);
 }



  }//end try
catch(IndexOutOfBoundsException indErr){
    split.add(val);
    val="";
    break;
}
}//end while

return split;
}



/**
 * method getFirstNumberSubstring takes a String object and returns the first
 * number string in the String.
 * @param val the string to analyze
 * @return the index of the first occurence of a number character in the sequence.
 * @throws StringIndexOutOfBoundsException
 */

public String getFirstNumberSubstring(String val) throws StringIndexOutOfBoundsException{



 /*The number model used to detect the number is....
  * numberstring...a
  * floating point.p
  * numberstring...b
  * exp symbol.....e
  * operator.......s
  * numberstring...c
  * There should be an operator symbol before the first number
  * but in this scenario we do not handle that.
  *
  *
  */
String num1="";
String num2="";
String point="";
String exp="";
String sign="";
String num3="";
    int plusCount=0;
    int minusCount=0;
    int expCount=0;
    int pointCount=0;

    int ind=getFirstIndexOfDigit(val);
    try{
if(val.substring(ind-1,ind).equals(".")){
    num2=".";
    pointCount++;
}
    }
    catch(IndexOutOfBoundsException in){
        num2="";num1="";
    }
//num=a+p+b+e+s+c
    if(num2.equals(".")){
        try{
for(int i = ind;i<val.length();i++){
      if(!isNumberComponent(val.substring(i,i+1))){
        break;
    }
      //a number cannot contain more than 1 decimal point.
     if(val.substring(i,i+1).equals(".")){
         break;
     }

       //if a + is encountered and no sign operator has been encountered then parse it accordingly
    if(val.substring(i,i+1).equals("+")){
        if(expCount==0){
            break;
        }
        if(plusCount==1||minusCount==1){
            break;
        }
        plusCount++;

    }
        //if a - is encountered and no sign operator has been encountered then parse it accordingly
    if(val.substring(i,i+1).equals("-")){
        if(expCount==0){
            break;
        }
   if(minusCount==1||plusCount==1){
            break;
        }
        minusCount++;

    }
    if(val.substring(i,i+1).equals("E")){
       if(expCount==1){
            break;
        }
        exp="E";
        expCount++;

    }


    //build the decimal part of the number on the condition that it must occur
    //before the exp symbol is encountered
if(isDigit(val.substring(i,i+1))&&(pointCount==1&&expCount==0)){
point+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals("+")&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals("-")&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}


if(isDigit(val.substring(i,i+1))&&( (pointCount==0&&expCount==1) ||( pointCount==1&&expCount==1 ) ) ){
num3+=val.substring(i,i+1);
}




}//end for

    }//end try
    catch(IndexOutOfBoundsException indexerr){

    }
    }//end if
//num=a+p+b+e+s+c
    else if(num2.equals("")){
        try{
for(int i = ind;i<val.length();i++){
    if(!isNumberComponent(val.substring(i,i+1))){
        break;
    }
if(isDigit(val.substring(i,i+1))&&(pointCount==0&&expCount==0)){
num1+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals(".")){
    //disallow decimal points after an exponent symbol has been parsed
         if(expCount==1){
            break;
        }
         //disallow more than 1 point from being parsed
        if(pointCount==1){
            break;
        }
        num2=".";
        pointCount++;

    }
    //if a + is encountered and no sign operator has been encountered then parse it accordingly
    if(val.substring(i,i+1).equals("+")){
        if(expCount==0){
            break;
        }
        if(plusCount==1||minusCount==1){
            break;
        }
        plusCount++;

    }
        //if a - is encountered and no sign operator has been encountered then parse it accordingly
    if(val.substring(i,i+1).equals("-")){
        if(expCount==0){
            break;
        }
   if(minusCount==1||plusCount==1){
            break;
        }
        minusCount++;

    }
    if(val.substring(i,i+1).equals("E")){
       if(expCount==1){
            break;
        }
        exp="E";
        expCount++;

    }


    //build the decimal part of the number on the condition that it must occur
    //before the exp symbol is encountered
if(isDigit(val.substring(i,i+1))&&(pointCount==1&&expCount==0)){
point+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals("+")&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals("-")&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}




if(isDigit(val.substring(i,i+1))&&( (pointCount==0&&expCount==1) ||( pointCount==1&&expCount==1 ) ) ){
num3+=val.substring(i,i+1);
}


}//end for
    }//end try
    catch(IndexOutOfBoundsException indexerr){

    }


    }//end else if


//klnsdkn23.94E2jwjji
    //klwejh.7856E++43
    String result=num1+num2+point+exp+sign+num3;
try{
if(lastElement(result).equals("+")||lastElement(result).equals("-")){
     result=result.substring(0, result.length()-1);
}
if(lastElement(result).equals("+")||lastElement(result).equals("-")){
     result=result.substring(0, result.length()-1);
}
if(lastElement(result).equals("E")){
    result=result.substring(0, result.length()-1);
}


}
catch(IndexOutOfBoundsException in){}
return result;
}//end method getFirstNumberSubstring
/**
 *
 * @param num The number string
 * @return true if it represents a valid number.
 */
public boolean isNumber(String num){
    return getFirstNumberSubstring(num).equals(num);
}//end method


/**
 *
 * @param val the String to analyze
 * @return an ArrayList consisting of 2 or 3 parts.
 * a. If a number substring starts the string,then it returns an ArrayList object
 * containing the number substring and the part of the string after the number.
 * e.g. for 231.62A+98B+sin45C, the method returns [231.62,A+98B+sin45C]
 * b.
 * If a number substring does not start the string but occurs later on in it,
 * the method returns an ArrayList object containing
 * (i.)the substring from index 0 up to, but not including the index where
 * the first number occurs.
 * (ii.)The number substring
 * (iii.)The remaining part of the string after the number.
 *
 * The method will not be used directly by developers but will be used as a tool
 * in math expression scanning.
 *
 *
 */

public ArrayList<String> splitStringAtFirstNumber(String val){
int firstOccDigit=getFirstIndexOfDigitOrPoint(val);//record the index where a digit or point first occurs.

if(firstOccDigit>-1){
//CASE 1: i.e a number starts the input string.
    if(firstOccDigit==0){
     String getFirstNumber=getFirstNumberSubstring(val);
     int firstNumberLen=getFirstNumber.length();
     scanner.add(getFirstNumber);
     scanner.add(val.substring(firstNumberLen));
    }
//CASE 2: a number does not start the input string.
    else if(firstOccDigit>0){

     String getFirstNumber=getFirstNumberSubstring(val);
     int firstNumberLen=getFirstNumber.length();
scanner.add(val.substring(0,firstOccDigit));
scanner.add(getFirstNumber);
scanner.add(val.substring(firstOccDigit+firstNumberLen));
    }

}
return scanner;
}

/**
 * method splitStringOnNumbers takes a String object and returns an ArrayList of substrings
 * in which the original string has been split into different parts based on the amount of
 * number substrings it has.
 * @param val the string to analyze
 * e.g if val="123.234+43.6",the output will be
 * [123.234,+,43.6]
 * @return An ArrayList of substrings consisting of number substrings and the other substrings
 * of the input.
 * @throws StringIndexOutOfBoundsException
 */

public ArrayList<String>splitStringOnNumbers(  String val ){

    ArrayList<String>scan=new ArrayList<String>();
    ArrayList<String>split=new ArrayList<String>();

boolean canSplit=getFirstIndexOfDigitOrPoint(val)>-1;
if(canSplit){
    int passes=0;
while(canSplit){
split.clear();
    split = splitStringAtFirstNumber(val);//the splitting
   int sizeAfterSplit=split.size();//the size after the splitting occurs

    val=split.get(sizeAfterSplit-1);//get the new string to split if it is still splittable on numbers.


     scan.addAll(split.subList(0, sizeAfterSplit-1));
    canSplit=getFirstIndexOfDigitOrPoint(val)>-1;


if(!canSplit){
    scan.add(val);
    break;
}


++passes;
}

}
else{
    scan.add(val);
}

    return scan;
 }

//





/**
 * method getNumberStrings takes a String object and returns an ArrayList of substrings
 * of all numbers in the input String
 * @param val the string to analyze
 * e.g if val="123.234+43.6",the output will be
 * [123.234,43.6]
 * @return An ArrayList of substrings consisting of number substrings and the other substrings
 * of the input.
 * @throws StringIndexOutOfBoundsException
 */

public  ArrayList<String>getNumberStrings( String val  ){
      ArrayList<String>split=new ArrayList<String>();
    int firstOcc=-1;
    int extent=0;//measures the length of each number substring

    int size=0;
    val=" "+val;
  while(firstOcc!=0){
firstOcc=getFirstIndexOfDigit(val);

if(firstOcc>0&&val.substring(firstOcc-1,firstOcc).equals(".")){
    firstOcc-=1;
}

    split.add(getFirstNumberSubstring(val));//store number token in split.
    //get the number of characters in the last element in split.
    size=split.get(split.size()-1).length();//get the size of the last element in the ArrayList
    extent = firstOcc+size;//index of the remaining part of the string ro be processed
    val=val.substring(extent);

    }
ArrayList<String> clearEmptyString=new ArrayList<String>();
clearEmptyString.add("");
  split.removeAll(clearEmptyString);
    return split;
}


/**
 * method getFirstOperatorInString takes a String object and returns the the first occurence
 * of an operator in the String.
 * @return the first occurence of an operator in the sequence.
 * @throws NullPointerException if no operator is found in the string
 */

public  String getFirstOperatorInString(  ){
    String opString="";


  for(int i=0;i<scannerInput.length();i++){
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+15))&&i+15<=scannerInput.length()){
    opString=scannerInput.substring(i,i+15); setFirstOpIndex(i); break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
 try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+14))&&i+14<=scannerInput.length()){
    opString=scannerInput.substring(i,i+14);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+13))&&i+13<=scannerInput.length()){
    opString=scannerInput.substring(i,i+13);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+12))&&i+12<=scannerInput.length()){
    opString=scannerInput.substring(i,i+12);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
 try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+11))&&i+11<=scannerInput.length()){
    opString=scannerInput.substring(i,i+11);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
  try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+10))&&i+10<=scannerInput.length()){
    opString=scannerInput.substring(i,i+10);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }

   try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+9))&&i+9<=scannerInput.length()){
    opString=scannerInput.substring(i,i+9);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }

   try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+8))&&i+8<=scannerInput.length()){
    opString=scannerInput.substring(i,i+8);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+7))&&i+7<=scannerInput.length()){
    opString=scannerInput.substring(i,i+7);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+6))&&i+6<=scannerInput.length()){
    opString=scannerInput.substring(i,i+6);setFirstOpIndex(i);   break;
   }
    }
        catch(IndexOutOfBoundsException ind){

 }
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+5))&&i+5<=scannerInput.length()){
    opString=scannerInput.substring(i,i+5);setFirstOpIndex(i);   break;
   }
      }//end try
      catch(IndexOutOfBoundsException ind){

 }
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+4))&&i+4<=scannerInput.length()){
    opString=scannerInput.substring(i,i+4);setFirstOpIndex(i);   break;
   }
      }//end try
             catch(IndexOutOfBoundsException ind){

 }//end catch
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+3))&&i+3<=scannerInput.length()){
    opString=scannerInput.substring(i,i+3);setFirstOpIndex(i);   break;
   }
       }//end try
             catch(IndexOutOfBoundsException ind){

 }//end catch
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+2))&&i+2<=scannerInput.length()){
    opString=scannerInput.substring(i,i+2);setFirstOpIndex(i);   break;
   }
    }//end try
             catch(IndexOutOfBoundsException ind){

 }//end catch
      try{
   if(MOperator.isOperatorString(scannerInput.substring(i,i+1))){
opString=scannerInput.substring(i,i+1);
    setFirstOpIndex(i);
    break;
   }
  }//end try
             catch(IndexOutOfBoundsException ind){

 }//end catch

  }

 return opString;
}//end method getFirstOperatorInString
/**
 *
 * @return the index of first occurence of the operator.
 */

public  int getFirstIndexOfOperator(){
    return firstOpIndex;
}
/**
 *
 * @return an ArrayList of all MOperator object names in the
 */
public  ArrayList<String> getOperatorStrings(){
  ArrayList<String>split=new ArrayList<String>();

while(scannerInput.length()>0){
    String op=getFirstOperatorInString();
    if(!op.equals("")){
    split.add(op);
    scannerInput=scannerInput.substring( scannerInput.indexOf(op)+op.length()       );

}//end if
    else if(op.equals("")){
        break;
    }
}//end loop
   return split;
}//end method getOperatorStrings






/**
 *
 * @return this string split on the operators.
 */
public  ArrayList<String> splitStringOnOperators( ){

   ArrayList<String>filter=new ArrayList<String>();
   filter.add("");
   filter.add(",");
int opIndex=2;
String op="<><./?>";
int pass=0;
while(!op.equals("")){
 op = getFirstOperatorInString();
 opIndex=getFirstIndexOfOperator();

if(opIndex==0){

    scanner.add(op);
    scannerInput=scannerInput.substring(opIndex+op.length());
}
else{
     scanner.add(scannerInput.substring(0,opIndex));
     scanner.add(op);
     scannerInput=scannerInput.substring(opIndex+op.length());
}
 op = getFirstOperatorInString();
 if(op.equals("")){
     scanner.add(scannerInput);
 }
++pass;



}//end while
//3AC-4+7sin5


//sort(5,3,2,1,-8,-9,12,34,98,-900,34,23,12,340)




for(int i=0;i<scanner.size();i++){//±–
    if(scanner.get(i).contains("±")){
        int index=scanner.get(i).indexOf("±");
     scanner.set(i, replace(scanner.get(i),"+", index, index+1));
    }
    else if(scanner.get(i).contains("–")){
              int index=scanner.get(i).indexOf("–");
     scanner.set(i, replace(scanner.get(i),"-", index, index+1));
    }
}//end if


scanner.removeAll(filter);
//23sin2-7cosh55-90+223-23sin2-7cosh55-90+223
   return scanner;
}
/**
 * Convenience form of splitStringOnOperators
 * that inserts the string to be split into
 * a List object and effects the split right inside it.
 * This is useful to the scanner method as it takes advantage
 * of ArrayList method subList to get Strings in locations inside it
 * and split them into multiple substrings of numbers and operators using
 * this method.
 * CAUTION!!!!! This method only allows the List to take contain one element
 * at the beginning and that is the String to be split into numbers and operators.
 * @param val the List object containing a single string value e.g 4+tan(12)
 * @return the the List object with the initial string split into
 * component number and operator parts.
 */

public  List<String> splitListOnVariables( List<String> val ){
    if(val.size()==1){
    val.addAll(splitStringOnVariables(val.get(0)));
    val.remove(0);

    }
   return val;
}













/**
 * Identifies that the input is a valid one by checking that all tokens are either MNumber objects, Variable objects
 * or MOperator objects. Then it registers any Variable object found and initializes it to zero.
 */
    public void validateTokens(){
       for(int i=0;i<scanner.size();i++){
        if(!isOperatorString(scanner.get(i))&&!isVariableString(scanner.get(i))
            && !validNumber(scanner.get(i))){
            setRunnable(false);
          errorList.add(scanner.get(i)+" is a strange math object!");
             setRunnable(false);
             scanner.clear();
             break;
        }//end if

    }//end for

    }//end validateTokens






/**
 * Utility method,more popularly used as a scanner into mathematical tokens of mathematical expressions.
 *
 * This method does not check expression variables against declared variables and is
 * only useful in breaking down the input into scanned tokens of numbers,variables and operators.
 *
 * The overloaded one takes a VariableManager object that models the store of
 * Variable objects in a computing application and checks if all variable names
 * entered by the user have been declared.
 *
 * @return an ArrayList containing a properly scanned version of the string such that all recognized number
 * substrings,variable substrings and operator substrings have been resolved.
 * To use this method as a scanner for math expressions all the programmer has to do
 * is to take care of the signs where need be(e.g in -2.873*5R+sinh34.2, the scanned view will be
 * [-,2.873,*,5,R,+,sinh,34.2]) To make sense of this, the programmer has to write minor code to concatenate
 * the - and the 2.873 and so on.
 *
 */
public ArrayList<String> scanner(){

    ArrayList<String>split=new ArrayList<String>();
    splitStringOnOperators();

removeCommas();

for(int i=0;i<scanner.size();i++){
    //The second boolean condition is added to avoid splitting exponential numbers because they contain E
    //the power of 10 operator
    split.clear();
        if( !MOperator.isOperatorString( scanner.get(i))&&!isNumber(scanner.get(i)) ){
            try{
         split=splitStringOnVariables(scanner.get(i));
         scanner.remove(i);
scanner.addAll(i,split);
       i+=split.size();
                  }//end if
        catch(IndexOutOfBoundsException indErr){
            break;
        }



       }//end if
        //

    }//end for



/*
 * Re-build the negative numbers in a statistical
 * data set. e.g 2,-3,4... is translated into 2, ~3, 4
 * At the end convert the ~3 back to -3
 */
for(int i=0;i<scanner.size();i++){
    if(scanner.get(i).substring(0,1).equals("~")){
        scanner.set(i, "-"+scanner.get(i).substring(1));
    }

}
/*
 * The for loop above does not properly handle
 * negative exponents of 10, e.g -3E-10
 * It splits it into -3,E, -10
 * So we need to manually couple the split objects together
 *
 */
for(int i=0;i<scanner.size();i++){
    try{
    if(  Number.isNegative(scanner.get(i)) && scanner.get(i+1).equals("E")&&
           Number.isNumber(scanner.get(i+2))){
        scanner.set(i, scanner.get(i)+scanner.get(i+1)+scanner.get(i+2));
        scanner.remove(i+1);
        scanner.remove(i+1);

    }
    }
    catch(IndexOutOfBoundsException indexErr){
        break;
    }
}


return scanner;
}

/**
 * Utility method,more popularly used as a scanner into mathematical tokens of mathematical expressions
 * @param varMan The store of all user variables already declared.
 * The scanner will not allow the user to use any variables he/she/it has not declared
 * @param matrixvarman The store of all user matrix variables.
 * The scanner references it here to ensure that nobidy uses a variable that
 * does not been declared in the store.
 * @return an ArrayList containing a properly scanned version of the string such that all recognized number
 * substrings,variable substrings and operator substrings have been resolved.
 * To use this method as a scanner for math expressions all the programmer has to do
 * is to take care of the signs where need be(e.g in -2.873*5R+sinh34.2, the scanned view will be
 * [-,2.873,*,5,R,+,sinh,34.2]) To make sense of this, the programmer has to write minor code to concatenate
 * the - and the 2.873 and so on.
 *
 */

public ArrayList<String> scanner(MatrixVariableManager matrixvarman,VariableManager varMan){
    ArrayList<String>split=new ArrayList<String>();//const:a,b,c=22E8,3E4,5
    splitStringOnOperators();
    //22A+3B+4C-96TU
removeCommas();
for(int i=0;i<scanner.size();i++){
    //The second boolean condition is added to avoid splitting exponential numbers because they contain E
    //the power of 10 operator
    split.clear();
        if( !MOperator.isOperatorString( scanner.get(i))&&!isNumber(scanner.get(i)) ){
            try{
         split=splitStringOnVariables(scanner.get(i));
         scanner.remove(i);
scanner.addAll(i,split);
       i+=split.size();
                  }//end if
        catch(IndexOutOfBoundsException indErr){
            break;
        }



       }//end if
        //

    }//end for



/*
 * Re-build the negative numbers in a statistical
 * data set. e.g 2,-3,4... is translated into 2, ~3, 4
 * At the end convert the ~3 back to -3
 */
for(int i=0;i<scanner.size();i++){
    if(scanner.get(i).substring(0,1).equals("~")){
        scanner.set(i, "-"+scanner.get(i).substring(1));
    }
}
/*
 * The for loop above does not properly handle
 * negative exponents of 10, e.g -3E-10
 * It splits it into -3,E, -10
 * So we need to manually couple the split objects together
 *
 */
for(int i=0;i<scanner.size();i++){
    try{
    if( Number.isNegative(scanner.get(i)) && scanner.get(i+1).equals("E")&&
             Number.isNumber(scanner.get(i+2))){
        scanner.set(i, scanner.get(i)+scanner.get(i+1)+scanner.get(i+2));
        scanner.remove(i+1);
        scanner.remove(i+1);

    }
    }
    catch(IndexOutOfBoundsException indexErr){
        break;
    }
}
//Build the matrixvariables:
for(int i=0;i<scanner.size();i++){
    try{
    if(scanner.get(i).equals("#")&&isVariableString(scanner.get(i+1))){
        scanner.set(i, scanner.get(i)+scanner.get(i+1));
        scanner.set(i+1,"");
    }
    }//end try
catch(IndexOutOfBoundsException indexErr){
    
}//end catch
}//end for loop

for(int i=0;i<scanner.size();i++){
    //disable use of variables that have not been afore created.
    if(Variable.isVariableString(scanner.get(i))&&!varMan.contains(scanner.get(i))){
        errorList.add(" Unknown Variable: "+scanner.get(i)+ "\n Please Declare And Initialize This Variable Before Using It.\n" +
                "Use The Command, \'var=val\' or\'vars:var1,var2...=val1,val2,... To Accomplish This.");
        setRunnable(false);
    }
    if(isMatrixVariableName(scanner.get(i))&&!matrixvarman.contains(scanner.get(i))){
        errorList.add(" Unknown Variable: "+scanner.get(i)+ "\n Please Declare And Initialize This Variable Before Using It.\n" +
                "Use The Command, \'var=val\' or\'vars:var1,var2...=val1,val2,... To Accomplish This.");
        setRunnable(false);
    }

    if(!isMatrixVariableName(scanner.get(i))&&!Variable.isVariableString(scanner.get(i))&&!MOperator.isOperatorString(scanner.get(i))&&!Number.isNumber(scanner.get(i))){
        errorList.add("Syntax Error! Strange Object Found: "+scanner.get(i));
        setRunnable(false);
    }
}

validateTokens();
if(!runnable){
    errorList.add("\n" +
            "Sorry, Errors Were Found In Your Expression." +
            "Please Consult The Help File For Valid Mathematical Syntax.");
    scanner.clear();
}
else{
    errorList.add("Scan SuccessFul.No Illegal Object Found.\n" +
            "Putting Scanner On StandBy");
}
return scanner;
}

 

}
