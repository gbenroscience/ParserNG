package parser;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import static java.lang.Math.*;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA.
 * This class has many methods. Method delete provides 2 constructors.The at takes 2 arguments,the string to manipulate
 * and the index to be deleted from the string.It returns the original string without the element at the
 * specified index.
 *
 * Method delete takes 3 arguments.The string,the index from which deletion is to begin and the index where deletion
 * ends.The returned string is the original string without the characters from the starting index
 * to one index behind the ending index.So the character at the ending index is not deleted.
 * So if we write   At( u,0,u.length() ),the whole string is deleted.
 *
 * Method isDouble1 takes a single argument;that is the number string which we wish to know if it is of type double or not.
 * It is a bold attempt by the author to devise a means of checking if a number entry qualifies as a double number string.
 * This checking is done at much higher speeds in comparison to method isDouble.
 */


public class STRING{
/**
 * contains allows you to verify if an input string contains a given substring.
 * @param str : the string to check
 * @param substr : the substring to check for
 * @return true or false depending on if or not the input consists of digits alone or otherwise.
 */
    public static boolean contains(String str,String substr){
    return (str.indexOf(substr)!=-1);
    }
/**
 * isOnlyDigits allows you to verify if an input string consists of just digits and the exponential symbol.
 * So the method will return true for 32000 but not for 3.2E4.It will also return true for 32E3.No decimal points are allowed
 * @param h ..The string to be analyzed.
 * @return true or false depending on if or not the input consists of digits alone or otherwise.
 */
public static boolean isOnlyDigits(String h){
boolean w=true;
for(int i=0;i<h.length();i++){
    if( !isDigit(h.substring(i, i+1)) ){
        w=false;
    }
}
return w;
}
/**
 * 
 * @param aChar The single character string to be examined
 * if it is a letter of the alphabet or not.
 * @return true if the parameter is a valid letter of the
 * English alphabet.
 */
public static boolean isLetter(String aChar){
    return Character.isLetter(aChar.toCharArray()[0]);
}
/**
 *
 * @param input The string from which we wish to remove all new line characters.
 * @return a string like the input but free of all new line characters.
 */
public static String removeNewLineChar( String input ){
String newString="";
    for(int i=0;i<input.length();i++){
        newString+=(!input.substring(i,i+1).equals("\n"))?input.substring(i,i+1):"";
    }//end for loop
return newString;
}//end method



//abcdabcde
//  234567

/**
 * isfullyDouble allows you to verify if an input string is in double number format or not.
 * @param num ..The string to be analyzed.
 * @return true or false depending on if or not the input represents a valid double number
 */
public static boolean isfullyDouble(String num){

boolean result=true;


String y=num;//store the original copy
int expCount=0;//counter for tracking the number of E symbols
int pointCount=0;//counter for tracking the number of floating point symbols
int minusCount=0;//counter for tracking the number of - operators
int plusCount=0;//counter for tracking the number of + operators
int minusPos=0;//get the index of the - symbol
int plusPos=0;//get the index of the + symbol
int expPos=0;//get the index of the E symbol
int pointPos=0;//get the index of the point symbol
int numCount=0;//detect if numbers have been encountered at all

int signCount=0;
boolean signed=false;
if(num.equals("")){//deal with empty strings
    result=false;
}
else if(lastElement(num).equals("E")||lastElement(num).equals(Operator.PLUS)||lastElement(num).equals(Operator.MINUS)){
    result=false;
}

else{
//sign+num1+pt+num2+exp+num3
for(int i=0;i<num.length();i++){

if(!isNumberComponent(num.substring(i,i+1))){
    result=false;
break;
}

if(isDigit(num.substring(i,i+1))){
   numCount=1;
    }
if(num.substring(i,i+1).equals("E")){
    //E cannot come before digits in the number
  if(numCount==0){
      result=false;
      break;
  }
    if(expCount==1){
        result=false;
        break;
    }
  expPos=i;
expCount++;
}
if(num.substring(i,i+1).equals(".")){
    //disallow decimal points after an exponent symbol has been parsed
         if(expCount==1){
             result=false;
            break;
        }
         //disallow more than 1 point from being parsed
        if(pointCount==1){
            result=false;
            break;
        }
         pointPos=i;
        pointCount++;

    }
    //if a + is encountered and no sign operator has been encountered then parse it accordingly
    if(num.substring(i,i+1).equals(Operator.PLUS)){
        signCount=plusCount+minusCount;
if(i==0){signed=true;}
//if you meet a plus operator anywhere between the 1 index and the index of the exp(E),the
//        number string is an invalid one
      if(i!=0&&expCount==0){
          result=false;
          break;
      }
 //if the + symbol is not the first char in the number then it must be the char that follows
        //the exp symbol E
if(i!=0&&i!=expPos+1){
    result=false;
    break;
}
  //signs (+|-) cannot be more than 2 if the number is unsigned
        if(!signed&&signCount==1){
            result=false;
            break;
        }
        //signs (+|-) cannot be more than 2 if the number is signed
        if(signed&&signCount==2){
            result=false;
            break;
        }
        plusPos=i;
        plusCount++;

    }
        //if a - is encountered and no sign operator has been encountered then parse it accordingly
    if(num.substring(i,i+1).equals("-")){
        signCount=plusCount+minusCount;
if(i==0){signed=true;}

 //if you meet a plus operator anywhere between the 1 index and the index of the exp(E),the
//        number string is an invalid one
      if(i!=0&&expCount==0){
          result=false;
          break;
      }
 //if the + symbol is not the first char in the number then it must be the char that follows
        //the exp symbol E
if(i!=0&&i!=expPos+1){
    result=false;
    break;
}
  //signs (+|-) cannot be more than 2 if the number is unsigned
        if(!signed&&signCount==1){
            result=false;
            break;
        }
        //signs (+|-) cannot be more than 2 if the number is signed
        if(signed&&signCount==2){
            result=false;
            break;
        }
        minusPos=i;
        minusCount++;
    }


}//end for


}//end else

return result;
}


/**
 * method firstElement takes one argument and that is the string to be modified.
 *
 * @param u the string to be modified
 * @return the first element of the string
 * @throws StringIndexOutOfBoundsException
 */
public  static String firstElement(String u){
return u.substring(0,1);
    }
/**
 * method lastElement takes one argument and that is the string to be modified.
 * @param u the string to be modified
 * @return the last element of the string
 * @throws StringIndexOutOfBoundsException
 */
public  static String lastElement(String u){
return u.substring(u.length()-1,u.length());
    }
/**
 * nextIndexOf allows you to get the next occurence of a substring searching
 * from a point in the string forwards
 * @param str ..The String being analyzed.
 * @param index ..The starting point of the search.
 * @param str2 ..The string whose next occurence we desire.
 * @return the next index of the str2 if found or -1  if not found
 */
public static int nextIndexOf(String str,int index,String str2){
    int result=-1;//contains the result of the search
    int L=str.length();
    int L1=str2.length();

if(index>=L){
    throw new StringIndexOutOfBoundsException("Searching forward from the end of this string is impossible");
}
else if(L1>(L-(index+1))){
    throw new StringIndexOutOfBoundsException("Searching forward from this point in the string is impossible");

}
else if(index<0){
    throw new StringIndexOutOfBoundsException("Specify a positive value for the search index");
}
else{
//i=index+1 so that the search can start from the next element after the search's starting position
    str=str.substring(index+1);
    result = str.indexOf(str2);

    if(result!=-1){
        result+=index+1;
    }

}//end else

    return result;
}

/**
 * prevIndexOf allows you to get the previous occurence of a substring searching
 * from a point in the string backwards
 * @param str ..The String being analyzed.
 * @param index ..The starting point of the search.
 * @param str2 ..The string whose previous occurence we desire.
 * @return the previous index of the str2 if found or -1  if not found
 */
public static int prevIndexOf(String str,int index,String str2){
    int result=-1;//contains the result of the search
    int L=str.length();
    int L1=str2.length();

if(index>=str.length()){
    throw new StringIndexOutOfBoundsException("Searching backwards from this point of the string is impossible");
}
else if(L1>index){
    throw new StringIndexOutOfBoundsException("Searched String Must Have More Characters Than Search String");

}
else if(index<=0){
    throw new StringIndexOutOfBoundsException("Specify a positive value greater than 0 for the search index");
}


else{
    str=str.substring(0,index);
    result = str.lastIndexOf(str2);



}

    return result;
}
/**
 *
 * @param val the entry being analyzed.
 * @return true if the entry is a + sign or a minus sign
 */
public static boolean isSign(String val){
    return val.equals(Operator.PLUS)||val.equals(Operator.MINUS);
}
/**
 *
 * @param val the entry being analyzed.
 * @return true if the entry is a floating point
 */
public static boolean  isDecimalPoint(String val){
    return val.equals(".");
}
/**
 *
 * @param val the entry being analyzed.
 * @return true if the entry is the power of 10 symbol
 * i.e E
 */
public static boolean  isExponentOf10(String val){
    return val.equals("E");
}
/**
 * method getFirstOccurenceOfDigit takes a String object and returns the first occurence
 * of a number character in the String.
 * @param value the string to analyze
 * @return the first occurence of a number character in the sequence.
 * @throws StringIndexOutOfBoundsException
 */
public  static String getFirstOccurenceOfDigit(String value) throws StringIndexOutOfBoundsException{
String firstnum="";
for(int i=0; i<value.length(); i++){
    if( isDigit(value.substring(i,i+1)) ){
        firstnum=value.substring(i,i+1);
        break;
    }
}
return firstnum;
    }
/**
 * method getFirstIndexOfDigit takes a String object and returns the index of the first occurence
 * of a number character in the String.
 * @param val the string to analyze
 * @return the index of the first occurence of a number character in the sequence.
 * @throws StringIndexOutOfBoundsException
 */
public  static int getFirstIndexOfDigit(String val) throws StringIndexOutOfBoundsException{
int i;
    for( i=0;i<val.length();i++){
    if( isDigit(val.substring(i,i+1)) ){
          return i;
    }
}
    return -1;
    }

/**
 * method getFirstIndexOfDigit takes a String object and returns the index of the first occurence
 * of a number character in the String.
 * @param val the string to analyze
 * @return the index of the first occurence of a number character in the sequence.
 * @throws StringIndexOutOfBoundsException
 */
public  static int getFirstIndexOfDigitOrPoint(String val) throws StringIndexOutOfBoundsException{
    int i;
for( i=0;i<val.length();i++){
    if(isDigit(val.substring(i,i+1))||val.substring(i,i+1).equals(".")){
     return i;
    }
}
    return -1;
    }
/**
 * Counts how many times a given single character
 * string occurs in a given String object.
 * @param c A single character String value whose occurences in another string we wish to count.
 * @param str The String object that we will search through.
 * @return The number of times c occurs in str
 */
public static int countOccurences(String c,String str){
if(c.length()==1&&str.length()>=1){
    int count = 0;
    for(int i=0;i<str.length();i++){
     if(c.equals(str.substring(i, i+1))){
         ++count;
     }//end if
    }//end for loop
    return count;
}//end if
else{
    throw new StringIndexOutOfBoundsException(str+" \'s length must be 1 or greater. "+c+" \'s length must be 1.");
}//end else


}//end method

/**
 * method getFirstNumberSubstring takes a String object and returns the first
 * number string in the String.
 * @param val the string to analyze
 * @return the index of the first occurrence of a number character in the sequence.
 * @throws StringIndexOutOfBoundsException
 */
public  static String getFirstNumberSubstring(String val) throws StringIndexOutOfBoundsException{



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
for(int i = ind;i<val.length();i++){
      if(!isNumberComponent(val.substring(i,i+1))){
        break;
    }
      //a number cannot contain more than 1 decimal point.
     if(val.substring(i,i+1).equals(".")){
         break;
     }

       //if a + is encountered and no sign operator has been encountered then parse it accordingly
    if(val.substring(i,i+1).equals(Operator.PLUS)){
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
if(val.substring(i,i+1).equals(Operator.PLUS)&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals("-")&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}


if(isDigit(val.substring(i,i+1))&&( (pointCount==0&&expCount==1) ||( pointCount==1&&expCount==1 ) ) ){
num3+=val.substring(i,i+1);
}




}//end for
    }//end if
//num=a+p+b+e+s+c
    else if(num2.equals("")){
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
    if(val.substring(i,i+1).equals(Operator.PLUS)){
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
if(val.substring(i,i+1).equals(Operator.PLUS)&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}
if(val.substring(i,i+1).equals("-")&&expCount==1&&num3.equals("")&&sign.equals("")){
sign+=val.substring(i,i+1);
}




if(isDigit(val.substring(i,i+1))&&( (pointCount==0&&expCount==1) ||( pointCount==1&&expCount==1 ) ) ){
num3+=val.substring(i,i+1);
}


}//end for
    }//end else if
//klnsdkn23.94E2jwjji
    //klwejh.7856E++43
    String result=num1+num2+point+exp+sign+num3;
try{
if(lastElement(result).equals(Operator.PLUS)||lastElement(result).equals("-")){
     result=result.substring(0, result.length()-1);
}
if(lastElement(result).equals(Operator.PLUS)||lastElement(result).equals("-")){
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
 * @return true if the input string ends with
 * a mathematical operator string name

public static double endsWithKeyWord( String val  ){





}//end method endsWithKeyWord
*/










/**
 * Allows you to remove commas from a string.
 * @param h
 * @return a string in which the contents of
 * the original have been freed from all
 * commas.
 */
public static String removeCommas(String h){
    String y="";
 if(h instanceof String){
    for(int i=0;i<h.length();i++){
      if(!h.substring(i,i+1).equals(",")){
          y+=h.substring(i,i+1);
      }
 }
 }
    return y;
}





/**
 * purifier allows you to remove all white spaces in a given string,which it receives as its argument
 * @param h The string to free from white space.
 * @return a string devoid of all white space.
 */
public static String purifier(String h){
    StringBuilder b = new StringBuilder();
    for(int i=0;i<h.length();i++){
      if(!h.substring(i,i+1).equals("")&&!h.substring(i,i+1).equals(" ")&&!h.substring(i,i+1).equals("\n")){
          b.append(h.charAt(i)); 
      }
 }
    return b.toString();
}


/**
 * purifier allows you to remove all white spaces in a given string,which it receives as its argument
 * @param h The string to remove white space from.
 * @return a string from which white space has been removed starting
 * from index fromIndex.
 */
public static String purifier(String h , int fromIndex){
    String begin = h.substring(0,fromIndex);
    String end = h.substring(fromIndex);
    return begin+purifier(end);
}

/**
 * removeAll allows you to remove all occurrences of an unwanted string from a given string,
 * which it receives as its argument
 * @param str  The string to remove the unwanted substring from.
 * @param unwanted The unwanted substring.
 * @return a string from which all instance of unwanted has been removed.
 */
public static String removeAll(String str , String unwanted) throws IndexOutOfBoundsException{
if( unwanted.length() <= str.length() ){
  String[]splitted = splits(str, unwanted);
String acc = "";
for( int i = 0; i < splitted.length; i++ ){
    acc+=splitted[i];
}

return acc;
}//end if
throw new IndexOutOfBoundsException("Substring to be removed cannot be longer than the String "
        +(unwanted.length() > str.length() ) );
}

/**
 * The Technology:
 *
 * 1. Split the string into an array on the replaced string.
 * For example, if the string is Shohkhohlhohkhohbanhghohshoh,
 * and the replaced string is hoh and the replacing string is now.
 *
 * Stage 1 generates: [S,k,l,k,banhg,s,]
 *
 * 2. Create a string object and loop through the
 * array, appending to this object in each index,
 * the contents of that index and the replacing string
 *
 *
 *
 *
 *
 * replaceAll allows you to replace all occurrences of an unwanted string in a given string
 * with a new string,
 * which it receives as its argument
 * @param str  The string on which replacement is to be done.
 * @param replaced The replaced string.
 * @param  replacing The replacing string.
 * @return a string from which all instance of unwanted has been removed.
 */
public static String replaceAll(String str , String replaced,String replacing){
if( replaced.length() <= str.length() ){

int len = str.length();
int replacedLen = replaced.length();
    /**
 * Check if the string ends with the splitting  string.
 * If so checkEnd is true
 * If so, then the splitting array will contain an empty string in its last index.
 * So do away with the ending substring.
 *
 * Next check if it starts with the splitting string.
 * If so checkStart is true
 * If so, then the splitting array will contain an empty string in its first index.
 *
 */
    boolean checkEnd = str.substring(len-replacedLen).equals(replaced);
    boolean checkStart = str.substring(0,replacedLen).equals(replaced);

if(checkEnd){
    str = str.substring(0,len-replacedLen);
    str += replacing;
}
if(checkStart){
    str = str.substring(replacedLen);
    str = replacing+str;
}

  String[]splitted = splits(str, replaced);
String acc = "";
int length = splitted.length;

for( int i = 0; i < length; i++ ){
    if( i == length - 1 ){
       acc+=splitted[i];
    }
 else{
       acc+=(splitted[i]+replacing);
 }
}//end for

return acc;
}//end if


throw new IndexOutOfBoundsException("Substring to be removed cannot be longer than the String "
        +(replaced.length() > str.length() ) );
}






/**
 * method delete takes 2 arguments,the string to be modified and the index of the character to be deleted
 *
 * @param u
 * @param i
 * @return the string after the character at index i has been removed
 * @throws StringIndexOutOfBoundsException
 */
public  static String delete(String u,int i) throws StringIndexOutOfBoundsException{
String t="";
String t1="";
 try{
t=u.substring(0,i);
t1=u.substring(i+1);
}
  catch(StringIndexOutOfBoundsException e){
       System.err.print("A stringindexoutofbounds exception has occured.");
   }
  return t+t1;
    }

/**
 * method delete takes 2 arguments,the string to be modified and the substring to be removed from it.
 *
 * @param u the string to be modified
 * @param str the substring to be removed from u
 * @return the string after the first instance of the specified substring has been removed
 * @throws StringIndexOutOfBoundsException
 */
public  static String delete(String u,String str){
    if(contains(u, str)){
try{
  int ind1=u.indexOf(str.substring(0,1));
  int ind2=u.indexOf(str.substring(str.length()-1,str.length()));
  u=delete(u, ind1,ind2+1);
}
catch(StringIndexOutOfBoundsException inder){u+="";}

    }

  return  u;
    }




/**
 * method delete takes 3 arguments,the string to be modified, the index at which deleting is to begin and the index one after the point where
 * deletion is to end. STRING.delete("Corinthian", 2,4)returns "Conthian"
 * @param u
 * @param i
 * @param j
 * @return
 * @throws StringIndexOutOfBoundsException
 */
public  static String delete(String u, int i,int j) throws StringIndexOutOfBoundsException{
 String t="";
 String t1="";
try{
    t=u.substring(0,i);
    t1=u.substring(j);
    t+=t1;
}
catch(StringIndexOutOfBoundsException e){
       t+="";
   }

return t;
    }

/**
 * method delete takes 3 arguments,the string to be modified, the index at which deleting is to begin and the index one after the point where
 * deletion is to end. STRING.delete("Corinthian", 2,4)returns "Conthian"
 * @param str..The original string
 * @param i..The index at which we start deleting
 * @param num The number of characters to delete after the index.
 * @return the resulting string after the chars between i and i+num-1
 * have been deleted.
 * @throws StringIndexOutOfBoundsException
 */
public  static String deleteCharsAfter(String str, int i,int num) throws StringIndexOutOfBoundsException{
 String t="";
 String t1="";
try{
    t=str.substring(0,i);
    t1=str.substring(i+num);//abcdefghijklmn del(this, 3,5);abcijklmn
    t+=t1;
}
catch(StringIndexOutOfBoundsException e){
       t+="";
   }

return t;
    }


      /**
       * boolean Method isDouble(Object a) detects if a string entry is in double number format or not.
       *
       * @param a
       * @return
       * @throws NumberFormatException
       */
 public static boolean isDouble(String a) throws NumberFormatException{
boolean truth=false;
 try{
   double v=
           Double.valueOf(a);
   truth=true;
    }
    catch(NumberFormatException num){truth=false; }
return truth;
    }

      /**
       * boolean Method isNumberComponent(args) detects if a string entry is a buiding block for a valid number or not
       * For our purposes,the building blocks are the digits,the "." symbol,the E symbol,the - symbol,but not the addition one.
       * @param comp the String object to check
       * @return true if the analyzed object is a valid component of a real number
       */
 public static boolean isNumberComponent(String comp){
    boolean truth=false;
    if(comp.length()==1&&(
            STRING.isDigit(comp)||comp.equals(Operator.MINUS)||comp.equals(".")||comp.equals("E")||comp.equals(Operator.PLUS)) ){
            truth=true;
        }


     return truth;

    }



/**
 * This is an optimized specialized substitute for String.split in
 * the Java API.
 * It runs about 10-20 times or more faster than the split method in the Java API and it returns
 * an ArrayList of values, instead of an array.
 *
 * Splits a string on all instances
 * of the splitting object
 * @param splittingObject The substring on which the string is to be split.
 * @param stringTosplit The string to be split.
 * @return an array containing substrings that the string has
 * been split into.
 */
public static ArrayList<String> split(String stringTosplit,String splittingObject) throws ArrayIndexOutOfBoundsException{
    ArrayList<String>split= new ArrayList<String>();
    String[]splitted = splits(stringTosplit, splittingObject);
        split.addAll(Arrays.asList(splitted));

    return split;
}//end method




/**
 * This is an highly optimized specialized substitute for String.split in
 * the Java API.
 * It runs about 40-100 times faster than the split method in the Java API and it returns an array, too.
 *
 * Splits a string on all instances
 * of the splitting object
 * @param splittingObject The substring on which the string is to be split.
 * @param stringTosplit The string to be split.
 * @return an array containing substrings that the string has
 * been split into.
 */
public static String[] splits(String stringTosplit,String splittingObject){

    
 if(splittingObject.equals("")){
     String[]chars=new String[stringTosplit.length()];
     for(int i=0;i<chars.length;i++){
         chars[i]=stringTosplit.substring(i, i+1);
     }
     return chars;
 }//end if   
    
    
int len = stringTosplit.length();
int replacedLen = splittingObject.length();
    /**
 * Check if the string ends with the splitting  string.
 * If so checkEnd is true
 * If so, then the splitting array will contain an empty string in its last index.
 * So do away with the ending substring.
 *
 * Next check if it starts with the splitting string.
 * If so checkStart is true
 * If so, then the splitting array will contain an empty string in its first index.
 *
 */
    boolean checkEnd = stringTosplit.substring(len-replacedLen).equals(splittingObject);
    boolean checkStart = stringTosplit.substring(0,replacedLen).equals(splittingObject);

if(checkEnd){
    stringTosplit = stringTosplit.substring(0,len-replacedLen);
}
if(checkStart){
    stringTosplit = stringTosplit.substring(replacedLen);
}

    int ind = stringTosplit.indexOf(splittingObject);


    if(splittingObject.length()<stringTosplit.length()&&ind!=-1){
String[] split = new String[stringTosplit.length()];
int subLen =splittingObject.length();
String check="";
int lastIndex=0;
int index=0;
int length = stringTosplit.length();
//int whitespacecount = 0;//counts the number of white spaces stored.
    for(int j = 0; j<length; j++){
        try{
check = stringTosplit.substring(j, j+subLen);

if(check.equals(splittingObject)){
    split[index] = stringTosplit.substring(lastIndex, j);
    lastIndex = j+subLen;
    index++;
    //detect the end of the splitting action and store the last
    //item.
    if(stringTosplit.substring(j+subLen).indexOf(splittingObject)==-1){
        split[index] = stringTosplit.substring(j+subLen);
        index++;
        break;
    }
}//end if
        }//end try
        catch( IndexOutOfBoundsException inds){
            System.out.println( "tot_len = "+stringTosplit.length()+", subLen = "+subLen+", j = "+j );
        }
    }//end for
String[]splitted=new String[index];
            System.arraycopy(split, 0, splitted, 0, splitted.length);
return splitted;
    }//end if
    else if(ind==-1){
return new String[]{stringTosplit};
    }
    else{
        throw new ArrayIndexOutOfBoundsException(stringTosplit+" must have more characters than "+splittingObject);
    }
}//end method










/**
 * This is an highly optimized specialized substitute for String.split in
 * the Java API.
 * It runs about 5-10 times faster than the split method in the Java API and it returns an Array object, too.
 * If the array of splitting objects contains nothing or contains elements that
 * are not found in the string, then an Array object containing
 * the original string is returned. But if the array of splitting objects contains an empty string,
 * it chops or slices the string into pieces, and so returns an Array object
 * containing the individual characters of the string.
 *
 *
 * Splits a string on all instances
 * of the splitting object
 * @param stringTosplit The string to be split.
 * @param splittingObjects An array containing substrings on which the string should be split.
 * @return an array containing substrings that the string has
 * been split into.
 * @throws NullPointerException if the array of splitting objects contains
 * a null value or is null.
 */
public static String[] splits(String stringTosplit,String[] splittingObjects) throws NullPointerException{

ArrayList<String>split = new ArrayList<String>();
split.add(stringTosplit);
for( int j=0;j<splittingObjects.length;j++){
for( int i=0;i<split.size();i++){
    ArrayList<String>sub = split(split.get(i), splittingObjects[j] );
    split.remove(i);
    split.addAll( i , sub );
    i+=sub.size()-1;
}//end for
}//end for
/**
 * Remove any indiscriminate white spaces produced.
 */
    for(int i=0;i<split.size();i++){
      if(split.get(i).equals("")){
          split.remove(i);i--;
      }
    }
return split.toArray(new String[]{});
}//end method








/**boolean Method isDigit can be used to check for single digits  i.e 0-9.
 *
 * @param a the String to check.
 * @return true if the string is a valid digit.
 */


public static boolean isDigit(String a) {
return ((a.equals("0")||a.equals("1")||a.equals("2")||a.equals("3")||
     a.equals("4")||a.equals("5")||a.equals("6")||a.equals("7")||a.equals("8")||a.equals("9") ) );
}
/**


*/
/**
* Method isDouble1 is specially built for parsing number strings in my parser.Please do not use it for your own purposes.
 * Its purpose is to detect already scanned number strings in a List object that contains many kinds of string objects
 * @param a
 * @return true if the string represents a number string( this in reality is true if the number has already being scanned and proven to be a valid number)
 */

public static boolean isDouble1(String a){

 int L=a.length();
 boolean truth=false;
 int s=1/L;
 if(L>3&&a.substring(0,1).equals("E") ){
     truth= false;
}
 else if(L>3&&STRING.isDigit(a.substring(0,1) ) ){
    truth= true;
}
 else if(L>3&&a.substring(0,1).equals(".") ){
     truth= true;
}
else if(L>3&&a.substring(0,1).equals("-") ){
     truth= true;
}
    if(L==3&&a.substring(0,1).equals("E") ){
     truth= false;
}
 else if(L==3&&a.substring(0,1).equals(".") ){
     truth= true;
}
  else if(L==3&&a.substring(0,1).equals("-") ){
     truth= true;
}
else if(L==3&&STRING.isDigit(a.substring(0,1) ) ){
     truth= true;
}
 if(L==2&&STRING.isDigit(a.substring(0,1) )){
     truth= true;
 }
 else if(L==2&&a.substring(0,1).equals(".") ){
     truth= true;
 }
else if(L==2&&a.substring(1,2).equals("-")  ){
     truth= true;

}
else if(L==2&&a.substring(0,1).equals("E") ){
     truth= false;
}

 else if(L==1&&a.substring(0,1).equals("-")){
     truth= false;
 }
 else if(L==1&&STRING.isDigit(a.substring(0,1) ) ){
     truth= true;
 }
  else if(L==1&&a.substring(0,1).equals(".")){
     truth= false;
 }
else if(L==1&&a.substring(0,1).equals("E")){
     truth= false;
 }
else{
     truth= false;
}





return truth;
}




public static boolean isWord(String a) {
boolean truth=false;
  String q="";
  for(int i=0; i<a.length(); i++){
        if(a.substring(i,i+1).equalsIgnoreCase("A")||a.substring(i,i+1).equalsIgnoreCase("B")||a.substring(i,i+1).equalsIgnoreCase("C")||
           a.substring(i,i+1).equalsIgnoreCase("D")||a.substring(i,i+1).equalsIgnoreCase("E")||a.substring(i,i+1).equalsIgnoreCase("F")||
           a.substring(i,i+1).equalsIgnoreCase("G")||a.substring(i,i+1).equalsIgnoreCase("H")||a.substring(i,i+1).equalsIgnoreCase("I")||
           a.substring(i,i+1).equalsIgnoreCase("J")||a.substring(i,i+1).equalsIgnoreCase("K")||a.substring(i,i+1).equalsIgnoreCase("L")||
           a.substring(i,i+1).equalsIgnoreCase("M")||a.substring(i,i+1).equalsIgnoreCase("N")||a.substring(i,i+1).equalsIgnoreCase("O")||
           a.substring(i,i+1).equalsIgnoreCase("P")||a.substring(i,i+1).equalsIgnoreCase("Q")||a.substring(i,i+1).equalsIgnoreCase("R")||
           a.substring(i,i+1).equalsIgnoreCase("S")||a.substring(i,i+1).equalsIgnoreCase("T")||a.substring(i,i+1).equalsIgnoreCase("U")||
           a.substring(i,i+1).equalsIgnoreCase("V")||a.substring(i,i+1).equalsIgnoreCase("W")||a.substring(i,i+1).equalsIgnoreCase("X")||
           a.substring(i,i+1).equalsIgnoreCase("Y")|| a.substring(i,i+1).equalsIgnoreCase("Z") ){
            q+=a.substring(i,i+1);
        }
  }

   if(a.equals("")){
      truth=false;
  }

  else if(q.equals(a ) ){
truth=true;
  }
  else if(!q.equals(a )  ){
     truth=false;
  }
 return truth;
}


/**
 * method isLowerCaseWord takes one argument and that is the string
 * whose case we wish to know.
 * @param a The string to manipulate.
 * @return  true if the string is a lowercase letter
 */

public static boolean isLowerCaseWord(String a) {
  String q="";
boolean truth=false;

  for(int i=0; i<a.length(); i++){
        if(a.substring(i,i+1).equals("a")||a.substring(i,i+1).equals("b")||a.substring(i,i+1).equals("c")||
           a.substring(i,i+1).equals("d")||a.substring(i,i+1).equals("e")||a.substring(i,i+1).equals("f")||
           a.substring(i,i+1).equals("g")||a.substring(i,i+1).equals("h")||a.substring(i,i+1).equals("i")||
           a.substring(i,i+1).equals("j")||a.substring(i,i+1).equals("k")||a.substring(i,i+1).equals("l")||
           a.substring(i,i+1).equals("m")||a.substring(i,i+1).equals("n")||a.substring(i,i+1).equals("o")||
           a.substring(i,i+1).equals("p")||a.substring(i,i+1).equals("q")||a.substring(i,i+1).equals("r")||
           a.substring(i,i+1).equals("s")||a.substring(i,i+1).equals("t")||a.substring(i,i+1).equals("u")||
           a.substring(i,i+1).equals("v")||a.substring(i,i+1).equals("w")||a.substring(i,i+1).equals("x")||
           a.substring(i,i+1).equals("y")||a.substring(i,i+1).equals("z") ){
            q+=a.substring(i,i+1);
        }
  }

   if(a.equals("")){
   truth=false;
  }

  else if(q.equals(a ) ){
      truth=true;
  }
  else if(!q.equals(a )  ){
    truth=false;
  }
return truth;
}
/**
 * method isUpperCaseWord takes one argument and that is the string
 * whose case we wish to know.
 * @param a The string to manipulate.
 * @return  true if the string is an uppercase letter
 */
public static boolean isUpperCaseWord(String a) {
  String q="";
  boolean truth=false;
for(int i=0; i<a.length(); i++){
        if(a.substring(i,i+1).equals("A")||a.substring(i,i+1).equals("B")||a.substring(i,i+1).equals("C")||
           a.substring(i,i+1).equals("D")||a.substring(i,i+1).equals("E")||a.substring(i,i+1).equals("F")||
           a.substring(i,i+1).equals("G")||a.substring(i,i+1).equals("H")||a.substring(i,i+1).equals("I")||
           a.substring(i,i+1).equals("J")||a.substring(i,i+1).equals("K")||a.substring(i,i+1).equals("L")||
           a.substring(i,i+1).equals("M")||a.substring(i,i+1).equals("N")||a.substring(i,i+1).equals("O")||
           a.substring(i,i+1).equals("P")||a.substring(i,i+1).equals("Q")||a.substring(i,i+1).equals("R")||
           a.substring(i,i+1).equals("S")||a.substring(i,i+1).equals("T")||a.substring(i,i+1).equals("U")||
           a.substring(i,i+1).equals("V")||a.substring(i,i+1).equals("W")||a.substring(i,i+1).equals("X")||
           a.substring(i,i+1).equals("Y")|| a.substring(i,i+1).equals("Z") ){
            q+=a.substring(i,i+1);
        }
  }

   if(a.equals("")){
     truth=false;
  }

  else if(q.equals(a ) ){
     truth=true;
  }
  else if(!q.equals(a )  ){
      truth=false;
  }

 return truth;
}
/**
 * method toUpperCase takes one argument only and that is the input string to be converted to upper case
 * @param m the single length string to be converted to upper case
 * @return   the upper case version of the input
 * @throws StringIndexOutOfBoundsException
 */
public static String toUpperCase(String m){

    if(m.length()<1||m.length()>1){
        throw new StringIndexOutOfBoundsException();
    }
    if(!STRING.isWord(m)){
        throw new InputMismatchException();
    }
    if(m.equals("a")){
        m="A";
    }
    else if(m.equals("b")){
        m="B";
    }
     else if(m.equals("c")){
        m="C";
     }
    else if(m.equals("d")){
        m="D";
    }
    else if(m.equals("e")){
        m="E";
    }
    else if(m.equals("f")){
        m="F";
    }
    else if(m.equals("g")){
        m="G";
    }
    else if(m.equals("h")){
        m="H";
    }
    else if(m.equals("i")){
        m="I";
    }
    else if(m.equals("j")){
        m="J";
    }
    else if(m.equals("k")){
        m="K";
    }
    else if(m.equals("l")){
        m="L";
    }
    else if(m.equals("m")){
        m="M";
    }
    else if(m.equals("n")){
        m="N";
    }
    else if(m.equals("o")){
        m="O";
    }
    else if(m.equals("p")){
        m="P";
    }
    else if(m.equals("q")){
        m="Q";
    }
    else if(m.equals("r")){
        m="R";
    }
    else if(m.equals("s")){
        m="S";
    }
    else if(m.equals("t")){
        m="T";
    }
    else if(m.equals("u")){
        m="U";
    }
    else if(m.equals("v")){
        m="V";
    }
    else if(m.equals("w")){
        m="W";
    }
    else if(m.equals("x")){
        m="X";
    }
    else if(m.equals("y")){
        m="Y";
    }
    else if(m.equals("z")){
        m="Z";
    }
    else if(STRING.isUpperCaseWord(m)){
       m+="";
    }
return m;
}

/**
 * method toLowerCase takes one argument only and that is input string to be converted to lower case
 * @param m the single length string to be converted to lower case
 * @return   the lower case version of the input
 * @throws StringIndexOutOfBoundsException
 */
public static String toLowerCase(String m){

    if(m.length()<1||m.length()>1){
        throw new StringIndexOutOfBoundsException();
    }
    if(!STRING.isWord(m)){
        throw new InputMismatchException();
    }
    if(m.equals("A")){
        m="a";
    }
    else if(m.equals("B")){
        m="b";
    }
     else if(m.equals("C")){
        m="c";
     }
    else if(m.equals("D")){
        m="d";
    }
    else if(m.equals("E")){
        m="e";
    }
    else if(m.equals("F")){
        m="f";
    }
    else if(m.equals("G")){
        m="g";
    }
    else if(m.equals("H")){
        m="h";
    }
    else if(m.equals("I")){
        m="i";
    }
    else if(m.equals("J")){
        m="j";
    }
    else if(m.equals("K")){
        m="k";
    }
    else if(m.equals("L")){
        m="l";
    }
    else if(m.equals("M")){
        m="m";
    }
    else if(m.equals("N")){
        m="n";
    }
    else if(m.equals("O")){
        m="o";
    }
    else if(m.equals("P")){
        m="p";
    }
    else if(m.equals("Q")){
        m="q";
    }
    else if(m.equals("R")){
        m="r";
    }
    else if(m.equals("S")){
        m="s";
    }
    else if(m.equals("T")){
        m="t";
    }
    else if(m.equals("U")){
        m="u";
    }
    else if(m.equals("V")){
        m="v";
    }
    else if(m.equals("W")){
        m="w";
    }
    else if(m.equals("X")){
        m="x";
    }
    else if(m.equals("Y")){
        m="y";
    }
    else if(m.equals("Z")){
        m="z";
    }
    else if(STRING.isLowerCaseWord(m)){
       m+="";
    }

return m;
}

/**
 * method reverse takes 1 argument and that is the string to be reversed
 * @param u1 The string to be reversed.
 * @return  The reversed string.
 */
 public static String reverse( String u1 ){
String v1=""; for(int i=0;i<u1.length(); i++){ v1=u1.substring(i, i+1)+v1;  }  return v1; }
/**
 * method replace takes 4 arguments,the string to be modified, the replacing string, the index at which replacement is to begin and the index where
 * replacement ends plus one i.e replacement ends at index j-1. STRING.replace("Corinthian","The" 3,6)returns "CorThehians"
 * @param u The string to manipulate.
 * @param u1 The replacing string
 * @param i Index at which to begin replacing
 * @param j Index at which replacement ends plus one
 * @return  A string in which the characters between index i and j-1 have been replaced with the replacing string
 * @throws StringIndexOutOfBoundsException
 */
public static String replace(String u, String u1,Integer i,Integer j )throws StringIndexOutOfBoundsException{
u=u.substring(0,i)+u1+u.substring(j);
return u;
 }

   public static boolean isEven(String a) {
   double ir=a.length();
 String h=String.valueOf(0.5*ir);
 if(h.substring(h.length()-1,h.length()).equals("0")){  return true;  }
 else if(!h.substring(h.length()-1,h.length()).equals("0")){  return false;  }


  return true;
}
private static double convertdoubleDigitToString(String num){
       double ans = -1;
       if(num.equals("0")){
         ans=0;
       }
       else if(num.equals("1")){
           ans=0;
       }
       else if(num.equals("2")){
           ans=3;
       }
       else if(num.equals("3")){
           ans=4;
       }
       else if(num.equals("4")){
           ans=4;
       }
       else if(num.equals("5")){
           ans=5;
       }
       else if(num.equals("6")){
           ans=6;
       }
       else if(num.equals("7")){
           ans=7;
       }
       else if(num.equals("8")){
           ans=8;
       }
       else if(num.equals("9")){
           ans=9;
       }
       else{
           throw new NoSuchElementException("Only 0 through 9 can be converted");
       }
       return ans;
   }
   private static String convertdoubleDigitToString(double a){
       String ans="";
       if(a==0){
         ans="0";
       }
       else if(a==1){
           ans="1";
       }
       else if(a==2){
           ans="2";
       }
       else if(a==3){
           ans="3";
       }
       else if(a==4){
           ans="4";
       }
       else if(a==5){
           ans="5";
       }
       else if(a==6){
           ans="6";
       }
       else if(a==7){
           ans="7";
       }
       else if(a==8){
           ans="8";
       }
       else if(a==9){
           ans="9";
       }
       else{
           throw new NoSuchElementException("Only 0 through 9 can be converted");
       }
       return ans;
   }
 //45.89===4589000000000000000
  public static String doubleToString(double num){
      //assume precision of 10^-17 for double
      //e.g 0.00000000000000001
      int count=0;
String number="";
    double absVal=Math.abs(num);

if(absVal<1){
    while(absVal<1){
    absVal*=10;
    count++;
    count*=-1;
    }
}
else if(absVal>10){
   while(absVal>10){
    absVal/=10;
    count++;
}
}
/*from count we may deduce the no. of d.p in the number
e.g in 123.55
    count = 2
 for 0.035, count = 2 also.
 * So count = base 10 exponent of the number when written in standard form
*/

while(number.length()<17){
    double abs=absVal;//shadow variable for absVal
    number+=convertdoubleDigitToString(floor(absVal));
    absVal-=floor(absVal);   //(   absVal = abs-Math.floor(abs))
    //the last step generates some error,try and get it out by using the shadow variable for absVal
    double error = abs-(absVal+floor(abs));
    //23.987
    absVal+=error;

    absVal*=10;
}


//number is always displayed in exponential format
number=number.substring(0,1)+"."+number.substring(1)+"E"+count;
if(num<0){
    number="-"+number;
}



    return number;
  }





/**
 *
 * @param str The String object to check.
 * @return true if it contains at least one non-white space characters.
 */
    public static boolean hasChars( String str ){
        int len = str.length();
        for( int i=0; i<len;i++){
            String sub = str.substring(i, i+1);
        if( sub.trim().length() > 0 ){
            return true;
        }//end if
        }//end for loop
    return false;
    }



/**
 * Checks if a given number is even
 * @param a the number
 * @return true if a is even
 */
public static boolean isEven(double a) {
   return (a%2)==0;
}
/**
 * Checks if a given number is even
 * @param a the number
 * @return true if a is even
 */
public static boolean isEven(float a) {
   return (a%2)==0;
}
/**
 * Checks if a given number is even
 * @param a the number
 * @return true if a is even
 */
public static boolean isEven(int a) {
    return (a%2)==0;
   }




public static void main(String args[]){//tester method for STRING methods

    String h = "F= @(x,y,z,w,...)mathexpr ";
    String v = "F(x,y,z,w,...) = mathexpr ";
    
    System.out.println(purifier(h));
    System.out.println(purifier(v));
String str = "Am I not trying a lot for you ?";



String arr[] = splits( str, " ");

for( String c:arr){
System.out.print( c+","  );
    }

/*
System.out.println( "Intent: TO REPLACE ALL OCCURRENCES\n"
        + "OF "+replaced+" IN "+str+"\n WITH "+replacing );
String aaa = replaceAll(str, replaced, replacing);
System.out.println( aaa );
*/
}//end main
}//end class STRING

//Shohkhohlhohkhohbanhghohsheh
//A+B+356.523E-235+4C
//Nebuchadrezzar