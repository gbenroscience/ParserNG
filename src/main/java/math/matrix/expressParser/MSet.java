/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import parser.STRING;
import java.util.List;
import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Random;
import math.Maths;
import util.FunctionManager;

/**
 *Class that provides utility methods for carrying
 * out statistical analysis on a data set consisting of real numbers.
 * @author GBENRO
 */
public class MSet {
/**
 * The data attribute of objects if this class
 */
private List<String>data=new ArrayList<String>();



/**
 *
 * @param data
 */
public MSet(double...data){

    for(int i=0;i<data.length;i++){
    this.data.add(data[i]+"");
    }
}//end constructor
/**
 * Creates a new MSet object initialized
 * with the specified data set.
 * @param data the data set used
 * to initialize the data attribute of this class
 */
public MSet(List<String>data ){
    this.data=data;
}

/**
 * Creates a new MSet object initialized with a set of data coming from a mathematical
 * MathExpression.
 * @param function the Math MathExpression from which the
 * set of data is coming.
 * @param data the incoming data set
 */
public MSet(MatrixFunction function, List<String>data ){
    System.out.println(data);
     this.data=function.solveSubPortions(data);
}//end constructor

/*    public MSet(math.MathExpression aThis, List<String> executable) {
       this.data=aThis.solveSubPortions(executable);
    }
*/
/**
 *
 * @param data sets the data to be operated on
 */
public void setData(ArrayList<String>data){
    this.data=data;
}
/**
 *
 * @return the data set
 */
public List<String>getData(){
    return data;
}
/**
 *
 * @return the number of elements in the data set
 */
public int size(){
 return data.size();
}
/**
 *
 * @return the sum of all elements in the data set
 */
public double sum(){
double u=0;
for(int i=0;i<data.size();i++){
   u+=Double.valueOf(data.get(i));
}

    return u;
    }//end sum

/**
 *
 * @return the sum of squares of values in the data set.
 */
public double sumOfSquares(){
double u=0;
for(int i=0; i<data.size(); i++){
u+=Math.pow(Double.valueOf( data.get(i) ), 2);
}
    return u;
}
/**
 *
 * @return the product of all elements in the data set.
 */

public double prod(){
double u=1;
for(int i=0;i<data.size();i++){
u*=Double.valueOf(data.get(i));
}
 return u;
}

/**
 * this method determines the least value in a set of numbers
 * @return the least value in a set of numbers
 * @throws NumberFormatException
 */
public double min(){
//the maximum finder algorithm begins
double minval = Double.valueOf( data.get(0) );
for(int i=0;i<data.size();i++){
double y=Double.valueOf( data.get(i) );
if(y<minval){minval=y;     }
}


return  minval;
}

/**
 *
 * @return the maximum value in the data set.
 */
public double max(){
//the maximum finder algorithm begins
double maxval=Double.valueOf( data.get(0) );
for(int i=0;i<data.size();i++){
double y=Double.valueOf( data.get(i) );
if(y>maxval){maxval=y; }
}


return maxval;
}


/**
 *
 * @return the meanor average value of a data set
 */
public double avg(){
    return  sum()/size();
    }//end avg
/**
 *
 * @return the root mean squared value of the data set
 */
public double rms(){
    return Math.sqrt(  sumOfSquares() )/size() ;
    }//end rms

/**
 *
 * @return the range of the data set
 */
public double rng(){
    return  max()-min();
    }//end range
/**
 *
 * @return the midrange of the data set
 */
public double mrng(){
    return  0.5*(  max()+min()   ) ;
    }//end midrange

/**
 *
 * @return the variance
 */
public double var(){
double u=0;double u1=0;
double avrg = avg();
double N = size();
for(int i=0;i<N;i++){
u=(Double.valueOf(data.get(i))-avrg);
u1+=Math.pow(u, 2);
}
return u1/N;
}//end variance
/**
 *
 * @return the standard deviation
 */
public double std_dev(){//Finds the variance
return Math.sqrt( var() );
}
/**
 *
 * @return the standard error
 */
public double std_err(){//Finds the variance
return ( std_dev()/Math.sqrt(size()) ) ;
}
/**
 *
 * @return the coefficient of variation
 */
public String cov(){//Finds the coefficient of variastion
return (100*std_dev()/avg())+"%";
}
/**
 *
 * @return displays the output of the sort method as the sorting process proceeds.
 */
private List <String> displayOuputLineByLine(){
   List<String>sort=new ArrayList<String>();
for(int i=0;i<size();i++){
    sort.add(data.get(i));
}
    return sort;
}

/**
 *
 * @return sorts a number set and returns the result as
 * a string of comma separated values sorted in ascending order
 */
public String sort(){
      List<String>sort=new ArrayList<String>();
    for(int j=0;j<size();j++){
    for(int i=0;i+1<size();i++){
        try{
        double swapcontrol=0;
          if(Double.valueOf( data.get(i) )>Double.valueOf( data.get(i+1) )){
              swapcontrol=Double.valueOf( data.get(i+1) );
              data.set( i+1,data.get(i) );
              data.set(i,String.valueOf( swapcontrol ) );
          }//end if
        }//end try
        catch(IndexOutOfBoundsException indErr){

        }
    }//end for
   if(j==size()-1){
       sort = displayOuputLineByLine();
    }//end if
        }//end for

    return STRING.delete(  STRING.delete( sort.toString(),"["),"]" );

    //return  sort;
}//end sort
/**
 *
 * @return a number list sorted in ascending order
 */
public List<String> sort1(){
          List<String>sort=new ArrayList<String>();
    for(int j=0;j<size();j++){
    for(int i=0;i+1<size();i++){
        try{
        double swapcontrol=0;
          if(Double.valueOf( data.get(i) )>Double.valueOf( data.get(i+1) )){
              swapcontrol=Double.valueOf( data.get(i+1) );
              data.set( i+1,data.get(i) );
              data.set(i,String.valueOf( swapcontrol ) );
          }//end if
        }
        catch(IndexOutOfBoundsException indErr){

        }
    }//end for
   if(j==size()-1){
       sort = displayOuputLineByLine();
    }//end if
        }//end for
    return  sort;
}//end sort
/**
 *
 * @return the median of the data set
 */
public double median(){
    double median = 0;
    List<String>scan=new ArrayList<String>();
scan=sort1();
    int g=size();
  if(STRING.isEven(g)){
double med = ( Double.valueOf( scan.get( g/2 ) ) + Double.valueOf( scan.get( (g/2)-1  ) ) )/2 ;
median= med  ;
  }
  else if(!STRING.isEven(g)){
median= Double.valueOf( scan.get( (g-1)/2) );
  }

 return median;
}

/**
 *
 * @return the mode of a number set as a list
 */
public String mode(){
    List<String>sorter=new ArrayList<String>();
    List<Double>freq=new ArrayList<Double>();
    sorter = sort1();
    int g = sorter.size();
    double count = 0;
for(int j=0;j < g;j++){
for(int i=0;i < g;i++){
    if( sorter.get(i).equals(sorter.get(j))){
    count++;
    }//end if
}//end for
freq.add(count);
count=0;
}//end for
//sorts records one occurence each of the entries
double big=0;
int index = -1;
for(int i = 0;i<freq.size();i++){
    if(big<freq.get(i)){
        big = freq.get(i);
        index = i;
    }//end if
}//end for

//get the values with common frequency
List<String> mode = new ArrayList<String>();
 for(int i=0;i<freq.size();i++){
  if(freq.get(i)==big && !mode.contains(sorter.get(i))){
      mode.add(sorter.get(i));
  }//end if

 }// end for
String dMode=String.valueOf(mode);
dMode=STRING.delete( STRING.delete(dMode, "["),"]"  );
return dMode;
    //return "The number(s) "+mode+" with "+big+" occurences has/have the highest frequency ";
}



/**
 *
 * if no value is found in the data set, the software will generate floating point values randomly between
 * 0.0 and 1.0 ( 0.0 inclusive and 1.0 exclusive).
 * Else:
 * If the data set has only one number, e.g [m]
 * this method will randomly generate a number
 * between 0 and m-1
 * If the list has 2 numbers, say m and n, e.g [m,n]
 * The method will generate n numbers between 0 and m-1
 *
 *
 *
 * @return a list of values generated randomly according to
 * the format of the random command.
 */
public List<String>random(){
    List<String> vals= new ArrayList<String>();
    Random rnd = new Random();
//check if the random command came with only one argument
    if(size()==1){
        int val=Integer.valueOf( data.get(0) );
        data.set(0,String.valueOf((int) val));
    }
    //check if it came with 2 arguments
    else if(size()==2){
        double val=   Double.valueOf( data.get(0) ) ;
        double val1= Double.valueOf( data.get(1) );
        data.set(0,String.valueOf((int)val));
        data.set(1,String.valueOf((int)val1));
    }



    if( size() == 0 ){
vals.add(String.valueOf(rnd.nextFloat()));
}

else if( size() == 1 ){
    String num =String.valueOf(rnd.nextInt(Integer.valueOf(data.get(0))));
    vals.add(num);
}
else if(size() == 2){

    for(int i=0;i<Integer.valueOf(data.get(1));i++){
    vals.add(String.valueOf(rnd.nextInt( Integer.valueOf( ( data.get(0) ) ))));
    }
}
else{
        throw new NoSuchElementException("Allowed Formats For rnd Are rnd(),rnd1(number)," +
                "rnd1( number1,number2)");
}
data.clear();
data.addAll(vals);
    return vals;
}

/**
 *
 * @return the permutation of 2 values.
 */
public String permutation(){
String n = data.get(0);
String r = data.get(1);

double n_Factorial = Double.parseDouble( Maths.fact(n) );
double n_Minus_r_Factorial = Double.parseDouble( Maths.fact(  String.valueOf( Double.valueOf(n)-Double.valueOf(r) )   ) );


return String.valueOf(  n_Factorial/n_Minus_r_Factorial  );
}
/**
 *
 * @return the combination of 2 values.
 */
public String combination(){
String n = data.get(0);
String r = data.get(1);

double n_Factorial = Double.parseDouble( Maths.fact(n) );
double n_Minus_r_Factorial = Double.parseDouble( Maths.fact(  String.valueOf( Double.valueOf(n)-Double.valueOf(r) )   ) );
double r_Factorial = Double.parseDouble( Maths.fact(r) );

return String.valueOf(  n_Factorial/( n_Minus_r_Factorial * r_Factorial ) );
}
/**
 *
 * @return Raises the number in index
 * 0 to a power equal to the number
 * in index 1.
 */
public String power(){
double n = Double.parseDouble( data.get(0) );
double pow = Double.parseDouble( data.get(1) );
return String.valueOf(  Math.pow(n, pow));
}


/**
 *
 * @param operator The operator.
 * @return the value of the user defined function.
 * @throws ClassNotFoundException if the function was never defined by the user.
 */
public String evaluateUserDefinedFunction(String operator) throws ClassNotFoundException{
int sz=data.size();

System.out.println("operator = "+operator);

System.out.println("data = "+data);
String fullname=operator.concat("(");
for(int i=0;i<sz;i++){
fullname=fullname.concat(data.get(i).concat(","));
}//end for loop

fullname = fullname.substring(0,fullname.length()-1);
fullname=fullname.concat(")");
try{
return FunctionManager.getFunction(operator).evalArgs(fullname);
}//end try
catch(Exception cnfe){
    throw new ClassNotFoundException("Could Not Find Function! "+fullname);
}


}



public static void main( String args[] ){

MSet set = new MSet(2,3000);
System.out.println( set.permutation() );
System.out.println( set.combination() );
System.out.println( set.random() );




}//end main



}//end class