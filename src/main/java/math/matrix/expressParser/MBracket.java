/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import java.util.ArrayList;



/**
 *
 * @author GBENRO
 */

public class MBracket extends MOperator {
/**
 * the name of the bracket i.e "(" or ")"
 */
private String name = "";

/**
 * The index of the bracket in the ArrayList containing the scanned function
 */
private int index;

/**
 * objects of this class keep a record of their counterpart or complementing bracket.
 *
 */
private MBracket complement;
/**
 * Return true if the contents of the bracket have been evaluated
 */
private boolean evaluated=false;

/**
 * Constructor of this class for creating its objects and initializing their names with either
 * a ( or a )
 * and initial
 * @param op
 */
    public MBracket(String op) {
        super(op);

    }
/**
 *
 * @param evaluated set whether or not this bracket's contents have been evaluated
 */
    public void setEvaluated(boolean evaluated) {
        this.evaluated = evaluated;
    }
/**
 *
 * @return true if this bracket's contents have been evaluated
 */
    public boolean isEvaluated() {
        return evaluated;
    }





/**
 * Used to create similar objects that are not equal
 * The object created by this class is similar to the parameter
 * because it contains the same data as the parameter.
 * However,its address in memory is different because
 * it refers to an entirely different object of the
 * same class,but having similar attributes.
 *
 * How can this method be of any use?
 * Imagine an Array of Brackets say array bracs filled with MBracket objects.
 *
 * If we create another MBracket array, say array moreBracs and copy
 * the objects in bracs into moreBracs.Now, both bracs and moreBracs will
 * hold references to these MBracket objects in memory.Java will not create new,
 * similar objects at another address in memory and store in the new array.
 * The command was most likely moreBracs=bracs;
 * or in a loop, it would look like:
 *
 * for(int i=0;i&lt;bracs.length;i++){
 * moreBracs=bracs[i];
 * }
 *
 * These statements will only ensure that both arrays will hold a reference to
 * the same objects in memory,i.e RAM.
 *
 * Hence whenever an unsuspecting coder modifies the contents of bracs, thinking He/She has a
 * backup in moreBracs,Java is effecting the modification on the objects referred to by
 * moreBracs, too.This can cause a serious logical error in applications.
 * To stop this, we use this method in this way:
 *
 * for(int i=0;i&lt;bracs.length;i++){
 * moreBracs[i]=createTwinBracket(bracs[i]);
 * }
 *
 *
 *
 *
 * Note that this can be applied to all storage objects too e.g Collection objects and so on.
 *
 * @param brac The object whose twin we wish to create.
 * @return a MBracket object that manifests exactly
 * the same attributes as brac but is a distinct object from brac.
 */
    public static MBracket createTwinBracket(MBracket brac) {
       MBracket newBrac=new MBracket(brac.getName());
       newBrac.setComplement(brac.getComplement());
       newBrac.setIndex(brac.getIndex());
return newBrac;
    }
/**
 * non-static version of the above method.
 * This one creates a twin for this MBracket object.
 * The one above creates a twin for the specified bracket object.
 * @return a MBracket object that manifests exactly
 * the same attributes as brac but is a distinct object from brac.
 */
       public MBracket createTwinBracket() {
       MBracket newBrac=new MBracket(getName());
       newBrac.setComplement(getComplement());
       newBrac.setIndex(getIndex());
return newBrac;
    }

/**
 *
 * @return the index of this MBracket object in a scanned function
 */
    public int getIndex() {
        return index;
    }
/**
 *
 * @param index the ne w index to assign to this MBracket object in a scanned Function
 */
    public void setIndex(int index) {
        this.index = index;
    }
/**
 *
 * @return the name of this MBracket either ( or )
 */
    public String getName() {
        return name;
    }
/**
 *
 * @param name sets the name of this bracket to either ( or )
 */
    @Override
    public void setName(String name) {
        this.name = name;
    }
/**
 *
 * @return the MBracket object which is the complement of this MBracket object
 */
    public MBracket getComplement() {
        return complement;
    }
/**
 *
 * @param complement sets the MBracket object which is to be the complement to this one in the scanned Function
 */
    public void setComplement(MBracket complement) {
        this.complement = complement;
    }
/**
 * checks if the MBracket object argument below is the
 * sane as the complement to this MBracket object.
 * @param brac The MBracket object whose identity is to be checked
 * whether or not it complements this MBracket object.
 * @return true if the parameter is the complement to this one.
 */
public boolean isComplement(MBracket brac){
        return brac==getComplement();
}
/**
 *
 * @param brac the bracket to be checked if or not it is enclosed
 * by this bracket and its complement.
 * @return true if the bracket is enclosed by this bracket and its counterpart.
 */
public boolean encloses(MBracket brac){
   boolean truth=false;

   if(this.getIndex()<brac.getIndex() && this.getComplement().getIndex()>brac.getIndex()  ){
    truth=true;
    }
   else if(this.getIndex()>brac.getIndex() && this.getComplement().getIndex()<brac.getIndex() ){
       truth=true;
   }



return truth;
}

/**
 *
 * @param brac an ArrayList object containing all brackets found in a function
 * @return the number of bracket pairs contained between this MBracket object and its complement
 */
public int getNumberOfInternalBrackets( ArrayList<MBracket>brac  ){
int num=0;
int i=0;
while( i<brac.size()  ){
  if( encloses( brac.get(i)) ){
      num++;
  }

i++;
}
return (num/2);
}





/**
 * @param isOpenBracket boolean variable that should be true if this bracket object whose complement
 *         we seek is an opening bracket i.e (, and should be set to false if this bracket object whose complement
 *         we seek is a closing bracket i.e )
 * @param start the index of the given bracket.
 * @param scan  the ArrayList containing the scanned function.
 * @return the index of the enclosing or complement bracket of this bracket object
 */
public static int getComplementIndex(boolean isOpenBracket,int start,ArrayList<String>scan){

    int open=0;
    int close=0;
    int stop=0;
    if(isOpenBracket){
        try{
    for(int i=start;i<scan.size();i++){

        if(scan.get(i).equals("(")){
            open++;
        }
        else if(scan.get(i).equals(")")){
            close++;
        }
if(open==close){
            stop=i;
            break;
        }

    }//end for
        }//end try
catch(IndexOutOfBoundsException ind){

}
    }//end if
    else if(!isOpenBracket){
        try{
    for(int i=start;i>=0;i--){
try{
        if(scan.get(i).equals("(")){
            open++;
        }
        else if(scan.get(i).equals(")")){
            close++;
        }
if(open==close){
            stop=i;
            break;
        }
}//end try
catch(IndexOutOfBoundsException ind){

}
    }//end for
        }//end try
        catch(IndexOutOfBoundsException ind){

        }


    }
return stop;
}
/**
 * 
 * @param bracket The String object.
 * @return true if the String object represents an open bracket
 */
public static boolean isOpenBracket(String bracket){
    return bracket.equals("(");
}

/**
 *
 * @param bracket The String object.
 * @return true if the String object represents a close bracket
 */
public static boolean isCloseBracket(String bracket){
    return bracket.equals(")");
}






/**
 * returns a List containing the contents of a bracket pair,including the bracket pair itself.
 * @param scan the ArrayList containing the scanner output for a Function
 * @return the bracket pair and its contents.
 */
public ArrayList<String> getBracketDomainContents( ArrayList<String>scan  ){
 if(isOpeningBracket(this.getName())){
     return (ArrayList<String>) scan.subList(this.getIndex(), this.getComplement().getIndex()+1);
 }
 else if(isClosingBracket(this.getName())){
       return (ArrayList<String>) scan.subList(this.getComplement().getIndex(), this.getIndex()+1);
 }
  return null;
}







}//end class MBracket
