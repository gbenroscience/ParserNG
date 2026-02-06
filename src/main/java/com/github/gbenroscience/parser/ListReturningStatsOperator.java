/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package parser;
import parser.methods.Method;

import java.util.ArrayList;

/**
 *
 * @author GBEMIRO
 */
public final class ListReturningStatsOperator extends Operator  implements Validatable{




    /**
     * The container of all ListTypeOperator objects in the scanned function.
     */
    private boolean superParent;
/**
 * The index of this Operator in its parent scanned function.
 */
private int index;

/**
 * The opening bracket operator that forms one of the bracket
 * pair used by this Operator to bound its data to the left
 */
    private Bracket openBracket;
/**
 * The closing bracket operator that forms one of the bracket
 * pair used by this Operator to bound its data to the right
 */
    private Bracket closeBracket;

    /**
     * The ListReturningStatsOperator object that immediately envelopes this one.
     */
    private ListReturningStatsOperator parent;


   private static String errorMessage="";

    /**
     *
     * @param op the name of the operator
     * @param index  the index of this Operator in its parent scanned function.
     * @param scan the ArrayList object that contains the scanned function that has this ListReturningStatsOperator object
     */
    public ListReturningStatsOperator(String op,int index,ArrayList<String>scan) {
         super( op);
         this.index=(scan.get(index).equals(op))?index:-1;
         this.openBracket=new Bracket("(");
         this.openBracket.setIndex(index+1);
         int compIndex=Bracket.getComplementIndex(true, index+1, scan);
         closeBracket=new Bracket(")");
         closeBracket.setIndex(compIndex);
         openBracket.setComplement(closeBracket);
         closeBracket.setComplement(openBracket);
         determineSuperParentStatus(scan);
         hasParent(scan);
         if(this.index==-1){
             throw new ArrayIndexOutOfBoundsException("PARSER COULD NOT FIND\n \'"+op+ "\' AT INDEX "+index+
                   ".\nFOUND \'"+scan.get(index)+"\' INSTEAD OF \'"+op+"\'");
         }

    }



    /**
     *
     * @return true if this object is the superParent
     * i.e the container for all data in the function.
     */
    private void determineSuperParentStatus(ArrayList<String>scan){
        boolean isSuper=true;
        int scanStartIndex=getCloseBracket().getIndex();
 loopForwards:{
     for(int i=scanStartIndex+1;i<scan.size();i++){
         if( !isClosingBracket( scan.get(i) ) ){
             isSuper=false;
             break loopForwards;
         }

        }//end for loop

 }//end loop

  loopBackwards:{
     for(int i=index-1;i>=0;i--){
         if( !isOpeningBracket( scan.get(i) ) ){
             isSuper=false;
             break loopBackwards;
         }

        }//end for loop

 }//end loop


setSuperParent(isSuper);
    }



/**
 *
 * @param openBracket sets the opening bracket
 * for this Operator
 */
    public void setOpenBracket(Bracket openBracket) {
        this.openBracket = openBracket;
    }
/**
 *
 * @return the opening bracket
 * for this Operator
 */
    public Bracket getOpenBracket() {
        return openBracket;
    }
/**
 *
 * @param closeBracket sets the closing bracket
 * for this Operator
 */
    public void setCloseBracket(Bracket closeBracket) {
        this.closeBracket = closeBracket;
    }
/**
 *
 * @return the closing bracket
 * for this Operator
 */
    public Bracket getCloseBracket() {
        return closeBracket;
    }
/**
 *
 * @param index sets the location of this Operator
 * object in its parent scanned function.
 */
    public void setIndex(int index) {
        this.index = index;
    }
/**
 *
 * @return the index of its parent scanned function.
 */
    public int getIndex() {
        return index;
    }
/**
 *
 * @param superParent sets whether or not
 * this is the container ListReturningStatsOperator object
 * for the data set.
 */
    public void setSuperParent(boolean superParent) {
        this.superParent = superParent;
    }
/**
 *
 * @return true if this is the container ListReturningStatsOperator object
 * for the data set.
 */
    public boolean isSuperParent() {
        return superParent;
    }



/**
 * The concept of a parent here is that
 * the first ListReturningStatsOperator object to
 * this Operator's left is one that encloses it.
 * i.e    sort(2,3,mode(4,1,1,3))
 * Here sort is a parent to mode.
 *
 * But in sort(2,3,sort(4,1,3,4),mode(4,5,1)),mode will
 * have no parent here by our definition. The outermost sort which would
 * have been its parent does not immediately enclose it.The outermost sort
 * however is a parent to the second sort(4,1,3,4).
 * We use this special definition for parent because we wish to ensure that each
 * object of this class will validate its own surroundings, not even its contents.
 * It will not validate beyond the next ListReturningStatsOperator to it.
 *
 * @param parent sets the ListReturningStatsOperator object that immediately envelopes this one.
 */
    public void setParent(ListReturningStatsOperator parent) {
        this.parent = parent;
    }
/**
 *
 * @return the ListReturningStatsOperator object that immediately envelopes this one.
 */
    public ListReturningStatsOperator getParent() {
        return parent;
    }
/**
 *
 * @param errorMessage sets the error message generated by objects of this class.
 */
    public static void setErrorMessage(String errorMessage) {
        ListReturningStatsOperator.errorMessage = errorMessage;
    }
/**
 *
 * @return the error message generated by objects of this class.
 */
    public static String getErrorMessage() {
        return errorMessage;
    }


/**
 *
 * @param scan the ArrayList containing the scanned function
 * @return true if the first ListReturningStatsOperator object to its
 * left is its parent. So it can have a parent and yet return false
 * here.
 */

    public boolean hasParent(ArrayList<String>scan){
boolean isEnveloped = false;
        int i=index;

        boolean foundLikelyEncloser=false;
        //search for an enclosing ListReturningStatsOperator object to the left of this
        //Operator's location.If one is found set the boolean to true and exit the search loop.
        for(i=index-1;i>=0;i--){

            if( Method.isListReturningStatsMethod( scan.get(i) ) ){
                foundLikelyEncloser=true;
                break;
            }//end if

        }//end for

    //Try to get the index of the enclosing bracket of the enclosing operator if one exists.
        if(foundLikelyEncloser){
        int compIndex=Bracket.getComplementIndex(true, i+1, scan);

        if(compIndex>closeBracket.getIndex()){
         isEnveloped=true;
   ListReturningStatsOperator listType = new ListReturningStatsOperator(scan.get(i), i, scan); 
setParent(listType);
        }// end if

       
        }//end if


return isEnveloped;
    }




/**
 * 
 * @param scan the ArrayList of the scanned function
 * that contains this object
 * @return true if this object scans through its surroundings to
 * the left and to the right and sees
 * a valid bracket structure around itself
 */
    @Override
public boolean validate(ArrayList<String>scan){
    boolean valid=true;


    if(!isSuperParent()){
try{
 if(isBinaryOperator( scan.get(index-1) )|| Method.isUnaryPreOperatorORDefinedMethod(scan.get(index-1))){
        valid=false;
errorMessage+="\n Bad Syntax! Do Not Concatenate operator "+scan.get(index-1)+" With "+getName();
    }//end if
 }//end try
catch(IndexOutOfBoundsException indexErr){

}//end catch
try{
if( isBinaryOperator( scan.get(closeBracket.getIndex()+1) )||isUnaryPostOperator(scan.get(closeBracket.getIndex()+1))){
        valid=false;
      errorMessage+="\n Bad Syntax! Do Not Append operator "+scan.get(index-1)+" To "+getName();
    }//end else if
}//end try 
catch(IndexOutOfBoundsException indexErr){

}//end catch
 //validate further
 if(valid){
    loopBackwards:{
        boolean openBracsOnly=true;
        for(int i=index-1;i>=0;i--){
   //this logic disallows non-listtypestatsoperators from enveloping listtypestatsoperators.
            if(isOpeningBracket(scan.get(i))&& Method.isUnaryPreOperatorORDefinedMethod(scan.get(i-1))){
                int compIndex=Bracket.getComplementIndex(true, i, scan);
                if(compIndex>closeBracket.getIndex()){
                    valid=false;
errorMessage+="\n Bad Syntax! Do Not Embed "+getName()+" In Parentheses Belonging To "+scan.get(i-1);
                break loopBackwards;
                }
}//end if

            //this logic recognizes the end point of backwards bracket
            //validation for a given object of this class. A close bracket
            //indicates the end of data that can affect this object
            if(isClosingBracket(scan.get(i))|| Method.isStatsMethod(scan.get(i))){
                errorMessage+="\n Ending Backwards Validation For "+getName();
                break loopBackwards;
            }//end if

            //this logic (the next two ifs)will disallow binary operations
            //on encapsulations of objects of this class
            // and their operands within brackets e.g sort(3,2,4)+((sort(1,3,1,-9,3))
if(!isOpeningBracket(scan.get(i))){
                openBracsOnly = false;
errorMessage+="\n MBracket Sequence Established. Trend Finished. Apllying Other Validation Techniques.";
}
       
if(openBracsOnly&&isBinaryOperator( scan.get(i-1) )|| Method.isUnaryPreOperatorORDefinedMethod(scan.get(i-1))){
                valid=false;
                errorMessage+="\n Bad Syntax For Data Set Returning Operator "+getName()+"\n" +
                        "REASON:::"+
                        "Cannot Perform Binary Operations On Data Set.";
                break loopBackwards;
            }


        }//end for
    }//end loopBackwards

    loopForwards:{
        boolean closeBracsOnly=true;

for(int i=closeBracket.getIndex()+1;i<scan.size();i++){

            //this logic recognizes the end point of backwards bracket
            //validation for a given object of this class. A close bracket
            //indicates the end of data that can affect this object
            if(isOpeningBracket(scan.get(i))|| Method.isStatsMethod(scan.get(i))){
                 break loopForwards;
            }//end if


            //this logic (the next two ifs)will disallow binary operations
            //on encapsulations of objects of this class
            // and their operands within brackets e.g sort(3,2,4)+((sort(1,3,1,-9,3))
if(!isClosingBracket(scan.get(i))){
                closeBracsOnly = false;
}//end if
try{
if(closeBracsOnly&&isBinaryOperator( scan.get(i+1) )||isUnaryPostOperator(scan.get(i+1))){ 
                valid=false;
                break loopForwards;
            }//end if
}
catch(IndexOutOfBoundsException indexErr){
    
}

}//end for loop



    }//end loopForwards

 }//end if
 
    }//end if


   else if (isSuperParent()){
       return true;
    }


    return valid;
}

/**
 * Takes an object of class Function and validates its ListReturningStatsOperators objects.
 * @param scan the scanner output
 * @return true if valid
 */
public static boolean validateFunction(ArrayList<String>scan){

ListReturningStatsOperator list;
   boolean validity=true;

   for(int i=0;i<scan.size();i++){
     if(Method.isListReturningStatsMethod(scan.get(i))){
            list = new ListReturningStatsOperator(scan.get(i), i, scan);
validity=list.validate(scan);
i=list.getCloseBracket().getIndex();
if(!validity){
    break;
}//end if
       }//end if
}//end for
return validity;
}//end method validateFunction(args)





    @Override
    public String toString() {
   String objectName="ListReturningStatsOperator \""+getName()+"\" located @ index "+index+" has its opening bracket @ "+openBracket.getIndex()+
           " and its closing bracket at "+closeBracket.getIndex()+
           ((isSuperParent())?" and is the root parent of this data set ":" and is not the root parent of this data set ");

   return objectName;
    }


/*
public static void main(String args[]){

  ArrayList<String>scan= new ArrayList<String>();

  scan = new MathScanner("sum(sort(2,3),sum(3,2))").scanner(null);

ListReturningStatsOperator list = new ListReturningStatsOperator("sum", 0, scan);

 util.Utils.logError(list.toString());

 util.Utils.logError(list.validate(scan));

}//end main

*/











}//end class