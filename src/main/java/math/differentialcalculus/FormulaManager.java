/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import static parser.Number.*;
import java.util.List;
/**
 *
 * @author GBEMIRO
 */
public class FormulaManager {
        /**
     * Records all unique Formulae(no duplicates allowed) used by the function to be simplified.
     */
        private final ArrayList<Formula> FORMULAE;
        
        
         

        public FormulaManager() {
            FORMULAE = new ArrayList<Formula>();
        }

        public ArrayList<Formula> getFORMULAE() {
            return new ArrayList<Formula>(FORMULAE);
        }
/**
 * 
 * @return The number of Formula objects stored
 * by an object of this class.
 */
public int count(){
    return FORMULAE.size();
}



/**
 *
 * @param name The name of the Formula.
 * @return true if a Formula exists by the name supplied.
 */
public boolean contains( String name ){
return indexOf(name)!=-1;
}//end method
/**
 * <b>
 * <font color='red'>
 * THIS UTILITY METHOD CAN BE USED TO COMPARE THE DATA OF A FORMULA
 * OBJECT WITH OTHER FORMULAE IN THE STORE TO SEE IF A FORMULA
 * WITH SIMILAR DATA EXISTS ALREADY.
 * ALSO, THE COMPARISON RETURNS MULTIPLE INFORMATION, NOT JUST
 * ONE.
 * </font>
 * </b>
 * @param data  The data of the Formula.
 * @return an array having 3 entries.
 * Index 0 contains 0 if a comparable Formula is found in the store.
 * A comparable Formula is one that is either a constant factor of this one
 * or is equivalent to this one...i.e. they evaluate to the same number, always.
 * If index 0 contains -1, then no comparable Formula exists in the store 
 * and so indices 1 and 2 will also be -1.
 * Index 1 contains the index at which the comparable Formula is found.
 * Index 2 contains the factor. If factor =1 , then they are equivalent,
 * else,they are constant factors of each other.
 * 
 * 
 * 
 */
public Object[] comparisonData( List<String> data ){
    Object[]info = new Object[]{-1.0,-1.0,-1.0};
    Formula f = new Formula("myForm_1", data);
 for(int i=0;i<FORMULAE.size();i++){
     Formula fj = FORMULAE.get(i);
     if(f.isEquivalentTo(fj)){
         info[0]=0;
         info[1]=i;
         info[2]=1;
         return info;
     }//end if
     else{
         double factor = f.getFactor(fj);
         if(!Double.isNaN(factor)){
         info[0]=0;
         info[1]=i;
         info[2]=factor;
         return info;
         }//end if
     }//end else
 }//end for loop  
return info;
}//end method
/**
 *
 * @param name The name of the dependent variable of the Formula.
 * @return the index of the Formula object that has the name supplied.
 * If no such Formula object exists, then it returns -1.
 */
public int indexOf( String name ){
    int sz=FORMULAE.size();
for(int i=0;i<sz;i++){

    if( name.equals(FORMULAE.get(i).getName()) ){
        return i;
    }//end if
}//end for
return -1;
}//end method

/**
 *
 * @param name The name of the Formula.
 * @return the Formula object that has the name supplied
 * if it exists.
 * If no such Formula object exists, then it returns null.
 * @throws ClassNotFoundException if no Formula object by that name exists.
 */
public Formula getFormula( String name ) throws ClassNotFoundException{
    //First try to get the exact location of the Formula from its name, this is very fast.
    //If this fails then manually search for it.
    int indexOfnumber = name.indexOf("_")+1;
    int index = -1;
    if(indexOfnumber!=-1){
     String numberpart = name.substring(indexOfnumber);
     boolean isAutoGenName = name.startsWith("myForm_")&&isNumber(numberpart);
     
    if(isAutoGenName){
     index = Integer.parseInt(numberpart);  
     }
     
    }//end if
    else{
    index = indexOf(name);  
    }//end else
  if( index != -1 ){//contains the Formula!
  return FORMULAE.get(index);
  }//end if
  else{
        throw new ClassNotFoundException(" Formula "+name+" Does Not Exist.");
  }//end else
}//end method
    
        
        


/**
 *
 * @param index  The index of the Formula in
 * this FormulaManager object.
 * If no such index exists, then it returns null.
 * @throws ClassNotFoundException if no Formula object by that name exists.
 */
public Formula getFormula( int index ) throws ClassNotFoundException{
    if(index<FORMULAE.size()&&index>=0){
 return FORMULAE.get(index);
    }//end if
    else{
        throw new ClassNotFoundException(" The Index \'"+index+"\' Does Not Exist.");
    }//end else


}//end method
/**
 * Attempts to retrieve a Formula
 * object from a FormulaManager based on its name.
 * @param name The name of the Formula object.
 * @return the Formula object that has that name or
 * null if the Formula is not found.
 */
public Formula lookUp( String name ){
    try{
return getFormula(name);
    }//end try
    catch( ClassNotFoundException classNotFoundException){
     throw new NullPointerException("Formula "+name+" does not exist!");
    }//end catch
}//end method



/**
 *
 * @param form The Formula object to add to this object.
 */
public void add( Formula form ){
if(!contains(form.getName())){   
   FORMULAE.add(form);
}//end if
else if(contains(form.getName())){
    update(form.getName(),form.getData());
}
}//end method

/**
 * Removes a Formula object from this FormulaManager.
 */
public void delete( String name ){
 int index = indexOf(name);
 if(index!=-1){
 FORMULAE.remove(index);
 }//end if
}//end method


public Formula firstFormula(){
    int sz = FORMULAE.size();
    if(sz>0){
        return FORMULAE.get(0);
    }
    throw new NullPointerException("Sorry,No Element Defined Yet!");
}//end method
public Formula lastFormula(){
    int sz = FORMULAE.size();
    if(sz>0){
        return FORMULAE.get(sz-1);
    }
    throw new NullPointerException("Sorry,No Element Defined Yet!");
    
}//end method


/**
 * It updates the name of a Formula object in this FormulaManager with the
 * parameter <code>newName</code>.
 * @param name The name of the Formula in this FormulaManager
 * that we wish to update.
 * @param newName The new name to give the Formula.
 * @return The index of the object, if a Formula by that
 * name is already recorded in the FormulaManager and
 * <code>-1</code> the Formula is not found.
 * 
 */
public int update( String name,String newName ){
    try {
        Formula form = getFormula(name);
        form.setName(newName);
   int index = indexOf(form.getName());
             FORMULAE.set(index, form);
             return index;
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(FormulaManager.class.getName()).log(Level.SEVERE, null, ex);
            return -1;
   }
}//end method
/**
 * Updates a Formula object in this FormulaManager.
 * @param name The name of the Formula object to be updated.
 * @param data The data to be used to update the object if found.
 * @return The index of the object, if a Formula by that
 * name is already recorded in the FormulaManager and
 * <code>-1</code> the Formula is not found.
 */
public int update( String name,List<String>data ){
    try {
        Formula form = getFormula(name);
        form.setData(data);
   int index = indexOf(form.getName());
       FORMULAE.set(index, form);
       return index;
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(FormulaManager.class.getName()).log(Level.SEVERE, null, ex);
            return -1;
   }
}//end method


public void clearAll(){
    FORMULAE.clear();
}

        


}
