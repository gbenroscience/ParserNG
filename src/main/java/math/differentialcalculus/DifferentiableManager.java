/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import static parser.Number.*;

/**
 *
 * @author GBEMIRO
 */
public class DifferentiableManager {
    /**
     * Records all differentiables used by the function to be differentiated.
     */
        private final ArrayList<Differentiable> DIFFERENTIABLES;

        public DifferentiableManager() {
            DIFFERENTIABLES = new ArrayList<>();
        }

        public ArrayList<Differentiable> getDIFFERENTIABLES() {
            return new ArrayList<>(DIFFERENTIABLES);
        }
/**
 * 
 * @return The number of Differentiable objects stored
 * by an object of this class.
 */
public int count(){
    return DIFFERENTIABLES.size();
}



/**
 *
 * @param name The name of the Differentiable.
 * @return true if a Differentiable exists by the name supplied.
 */
public boolean contains( String name ){
return indexOf(name)!=-1;
}//end method
/**
 *
 * @param name The name of the dependent variable of the Differentiable.
 * @return the index of the Differentiable object that has the name supplied.
 * If no such Differentiable object exists, then it returns -1.
 */
public int indexOf( String name ){
    int sz=DIFFERENTIABLES.size();
for(int i=0;i<sz;i++){

    if( name.equals(DIFFERENTIABLES.get(i).getName()) ){
        return i;
    }//end if
}//end for
return -1;
}//end method

/**
 *
 * @param name The name of the Differentiable.
 * @return the Differentiable object that has the name supplied
 * if it exists.
 * If no such Differentiable object exists, then it returns null.
 * @throws ClassNotFoundException if no Differentiable object by that name exists.
 */
public Differentiable getDifferentiable( String name ) throws ClassNotFoundException{
    //First try to get the exact location of the Differentiable from its name, this is very fast.
    //If this fails then manually search for it.
    int indexOfnumber = name.indexOf("_")+1;
    int index = -1;
    if(indexOfnumber!=-1){
     String numberpart = name.substring(indexOfnumber);
     boolean isAutoGenName = name.startsWith("myDiff_")&&isNumber(numberpart);
     
    if(isAutoGenName){
     index = Integer.parseInt(numberpart);  
     }
     
    }//end if
    else{
    index = indexOf(name);  
    }//end else
  if( index != -1 ){//contains the Differentiable!
  return DIFFERENTIABLES.get(index);
  }//end if
  else{
        throw new ClassNotFoundException(" Differentiable "+name+" Does Not Exist.");
  }//end else
}//end method
    
        
        


/**
 *
 * @param index  The index of the Differentiable in
 * this DifferentiableManager object.
 * If no such index exists, then it returns null.
 * @throws ClassNotFoundException if no Differentiable object by that name exists.
 */
public Differentiable getDifferentiable( int index ) throws ClassNotFoundException{
    if(index<DIFFERENTIABLES.size()&&index>=0){
 return DIFFERENTIABLES.get(index);
    }//end if
    else{
        throw new ClassNotFoundException(" The Index \'"+index+"\' Does Not Exist.");
    }//end else


}//end method
/**
 * Attempts to retrieve a Differentiable
 * object from a DifferentiableManager based on its name.
 * @param name The name of the Differentiable object.
 * @return the Differentiable object that has that name or
 * null if the Differentiable is not found.
 */
public Differentiable lookUp( String name ){
    try{
return getDifferentiable(name);
    }//end try
    catch( ClassNotFoundException classNotFoundException){
     throw new NullPointerException("Differentiable "+name+" does not exist!");
    }//end catch
}//end method



/**
 *
 * @param diff The Differentiable object to add to this object.
 */
public void add( Differentiable diff ){
if(!contains(diff.getName())){   
   DIFFERENTIABLES.add(diff);
}//end if
else if(contains(diff.getName())){
    update(diff.getName(),diff.getData());
}
}//end method

/**
 * Removes a Differentiable object from this DifferentiableManager.
 */
public void delete( String name ){
 int index = indexOf(name);
 if(index!=-1){
 DIFFERENTIABLES.remove(index);
 }//end if
}//end method


public Differentiable firstDifferentiable(){
    int sz = DIFFERENTIABLES.size();
    if(sz>0){
        return DIFFERENTIABLES.get(0);
    }
    throw new NullPointerException("Sorry,No Element Defined Yet!");
}//end method
public Differentiable lastDifferentiable(){
    int sz = DIFFERENTIABLES.size();
    if(sz>0){
        return DIFFERENTIABLES.get(sz-1);
    }
    throw new NullPointerException("Sorry,No Element Defined Yet!");
    
}//end method


/**
 * It updates the name of a Differentiable object in this DifferentiableManager with the
 * parameter <code>newName</code>.
 * @param name The name of the Differentiable in this DifferentiableManager
 * that we wish to update.
 * @param newName The new name to give the Differentiable.
 * @return The index of the object, if a Differentiable by that
 * name is already recorded in the DifferentiableManager and
 * <code>-1</code> the Differentiable is not found.
 * 
 */
public int update( String name,String newName ){
    try {
        Differentiable diff = getDifferentiable(name);
        diff.setName(newName);
   int index = indexOf(diff.getName());
             DIFFERENTIABLES.set(index, diff);
             return index;
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(DifferentiableManager.class.getName()).log(Level.SEVERE, null, ex);
            return -1;
   }
}//end method
/**
 * Updates a Differentiable object in this DifferentiableManager.
 * @param name The name of the Differentiable object to be updated.
 * @param data The data to be used to update the object if found.
 * @return The index of the object, if a Differentiable by that
 * name is already recorded in the DifferentiableManager and
 * <code>-1</code> the Differentiable is not found.
 */
public int update( String name,ArrayList<String>data ){
    try {
        Differentiable diff = getDifferentiable(name);
        diff.setData(data);
   int index = indexOf(diff.getName());
       DIFFERENTIABLES.set(index, diff);
       return index;
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(DifferentiableManager.class.getName()).log(Level.SEVERE, null, ex);
            return -1;
   }
}//end method


public void clearAll(){
    DIFFERENTIABLES.clear();
}

        


}//end inner class. 