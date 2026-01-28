/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix;

import java.util.ArrayList;

import util.Utils;

import math.matrix.expressParser.MatrixValueParser;
import math.matrix.expressParser.MatrixVariable;
import parser.STRING;


/**
 * Objects of this class have data storing capability and store this data in the form of Constant( Once created its value cannot be changed,
 * neither can they be duplicated in an object of this class.) and Variable objects.
 * The class seres as storage and manages various desirable properties of the class.
 * e.g Display of data on tables, creation, deletion, guarding against de-naturing( of Constant objects by making their values
 * immutable.)
 *
 *The class is created for working with the parser and the releveant calculating ware that will use it.
 * The parser will check with objects of this class (maintained by the calculator as a field)
 * for the value of its variables and place the value in its input before proceeding with parsing operations.
 * This implementation of the class uses a Timer object to remove earlier duplicates of Variable objects that it stores
 * at regular intervals.
 * The definitiion of a duplicate Variable here is that which any calculator user will automatically attach to it.i.e
 * A symbol represented with a particular character sequence to which any value can be assigned.
 * The current value stored by the Variable is the last value assigned to it.
 *
 * In order to avoid confusion and prevent an over-growth of the size of the Variable store,
 * Later duplicates must be removed periodically or just before the user performs important operations
 * like viewing the Variable database, storing the Variable objects on a persistent storage device,
 * closing the program e.t.c.
 *
 * A timer object can do this (remove later duplicates)automatically or the programmer can watch for actions that might require
 * the user or the program to perform the above specified operations and manually remove the duplicates.
 *
 * If you do not wish this action to be automatic please either you remove the object entirely or set the
 * delay to a very large value say 2 billion, or Integer.MAX_VALUE or create objects of this class with
 * the no-argument constructor of the class, which does not start the timer object.
 * @author GBEMIRO
 *
 */
public class MatrixVariableManager {
    /*This storage contains the variable objects.
     *It can contain duplicate values of a Variable,
     * This is due to the space-time trade-off.Here we go for time in preference to space.
     * This means that we use more space so that we can quickly get the value of a given Variable.
     * To use less space we would need to use a double for loop to compare the name of each Variable object
     * with the names of other Variable objects in the storage. This would take more time but use less space.
     *
     * However in this device, speed is essential and so we use a larger space by placing Variables having the same name
     * in the store.But we use a floating-mechanism to recognize the current values.That is we place the current Variable
     * object at the beginning of the store.
     *
     * For secondary storage or before displaying the contents of the store however,
     * we have the time to remove duplicate Variables.
     *
     */
private final ArrayList<MatrixVariable>varStore=new ArrayList<MatrixVariable>(100);

private int interval;
private boolean intervalChanged=false;
private static boolean validCommandString=true;

/**
 *
 */
public MatrixVariableManager(){

}
/**
 * Creates a new VariableManager object with a given time interval in which it searches through its
 * variable store and removes all duplicate elements.
 * @param interval the interval between which the object automatically refreshes its Variable store
 */
public MatrixVariableManager(int interval){
    this.interval=interval;
}

    public int getInterval() {
        return interval;
    }

    public void setInterval(int interval) {
        if(interval!=this.interval){
            setIntervalChanged(true);
        }
        this.interval = interval;
    }

    public boolean isIntervalChanged() {
        return intervalChanged;
    }

    public void setIntervalChanged(boolean intervalChanged) {
        this.intervalChanged = intervalChanged;
    }

    public static boolean isValidCommandString() {
        return MatrixVariableManager.validCommandString;
    }

    public static void setValidCommandString(boolean validCommandString) {
        MatrixVariableManager.validCommandString = validCommandString;
    }






/**
 *
 * @param index the Variable object located at the given index.
 * @return the Variable object located at the index.
 */
public MatrixVariable getVarFromIndex(int index){
    return varStore.get(index);
}


/**
 *
 * @param name the name of the Variable object whose index in the manager is desired.
 * @return the index of the Variable if found and -1 if not.
 */
public int getVarIndexFromVarName(String name){
    int index=-1;
   for(int i=0;i<varStore.size();i++){
       if(varStore.get(i).getName().equals(name)){
       index=i;
       break;
       }
   }//end for
    return index;
}




/**
 *
 * @param name the name of the Variable.
 * @return true if the manager contains the Variable that goes by this name.
 */
public boolean contains(String name){
  return getVarIndexFromVarName(name)!=-1;
}



 

/**
 *
 * @param cmd The command used to create the variable...must
 * have the format....variable name = matrix value. e.g..#A = [123,43,4,54,5,55,5:87,43,4,5,6,7,89:12,32,1,4,5,3,56:]
 *
 * The import of this is that constants cannot
 * be create in this manner.
 * @return the created Variable
 * @throws NullPointerException if the command syntax is wrong.
 */
private MatrixVariable createMatrixVariable(String cmd) throws NullPointerException{
cmd = STRING.purifier(cmd);
    MatrixValueParser parser = new MatrixValueParser(cmd.substring(cmd.indexOf("["))); 

    return new MatrixVariable( cmd.substring(0,cmd.indexOf("="))  , parser.getMatrix()  );
}//end method.
/**
 *
 * @param cmd The command used to create and store the variable
 */
public void storeVariable(String cmd){
    try{
    MatrixVariable var = createMatrixVariable(cmd);
    varStore.add(0,var);
removeDuplicates();
    }
    catch(NullPointerException nolian){
       Utils.logError( "Variable Creation Syntax Error. ERROR IN SYNTAX" );
    }
}

/**
 * scans the storage for Variables with similar names and removes all earlier occurences (found farther down the storage)
 * from the storage.Note that later or more recent versions of Variables are found towards the beginning of the storage.
 */



public void removeDuplicates(){
      ArrayList<MatrixVariable>constCleaner=new ArrayList<MatrixVariable>();//stores the Variables that match in name only, but not in value.
for(int j=0;j<varStore.size();j++){
 for(int i=j;i<varStore.size();i++){
     MatrixVariable var1=varStore.get(j);
     MatrixVariable var2=varStore.get(i);


    if(var1.getName().equals(var2.getName())&&!(j>=i)){
   constCleaner.add(varStore.get(i));
    }

}//end inner for
}//end outer for
    varStore.removeAll(constCleaner);

}//end method removeDuplicateConstants


/**
 *
 * @return the storage object of this class
 */
    public ArrayList<MatrixVariable> getVarStore() {
        return varStore;
    }
/**
 * set the storage to the new one specified by the parameter of this method
 * @param varStore a new storage to be used as the storage for objects of this class
 */
    public void setVarStore(ArrayList<MatrixVariable> varStore) {
        this.varStore.clear();
        this.varStore.addAll(varStore);
    }





/**
 * deletes a Variable or constant whose name is known
 * @param varName the name of the Variable object to be deleted
 */
public void deleteVar(String varName){
 try{
        for(int i=0;i<varStore.size();i++){
            if(varStore.get(i).getName().equals(varName)){
varStore.remove(i);
                break;
            }
        }//end for

}//end try
catch(IndexOutOfBoundsException indErr){
    util.Utils.logError("Variable "+varName +" Does Not Exist.");
}

}//end method
/**
 *deletes a Variable or constant whose location in varStore is known
 * @param index the index of the Variable object to be deleted
 */
public void deleteVar(int index){
    try{
varStore.remove(index);
     }//end try
catch(IndexOutOfBoundsException indErr){
    util.Utils.logError("No Variable Exists At Location "+(index));
}


}//end method



/**
 * Clears all Variables
 */

public void clearVariables(){
    ArrayList<MatrixVariable>varDeleter=new ArrayList<MatrixVariable>();
    for(int i=0;i<varStore.size();i++){
        varDeleter.add(varStore.get(i));
    }
    varDeleter.clear();
}

/**
 * Clears All Variables and Constants
 */
public void clearVariablesAndConstants(){
    varStore.clear();
}


 


















public static void main(String args[]){
MatrixVariableManager mgr = new MatrixVariableManager(5);
mgr.storeVariable("#dc=[-90,-20,23,12,3.0:2,1,3,4,12:,12,34,56,32,12:12,23,1,25,98:12,21,3,1,4:]");
System.out.println(mgr.getVarStore());
}















}//end class VariableManager
