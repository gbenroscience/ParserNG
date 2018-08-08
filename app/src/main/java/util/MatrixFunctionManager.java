/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

import java.util.ArrayList;
 
import math.matrix.expressParser.MatrixFunction;

/**
 *
 * @author GBEMIRO
 */
public class MatrixFunctionManager {
private ArrayList<MatrixFunction> functions=new ArrayList<MatrixFunction>();

/**
 *
 * @return all function objects stored by this FunctionManager object
 */
    public ArrayList<MatrixFunction> getFunctions() {
        return functions;
    }
/**
 *
 * @param functions the store of Function objects that this FunctionManager object will store
 */
    public void setFunctions(ArrayList<MatrixFunction> functions) {
        this.functions = functions;
    }

/**
 * checks if the Function object passed to it as an argument can be optimized or not
 * @param name
 * @return true if the Function is optimizable
 */
    public boolean canOptimizeFunction(String name){
    return getFunctionByName(name)!=null;
    }

/**
 * stores a Function in objects of this class.
 * The storage is done in such a way that the object is inserted
 * from the front.
 * @param function the new Function to store
 */
    public void storeFunction(MatrixFunction function){
        if(!contains(function)){
        functions.add(0,function);
        }
    }
/**
 *
 * @param func the Function object to search for
 * @return true if the Function object is found
 */
    public boolean contains(MatrixFunction func){
        return functions.indexOf(func)!=-1;
    }


public MatrixFunction getFunction(MatrixFunction function){
    return functions.get(functions.indexOf(function));
}


/**
 *
 * @param index the location from which we wish to retrieve the Function object
 * @return the Function object stored at the specified index.
 */
    public MatrixFunction getFunctionAt(int index){
        if(index>=0&&index<functions.size()){
            return functions.get(index);
        }
        else{
            throw new ArrayIndexOutOfBoundsException("Function Store Access Error");
        }
    }//end method getFunctionAt
    /**
     *
     * @param funcName the String representation of the Function object
     * @return the Function object in the store that goes by that name.
     * @throws NullPointerException
     */
public MatrixFunction getFunctionByName(String funcName) throws NullPointerException{
    int size=functions.size();
    for(int i=0;i<size;i++){
        if(functions.get(i).getExpression().equals(funcName)){
            return functions.get(i);
        }
    }//end for
return null;
}//end method getFunctionByName

/**
 *
 * @param scanner the scanner object to search for in the store
 * @return the Function object in the store that possesses that scanned form
 */
public MatrixFunction getFunctionByScanner(ArrayList<String>scanner){
    int size=functions.size();
    for(int i=0;i<size;i++){
        if(functions.get(i).getScanner()==scanner){
            return functions.get(i);
        }
    }//end for
return null;

}
/**
 * removes the Function object at the specified location
 * @param index the location from which the Function is to be removed
 */
public void removeFunctionAt(int index){
    if(contains(functions.get(index))){
      functions.remove(index);
    }
}
    /**
     * Removes the Function object in the store that goes by that name
     * @param funcName the String representation of the Function object
     */
public void removeFunctionByName(String funcName){
    int size=functions.size();
    for(int i=0;i<size;i++){
        if(functions.get(i).getExpression().equals(funcName)){
            functions.remove(i);
        }//end if
    }//end for
}//end method removeFunctionByName

/**
 *removes the Function object that has that scanner
 * @param scanner the scanner object to search for in the store
 */
public void removeFunctionByScanner(ArrayList<String>scanner){
    int size=functions.size();
    for(int i=0;i<size;i++){
        if(functions.get(i).getScanner()==scanner){
            functions.remove(i);
        }
    }//end for


}//end method removeFunctionByScanner

/**
 *
 * @param name the name or String format of the Function object to  be optimized
 * @return the optimized Function object such that it is given its attributes already processed and ready for use
 */
public MatrixFunction optimizeFunction(String name){
return getFunctionByName(name);
}







}//end class FunctionManager

