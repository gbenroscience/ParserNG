/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

import java.util.ArrayList; 

/**
 *
 * @author GBEMIRO
 */
public class FormulaeManager {
    ArrayList<Formulae>formulaStore=new ArrayList<Formulae>();


/**
 * records formulae as they are stored by the user
 * @param expr the formula to store
 */
    public void recordFormula(String expr){
        Formulae formulae=new Formulae(expr);
        if(!hasFormula(formulae)){
        formulaStore.add( formulae ); 
        } 
    }
    /**
     * deletes Formulae objects based on their indices
     * @param index the position of the Formulae object within the store
     */
    public void deleteFormula(int index){
        formulaStore.remove(index);
    }
    /**
     * deletes Formulae objects based on their original expressions
     * @param expr The expression
     */
    public void deleteFormula(String expr){
        for(int i=0;i<formulaStore.size();i++){
            Formulae form = new Formulae(expr);
            if(formulaStore.get(i).getExpression().equals(form.getExpression())&&
                    formulaStore.get(i).getVariable().equals( form.getVariable() )  ){
                formulaStore.remove( i );
                break;
            }//end if
        }
        
    }

    /**
     *
     * @param formula
     * @return true if the Formulae manager already contains this
     * Formulae object.
     */
    public boolean hasFormula(Formulae formula){

    for(int i = 0;i<formulaStore.size();i++){
    if(  (formulaStore.get(i).getVariable()+"="+formulaStore.get(i).getExpression()).toUpperCase().equals(
            (formula.getVariable()+"="+formula.getExpression()).toUpperCase())  ){
        return true;
    }//end if


    }//end for loop
return false;
    }//end method


    /**
     * deletes all Formulae objects in the storage.
     */
    public void clearFormula(){
        formulaStore.clear();
    }






 





}
