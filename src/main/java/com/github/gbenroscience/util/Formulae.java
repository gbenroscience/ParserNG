/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

/**
 *
 * @author GBEMIRO
 */
import parser.Variable;
import static parser.STRING.*;
public class Formulae {
private String expression="";
private String variable="";
private boolean validFormula = true;
private String description;//describes the formula
    public Formulae(String expr){
    parseFormulae(expr) ;
    }
/**
 *
 * @return the formula's expression
 * i.e the quantity to the right of the equals sign.
 */
    public String getExpression() {
        return expression;
    }
/**
 * sets the formula or rather the expression( to the right of the equals sign )
 * to be equal to the parameter given below.
 * @param expression the formula itself i.e the quantity to the right of the equals sign.
 */
    public void setFormula(String expression) {
        this.expression = expression;
    }
/**
 *
 * @return the variable string that we set to be equal to the formula
 */
    public String getVariable() {
        return variable;
    }
/**
 * sets the variable name to the given value.
 * @param variable the dependent variable that the formulae after evaluation will
 * store its value in.
 */
    public void setVariable(String variable) {
        this.variable = variable;
    }
/**
 *
 * @return returns true if the software can model
 * a valid Formula object from the user's input
 */
    public boolean isValidFormula() {
        return validFormula;
    }
/**
 *
 * @param validFormula set this to true if the software can model
 * a valid Formula object from the user's input
 */
    public void setValidFormula(boolean validFormula) {
        this.validFormula = validFormula;
    }
/**
 *
 * @return the description of the Formula
 */
    public String getDescription() {
        return description;
    }
/**
 * Sets the description of the Formula
 * @param description the description of the Formulae
 */
    public void setDescription(String description) {
        this.description = description;
    }



/**
 *
 * @return the String representaion of the Formulae object
 */
    @Override
    public String toString(){
        return variable+"="+expression;
    }







/*
 * the format of a formula storing command is:
 * STORE:variable=expression
 */
    private void parseFormulae(String formulae){
      formulae=purifier(formulae);

      if(!formulae.substring(0,6).equals("store:")){
          setValidFormula(false);
      }
      else{
          formulae=formulae.substring(6);
           int indexOfEquals=formulae.indexOf("=");

           try{
               String var=formulae.substring(0,indexOfEquals);
        if(Variable.isVariableString( var ) ){


          
           if(indexOfEquals!=-1){
              this.variable=var;
              this.expression=formulae.substring(indexOfEquals+1);
           }//end if
         else{
             Utils.logError( "The formula lacks a dependent variable...\n" +
                     "An example of a valid formula format is  y=7*x^5.  Formula Format Error" );
         setValidFormula(false);
           }


        }//end if
        else if(!Variable.isVariableString(var)){
             Utils.logError( "The formula lacks a dependent variable...\n" +
                     "An example of a valid formula format is  y=7*x^5.  Formula Format Error" );
             setValidFormula(false);
        }

        }//end try
        catch(IndexOutOfBoundsException ind){
     Utils.logError( "Error Occured During Formula Parse:\n" +
                     "Cause:\n" +
                     "Invalid Or No Formula Variable used.\n" +
                     "Use The Format When Storing Your Formula:\n" +
                     "Formula Variable = Formula Expression. e.g:" +
                     "y = a*x^6.  Formula Variable Error" );
     setValidFormula(false);
        }



      }//end else



    }//end method parseFormulae





}
