/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;
import parser.CustomScanner;
import parser.Function;
import parser.MathExpression;
import parser.STRING;
import parser.TYPE;

import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;


/**
 * Objects of this class attempt to optimize and then evaluate a math
 * expression. If the expression's MathExpression object is present in the
 * functions database (actually an ArrayList of MathExpression objects)
 * then it just retrieves the scanned and validated code
 * and solves it. Else,it must go through all the steps and then solve it.
 * This happens when:
 * 1. The expression is a new one.
 * 2. The expression has been computed before but
 * to limit memory usage, it has been deleted from memory.
 * @author GBEMIRO
 */
public class MathExpressionManager {
    private ArrayList<MathExpression> functions=new ArrayList<>();
    /**
     * Determines the maximum number of MathExpression objects that objects
     * of this class use to optimize an evaluation of an expression.
     */
    private int maxSize=200;
    /**
     * Whether to compute trigonometric functions
     * in degrees, rads or grads. Set to 0 for degrees, set to 1 for rads
     * and set to 2 for grads.
     */
    private int drgStatus=1;


    public MathExpressionManager() {
        storeFunction(new MathExpression("1+1"));
    }


    /**
     *
     * @return all function objects stored by this MathExpressionManager object
     */
    public ArrayList<MathExpression> getFunctions() {
        return functions;
    }

    /**
     *
     * @param functions the store of MathExpression objects that this MathExpressionManager object will store
     */
    public void setFunctions(ArrayList<MathExpression> functions) {
        this.functions = functions;
    }

    public int getMaxSize() {
        return maxSize;
    }

    public void setMaxSize(int maxSize) {
        this.maxSize = maxSize;
    }

    public void setDrgStatus(int drgStatus) {
        this.drgStatus = drgStatus>=0&&drgStatus<3?drgStatus:1;
        for(int i=0;i<functions.size();i++){
            functions.get(i).setDRG(drgStatus);
        }
    }//end method

    public int getDrgStatus() {
        return drgStatus;
    }

    /**
     * A workspace uses a MathExpressionManager to
     * manage and optimize functions used in calculations.
     * Each of these functions have different VariableManager
     * objects but all these objects are actually accessing, storing
     * creating and modifying their variables and constants from one
     * single Variable database( actually an ArrayList of Variable objects ) in the program.
     * This method hence allows an object of this class to return a reference to any of
     * the VariableManager objects associated with the MathExpression objects it manages,since it
     * knows that the destination database(ArrayList) is the same,i.e all the MathExpression objects
     * are only accessing the same set and copy of variables and constants.
     * @return the VariableManager object associated with the first MathExpression
     * object stored by objects of this class.
     */
    public VariableManager getVariableManager(){
        if(functions.size()>0){
            return functions.get( 0 ).getVariableManager();
        }
        else{
            throw new NullPointerException("No function created yet!");
        }
    }
    /**
     * checks if the MathExpression object passed to it as an argument can be optimized or not
     * @param expression: The expression to evaluate
     * @return true if the MathExpression is optimizable
     */
    public boolean canOptimizeFunction(String expression){
        return getFunctionByExpression(expression)!=null;
    }

    /**
     * stores a MathExpression in objects of this class.
     * The storage is done in such a way that the object is inserted
     * from the front.
     * @param function the new MathExpression to store
     */
    public void storeFunction(MathExpression function){
        if(!contains(function) && function.isCorrectFunction()){
            functions.add(function);
            int sz;
            while( (sz=functions.size()) >= maxSize){
                functions.remove(sz-1);
            }
        }
    }
    /**
     *
     * @param func the MathExpression object to search for
     * @return true if the MathExpression object is found
     */
    public boolean contains(MathExpression func){
        return getFunctionByExpression(func.getExpression())!=null;
    }//end method


    public MathExpression getFunction(MathExpression function){
        return functions.get(functions.indexOf(function));
    }


    /**
     *
     * @param index the location from which we wish to retrieve the MathExpression object
     * @return the MathExpression object stored at the specified index.
     */
    public MathExpression getFunctionAt(int index){
        if(index>=0&&index<functions.size()){
            return functions.get(index);
        }
        else{
            throw new ArrayIndexOutOfBoundsException("Function Store Access Error");
        }
    }//end method getFunctionAt
    /**
     *
     * @param expression  the String representation of the MathExpression object
     * @return the MathExpression object in the store that goes by that name or null if none is found.
     * @throws NullPointerException
     */
    public MathExpression getFunctionByExpression(String expression) throws NullPointerException{
        int size=functions.size();
        String expr = STRING.purifier(expression);
        for(int i=0;i<size;i++){
            String checkExpr = STRING.purifier( functions.get(i).getExpression() );
            if(expr.equals(checkExpr)){
                return functions.get(i);
            }
        }//end for
        return null;
    }//end method getFunctionByName

    /**
     *
     * @param scanner the scanner object to search for in the store
     * @return the MathExpression object in the store that possesses that scanned form
     */
    public MathExpression getFunctionByScanner(ArrayList<String>scanner){
        int size=functions.size();
        for(int i=0;i<size;i++){
            if(functions.get(i).getScanner()==scanner){
                return functions.get(i);
            }
        }//end for
        return null;

    }
    /**
     * removes the MathExpression object at the specified location
     * @param index the location from which the MathExpression is to be removed
     */
    public void removeFunctionAt(int index){
        if(contains(functions.get(index))){
            functions.remove(index);
        }
    }
    /**
     * Removes the MathExpression object in the store that goes by that name
     * @param funcName the String representation of the MathExpression object
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
     *removes the MathExpression object that has that scanner
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
     * @param name the name or String format of the MathExpression object to  be optimized
     * @return the optimized MathExpression object such that it is given its attributes already processed and ready for use
     */
    public MathExpression optimizeFunction(String name){
        return getFunctionByExpression(name);
    }

    /**
     *
     * @return true if this object has no
     * MathExpression objects to manage.
     */
    public boolean isEmpty(){
        return functions.isEmpty()||this==null;
    }

    /**
     *
     * @param expr The expression to be evaluated
     * by the MathExpression to be created.
     */
    public MathExpression createFunction( String expr ){
        MathExpression f = new MathExpression(expr);
        f.setDRG(1);
        storeFunction( f );
        return f;
    }//end method createFunction

    /**
     * Takes a math expression,
     * optimizes it if possible,
     * and then solves it.
     * Else if the expression cannot be optimized,
     * it has to interpret and then evaluate it.
     * It then stores the expression.
     * @param expr The expression to evaluate.
     * @return the result.
     */
    public String solve( String expr ) throws NullPointerException{
        try {
            CustomScanner cs = new CustomScanner( STRING.purifier(expr), false, VariableManager.endOfLine);

            List<String> scanned = cs.scan();
            String mathExpr = null;
            int exprCount = 0;

            for(String code : scanned) {
                if(code.contains("=")){
                    boolean success =  Function.assignObject(code+";");
                    if(!success) {
                        throw new Exception("Bad Variable or Function assignment!");
                    }
                }
                else{
                    mathExpr = code;
                    ++exprCount;
                }
            }

            MathExpression f = null;


            if (mathExpr!=null && !mathExpr.isEmpty() && exprCount == 1) {
                if ((f = getFunctionByExpression(mathExpr)) != null) {//optimize function
                    f.setDRG(drgStatus);
                }//end if
                else {
/**
 * can take care of the raw input
 * since it will still scan for variables and
 * constants in the constructor
 */
                    f = new MathExpression("("+mathExpr+")");
                    f.setDRG(drgStatus);
                    storeFunction(f);
                }//end else

            }//end if

            if (f != null) {
                return analyzeSolution(f);
            } else {
                return "0";//variable or function init or update
            }
        }
        catch(Exception e){
            e.printStackTrace();
            return "Syntax Error!";
        }
    }//end method




    public static String analyzeSolution(MathExpression expr){
        TYPE type;
        try {
            String val = expr.solve();
            type = expr.getReturnType();
            if(val.contains(",")|| val.contains("%")){
                return val;
            }
            for(int i=0;i<val.length();i++){
                String token = val.substring(i,i+1);
                if(STRING.isNumberComponent(token)){
                    break;
                }
            }

            if(type == TYPE.MATRIX){
                return val;
            }

            double v = Double.parseDouble(val);//test to see that a valid number was returned
            return val;

        }
        catch(Exception e){e.printStackTrace();
            if(e instanceof NumberFormatException || e instanceof InputMismatchException){
                return "Syntax Error!";
            }
            else if(e instanceof ArithmeticException){
                return "Valid Range Error";
            }
            else{
                return "Math Input Error";
            }
        }
    }


}//end class MathExpressionManager