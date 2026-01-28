/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

import parser.Function;
import parser.Variable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class FunctionManager {

    public static final Map<String, Function> FUNCTIONS = Collections.synchronizedMap(new HashMap<String, Function>());

    /**
     *
     * @param fName The name of the dependent variable of the function or the
     * full name of the function which is a combination of the name of its
     * dependent variable and its independent variables enclosed in circular
     * parentheses. e.g in y = x^3, either y or y(x) may be supplied.
     * @return true if a Function exists by the name supplied.
     */
    public static boolean contains(String fName) {
        return lookUp(fName) != null;
    }//end method

    /**
     *
     * @param fName The name of the dependent variable of the function.
     * @return the Function object that has the name supplied if it exists. If
     * no such Function object exists, then it returns null.
     */
    public static Function getFunction(String fName) {
        return FUNCTIONS.get(fName);
    }//end method

    /**
     * Attempts to retrieve a Function object from a FunctionManager based on
     * its name.
     *
     * @param functionName The name of the Function object.
     * @return the Function object that has that name or null if the Function is
     * not found.
     */
    public static Function lookUp(String functionName) {
        return FUNCTIONS.get(functionName);
    }//end method

    /**
     * Adds a Function object to this FunctionManager.
     *
     * @param expression The expression that creates the Function to add. The
     * form is:F=@(x,y,z,...)mathexpr. e.g y=@(x)3x-x^2; Functions take
     * precedence over variables.. so if a function called sin_func is created
     * and there exists a variable with that name, the system discards that
     * variable
     */
    public static Function add(String expression) {
        Function f = new Function(expression);
        add(f);
        return f;
    }//end method

    /**
     *
     * @param f The Function object to add to this object.
     */
    public static void add(Function f) {
        String fName = f.getName();

        Function oldFunc = FUNCTIONS.get(fName);

        if (oldFunc == null) {//function does not exist in registry
            Variable v = VariableManager.lookUp(fName);//check if a Variable has this name in the Variables registry
            if (v != null) {
                VariableManager.delete(fName);//if so delete it.
            }//end if
            FUNCTIONS.put(fName, f);
        } else {
            update(f.toString());
        }
        update();
    }//end method

    /**
     * Loads the {@link Function} objects in this {@link Map} into the
     * {@link FunctionManager#FUNCTIONS} {@link Map}.
     *
     * @param functions A {@link Map} of {@link Function} objects.
     */
    public static void load(Map<String, Function> functions) {
        load(functions, false);
    }

    /**
     * Loads the {@link Function} objects in this {@link Map} into the
     * {@link FunctionManager#FUNCTIONS} {@link Map}.
     *
     * @param functions A {@link Map} of {@link Function} objects.
     * @param clearFirst If true, then the {@link FunctionManager#FUNCTIONS} is
     * cleared first before new content is loaded into it.
     */
    public static void load(Map<String, Function> functions, boolean clearFirst) {
        synchronized (FUNCTIONS) {
            if (clearFirst) {
                FUNCTIONS.clear();
            }

            FUNCTIONS.putAll(functions);

        }
    }

    /**
     * Removes a Function object from this FunctionManager.
     */
    public static void delete(String fName) {
        FUNCTIONS.remove(fName);
        update();
    }//end method

    /**
     * Updates a Function object in this FunctionManager.
     */
    public static void update(String expression) {
        try {
            Function f = new Function(expression);
            String name = f.getName();
            FUNCTIONS.put(name, f);
        } catch (Exception ex) {
            Logger.getLogger(FunctionManager.class.getName()).log(Level.SEVERE, null, ex);
        }

        update();
    }//end method

    /**
     * Deletes all anonymous functions
     */
    public static void clearAnonymousFunctions() {
        List<String> anonKeys = new ArrayList<>();
        Set<Map.Entry<String, Function>> entrySet = FUNCTIONS.entrySet();
        synchronized (entrySet) {
            for (Map.Entry<String, Function> entry : entrySet) {
                Function function = entry.getValue();
                if (function.isAnonymous()) {
                    anonKeys.add(entry.getKey());
                }
            }//end for
            FUNCTIONS.keySet().removeAll(anonKeys);
        }
    }

    /**
     *
     * @return the number of anonymous functions in the FunctionManager.
     */
    public static int countAnonymousFunctions() {
        int count = 0;
        synchronized (FUNCTIONS) {
            for (Map.Entry<String, Function> entry : FUNCTIONS.entrySet()) {
                Function function = entry.getValue();
                if (function.getName().startsWith("anon")) {
                    ++count;
                }//end if
            }//end for
        }
        return count;
    }

    /**
     *
     * @return An {@link ArrayList} of all non-anonymous functions.
     */
    public static final ArrayList<Function> getDefinedFunctions() {
        ArrayList<Function> functions = new ArrayList<>();

        synchronized (FUNCTIONS) {
            for (Map.Entry<String, Function> entry : FUNCTIONS.entrySet()) {
                Function function = entry.getValue();
                if (function != null && !function.isAnonymous()) {
                    functions.add(function);
                }
            }//end for
        }
        return functions;
    }

    /**
     * Saves stored functions and: updates the client UIs that use this manager.
     */
    public static void update() {

    }

    /**
     * Registers the parameters of all registered functions as Variables on the
     * Variable Registry.
     */
    public static void initializeFunctionVars() {
        Set<Map.Entry<String, Function>> entrySet = FUNCTIONS.entrySet();
        synchronized (FUNCTIONS) {
            for (Map.Entry<String, Function> entry : entrySet) {
                Function f = entry.getValue();
                VariableManager.add(f.getIndependentVariables().toArray(new Variable[]{}));
            }
        }
    }

}//end class
