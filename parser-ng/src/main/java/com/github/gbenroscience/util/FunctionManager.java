/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.util;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.Variable;
import com.github.gbenroscience.parser.turbo.QuickTime;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class FunctionManager {

    public static final String ANON_PREFIX = "anon";
    /**
     * This is an indicator of the total number of anonymous functions ever
     * created since the code was run in this session.
     */
    public static final AtomicInteger ANON_CURSOR = new AtomicInteger(0);
    public static final Map<String, Function> FUNCTIONS = Collections.synchronizedMap(new HashMap<>());

    /**
     *
     * @param fName The name of the dependent variable of the function or the
     * full name of the function which is a combination of the name of its
     * dependent variable and its independent variables enclosed in circular
     * parentheses. e.g in y = x^3, either y or y(x) may be supplied.
     * @return true if a Function exists by the name supplied, but it is not a matrix
     */
    public static boolean containsAlgebraicFunction(String fName) {
        Function f = lookUp(fName);
        return f != null && f.getType() != TYPE.MATRIX;
    }//end method
/**
 * 
 * @param fName
 * @return true if the function exists and it is matrix type
 */
    public static boolean containsMatrix(String fName) {
        Function f = lookUp(fName);
        return f != null && f.getType() == TYPE.MATRIX;
    }//end method
/**
 * 
 * @param fName
 * @return true if the function exists and is either Matrix or Algebraic
 */
    public static boolean containsAny(String fName) {
        Function f = lookUp(fName);
        return f != null;
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
     * Some actions require a handle being gotten on a function name. This
     * method allows you to acquire the name. Create a dummy function to be
     * populated with its true values later.
     *
     * @param fName
     * @param independentVars
     * @return
     */
    public static Function lockDown(String fName, String... independentVars) {
        Function f = lookUp(fName);
        if (f != null) {
            return f;
        }
        f = new Function();
        f.setDependentVariable(new Variable(fName));
        for (String d : independentVars) {
            if (Variable.isVariableString(d)) {
                Variable v = VariableManager.lookUp(d);
                if (v != null) {
                    f.getIndependentVariables().add(v);
                } else {
                    f.getIndependentVariables().add(new Variable(d));
                }
            }
        }
        if (independentVars.length == 2 && com.github.gbenroscience.parser.Number.isNumber(independentVars[0])
                && com.github.gbenroscience.parser.Number.isNumber(independentVars[1])) {
            int rows = Integer.parseInt(independentVars[0]);
            int cols = Integer.parseInt(independentVars[1]);
            Matrix m = new Matrix(rows, cols);
            m.setName(fName);
            f.setMatrix(m);
        } else if (independentVars.length == 1 && com.github.gbenroscience.parser.Number.isNumber(independentVars[0])) {
            int cols = Integer.parseInt(independentVars[0]);
            Matrix m = new Matrix(1, cols);
            m.setName(fName);
            f.setMatrix(m);
            f.setType(TYPE.VECTOR);
        } else {
            f.setMathExpression(new MathExpression());
            f.setType(TYPE.ALGEBRAIC_EXPRESSION);
        }
        FUNCTIONS.put(fName, f);
        return FUNCTIONS.get(fName);

    }//end method

    /**
     * Create a dummy function to be populated with its true values later.
     *
     * @param independentVars
     * @return
     */
    public static synchronized Function lockDownAnon(String... independentVars) {
        String fName = ANON_PREFIX + ANON_CURSOR.incrementAndGet();
        return lockDown(fName, independentVars);
    }//end method

    /**
     * Creates the anonymous copy of a Function
     *
     * @param f
     * @return
     */
    public static synchronized Function lockDownAnon(Function f) {
        String fName = ANON_PREFIX + ANON_CURSOR.incrementAndGet();
        if (f.getType() == TYPE.MATRIX || f.getType() == TYPE.VECTOR) {
            if (f.getMatrix() != null) {
                f.getMatrix().setName(fName);
            }
        }
        return FUNCTIONS.put(fName, f);
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
        try {
            
            Function f = new Function(expression); 
            String name = f.getName();
            FUNCTIONS.put(name, f);
            Function fn = FUNCTIONS.get(name);
            update();
            return fn;
        } catch (Exception ex) {
            Logger.getLogger(FunctionManager.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }//end method

    /**
     *
     * @param f The Function object to add to this object.
     */
    public static Function add(Function f) {
        String fName = f.getName();
        delete(fName);
        VariableManager.delete(fName);//if so delete it.
        Function fn = FUNCTIONS.put(fName, f);
        update();
        return fn;
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
     *
     * @param fName
     */
    public static void delete(String fName) {
        FUNCTIONS.remove(fName);
        update();
    }//end method

    /**
     * Used to update Functions by name. A great use is to promote anonymous
     * functions into named functions
     *
     * @param oldFuncName
     * @param newName
     */
    public static void update(String oldFuncName, String newName) {
        try {
            Function f = FUNCTIONS.remove(oldFuncName);
            if (f != null && f.getType() == TYPE.MATRIX) {
                f.getMatrix().setName(newName);
            }
            FUNCTIONS.put(newName, f);
        } catch (Exception ex) {
            Logger.getLogger(FunctionManager.class.getName()).log(Level.SEVERE, null, ex);
        }

        update();
    }//end method

    /**
     * Updates the Map with the most recent version of this Function.
     *
     * @param f
     */
    public static void update(Function f) {
        FUNCTIONS.put(f.getName(), f);
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

    public static final void clear() {
        FUNCTIONS.clear();
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
                if (function.getName().startsWith(ANON_PREFIX)) {
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

    public static void main(String[] args) {
        Function f = FunctionManager.lockDown("v", "x", "y", "z", "w");
        System.out.println(FUNCTIONS);
        update(new Function("v=@(x,y,z,w)3*x-2*y+z-4*w"));
        System.out.println(FUNCTIONS);
        Function func = FunctionManager.lookUp("v");
        System.out.println("Function = " + f.toString());

        int wm = 1000000;
        int n = 1000000;
//        QuickTime.benchmarkNano("func-string-args", wm, n, () -> {
//            func.evalArgs("v(2,3,4,5)");
//        });
        QuickTime.benchmarkNano("func-std-mode", wm, n, () -> {
            func.evalArgs(2, 3, 4, 5);
        });
        QuickTime.benchmarkNano("func-turbo-mode", wm, n, () -> {
            func.evalArgsTurbo(2, 3, 4, 5);
        });

    }

}//end class
