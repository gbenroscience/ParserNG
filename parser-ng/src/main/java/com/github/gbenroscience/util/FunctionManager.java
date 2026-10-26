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
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Static access to the registry of user-defined functions.
 * <p>
 * Every method here works on the <b>active scope</b> of the calling thread. By
 * default that is the global scope, backed by {@link #FUNCTIONS}, and nothing
 * differs from earlier versions. When a {@link MathExpression} that belongs to
 * a {@link FunctionScope} is parsing, compiling or evaluating (or code runs
 * inside {@link FunctionScope#activate()}), the active scope is that local
 * scope: lookups see its functions first and then fall back to its parents,
 * while definitions and deletions affect only the local scope. See
 * {@link FunctionScope} for the full model.
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

    /**
     * The functions of the <b>global</b> scope. Functions defined in a local
     * {@link FunctionScope} are not stored here.
     */
    public static final Map<String, Function> FUNCTIONS = Collections.synchronizedMap(new HashMap<>());

    private static final FunctionScope GLOBAL_SCOPE = new FunctionScope(FUNCTIONS);

    /**
     * Per-thread slot holding the active local scope. It is one small object
     * per thread, created on first use and then only ever written to: setting
     * and restoring the scope are plain field writes, so entering and leaving a
     * scope allocates nothing (a {@code ThreadLocal.set} after a {@code remove}
     * would allocate a map entry every time).
     * <p>
     * This is the low-level, single-lookup form of
     * {@link #enter(FunctionScope)}/{@link #exit(FunctionScope)} used on the
     * evaluation hot path: fetch the slot once, then swap and restore without
     * touching the thread-local again.
     */
    public static final class Slot {

        private FunctionScope scope; // null means the global scope

        private Slot() {
        }

        /**
         * Makes {@code s} the active scope (null for global) and returns the one
         * it replaced. Pass the result to {@link #restore(FunctionScope)}.
         */
        public FunctionScope swap(FunctionScope s) {
            FunctionScope prev = scope;
            if (prev != s) {
                scope = s;
            }
            return prev;
        }

        /**
         * Reinstates a scope returned by {@link #swap(FunctionScope)}.
         */
        public void restore(FunctionScope previous) {
            scope = previous;
        }
    }

    private static final ThreadLocal<Slot> ACTIVE = new ThreadLocal<Slot>() {
        @Override
        protected Slot initialValue() {
            return new Slot();
        }
    };

    /**
     * Flips to true the first time a local {@link FunctionScope} is created.
     * Until then every scope check below is skipped, so programs that never use
     * local scopes pay nothing for them.
     */
    private static volatile boolean localScopesInUse = false;

    static void markLocalScopesInUse() {
        localScopesInUse = true;
    }

    /**
     * @return true once any local {@link FunctionScope} has been created in
     * this process. While false, every scope check is a single volatile read.
     */
    public static boolean isScopingActive() {
        return localScopesInUse;
    }

    /**
     * @return this thread's {@link Slot}, or null if no local scope has ever
     * been created (in which case there is nothing to switch).
     */
    public static Slot slot() {
        return localScopesInUse ? ACTIVE.get() : null;
    }

    // ------------------------------------------------------------------ scope plumbing
    /**
     * @return the process-wide scope, the one backed by {@link #FUNCTIONS}.
     */
    public static FunctionScope globalScope() {
        return GLOBAL_SCOPE;
    }

    /**
     * @return the scope that the static methods of this class are currently
     * using on this thread. Never null: it is the global scope unless a local
     * scope has been activated.
     */
    public static FunctionScope currentScope() {
        FunctionScope s = activeLocalScope();
        return s == null ? GLOBAL_SCOPE : s;
    }

    /**
     * @param scope a scope, or null meaning global
     * @return {@code scope}, or the global scope if it is null
     */
    public static FunctionScope scopeOrGlobal(FunctionScope scope) {
        return scope == null ? GLOBAL_SCOPE : scope;
    }

    /**
     * @return the local scope active on this thread, or null if the global
     * scope is in effect.
     */
    public static FunctionScope activeLocalScope() {
        return localScopesInUse ? ACTIVE.get().scope : null;
    }

    /**
     * Makes {@code scope} the active scope on this thread. Pass null (or the
     * global scope) to activate the global scope. Always pair with
     * {@link #exit(FunctionScope)} in a {@code finally} block:
     * <pre>{@code
     * FunctionScope prev = FunctionManager.enter(scope);
     * try { ... } finally { FunctionManager.exit(prev); }
     * }</pre>
     * {@link FunctionScope#activate()} wraps this in try-with-resources form.
     *
     * @param scope the scope to activate
     * @return the previously active local scope (null for global); give it to
     * {@link #exit(FunctionScope)}
     */
    public static FunctionScope enter(FunctionScope scope) {
        if (!localScopesInUse) {
            return null;
        }
        if (scope != null && scope.isGlobal()) {
            scope = null;
        }
        Slot slot = ACTIVE.get();
        FunctionScope prev = slot.scope;
        if (prev != scope) {
            slot.scope = scope;
        }
        return prev;
    }

    /**
     * Restores the scope that {@link #enter(FunctionScope)} replaced.
     *
     * @param previous the value returned by {@code enter}
     */
    public static void exit(FunctionScope previous) {
        if (!localScopesInUse) {
            return;
        }
        ACTIVE.get().scope = previous;
    }

    /**
     * The map that definitions and deletions go to: the active local scope's
     * own map, or the global map.
     */
    private static Map<String, Function> own() {
        FunctionScope s = activeLocalScope();
        return s == null ? FUNCTIONS : s.ownMap();
    }

    /**
     * @return a snapshot of the names of all functions visible from the active
     * scope (its own and its parents'), without duplicates.
     */
    public static String[] visibleFunctionNames() {
        FunctionScope s = activeLocalScope();
        if (s != null) {
            return s.visibleNames();
        }
        synchronized (FUNCTIONS) {
            return FUNCTIONS.keySet().toArray(new String[0]);
        }
    }

    /**
     * Stores a function in the active scope under the given name, replacing any
     * function of that name defined there.
     *
     * @param name the key
     * @param f the function
     * @return the previous function stored under that name in the active scope
     */
    public static Function put(String name, Function f) {
        return own().put(name, f);
    }

    // ------------------------------------------------------------------ queries
    /**
     *
     * @param fName The name of the dependent variable of the function or the
     * full name of the function which is a combination of the name of its
     * dependent variable and its independent variables enclosed in circular
     * parentheses. e.g in y = x^3, either y or y(x) may be supplied.
     * @return true if a Function exists by the name supplied, but it is not a
     * matrix
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
        return lookUp(fName);
    }//end method

    /**
     * Attempts to retrieve a Function object from a FunctionManager based on
     * its name. Resolves through the active scope: local first, then its
     * parents, ending at the global scope.
     *
     * @param functionName The name of the Function object.
     * @return the Function object that has that name or null if the Function is
     * not found.
     */
    public static Function lookUp(String functionName) {
        FunctionScope s = activeLocalScope();
        return s == null ? FUNCTIONS.get(functionName) : s.lookUp(functionName);
    }//end method

    /**
     * Some actions require a handle being gotten on a function name. This
     * method allows you to acquire the name. Create a dummy function to be
     * populated with its true values later.
     * <p>
     * The handle is looked for, and created, in the active scope itself, so a
     * local call never hands back (and later mutates) a parent's function.
     *
     * @param fName
     * @param independentVars
     * @return
     */
    public static Function lockDown(String fName, String... independentVars) {
        Function f = own().get(fName);
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
        Map<String, Function> target = own();
        target.put(fName, f);
        return target.get(fName);

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
        return own().put(fName, f);
    }//end method

    /**
     * Adds a Function object to the active scope.
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
            Map<String, Function> target = own();
            target.put(name, f);
            Function fn = target.get(name);
            update();
            return fn;
        } catch (Exception ex) {
            Logger.getLogger(FunctionManager.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }//end method

    /**
     * Adds a Function object to the active scope. In the global scope a
     * variable of the same name is discarded (functions take precedence over
     * variables); a local scope leaves global variables alone.
     *
     * @param f The Function object to add to this object.
     */
    public static Function add(Function f) {
        String fName = f.getName();
        delete(fName);
        if (activeLocalScope() == null) {
            VariableManager.delete(fName);//if so delete it.
        }
        Function fn = own().put(fName, f);
        update();
        return fn;
    }//end method

    /**
     * Loads the {@link Function} objects in this {@link Map} into the active
     * scope (the global {@link FunctionManager#FUNCTIONS} {@link Map} unless a
     * local scope is active).
     *
     * @param functions A {@link Map} of {@link Function} objects.
     */
    public static void load(Map<String, Function> functions) {
        load(functions, false);
    }

    /**
     * Loads the {@link Function} objects in this {@link Map} into the active
     * scope (the global {@link FunctionManager#FUNCTIONS} {@link Map} unless a
     * local scope is active).
     *
     * @param functions A {@link Map} of {@link Function} objects.
     * @param clearFirst If true, then the active scope is cleared first before
     * new content is loaded into it.
     */
    public static void load(Map<String, Function> functions, boolean clearFirst) {
        Map<String, Function> target = own();
        synchronized (target) {
            if (clearFirst) {
                target.clear();
            }

            target.putAll(functions);

        }
    }

    /**
     * Removes a Function object from the active scope. A function of the same
     * name in a parent scope is not touched.
     *
     * @param fName
     */
    public static void delete(String fName) {
        if (fName != null) {
            own().remove(fName);
        }
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
            Map<String, Function> target = own();
            Function f = target.remove(oldFuncName);
            if (f != null && f.getType() == TYPE.MATRIX) {
                f.getMatrix().setName(newName);
            }
            if (f != null) {
                target.put(newName, f);
            }
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
        own().put(f.getName(), f);
        update();
    }//end method

    /**
     * Deletes all anonymous functions of the active scope
     */
    public static void clearAnonymousFunctions() {
        Map<String, Function> target = own();
        List<String> anonKeys = new ArrayList<>();
        Set<Map.Entry<String, Function>> entrySet = target.entrySet();
        synchronized (target) {
            for (Map.Entry<String, Function> entry : entrySet) {
                Function function = entry.getValue();
                if (function.isAnonymous()) {
                    anonKeys.add(entry.getKey());
                }
            }//end for
            target.keySet().removeAll(anonKeys);
        }
    }

    /**
     * Removes every function of the active scope.
     */
    public static final void clear() {
        own().clear();
    }

    /**
     *
     * @return the number of anonymous functions in the active scope.
     */
    public static int countAnonymousFunctions() {
        int count = 0;
        Map<String, Function> target = own();
        synchronized (target) {
            for (Map.Entry<String, Function> entry : target.entrySet()) {
                Function function = entry.getValue();
                if (function.getName().startsWith(ANON_PREFIX)) {
                    ++count;
                }//end if
            }//end for
        }
        return count;
    }

    public static final boolean isAnonymousFormat(String fName) {
        if ( (fName == null) || (fName.length() < (ANON_PREFIX.length()+1)) || !fName.startsWith(ANON_PREFIX)  ) {
            return false;
        }

        for (int i = 4; i < fName.length(); i++) {
            char c = fName.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }

        return true;
    }

    /**
     *
     * @return An {@link ArrayList} of all non-anonymous functions of the active
     * scope.
     */
    public static final ArrayList<Function> getDefinedFunctions() {
        ArrayList<Function> functions = new ArrayList<>();

        Map<String, Function> target = own();
        synchronized (target) {
            for (Map.Entry<String, Function> entry : target.entrySet()) {
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
     * Registers the parameters of all functions of the active scope as
     * Variables on the Variable Registry.
     */
    public static void initializeFunctionVars() {
        Map<String, Function> target = own();
        Set<Map.Entry<String, Function>> entrySet = target.entrySet();
        synchronized (target) {
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
