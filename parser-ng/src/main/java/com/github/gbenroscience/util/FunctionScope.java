/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.util;

import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * A namespace for user-defined functions (and the matrices, vectors and
 * anonymous functions that ParserNG stores alongside them).
 * <p>
 * Historically every {@link MathExpression} shared one process-wide map,
 * {@link FunctionManager#FUNCTIONS}. That map is still there and is now the
 * <b>global scope</b>. A <b>local scope</b> is a second, private map that sits
 * in front of its parent:
 * <ul>
 * <li><b>Lookup</b> checks this scope first, then its parent, and so on up to
 * the global scope (unless the scope was created with {@link #isolated()}, in
 * which case the chain stops at the scope itself).</li>
 * <li><b>Definition</b> always writes into the scope that is active, never
 * into a parent. A local {@code f} therefore <i>shadows</i> a global
 * {@code f} without touching it.</li>
 * <li><b>Deletion</b> only removes from the scope that is active. A local
 * scope can never delete a global function.</li>
 * </ul>
 * <h3>Using a scope</h3>
 * <pre>{@code
 * FunctionManager.add("f(x)=x+1");              // global, as before
 *
 * FunctionScope mine = FunctionScope.create();   // child of the global scope
 * mine.add("f(x)=x*10");                         // local, shadows the global f
 *
 * new MathExpression("f(2)").solve();            // 3.0  (global f)
 * new MathExpression("f(2)", mine).solve();      // 20.0 (local f)
 * mine.solve("g(x)=x^2; g(5)");                  // 25.0, and g is local to 'mine'
 * }</pre>
 * A scope is a plain object: share one between many expressions, or give each
 * expression its own. Expressions that never name a scope behave exactly as
 * they always did.
 *
 * <h3>How it works</h3>
 * ParserNG resolves function names through the static {@link FunctionManager}
 * in ~200 places. Rather than thread a scope argument through every one of
 * them, each thread has an <i>active scope</i> (see
 * {@link FunctionManager#currentScope()}). Every static {@code FunctionManager}
 * method reads and writes the active scope. A {@link MathExpression} that
 * belongs to a scope activates it around parsing, compiling and evaluating, and
 * restores the previous scope afterwards, so scoping follows the expression
 * onto whichever thread uses it. Functions are lexically scoped: a function
 * defined in scope S always runs in S, whoever calls it.
 *
 * <h3>Limits</h3>
 * <ul>
 * <li>Only functions are scoped. Variables keep their existing model (a
 * per-expression registry backed by the global {@link VariableManager}).</li>
 * <li>Serialization does not carry the scope: a deserialized
 * {@link MathExpression} is global.</li>
 * <li>{@link FunctionManager#FUNCTIONS} remains the global map. Code that reads
 * it directly sees global functions only.</li>
 * </ul>
 *
 * @author GBEMIRO
 */
public final class FunctionScope {

    private final FunctionScope parent;
    private final Map<String, Function> functions;
    private final boolean global;
    private IdentityHashMap<Function, String> aliases; // guarded by 'this'; created on first use

    /**
     * Creates the process-wide scope over the given backing map. Only
     * {@link FunctionManager} calls this.
     */
    FunctionScope(Map<String, Function> backing) {
        this.parent = null;
        this.functions = backing;
        this.global = true;
    }

    private FunctionScope(FunctionScope parent) {
        this.parent = parent;
        // Lock-free reads: a local scope is read on every evaluation and written rarely.
        this.functions = new ConcurrentHashMap<>(8);
        this.global = false;
    }

    // ------------------------------------------------------------------ factories
    /**
     * @return the process-wide scope that backs {@link FunctionManager#FUNCTIONS}.
     */
    public static FunctionScope global() {
        return FunctionManager.globalScope();
    }

    /**
     * @return a new, empty scope whose parent is the global scope. Functions
     * not defined locally are found in the global scope.
     */
    public static FunctionScope create() {
        return create(global());
    }

    /**
     * @param parent the scope to fall back to for names not defined here. May
     * be a local scope, giving you nested scopes.
     * @return a new, empty scope that falls back to {@code parent}.
     */
    public static FunctionScope create(FunctionScope parent) {
        if (parent == null) {
            throw new IllegalArgumentException("parent must not be null; use isolated() for a scope with no fallback");
        }
        FunctionManager.markLocalScopesInUse();
        return new FunctionScope(parent);
    }

    /**
     * @return a new, empty scope with <b>no</b> fallback. Only functions
     * defined in it are visible; global user functions are not. Built-in
     * methods ({@code sin}, {@code sum}, ...) are always available.
     */
    public static FunctionScope isolated() {
        FunctionManager.markLocalScopesInUse();
        return new FunctionScope((FunctionScope) null);
    }

    // ------------------------------------------------------------------ inspection
    /**
     * @return the scope this one falls back to, or null for the global scope
     * and for isolated scopes.
     */
    public FunctionScope getParent() {
        return parent;
    }

    /**
     * @return true if this is the process-wide scope.
     */
    public boolean isGlobal() {
        return global;
    }

    /**
     * @return true if this scope has no fallback and is not the global scope.
     */
    public boolean isIsolated() {
        return !global && parent == null;
    }

    // ------------------------------------------------------------------ lookup
    /**
     * Finds a function by name: this scope first, then each parent in turn.
     *
     * @param name the function name
     * @return the nearest function with that name, or null
     */
    public Function lookUp(String name) {
        if (name == null) { // number and operator tokens have no name; ConcurrentHashMap rejects null keys
            return null;
        }
        for (FunctionScope s = this; s != null; s = s.parent) {
            Function f = s.functions.get(name);
            if (f != null) {
                return f;
            }
        }
        return null;
    }

    /**
     * @param name the function name
     * @return the function defined in <i>this</i> scope under that name,
     * ignoring parents, or null.
     */
    public Function lookUpLocal(String name) {
        return name == null ? null : functions.get(name);
    }

    /**
     * @return true if a function of that name is visible from this scope.
     */
    public boolean contains(String name) {
        return lookUp(name) != null;
    }

    /**
     * @return true if a function of that name is defined in this scope itself.
     */
    public boolean containsLocal(String name) {
        return name != null && functions.get(name) != null;
    }

    /**
     * @return the number of functions defined in this scope itself.
     */
    public int size() {
        return functions.size();
    }

    /**
     * @return true if no function is defined in this scope itself.
     */
    public boolean isEmpty() {
        return functions.isEmpty();
    }

    /**
     * @return a snapshot of the names defined in this scope itself.
     */
    public Set<String> localNames() {
        return new LinkedHashSet<>(functions.keySet());
    }

    /**
     * @return a snapshot of every name visible from this scope (its own, then
     * its parents'), without duplicates.
     */
    public String[] visibleNames() {
        return snapshotNames().names();
    }

    // ------------------------------------------------------------------ mutation
    /**
     * Defines a function in <b>this</b> scope.
     *
     * @param expression e.g. {@code "f(x)=x^2"} or {@code "f=@(x)x^2"}
     * @return the stored function, or null if the expression is invalid
     */
    public Function add(String expression) {
        FunctionScope prev = FunctionManager.enter(this);
        try {
            return FunctionManager.add(expression);
        } finally {
            FunctionManager.exit(prev);
        }
    }

    /**
     * Stores a function in <b>this</b> scope, replacing any local function of
     * the same name. For a function whose body should resolve names in this
     * scope, build it with {@link #add(String)} (or inside {@link #run}).
     *
     * @param f the function
     * @return the function that was replaced, or null
     */
    public Function add(Function f) {
        FunctionScope prev = FunctionManager.enter(this);
        try {
            return FunctionManager.add(f);
        } finally {
            FunctionManager.exit(prev);
        }
    }

    /**
     * Removes a function from <b>this</b> scope only. A function of the same
     * name in a parent scope becomes visible again.
     *
     * @param name the function name
     * @return the removed function, or null
     */
    public Function remove(String name) {
        return name == null ? null : functions.remove(name);
    }

    /**
     * Removes every function defined in this scope itself.
     */
    public void clear() {
        functions.clear();
    }

    // ------------------------------------------------------------------ activation and evaluation
    /**
     * Makes this scope the active one on the current thread until the returned
     * {@link Binding} is closed. Use it with try-with-resources to run
     * arbitrary code (including the static {@link FunctionManager} API and code
     * that predates scopes) against this scope:
     * <pre>{@code
     * try (FunctionScope.Binding b = scope.activate()) {
     *     FunctionManager.add("k(x)=x+7");   // lands in 'scope'
     * }
     * }</pre>
     *
     * @return a handle that restores the previously active scope
     */
    public Binding activate() {
        return new Binding(FunctionManager.enter(this));
    }

    /**
     * Runs the task with this scope active.
     */
    public void run(Runnable task) {
        FunctionScope prev = FunctionManager.enter(this);
        try {
            task.run();
        } finally {
            FunctionManager.exit(prev);
        }
    }

    /**
     * Runs the supplier with this scope active and returns its result.
     */
    public <T> T call(Supplier<T> task) {
        FunctionScope prev = FunctionManager.enter(this);
        try {
            return task.get();
        } finally {
            FunctionManager.exit(prev);
        }
    }

    /**
     * @param input an expression, optionally with function definitions
     * @return a new {@link MathExpression} that belongs to this scope
     */
    public MathExpression expression(String input) {
        return new MathExpression(input, this);
    }

    /**
     * Parses and evaluates the input in this scope. Equivalent to
     * {@code expression(input).solve()}.
     */
    public String solve(String input) {
        return expression(input).solve();
    }

    /**
     * Restores the previously active scope on {@link #close()}.
     */
    public static final class Binding implements AutoCloseable {

        private final FunctionScope previous;
        private boolean closed;

        Binding(FunctionScope previous) {
            this.previous = previous;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                FunctionManager.exit(previous);
            }
        }
    }

    /**
     * <b>Internal.</b> Gives a function that lives in another scope a hidden
     * name in <i>this</i> scope, and returns that name. The scanner uses it when
     * it inlines a function body into an expression of this scope: a name in the
     * body that this scope would resolve to something else (or to nothing) is
     * replaced by the alias, so the inlined tokens keep meaning what they meant
     * where the function was defined. The same function always gets the same
     * alias in a given scope, so a long-lived scope does not accumulate entries.
     *
     * @param target the function to alias
     * @return a name that resolves to {@code target} from this scope
     */
    public String aliasFor(Function target) {
        synchronized (this) {
            if (aliases == null) {
                aliases = new IdentityHashMap<>();
            }
            String alias = aliases.get(target);
            if (alias != null && functions.get(alias) == target) {
                return alias;
            }
            alias = FunctionManager.ANON_PREFIX + FunctionManager.ANON_CURSOR.incrementAndGet();
            functions.put(alias, target);
            aliases.put(target, alias);
            return alias;
        }
    }

    // ------------------------------------------------------------------ package / cache support
    /**
     * @return the map holding the functions defined in this scope itself.
     */
    Map<String, Function> ownMap() {
        return functions;
    }

    /**
     * Opaque record of the names visible from a scope at one moment, used by
     * {@code TokenTables} to know when a cached scanner vocabulary is stale.
     */
    public static final class NameSnapshot {

        private final String[][] perScope;
        private final String[] names;

        private NameSnapshot(String[][] perScope, String[] names) {
            this.perScope = perScope;
            this.names = names;
        }

        /**
         * @return every visible name, without duplicates. Shared; do not
         * modify.
         */
        public String[] names() {
            return names;
        }
    }

    /**
     * @return a snapshot of the names visible from this scope.
     */
    public NameSnapshot snapshotNames() {
        List<String[]> per = new ArrayList<>(4);
        Set<String> union = new LinkedHashSet<>();
        for (FunctionScope s = this; s != null; s = s.parent) {
            String[] keys;
            if (s.global) {
                synchronized (s.functions) { // the mutex of a Collections.synchronizedMap is the wrapper itself
                    keys = s.functions.keySet().toArray(new String[0]);
                }
            } else {
                keys = s.functions.keySet().toArray(new String[0]);
            }
            per.add(keys);
            Collections.addAll(union, keys);
        }
        return new NameSnapshot(per.toArray(new String[per.size()][]), union.toArray(new String[0]));
    }

    /**
     * @return true if the names defined in every scope of this chain are
     * exactly what they were when the snapshot was taken (same size and every
     * recorded key still present). Allocation-free.
     */
    public boolean unchangedSince(NameSnapshot snapshot) {
        int i = 0;
        for (FunctionScope s = this; s != null; s = s.parent, i++) {
            if (i >= snapshot.perScope.length) {
                return false;
            }
            String[] keys = snapshot.perScope[i];
            if (s.global) {
                synchronized (s.functions) {
                    if (!sameKeys(s.functions, keys)) {
                        return false;
                    }
                }
            } else if (!sameKeys(s.functions, keys)) {
                return false;
            }
        }
        return i == snapshot.perScope.length;
    }

    private static boolean sameKeys(Map<String, Function> map, String[] keys) {
        if (map.size() != keys.length) {
            return false;
        }
        for (String k : keys) {
            if (!map.containsKey(k)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @return true if the function names visible from this scope are exactly
     * the global ones: the chain ends at the global scope and every local scope
     * on the way is empty. Such a scope can share the global scanner
     * vocabulary. Allocation-free.
     */
    public boolean addsNoNames() {
        FunctionScope s = this;
        while (!s.global) {
            if (!s.functions.isEmpty()) {
                return false;
            }
            s = s.parent;
            if (s == null) {
                return false; // isolated: global names are not visible
            }
        }
        return true;
    }

    @Override
    public String toString() {
        if (global) {
            return "FunctionScope[global]";
        }
        return "FunctionScope[" + (parent == null ? "isolated" : "local") + ", " + functions.size() + " function(s)]";
    }
}
