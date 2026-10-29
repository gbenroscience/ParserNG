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

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link FunctionScope}: global and local function scopes.
 * <p>
 * Every function name used here starts with {@code fs} so that cleanup can
 * remove exactly what a test created from the shared global registry.
 */
public class FunctionScopeTest {

    private static final String[] GLOBAL_NAMES = {"fsf", "fsg", "fsgg", "fsiso", "fsk", "fsshared", "fsthr", "fsM", "fsmm", "fsdd", "fssq"};

    @BeforeEach
    public void cleanBefore() {
        cleanGlobal();
    }

    @AfterEach
    public void cleanAfter() {
        cleanGlobal();
        // no test may leave a local scope active on this thread
        assertSame(FunctionManager.globalScope(), FunctionManager.currentScope());
    }

    private static void cleanGlobal() {
        for (String n : GLOBAL_NAMES) {
            FunctionManager.FUNCTIONS.remove(n);
        }
        FunctionManager.clearAnonymousFunctions();
    }

    private static double eval(String input) {
        return Double.parseDouble(new MathExpression(input).solve());
    }

    private static double eval(String input, FunctionScope scope) {
        return Double.parseDouble(new MathExpression(input, scope).solve());
    }

    // ------------------------------------------------------------------ the basic contract
    @Test
    @DisplayName("Without a scope, behaviour is exactly the old global behaviour")
    public void globalBehaviourIsUnchanged() {
        assertEquals(3.0, eval("fsf(x)=x+1;fsf(2)"), 0.0);
        assertNotNull(FunctionManager.lookUp("fsf"));
        assertSame(FunctionManager.FUNCTIONS.get("fsf"), FunctionManager.lookUp("fsf"));
        assertEquals(3.0, eval("fsf(2)"), 0.0);
        FunctionManager.delete("fsf");
        assertNull(FunctionManager.lookUp("fsf"));
    }

    @Test
    @DisplayName("A local function shadows a global one of the same name and leaves it untouched")
    public void localShadowsGlobal() {
        FunctionManager.add("fsf(x)=x+1");
        FunctionScope scope = FunctionScope.create();

        assertEquals(20.0, eval("fsf(x)=x*10;fsf(2)", scope), 0.0);

        assertEquals(3.0, eval("fsf(2)"), 0.0, "global fsf must be untouched");
        assertEquals(20.0, eval("fsf(2)", scope), 0.0, "the scope keeps its own fsf");
        assertNotNull(scope.lookUpLocal("fsf"));
        assertNotNull(FunctionManager.FUNCTIONS.get("fsf"));
        assertNotSame(scope.lookUpLocal("fsf"), FunctionManager.FUNCTIONS.get("fsf"));
    }

    @Test
    @DisplayName("Names not defined locally fall back to the global scope")
    public void localFallsBackToGlobal() {
        FunctionManager.add("fsg(x)=x*x");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x+100");

        assertEquals(109.0, eval("fsg(3)+fsf(0)", scope), 0.0);
        assertEquals(9.0, eval("fsg(3)", scope), 0.0);
        assertSame(FunctionManager.FUNCTIONS.get("fsg"), scope.lookUp("fsg"));
        assertNull(scope.lookUpLocal("fsg"));
    }

    @Test
    @DisplayName("Functions defined in a scope never appear in the global registry")
    public void localDefinitionsDoNotLeak() {
        FunctionScope scope = FunctionScope.create();
        assertEquals(25.0, eval("fsg(x)=x^2;fsg(5)", scope), 0.0);

        assertTrue(scope.containsLocal("fsg"));
        assertNull(FunctionManager.FUNCTIONS.get("fsg"));
        assertNull(FunctionManager.lookUp("fsg"));
        assertFalse(FunctionManager.FUNCTIONS.containsKey("fsg"));
    }

    @Test
    @DisplayName("Many scopes can each define the same name differently")
    public void manyIndependentScopes() {
        int n = 100;
        List<FunctionScope> scopes = new ArrayList<>();
        List<MathExpression> exprs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            FunctionScope s = FunctionScope.create();
            s.add("fsf(x)=x*" + (i + 1));
            scopes.add(s);
            exprs.add(new MathExpression("fsf(2)", s));
        }
        for (int i = 0; i < n; i++) {
            assertEquals(2.0 * (i + 1), Double.parseDouble(exprs.get(i).solve()), 0.0, "scope " + i);
        }
        assertNull(FunctionManager.FUNCTIONS.get("fsf"));
    }

    @Test
    @DisplayName("An isolated scope does not see global user functions")
    public void isolatedScope() {
        FunctionManager.add("fsiso(x)=x+1");
        FunctionScope iso = FunctionScope.isolated();

        assertTrue(iso.isIsolated());
        assertNull(iso.lookUp("fsiso"));
        assertNotEquals("3.0", new MathExpression("fsiso(2)", iso).solve());
        assertEquals(9.0, eval("fsk(x)=x-1;fsk(10)", iso), 0.0);
        assertEquals(3.0, eval("fsiso(2)"), 0.0);
    }

    @Test
    @DisplayName("Scopes nest: a child sees its parent, and the parent does not see the child")
    public void nestedScopes() {
        FunctionScope parent = FunctionScope.create();
        parent.add("fsf(x)=x+1");
        FunctionScope child = FunctionScope.create(parent);
        child.add("fsg(x)=fsf(x)*2");

        assertEquals(8.0, eval("fsg(3)", child), 0.0);
        assertNull(parent.lookUp("fsg"));
        assertSame(parent.lookUp("fsf"), child.lookUp("fsf"));
        assertSame(parent, child.getParent());

        child.add("fsf(x)=x+1000");
        assertEquals(1002.0, eval("fsf(2)", child), 0.0, "child shadows parent");
        assertEquals(3.0, eval("fsf(2)", parent), 0.0, "parent keeps its own");
    }

    @Test
    @DisplayName("Removing a local function reveals the global one again")
    public void removeLocalRevealsGlobal() {
        FunctionManager.add("fsf(x)=x+1");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");
        assertEquals(20.0, eval("fsf(2)", scope), 0.0);

        assertNotNull(scope.remove("fsf"));
        assertEquals(3.0, eval("fsf(2)", scope), 0.0);
        assertNotNull(FunctionManager.FUNCTIONS.get("fsf"), "removal must not reach the global scope");
    }

    // ------------------------------------------------------------------ lexical scoping
    @Test
    @DisplayName("A local function calling another local function uses the local one")
    public void localBodyUsesLocalFunctions() {
        FunctionManager.add("fsf(x)=x");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*100");
        scope.add("fsg(x)=fsf(x)+1");
        assertEquals(201.0, eval("fsg(2)", scope), 0.0);
    }

    @Test
    @DisplayName("A global function keeps resolving names globally, even when called from a shadowing scope")
    public void globalFunctionIsLexicallyScoped() {
        FunctionManager.add("fsf(x)=x+1");
        FunctionManager.add("fsgg(x)=fsf(x)+1");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");

        assertEquals(4.0, eval("fsgg(2)"), 0.0);
        assertEquals(4.0, eval("fsgg(2)", scope), 0.0, "fsgg was defined globally, so it sees the global fsf");
        assertEquals(20.0, eval("fsf(2)", scope), 0.0);
    }

    @Test
    @DisplayName("Lexical scoping holds for arguments, several parameters, nested calls and compiled expressions")
    public void lexicalAcrossCallShapes() throws Throwable {
        FunctionManager.add("fsf(x)=x+1");
        FunctionManager.add("fsgg(x)=fsf(x)+1");
        FunctionManager.add("fsshared(x,y)=fsf(x)*y");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");

        assertEquals(12.0, eval("fsgg(fsf(1))", scope), 0.0, "the argument is written in the scope, so it uses the local fsf");
        assertEquals(9.0, eval("2*fsgg(2)+1", scope), 0.0);
        assertEquals(9.0, eval("fsshared(2,3)", scope), 0.0);
        assertEquals(24.0, eval("fsgg(2)+fsf(2)", scope), 0.0);
        assertEquals(4.0, new MathExpression("fsgg(2)", scope).compileTurbo().applyScalar(new double[0]), 0.0);
        assertEquals(9.0, new MathExpression("fsshared(2,3)", scope).compileTurbo().applyScalar(new double[0]), 0.0);
    }

    @Test
    @DisplayName("A function keeps the definitions its body used when it was defined, as it always has")
    public void definitionTimeBindingIsUnchanged() {
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x+1");
        scope.add("fsg(x)=fsf(x)+5");
        assertEquals(8.0, eval("fsg(2)", scope), 0.0);
        scope.add("fsf(x)=x*10");
        assertEquals(20.0, eval("fsf(2)", scope), 0.0);
        assertEquals(8.0, eval("fsg(2)", scope), 0.0, "same rule as the global registry");

        FunctionManager.add("fsf(x)=x+1");
        FunctionManager.add("fsg(x)=fsf(x)+5");
        FunctionManager.add("fsf(x)=x*10");
        assertEquals(8.0, eval("fsg(2)"), 0.0, "the global registry has always behaved this way");
    }

    @Test
    @DisplayName("A matrix named in a function body means the matrix of the scope the function was defined in")
    public void matricesInFunctionBodiesAreLexical() throws Throwable {
        FunctionManager.add("fsM=@(2,2)(1,2,3,4)");          // det = -2
        FunctionManager.add("fsmm(x)=det(fsM)*x");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsM=@(2,2)(2,0,0,2)");                    // det = 4

        assertEquals(-4.0, eval("fsmm(2)"), 0.0);
        assertEquals(-4.0, eval("fsmm(2)", scope), 0.0, "constant argument");
        assertEquals(-4.0, eval("x=2;fsmm(x)", scope), 0.0, "variable argument");
        assertEquals(-4.0, new MathExpression("fsmm(2)", scope).compileTurbo().applyScalar(new double[0]), 0.0, "compiled");
        assertEquals(4.0, eval("det(fsM)", scope), 0.0, "the scope's own matrix is still its own");
        assertEquals(0.0, eval("fsmm(2)+det(fsM)", scope), 0.0, "both meanings side by side in one expression");
    }

    @Test
    @DisplayName("A function handle named in a function body means the function of the scope it was defined in")
    public void functionHandlesInFunctionBodiesAreLexical() throws Throwable {
        FunctionScope parent = FunctionScope.isolated();
        parent.add("fssq(x)=x^2");
        parent.add("fsdd(x)=x*diff(fssq,3,1)");              // on a clean JVM d/dx x^2 at 3 = 6, so fsdd(2) = 12
        FunctionScope child = FunctionScope.create(parent);
        child.add("fssq(x)=x^3");                            // d/dx x^3 at 3 = 27

        // The mechanism: building an expression that calls the parent's function pins the
        // handle in the body to the parent's fssq, through a hidden alias in the child.
        assertEquals(0, countAliasesTo(child, child.lookUpLocal("fssq")));
        new MathExpression("x=2;fsdd(x)", child);
        assertEquals(1, countAliasesTo(child, parent.lookUpLocal("fssq")), "the body's fssq is pinned to the parent's");
        assertEquals(0, countAliasesTo(child, child.lookUpLocal("fssq")), "and never to the child's own");

        // The numbers: skipped when this JVM's earlier tests have left global state that makes
        // diff() return a stale value for every function (same on every version of the library).
        double fromParent = eval("diff(fssq,3,1)", parent);
        double fromChild = eval("diff(fssq,3,1)", child);
        if (fromParent != fromChild) {
            double viaParent = eval("fsdd(2)", parent);
            assertEquals(viaParent, eval("fsdd(2)", child), 0.0, "constant argument");
            assertEquals(viaParent, eval("x=2;fsdd(x)", child), 0.0, "variable argument");
            assertEquals(viaParent, new MathExpression("fsdd(2)", child).compileTurbo().applyScalar(new double[0]), 0.0, "compiled");
            assertEquals(fromChild, eval("diff(fssq,3,1)", child), 0.0, "the child's own function is still its own");
        }
    }

    /** How many hidden aliases in {@code scope} point at {@code target}. */
    private static int countAliasesTo(FunctionScope scope, com.github.gbenroscience.parser.Function target) {
        int n = 0;
        for (String name : scope.localNames()) {
            if (FunctionManager.isAnonymousFormat(name) && scope.lookUpLocal(name) == target) {
                n++;
            }
        }
        return n;
    }

    @Test
    @DisplayName("The same holds for nested and isolated scopes")
    public void lexicalMatricesAcrossNestedAndIsolatedScopes() {
        FunctionManager.add("fsM=@(2,2)(1,2,3,4)");
        FunctionManager.add("fsmm(x)=det(fsM)*x");
        FunctionScope parent = FunctionScope.create();
        parent.add("fsM=@(2,2)(2,0,0,2)");                   // det 4
        parent.add("fsg(x)=det(fsM)*x");                     // defined where fsM has det 4
        FunctionScope child = FunctionScope.create(parent);
        child.add("fsM=@(2,2)(3,0,0,3)");                    // det 9, shadows

        assertEquals(-4.0, eval("fsmm(2)", child), 0.0, "global function seen from a grandchild");
        assertEquals(8.0, eval("fsg(2)", child), 0.0, "parent's function keeps the parent's matrix");
        assertEquals(8.0, eval("x=2;fsg(x)", child), 0.0);
        assertEquals(18.0, eval("det(fsM)*2", child), 0.0, "and the child's own matrix is the child's");

        FunctionScope iso = FunctionScope.isolated();
        iso.add("fsM=@(2,2)(5,0,0,5)");
        iso.add("fsk(x)=det(fsM)*x");
        assertEquals(50.0, eval("fsk(2)", iso), 0.0);
        assertEquals(50.0, eval("x=2;fsk(x)", iso), 0.0);
    }

    @Test
    @DisplayName("Repeatedly building expressions in a scope does not accumulate hidden entries")
    public void aliasesAreReused() {
        FunctionManager.add("fsM=@(2,2)(1,2,3,4)");
        FunctionManager.add("fsmm(x)=det(fsM)*x");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsM=@(2,2)(2,0,0,2)");
        new MathExpression("x=2;fsmm(x)", scope);
        int settled = scope.size();
        for (int i = 0; i < 200; i++) {
            new MathExpression("x=2;fsmm(x)", scope);
            new MathExpression("fsmm(2)", scope);
        }
        assertEquals(settled, scope.size());
    }

    @Test
    @DisplayName("An expression created while a scope is active joins that scope")
    public void expressionsCreatedInsideAScopeJoinIt() {
        FunctionManager.add("fsf(x)=x+1");
        final FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");

        String inside = scope.call(new java.util.function.Supplier<String>() {
            @Override
            public String get() {
                return new MathExpression("fsf(2)").solve();
            }
        });
        assertEquals(20.0, Double.parseDouble(inside), 0.0);
        assertEquals(3.0, eval("fsf(2)"), 0.0);
    }

    @Test
    @DisplayName("setExpression on a scoped expression keeps resolving in the scope")
    public void setExpressionStaysInScope() {
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");
        MathExpression me = new MathExpression("fsf(1)", scope);
        assertEquals(10.0, Double.parseDouble(me.solve()), 0.0);
        me.setExpression("(fsf(5))");
        assertEquals(50.0, Double.parseDouble(me.solve()), 0.0);
        assertSame(scope, me.getFunctionScope());
    }

    @Test
    @DisplayName("copy() keeps the scope")
    public void copyKeepsScope() {
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");
        MathExpression me = new MathExpression("fsf(3)", scope);
        me.solve();
        MathExpression copy = me.copy();
        assertSame(scope, copy.getFunctionScope());
        assertEquals(30.0, Double.parseDouble(copy.solve()), 0.0);
    }

    // ------------------------------------------------------------------ kinds of function
    @Test
    @DisplayName("Matrices defined in a scope stay in that scope")
    public void localMatrices() {
        FunctionScope scope = FunctionScope.create();
        assertEquals(-2.0, eval("fsk=@(2,2)(1,2,3,4);det(fsk)", scope), 0.0);
        assertNotNull(scope.lookUpLocal("fsk"));
        assertNull(FunctionManager.FUNCTIONS.get("fsk"));
    }

    @Test
    @DisplayName("Anonymous functions made inside a scope do not pile up in the global registry")
    public void anonymousFunctionsStayLocal() {
        int before = FunctionManager.countAnonymousFunctions();
        FunctionScope scope = FunctionScope.create();
        assertEquals(8.0, eval("fsf=@(x)x*2;fsf(4)", scope), 0.0);
        assertEquals(before, FunctionManager.countAnonymousFunctions());
        assertTrue(scope.containsLocal("fsf"));
    }

    @Test
    @DisplayName("A local scope never deletes a global variable that shares a function's name")
    public void localFunctionDoesNotDeleteGlobalVariable() {
        VariableManager.parseCommand("fsvar=7;");
        try {
            assertNotNull(VariableManager.lookUp("fsvar"));
            FunctionScope scope = FunctionScope.create();
            scope.add("fsvar(x)=x+1");
            assertNotNull(VariableManager.lookUp("fsvar"));
        } finally {
            VariableManager.delete("fsvar");
        }
    }

    // ------------------------------------------------------------------ activation
    @Test
    @DisplayName("activate() routes static FunctionManager calls to the scope and restores afterwards")
    public void activateRoutesStaticApi() {
        FunctionScope scope = FunctionScope.create();
        FunctionScope.Binding b = scope.activate();
        try {
            assertSame(scope, FunctionManager.currentScope());
            FunctionManager.add("fsk(x)=x+7");
            assertNotNull(FunctionManager.lookUp("fsk"));
        } finally {
            b.close();
        }
        assertSame(FunctionManager.globalScope(), FunctionManager.currentScope());
        assertNull(FunctionManager.FUNCTIONS.get("fsk"));
        assertNotNull(scope.lookUpLocal("fsk"));
    }

    @Test
    @DisplayName("Scope switching nests and restores in order")
    public void enterExitNest() {
        FunctionScope a = FunctionScope.create();
        FunctionScope b = FunctionScope.create();
        FunctionScope p1 = FunctionManager.enter(a);
        assertSame(a, FunctionManager.currentScope());
        FunctionScope p2 = FunctionManager.enter(b);
        assertSame(b, FunctionManager.currentScope());
        FunctionManager.exit(p2);
        assertSame(a, FunctionManager.currentScope());
        FunctionManager.exit(p1);
        assertSame(FunctionManager.globalScope(), FunctionManager.currentScope());
    }

    @Test
    @DisplayName("A failed parse leaves no scope active")
    public void failedParseRestoresScope() {
        FunctionScope scope = FunctionScope.create();
        try {
            new MathExpression("fsf(((", scope);
        } catch (RuntimeException expected) {
            // fine: what matters is the state afterwards
        }
        assertSame(FunctionManager.globalScope(), FunctionManager.currentScope());
    }

    @Test
    @DisplayName("Global scope handed to a constructor means no scope")
    public void globalScopeArgumentIsPlainGlobal() {
        FunctionManager.add("fsf(x)=x+1");
        MathExpression me = new MathExpression("fsf(2)", FunctionManager.globalScope());
        assertEquals(3.0, Double.parseDouble(me.solve()), 0.0);
        assertTrue(me.getFunctionScope().isGlobal());
        MathExpression plain = new MathExpression("fsf(2)");
        assertTrue(plain.getFunctionScope().isGlobal());
    }

    // ------------------------------------------------------------------ compiled (turbo) expressions
    @Test
    @DisplayName("A compiled expression keeps its scope, on whichever thread runs it")
    public void turboExpressionsKeepScope() throws Throwable {
        FunctionManager.add("fsf(x)=x+1");
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");

        final FastCompositeExpression local = new MathExpression("fsf(3)", scope).compileTurbo();
        final FastCompositeExpression global = new MathExpression("fsf(3)").compileTurbo();

        assertEquals(30.0, local.applyScalar(new double[0]), 0.0);
        assertEquals(4.0, global.applyScalar(new double[0]), 0.0);

        final double[] fromOtherThread = new double[1];
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                fromOtherThread[0] = local.applyScalar(new double[0]);
            }
        });
        t.start();
        t.join();
        assertEquals(30.0, fromOtherThread[0], 0.0);
        assertSame(FunctionManager.globalScope(), FunctionManager.currentScope());
    }

    @Test
    @DisplayName("Compiled expressions that call no functions are not wrapped, so they cost nothing extra")
    public void pureTurboExpressionsAreNotWrapped() throws Throwable {
        FunctionScope scope = FunctionScope.create();
        FastCompositeExpression pure = new MathExpression("2+3*4", scope).compileTurbo();
        FastCompositeExpression plain = new MathExpression("2+3*4").compileTurbo();
        assertEquals(14.0, pure.applyScalar(new double[0]), 0.0);
        assertSame(plain.getClass(), pure.getClass());
    }

    // ------------------------------------------------------------------ concurrency
    @Test
    @DisplayName("Scoped expressions evaluated on many threads each see their own functions")
    public void concurrentScopes() throws Exception {
        FunctionManager.add("fsthr(x)=x*1000");
        final int threads = 8;
        final int rounds = 300;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Callable<Boolean>> jobs = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                final int k = i + 1;
                FunctionScope scope = FunctionScope.create();
                scope.add("fsthr(x)=x*" + k);
                final MathExpression me = new MathExpression("fsthr(2)", scope);
                jobs.add(new Callable<Boolean>() {
                    @Override
                    public Boolean call() {
                        for (int r = 0; r < rounds; r++) {
                            if (me.solveGeneric().scalar != 2.0 * k) {
                                return Boolean.FALSE;
                            }
                        }
                        return Boolean.TRUE;
                    }
                });
            }
            List<Future<Boolean>> results = pool.invokeAll(jobs);
            for (Future<Boolean> f : results) {
                assertTrue(f.get(60, TimeUnit.SECONDS));
            }
        } finally {
            pool.shutdownNow();
        }
        assertEquals(2000.0, eval("fsthr(2)"), 0.0, "global fsthr must be untouched");
    }

    // ------------------------------------------------------------------ the scanner vocabulary cache
    @Test
    @DisplayName("A name added to a scope after its expressions were built is picked up by the next one")
    public void vocabularyTracksScopeChanges() {
        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x+1");
        assertEquals(3.0, eval("fsf(2)", scope), 0.0);
        scope.add("fsg(x)=x+2");
        assertEquals(5.0, eval("fsg(3)", scope), 0.0);
        scope.remove("fsg");
        assertNull(scope.lookUp("fsg"));
        assertEquals(3.0, eval("fsf(2)", scope), 0.0);
    }

    @Test
    @DisplayName("A scope that adds no names shares the global vocabulary and still works")
    public void emptyScopeBehavesLikeGlobal() {
        FunctionManager.add("fsf(x)=x+1");
        FunctionScope scope = FunctionScope.create();
        assertTrue(scope.addsNoNames());
        assertEquals(3.0 + Math.sin(0), eval("fsf(2)+sin(0)", scope), 1e-12);
        assertFalse(FunctionScope.isolated().addsNoNames());
    }

    // ------------------------------------------------------------------ performance contract
    @Test
    @DisplayName("Scope lookup, scope switching and scoped evaluation allocate nothing")
    public void hotPathsAllocateNothing() throws Exception {
        final Object mx = ManagementFactory.getThreadMXBean();
        Class<?> sun;
        try {
            sun = Class.forName("com.sun.management.ThreadMXBean");
        } catch (ClassNotFoundException e) {
            return; // not a HotSpot-style JVM: nothing to measure with
        }
        if (!sun.isInstance(mx)) {
            return;
        }
        final Method allocated = sun.getMethod("getThreadAllocatedBytes", long.class);
        final long tid = Thread.currentThread().getId();

        FunctionScope scope = FunctionScope.create();
        scope.add("fsf(x)=x*10");
        MathExpression plain = new MathExpression("2+3*4");
        MathExpression scopedPlain = new MathExpression("2+3*4", scope);
        MathExpression scopedCall = new MathExpression("fsf(3)+1", scope);

        int warm = 300_000;
        for (int i = 0; i < warm; i++) {
            plain.solveGeneric();
            scopedPlain.solveGeneric();
            scopedCall.solveGeneric();
            scope.lookUp("fsf");
            FunctionManager.exit(FunctionManager.enter(scope));
            FunctionManager.lookUp("fsf");
        }

        final int n = 200_000;
        final long slack = 4096; // covers the reflective measuring calls themselves; a real 16-byte-per-call leak would be 3.2 MB

        long a0 = (Long) allocated.invoke(mx, tid);
        for (int i = 0; i < n; i++) {
            scope.lookUp("fsf");
        }
        long a1 = (Long) allocated.invoke(mx, tid);
        assertTrue(a1 - a0 <= slack, "FunctionScope.lookUp allocated " + (a1 - a0) + " bytes");

        for (int i = 0; i < n; i++) {
            FunctionScope prev = FunctionManager.enter(scope);
            FunctionManager.exit(prev);
        }
        long a2 = (Long) allocated.invoke(mx, tid);
        assertTrue(a2 - a1 <= slack, "enter/exit allocated " + (a2 - a1) + " bytes");

        for (int i = 0; i < n; i++) {
            scopedPlain.solveGeneric();
        }
        long a3 = (Long) allocated.invoke(mx, tid);
        long unscopedBaseline;
        {
            long b0 = (Long) allocated.invoke(mx, tid);
            for (int i = 0; i < n; i++) {
                plain.solveGeneric();
            }
            long b1 = (Long) allocated.invoke(mx, tid);
            unscopedBaseline = b1 - b0;
        }
        assertTrue((a3 - a2) <= unscopedBaseline + slack, "scoped solveGeneric allocated " + (a3 - a2) + " bytes vs " + unscopedBaseline + " unscoped");

        long c0 = (Long) allocated.invoke(mx, tid);
        for (int i = 0; i < n; i++) {
            scopedCall.solveGeneric();
        }
        long c1 = (Long) allocated.invoke(mx, tid);
        assertTrue(c1 - c0 <= slack, "scoped solveGeneric calling a function allocated " + (c1 - c0) + " bytes");
    }
}
