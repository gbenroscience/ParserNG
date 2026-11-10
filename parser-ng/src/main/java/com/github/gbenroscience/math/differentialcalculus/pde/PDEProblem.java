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
package com.github.gbenroscience.math.differentialcalculus.pde;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Mathematical description of a PDE (or system of PDEs): the equations, the
 * unknowns, the domain, the initial conditions and the boundary conditions.
 * It says nothing about HOW the problem is solved (see {@link PDEOptions}).
 * <p>
 * Equations are ordinary ParserNG expressions of the form {@code LHS = RHS}.
 * Derivatives are written with ParserNG's {@code diff} notation:
 * <pre>
 * diff(u,t)         u_t              diff(u,x,2)      u_xx
 * diff(u,t,2)       u_tt             diff(u,x,y)      u_xy   (also diff(diff(u,x),y))
 * </pre>
 * The independent variables are the names given to {@link Builder#domain}
 * (conventionally {@code x}, {@code y}, {@code z}) and {@code t} for time.
 * Anything else that is not an unknown must be declared with
 * {@link Builder#param(String, double)}.
 * <p>
 * Example (heat equation):
 * <pre>{@code
 * PDEResult r = PDE.problem("diff(u,t) = alpha*diff(u,x,2)")
 *         .param("alpha", 0.1)
 *         .domain("x", 0, 1, 101)
 *         .initial("u", "sin(pi*x)")
 *         .dirichlet("u", "0")
 *         .solve(0.5, 1e-3, "rk4");
 * }</pre>
 */
public final class PDEProblem {

    /** Name of the time variable. */
    public static final String TIME = "t";

    private final List<String> equations;
    private final List<String> unknowns;
    private final String[] axisNames;
    private final double[] axisMin;
    private final double[] axisMax;
    private final int[] axisN;
    private final Map<String, Double> params;
    private final Map<String, String> initial;
    private final Map<String, String> initialVelocity;
    private final Map<String, BoundaryCondition> faceBcs;
    private final Map<String, BoundaryCondition> defaultBcs;
    private final Set<Integer> periodicAxes;

    private PDEProblem(Builder b) {
        this.equations = Collections.unmodifiableList(new ArrayList<String>(b.equations));
        this.unknowns = Collections.unmodifiableList(new ArrayList<String>(b.unknowns));
        int d = b.axisNames.size();
        this.axisNames = b.axisNames.toArray(new String[d]);
        this.axisMin = new double[d];
        this.axisMax = new double[d];
        this.axisN = new int[d];
        for (int a = 0; a < d; a++) {
            axisMin[a] = b.axisMin.get(a);
            axisMax[a] = b.axisMax.get(a);
            axisN[a] = b.axisN.get(a);
        }
        this.params = Collections.unmodifiableMap(new LinkedHashMap<String, Double>(b.params));
        this.initial = Collections.unmodifiableMap(new LinkedHashMap<String, String>(b.initial));
        this.initialVelocity = Collections.unmodifiableMap(new LinkedHashMap<String, String>(b.initialVelocity));
        this.faceBcs = Collections.unmodifiableMap(new LinkedHashMap<String, BoundaryCondition>(b.faceBcs));
        this.defaultBcs = Collections.unmodifiableMap(new LinkedHashMap<String, BoundaryCondition>(b.defaultBcs));
        this.periodicAxes = Collections.unmodifiableSet(new LinkedHashSet<Integer>(b.periodicAxes));
    }

    /** @return a new builder preloaded with the given equation(s). */
    public static Builder builder(String... equations) {
        Builder b = new Builder();
        if (equations != null && equations.length > 0) {
            b.equation(equations);
        }
        return b;
    }

    public List<String> getEquations() {
        return equations;
    }

    /** @return declared unknowns (empty if they are to be inferred from the equations). */
    public List<String> getDeclaredUnknowns() {
        return unknowns;
    }

    /** @return number of spatial dimensions. */
    public int getDimension() {
        return axisNames.length;
    }

    public String getAxisName(int a) {
        return axisNames[a];
    }

    public double getAxisMin(int a) {
        return axisMin[a];
    }

    public double getAxisMax(int a) {
        return axisMax[a];
    }

    public int getAxisPoints(int a) {
        return axisN[a];
    }

    public Map<String, Double> getParameters() {
        return params;
    }

    public Map<String, String> getInitialConditions() {
        return initial;
    }

    public Map<String, String> getInitialVelocities() {
        return initialVelocity;
    }

    /**
     * @return the boundary condition for an unknown on a face, or null when
     * none was declared. Resolution order: face-specific, periodic axis,
     * per-unknown default, wildcard default.
     */
    public BoundaryCondition getBoundaryCondition(String unknown, Face face) {
        BoundaryCondition bc = faceBcs.get(unknown + "|" + face.name());
        if (bc != null) {
            return bc;
        }
        bc = faceBcs.get("*|" + face.name());
        if (bc != null) {
            return bc;
        }
        if (periodicAxes.contains(face.axis())) {
            return BoundaryCondition.periodic();
        }
        bc = defaultBcs.get(unknown);
        if (bc != null) {
            return bc;
        }
        return defaultBcs.get("*");
    }

    /** Fluent builder of {@link PDEProblem}. */
    public static final class Builder {

        private final List<String> equations = new ArrayList<String>();
        private final List<String> unknowns = new ArrayList<String>();
        private final List<String> axisNames = new ArrayList<String>();
        private final List<Double> axisMin = new ArrayList<Double>();
        private final List<Double> axisMax = new ArrayList<Double>();
        private final List<Integer> axisN = new ArrayList<Integer>();
        private final Map<String, Double> params = new LinkedHashMap<String, Double>();
        private final Map<String, String> initial = new LinkedHashMap<String, String>();
        private final Map<String, String> initialVelocity = new LinkedHashMap<String, String>();
        private final Map<String, BoundaryCondition> faceBcs = new LinkedHashMap<String, BoundaryCondition>();
        private final Map<String, BoundaryCondition> defaultBcs = new LinkedHashMap<String, BoundaryCondition>();
        private final Set<Integer> periodicAxes = new LinkedHashSet<Integer>();
        private PDEOptions options = new PDEOptions();

        Builder() {
        }

        /**
         * Adds one or more equations. A single argument may also use
         * ParserNG's system-array syntax {@code @(2)("eq1","eq2")}.
         */
        public Builder equation(String... eqs) {
            for (String e : eqs) {
                if (e == null || e.trim().isEmpty()) {
                    throw new PDEConfigurationException("Equation must not be empty");
                }
                String s = e.trim();
                if (s.startsWith("@(")) {
                    equations.addAll(parseArraySyntax(s));
                } else {
                    equations.add(s);
                }
            }
            return this;
        }

        /** Declares the unknown functions explicitly (their order fixes the order of {@code PDEResult} fields). */
        public Builder unknowns(String... names) {
            for (String n : names) {
                unknowns.add(n.trim());
            }
            return this;
        }

        /** Adds a spatial axis: {@code name} runs over [min,max] on {@code points} nodes. Call 1-3 times (x, y, z order). */
        public Builder domain(String name, double min, double max, int points) {
            if (axisNames.size() >= 3) {
                throw new PDEConfigurationException("At most three spatial dimensions are supported");
            }
            if (name == null || name.trim().isEmpty()) {
                throw new PDEConfigurationException("Axis name must not be empty");
            }
            if (!(max > min) || Double.isNaN(min) || Double.isInfinite(max)) {
                throw new PDEConfigurationException("Invalid domain [" + min + "," + max + "] for '" + name + "': need min < max");
            }
            if (points < 3) {
                throw new PDEConfigurationException("Axis '" + name + "' needs at least 3 grid points, got " + points);
            }
            axisNames.add(name.trim());
            axisMin.add(min);
            axisMax.add(max);
            axisN.add(points);
            return this;
        }

        /** Declares a named constant usable in equations and conditions. */
        public Builder param(String name, double value) {
            params.put(name.trim(), value);
            return this;
        }

        /** Initial condition {@code u(x,y,z,t0) = expr}. */
        public Builder initial(String unknown, String expr) {
            initial.put(unknown.trim(), expr.trim());
            return this;
        }

        /** Initial velocity {@code u_t(x,y,z,t0) = expr} for unknowns that are second order in time (default 0). */
        public Builder initialVelocity(String unknown, String expr) {
            initialVelocity.put(unknown.trim(), expr.trim());
            return this;
        }

        /** Boundary condition on one face. Use {@code "*"} as unknown to mean "every unknown". */
        public Builder boundary(Face face, String unknown, BoundaryCondition bc) {
            faceBcs.put(unknown.trim() + "|" + face.name(), bc);
            return this;
        }

        /** Boundary condition on every face that has no face-specific condition. */
        public Builder boundary(String unknown, BoundaryCondition bc) {
            defaultBcs.put(unknown.trim(), bc);
            return this;
        }

        /** Dirichlet {@code u = g} on all faces without a more specific condition. */
        public Builder dirichlet(String unknown, String g) {
            return boundary(unknown, BoundaryCondition.dirichlet(g));
        }

        /** Dirichlet on one face. */
        public Builder dirichlet(Face face, String unknown, String g) {
            return boundary(face, unknown, BoundaryCondition.dirichlet(g));
        }

        /** Neumann {@code du/dn = g} (outward normal) on all faces without a more specific condition. */
        public Builder neumann(String unknown, String g) {
            return boundary(unknown, BoundaryCondition.neumann(g));
        }

        /** Neumann on one face. */
        public Builder neumann(Face face, String unknown, String g) {
            return boundary(face, unknown, BoundaryCondition.neumann(g));
        }

        /** Robin {@code a*u + b*du/dn = g} on one face. */
        public Builder robin(Face face, String unknown, String a, String b, String g) {
            return boundary(face, unknown, BoundaryCondition.robin(a, b, g));
        }

        /** Makes the named axis periodic for every unknown. */
        public Builder periodic(String axisName) {
            int a = axisNames.indexOf(axisName);
            if (a < 0) {
                throw new PDEConfigurationException("periodic(): axis '" + axisName
                        + "' is not declared; call domain(...) first");
            }
            periodicAxes.add(a);
            return this;
        }

        /** Numerical options used by the terminal {@code solve*} calls. */
        public Builder options(PDEOptions o) {
            this.options = o == null ? new PDEOptions() : o;
            return this;
        }

        /** Validates and freezes the problem. */
        public PDEProblem build() {
            if (equations.isEmpty()) {
                throw new PDEConfigurationException("No equation given");
            }
            if (axisNames.isEmpty()) {
                throw new PDEConfigurationException("No spatial domain given; call domain(\"x\", min, max, points)");
            }
            Set<String> seen = new LinkedHashSet<String>();
            for (String a : axisNames) {
                if (!seen.add(a)) {
                    throw new PDEConfigurationException("Duplicate axis name '" + a + "'");
                }
                if (a.equals(TIME)) {
                    throw new PDEConfigurationException("'" + TIME + "' is reserved for time and cannot be a spatial axis");
                }
            }
            for (String p : params.keySet()) {
                if (seen.contains(p) || p.equals(TIME)) {
                    throw new PDEConfigurationException("Parameter '" + p + "' clashes with an independent variable");
                }
            }
            return new PDEProblem(this);
        }

        /** Builds the solver without running it (for real-time stepping). */
        public PDESolver solver() {
            return PDE.solver(build(), options);
        }

        /** Builds the solver with explicit options. */
        public PDESolver solver(PDEOptions o) {
            return PDE.solver(build(), o);
        }

        /** Time-dependent solve from {@code t0} (default 0) to {@code tEnd} with the given step and integrator name. */
        public PDEResult solve(double tEnd, double dt, String integrator) {
            options.dt(dt).integrator(integrator);
            return PDE.solve(build(), options, tEnd);
        }

        /** Time-dependent solve with the options set via {@link #options(PDEOptions)}. */
        public PDEResult solve(double tEnd) {
            return PDE.solve(build(), options, tEnd);
        }

        /** Steady-state (elliptic) solve with the default iterative method. */
        public PDEResult solveSteady() {
            return PDE.solveSteady(build(), options);
        }

        /** Steady-state solve with a named method, e.g. {@code "sor"}, {@code "gauss_seidel"}. */
        public PDEResult solveSteady(String method) {
            options.ellipticMethod(method);
            return PDE.solveSteady(build(), options);
        }
    }

    // ---------------------------------------------------------------- helpers
    /** Parses ParserNG's {@code @(n)("eq1","eq2",...)} system syntax. */
    static List<String> parseArraySyntax(String s) {
        int close = s.indexOf(')');
        int open2 = s.indexOf('(', close + 1);
        int last = s.lastIndexOf(')');
        if (close < 0 || open2 < 0 || last <= open2) {
            throw new PDEConfigurationException("Malformed equation array '" + s + "'; expected @(n)(\"eq1\",\"eq2\",...)");
        }
        String countText = s.substring(2, close).trim();
        String body = s.substring(open2 + 1, last);
        List<String> out = new ArrayList<String>();
        int i = 0;
        while (i < body.length()) {
            char c = body.charAt(i);
            if (c == '"') {
                int j = body.indexOf('"', i + 1);
                if (j < 0) {
                    throw new PDEConfigurationException("Unterminated string in equation array '" + s + "'");
                }
                out.add(body.substring(i + 1, j).trim());
                i = j + 1;
            } else {
                i++;
            }
        }
        try {
            int n = Integer.parseInt(countText);
            if (n != out.size()) {
                throw new PDEConfigurationException("Equation array declares " + n + " equations but contains " + out.size());
            }
        } catch (NumberFormatException ex) {
            throw new PDEConfigurationException("Malformed equation count '" + countText + "' in '" + s + "'");
        }
        return out;
    }
}
