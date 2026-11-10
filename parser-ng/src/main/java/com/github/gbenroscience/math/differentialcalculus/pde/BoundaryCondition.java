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

/**
 * Mathematical description of a boundary condition on one face for one
 * unknown, independent of any numerical solver.
 * <ul>
 * <li><b>Dirichlet</b>: {@code u = g}</li>
 * <li><b>Neumann</b>: {@code du/dn = g} with {@code n} the OUTWARD normal</li>
 * <li><b>Robin</b>: {@code a*u + b*du/dn = g}</li>
 * <li><b>Periodic</b>: {@code u} at the lower face equals {@code u} at the upper
 * face (declare it on both faces of the axis, or use
 * {@link PDEProblem.Builder#periodic(String)})</li>
 * </ul>
 * The coefficient expressions {@code a}, {@code b}, {@code g} are ordinary
 * ParserNG expressions. They may use the domain variables (evaluated at the
 * boundary node), {@code t}, and the problem parameters. They are compiled
 * once, never re-parsed during time stepping.
 */
public final class BoundaryCondition {

    /** Kind of boundary condition. */
    public enum Type {
        DIRICHLET, NEUMANN, ROBIN, PERIODIC
    }

    private final Type type;
    private final String a;
    private final String b;
    private final String g;

    private BoundaryCondition(Type type, String a, String b, String g) {
        this.type = type;
        this.a = a;
        this.b = b;
        this.g = g;
    }

    /** {@code u = g}. */
    public static BoundaryCondition dirichlet(String g) {
        return new BoundaryCondition(Type.DIRICHLET, "1", "0", requireExpr(g));
    }

    /** {@code du/dn = g}, outward normal. */
    public static BoundaryCondition neumann(String g) {
        return new BoundaryCondition(Type.NEUMANN, "0", "1", requireExpr(g));
    }

    /** {@code a*u + b*du/dn = g}, outward normal. */
    public static BoundaryCondition robin(String a, String b, String g) {
        return new BoundaryCondition(Type.ROBIN, requireExpr(a), requireExpr(b), requireExpr(g));
    }

    /** Periodic coupling with the opposite face. */
    public static BoundaryCondition periodic() {
        return new BoundaryCondition(Type.PERIODIC, "0", "0", "0");
    }

    private static String requireExpr(String s) {
        if (s == null || s.trim().isEmpty()) {
            throw new PDEConfigurationException("Boundary condition expression must not be empty");
        }
        return s.trim();
    }

    public Type getType() {
        return type;
    }

    /** @return expression of coefficient {@code a}. */
    public String getA() {
        return a;
    }

    /** @return expression of coefficient {@code b}. */
    public String getB() {
        return b;
    }

    /** @return expression of the right-hand side {@code g}. */
    public String getG() {
        return g;
    }

    @Override
    public String toString() {
        switch (type) {
            case DIRICHLET:
                return "u = " + g;
            case NEUMANN:
                return "du/dn = " + g;
            case ROBIN:
                return "(" + a + ")*u + (" + b + ")*du/dn = " + g;
            default:
                return "periodic";
        }
    }
}
