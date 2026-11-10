# ParserNG PDE engine

> **Contents:** 1 writing a problem · 2 solving · 3 safeguards · 4 limitations · 5 performance · 6 real-time use · 7 tests · **8 MathExpression integration and the `pde` command** · **9 incompressible flow** · **10 multigrid and preconditioning** · **11 status report** · **12 finite volume, shock capturing and compressible flow**

Package `com.github.gbenroscience.math.differentialcalculus.pde` (core `parser-ng` module, pure Java, Java 8 compatible, no new dependencies).

A numerical solver for partial differential equations in 1, 2 and 3 space dimensions: elliptic, parabolic and hyperbolic; linear and nonlinear; single equations and coupled systems. It is a sibling of the existing ODE engine (`differentialcalculus.equations`), which it does not modify, and reuses ParserNG's expression parser, its turbo compiler, and its forward-mode automatic differentiator (`SystemAutoDiffEvaluator`).

```java
PDEResult r = PDE.problem("diff(u,t) = alpha*diff(u,x,2)")      // u_t = alpha u_xx
        .param("alpha", 0.1)
        .domain("x", 0, 1, 101)                                  // 101 nodes on [0,1]
        .initial("u", "sin(pi*x)")
        .dirichlet("u", "0")                                     // u = 0 on both ends
        .solve(0.5, 1e-3, "crank_nicolson");                     // tEnd, dt, integrator

double err = r.maxError("u", (x, y, z, t) -> Math.exp(-Math.PI*Math.PI*0.1*t) * Math.sin(Math.PI*x));
```

## 1. Writing a problem

### Equations

`LHS = RHS`, in ordinary ParserNG expression syntax. Derivatives use ParserNG's `diff` notation:

| write | meaning |
|---|---|
| `diff(u,t)` / `diff(u,t,2)` | u_t / u_tt |
| `diff(u,x)` / `diff(u,x,2)` | u_x / u_xx |
| `diff(u,x,y)` or `diff(diff(u,x),y)` | u_xy (mixed) |
| `diff(u,x,2,y,1)` | u_xxy |

* Independent variables are the names given to `domain(...)` (conventionally `x`, `y`, `z`) and `t`.
* Anything else must be a declared `param(name, value)`, an unknown, or a constant (`pi`, `e`).
* Systems: several equations (`PDE.problem(eq1, eq2)`), or ParserNG's array syntax `@(2)("eq1","eq2")`. The system must be square: as many equations as unknowns. `unknowns("v","u")` fixes the field order.
* Equations are solved for their highest time derivative automatically; it may carry a coefficient or sit on either side (`2*diff(u,t) = ...`, `diff(u,t) - 0.5*diff(u,x,2) = 0`). It must appear linearly (checked exactly with automatic differentiation).
* **Scientific notation:** ParserNG reads a lower-case `e` as Euler's number, so `1e-3` would silently mean `1*e-3`. The PDE front end rewrites numeric literals such as `1e-3` to `1E-3`. (Plain `ParserNG` expressions outside the PDE engine are unchanged.)

### Domain, initial and boundary conditions

```java
.domain("x", 0, 1, 101).domain("y", 0, 2, 201)     // up to three axes, nodes include both end points
.initial("u", "exp(-50*((x-0.5)^2 + (y-1)^2))")    // u(x,y,t0)
.initialVelocity("u", "0")                         // u_t(x,y,t0), second-order-in-time unknowns (default 0)
.dirichlet("u", "0")                               // u = g          (all faces without a face-specific rule)
.neumann("u", "0")                                 // du/dn = g      (OUTWARD normal)
.robin(Face.X_MAX, "u", "1", "1", "0")             // a*u + b*du/dn = g
.periodic("x")                                     // periodic in x for every unknown
.dirichlet(Face.Y_MAX, "u", "sin(pi*x)*sin(t)")    // data may depend on the tangential coordinates and on t
.dirichlet("*", "0")                               // "*" = every unknown
```

Every unknown needs a condition on every face; missing ones are reported by name. Periodic axes have the end node `n-1` duplicate node `0`. If the initial data contradict a Dirichlet/Neumann/Robin condition on boundary nodes, the boundary condition wins and a warning is recorded in the diagnostics.

## 2. Solving

```java
PDEResult r = builder.solve(tEnd, dt, "rk4");      // time dependent
PDEResult r = builder.solveSteady("sor");          // steady (no time derivative)
PDESolver s = builder.solver();                    // reusable, real-time (section 6)
```

Numerical choices live in `PDEOptions` (every field has a default):

```java
new PDEOptions().integrator("rk4").dt(1e-3).spatialOrder(4)
        .firstDerivativeScheme(DerivativeScheme.BACKWARD)   // upwinding for transport
        .stabilityPolicy(StabilityPolicy.SUBSTEP).threads(0)
```

### Methods

| class | methods | notes |
|---|---|---|
| time integration | `euler`, `heun`, `rk4`, `rk45` (Dormand–Prince 5(4), adaptive), `backward_euler`, `crank_nicolson`, `verlet` (leapfrog, 2nd-order-in-time only) | `auto`: RK4 for hyperbolic, Crank–Nicolson for parabolic |
| elliptic | `jacobi`, `gauss_seidel`, `sor`, `red_black_sor`, `newton_krylov` | `auto`: red-black SOR (single unknown), Newton–Krylov (systems) |
| spatial | central differences of order 2, 4, 6 (`spatialOrder`), one-sided (upwind) first derivatives of order 1–6 | weights generated by Fornberg's algorithm; second derivatives are always central |

* **Method of lines**: space is discretised on a uniform grid; the resulting ODE system `y' = F(t,y)` is integrated in time. Second-order-in-time equations (`diff(u,t,2)`) become first-order systems in `(u, u_t)`.
* **Implicit steps** solve the nonlinear step equations by Newton iteration with matrix-free BiCGStab. The Jacobian comes from ParserNG's automatic differentiator (exact; one linearisation per Newton iteration, then cheap multiplications) or from finite differences if an expression is outside the differentiator's operator set (for example comparisons inside `if`). The method string in the diagnostics says which.
* **Boundary nodes are constrained**, not advanced: they are recomputed after every update from the boundary condition (Neumann/Robin by a one-sided difference of the configured order, solved for the node value). Halo cells beyond the boundary are filled by extrapolation of degree `spatialOrder+1`, so stencils never branch.
* **Classification** (`PDEType`): no time derivative → elliptic; `diff(u,t,2)` or first-order transport → hyperbolic; first time derivative with a second space derivative → parabolic. This chooses defaults; it is a heuristic, not a principal-symbol analysis.

## 3. Correctness safeguards

* **Stability:** explicit integrators are checked before stepping. The Jacobian spectral radius is estimated (power iteration, for CFL-type limits and a suggested step), and the amplification of the integrator's one-step map is probed, which rejects forward Euler on oscillatory problems. A violating step throws `UnstableTimeStepException` carrying the estimated limit (`getMaxStableDt()`), or is sub-stepped / only warned about according to `StabilityPolicy`. A divergence guard (non-finite values, or growth beyond `blowupFactor` times the initial scale) catches marginal cases.
* **Convergence:** elliptic iterations report iterations, final exact residual, contraction factor, and an error estimate; a non-converging solve returns `isConverged() == false` with warnings (or throws `NonConvergenceException` with `failOnNonConvergence(true)`). Implicit steps retry with halved sub-steps before failing.
* **Validation:** missing conditions, unknown symbols, unsupported equation forms, incompatible method choices, too few grid points, and so on all throw `PDEConfigurationException` with a message that says what to change.
* **Diagnostics:** `result.getDiagnostics()` — method, steps, rejected steps, Newton/Krylov counts, residuals, spectral radius, stability limit, warnings.

## 4. Documented limitations

* Rectangular domains on uniform tensor grids; no curvilinear or unstructured meshes, no immersed boundaries.
* **No shock capturing.** Central and linear upwind stencils are not monotone; nonlinear hyperbolic problems with shocks (Burgers, Euler equations) need limiters/Riemann solvers that are not included. Upwind direction is chosen by the user (`BACKWARD`/`FORWARD`), not automatically from the local velocity.
* Time derivatives up to second order, no mixed space-time derivatives, no algebraic constraints (DAEs).
* Point-iteration elliptic methods solve a single unknown; use `newton_krylov` for systems. BiCGStab is unpreconditioned, and there is no multigrid, so iteration counts grow with grid size (the Poisson figures below use a one-mode right-hand side, which Krylov methods solve in one iteration — a general right-hand side needs on the order of hundreds).
* At domain corners, conditions of different faces are applied in axis order (x, then y, then z); the last one wins.
* Explicit SIMD (Java Vector API) and GPU back ends are not implemented (see §5).

## 5. Performance design

Steady-state stepping allocates **nothing** (verified by tests, below), and all expressions are compiled once.

* Fields are flat `double[]`, x-fastest, with a halo; `Grid.index(i,j,k)` gives the flat index. Stencils are precomputed as `(offset, weight)` pairs, so interior kernels have no branches and no index arithmetic beyond one add.
* Expressions go through ParserNG's turbo compiler once; a node evaluation reads a per-thread `double[]` frame (no lookups, no allocation).
* **Specialised kernels.** At compile time the automatic differentiator and random-point checks decide whether an equation is *affine with constant coefficients* in the unknowns and their derivatives (heat, wave, advection, Laplace, …). Such equations are replaced by a merged tap list and run as plain contiguous row loops that the JIT can auto-vectorise; nonlinear / variable-coefficient equations use the general compiled-expression kernel.
* **Parallelism:** static partitioning of the outermost axis over a reusable worker pool (no tasks, futures or lambdas per step). Results are bit-identical for any thread count. `threads(0)` (default) stays serial below 20 000 nodes.
* Work buffers (stages, Newton and Krylov vectors, probes) are allocated at construction. Explicit schemes never copy the solution: `PDESolver.state(name)` returns the live storage.
* The numerical engine is separated from the mathematical description (`Model` kernels vs `EvolutionEngine` / `EllipticEngine`), so another back end (explicit Vector API, GPU) can replace the kernels without touching the public API. That back end is future work.

### Measured (single core, JDK 21, `PDEBenchmarkHarness full`)

Ratios are against a hand-written, flat-array Java implementation of the same algorithm. B/step is allocation per step on the stepping thread and all engine workers.

| workload | nodes | ms/step | M updates/s | vs hand-written | B/step |
|---|---:|---:|---:|---:|---:|
| 1D heat, Euler | 1,048,576 | 2.8 | 369 | – | 0 |
| 2D heat, Euler | 128² | 0.048 | 344 | 1.5× | 0 |
| 2D heat, Euler | 512² | 0.84 | 313 | 2.6× | 0 |
| 2D heat, Euler | 1024² | 3.09 | 339 | 1.4× | 0 |
| 2D heat, RK4 | 512² | 4.1 | 64 | – | 0 |
| 2D wave, Verlet | 1024² | 5.7 | 184 | 1.9× | 0 |
| 3D diffusion, Euler | 128³ | 8.7 | 242 | 2.2× | 0 |
| 3D diffusion, Euler | 256×256×128 | 45.7 | 184 | 1.7× | 0 |
| reaction–diffusion (nonlinear), RK4 | 512² | 26.0 | 10 | – | 0 |
| Gray–Scott (2 coupled PDEs), RK4 | 512² | 56.4 | 4.6 | – | 0 |
| heat 2D, Crank–Nicolson (exact AD Jacobian) | 129² | 3.2 | 5.2 | – | 0 |

GC activity during every measured window: none. Honest reading: constant-coefficient linear problems run within about 1.4–2.6× of hand-written code; nonlinear problems use the general compiled kernel (~100–200 ns per node-update with RK4) and are 20–50× slower than the fused path. On a **one-core** machine the thread rows show no speedup (a 4-thread run on one core is slightly slower); multi-core scaling has *not* been measured. JMH versions of these benchmarks are in `pde-jmh/PDEBench.java` (written to the conventions of the `parser-ng-bench` module; they compile against the JMH annotations but could not be run in the build environment).

## 6. Real-time use and ODE interoperability

```java
try (PDESolver s = PDE.problem("diff(u,t,2) = diff(u,x,2) + diff(u,y,2)")
        .domain("x", 0, 1, 65).domain("y", 0, 1, 65)
        .initial("u", "exp(-100*((x-0.3)^2 + (y-0.6)^2))").dirichlet("u", "0")
        .options(new PDEOptions().integrator("verlet")).solver()) {
    s.initialize();
    double dt = 0.5 * s.maxStableTimeStep();
    double[] u = s.state("u");                    // live storage, never reallocated
    Grid g = s.grid();
    while (running) {
        s.step(dt);                               // allocation free
        draw(u, g);                               // u[g.index(i,j,0)]
    }
}
```

`PDESolver.toOdeSystem()` returns a `MethodOfLinesSystem`: the semi-discrete system as a `MethodHandle void f(double[] vars, double[] out)` (`vars[0] = t`), so the existing ODE integrators (`DifferentialEquations.stepRK4`, `stepRK45Adaptive`, …) can drive a PDE directly. Those integrators allocate per step, so this is for reuse and cross-checking, not for the allocation-free path.

## 7. Tests and benchmarks

`src/test/java/.../differentialcalculus/pde/` (JUnit 5, 189 tests across the PDE package and `parser/PdeExpressionTest`; 754 with the module's existing tests):

* `PDEParabolicTest` — heat equation in 1/2/3-D against analytic solutions with every integrator; Neumann, Robin, periodic and mixed conditions; time-dependent boundary data; nonlinear and manufactured solutions; observed temporal order 2 (Crank–Nicolson) / 1 (backward Euler) and spatial order 2 / 4; adaptive tolerance control.
* `PDEHyperbolicTest` — wave equation (all integrators, initial velocity, driven boundaries, Neumann, damping), vibrating membrane, advection (central 4th order, upwind orders 1–2 with measured convergence, inflow boundary), a hyperbolic system with mixed boundary types.
* `PDEEllipticTest` — Laplace, Poisson (all methods; observed order 2 and 4), 3-D, mixed Dirichlet/Neumann/Robin, periodic, variable coefficients, nonlinear, coupled system, non-convergence (reported and raised), iteration-count ratios, bit-identical parallel results.
* `PDEValidationTest` — invalid configurations with their messages, unstable steps, the stability policies, divergence guard.
* `PDEPerformanceContractTest` — allocation per step (JVM per-thread counters on the stepping thread and every worker) for every integrator, 1-D/2-D/3-D, Neumann/periodic/time-dependent boundaries, multi-threaded; thread-count independence; live storage; ODE-engine interoperability.
* `PDESyntaxTest`, `PDEExamplesTest` — notation, equation arrays, the AD fallback, kernel equivalence; the worked examples (`PDEExamples`).

Benchmarks: `PDEBenchmarkHarness` (test sources, plain `main`) and `pde-jmh/PDEBench.java`.

```
java -Xmx3g -cp <classes>:<test-classes> com.github.gbenroscience.math.differentialcalculus.pde.PDEBenchmarkHarness full
```

## 8. MathExpression integration and the `pde` command

`MathExpression` stays the single entry point: a PDE is recognised by the constructor, represented, and handed to the PDE subsystem by composition. `MathExpression` never solves anything numerically.

```java
MathExpression equation = new MathExpression("diff(u,t) = alpha*diff(u,x,2)");   // ExpressionKind.PDE
PDEResult r = equation.asPDE()                       // PDEProblem.Builder, preloaded with the equation
        .param("alpha", 0.1).domain("x", 0, 1, 101)
        .initial("u", "sin(pi*x)").dirichlet("u", "0")
        .solve(0.5, 1e-3, "crank_nicolson");
```

| notation | meaning |
|---|---|
| `diff(u,t) = alpha*diff(u,x,2)` | bare equation (recognised by shape) |
| `pde(diff(u,t) = alpha*diff(u,x,2))` | the explicit `pde` command (registered like the other inbuilt commands, in `Declarations`) |
| `eq1; eq2` / `pde(eq1, eq2)` / `pde(@(2)("eq1","eq2"))` | systems (`ExpressionKind.PDE_SYSTEM`); the array form is the same convention as the ODE systems |
| `alpha=0.1; diff(u,t) = alpha*diff(u,x,2)` | constants assigned next to the equation become problem parameters automatically |

* `MathExpression.getExpressionKind()` returns `SCALAR`, `ODE`, `ODE_SYSTEM`, `PDE` or `PDE_SYSTEM`; `isPDE()`, `getPDEModel()` and `asPDE()` expose the model. `PDEModel` is the grid-free mathematical description (equations, unknowns, independent variables, classification); the numerical problem (domain, conditions, method) is the builder `asPDE()` returns, so one expression can be solved with different grids and methods.
* **Recognition rules.** A statement is a PDE when it has a single top-level `=`, its left side is *not* an assignment target (`name`, or `f(params)` for a user function) and it contains a PDE-shaped derivative; or when the left side starts with `diff(` (an inbuilt name can never be the function being defined); or when it is a `pde(...)` call. `y = diff(v,x)` therefore stays an assignment, `x==3`, `x<=3` and `h(x)=diff(x^2,x)` are untouched, and `diff(F,x,2)` with a *defined* ParserNG function `F` keeps its ordinary meaning in a bare statement. Inside an explicit `pde(...)` a bare identifier is always an unknown function. Use `pde(...)` to force the PDE reading.
* **`MathExpressionTreeDepth`.** `pde` follows the diffeqn-family rule (it must be the root of the expression). Its arguments are validated equation by equation: one top-level `=`, at least one PDE-shaped derivative, well-formed `diff(...)` calls, and each side checked as an expression by a nested `MathExpressionTreeDepth`, with unknowns, parameters and coordinates left undeclared (they are declared when the numerical problem is configured). Nothing is bypassed, no existing code path changes, and the scalar and ODE paths are untouched (`PdeExpressionTest` covers recognition, pass-through, regressions and diagnostics).
* A PDE expression is compiled to nothing: `solve()` reports the canonical model text and `compileTurbo()` explains that a PDE is a model. Evaluating `pde(...)` where a number is expected throws a descriptive exception instead of returning NaN.
* Files touched for the command: `Declarations` (constant, arity, name list, return type), `Method` (`isMultiArgsMethod`, `isSupportedPdeMethod`), `MethodRegistry`, `MathExpressionTreeDepth`, `MathExpression`, plus the new `ExpressionKind` and `pde.PDEModel`.

## 9. Incompressible Navier–Stokes (`IncompressibleFlow`)

```java
IncompressibleFlow cavity = IncompressibleFlow.builder().cells(64, 64).reynolds(100)
        .boundary(Face.Y_MAX, FlowBoundary.movingWall("1", "0")).build().initialize();
cavity.runToSteadyState(1e-6, 60, 0.25 / 64);
FlowResult r = cavity.result();                       // velocityAt(...), streamFunction2D(), vorticity2D(...)
FlowDiagnostics d = cavity.diagnostics();             // divergence, mass flux, residuals, energy, CFL, ...
```

`u_t + (u·∇)u = −∇p/ρ + ν∇²u + f`, `∇·u = 0` in 2-D and 3-D on a **staggered MAC grid** (velocity on faces, pressure at cell centres: no checkerboard modes). Incremental pressure-correction projection: Adams–Bashforth-2 predictor (forward Euler on the first step) → pressure Poisson equation solved by `PoissonMultigrid` → correction `u = u* − dt ∇φ`, `p += φ`. The discrete divergence and gradient are exact adjoints, so the corrected velocity is divergence-free to the Poisson tolerance (≈1e-14 observed). Conservative second-order central convection with an optional donor-cell blend (`upwindBlend`).

Boundaries (`FlowBoundary`): `noSlip()`, `movingWall(...)`, `slip()`, `inflow(...)` (velocity expressions of x, y, z, t), `outflow()` (zero velocity gradient, p = 0), `periodic()`. Faces default to no-slip walls; corners need no special handling because ghost values are set per component and axis.

Diagnostics (`FlowDiagnostics`): divergence (max, L2), net boundary flux (mass-conservation error), pressure residual and cycles, velocity residual (`max|Δu|/dt`, the steady-state indicator), kinetic energy, max vorticity, enstrophy, max speed, Reynolds number, cell Reynolds number, CFL number, diffusion number. A diverging run throws `UnstableTimeStepException` rather than returning NaN.

Cost: O(N) per step. Memory: about (3d+3) velocity-sized arrays plus the multigrid hierarchy. Allocation: **0 B/step** (tests, 2-D with every boundary type, periodic, 3-D). Parallelism: single-threaded in this release. Measured on one core: 2-D ≈ 3.5 M cell-steps/s, 3-D ≈ 2.1 M cell-steps/s (the pressure solve, 9–17 cycles at the default 1e-10 tolerance, dominates; loosening `poissonTolerance` trades divergence for speed).

### Validation performed

| case | result |
|---|---|
| Taylor–Green vortex (periodic, exact decay) | velocity order 1.98 / 1.99 (n = 16→32→64); divergence 1e-14; kinetic energy and pressure match the exact solution |
| Couette flow, transient (series solution) | order 2.0 (error 4.3e-3 → 1.1e-3 → 2.6e-4); steady state 1.7e-9 |
| Poiseuille, body-force driven | order 2.0; steady parabola |
| Poiseuille, inflow/outflow | exact profile to 3e-3, pressure gradient −0.7952 vs −0.8000 (0.6%), net flux 1e-18 |
| Kovasznay flow (Re = 40, nonlinear, exact) | velocity order 1.97 / 1.99, pressure converging; divergence 1e-14 |
| 3-D Ethier–Steinman (exact, time-dependent Dirichlet on 6 faces) | error 8.7e-3 → 2.9e-3 → 9.7e-4 for n = 8, 16, 32 (order ≈ 1.6 in the max norm); divergence ≤ 2e-13 |
| 2-D flow extruded in z, run in 3-D | equals the 2-D result to 1e-7 |
| lid-driven cavity vs mirror image (lid reversed) | identical to 1e-16: no min/max-face asymmetry |
| lid-driven cavity, Re = 100 vs Ghia, Ghia & Shin (1982) | u centreline within 0.004; v centreline within 0.0086; ψ_min −0.1032 at (0.609, 0.734) vs −0.1034 at (0.617, 0.734) |

**Open item — be careful before citing the cavity numbers as "validated against Ghia".** The Ghia reference values in the tests were entered from memory. Along the horizontal centreline our v agrees with them to about 0.003 for x < 0.6 but is 3–4 % more negative (up to 0.0086) for x > 0.8. That difference does not change between n = 32 and n = 64 (so it is not discretisation error), and the solver is mirror-symmetric to 1e-16, and it passes every exact-solution test above. Either a table entry was misremembered or the 1982 reference (first-order wall vorticity) is itself that inaccurate; I could not decide which without the published table. The test bound documents the current agreement rather than hiding it; please check the table. A Re = 1000, n = 64 run (115 s) had not reached steady state by t = 80 and gave ψ_min = −0.1136 at the right location against −0.1179 (3.7 %, not grid-refined); it is **not** included in the tests.

### Limitations
Uniform Cartesian boxes only (**no flow around a cylinder or any internal obstacle**: not implemented). Explicit AB2 diffusion limits `dt ≲ h²/(4 d ν)`: the step count grows like 1/h² (the 64² Re = 100 cavity needed 10 788 steps to reach steady state, the 32² one 3 376); an implicit/semi-implicit diffusion step would remove this and is future work. Central convection becomes oscillatory above cell Reynolds ≈ 2 (use `upwindBlend`). The incremental scheme is first-order in the pressure near boundaries. No turbulence model.

## 10. Multigrid and preconditioning

`PoissonMultigrid` — matrix-free geometric multigrid for the cell-centred Poisson equation in 2-D/3-D: Neumann, Dirichlet or periodic on each face; restriction by cell averaging, (bi/tri)linear prolongation; V and W cycles; weighted Jacobi, Gauss–Seidel, red–black Gauss–Seidel and SOR smoothers; configurable pre/post sweeps, tolerances and cycle cap; singular (all-Neumann/periodic) problems are made compatible and normalised. Measured: convergence factor 0.081 (n = 64) … 0.095 (n = 1024) per V(2,2) cycle, independent of grid size; 0.026 for W cycles; second-order accurate solutions; ≈ 21 ns per cell per cycle; `solve()` allocates nothing; memory ≈ 4 arrays × N × (1 + 1/2^d + …). It is the pressure solver of `IncompressibleFlow` and can be used directly.

BiCGStab (matrix-free Newton–Krylov) gained **Jacobi preconditioning** from the exact diagonal of the automatic-differentiation Jacobian (`PDEOptions.jacobiPreconditioner`, default on). Measured Krylov iterations with → without: variable-coefficient Poisson 234 vs 561 (n = 65), 2.4× fewer; Crank–Nicolson reaction–diffusion steps 4460 vs 6063; constant-coefficient Poisson identical (119 vs 119, 235 vs 235), strongly anisotropic 385 vs 403 and the nonlinear problem 357 vs 357 essentially unchanged (a diagonal preconditioner cannot help there). Iteration counts still grow like 1/h (119 → 235 for n = 33 → 65).

Not done: multigrid is **not** wired into the general expression-driven elliptic engine (`EllipticMethod.MULTIGRID`), into implicit diffusion steps, or as a BiCGStab preconditioner; incomplete-factorisation preconditioners were not investigated.

## 11. Status report (Phase 2)

| area | status |
|---|---|
| MathExpression recognises PDEs / PDE systems; `pde` command; `MathExpressionTreeDepth`; `asPDE()`; `ExpressionKind` | **implemented and tested** (31 tests; 754-test suite green) |
| 2-D incompressible Navier–Stokes (MAC projection, BCs, diagnostics) | **implemented and validated** against exact solutions (Taylor–Green, Couette, Poiseuille, Kovasznay); cavity comparison with Ghia is **partially resolved** (open item above) |
| 3-D incompressible Navier–Stokes | **implemented, validated** against Ethier–Steinman and 2-D extrusion; order ≈ 1.6 with time-dependent Dirichlet data |
| flow around a cylinder / obstacles | **not implemented** |
| geometric multigrid (cell-centred, 2-D/3-D, V/W, 4 smoothers) | **implemented and validated** (rates, accuracy, BC types, allocation) |
| multigrid integrated with pressure projection | **done** |
| multigrid in the generic elliptic engine, implicit diffusion, Newton–Krylov preconditioning | **not done** |
| Jacobi-preconditioned BiCGStab | **implemented and measured** (helps only where the diagonal varies) |
| finite-volume framework (cell-centred, fluxes LLF/Rusanov/HLL/HLLC, MUSCL limiters, SSP-RK, boundaries), Burgers, conservation laws | **implemented and validated** (section 12) |
| compressible Euler (1/2/3-D, HLLC), Sod shock tube vs the exact Riemann solution | **implemented and validated** for Sod (positions, plateaus, errors) and the smooth isentropic vortex (order ≈ 2.2); strong-shock / near-vacuum robustness **not tested** |
| finite elements (triangles/tetrahedra, assembly, sparse solvers) | **not started** |
| Fourier / Chebyshev spectral methods | **not started** |
| non-uniform / curvilinear / adaptive / unstructured grids | **not started** (the flow and PDE engines remain uniform-grid only) |
| SIMD / GPU back ends | **not started**; the existing array kernels remain the scalar baseline |

## 12. Finite volume, shock capturing and compressible flow (`FiniteVolumeSolver`)

A conservative cell-centred scheme for `U_t + div F(U) = S` on a uniform Cartesian grid in 1, 2 or 3 dimensions: `dU/dt = -sum_d (F_{i+1/2} - F_{i-1/2}) / h_d` from numerical fluxes at the faces. Components:

* **Conservation laws** (`ConservationLaw`): `EulerLaw` (ideal gas, 1–3-D: exact wave speeds, HLLC, reflecting wall) and `ExpressionLaw` (fluxes and sources as ParserNG expressions; wave speeds `dF/du` by the automatic differentiator, exact for scalar laws; for systems a Gershgorin bound from the AD Jacobian).
* **Numerical fluxes** (`NumericalFlux`): `LLF` (local Lax–Friedrichs), `RUSANOV` (same flux here; both names kept), `HLL`, `HLLC` (Euler; Einfeldt/Roe wave-speed estimates).
* **Reconstruction**: MUSCL with `Limiter` `MINMOD`, `MC`, `VAN_LEER` (or `NONE` for first order), on primitive variables where the law has them (Euler) and on conserved variables otherwise. A face whose limited states are unphysical falls back to the cell values (counted in `FVDiagnostics.getReconstructionFallbacks()`); a cell that becomes unphysical raises `UnstableTimeStepException` — nothing is clipped silently.
* **Time**: forward Euler, SSP-RK2, SSP-RK3 (default); `suggestTimeStep()` uses the Courant number `cfl` (default 0.4) on `sum_d s_d/h_d`.
* **Boundaries** (`FiniteVolumeSolver.Boundary`): periodic, outflow, reflecting wall (Euler), fixed inflow state; two ghost layers.
* **Diagnostics**: `getConservationDefect()` is `total(t) - total(0) + time-integrated net boundary flux - time-integrated source`; for a conservative scheme it is zero to round-off for *any* boundary condition, so it checks conservation even with open or reflecting boundaries.

### From a MathExpression

A conservation law is written with the flux inside the derivative and recognised by the same expression machinery as every other PDE:

```java
MathExpression burgers = new MathExpression("diff(u,t) + diff(u^2/2,x) = 0");   // PDEModel.isConservationLaw()
FiniteVolumeSolver s = burgers.asFiniteVolume().cells(400).size(2)
        .limiter(Limiter.MC).numericalFlux(NumericalFlux.HLL).initial("if(x<0.4,2,0.5)").build().initialize();
s.advanceTo(0.5);
FVDiagnostics d = s.diagnostics();
```

Recognised forms: `diff(u,t) + diff(F(u),x) [+ diff(G(u),y)] = S`, flux terms on either side, systems (`pde(eq1, eq2)` or `;`), sources (`= -k*u`), constants bound next to the equation (`a=1.5; ...`) or given with `.param(...)` on the builder. A flux may depend on the unknowns and constants only. A model that is not in this form (`diff(u,x)` as a non-conservative term, a coefficient on `diff(u,t)`, a second derivative) is refused with the reason (`PDEModel.getConservationFormReason()`). Euler is built directly: `FiniteVolumeSolver.builder().law(new EulerLaw(2, 1.4))`, with `initialPrimitive(rho, u, v, p)`.

### Validation performed (`FiniteVolumeTest`, 30 tests)

| case | result |
|---|---|
| exact Euler Riemann solver (reference used below) | reproduces Toro's tabulated Sod values: p* = 0.30313, u* = 0.92745, shock speed 1.75216 |
| Sod shock tube, n = 400, HLLC + MC | L1(density) = 0.0014; star-state density/velocity/pressure within 1 %; shock and contact positions within 0.008 / 0.01 of the exact ones |
| Sod, error ranking | first order 0.008–0.012; MUSCL 0.0014–0.003; LLF > HLL > HLLC (diffusion), MINMOD > VAN_LEER, MC (sharpness); error falls with refinement at the rate expected for discontinuities (> 0.6) |
| conservation | conservation defect ≤ 1e-12 for every flux; closed box with reflecting walls conserves mass and energy to 1e-13; open boundaries balance exactly against the boundary flux |
| linear advection | order 2.03 / 2.05 (van Leer), 1 without reconstruction, between 1.4 and 2 for minmod (it clips smooth extrema) |
| TVD | square wave: no overshoot/undershoot (≤ 1e-12) for MINMOD, MC, VAN_LEER with LLF and HLL |
| Burgers | shock speed equals Rankine–Hugoniot (1.25 for 2 → 0.5) within 0.012, L1 0.004; rarefaction is the entropy solution (monotone fan, no expansion shock), L1 0.003; sine wave forms a one-cell shock with maximum principle and exact mass conservation |
| scalar source + constants | `u_t + (a u)_x = -k u` against the exact decaying wave; defect includes the source integral |
| linear 2×2 system from two expressions | second order |
| isentropic vortex (exact smooth 2-D Euler solution) | order ≈ 2.2 (L1, n = 32 → 64 → 128), no positivity fallbacks needed |
| direction handling | a Sod tube along x, y (2-D) and z (3-D) equals the 1-D solution to 1e-12, with no spurious transverse momentum |

Two defects found and fixed by these tests: the reconstruction of scalar laws read the base state instead of the current Runge–Kutta stage (order 0.9 instead of 2; the Euler path was unaffected), and a reference-solver sign error in the rarefaction fan.

### Cost, memory, parallelism
`O(N)` per stage; per-cell state (conserved), one stage array, one right-hand-side array, plus one primitive array for Euler; row buffers; **0 B/step** (tests and harness) and no GC. Measured on one core (`FVBenchmarkHarness`, SSP-RK3 MUSCL): Euler 1-D ≈ 3.6–4.8 M cell-updates/s (HLLC ≈ 3.6 M at 10⁶ cells), 2-D 256²–512² ≈ 1.1–1.6 M/s, 3-D 32³–64³ ≈ 0.8–1.0 M/s; MUSCL costs 1.45× (minmod), 1.57× (van Leer) and 1.8× (MC) the first-order scheme; an expression-defined Burgers law runs about 1.7 M/s (the flux and AD wave speed are evaluated by the expression engine). Single-threaded; the face loops are row-partitionable.

### Limitations (honest list)
Dimension-by-dimension fluxes (no genuinely multidimensional Riemann solvers); at most second order in space; slopes are limited component by component (no characteristic limiting); HLLC exists for Euler only; for expression *systems* the wave-speed bound makes HLL behave like Rusanov; positivity is protected by the face fallback and reported, not proved; no well-balancing of source terms; reflecting walls are defined for Euler only; **not tested**: strong shocks, near-vacuum states (e.g. the Toro 123 problem), 2-D Riemann problems and shock-reflection benchmarks, so do not rely on this as a production CFD code for those regimes.
