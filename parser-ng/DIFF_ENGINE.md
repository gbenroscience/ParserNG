# Solving Differential Equations in ParserNG

ParserNG can solve ordinary differential equations — single equations, systems of equations, and higher-order equations — directly inside an expression, using four related functions: `diffeqn`, `diffeqnPath`, `diffeqnHO`, and `diffeqnPathHO`. Results can be assigned to a variable or matrix for later use, the same way any other ParserNG expression's result can. This document explains what each function does, exactly what every argument means, how to capture and read back a result, which solver methods are available and when to reach for each, and the handful of nuances that will save you a debugging session if you know them up front.

---

## 1. The one hard rule: the diffeqn call must be the whole expression

Before anything else, the rule that matters most:

**A `diffeqn`-family call must be the entire input expression. It cannot be embedded inside a larger expression, and nothing else can be embedded inside it.**

That means this is fine:

```
diffeqn(y[1] - 2*y[0], 0, 1, 5)
```

But none of these are:

```
sin(2*x) + diffeqn(y[1] - 2*y[0], 0, 1, 5)        ❌ diffeqn is not the root
diffeqn(y[1] - 2*y[0], 0, 1, 5) + 3                ❌ diffeqn is not the root
sin(diffeqn(y[1] - 2*y[0], 0, 1, 5) + 5)            ❌ diffeqn is nested inside sin
diffeqnPath(...) * diffeqnPath(...)            ❌ two calls, neither is the root alone
```

Why this restriction exists: solving a differential equation is a fundamentally different kind of computation from evaluating an arithmetic expression. `diffeqn` and its siblings don't return a single number the way `sin(x)` or `2*x+1` do in the ordinary sense — they run an entire numerical integration (potentially producing a whole trajectory of values, or a full system state) and the result has its own shape and its own downstream handling. ParserNG's evaluator needs to recognize "this whole input is a differential-equation solve" up front, before it starts walking the expression tree — it can't discover that fact halfway through evaluating some other, larger expression that happens to have a `diffeqn` call buried inside it.

**Practically:** if you want to combine the result of a `diffeqn` call with something else (add 5 to it, take its sine, multiply it by another value), solve first and combine second, as two separate steps — not as one nested expression.

This applies to all four functions: `diffeqn`, `diffeqnPath`, `diffeqnHO`, `diffeqnPathHO`.

**One sanctioned exception: assignment.** Writing `A = diffeqn(...)` (or `diffeqnPath`, `diffeqnHO`, `diffeqnPathHO`) is fine, and is in fact the normal way to capture and reuse a result — see §2. Assignment isn't "embedding" in the sense this rule is guarding against: the right-hand side is still nothing but the `diffeqn`-family call on its own, the call is still the entire expression being evaluated, and `A =` is just naming the result for later use. What's still off-limits is wrapping the call in further arithmetic on the same line, or nesting one inside another function call, as shown above.

---

## 2. Capturing results: assigning to a variable or matrix

A `diffeqn`-family call doesn't have to be a one-off, throwaway evaluation — its result can be assigned to a name, just like any other ParserNG expression, and picked up again afterward. This is the normal way to use these functions when you want to inspect, print, or reuse the outcome rather than just seeing a value flash by.

Because a scalar `diffeqn`/`diffeqnHO` call and a trajectory-producing `diffeqnPath`/`diffeqnPathHO` call don't return the same *shape* of result, what you get back on the other end of the assignment differs accordingly:

- **`diffeqn`/`diffeqnHO`** with a scalar system returns a single number — assign it and read it back as an ordinary variable's value.
- **`diffeqnPath`/`diffeqnPathHO`**, or any call whose system size is greater than 1, returns a matrix (rows of `t`, `y`, and — depending on `presentationStrategy` — the rest of the state) — assign it and read it back as a matrix.

### Example: assigning a trajectory to a matrix

```java
MathExpression me = new MathExpression(
    "A=diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");
me.solve();
FunctionManager.lookUp("A").getMatrix().print();
```

Here, `A` is bound to the full trajectory produced by the `diffeqnPathHO` call — since `presentationStrategy` is `state`, each row of the resulting matrix holds `t, y[0], y[1], y[2]`. Once `me.solve()` has run, `A` is a real, addressable name in ParserNG's function/variable table — `FunctionManager.lookUp("A")` retrieves it, `.getMatrix()` gives you the underlying matrix, and `.print()` shows it. From here, `A` behaves like any other matrix-valued variable: it can be indexed, passed to other functions, or referenced in later expressions, entirely independently of the `diffeqnPathHO` call that produced it — that call has already finished and handed off a plain result.

### Example: assigning a scalar endpoint to a variable

```java
MathExpression m = new MathExpression(
    "b=diffeqn((3t^2)*y[1]+(5*sin(t))*y[0]+(5/t)*sin(t), 1, 3, 10, 0.01, rk45)");
m.solve();
System.out.println("b = " + m.getValue("b"));
```

Here, `b` is bound to the single endpoint value `y(10)` that `diffeqn` returns. Since this is a scalar result (not a matrix), it's read back with `m.getValue("b")` rather than through `FunctionManager`'s matrix accessor.

**The takeaway:** whether you read a `diffeqn`-family result back as a scalar value or as a matrix is entirely determined by what that particular call actually produces — a scalar endpoint solve gives you a scalar, a path/HO/system solve gives you a matrix — not by how you choose to retrieve it afterward. The assignment itself (`name = diffeqn(...)`) is uniform across all four functions; only the shape of what's sitting behind the name differs.

---

## 3. The four functions, at a glance

**The dividing line that matters most:** `diffeqn`/`diffeqnPath` handle *first-order* problems — a single equation, or an explicit system of several coupled first-order equations. `diffeqnHO`/`diffeqnPathHO` handle exactly one equation of *higher order* (second, third, ...), which ParserNG reduces internally to a first-order companion system on your behalf. These two families never overlap: `diffeqnHO`/`diffeqnPathHO` will not accept the explicit-system array syntax (§5), and `diffeqn`/`diffeqnPath`'s array syntax is not a way to sneak in a higher-order equation — each array element is still just an ordinary first-order relation.

| Function | Returns | Handles | Use it for |
|---|---|---|---|
| `diffeqn` | A single endpoint value, or a vector for a system | One first-order equation, or an explicit system of them | "What is the state at t = tEnd?" |
| `diffeqnPath` | A trajectory — many rows | Same as `diffeqn`, over the whole interval | "Show me y over the whole interval." |
| `diffeqnHO` | A single endpoint state | Exactly one higher-order equation, order ≥ 2 | Second/third/... order equations — endpoint only |
| `diffeqnPathHO` | A trajectory of state vectors | Same as `diffeqnHO`, over the whole interval | Second/third/... order equations — full trajectory |

The `HO` suffix stands for **Higher Order** — it's for a single equation that involves second derivatives, third derivatives, and so on (`y''`, `y'''`, …), not just `y'`. Internally, ParserNG reduces that one higher-order equation to an equivalent *system* of first-order equations (the standard "companion system" technique), and `y0` becomes a vector holding the initial values of `y[0]`, `y[1]`, `y[2]`, and so on, up to one less than the equation's order.

---

## 4. Syntax and argument meanings

### `diffeqn` and `diffeqnHO`

```
diffeqn(equation, t0, y0, tEnd, h?, method?)
diffeqnHO(equation, t0, y0, tEnd, h?, method?)
```

### `diffeqnPath` and `diffeqnPathHO`

```
diffeqnPath(equation, t0, y0, tEnd, h?, method?, points?, presentationStrategy?)
diffeqnPathHO(equation, t0, y0, tEnd, h?, method?, points?, presentationStrategy?)
```

Arguments marked with `?` are optional and can be left out entirely — if you leave one out, a sensible default is used (see §6). You cannot, however, skip an earlier optional argument and still supply a later one — arguments are positional, so if you want to specify `points`, you must also supply `h` and `method` before it (even if you just want them at their defaults, write them out).

### What each argument means

**`equation`**
The differential equation itself, written as an expression that's already been rearranged so that everything is on one side and the other side is implicitly zero — in other words, write `LHS - RHS`, and drop the `=` sign entirely. For example, the equation `y' = -2y` becomes:

```
y[1] - (-2*y[0])
```

or more simply:

```
y[1] + 2*y[0]
```

For systems and higher-order equations, individual state components are referenced using `y[0]`, `y[1]`, `y[2]`, and so on — `y[0]` is `y` itself, `y[1]` is `y'`, `y[2]` is `y''`, and so forth. So a third-order equation like `y''' + 3y'' - y' = f(t)` would be written using `y[0]`, `y[1]`, `y[2]`, `y[3]` to refer to `y`, `y'`, `y''`, `y'''` respectively, rearranged onto one side.

**`t0`**
The starting value of the independent variable (usually time), where the solve begins. A plain number.

**`y0`**
The initial condition(s) — the value(s) of `y` (and its derivatives, for higher-order equations) at `t0`.
- For a **scalar** equation, this is a single number: `1`, `0.5`, `-2`, etc.
- For a **system or higher-order equation**, this is a vector, written as a bracketed or parenthesized literal, e.g. `(1, 0, 0)` or `@(1,3)(1, 0, 0)` — a vector of length equal to the system size (for `diffeqnHO`/`diffeqnPathHO`, that's the order of the original equation: a 3rd-order equation needs 3 initial values, for `y[0]`, `y[1]`, and `y[2]`).

**`tEnd`**
The value of the independent variable at which the solve stops. A plain number. `tEnd` can be less than `t0` — ParserNG solves in whichever direction is implied (forward or backward in the independent variable) — but `tEnd` cannot equal `t0` in a way that asks for actual integration; if they're equal, the solve is a no-op that simply hands back `y0`.

**`h`** *(optional — default `0.01`)*
The step size. For the fixed-step methods (`euler`, `rk4`, `implicit_euler`, `bdf2`), this is the exact interval used for every step, and it determines exactly how many steps the solve takes to get from `t0` to `tEnd`. For the adaptive method (`rk45`), this is only the *initial* step size — the solver will grow or shrink it automatically as it goes, based on its own error estimates, so think of `h` as a starting suggestion rather than a fixed contract when you're using `rk45`.

**`method`** *(optional — default `rk4`)*
Which numerical method to use. One of: `euler`, `rk4`, `rk45`, `implicit_euler`, `bdf2`. See §7 for a full breakdown of each, and how to choose.

**`points`** *(optional, path variants only — default: the solver's natural step count, no resampling)*
How many (t, y) rows you want back in the trajectory, evenly spaced across `[t0, tEnd]`. If you leave this out, or pass a non-positive number, you get the solver's own natural output — for a fixed-step method, that's exactly one row per step (evenly spaced already); for `rk45`, that's exactly the accepted steps the adaptive algorithm actually took, which will *not* be evenly spaced in general (adaptive solvers take bigger steps where the solution is well-behaved and smaller steps where it changes rapidly). If you specify `points` and the natural output doesn't already match that count, ParserNG resamples the trajectory onto a uniform grid via linear interpolation between the nearest bracketing points.

**`presentationStrategy`** *(optional, path variants only — default `trajectory`)*
One of two literal words: `trajectory` or `state`.
- `trajectory` gives you back just `t` and `y` (the value itself) per row — the classic "plot this" shape.
- `state` gives you back `t` plus the *entire* internal state vector per row — for a higher-order equation, that means `t, y[0], y[1], y[2], …` all together, not just `y[0]`.

A note on where this actually takes effect right now: **`presentationStrategy` is fully wired up and consumed by `diffeqnPathHO`.** `diffeqnPath` accepts and parses the argument too (so it won't error if you pass it), but nothing downstream currently changes behavior based on it for the non-higher-order path — support there is expected but not yet active. If you're solving a plain (non-higher-order) system and want to see anything beyond `t, y`, you'll want `diffeqnPathHO` for now.

---

## 5. Solving an explicit system of equations

Everything so far has covered a single equation — one unknown, one derivative relation. `diffeqn` and `diffeqnPath` also support genuine **systems**: several coupled equations, each governing its own state component, solved together. (`diffeqnHO`/`diffeqnPathHO` do **not** support this form — see the note at the end of this section for why, and what to use instead.)

### The syntax

Wrap the equations in an array literal, `@(n)(...)`, with each equation written as a **quoted string**:

```
diffeqn(@(n)("eq1", "eq2", ..., "eqN"), t0, y0, tEnd, h?, method?)
```

`n` is the number of equations, and it must match `y0`'s length exactly — one equation per state component. `y0` is still a plain vector literal, same as always: `@(1,n)(v0, v1, ..., v_{n-1})`.

Why quoted strings, specifically? A single equation (the classic unwrapped form) can contain symbolic placeholders like `y[2]` because ParserNG treats the whole first argument specially and never tries to evaluate it as a live expression. An *array* argument, on the other hand, is a literal ParserNG can resolve immediately — and it can only do that immediately for values that are already fully determined at parse time, like numbers or quoted strings. An unquoted `y[2]` inside an array isn't a determined value the moment the array is built; a quoted `"y[2]-y[1]"` is just text, resolved later, the same way a single equation always has been. So: quote every equation string, exactly like any other ParserNG string literal (`"..."` or `'...'`).

### The one rule that trips people up: every equation targets `y[n]`, not its own row

This is the single most important thing to get right, and it's the opposite of what feels natural at first.

Each equation in the array is written the same way a single `diffeqn` equation always has been — as an implicit `LHS - RHS = 0` form (`= 0` omitted), with one designated symbol divided out to become "the derivative this equation computes." For a system, **that designated symbol is always `y[n]`, where `n` is the total number of equations in the system — the same symbol for every equation in the array, regardless of which physical row that equation represents.**

So for a 2-equation system, both equations divide out `y[2]`:

```
diffeqn(@(2)("y[2]-(0.6*y[0]-0.03*y[0]*y[1])", "y[2]-(-0.9*y[1]+0.02*y[0]*y[1])"), 0, @(1,2)(30, 4), 20, 0.01, rk4)
```

The first equation computes `dy[0]/dt`; the second computes `dy[1]/dt`. Neither one writes `y[0]` or `y[1]` as the thing being divided out — both use `y[2]`, because `n = 2` here. For a 4-equation system, every equation divides out `y[4]`:

```
diffeqn(@(4)("y[4]-y[1]", "y[4]-(-2*y[0]+y[2])", "y[4]-y[3]", "y[4]-(y[0]-2*y[2])"), 0, @(1,4)(1,0,0,1), 10, 0.01, rk4)
```

**Why it works this way:** this is the exact same convention `diffeqnHO` already uses for a single higher-order equation — the target symbol is always "one past the state," i.e. `y[order]`. A system just applies that same rule independently to N equations rather than one. The upside of keeping it consistent: `EquationDivider`'s extraction logic is identical whether it's dividing a single HO equation's top derivative or one row of a system — no separate algorithm, no separate set of rules to learn.

Getting this wrong produces a clear error rather than a silently wrong answer — if an equation never mentions `y[n]`, you'll see:

```
Equation never references the top-order state y[n] -- check that the equation's highest derivative matches y0's length (order).
```

If you see that error on a system call, the fix is almost always: go back through every equation string and replace whatever target symbol you used with `y[n]`.

### Each equation is parsed independently — what that does and doesn't mean for you

Under the hood, each equation string in the array is compiled on its own, as if it had been handed to `diffeqn` by itself. This has two practical consequences worth knowing:

- **Every equation can reference any state variable freely**, including nonlinear combinations like `y[0]*y[1]` (as the Lotka-Volterra example above shows) — nothing about the system form restricts you to linear equations, and an equation can reference its own row's state component without issue.
- **A state variable that no equation ever mentions is simply treated as having zero effect from that equation** — this only matters if you're supplying an analytic (implicit-method) Jacobian; for the finite-difference default, it's invisible.

### What this means for `implicit_euler` and `bdf2`

Both stiff-system methods work on an explicit system, with the same accuracy/stability tradeoffs described in §7 — `implicit_euler` for guaranteed stability, `bdf2` when you need better accuracy at that same stability. The Jacobian ParserNG builds internally accounts for every equation's dependence on every state component, computed exactly (via automatic differentiation), not by finite differences.

### `diffeqnHO`/`diffeqnPathHO` and the array form don't mix

Passing `@(n)(...)` as the equation argument to `diffeqnHO` or `diffeqnPathHO` is rejected outright:

```
DIFFEQN_HO takes a single higher-order equation, not an equation array — use diffeqn/diffeqnPath with @(n)("eq1", ..., "eqN") for an explicit system instead.
```

This is intentional, not a missing feature. `diffeqnHO`'s whole premise is a *single* equation whose companion first-order chain (`dY_0/dt = Y_1`, `dY_1/dt = Y_2`, …) is derived automatically from one top-derivative expression — mixing that with an explicit, independently-specified N-equation array would be genuinely ambiguous (which equation is "the" top derivative? what happens to the automatic chain?). If what you actually have is a set of coupled first-order equations — even ones that came from reducing a higher-order equation by hand — write them out explicitly with `diffeqn`/`diffeqnPath` and the array syntax; don't route them through `diffeqnHO`.

### `presentationStrategy` on a system call

Same caveat as for `diffeqnPath` generally (see §4): accepted and parsed without error, but not yet consumed for a plain system the way it is for `diffeqnPathHO`. A system `diffeqnPath` call currently always returns `t, y[0], ..., y[n-1]` regardless of what you pass for `presentationStrategy`.

---

## 6. Defaults, and when it's safe to omit an argument

Two arguments have defaults you can lean on:

- **`h`** defaults to `0.01` if omitted.
- **`method`** defaults to `rk4` if omitted.

If your equation is well-behaved and you don't have strong opinions about accuracy or performance, it's entirely reasonable to write:

```
diffeqn(y[1] + 2*y[0], 0, 1, 5)
```

and let both defaults do their job. But if you care about the exact numerical behavior you're getting — stiffness handling, accuracy order, step-size philosophy — always pass both explicitly. Silently inheriting a default is fine for a quick calculation; it's the wrong call for anything you're going to depend on for correctness.

`points` and `presentationStrategy` are each **independently optional** on the path variants — you can supply neither, just one, or both, in that order (points first, then presentation strategy). You do not need to pass an empty placeholder for one just to reach the other; ParserNG figures out from the content of the trailing arguments which is which.

---

## 7. The five solver methods

### `euler` — the simplest, fastest, least accurate

Explicit (forward) Euler. Takes the derivative at the current point and steps forward in a straight line. Error shrinks linearly with step size (first-order accurate) — halve `h`, halve the error.

**Use it for:** real-time graphics, particle simulations, anywhere you need speed over precision and can afford a small step size to compensate. It's the cheapest method per step by a wide margin.

**Don't use it for:** anything where accuracy actually matters, or any system that's even mildly stiff — Euler can become numerically unstable (the solution blows up) on stiff systems unless you shrink the step size dramatically, at which point you're paying a lot for not much accuracy.

### `rk4` — the classical workhorse

Classical fourth-order Runge-Kutta. Evaluates the derivative at four cleverly chosen points within each step and combines them — error shrinks with the fourth power of the step size, so halving `h` cuts the error by roughly 16x. This is the library's own default method.

**Use it for:** general-purpose, non-stiff systems where you want good accuracy without adaptive-step bookkeeping. It's the right first choice for most equations you'd hand-write.

**Don't use it for:** genuinely stiff systems — like Euler, RK4 can require an impractically small step size to stay stable on stiff problems.

### `rk45` — adaptive, industry-standard

Adaptive-step Dormand-Prince Runge-Kutta. Runs a 4th-order and a 5th-order estimate side by side at each step, compares them to estimate the local error, and grows or shrinks the step size automatically to hit a target tolerance. This is the same family of method used as the default in most serious scientific computing environments.

**Use it for:** situations where you don't know ahead of time what step size will give you good accuracy, or where the solution's behavior changes character across the interval (slow in some places, fast in others) — `rk45` will automatically spend more steps where it needs to and fewer where it doesn't.

**A nuance worth knowing:** because the step size changes adaptively, the accepted steps are *not evenly spaced in time*. If you use `diffeqnPath` or `diffeqnPathHO` with `rk45` and don't specify `points`, you'll get back a trajectory with irregular time spacing — completely correct, but not directly suitable for something expecting a uniform grid (like certain plotting or export formats) without resampling. Specify `points` if you need evenly spaced output.

**Don't use it for:** stiff systems — `rk45`, like `rk4` and `euler`, is an explicit method, and explicit methods generally struggle with stiffness regardless of step-size adaptivity.

### `implicit_euler` — the stiff-system default

Backward (implicit) Euler. Instead of stepping forward using the derivative at the *current* point, it solves an equation for the derivative at the *next* point — this requires an internal root-finding step (Newton's method) at every single step, which costs more per step than the explicit methods, but buys unconditional *linear* stability (technically: **L-stability**) — the underlying method itself won't diverge, regardless of how stiff the system is or how large a step you take.

**A caveat worth being precise about:** that stability guarantee is a property of the *linear* method, not an ironclad promise about every solve. Each step still runs Newton's method to convergence, and if Newton fails to converge on a genuinely hard nonlinear step, the solver currently logs a warning and proceeds with its last iterate rather than halting — so a pathological system can still produce a degraded step even though the method's underlying stability theory is unconditional. In practice this is rare for well-posed physical systems, but if you see convergence warnings in your logs, that's real signal worth investigating, not noise to ignore.

Like `euler`, it's only first-order accurate (error shrinks linearly with `h`), so for a given step size it's less precise than `rk4` or `bdf2` — but on a stiff system, that tradeoff is usually well worth it, since the explicit methods may not even produce a stable answer at all.

**Use it for:** stiff systems (chemical kinetics, circuits with widely separated time constants, certain mechanical systems) where stability matters more than squeezing out extra accuracy, or as a safe fallback when you're not sure whether a system is stiff and want something that won't diverge either way.

**A nuance worth knowing:** "it works" on a system doesn't necessarily mean it's giving you tight accuracy — implicit Euler's *stability* is unconditional, but its *accuracy* is still only first-order. It's entirely possible to get a stable, plausible-looking, but numerically imprecise answer from implicit Euler on a system where a higher-order method (like `bdf2`) would visibly disagree at a tighter tolerance. If your results need to be trustworthy to more than a rough approximation, don't assume "it runs without blowing up" is the same as "it's accurate" — cross-check against `bdf2` or `rk45` if you're unsure.

### `bdf2` — the higher-accuracy stiff-system option

Second-order Backward Differentiation Formula. Like `implicit_euler`, it's an implicit method solved via Newton's method internally, and — like implicit Euler — it's L-stable: bounded and cleanly damped even in the stiffest regime, with the amplification factor going to zero rather than merely staying bounded. (Among the BDF family, this L-stability only holds for orders 1 and 2; BDF3 and above trade it away for higher order.) The real difference between BDF2 and implicit Euler is accuracy: BDF2 is second-order (error shrinks with the *square* of `h`), a real step up from implicit Euler's first-order behavior, for the same stability guarantees.

**Use it for:** stiff systems where `implicit_euler`'s first-order accuracy isn't tight enough — this is the method to reach for when you need both the stability of a backward method *and* meaningfully better precision.

**A nuance worth knowing:** BDF2 needs two previous points to take a step (it's a *multistep* method, not a single-step one), so it can't be used for the very first step of a solve — there's no "previous-previous" point yet. Internally, the very first step of any `bdf2` solve is quietly bootstrapped using one implicit-Euler step, and BDF2 proper takes over from the second step onward. This is standard, well-established practice, not a shortcut or a bug — but it does mean that a `bdf2` solve with only one step in it is really just implicit Euler in disguise, and won't show BDF2's improved accuracy. If you're relying on BDF2's second-order behavior, make sure your step size gives you more than a single step across the interval.

### Choosing quickly

- **Not stiff, want speed, don't care much about precision:** `euler`
- **Not stiff, want solid general-purpose accuracy:** `rk4` (the default)
- **Not stiff, behavior varies a lot across the interval, want automatic step control:** `rk45`
- **Stiff, want guaranteed stability, first-order accuracy is enough:** `implicit_euler`
- **Stiff, want guaranteed stability *and* better accuracy:** `bdf2`

If you're not sure whether your system is stiff: try `rk4` or `rk45` first. If the solution behaves erratically, blows up, or forces you to use an absurdly tiny step size to get a sane answer, that's usually a sign of stiffness — switch to `implicit_euler` or `bdf2`.

---

## 8. Worked examples

**A simple scalar decay equation, endpoint only, using defaults:**

```
diffeqn(y[1] + 2*y[0], 0, 1, 5)
```
Solves `y' = -2y`, `y(0) = 1`, from `t=0` to `t=5`, using `rk4` with `h=0.01` (both defaults). Returns the single value `y(5)`.

**The same equation, full trajectory, evenly sampled at 50 points:**

```
diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 50)
```

**A stiff system, solved with BDF2 and an explicit Jacobian-friendly method, adaptive tolerance not needed since it's fixed-step:**

```
diffeqn(y[1] + 1000*y[0], 0, 1, 2, 0.001, bdf2)
```

**An explicit two-equation system — Lotka-Volterra predator-prey dynamics, solved to an endpoint state:**

```
diffeqn(@(2)("y[2]-(0.6*y[0]-0.03*y[0]*y[1])", "y[2]-(-0.9*y[1]+0.02*y[0]*y[1])"), 0, @(1,2)(30, 4), 20, 0.01, rk4)
```
Both equations divide out `y[2]` — the system's total component count — not `y[0]`/`y[1]`. `y0 = (30, 4)` gives the two starting populations; the call returns their values at `t=20`.

**A third-order equation, higher-order form, full state trajectory:**

```
diffeqnPathHO(3*x*sin(x)*y[3] + 4*x*y[2] + 3*ln(x)*y[1] + 4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, bdf2, state)
```
Here `y0` is the vector `(1, 0, 0)` — initial values for `y[0], y[1], y[2]` — and `points` is omitted (so you get the solver's natural steps), while `presentationStrategy` is set to `state`, so each output row contains `t, y[0], y[1], y[2]`, not just `t, y[0]`.

**The same call, but capped to 100 evenly spaced points, still with full state:**

```
diffeqnPathHO(3*x*sin(x)*y[3] + 4*x*y[2] + 3*ln(x)*y[1] + 4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, bdf2, 100, state)
```

---

## 9. Quick troubleshooting

- **"My diffeqn call combined with other math throws an error / doesn't parse."** — See §1. The call has to be the whole expression (assignment, per §2, is the one exception). Solve first, combine the result afterward as a separate step.
- **"How do I actually get the result back out after solving?"** — See §2: assign it (`name = diffeqn(...)`), then retrieve it via `FunctionManager.lookUp("name").getMatrix()` for a matrix-valued result, or `.getValue("name")` for a scalar one.
- **"My `rk45` trajectory isn't evenly spaced."** — Expected behavior; `rk45` is adaptive. Pass `points` if you need a uniform grid.
- **"My stiff system diverges with `rk4`/`euler`/`rk45`."** — Switch to `implicit_euler` or `bdf2`.
- **"My results look stable but seem slightly off compared to a reference."** — If you're using `implicit_euler`, try `bdf2` at the same step size — implicit Euler's stability doesn't guarantee tight accuracy.
- **"I asked for `state` on `diffeqnPath` (not the HO version) and got the same output as `trajectory`."** — Expected, for now — `presentationStrategy` is currently only wired up for `diffeqnPathHO`. Use `diffeqnPathHO` if you need full-state rows.
- **"My system call throws 'Equation never references the top-order state y[n]'."** — Almost always means an equation in the array targets the wrong symbol — check that *every* equation divides out `y[n]` (the system's total component count), not its own row index. See §5.
- **"I tried to pass an array of equations to `diffeqnHO`/`diffeqnPathHO` and it was rejected."** — Expected; those two functions only accept a single equation. Use `diffeqn`/`diffeqnPath` with the array form for an explicit system instead. See §5.