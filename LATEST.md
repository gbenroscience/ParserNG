# ParserNG

### ParserNG 3.1.1 PDE Solver got bug fixes in `IncompressibleFlow` where channel settings diverge.

### ParserNG 3.1.0 Partial Differential Equation Solver
This update upgrades the platform with its own pure Java, near-zero allocation PDE solver, extending ParserNG's reach further across the spectrum of calculus. Pure Java 8, no new dependencies, and it sits beside the ODE engine without modifying it.
 
- **What it solves:** elliptic, parabolic and hyperbolic problems in 1, 2 and 3 space dimensions; linear and nonlinear; single equations and coupled systems. Equations are written in ordinary ParserNG syntax:
```java
PDEResult r = PDE.problem("diff(u,t) = alpha*diff(u,x,2)")   // u_t = alpha u_xx
        .param("alpha", 0.1)
        .domain("x", 0, 1, 101)
        .initial("u", "sin(pi*x)")
        .dirichlet("u", "0")
        .solve(0.5, 1e-3, "crank_nicolson");
```
- **First class in `MathExpression`:** a PDE is recognised by the constructor, either as a bare equation (`diff(u,t) = alpha*diff(u,x,2)`) or through the new `pde(...)` command, including systems (`pde(eq1, eq2)` or the `@(2)("eq1","eq2")` array form). `getExpressionKind()` reports `PDE` or `PDE_SYSTEM`, and `asPDE()` hands you a builder preloaded with the equation. Existing scalar and ODE paths are unchanged.
- **Methods:** `euler`, `heun`, `rk4`, adaptive `rk45`, `backward_euler`, `crank_nicolson` and `verlet` in time; `jacobi`, `gauss_seidel`, `sor`, `red_black_sor` and `newton_krylov` for steady problems; central differences of order 2, 4 and 6 and upwind first derivatives up to order 6. Boundary conditions: Dirichlet, Neumann, Robin and periodic, with data that may depend on position and time. Implicit steps use an exact Jacobian from ParserNG's automatic differentiator.
- **Safeguards:** explicit steps are checked for stability before they run (`UnstableTimeStepException` reports the largest stable step), elliptic solves report convergence and residuals, and bad setups fail with a message that says what to change.
- **Fast and allocation free:** stepping allocates 0 bytes (verified by tests), constant-coefficient linear problems are fused into branch-free kernels that run within about 1.4x to 2.6x of hand-written Java (300+ million node updates per second on one core), and results are bit-identical for any thread count. For real-time use, `PDESolver.step(dt)` advances the live `double[]` field in place, ready to draw every frame.
- **Incompressible Navier-Stokes (`IncompressibleFlow`):** 2-D and 3-D on a staggered MAC grid with a geometric multigrid pressure solver (`PoissonMultigrid`), no-slip, moving-wall, inflow, outflow and periodic boundaries, and a full set of flow diagnostics. Validated against exact solutions (Taylor-Green, Couette, Poiseuille, Kovasznay, Ethier-Steinman). The lid-driven cavity agrees with the published Ghia values only partly, and the doc lists this as an open item.
- **Finite volume and shock capturing (`FiniteVolumeSolver`):** conservation laws such as Burgers and the compressible Euler equations in 1, 2 and 3-D, with LLF, HLL and HLLC fluxes, MUSCL reconstruction (minmod, MC, van Leer) and SSP Runge-Kutta. A law is recognised straight from `diff(u,t) + diff(u^2/2,x) = 0`. Validated on the Sod shock tube against the exact Riemann solution, and on a smooth isentropic vortex (order about 2.2).
- **Not in this release:** non-uniform, curvilinear or unstructured grids, flow around obstacles, finite elements, spectral methods, and SIMD or GPU back ends. Strong shocks and near-vacuum states are untested, and multi-core scaling has not been measured.
Full syntax, methods, measured numbers and limitations are in [PDE.md](parser-ng/PDE.md).
 

### ParserNG 3.0.9.3 Major and Minor updates.
#### 1. Minor
Fixes to `MathScanner`, `Matrix` and `RootFinder`.
Needed syntax error fixes. Singular matrix determinants were returning an error instead of 0.0(for Matrix)

#### 2. Major(ParserNG LLM)
<b>INTEL HD520 GPU(2015 or 2016):</b><br>
a. ParserNG's OpenCL Llama Engine now runs faster than llama.cpp on Vulkan, both on Windows 10, same machine<br>

b. llama.cpp - pre-fill: 48 toks/sec, decode: 5.7-9.7 toks/sec(conversational 5.70 toks/sec, benchmark 9.7 toks/sec)<br>
ParserNG -  pre-fill: 32 toks/sec, decode: 10.5-11.5 toks/sec.

<b>AMD Hainan(from 2013-2014 or so)</b><br>
ParserNG's OpenCL Llama engine alone is supported on this hardware. llama.cpp has no AMD Hainan story.<br>
pre-fill: 44 toks/sec pre-fill and 8 toks/sec decode.

### ParserNG 3.0.9.2 Minor updates. 
ParserNG's core class, `MathExpression`, has been armed with a `strict` flag(instance field)
This allows it to determine whether to allow undeclared variable/method names fly as valid variables initialized to zero or not. When the flag is set to true, undeclared variables/methods throw a error at constructor initialization time.

### ParserNG 3.0.9 lands with 
- `parser-ng-parquet` 
- upgrades to `parser-ng-sql. Allows parquet files to be read at low alloc and very high rates, multiple times faster than the standard parquet-java reader. `parser-ng-sql` now selects over both local and remote parquet files.

### ParserNG 3.0.8 is out on maven-central
 A huge speed update over the first edition of `parser-ng-sql`. Race fixes and alloc. updates, further cementing the zero copy promises of the module.
Now brings the full power of `parser-ng-arrow` to the user, by introducing consierably lesser overhead while allowing `parser-ng-arrow` do its work on the substrate data. Now comes with an execute(args) overload that allows you to use reusable output buffers for even greater speed.
Approaches Gandiva speed on simple `arithmetic/sqrt/logic` and beats it by 10x to 30x on every other thing. No LLVM, No JNI. Pure Java.


### ParserNG 3.0.7 is out on maven-central
   Introduces `parser-ng-sql`, a new extension which is a SQL-shaped front end over parser-ng-arrow: SELECT ... FROM ... [WHERE ...] compiles to a vectorized Arrow filter/projection pipeline built on ArrowExpressionEvaluators. 
This is deliberately NOT a database - no JOIN, INSERT, UPDATE, DELETE, transactions. 
It exists purely to make SQL a convenient surface syntax for describing vectorized Arrow computations that parser-ng-arrow already knows how to run.
### ParserNG 3.0.6 is out on maven-central
This version takes `parser-ng-arrow` to the next level by adding the `filter`, the `project` and the `filterProject` methods.
This extends ParserNG's reach into BigData as not just an evaluator, but also as a query(non-SQL) engine fo BigData. This functionalities are available across the CPU and GPU.
Enjoy!

### ParserNG 3.0.5 is out on maven-central!
- This version features the gradual phaseout of SIMDEngineEvaluator and the phase-in of the SIMDCommand series. 
The former SIMDCommandEvaluator is also being retired for a suite of SIMDCommand classes; one for handling float64(double) data, one for handling float32(Java's float primitive) data, one each for handling MemorySegment storing doubles and MemorySegment storing floats.
SIMDCommandF32,
SIMDCommandF64,
SIMDCommandSegmentF32,
SIMDCommandSegmentF64,
The API remains unchanged from that of the SIMDEngineEvaluator.
SIMDCommandF32 runs faster by default than the others since it is handling float32 data.
- The second big addition to the ParserNG runtime is the `parser-ng-arrow` extension which brings the SIMD and the GPU
leveraging power of ParserNG to the Apache Arrow library.
Out of the box, `ArrowBulkEvaluator` uses ParserNG's SIMD integration to beat the existing evaluator, Apache Gandiva on most expressions, save pure arithmetic and expressions where Gandiva can use the native hardware intrinsics (sqrt)
where Gandiva is about 1.5x to 2x as fast. Since ParserNG comes with a parallel mode in its SIMD evaluators, if you set it to true, out of the box, you will see even more throughput depending on how many processors and how much RAM your device possesses.

The second class is `ArrowGpuBulkEvaluator` which lets you select the GPU to use on your system to process Arrow data. It supports OpenCL and CUDA.(Feedback is needed on both paths, but especially on the CUDA path.)
Both `ArrowBulkEvaluator` and `ArrowGpuBulkEvaluator` support `Float4Vector` and `Float8Vector`.
For more on `ArrowBulkEvaluator`, see: 
[parser-ng-arrow's README.md](parser-ng-arrow/README.md)

For more on `ArrowGpuBulkEvaluator`, see: 
[parser-ng-arrow's GPU README.md](parser-ng-arrow/ARROW-GPU-EVAL.md)

- Thirdly, Metal framework support has been added to the  GPU bulk evaluator and the incubating LLM engines, which formerly supported CUDA and OpenCL alone, bringing the macOS ecosystem into the ParserNG GPU bulk evaluation/LLM world, or vice versa.


### ParserNG 3.0.4 is out on maven-central!
This update strengthens the differential equation engine by adding systems solving capability explicitly
in the `diffeqn` and `diffeqnPath` paths. The engine has also been given a battery of tests, 120 in number
to check the engine for correctness and to give users an insight into the usage.
Equally important in this release is the emergence of the `ARRAY` type in ParserNG.
This is the piece that made the systems solving ability possible in ParserNG.
Pre-dating the ARRAY type was the `VECTOR` type which was simply an array of numbers.
The `ARRAY` type allows one to store both numbers, strings etc.
It is also bound to the `Function` class. So a Function in ParserNG may be a VECTOR(a 1D matrix), a MATRIX, an ARRAY, or an ALGEBRAIC_EXPRESSION. You may access it with the getArray() method on the Function object.


More methods are layered on the MathExpression object such as `solveGenericWithThrows` and
`solveWithThrows`, which throw java.lang.Throwable`, if the expression errors out during evaluation.
There may be a slight breaking change also with the MathExpression constructor which now throws an InputMismatchException.

### ParserNG 3.0.3 is out on maven-central!
It features a differential equation solver that supports rk4, rk45, euler, implicit_euler, bdf2.
Incubating in parser-ng-gpu-simd extension is a pure Java Llama model runner which uses the GPU(CUDA and OpenCL)
to execute Llama models. We will need a lot of feedback from our users on this, as it is in active development.



### ParserNG 3.0.2 is out on maven-central!
### ParserNG 3.0.1 is out on maven-central!
Stabilized v3.0.0
### ParserNG 3.0.0 is out on maven-central! 
1. GPU bulk evaluator(CUDA and OpenCL) comes to ParserNG with MemorySegments and zero dependencies; no JNI hell.
2. SIMDEngineEvaluator with MemorySegments also and SIMDCommandTurboEvaluator from ParserNG Enterprise now open-sourced and released with version 3.0.0
3. Nth-order Automatic Differentiation, A fresh symbolic differentiator and a resilient numerical integrator, fronted by a symbolic integrator to expand the range of numerically integrable functions
4. Conditionals; if, &&, || implemented all the way from parser frontend to all backends(std evaluator, 2 turbo scalar evaluators and 1 turbo matrix evaluator, bulk evaluators(vector simd and auto-vectorization type) and GPU bulk evaluator)

### ParserNG 2.0.7 is out on maven-central! 
Full migration of all inbuilt functions and UDFs to the Vector API(not stats functions), leading to 2.0x to 14x speedup in bulk evaluations


### ParserNG 2.0.6 is out on maven-central! 
Very stable release. Lots of bug fixes in bulk gelu, swiglu etc

### ParserNG 2.0.5 is out on maven-central! 
More stable. Cleaned up API.


### ParserNG 2.0.3 is out on maven-central! 
More stable. Indexing bugs fixed in bulk processors

### ParserNG 2.0.0 is out on maven-central! 
Guess who the kid on the block is? Vector API bulk evaluator(SIMDVectorTurboEvaluator)and its compatibility partner, VectorTurboEvaluator.
Both run bulk evaluations at roughly same speed(competitive with Janino), and come with workers out of the box!
`VectorTurboEvaluator` works because its code is mechanically sympathetic to the hardware running it, so auto-vectorization occurs.



### Parser 1.2.1 has been released on maven-central!
Version 1.2.1 fixes bugs in Matrix Algebra in the standard mode.


### Parser 1.2.0 has been released on maven-central!
Version 1.2.0 introduces pure Matrix algebra into ParserNG standard. Its great performance is not at par with what
ParserNG Turbo can do( with MatrixTurboEvaluator), in terms of memory allocation optimization and matrix evaluation speed, but it ensures that
Matrix Algebra is fully available alongside matrix functionality on ParserNG Standard also.

### Parser 1.1.5 has been released on maven-central!
#### ParserNG Turbo: Zero-Allocation Optimization Pass
We have completely refactored the runtime variable mapping layer inside `ScalarTurboEvaluator1` and `ScalarTurboEvaluator2`. 

* **Runtime Remapping Eliminated:** Variable array positions are now baked directly into the `MethodHandle` topology at compile-time. 
The runtime engine now evaluates expressions by reading straight from the user's input arrays.
* **30%+ Evaluation Speed Burst:** Microbenchmarks show arithmetic evaluation speeds dropping from ~18ns down to **~12.2ns**, pulling within arm's reach of raw native Java performance (~6.4ns).
* **Flat Memory Profile:** GC allocation churn on hot evaluation paths remains at **absolute zero**. 
This guarantees stutter-free performance during heavy graph plotting or the soon-coming multi-million step differential equation loops.


### Parser 1.1.4 has been released on maven-central!
Version 1.1.4 squashes a bug where an over active syntax checker disables nested stats functions e.g. sort(3,1,5,listsum(4,12,18,-9),5,2,31,4) returns a syntax error Added tests


### Parser 1.1.3 has been released on maven-central!
Fixed validation bugs in parser, made relevant matrix maethods support algebraic operations, like A*invert(B) etc, fixed bad bugs in flat matrix turbo implementations and optimized them further. Added tests

### Parser 1.1.2 has been released on maven-central!
1. Fixes bugs and makes `MatrixTurboEvaluator` natively support turbo execution of the rot function. Note that the `ScalarTurboEvaluator`s already support it.
2. Also `FastCompositeExpression` is now aware of its compiler as it now sports a `getCompiler` default method(which can be overriden to specify the turbo class that compiled it) 

### Parser 1.1.1 has been released on maven-central!
Implemented version retrieval for ParserNG

### Parser 1.1.0 has been released on maven-central!
Bug fixes in Rotor and ErrorLog. Matrix of Points upgrade for Rotor

### Parser 1.0.8 has been released on maven-central!
Bug fixes and more Android compatibility issues resolved.

### Parser 1.0.6 has been released on maven-central!
Bug fixes and Android compatibility issues resolved.


### Parser 1.0.5 has been released on maven-central!
Features bug fixes and optimizations in the scanning/semantic analysis stages.

### Parser 1.0.4 has been released on maven-central!
This version features various optimizations and turbo capability for the Function
class.

Parser 1.0.3 has been released on maven-central!
Maintaining the industry standard besting speeds of v1.0.x, it adds the functionality of rotational geometry.
In v1.0.3, you can use the rotor function, `rot` to rotate raw points in 3D space and other functions such as curves, lines, surfaces(both plane and curved) and 3D equations of all sorts.

### ParserNG 1.0.2 has been released on maven-central!
Version 1.0.2 retains the wild speeds of Version 1.0.1. Adds an extra widening technique of variable passing to the Turbo mode,
In addition to the current method of array based passing. The widening technique can be sometimes faster than the array based methods,
but their speed profiles and memory profiles are similar. Its weakness though is that it cannot use more than 63 variables per expression, whereas the array based approach allows in theory any number up to the max integer size.


### ParserNG 1.0.1 has been released on maven-central!
This close update ensures that Turbo mode's  memory profile stays close to that of the normal mode, which is, nigh zero.


### ParserNG 1.0.0 has been released on maven-central!
The library has finally come of age with the introduction of its Turbo mode, which offers a massive speed boost over its normal mode.
The nomal mode already beats famous libraries like exp4J, and rivals Janino, the widely acclaimed Gold Standard of Java math parser speed measurements, very closely

 
### ParserNG 0.2.5 has been released on maven-central

1. Functions like `intg`, `root`, `t_root` and `quadratic` have been fixed and are working well. .

2. Frame based args passing is used to milk to milk the last drops of performance during iterations.

3. Constant folding and strength reduction make the evaluation process feel much faster.

4. The `print` function that can be used to view the contents of an `EvalResult` which the `solveGeneric()` method returns is also functional.

5. If you need a rich, fully featured parser that can do 3 million to 10 million evaluations per second, ParserNG v0.2.5 is the one for you.

### ParserNG 0.2.4 has been released on maven-central!

### ParserNG 0.2.4 drives the limits of expression interpretation velocity even further than all previous versions, beating many lighterweight and fast Java math parsers(interpreted) in many benchmarks.
Check [ParserNG-Wars](https://github.com/gbenroscience/ParserNG) for some shootouts between ParserNG and other parsers, both handrafted benchmarks and JMH based ones

ParserNG evaluates expressions at almost the speed at which the expressions would run if they were compiled statements in Java code. Typical values for moderate expression evaluation speeds are between `85ns`(algebraic expressions e.g.`((12+5)*3 - (45/9))^2` to `176ns`(methods with trig. functions, e.g. `(sin(3) + cos(4 - sin(2))) ^ (-2))`.


Applications that need 5 million to 10 million points generated per second would benefit from `ParserNG v0.2.4`
 

### **ParserNG 0.2.3** has been released on maven central!<br>
comes with couple microsecond (on decent hardware and) expression solving ability while maintaining its full feature stack. Graphing feels butter-smooth and iterations shouldn't feel so iterative. 

At 5 microsecond, moderately complex expressions such as
```Java
String s6 = "5*sin(3+2)/(4*3-2)";
```
 can be evaluated almost 200 thousand times per second.

 The model of a `Matrix` has also been optimized to use a 1D array internally. This makes it faster due to memory locality of Matrix data.

 Also, we support eigenvalues and eigenvectors as inbuilt methods, so enjoy! 

### ParserNG 0.2.2 has been released on maven-central!**


## What's new?

1. Breaking change!!! Package name change. **ParserNG** now has a proper root package name, com.github.gbenroscience
2. Really cool!!! -> Speed upgrades of up to 10x to 40x in the base algebraic expression parser. This means more speedy and energy efficient iterative calculations and graphing!
3. Very nice to have -> **ParserNG** now comes with a Java platform agnostic graphing capability. All the developer needs to do is to implement 2 interfaces: 
```java
com.github.gbenroscience.math.graph.DrawingContext
``` 
```java
com.github.gbenroscience.math.graph.AbstractView
``` 

and pass the instance to  
```java
com.github.gbenroscience.math.graph.Grid
```

So you can use the same codebase to plot graphs on Android, JavaFX, Swing and other Java platforms. We have made the required implementations of DrawingContext(so called adapters) for the most popular Java platforms(Android, Swing and JavaFX) available in [GRAPHING.md](./GRAPHING.md)
If you have more platforms in mind, we will be excited to have you contribute the code and make **ParserNG** even more versatile.
Enjoy!

Fun fact: here are some interactions with ParserNG hosted on maven-central within Dec 2025 and Feb 11 2026:
We had 970 total downloads from 193 unique sources across 82 companies
[See here](./maven-central-3-month-data.png)
