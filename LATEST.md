# ParserNG

### ParserNG 3.0.4 is out on maven-central!
This update strengthens the differential equation engine by adding systems solving capability explicitly
in the `diffeqn` and `diffeqnPath` paths. The engine has also been given a battery of tests, 120 in number
to check the engine for correctness and to give users an insight into the usage.
Equally important in this release is the emergence of the `ARRAY` type in ParserNG.
This is the piece that made the systems solving ability possible in ParserNG.
# ParserNG 3.0.4 🧮⚡

[![Maven Central](https://img.shields.io/maven-central/v/com.github.gbenroscience/parser-ng.svg?style=flat-square&color=blue)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Downloads](https://img.shields.io/badge/Downloads-11k%2B-brightgreen?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Growth](https://img.shields.io/badge/Growth-%2B250%25%20(90%20days)-orange?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![JDK Compatibility](https://img.shields.io/badge/JDK-8%20to%2026%2B-red?style=flat-square)](https://www.oracle.com/java/)

> **The fastest pure-Java math runtime, now with GPU (CUDA and OpenCL) bulk evaluators and a fully open-sourced Vector API (SIMD) kernel. Zero JNI. Zero native binaries. Zero bytecode-safety risk.**

**ParserNG 3.0.4 is live**, and it's the biggest release yet: your math expressions can now run on the GPU, CUDA or OpenCL, your choice, through nothing but standard Java. No driver bundling, no native wrapper to compile, no JNI surface to audit. And the SIMD engine that used to be an Enterprise-only feature is now fully open source.


**ParserNG** is an ultra-high-performance mathematical runtime built for modern JVM workloads, real-time plotting pipelines, financial modeling, and deep learning activation functions (SwiGLU, GELU, and the rest of the Transformer toolkit). By adopting a hardware-aligned, fast-interpreted memory model instead of risky dynamic bytecode generation, ParserNG eliminates classloader bloat, protects your runtime from native segmentation faults, and dramatically simplifies your Software Bill of Materials (SBOM) compliance posture.

Five execution tiers, one parser, one syntax. Pick the tier that matches how hot your loop is, you never rewrite the expression to move up a tier.
 
---

## 🏢 Globally Trusted

ParserNG powers mission-critical math visualization, analytical pipelines, and platform tooling at **167 organizations** around the globe.

### Need Enterprise-Grade Confidence?

Book a **free 30-minute introductory call** to discuss your production needs. Ongoing consulting, custom development, and support engagements are available after the initial conversation.

**[Schedule a call →](https://calendly.com/gbenroscience/30min)**
---

## 🔥 Start here: GPU, SIMDEngineEvaluator, SIMDVectorTurboEvaluator

This is the part of ParserNG worth losing sleep over, so it goes first.

### GPU bulk evaluation, CUDA and OpenCL, zero native code

One kernel, two backends, your choice of double or genuinely native float32 (not double silently upcast, a lot of consumer GPUs run fp64 at a fraction of their fp32 rate, so faking float32 would defeat the entire point). Auto-detects whichever backend is installed, or you pick explicitly:

```java
MathExpression me = new MathExpression("3*cos(x-2)+ln(3*x^3-5*x-4*tan(x))");
VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {   // auto-picks CUDA, falls back to OpenCL
    double[] flat = /* your sample buffer, column-major per variable slot */;
    double[] out  = new double[flat.length];
    gpu.applyBulk(flat, out);          // full double precision
    // gpu.applyBulk(floatIn, floatOut) // native float32 kernel -- real throughput gain, not a cast
}
```

Multiple GPUs installed? Select by vendor, not driver-string guesswork:

```java
OpenClCompositeExpression.selectDevice(OpenClCompositeExpression.GpuVendor.AMD);
// or: OpenClCompositeExpression.listAvailableDevices().forEach(System.out::println);
```

And MemorySegments allow for zero-copy!

```Java

import com.github.gbenroscience.gpu.opencl.OpenClCompositeExpression;
import com.github.gbenroscience.gpu.opencl.OpenClExpressionBridge;
import com.github.gbenroscience.gpu.opencl.OpenClKernelSource;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import static org.junit.jupiter.api.Assertions.*;


  @Test
 void memorySegmentOverloadMatchesFloatArrayOverload() throws Throwable {
        MathExpression me = new MathExpression("2*x^2-3*x+1");
        VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

        try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte); Arena arena = Arena.ofConfined()) {

            int dataSize = 2_000_000;
            float[] flat = new float[dataSize];
            for (int i = 0; i < dataSize; i++) {
                flat[i] = (float) (-25.6 + i * 0.1);
            }

            float[] outViaArray = new float[dataSize];
            long tArray = System.nanoTime();
            gpu.applyBulk(flat, outViaArray);
            tArray = System.nanoTime() - tArray;
            System.out.println("float[] path: " + (tArray / 1000) + " us");

            MemorySegment inSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(flat, 0, inSeg, ValueLayout.JAVA_FLOAT, 0, dataSize);

            long tSeg = System.nanoTime();
            gpu.applyBulkF32(inSeg, outSeg);
            tSeg = System.nanoTime() - tSeg;
            System.out.println("MemorySegment path (f32): " + (tSeg / 1000) + " us");

            float[] outViaSegment = new float[dataSize];
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0, outViaSegment, 0, dataSize);

            assertArrayEquals(outViaArray, outViaSegment, 0.0f,
                    "float[] and MemorySegment (applyBulkF32) overloads diverged for identical input -- "
                    + "bug is in the staging/copy layer, not the kernel");
        }
    }
```

### SIMDEngineEvaluator, CPU-pinned parallelism (JDK 22+)

The most efficient CPU-bound tier ParserNG has. CPU pinning (best on Linux) means 2 workers on 2 cores get you ~1.8×–2.0× the throughput of 1 worker on 1 core, real scaling, not the diminishing returns you get from unpinned thread pools fighting the scheduler.

```java
MathExpression me = new MathExpression("3*sin(x)*cos(y)+sqrt(abs(x*y))");
var evaluator = new SIMDEngineEvaluator(me).compile();

double[][] inputs = new double[2][dataSize]; // x, y
double[] out = new double[dataSize];

evaluator.applyBulkParallel(inputs, out, 2); // 2 pinned workers
```


Feel free to use the SIMDEngineEvaluator with MemorySegments also, for greater throughput - 
```Java

   /**
     * Test API Call: Parallel MemorySegment Bulk Execution Uses
     * Arena.ofShared() to allow worker threads to access the memory segments.
     */
    @Test
    public void testMathematicalPrecisionVsNativeMemorySegmentBulkParallel() throws Throwable {
        MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
        // Instantiate with 4 workers to ensure parallel execution pool is created
        var evaluator = SIMDEngineEvaluator.getEvaluator(me, 2);// the second argument is the number of workers to use.
        //If not specified, the worker count
        //is equal to the number of cpus. NOTE: parallel processing only occurs when applyBulkParallel is called

        logDetails(me, evaluator, !active);

        // Use a dataset large enough to exceed PARALLEL_OPS_THRESHOLD
        long totalElements = 300_000_00L;
        int varCount = 3; // x1, x2, x3
        double start = System.nanoTime();
        // MUST use shared arena for multi-threaded memory access
        try (Arena arena = Arena.ofShared()) {
            MemorySegment inputSegment = arena.allocate(varCount * totalElements * 8L);
            MemorySegment outputSegment = arena.allocate(totalElements * 8L);
            System.out.println("Started loading input data");
            for (long i = 0; i < totalElements; i++) {
                double x1Val = 1.5 + (i * 0.001);
                double x2Val = 2.0 + (i * 0.005);
                double x3Val = 0.5;

                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((0 * totalElements) + i) * 8L, x1Val);
                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((1 * totalElements) + i) * 8L, x2Val);
                inputSegment.set(ValueLayout.JAVA_DOUBLE, ((2 * totalElements) + i) * 8L, x3Val);
            }
            double t0 = System.nanoTime() - start;
            System.out.println("Done loading input data in " + t0 + "ns");

            start = System.nanoTime();
            // Execute Parallel Segment Path
            evaluator.applyBulkParallel(inputSegment, outputSegment);
            double t1 = System.nanoTime() - start;
            System.out.println("evalTime = " + t1 + "ns");

            System.out.println("Started testing output results");
            start = System.nanoTime();
            for (long i = 0; i < totalElements; i++) {
                double x1 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((0 * totalElements) + i) * 8L);
                double x2 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((1 * totalElements) + i) * 8L);
                double x3 = inputSegment.get(ValueLayout.JAVA_DOUBLE, ((2 * totalElements) + i) * 8L);

                double expected = (1.0 / (x1 * Math.sqrt(2.0 * 3.14159)))
                        * Math.exp((-Math.pow((x2 - x3), 2.0)) / (2.0 * Math.pow(x1, 2.0)));

                double actual = outputSegment.get(ValueLayout.JAVA_DOUBLE, i * 8L);
                assertEquals(expected, actual, EPSILON, "SIMD MemorySegment (Parallel) math drifted at index: " + i);
            }
            double t2 = System.nanoTime() - start;
            System.out.println("Ended testing output results in "+t2+"ns");
        }
    }

```

`SIMDCommandTurboEvaluator` sits right beside it in `parser-ng-gpu-simd` with the same CPU-pinned parallel model, `SIMDEngineEvaluator` is typically a few nanoseconds/op faster, so it's the default recommendation.

### SIMDVectorTurboEvaluator, explicit Vector API (JDK 21+)

Maps your expression straight to 256/512-bit AVX lanes via `jdk.incubator.vector`, not just hoping the JIT auto-vectorizes:

```java
MathExpression me = new MathExpression("4*x+3*sin(5+x^2)");
var evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression)
        new SIMDVectorTurboEvaluator(me).compile();

double[][] inputs = new double[1][dataSize];
double[] out = new double[dataSize];
evaluator.applyBulk(inputs, out);
// evaluator.applyBulkParallel(inputs, out); // works, but JDK 21 has no CPU pinning -- SIMDEngineEvaluator scales better
```

`VectorTurboEvaluator` (the non-`SIMD`-prefixed sibling) takes the opposite bet: no explicit Vector API calls at all, just memory layout and loop shape engineered to *coerce* the JIT into auto-vectorizing. Use it where the Vector API's incubator status is a concern; use `SIMDVectorTurboEvaluator` everywhere else.

---

## 🧬 The Evolution of ParserNG

| Version | What shipped |
| :--- | :--- |
| **< 1.0.0** | `MathExpression`, the interpreter. ParserNG Standard. |
| **1.0.0 – 1.x** | Turbo tier arrives: `ScalarTurboEvaluator1` (variable args as an array), `ScalarTurboEvaluator2` (variable args as widened primitives internally), and `MatrixTurboEvaluator`, all built on `MethodHandles`. |
| **2.0.0 – 2.x** | Bulk evaluation, via mechanical sympathy *and* SIMD. `VectorTurboEvaluator` coerces auto-vectorization through code shape alone; `SIMDVectorTurboEvaluator` forces it via the explicit Vector API. Both support `applyBulkParallel(in, out)`, but JDK 21 has no CPU pinning, capping the parallel win. |
| **3.0.4** | `SIMDEngineEvaluator` and `SIMDCommandTurboEvaluator` add CPU pinning (best on Linux): 2 workers on 2 cores ≈ 1.8×–2.0× the work of 1 worker on 1 core. `SIMDEngineEvaluator` edges out `SIMDCommandTurboEvaluator` by a few ns/op. Both live in **`parser-ng-gpu-simd`** (JDK 22+), the module that also houses the star of this release: native **GPU bulk evaluators for CUDA and OpenCL**. |

Same `MathExpression` syntax at every tier. You scale up by choosing a different evaluator, never by rewriting the expression.

---

## 📊 Performance & Throughput Profiles

### Throughput Horizons
| Execution Mode | Throughput Capacity (Evaluations / Sec) - on a 2015, Dell Inspiron 5759 - Processor: Intel(R) Core(TM) 76500U CPU @2.50GHz (4 CPUs), ~2.6GHz |
| :--- | :--- |
| **Standard Mode** | 3,000,000 – 10,000,000 |
| **Turbo Mode** | 10,000,000 – 90,000,000 |
| **Bulk Turbo (Single Core)** | **200,000,000+** |
| **Bulk Turbo (Parallel Workers)** | **Even Higher (Hardware Saturation)** |
| **GPU Bulk Turbo (Massive Parallelism)** | **Highest (Hardware Saturation)** |

### Amortized Activation Latencies (JDK 21+)
* **SwiGLU Kernel:** `1.8 ns / element`
* **GELU Kernel:** `2.1 ns / element`
* **Custom Expressions:** Highly scalable, consistently executing multiple times faster than standard Janino scalar compilations on single-core setups.

Full measured breakdowns, including GPU throughput at scale, live in [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md).

---

## ✨ Key Capabilities

* **Calculus & Advanced Algebra:** Symbolic differentiation (`diff`), automatic differentiation (`autodiff`) to arbitrary order, numerical integration (`intg`), and full matrix algebra, determinants, `eigvalues`, `adjoint`, `cofactor`, `echelon`, `linear_sys`, matrix division.
* **Vectorized Orchestration:** Hardware vector registers via `jdk.incubator.vector`, with a smooth Android-compliant scalar fallback.
* **Versatile Execution Architecture:** Logical/conditional expressions (`if(cond, a, b)`), user-defined functions (`f(x,y)=...` and `@(x,y)...` anonymous forms), equation root solvers (`quadratic`, `t_root` for cubics via Tartaglia, general `root`), point/line/function rotation (`rot`), and real-time geometric plotting buffers.
* **Zero Dependencies:** Portable and self-contained across Java SE, Android, and legacy JavaME profiles.

---

## 🗃️ New in 3.0.4: the `ARRAY` type

ParserNG has always had `@(dim)(...)` vector/matrix literals, but those are strictly numeric, backed by a `Matrix`. 3.0.4 adds a sibling: the **`ARRAY`** type, using the exact same `@(dim)(...)` syntax, but able to hold a genuine *mix* of content, numbers and strings side by side, not just numbers.

```java
MathExpression m = new MathExpression("a=@(4)('3.14', 5, \"I am here\", 32.34)");
m.solve();

String[] array = FunctionManager.lookUp("a").getArray();
// array = ["3.14", "5", "I am here", "32.34"]
```

An array literal accepts both single- and double-quoted strings (`'3.14'` and `"I am here"` are equally valid) alongside bare numeric literals, ParserNG figures out on its own whether the whole `@(dim)(...)` is a pure numeric `VECTOR` (still `Matrix`-backed, unchanged from before) or a mixed `ARRAY`, and compiles it accordingly. Like every other named function-valued literal, an `ARRAY` is registered with `FunctionManager` under its assigned name and retrieved the same way any `Function` is, `getArray()` sits right alongside the existing `getMatrix()` accessor, always returning `String[]`, since that's the one representation that can hold whatever mix of numbers and strings the literal contained.

**This is what makes the differential-equation engine's system-solving support possible.** Each equation in an explicit system (see above) is a full algebraic expression, not a constant, and not something ParserNG can evaluate eagerly the way a plain number can. Wrapping the set of equations in an `ARRAY` of quoted strings is what lets ParserNG treat "here are N equations" as a single, ordinary, eagerly-resolvable literal argument, the same trick that already worked for a numeric `y0` vector, extended to a type that can hold text.

---

## 🧮 Solving Differential Equations

ParserNG solves ODEs, single equations, systems of equations, and higher-order equations, as a first-class expression, not a bolted-on API. Four functions cover it: `diffeqn` (endpoint), `diffeqnPath` (full trajectory), `diffeqnHO` and `diffeqnPathHO` (the higher-order equivalents, for equations involving `y''`, `y'''`, and beyond).

Five solver methods are available, spanning the usual accuracy/stiffness tradeoff:

| Method | Order | Stiff-safe? | Use it for |
| :--- | :--- | :--- | :--- |
| `euler` | 1st | No | Real-time graphics, particle sims, speed over precision |
| `rk4` | 4th | No | General-purpose default, solid accuracy, no adaptive bookkeeping |
| `rk45` | 4th/5th, adaptive | No | Behavior that varies across the interval; industry-standard adaptive stepping |
| `implicit_euler` | 1st | **Yes** | Stiff systems where stability matters more than tight accuracy |
| `bdf2` | 2nd | **Yes** | Stiff systems needing better accuracy than `implicit_euler` at the same stability |

```java
// Endpoint only, using RK4 defaults
MathExpression me = new MathExpression("diffeqn(y[1] + 2*y[0], 0, 1, 5)");
me.solve();

// Full trajectory, resampled to 50 evenly-spaced points
MathExpression path = new MathExpression(
    "diffeqnPath(y[1] + 2*y[0], 0, 1, 5, 0.01, rk4, 50)");
path.solve();

// A 3rd-order equation, stiff-safe BDF2, full state trajectory (t, y, y', y'')
MathExpression ho = new MathExpression(
    "A=diffeqnPathHO(3*x*sin(x)*y[3]+4*x*y[2]+3*ln(x)*y[1]+4*y[0], 1, @(1,3)(1, 0, 0), 3, 0.01, bdf2, state)");
ho.solve();
FunctionManager.lookUp("A").getMatrix().print();
```

Results assign to a variable or matrix just like any other ParserNG expression, but a `diffeqn`-family call must always be the *entire* input expression (`A = diffeqn(...)` is fine; `sin(diffeqn(...))` is not, see the docs for why).

### New in 3.0.4: explicit systems of coupled equations

`diffeqn` and `diffeqnPath` now solve genuine systems, several coupled first-order equations, not just one, by passing an **array of quoted equation strings** in place of a single equation:

```java
// Lotka-Volterra predator-prey dynamics: two coupled equations, solved together
MathExpression system = new MathExpression(
    "diffeqn(@(2)(\"y[2]-(0.6*y[0]-0.03*y[0]*y[1])\", \"y[2]-(-0.9*y[1]+0.02*y[0]*y[1])\"), " +
    "0, @(1,2)(30, 4), 20, 0.01, rk4)");
double[] result = (double[]) system.solve(); // populations at t = 20
```

**The one rule worth knowing up front:** every equation in the array divides out `y[n]`, the system's *total* component count, the same symbol for every row, not each equation's own index. It's the identical convention `diffeqnHO` already uses for a single higher-order equation, just applied to N equations independently rather than one. `diffeqnHO`/`diffeqnPathHO`, by contrast, remain single-equation-only and will reject the array form outright, the two families don't mix.

Full syntax reference, argument-by-argument breakdown, the `y[n]` convention explained in depth, and per-method accuracy/stability nuances: [DIFF_ENGINE.md](parser-ng/DIFF_ENGINE.md).

---



## 📐 Examples Across Every Backend

Same expression syntax, five different execution tiers. Pick based on how hot the loop is.

### 1. Standard, `MathExpression`, direct interpretation

No compile step. Right for one-off evaluations, or expressions that change shape every call.

```java
MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,5)");
System.out.println(me.solve());

// Conditionals and multiple return types come free:
MathExpression cond = new MathExpression("if(3*x+7>5, sin(x), -3)");
MathExpression.EvalResult r = cond.solveGeneric(-42.0);   // r.scalar, r.vector, r.matrix, r.boolVal, r.textRes
```

### 2. Turbo Scalar, `ScalarTurboEvaluator1` / `ScalarTurboEvaluator2`

`TurboEvaluatorFactory` picks the right one for you; reach for a specific evaluator directly when you already know your shape (e.g. point/line/function rotation compiles cleanly on `ScalarTurboEvaluator1`).

```java
MathExpression me = new MathExpression("rot(@(1,3)(0,2,0), pi, @(1,3)(0,0,0), @(1,3)(1,0,0))");
FastCompositeExpression turbo = new ScalarTurboEvaluator1(me).compile();
double result = turbo.applyScalar(new double[0]);

// Or let the factory decide:
FastCompositeExpression auto = TurboEvaluatorFactory.getCompiler(me).compile();
```

### 3. Turbo Matrix, `MatrixTurboEvaluator`

Matrices are first-class function values. Solve, invert, decompose, compiled.

```java
Matrix m = new Matrix(coefficientData, n, n + 1);
m.setName("M");
new Function(m); // registers M for lookup by name

MathExpression expr = new MathExpression("linear_sys(M)");
FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(expr).compile();
Matrix solution = turbo.applyMatrix(new double[0]);
```

### 4. SIMD, `VectorTurboEvaluator` / `SIMDVectorTurboEvaluator` (JDK 21+)

```java
MathExpression me = new MathExpression(
    "(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
var evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression)
        new SIMDVectorTurboEvaluator(me).compile();

double[][] inputs = new double[3][totalElements]; // x1, x2, x3
double[] out = new double[totalElements];
evaluator.applyBulk(inputs, out); // tail elements auto-masked if totalElements isn't lane-aligned
```

### 5. GPU, OpenCL / CUDA (`parser-ng-gpu-simd`, JDK 22+)

```java
MathExpression me = new MathExpression("2*x^2-3*x+1");
VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte, GpuBackend.OPENCL)) {
    float[] in = /* ... */;
    float[] out = new float[in.length];
    gpu.applyBulk(in, out); // native float32 kernel, no double bridging
}
```

Full GPU quick-start, including multi-vendor device selection, is at the top of this document.

---

## 📦 Installation & Configuration

Module choice maps directly onto the evolution table above, pick the highest tier you need; each module pulls in everything below it.

### For Standard Android or Legacy Pre-JDK 21 Runtimes

Core interpreter, Turbo scalar/matrix tiers, and `BulkTurboEvaluator` for mechanically-sympathetic bulk loops with an inbuilt worker system:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.0.4</version>
</dependency>

```
```Java
BulkTurboEvaluator.BatchedVectorCompositeExpression evaluator =
    (BulkTurboEvaluator.BatchedVectorCompositeExpression) new BulkTurboEvaluator(me).compile();
double[] out = new double[1];
evaluator.applyBulk(new double[]{5, 4, 1}, out);
```

### For Modern JDK 21+ SIMD Vector Acceleration Environments

Adds `VectorTurboEvaluator` and `SIMDVectorTurboEvaluator`:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.0.4</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>3.0.4</version>
</dependency>

```

### For Modern JDK 22+ GPU and/or CPU-Pinned SIMD Environments

Adds `SIMDEngineEvaluator`, `SIMDCommandTurboEvaluator`, and the GPU bulk evaluators (CUDA + OpenCL):

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.0.4</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>3.0.4</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-gpu-simd</artifactId>
    <version>3.0.4</version>
</dependency>

```

### For Microbenchmarking and Profiling Verification Harnesses

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-bench</artifactId>
    <version>3.0.4</version>
</dependency>

```

---

## 📊 Comprehensive Feature Matrix

| Category | Supported Mathematical Tokens & Functions | Turbo Optimization Support |
| --- | --- | --- |
| **Arithmetic Operators** | `+`, `-`, `*`, `/`, `^`, `%`, `and`, `or`, `==`, `!=` | **Full Hardware Mapping** |
| **Trigonometric Functions** | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh` | **Full Hardware Mapping** |
| **Calculus Engines** | `diff` (Symbolic Engine), `intg` (Numerical Boundaries) | **Yes** |
| **Matrix Algebra** | `det`, `eigvalues`, `eigvec`, `adjoint`, `linear_sys` | **Optimized Linear Path** |
| **Statistical Functions** | `avg`, `variance`, `rms`, `sort` | **Yes** |

---

## 🏢 Trusted in Production by Global Organizations

ParserNG is used globally by **167 organizations** to power mission-critical math visualization, analytical pipelines, and platform tooling. Special thanks to Jiri Vanek for his contribution of `ExpandingParser` and its integration within the [jenkins-report-generic-chart-column](https://github.com/jenkinsci/report-generic-chart-column-plugin/) plugin ecosystem.

### Looking for Enterprise Guarantees?

Running any of this in production? Production infrastructures requiring predictable performance, safety assurances, and expert engineering access can upgrade to **ParserNG Enterprise**:

* **Priority Operational Support:** 24/48-hour SLAs for immediate bug tracking, profiling, and compliance vulnerabilities.
* **GraalVM Native Image Deployment:** Turn-key integration configurations for compiling down to lightning-fast native binary footprints.
* **Direct Consultative Access:** Architectural reviews, customized functions (extension), and hand-tailored vector/GPU kernel design directly from the author.

📧 **Contact Corporate Licensing & Consultations:** `gbenroscience@gmail.com`

☕ **Support Open Source Development:** [GitHub Sponsors Corporate Tiers](https://buymeacoffee.com/gbenroscience/membership), the SIMD engine you're reading about above was Enterprise-only until this release. Sponsorship is what funds moving the next tier out from behind that wall.

---

## 📚 Documentation & Technical Resources

* **Deep Dive Benchmarking Logs:** [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md), Comprehensive execution breakdowns versus competitor runtimes.
* **High-Fidelity Graphical Plotting:** [GRAPHING.md](parser-ng/GRAPHING.md), Render configuration rules for JavaFX, Swing, and Android surfaces.
* **Bulk Vectorization Blueprints:** [BULK.md](https://www.google.com/search?q=parser-ng/BULK.md), Optimization techniques for massive array processing.
* **Differential Equations:** [DIFF_ENGINE.md](parser-ng/DIFF_ENGINE.md), Full `diffeqn`/`diffeqnPath`/`diffeqnHO`/`diffeqnPathHO` syntax, solver selection guide, and result-capture patterns.
* **Release Artifact Logs:** [LATEST.md](LATEST.md), Change logs and technical notes for v3.0.4.
* [MORE.md](MORE.md), Even more to know
* [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md), Original readme for pre-1.0 versions with a lot of, still valid, examples

---

*Developed with mathematical rigor and mechanical sympathy by **Gbemiro Jiboye** (@gbenroscience).*Pre-dating the ARRAY type was the `VECTOR` type which was simply and array of numbers.
The `ARRAY` type allows one to store both numbers, strings etc.
It is also bound to the `Function` class. Soa a Funciton in ParserNG may be a VECTOR(a 1D matrix), a MATRIX, an ARRAY, or an ALGEBRAIC_EXPRESSION. You may acces it with the getArray() method on the Fucntion object.


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
