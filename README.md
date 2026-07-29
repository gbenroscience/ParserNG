# ParserNG 3.0.1 🧮⚡

[![Maven Central](https://img.shields.io/maven-central/v/com.github.gbenroscience/parser-ng.svg?style=flat-square&color=blue)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Downloads](https://img.shields.io/badge/Downloads-11k%2B-brightgreen?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Growth](https://img.shields.io/badge/Growth-%2B250%25%20(90%20days)-orange?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![JDK Compatibility](https://img.shields.io/badge/JDK-8%20to%2026%2B-red?style=flat-square)](https://www.oracle.com/java/)

> **The fastest pure-Java math runtime — now with GPU (CUDA and OpenCL) bulk evaluators and a fully open-sourced Vector API (SIMD) kernel. Zero JNI. Zero native binaries. Zero bytecode-safety risk.**

**ParserNG 3.0.1 is live**, and it's the biggest release yet: your math expressions can now run on the GPU — CUDA or OpenCL, your choice — through nothing but standard Java. No driver bundling, no native wrapper to compile, no JNI surface to audit. And the SIMD engine that used to be an Enterprise-only feature is now fully open source.


**ParserNG** is an ultra-high-performance mathematical runtime built for modern JVM workloads — real-time plotting pipelines, financial modeling, and deep learning activation functions (SwiGLU, GELU, and the rest of the Transformer toolkit). By adopting a hardware-aligned, fast-interpreted memory model instead of risky dynamic bytecode generation, ParserNG eliminates classloader bloat, protects your runtime from native segmentation faults, and dramatically simplifies your Software Bill of Materials (SBOM) compliance posture.

Five execution tiers, one parser, one syntax. Pick the tier that matches how hot your loop is — you never rewrite the expression to move up a tier.

---

## 🔥 Start here: GPU, SIMDEngineEvaluator, SIMDVectorTurboEvaluator

This is the part of ParserNG worth losing sleep over, so it goes first.

### GPU bulk evaluation — CUDA and OpenCL, zero native code

One kernel, two backends, your choice of double or genuinely native float32 (not double silently upcast — a lot of consumer GPUs run fp64 at a fraction of their fp32 rate, so faking float32 would defeat the entire point). Auto-detects whichever backend is installed, or you pick explicitly:

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

### SIMDEngineEvaluator — CPU-pinned parallelism (JDK 22+)

The most efficient CPU-bound tier ParserNG has. CPU pinning (best on Linux) means 2 workers on 2 cores get you ~1.8×–2.0× the throughput of 1 worker on 1 core — real scaling, not the diminishing returns you get from unpinned thread pools fighting the scheduler.

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

`SIMDCommandTurboEvaluator` sits right beside it in `parser-ng-gpu-simd` with the same CPU-pinned parallel model — `SIMDEngineEvaluator` is typically a few nanoseconds/op faster, so it's the default recommendation.

### SIMDVectorTurboEvaluator — explicit Vector API (JDK 21+)

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

`VectorTurboEvaluator` (the non-`SIMD`-prefixed sibling) takes the opposite bet: no explicit Vector API calls at all — just memory layout and loop shape engineered to *coerce* the JIT into auto-vectorizing. Use it where the Vector API's incubator status is a concern; use `SIMDVectorTurboEvaluator` everywhere else.

---

## 🧬 The Evolution of ParserNG

| Version | What shipped |
| :--- | :--- |
| **< 1.0.0** | `MathExpression` — the interpreter. ParserNG Standard. |
| **1.0.0 – 1.x** | Turbo tier arrives: `ScalarTurboEvaluator1` (variable args as an array), `ScalarTurboEvaluator2` (variable args as widened primitives internally), and `MatrixTurboEvaluator` — all built on `MethodHandles`. |
| **2.0.0 – 2.x** | Bulk evaluation, via mechanical sympathy *and* SIMD. `VectorTurboEvaluator` coerces auto-vectorization through code shape alone; `SIMDVectorTurboEvaluator` forces it via the explicit Vector API. Both support `applyBulkParallel(in, out)` — but JDK 21 has no CPU pinning, capping the parallel win. |
| **3.0.1** | `SIMDEngineEvaluator` and `SIMDCommandTurboEvaluator` add CPU pinning (best on Linux): 2 workers on 2 cores ≈ 1.8×–2.0× the work of 1 worker on 1 core. `SIMDEngineEvaluator` edges out `SIMDCommandTurboEvaluator` by a few ns/op. Both live in **`parser-ng-gpu-simd`** (JDK 22+) — the module that also houses the star of this release: native **GPU bulk evaluators for CUDA and OpenCL**. |

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

* **Calculus & Advanced Algebra:** Symbolic differentiation (`diff`), automatic differentiation (`autodiff`) to arbitrary order, numerical integration (`intg`), and full matrix algebra — determinants, `eigvalues`, `adjoint`, `cofactor`, `echelon`, `linear_sys`, matrix division.
* **Vectorized Orchestration:** Hardware vector registers via `jdk.incubator.vector`, with a smooth Android-compliant scalar fallback.
* **Versatile Execution Architecture:** Logical/conditional expressions (`if(cond, a, b)`), user-defined functions (`f(x,y)=...` and `@(x,y)...` anonymous forms), equation root solvers (`quadratic`, `t_root` for cubics via Tartaglia, general `root`), point/line/function rotation (`rot`), and real-time geometric plotting buffers.
* **Zero Dependencies:** Portable and self-contained across Java SE, Android, and legacy JavaME profiles.

---

## 📐 Examples Across Every Backend

Same expression syntax, five different execution tiers. Pick based on how hot the loop is.

### 1. Standard — `MathExpression`, direct interpretation

No compile step. Right for one-off evaluations, or expressions that change shape every call.

```java
MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,5)");
System.out.println(me.solve());

// Conditionals and multiple return types come free:
MathExpression cond = new MathExpression("if(3*x+7>5, sin(x), -3)");
MathExpression.EvalResult r = cond.solveGeneric(-42.0);   // r.scalar, r.vector, r.matrix, r.boolVal, r.textRes
```

### 2. Turbo Scalar — `ScalarTurboEvaluator1` / `ScalarTurboEvaluator2`

`TurboEvaluatorFactory` picks the right one for you; reach for a specific evaluator directly when you already know your shape (e.g. point/line/function rotation compiles cleanly on `ScalarTurboEvaluator1`).

```java
MathExpression me = new MathExpression("rot(@(1,3)(0,2,0), pi, @(1,3)(0,0,0), @(1,3)(1,0,0))");
FastCompositeExpression turbo = new ScalarTurboEvaluator1(me).compile();
double result = turbo.applyScalar(new double[0]);

// Or let the factory decide:
FastCompositeExpression auto = TurboEvaluatorFactory.getCompiler(me).compile();
```

### 3. Turbo Matrix — `MatrixTurboEvaluator`

Matrices are first-class function values. Solve, invert, decompose — compiled.

```java
Matrix m = new Matrix(coefficientData, n, n + 1);
m.setName("M");
new Function(m); // registers M for lookup by name

MathExpression expr = new MathExpression("linear_sys(M)");
FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(expr).compile();
Matrix solution = turbo.applyMatrix(new double[0]);
```

### 4. SIMD — `VectorTurboEvaluator` / `SIMDVectorTurboEvaluator` (JDK 21+)

```java
MathExpression me = new MathExpression(
    "(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
var evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression)
        new SIMDVectorTurboEvaluator(me).compile();

double[][] inputs = new double[3][totalElements]; // x1, x2, x3
double[] out = new double[totalElements];
evaluator.applyBulk(inputs, out); // tail elements auto-masked if totalElements isn't lane-aligned
```

### 5. GPU — OpenCL / CUDA (`parser-ng-gpu-simd`, JDK 22+)

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

Module choice maps directly onto the evolution table above — pick the highest tier you need; each module pulls in everything below it.

### For Standard Android or Legacy Pre-JDK 21 Runtimes

Core interpreter, Turbo scalar/matrix tiers, and `BulkTurboEvaluator` for mechanically-sympathetic bulk loops with an inbuilt worker system:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.0.1</version>
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
    <version>3.0.1</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>3.0.1</version>
</dependency>

```

### For Modern JDK 22+ GPU and/or CPU-Pinned SIMD Environments

Adds `SIMDEngineEvaluator`, `SIMDCommandTurboEvaluator`, and the GPU bulk evaluators (CUDA + OpenCL):

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.0.1</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>3.0.1</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-gpu-simd</artifactId>
    <version>3.0.1</version>
</dependency>

```

### For Microbenchmarking and Profiling Verification Harnesses

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-bench</artifactId>
    <version>3.0.1</version>
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

ParserNG is used globally by **618 organizations** to power mission-critical math visualization, analytical pipelines, and platform tooling. Special thanks to Jiri Vanek for his contribution of `ExpandingParser` and its integration within the [jenkins-report-generic-chart-column](https://github.com/jenkinsci/report-generic-chart-column-plugin/) plugin ecosystem.

### Looking for Enterprise Guarantees?

Running any of this in production? Production infrastructures requiring predictable performance, safety assurances, and expert engineering access can upgrade to **ParserNG Enterprise**:

* **Priority Operational Support:** 24/48-hour SLAs for immediate bug tracking, profiling, and compliance vulnerabilities.
* **GraalVM Native Image Deployment:** Turn-key integration configurations for compiling down to lightning-fast native binary footprints.
* **Direct Consultative Access:** Architectural reviews, customized functions (extension), and hand-tailored vector/GPU kernel design directly from the author.

📧 **Contact Corporate Licensing & Consultations:** `gbenroscience@gmail.com`

☕ **Support Open Source Development:** [GitHub Sponsors Corporate Tiers](https://buymeacoffee.com/gbenroscience/membership) — the SIMD engine you're reading about above was Enterprise-only until this release. Sponsorship is what funds moving the next tier out from behind that wall.

---

## 📚 Documentation & Technical Resources

* **Deep Dive Benchmarking Logs:** [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md) — Comprehensive execution breakdowns versus competitor runtimes.
* **High-Fidelity Graphical Plotting:** [GRAPHING.md](parser-ng/GRAPHING.md) — Render configuration rules for JavaFX, Swing, and Android surfaces.
* **Bulk Vectorization Blueprints:** [BULK.md](https://www.google.com/search?q=parser-ng/BULK.md) — Optimization techniques for massive array processing.
* **Release Artifact Logs:** [LATEST.md](LATEST.md) — Change logs and technical notes for v3.0.1.
* [MORE.md](MORE.md) — Even more to know
* [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md) — Original readme for pre-1.0 versions with a lot of, still valid, examples

---

*Developed with mathematical rigor and mechanical sympathy by **Gbemiro Jiboye** (@gbenroscience).*