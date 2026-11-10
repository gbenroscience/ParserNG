# ParserNG 3.1.0 🧮⚡

[![Maven Central](https://img.shields.io/maven-central/v/com.github.gbenroscience/parser-ng.svg?style=flat-square&color=blue)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Downloads](https://img.shields.io/badge/Downloads-11k%2B-brightgreen?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Growth](https://img.shields.io/badge/Growth-%2B250%25%20(90%20days)-orange?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![JDK Compatibility](https://img.shields.io/badge/JDK-8%20to%2026%2B-red?style=flat-square)](https://www.oracle.com/java/)

> **The fastest pure-Java math runtime, now with GPU (CUDA, Metal and OpenCL) bulk evaluators and a fully open-sourced Vector API (SIMD) kernel. Zero JNI. Zero native binaries. Zero bytecode-safety risk.**

**ParserNG** is an ultra-high-performance mathematical runtime built for modern JVM workloads: real-time plotting pipelines, financial modeling, and deep learning activation functions (SwiGLU, GELU, and the rest of the Transformer toolkit). By adopting a hardware-aligned, fast-interpreted memory model instead of risky dynamic bytecode generation, ParserNG eliminates classloader bloat, protects your runtime from native segmentation faults, and dramatically simplifies your Software Bill of Materials (SBOM) compliance posture.

Five execution tiers, one parser, one syntax, and now big data. Pick the tier that matches how hot your loop is; you never rewrite the expression to move up a tier.

ParserNG powers math visualization, analytical pipelines, and platform tooling at **167 organizations** around the globe. Need enterprise-grade confidence? Book a **free 30-minute introductory call** to discuss your production needs. Ongoing consulting, custom development, and support engagements are available after the initial conversation: **[Schedule a call](https://calendly.com/gbenroscience/30min)**.

---

## 🆕 What's new

The full change log for each release is in [LATEST.md](LATEST.md).

### 3.1.0: Expands the calculus capabilities by including a partial differential equations engine in pure Java.

### 3.0.9.3: Incubating OpenCL LLM Engine optimization and bug fixes to `MathScanner`, `RootFinder` and `Matrix#det`

### 3.0.9.2: strict mode

`MathExpression`, ParserNG's core class, now has a `strict` flag (an instance field). It decides what happens when an expression uses a variable or method name that has not been declared:

* `strict = false` (the default, so existing behaviour is preserved): an undeclared name is accepted as a variable initialised to zero.
* `strict = true`: an undeclared variable or method name throws a parser exception when the expression is constructed.

### 3.0.9: Parquet, and a faster constructor

* **`parser-ng-parquet`** joins the ParserNG family. It reads Parquet files, local and remote (for example on S3), at perhaps the highest rates available on the JVM: about 3x to 4x the rate of `parquet-java` itself.
* **`parser-ng-sql`** now supports `select` over Parquet files, local and remote.
* The base `MathExpression` constructor is much faster, and `BaseNFunction`, the `MathExpression` class extended to every number base from 2 to 36 inclusive, is now complete.
* A hidden allocation source that occasionally appeared in the SIMD bulk evaluators has been removed.
* The float32 SIMD bulk evaluators (`SIMDCommandF32`, `SIMDCommandSegmentF32`) passed 1.4 billion evaluations per second on the quadratic `x^2+2*x+1`, using just 2 workers on an Intel mobile Skylake processor (2015 or 2016 vintage) in a Dell Inspiron 5759.
* Please read the READMEs of `parser-ng-sql` and `parser-ng-parquet`, as there are many new updates.

### 3.0.8 and earlier: Arrow, SQL and the GPU

* Version 3.0.5 introduced float32 (Java's `float`) and float64 (Java's primitive `double`) bulk evaluators, alongside MemorySegment-based bulk evaluators for both precisions.
* **`parser-ng-arrow`** puts all of that to work: a true zero-copy bridge between ParserNG and Apache Arrow. You can evaluate any runtime string expression, filter rows, and project computed columns directly over Arrow's columnar memory, with no serialization, no staging buffers, and no leaving the Arrow buffer at all. In its SIMD mode (`ArrowBulkEvaluator`), it beats Apache Gandiva by 10 to 30 times on every transcendental function: `sin`, `cos`, `log`, `exp`, and the rest of the set. Gandiva still edges it out on hardware-intrinsic `sqrt` and plain arithmetic, by 1.5x to 2x, but turn on `ArrowBulkEvaluator`'s built-in parallelism and that gap is gone at just 2 workers.
* For workloads that want more than the CPU can give, `ArrowGpuBulkEvaluator` dispatches the same expressions straight to CUDA, OpenCL or Metal, with zero JNI and zero native binaries anywhere in your build.
* **`parser-ng-sql`**, introduced in 3.0.7 for users of `parser-ng-arrow`, received major speed and bug fixes in 3.0.8.
* 3.0.8 also added real query power to `parser-ng-arrow`: `filter`, `project` and `filterProject` (see [below](#-arrow-filter-project-and-filterproject)).
* 3.0.8 added `SIMDEngineEvaluator` and `SIMDCommandTurboEvaluator` with CPU pinning, the native GPU bulk evaluators, and the `ARRAY` type (see [the table below](#-the-evolution-of-parserng)).

---

## 🔥 Start here: GPU, SIMDEngineEvaluator, SIMDVectorTurboEvaluator

This is the part of ParserNG worth losing sleep over, so it goes first.

### GPU bulk evaluation: CUDA, OpenCL and Metal, zero native code

One kernel, three backends, your choice of double or genuinely native float32 (not double silently upcast: a lot of consumer GPUs run fp64 at a fraction of their fp32 rate, so faking float32 would defeat the entire point). ParserNG auto-detects whichever backend is installed, or you pick explicitly:

```java
MathExpression me = new MathExpression("3*cos(x-2)+ln(3*x^3-5*x-4*tan(x))");
VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte)) {   // auto-picks CUDA, falls back to OpenCL/Metal
    double[] flat = /* your sample buffer, column-major per variable slot */;
    double[] out  = new double[flat.length];
    gpu.applyBulk(flat, out);          // full double precision
    // gpu.applyBulk(floatIn, floatOut) // native float32 kernel: a real throughput gain, not a cast
}
```

Multiple GPUs installed? Select by vendor, not by driver-string guesswork:

```java
OpenClCompositeExpression.selectDevice(OpenClCompositeExpression.GpuVendor.AMD);
// or: OpenClCompositeExpression.listAvailableDevices().forEach(System.out::println);
```

`MemorySegment`s allow zero-copy transfers. The segment path and the `float[]` path give identical results:

```java
import com.github.gbenroscience.gpu.opencl.OpenClCompositeExpression;
import com.github.gbenroscience.gpu.opencl.OpenClExpressionBridge;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

MathExpression me = new MathExpression("2*x^2-3*x+1");
VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

try (OpenClCompositeExpression gpu = (OpenClCompositeExpression) OpenClExpressionBridge.from(vte);
     Arena arena = Arena.ofConfined()) {

    int dataSize = 2_000_000;
    float[] flat = new float[dataSize];
    for (int i = 0; i < dataSize; i++) {
        flat[i] = (float) (-25.6 + i * 0.1);
    }

    // float[] path
    float[] outViaArray = new float[dataSize];
    gpu.applyBulk(flat, outViaArray);

    // MemorySegment path (float32)
    MemorySegment inSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
    MemorySegment outSeg = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
    MemorySegment.copy(flat, 0, inSeg, ValueLayout.JAVA_FLOAT, 0, dataSize);
    gpu.applyBulkF32(inSeg, outSeg);

    // outSeg now holds exactly the values in outViaArray
}
```

### SIMDEngineEvaluator: CPU-pinned parallelism (JDK 22+)

The most efficient CPU-bound tier ParserNG has. CPU pinning (best on Linux) means 2 workers on 2 cores get you about 1.8x to 2.0x the throughput of 1 worker on 1 core: real scaling, not the diminishing returns you get from unpinned thread pools fighting the scheduler.

```java
MathExpression me = new MathExpression("3*sin(x)*cos(y)+sqrt(abs(x*y))");
var evaluator = new SIMDEngineEvaluator(me).compile();

double[][] inputs = new double[2][dataSize]; // x, y
double[] out = new double[dataSize];

evaluator.applyBulkParallel(inputs, out, 2); // 2 pinned workers
```

`SIMDEngineEvaluator` also works over `MemorySegment`s, for greater throughput. Parallel work only happens when you call `applyBulkParallel`, and the dataset must be large enough to exceed the parallel threshold. Use a shared arena so that the worker threads can all read the segments:

```java
MathExpression me = new MathExpression(
        "(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
var evaluator = SIMDEngineEvaluator.getEvaluator(me, 2); // 2 workers; omit the argument to use one per CPU

long n = 30_000_000L; // elements per variable
try (Arena arena = Arena.ofShared()) {
    MemorySegment in = arena.allocate(3 * n * 8L);   // x1, x2, x3: one block of n doubles per variable
    MemorySegment out = arena.allocate(n * 8L);

    for (long i = 0; i < n; i++) {
        in.set(ValueLayout.JAVA_DOUBLE, (0 * n + i) * 8L, 1.5 + i * 0.001); // x1
        in.set(ValueLayout.JAVA_DOUBLE, (1 * n + i) * 8L, 2.0 + i * 0.005); // x2
        in.set(ValueLayout.JAVA_DOUBLE, (2 * n + i) * 8L, 0.5);             // x3
    }

    evaluator.applyBulkParallel(in, out);
    double first = out.get(ValueLayout.JAVA_DOUBLE, 0);
}
```

`SIMDCommandTurboEvaluator` sits right beside it in `parser-ng-gpu-simd` with the same CPU-pinned parallel model. `SIMDEngineEvaluator` is typically a few nanoseconds per op faster, so it is the default recommendation.

### SIMDVectorTurboEvaluator: explicit Vector API (JDK 21+)

Maps your expression straight to 256 or 512-bit AVX lanes via `jdk.incubator.vector`, instead of hoping the JIT auto-vectorizes:

```java
MathExpression me = new MathExpression("4*x+3*sin(5+x^2)");
var evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression)
        new SIMDVectorTurboEvaluator(me).compile();

double[][] inputs = new double[1][dataSize];
double[] out = new double[dataSize];
evaluator.applyBulk(inputs, out);
// evaluator.applyBulkParallel(inputs, out); // works, but JDK 21 has no CPU pinning, so SIMDEngineEvaluator scales better
```

`VectorTurboEvaluator` (the non-`SIMD`-prefixed sibling) takes the opposite bet: no explicit Vector API calls at all, just memory layout and loop shape engineered to *coerce* the JIT into auto-vectorizing. Use it where the Vector API's incubator status is a concern; use `SIMDVectorTurboEvaluator` everywhere else.

---

## 🏹 Arrow: `filter`, `project` and `filterProject`

`parser-ng-arrow` evaluates ParserNG expressions directly over Apache Arrow batches. These are not toy expressions: they are the kind of thing that runs in production, and the kind an engineering manager cares about because it touches fraud losses, cloud spend or patient safety. They are not a SQL engine, and they are no replacement for Gandiva or DataFusion, but for math-heavy numeric expressions over Arrow batches they are the faster tool for the job. `filterProject` fuses row selection with column computation, so your projection never runs on a row that is about to be thrown away. When your expressions are float or float64 heavy, `parser-ng-arrow` is the faster choice, by a lot.

**`filter`, fintech: screening a payments batch for fraud review before a human ever sees it**

```java
try (ArrowBulkEvaluator highRisk = ArrowBulkEvaluator.compile("txn_amount > 10000 && risk_score > 0.8")) {
    VectorSchemaRoot flagged = highRisk.filter(paymentsBatch);
    // Out of a batch that might be millions of transactions wide, "flagged" holds only
    // the ones your fraud analysts actually need to look at, computed at SIMD speed.
}
```

**`project`, platform engineering: attaching a real dollar cost to every trace span, at fleet scale**

```java
try (ArrowBulkEvaluator costPerRequest =
             ArrowBulkEvaluator.compile("(cpu_ms * cpu_rate + mem_gb_s * mem_rate) / request_count")) {
    VectorSchemaRoot withCost = costPerRequest.project(traceBatch, "cost_per_request");
    // Every span in this minute's trace batch now carries its own "cost_per_request" column,
    // computed once over however many million rows your tracing pipeline just handed you:
    // the kind of number that ends up on a cloud-spend dashboard your VP actually reads.
}
```

**`filterProject`, healthcare and IoT: two ways to score early-warning risk from a stream of patient vitals**

The first is `ArrowExpressionEvaluators.filterProject`, an ad-hoc, SQL-`WHERE`/`SELECT`-shaped convenience. Every expression, predicate included, is plain text compiled fresh on this one call. That is the right shape when the alert rule is something a clinician configured at runtime and is not worth precompiling:

```java
VectorSchemaRoot atRisk = ArrowExpressionEvaluators.filterProject(
        vitalsBatch,
        "heart_rate > 120 || spo2 < 92",
        "patient_id", "heart_rate", "spo2",
        "0.4 * (heart_rate - 100) + 0.6 * (94 - spo2)");
// Only the patients whose vitals actually crossed the threshold survive the filter.
// "patient_id", "heart_rate" and "spo2" pass straight through unchanged, zero copy,
// recognized as bare column names rather than routed through the expression engine at all.
// The last one is computed fresh, only over the rows that survived, and comes back
// named after its own trimmed expression text. The result holds exactly these four
// columns, nothing else from vitalsBatch tags along uninvited.
```

The second is the precompiled instance-method form, shown in the `filter` and `project` examples above. It is the right shape once the same rule is going to run against every batch streaming off a bedside monitor, not just once:

```java
try (ArrowBulkEvaluator isAbnormal = ArrowBulkEvaluator.compile("heart_rate > 120 || spo2 < 92");
     ArrowBulkEvaluator earlyWarningScore =
             ArrowBulkEvaluator.compile("0.4 * (heart_rate - 100) + 0.6 * (94 - spo2)")) {

    VectorSchemaRoot scored = isAbnormal.filterProject(vitalsBatch, earlyWarningScore, "ews_score");
    // Same fusion guarantee (the score never runs on a normal reading), but the predicate
    // and the projection are compiled once and reused across every batch this monitor
    // streams, instead of being recompiled on every call as in the ad-hoc form above.
    // This form also keeps vitalsBatch's full schema and simply appends "ews_score"
    // as a new column, rather than returning only the columns you named.
}
```

---

## 🧬 The Evolution of ParserNG

| Version | What shipped |
| :--- | :--- |
| **Before 1.0.0** | `MathExpression`, the interpreter. ParserNG Standard. |
| **1.0.0 to 1.x** | Turbo tier arrives: `ScalarTurboEvaluator1` (variable args as an array), `ScalarTurboEvaluator2` (variable args as widened primitives internally), and `MatrixTurboEvaluator`, all built on `MethodHandles`. |
| **2.0.0 to 2.x** | Bulk evaluation, via mechanical sympathy *and* SIMD. `VectorTurboEvaluator` coerces auto-vectorization through code shape alone; `SIMDVectorTurboEvaluator` forces it via the explicit Vector API. Both support `applyBulkParallel(in, out)`, but JDK 21 has no CPU pinning, capping the parallel win. |
| **3.0.5** | float32 and float64 bulk evaluators, plus MemorySegment-based bulk evaluators for both precisions. |
| **3.0.7** | `parser-ng-sql`, introduced to help users of `parser-ng-arrow`, the zero-copy Apache Arrow bridge. |
| **3.0.8** | `SIMDEngineEvaluator` and `SIMDCommandTurboEvaluator` add CPU pinning (best on Linux): 2 workers on 2 cores give about 1.8x to 2.0x the work of 1 worker on 1 core. `SIMDEngineEvaluator` edges out `SIMDCommandTurboEvaluator` by a few ns/op. Both live in **`parser-ng-gpu-simd`** (JDK 22+), the module that also houses native **GPU bulk evaluators for CUDA, OpenCL and Metal**. Also new: `filter`, `project` and `filterProject` in `parser-ng-arrow`, and the `ARRAY` type. |
| **3.0.9** | `parser-ng-parquet`, `select` over Parquet in `parser-ng-sql`, a much faster `MathExpression` constructor, and a complete `BaseNFunction`. |
| **3.0.9.2** | The `strict` flag on `MathExpression`. |

Same `MathExpression` syntax at every tier. You scale up by choosing a different evaluator, never by rewriting the expression.

---

## 📊 Performance & Throughput Profiles

### Throughput horizons

Measured on a 2015 Dell Inspiron 5759 with an Intel Core i7-6500U CPU @ 2.50GHz (4 logical CPUs, about 2.6GHz).

| Execution mode | Throughput capacity (evaluations / sec) |
| :--- | :--- |
| **Standard Mode** | 3,000,000 to 10,000,000 |
| **Turbo Mode** | 10,000,000 to 90,000,000 |
| **Bulk Turbo (Single Core)** | **200,000,000+** |
| **Bulk Turbo (Parallel Workers)** | **Even higher (hardware saturation)** |
| **GPU Bulk Turbo (Massive Parallelism)** | **Highest (hardware saturation)** |

### Amortized activation latencies (JDK 21+)

* **SwiGLU kernel:** `1.8 ns / element`
* **GELU kernel:** `2.1 ns / element`
* **Custom expressions:** highly scalable, consistently executing multiple times faster than standard Janino scalar compilations on single-core setups.

Full measured breakdowns, including GPU throughput at scale, live in [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md).

---

## ✨ Key Capabilities

* **Calculus & advanced algebra:** symbolic differentiation (`diff`), automatic differentiation (`autodiff`) to arbitrary order, numerical integration (`intg`), and full matrix algebra: determinants, `eigvalues`, `adjoint`, `cofactor`, `echelon`, `linear_sys`, matrix division.
* **Differential equations:** `diffeqn`, `diffeqnPath`, `diffeqnHO` and `diffeqnPathHO` for single equations, higher-order equations and systems. See [DIFF_ENGINE.md](parser-ng/DIFF_ENGINE.md).
* **Vectorized orchestration:** hardware vector registers via `jdk.incubator.vector`, with a smooth Android-compliant scalar fallback.
* **Versatile execution architecture:** logical and conditional expressions (`if(cond, a, b)`), user-defined functions (`f(x,y)=...` and the anonymous `@(x,y)...` form), equation root solvers (`quadratic`, `t_root` for cubics via Tartaglia, general `root`), point, line and function rotation (`rot`), and real-time geometric plotting buffers.
* **Global and local function scopes:** give each `MathExpression` (or group of them) its own private function registry with `FunctionScope`, shadowing or isolating from the global one, with zero allocation on the hot paths. See [global and local function scopes](#-new-global-and-local-function-scopes).
* **Mixed-content arrays:** the `ARRAY` type holds numbers and strings side by side. See the `ARRAY` type section below.
* **Zero dependencies:** portable and self-contained across Java SE, Android, and legacy JavaME profiles.

---

## 🗃️ New in 3.0.8: the `ARRAY` type

ParserNG has always had `@(dim)(...)` vector and matrix literals, but those are strictly numeric, backed by a `Matrix`. 3.0.8 adds a sibling: the **`ARRAY`** type, using the exact same `@(dim)(...)` syntax, but able to hold a genuine *mix* of content: numbers and strings side by side, not just numbers.

```java
MathExpression m = new MathExpression("a=@(4)('3.14', 5, \"I am here\", 32.34)");
m.solve();

String[] array = FunctionManager.lookUp("a").getArray();
// array = ["3.14", "5", "I am here", "32.34"]
```

An array literal accepts both single and double-quoted strings (`'3.14'` and `"I am here"` are equally valid) alongside bare numeric literals. ParserNG figures out on its own whether the whole `@(dim)(...)` is a pure numeric `VECTOR` (still `Matrix`-backed, unchanged from before) or a mixed `ARRAY`, and compiles it accordingly. Like every other named function-valued literal, an `ARRAY` is registered with `FunctionManager` under its assigned name and retrieved the same way any `Function` is. `getArray()` sits right alongside the existing `getMatrix()` accessor and always returns a `String[]`, since that is the one representation that can hold whatever mix of numbers and strings the literal contained.

**This is what makes the differential-equation engine's system-solving support possible.** Each equation in an explicit system (see [DIFF_ENGINE.md](parser-ng/DIFF_ENGINE.md)) is a full algebraic expression, not a constant, and not something ParserNG can evaluate eagerly the way a plain number can. Wrapping the set of equations in an `ARRAY` of quoted strings lets ParserNG treat "here are N equations" as a single, ordinary, eagerly-resolvable literal argument. It is the same trick that already worked for a numeric `y0` vector, extended to a type that can hold text.

---

## 🔒 New: global and local function scopes

Until now every `MathExpression` in the process shared one function registry, `FunctionManager.FUNCTIONS`. That is still the default, and nothing changes for code that ignores this section. But when many expressions, tenants, plugins or tests must each define *their own* `f(x)`, a single shared registry means they overwrite one another. A **`FunctionScope`** is a private registry that sits in front of the global one.

```java
FunctionManager.add("f(x)=x+1");                        // global, exactly as before

FunctionScope mine = FunctionScope.create();            // a new, empty local scope
new MathExpression("f(x)=x*10;f(2)", mine).solve();     // "20.0"  defined in `mine`, shadowing the global f
new MathExpression("f(2)", mine).solve();               // "20.0"  `mine` remembers its own f
new MathExpression("f(2)").solve();                     // "3.0"   the global f was never touched
```

### The rules

| | |
|---|---|
| **Lookup** | The expression's scope first, then its parent, and so on up to the global scope. The first match wins, so a local function *shadows* a global one of the same name. |
| **Definition** | `f(x)=...`, `f=@(x)...`, matrices (`M=@(2,2)(...)`), vectors, arrays and anonymous functions all go into the expression's own scope, never into a parent. |
| **Deletion** | Only removes from the scope you are in. A local scope can never delete a global function, or a global variable. |
| **Lexical** | A function runs in the scope it was *defined* in, whoever calls it. A global `g(x)=f(x)+1` keeps using the global `f`, even when called from a scope that defines its own `f`. A local `g` that calls `f` uses the local `f`. This also holds when the scanner inlines `g`'s body into the caller: matrices and function handles named in the body are pinned to the function the body meant (via hidden per-scope aliases), so they cannot leak to a same-named definition in the caller's scope. |
| **No scope given** | The global scope, which is the old behaviour. `FunctionManager.FUNCTIONS` remains the global map. |

### Creating and using scopes

```java
// An expression that belongs to a scope
MathExpression me = new MathExpression("f(3)", mine);
me.solve();
me.getFunctionScope();                                  // `mine`

// A scope you can define things in directly
mine.add("g(x)=x^2");                                   // local
mine.solve("h(x)=g(x)+1; h(3)");                        // "10.0", and h is local too

// Where do names come from?
mine.lookUp("f");                                       // local f if there is one, else the global f
mine.lookUpLocal("f");                                  // only what `mine` itself defines
mine.localNames();                                      // names defined in `mine`
mine.remove("f");                                       // the global f is visible again
```

A few shapes, depending on how much you want to share:

```java
FunctionScope.create();                  // child of the global scope: sees global functions, adds its own
FunctionScope.create(parent);            // child of another scope: scopes nest
FunctionScope.isolated();                // no fallback at all: global user functions are invisible
```

Give each expression its own scope for full separation (100 expressions, 100 different `f`), or share one scope between many expressions that should agree on their definitions.

Code that predates scopes, or that calls the static `FunctionManager` API, can be pointed at a scope for a block:

```java
try (FunctionScope.Binding b = mine.activate()) {
    FunctionManager.add("k(x)=x+7");                    // lands in `mine`
    new MathExpression("k(1)").solve();                 // an expression created here joins `mine`
}                                                       // the previous scope is restored
```

### How it works

ParserNG resolves function names through the static `FunctionManager` in a couple of hundred places, so rather than thread a scope argument through all of them, **each thread has an active scope** and every `FunctionManager` method reads and writes it. A `MathExpression` that belongs to a scope makes it active while it parses, compiles and evaluates, then puts the previous one back. An expression created while a scope is active (a function body, a nested sub-expression) joins that scope automatically; that is what makes scoping lexical. Compiled (`compileTurbo()`) expressions carry their scope with them and keep working on any thread. The scanner's cached vocabulary is kept per scope, so a scope's function names are recognised without ever appearing in the global one.

### Performance

Scoping is built to cost nothing when unused and next to nothing when used.

* **No allocation, ever, on the hot paths.** Scope lookup, scope switching, and evaluation of scoped expressions all allocate zero bytes per call (checked by a test that measures per-thread allocation).
* **Unscoped code pays one volatile read.** Until the first local scope is created in the process, the only added cost is a single flag check per evaluation. Global compiled expressions run at the same ~5 ns as before.
* **Local lookups are lock-free.** A local scope is a `ConcurrentHashMap`, so reading it takes no monitor; a lookup is about 10 ns against about 24 ns for the global (synchronized) map.
* **Compiled expressions that call no user functions are returned unwrapped**, with no scope handling at all.

Measured on one machine (JDK 21, best of seven runs of two million calls); your numbers will differ, the proportions should not:

| Operation | Before | After |
|---|---|---|
| `solveGeneric`, global expression, no scopes in use | 12.3 ns | 12.9 ns |
| compiled global expression, `applyScalar` | 5.4 ns | 5.4 ns |
| compiled scoped expression, `applyScalar` | n/a | 5.5 ns |
| global function lookup | 23.7 ns | 23.5 ns |
| local function lookup | n/a | 10.2 ns |

Once a local scope exists anywhere in the process, evaluating an expression costs one extra thread-local read (about 3 ns), because a global expression has to be able to tell it is not running inside a local scope.

### Limits

* Only **functions** are scoped. Variables keep their existing model: a per-expression registry backed by the global `VariableManager`.
* As in the global registry, a function keeps what its body referred to **when it was defined**. If `pp(x)=f(x)+5` is defined and `f` is redefined afterwards, `pp` still uses the old `f`. This is unchanged from before; it applies inside a scope exactly as it does globally.
* Serialization does not carry the scope: a deserialized `MathExpression` is global.
* A `MathExpression` is still not thread-safe; give each thread its own, as before. Scopes themselves are safe to share.
* Code that reads `FunctionManager.FUNCTIONS` directly sees the **global** functions only.

---

## 📐 Examples Across Every Backend

Same expression syntax, five different execution tiers. Pick based on how hot the loop is.

### 1. Standard: `MathExpression`, direct interpretation

No compile step. Right for one-off evaluations, or expressions that change shape every call.

```java
MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(3,4,5)");
System.out.println(me.solve());

// Conditionals and multiple return types come free:
MathExpression cond = new MathExpression("if(3*x+7>5, sin(x), -3)");
MathExpression.EvalResult r = cond.solveGeneric(-42.0);   // r.scalar, r.vector, r.matrix, r.boolVal, r.textRes
```

### 2. Turbo Scalar: `ScalarTurboEvaluator1` / `ScalarTurboEvaluator2`

`TurboEvaluatorFactory` picks the right one for you; reach for a specific evaluator directly when you already know your shape (for example, point/line/function rotation compiles cleanly on `ScalarTurboEvaluator1`).

```java
MathExpression me = new MathExpression("rot(@(1,3)(0,2,0), pi, @(1,3)(0,0,0), @(1,3)(1,0,0))");
FastCompositeExpression turbo = new ScalarTurboEvaluator1(me).compile();
double result = turbo.applyScalar(new double[0]);

// Or let the factory decide:
FastCompositeExpression auto = TurboEvaluatorFactory.getCompiler(me).compile();
```

### 3. Turbo Matrix: `MatrixTurboEvaluator`

Matrices are first-class function values. Solve, invert, decompose, all compiled.

```java
Matrix m = new Matrix(coefficientData, n, n + 1);
m.setName("M");
new Function(m); // registers M for lookup by name

MathExpression expr = new MathExpression("linear_sys(M)");
FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(expr).compile();
Matrix solution = turbo.applyMatrix(new double[0]);
```

### 4. SIMD: `VectorTurboEvaluator` / `SIMDVectorTurboEvaluator` (JDK 21+)

```java
MathExpression me = new MathExpression(
    "(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
var evaluator = (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression)
        new SIMDVectorTurboEvaluator(me).compile();

double[][] inputs = new double[3][totalElements]; // x1, x2, x3
double[] out = new double[totalElements];
evaluator.applyBulk(inputs, out); // tail elements auto-masked if totalElements isn't lane-aligned
```

### 5. GPU: OpenCL / CUDA / Metal (`parser-ng-gpu-simd`, JDK 22+)

```java
MathExpression me = new MathExpression("2*x^2-3*x+1");
VectorTurboEvaluator vte = new VectorTurboEvaluator(me);

try (GpuCompositeExpression gpu = GpuExpressionBridge.from(vte, GpuBackend.OPENCL)) {
    float[] in = /* ... */;
    float[] out = new float[in.length];
    gpu.applyBulk(in, out); // native float32 kernel, no double bridging
}
```

The full GPU quick-start, including multi-vendor device selection, is at the top of this document.

---

## 📦 Installation & Configuration

Module choice maps directly onto the evolution table above: pick the highest tier you need, and each module pulls in everything below it. All examples below use version `3.1.0`.

### For standard Android or legacy pre-JDK 21 runtimes

Core interpreter, Turbo scalar and matrix tiers, and `BulkTurboEvaluator` for mechanically-sympathetic bulk loops with an inbuilt worker system:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.1.0</version>
</dependency>
```

```java
BulkTurboEvaluator.BatchedVectorCompositeExpression evaluator =
    (BulkTurboEvaluator.BatchedVectorCompositeExpression) new BulkTurboEvaluator(me).compile();
double[] out = new double[1];
evaluator.applyBulk(new double[]{5, 4, 1}, out);
```

### For modern JDK 21+ SIMD Vector API environments

Adds `VectorTurboEvaluator` and `SIMDVectorTurboEvaluator`:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.1.0</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>3.1.0</version>
</dependency>
```

### For modern JDK 22+ GPU and CPU-pinned SIMD environments

Adds `SIMDEngineEvaluator`, `SIMDCommandTurboEvaluator`, and the GPU bulk evaluators (CUDA, OpenCL and Metal):

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>3.1.0</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>3.1.0</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-gpu-simd</artifactId>
    <version>3.1.0</version>
</dependency>
```

### For data pipelines: Arrow, SQL and Parquet

| Module | What it does |
|---|---|
| `parser-ng-arrow` | Zero-copy evaluation, filtering and projection over Apache Arrow memory, with SIMD and GPU modes. |
| `parser-ng-sql` | SQL `select` over Arrow data and, since 3.0.9, over Parquet files (local and remote). |
| `parser-ng-parquet` | Fast reading of Parquet files, local and remote (for example S3). |

Add them the same way as the modules above. Each module's own README has the details.

### For microbenchmarking and profiling harnesses

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-bench</artifactId>
    <version>3.1.0</version>
</dependency>
```

---

## 📊 Comprehensive Feature Matrix

| Category | Supported mathematical tokens & functions | Turbo optimization support |
| --- | --- | --- |
| **Arithmetic operators** | `+`, `-`, `*`, `/`, `^`, `%`, `and`, `or`, `==`, `!=` | **Full hardware mapping** |
| **Trigonometric functions** | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh` | **Full hardware mapping** |
| **Calculus engines** | `diff` (symbolic engine), `intg` (numerical boundaries) | **Yes** |
| **Differential equations** | `diffeqn`, `diffeqnPath`, `diffeqnHO`, `diffeqnPathHO`. See [DIFF_ENGINE.md](parser-ng/DIFF_ENGINE.md). | **Yes** |
| **Matrix algebra** | `det`, `eigvalues`, `eigvec`, `adjoint`, `linear_sys` | **Optimized linear path** |
| **Statistical functions** | `avg`, `variance`, `rms`, `sort`, `erf`, `erfc` | **Yes** |
| **Activation functions** | `gelu`, `geglu`, `swiglu` | **Yes** |

---

## 🏢 Enterprise and support

ParserNG is used by 167 organizations to power mission-critical math visualization, analytical pipelines, and platform tooling. Special thanks to Jiri Vanek for contributing `ExpandingParser` and integrating it into the [jenkins-report-generic-chart-column](https://github.com/jenkinsci/report-generic-chart-column-plugin/) plugin ecosystem.

Production infrastructures that need predictable performance, safety assurances, and expert engineering access can upgrade to **ParserNG Enterprise**:

* **Priority operational support:** 24 to 48-hour SLAs for immediate bug tracking, profiling, and compliance vulnerabilities.
* **GraalVM Native Image deployment:** turn-key integration configurations for compiling down to fast native binaries.
* **Direct consultative access:** architectural reviews, customized functions (extensions), and hand-tailored vector and GPU kernel design directly from the author.

📧 **Contact for corporate licensing and consultations:** `gbenroscience@gmail.com`

☕ **Support open source development:** [GitHub Sponsors corporate tiers](https://buymeacoffee.com/gbenroscience/membership). The SIMD engine described above was Enterprise-only until the 3.0.8 release. Sponsorship is what funds moving the next tier out from behind that wall.

---

## 📚 Documentation & Technical Resources

* **Deep-dive benchmarking logs:** [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md), comprehensive execution breakdowns versus competitor runtimes.
* **High-fidelity graphical plotting:** [GRAPHING.md](parser-ng/GRAPHING.md), render configuration rules for JavaFX, Swing, and Android surfaces.
* **Bulk vectorization blueprints:** [BULK.md](parser-ng/BULK.md), optimization techniques for massive array processing.
* **Differential equations:** [DIFF_ENGINE.md](parser-ng/DIFF_ENGINE.md), full `diffeqn`/`diffeqnPath`/`diffeqnHO`/`diffeqnPathHO` syntax, solver selection guide, and result-capture patterns.
* **Release notes:** [LATEST.md](LATEST.md), change logs and technical notes for the latest release.
* [MORE.md](MORE.md): even more to know.
* [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md): the original readme for pre-1.0 versions, with a lot of examples that are still valid.

---

*Developed with mathematical rigor and mechanical sympathy by **Gbemiro Jiboye** (@gbenroscience).*