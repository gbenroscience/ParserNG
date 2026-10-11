# ParserNG SIMD/GPU Extension (Enterprise version now open-source!)

> **Hardware-Accelerated Mathematical Kernels for Enterprise JVM Ecosystems. Zero JNI. Zero Native Binaries. Zero Garbage Collection Churn.**

**ParserNG SIMD/GPU extension** provides an architectural extension for ParserNG via the proprietary `parser-ng-gpu-simd` module. It enables zero-dependency, ultra-high-throughput, tile-based bulk evaluations powered entirely by the native JDK Vector API (`jdk.incubator.vector`) and pure Java native GPU access.

By compiling mathematical expression trees into hyper-optimized, vectorized machine loops at runtime, this extension delivers native-C/C++ throughput directly within a secure, managed Java runtime. This completely eliminates the stability risks, memory leaks, security vulnerabilities, and deployment complexities of native toolchains (e.g., ND4J, `libnd4j`, or custom JNI bindings).

---

## ⚠️ Deprecation Notice: `SIMDEngineEvaluator` / `SIMDCommandEvaluator`

`SIMDEngineEvaluator` and `SIMDCommandEvaluator` are **deprecated** and are being phased out. They remain in the codebase for backward compatibility, but they should be treated as legacy — no further optimization work is planned for them, and they may be removed in a future major version.

**All new work should target the `com.github.gbenroscience.simdext.turbo.tools.command` package.** These evaluators use a JIT-style, precompiled command structure: instead of walking an expression tree node-by-node at evaluation time, `compile()` lowers the expression once into a flat array of small, monomorphic `VectorCommand` records (`LoadCommand`, `AddCommand`, fused `LoadLoad*Command`s, `UnaryMathCommand`, etc.) that are then replayed directly against the data on every bulk call. Because each command type is a small, self-contained record operating on primitive arrays and `int` offsets, this shape is exactly what the JVM's escape analysis and JIT (C2) are good at reasoning about — commands allocated during `compile()` don't escape the hot execution loop, so C2 can scalar-replace them and inline through the interpreter loop far more aggressively than it could through the older, more polymorphic tree-walking evaluators. The practical effect is fewer allocations, better inlining, and more consistent vectorization on the actual hot path.

The four evaluators below are the current, supported entry points into that package.

| Class | Input shape | Use when |
| :--- | :--- | :--- |
| `SIMDCommandF32` | Flat `float[]` **and** 2D `float[][]` variable arrays | Working in `float32` precision with data already resident on the Java heap |
| `SIMDCommandF64` | Flat `double[]` **and** 2D `double[][]` variable arrays | Working in `float64` precision with data already resident on the Java heap |
| `SIMDCommandSegmentF32` | `MemorySegment` of `float` inputs | `float32` data living off-heap (native memory, mmap'd files, FFI buffers) via `java.lang.foreign.MemorySegment`, avoiding a heap copy |
| `SIMDCommandSegmentF64` | `MemorySegment` of `double` inputs | `float64` data living off-heap, same rationale as above |

All four share the same compile-once/execute-many lifecycle: build a `MathExpression`, compile it once into a `SIMDVectorCompositeExpression`, then call one of the `applyBulk*` methods as many times as you like against fresh data.

---

## Key Architectural Pillars

### 1. Tile-Based Workload Dispatch
Compute tasks are segmented into deterministic, block-aligned matrices (ranging from 70 x 70 up to 200 x 200). This structural orchestration guarantees that data tiles remain 100% resident within L1/L2 CPU caches during evaluation sweeps, maximizing hardware cache utilization and minimizing memory controller bus contention.

### 2. C2 Loop Fusion
The internal optimization layer collapses complex abstract syntax trees (ASTs) into single, unified native vector loops. Chained transformations (e.g., `exp + tanh + FMA`) do not generate intermediate array allocations or separate loop passes; the HotSpot C2 compiler fuses them into 3 to 4 sequential AVX2/AVX-512 machine instructions. The command-package evaluators take this further with explicit peephole fusion at `compile()` time — a bare variable load feeding directly into a binary op or unary math function is collapsed into a single fused command (`LoadLoadAddCommand`, `LoadUnaryMathCommand`, etc.) that reads straight from the source array, skipping an otherwise-redundant scratch-buffer round trip.

### 3. Pure Java Intrinsic Pipelines
Built on top-tier JVM semantics utilizing `FloatVector`/`DoubleVector` and `VectorMask` primitives. The framework interacts directly with hardware vector registers without relying on `sun.misc.Unsafe`, out-of-heap off-heap hacks, or platform-dependent compilation chains. It is fully cross-platform and cross-architecture compliant out of the box.

### 4. Zero-Allocation Hot Paths
Execution paths utilize pre-allocated internal memory frames and Thread-Local Allocation Buffers (TLABs). Complete insulation from the Java Heap prevents GC promotion, ensuring perfectly predictable, real-time latency characteristics under sustained, multi-million-record production workloads. The precompiled command structure of the `command` package evaluators is specifically designed to fall within the JVM's escape analysis boundaries, letting C2 scalar-replace command objects rather than allocate them on the heap during hot execution.

---

## Performance Benchmarks

### 1. Bulk Execution Performance (JDK 24)
*Executed on an Intel Core i5-1135G7 environment running a JDK 24 runtime using 2 execution cores.*

| Operation | Matrix Scale: 70 x 70 | Matrix Scale: 100 x 100 | Matrix Scale: 200 x 200 |
| :--- | :--- | :--- | :--- |
| **GELU** | 2.01 ns/elt (9.85μs) | 1.95 ns/elt (19.5μs) | **1.96 ns/elt** (78.4μs) |
| **SwiGLU** | 1.57 ns | 1.57 ns | **1.54 ns/elt** (61.6μs) |

> 📊 **Peak Hardware Efficiency:** A measurement of `2.00 ns/element` for a complex activation function like GELU equates to roughly 8 raw CPU cycles. This securely reaches the physical throughput ceiling of what AVX2 vector units combined with hardware `vdexp` primitives can achieve in a managed code layer.

### 2. Industry Ecosystem Comparison (40,000 Elements)

| Library / Engine | GELU Execution Profile | Architectural Mechanism | Native Overhead / GC Churn |
| :--- | :--- | :--- | :--- |
| **ParserNG** | **80 µs (2.00 ns/elt)** | **Pure Java Vector API (Direct SIMD)** | **None (Zero Allocation)** |
| Standard JIT Scalar | ~3,800 µs (95.0 ns/elt) | Unrolled C2 Scalar Loops | None |
| Typical Native-JNI Bridge | ~1,200 µs (30.0 ns/elt) | Off-Heap C++ Context Switch | High (JNI Boundary Overhead) |

### 3. Data Scaling & Core Linearity
* **Single-Worker Purity:** In strict single-threaded environments, ParserNG's contiguous memory alignment outperforms traditional scalar bytecode compilers (like Janino) by over **2× to 14.6×**.
* **Linear Parallel Scaling:** Unlike scalar compilers that suffer from "negative scaling" due to cache line thrashing across multiple threads, ParserNG's Structure of Arrays (SoA) layout scales with **~80% parallel efficiency** when assigning multiple pinned hardware cores. *(Those figures were taken with worker pinning enabled. Pinning is now opt-in — add `-Dparserng.pin=true` to reproduce them, and see [Parallel Execution, CPU Pinning & Tuning](#parallel-execution-cpu-pinning--tuning) for why it is no longer the default.)*
* **Bandwidth Saturation:** At extreme dataset limits (e.g., 67M+ elements), throughput safely stabilizes to track physical DRAM bandwidth caps, maintaining a consistent `1.3x` advantage over optimized scalar loops.

---

## Deep Comparison: ParserNG vs. Janino JIT Compiler
To demonstrate the limitations of standard runtime-compilation engines, the micro-benchmarks below contrast Janino's scalar bytecode output with ParserNG's hardware-aligned SIMD execution architecture across 2,000,000 elements.

### Workload A: Transcendental Composition (21 Sines + 20 Arithmetic Steps)
When tasked with deep, complex mathematical curves containing intense instruction depths, traditional scalar JIT execution hits a hard processing wall.
* **Janino / Traditional Scalar Baseline:** ~590.00 ms
* **ParserNG (Parallel Vector Engine):** **~104.00 ms** (A massive **5.6×** throughput explosion)

### Workload B: Multi-Variable Algebraic Complexity (x1^3+x2^3+x3^3+x4^3+x5^3+x6^3)
The performance gap scales exponentially when moving to a Structure of Arrays (SoA) layout across multiple independent variables. Because scalar compilers cannot vectorize execution loops, they force the CPU to cycle across separate memory streams, inducing devastating cache thrashing and register pressure.

#### Single-Thread Control (1 Pinned Worker)
* **Janino AoS (Array of Structures):** 260.31 ms
* **Janino SoA (Structure of Arrays):** 277.28 ms
* **ParserNG (Pure SoA SIMD):** **19.03 ms** (Up to **14.6× faster** on a single core)

#### Multi-Thread Scaling Execution (e.g. 2 Active Workers)
* **Janino AoS (Array of Structures):** 271.40 ms *(Stagnated / Degraded by -4.2%)*
* **Janino SoA (Structure of Arrays):** 301.36 ms *(Choked / Degraded by -8.7%)*
* **ParserNG (Pure SoA SIMD):** **11.86 ms** (**🚀 1.60× Near-Linear Speedup**)

> 💡 **Concurrency handling:** Adding threads to Janino is not straightforward, unlike in ParserNG. ParserNG's vector loop design tracks contiguous chunks smoothly, converting raw compute addition directly into clean, scalable throughput.

---

## Prerequisites & JVM Runtime Configurations

### Hardware Requirements
* **x86_64:** Processors with AVX2 or AVX-512 vector extensions.
* **ARM64:** Apple Silicon or Enterprise ARM servers supporting NEON or SVE vector extensions.

### Supported Environments
* **Target Runtime:** Java 22, 24, or 26. *(JDK 24+ is highly recommended for highly advanced C2 vector unrolling optimization passes.)* Optional CPU pinning (off by default; see [CPU pinning (opt-in)](#cpu-pinning-opt-in)) uses the `java.lang.foreign` API and requires JDK 22+ plus `--enable-native-access=ALL-UNNAMED`.

### Mandatory JVM Execution Flags
To grant access to hardware vector extensions, you must specify the incubator flags upon application startup:

* **For JDK 22 through JDK 26+:**
```bash
java --add-modules jdk.incubator.vector -jar parser-ng-enterprise.jar
```

---

## Enterprise Quick Start & Integration

### 1. Maven Dependency Configuration

Enterprise binaries are hosted securely inside your designated private repository. Include the foundational open-source core along with the premium enterprise SIMD module coordinates inside your project's `pom.xml`:

```xml
<repositories>
    <repository>
        <id>parserng-enterprise-repo</id>
        <name>ParserNG Private Artifact Registry</name>
        <url>https://maven.pkg.github.com/gbenroscience/parserng-enterprise</url>
        <releases><enabled>true</enabled></releases>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>com.github.gbenroscience</groupId>
        <artifactId>parser-ng</artifactId>
        <version>3.0.9</version>
    </dependency>

    <dependency>
        <groupId>com.github.gbenroscience</groupId>
        <artifactId>parser-ng-simd</artifactId>
        <version>3.0.9</version>
    </dependency>
    <dependency>
        <groupId>com.github.gbenroscience</groupId>
        <artifactId>parser-ng-gpu-simd</artifactId>
        <version>3.0.9</version>
    </dependency>
</dependencies>
```

### 2. Core Execution Paradigm — `com.github.gbenroscience.simdext.turbo.tools.command`

Every evaluator in this package follows the same lifecycle:

1. Build a `MathExpression` from your expression string.
2. Compile it **once** into a `SIMDVectorCompositeExpression` via the class's static `getEvaluator(...)` factory (or `new SIMDCommandXxx(mathExpr).compile()`).
3. Call one of the `applyBulk*` methods as many times as you need against fresh input/output buffers. Compilation is the expensive step — reuse the compiled evaluator across calls rather than rebuilding it per batch.

Each class below also exposes a `(MathExpression, int numWorkers)` / `(String, int numWorkers)` constructor and matching `getEvaluator` overload for multi-worker parallel execution (CPU pinning is opt-in — see [Parallel Execution, CPU Pinning & Tuning](#parallel-execution-cpu-pinning--tuning)), plus `applyBulkParallel(...)` and `applyBulkBatched(..., batchSize)` variants alongside the plain `applyBulk(...)` shown here.

#### `SIMDCommandF32` — flat `float[]` and 2D `float[][]` variables

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF32;

public class F32FlatExample {
    public static void main(String[] args) throws Throwable {
        String expression = "x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x^3)))"; // GELU

        // 1. Compile once
        MathExpression mathExpr = new MathExpression(expression, true);
        SIMDCommandF32.SIMDVectorCompositeExpression evaluator =
                SIMDCommandF32.getEvaluator(mathExpr);

        // 2. Flat Structure-of-Arrays buffers (single variable "x" here)
        int dataSize = 2_000_000;
        float[] inputSource = new float[dataSize];
        float[] outputTarget = new float[dataSize];
        java.util.Arrays.fill(inputSource, 1.5f);

        // 3. Hardware-accelerated bulk execution
        evaluator.applyBulk(inputSource, outputTarget);

        System.out.println("First result: " + outputTarget[0]);
    }
}
```

`SIMDCommandF32` also accepts variables laid out as a 2D `float[][]` (one row per variable, matching multi-variable expressions like `x1^3+x2^3+...`) via the `applyBulk(float[][] variables, float[] output)` overload, with no change to the `getEvaluator` setup step above.

#### `SIMDCommandF64` — flat `double[]` and 2D `double[][]` variables

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandF64;

public class F64FlatExample {
    public static void main(String[] args) throws Throwable {
        String expression = "x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x^3)))"; // GELU

        // 1. Compile once
        MathExpression mathExpr = new MathExpression(expression, true);
        SIMDCommandF64.SIMDVectorCompositeExpression evaluator =
                SIMDCommandF64.getEvaluator(mathExpr);

        // 2. Flat Structure-of-Arrays buffers, double precision
        int dataSize = 2_000_000;
        double[] inputSource = new double[dataSize];
        double[] outputTarget = new double[dataSize];
        java.util.Arrays.fill(inputSource, 1.5);

        // 3. Hardware-accelerated bulk execution
        evaluator.applyBulk(inputSource, outputTarget);

        System.out.println("First result: " + outputTarget[0]);
    }
}
```

As with `SIMDCommandF32`, a `double[][]` overload of `applyBulk` is available for multi-variable expressions.

#### `SIMDCommandSegmentF32` / `SIMDCommandSegmentF64` — off-heap `MemorySegment` inputs

Both classes expose **two** zero-copy entry points, mirroring the flat-array and 2D-array split on `SIMDCommandF32`/`SIMDCommandF64`:

* `applyBulk(MemorySegment variables, MemorySegment output)` — one flat segment holding every variable back-to-back (sample count is derived from `output`'s byte size), for single-variable expressions or pre-interleaved data.
* `applyBulk(MemorySegment[] variables, MemorySegment output)` — **one segment per variable**, the direct off-heap analog of the `float[][]`/`double[][]` overload; sample count is derived from `variables[0]`'s byte size. This is the one to reach for with multi-variable expressions when you want to hand in independently-sourced off-heap buffers with no copy into a combined layout first.

Both also have `applyBulkParallel(...)` overloads for multi-worker execution (use `Arena.ofShared()` for segments that worker threads will read), following the same `(MathExpression, int numWorkers)` / `getEvaluator(..., int numWorkers)` setup shown for `SIMDCommandF32`/`SIMDCommandF64` above.

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF32;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class SegmentF32Example {
    public static void main(String[] args) throws Throwable {
        // Multi-variable expression: two independent off-heap variables, no interleaving needed
        String expression = "x1^3 + x2^3";

        // 1. Compile once
        MathExpression mathExpr = new MathExpression(expression, true);
        SIMDCommandSegmentF32.SIMDVectorCompositeExpression evaluator =
                SIMDCommandSegmentF32.getEvaluator(mathExpr);

        // 2. Independent off-heap float buffers, one per variable — zero-copy: no
        // interleaving/flattening step, each segment is read straight from where it lives
        int dataSize = 2_000_000;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment x1 = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment x2 = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment outputTarget = arena.allocate((long) dataSize * ValueLayout.JAVA_FLOAT.byteSize());

            for (int i = 0; i < dataSize; i++) {
                x1.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.5f);
                x2.setAtIndex(ValueLayout.JAVA_FLOAT, i, 2.0f);
            }

            // 3. Zero-copy bulk execution — one segment per variable
            evaluator.applyBulk(new MemorySegment[]{x1, x2}, outputTarget);

            System.out.println("First result: " + outputTarget.getAtIndex(ValueLayout.JAVA_FLOAT, 0));
        }
    }
}
```

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simdext.turbo.tools.command.SIMDCommandSegmentF64;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class SegmentF64Example {
    public static void main(String[] args) throws Throwable {
        String expression = "x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x^3)))"; // GELU

        // 1. Compile once
        MathExpression mathExpr = new MathExpression(expression, true);
        SIMDCommandSegmentF64.SIMDVectorCompositeExpression evaluator =
                SIMDCommandSegmentF64.getEvaluator(mathExpr);

        // 2. Single flat off-heap double buffer (one variable — no need for the
        // MemorySegment[] overload here)
        int dataSize = 2_000_000;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputSource = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());
            MemorySegment outputTarget = arena.allocate((long) dataSize * ValueLayout.JAVA_DOUBLE.byteSize());

            for (int i = 0; i < dataSize; i++) {
                inputSource.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 1.5);
            }

            // 3. Hardware-accelerated bulk execution, straight off-heap — no array copy
            evaluator.applyBulk(inputSource, outputTarget);

            System.out.println("First result: " + outputTarget.getAtIndex(ValueLayout.JAVA_DOUBLE, 0));
        }
    }
}
```

### Supported Activation Kernels

Out of the box, the enterprise kernel includes optimized presets for:

* Modern Neural layers: `gelu`, `swiglu`, `silu`, `rmsnorm`.
* Standard Core Transcendental operations: `tanh`, `exp`, `log`, `relu`.
* Fully customized, multivariant expression trees via custom user lambdas.

---

## Parallel Execution, CPU Pinning & Tuning

### How the worker pool behaves
Every `command` evaluator created with `numWorkers` (or the default worker count, chosen from the machine's CPU count) owns a small pool of daemon worker threads. `applyBulkParallel(...)` splits the batch, the **calling thread processes one share itself**, and the workers process the rest.

* **Handoff:** workers spin briefly (`SPIN_LIMIT`), then block with an **untimed** `LockSupport.park()`. Dispatch calls `unpark(worker)` and a finishing worker calls `unpark(caller)`, so there is no polling loop and no dependence on OS timer granularity. *(Earlier revisions of the `SIMDCommandSegment*` classes polled with `parkNanos(20 µs)`, which on Windows rounds up to the scheduler tick and made parallel runs slower than serial; they now use the same handoff as `SIMDCommandF32`/`SIMDCommandF64`.)*
* **Lifecycle:** `SIMDVectorCompositeExpression` is `AutoCloseable`. Use try-with-resources (or call `close()`) when you are done so the worker threads stop; otherwise a `Cleaner` stops them when the evaluator is garbage-collected. Benchmark loops that create many evaluators should close each one.
* **One caller at a time:** to be safe, do not call `applyBulkParallel` concurrently from several threads on the *same* evaluator instance; create one evaluator per calling thread.
* **Small inputs run serially:** below an internal threshold `applyBulkParallel` falls back to the serial path.
* **Shared memory:** `MemorySegment`s handed to `applyBulkParallel` must be accessible from the worker threads — allocate them from `Arena.ofShared()` (or `Arena.global()`/`ofAuto()`), not `Arena.ofConfined()`.

### CPU pinning (opt-in)
**Pinning is off by default.** Hard CPU affinity can help on dedicated, isolated cores, but on shared desktops/laptops it removes the scheduler's ability to move a worker off a busy core. In our measurements it gave no consistent benefit and produced occasional multi-millisecond stalls, so you must ask for it:

```bash
# enable from the command line
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Dparserng.pin=true ...
```

```java
// or programmatically, BEFORE creating any evaluator (workers decide when they start)
com.github.gbenroscience.simdext.turbo.tools.utils.CPUPinner.setPinningEnabled(true);
```

When enabled, each worker calls `CPUPinner.pinWorker(workerId)`:

* **Topology-aware:** `CPUPinner.pinOrder()` lists the first logical CPU of every *physical* core, then each core's SMT siblings (e.g. `[[0,1],[2,3]]` → `[0,2,1,3]`), so workers land on distinct physical cores before sharing one. The first slot is left for the calling thread.
* **Platforms:** Linux (`sched_setaffinity`) and Windows (`SetThreadAffinityMask`; processor group 0 only, i.e. at most 64 logical CPUs). On macOS pinning is skipped (a no-op). Without `--enable-native-access=ALL-UNNAMED` the JDK prints a restricted-method warning on first use (and a future release will refuse the call), in which case pinning is skipped as well. Every pinning request in this module goes through `CPUPinner.pinCurrentThread`, which honours the flag.
* **Introspection:** `CPUPinner.isPinningEnabled()`, `CPUPinner.pinOrder()` and `CPUPinner.detectPhysicalCoreGroups()`.

### What we measured (and what to expect)
Measured on an Intel Core i7-6500U laptop (2 physical cores / 4 logical), Windows 10 and WSL2, 1,000,017 elements per variable. These numbers are specific to that machine — measure on yours.

* Parallel speedup over serial was roughly **1.3–1.9×**, higher for compute-heavy expressions (`exp`, `sin`) where the second hyper-thread helps, lower for cheap expressions that are limited by memory bandwidth (about 10 GB/s on that machine, so F32 runs about twice as fast as F64).
* The array paths and the `MemorySegment` / `MemorySegment[]` paths performed within about ±15% of each other. The segment paths exist to avoid copies when data already lives off-heap, not to beat heap arrays.
* Pinning made no consistent difference on WSL2 and was never clearly better on Windows; the worst stalls (a worker taking about twice as long as expected) occurred both pinned and unpinned, so report the **median of several rounds**, not a single average.
* The default worker count was a good setting on that machine; try `getEvaluator(expr, n)` with your own `n` if you have a different topology.

### Benchmark harness: `SIMDThrptBench`
`com.github.gbenroscience.simdext.turbo.tools.bench.SIMDThrptBench` benchmarks every bulk path, serial and parallel, and prints a summary table.

```bash
mvn compile exec:exec -Dexec.executable="java" \
  -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -classpath %classpath com.github.gbenroscience.simdext.turbo.tools.bench.SIMDThrptBench"
```

| Option | Path exercised |
| :--- | :--- |
| 1 | Auto-vectorized `double[][]` |
| 2 | Base SIMD `double[][]` |
| 3 / 4 | `SIMDCommandF32` / `SIMDCommandF64` arrays |
| 5 / 6 | `SIMDCommandSegmentF32` / `SIMDCommandSegmentF64`, array overloads |
| 7 / 8 | `SIMDCommandSegmentF64` / `F32`, **flat** `MemorySegment` (all variables in one segment, variable-major: element *(v, i)* at byte offset `(v*N + i) * elemSize`) |
| 9 / 10 | `SIMDCommandSegmentF64` / `F32`, **`MemorySegment[]`** (one segment per variable) |
| 0 | All of the above, then a summary table |

| System property | Effect |
| :--- | :--- |
| `-Dbench.workers=N` | Worker count for variants 3–10 (default: the evaluator's own default) |
| `-Dbench.rounds=N` | Measurement rounds per row (default 5); the table reports the **median** round, and each row also prints best and worst |
| `-Dparserng.pin=true` | Enable CPU pinning (default: off) |
| `-Dparserng.diag=true` | Print a per-row timing line: dispatch→pickup latency, worker busy time, caller wait time |

The first line of output shows the pin order, whether pinning is on, and the worker count, so every saved log is self-describing. After each prompt the harness prints `Expression :` and `Variant    :` lines confirming what it received — on some Windows launch paths (such as `mvn exec:exec`) the console does not echo typed text even though the input is read correctly. Each row also prints the last output element as a cheap cross-variant correctness check (all F64 rows should match, and all F32 rows should match). Pass `<expression> <variant>` as program arguments to run one variant once and exit — handy for giving each variant its own fresh JVM:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp ... \
  com.github.gbenroscience.simdext.turbo.tools.bench.SIMDThrptBench "x^2+1" 7
```

### JMH benchmark: `SIMDThrptJmh`
`SIMDThrptJmh` covers the same ten paths with [JMH](https://github.com/openjdk/jmh). Every parameter combination runs in its own forked JVM, so there is no JIT-profile pollution between variants and no leftover worker threads. The score is **milliseconds per bulk payload** (convert to M elements/s with `elements / 1000 / score_ms`).

Add JMH to the module (benchmarks live in `src/test/java`, so JMH stays out of the shipped jar):

```xml
<properties><jmh.version>1.37</jmh.version></properties>

<dependency>
    <groupId>org.openjdk.jmh</groupId>
    <artifactId>jmh-core</artifactId>
    <version>${jmh.version}</version>
    <scope>test</scope>
</dependency>

<plugin>
    <artifactId>maven-compiler-plugin</artifactId>
    <configuration>
        <annotationProcessorPaths>
            <path>
                <groupId>org.openjdk.jmh</groupId>
                <artifactId>jmh-generator-annprocess</artifactId>
                <version>${jmh.version}</version>
            </path>
        </annotationProcessorPaths>
    </configuration>
</plugin>
```

> If the module already relies on other annotation processors, add them to `annotationProcessorPaths` too — setting that list disables classpath discovery.

Run it through the class's **own `main`** — it prompts for an expression and a variant (0 = all), exactly like `SIMDThrptBench`, and includes *only* this class's benchmark, so other JMH classes in the same directory are never started. Pass `<expression> <variant>` as program arguments to skip the prompts.

```bash
mvn test-compile exec:exec -Dexec.classpathScope=test -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -classpath %classpath com.github.gbenroscience.simdext.turbo.tools.bench.SIMDThrptJmh"
```

| JMH parameter | Values |
| :--- | :--- |
| `variant` | 1–10 (0 = all in the prompt), same numbering as `SIMDThrptBench` (7/8 flat `MemorySegment`, 9/10 `MemorySegment[]`) |
| `mode` | `serial` or `parallel` |
| `expression` | Prompted for by `main` (any expression, commas and spaces are fine). Defaults, used only if you launch through `org.openjdk.jmh.Main`: a cheap one (`x^2+1`) and a three-variable Gaussian |
| `workers` | `0` = evaluator default, otherwise `getEvaluator(expr, workers)` for variants 3–10 |
| `elements` | Samples per variable (default 1,000,017) |

Optional system properties on the launching JVM: `-Dbench.workers=N`, `-Dbench.elements=N`, `-Dbench.mode=serial|parallel` (default both), `-Dparserng.pin=true`, `-Dparserng.diag=true`. The module flags and the two `parserng.*` properties are forwarded to the forked JVMs explicitly, and each fork prints a `[config]` line showing what it saw. To use JMH profilers such as `-prof gc`, run `org.openjdk.jmh.Main` with this class's name as the include pattern instead.

### Known limitations
* `validate(MemorySegment, MemorySegment)` on the segment evaluators requires the input and output segments to have the **same byte size**. That only matches the flat layout for single-variable expressions; multi-variable flat segments (`variables.byteSize() == numVars * output.byteSize()`) are processed by `applyBulk` but would not pass this check. `applyBulk` does not call it.
* The Windows topology/pinning code handles processor group 0 only.

---

## Change Log

* **Segment evaluators (`SIMDCommandSegmentF32`/`F64`):** replaced `parkNanos(20 µs)` polling in worker idle waits and in the caller's completion wait with the untimed `park()`/`unpark()` handoff used by `SIMDCommandF32`/`F64`. Fixes parallel runs being slower than serial on Windows.
* **CPU pinning is now opt-in everywhere:** `-Dparserng.pin=true` or `CPUPinner.setPinningEnabled(true)`. `CPUPinner.pinCurrentThread` returns `false` unless enabled.
* **Topology-aware pinning:** new `CPUPinner.pinOrder()` and `CPUPinner.pinWorker(workerId)` (one logical CPU per physical core first; first slot left for the caller). `SIMDCommandF32`, `SIMDCommandF64`, `SIMDCommandSegmentF32` and `SIMDCommandSegmentF64` now pin through `pinWorker`.
* **`CPUPinner` fixes:** Windows indices of 64 and above are rejected instead of silently wrapping onto another CPU; the Windows topology buffer is 8-byte aligned.
* **Opt-in diagnostics** (`-Dparserng.diag=true`) in all four evaluators: per-task dispatch→pickup latency, worker busy time and caller wait time. Compiled out by the JIT when the flag is off.
* **New JMH harness `SIMDThrptJmh`:** all ten paths, serial and parallel, one forked JVM per combination, evaluators closed after each trial.
* **`SIMDThrptBench` rewritten:** one shared measurement routine; new variants 7–10 for the flat `MemorySegment` and `MemorySegment[]` paths in F64 and F32; evaluators are closed after each variant; median-of-rounds reporting; `-Dbench.workers`, `-Dbench.rounds`, one-shot `<expr> <variant>` mode; correct handling of repeated expression/variant prompts.

---

## Local Validation Testing

To run microbenchmarks, check hardware lane allocations, and verify vector compilation paths on your machine architecture, execute the included target test harness:

```bash
mvn clean test -Dtest=SIMDCommandF64SegmentTest
```

The validation tool tests across variable scale profiles (20 x 20, 70 x 70, 100 x 100, 200 x 200, 512 x 512, and 1024 x 1024).

For throughput numbers on your own hardware across every bulk path (including the off-heap `MemorySegment` and `MemorySegment[]` paths), use the [`SIMDThrptBench` harness](#benchmark-harness-simdthrptbench).
 

## Enterprise Support, SLAs, & Custom DSL Services

ParserNG includes full support contracts designed for mission-critical banking, financial technology, and AI inference runtimes:

* **Commercial Support SLAs:** Guaranteed response times (24-hour response windows for tier-1 critical processing interruptions).
* **Custom Architecture Consultative Audits:** Direct assistance with cache layouts, hardware tuning, pinning worker threads to prevent bus contention, and maximizing mechanical sympathy profiles.
* **Custom Domain-Specific Language (DSL) Design:** Implementation services for domain-specific mathematical grammars, secure custom operations, and isolated macro processors tailored to your corporate financial modeling rules.

For enterprise license tokens, support inquiries, or custom engagement keys, open a ticket via the enterprise service portal or contact `gbenroscience@gmail.com`.

---