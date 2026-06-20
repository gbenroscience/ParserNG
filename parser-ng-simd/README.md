# parser-ng-simd

### ParserNG 2.0.0

High-performance, hardware-accelerated mathematical kernels for Java. **No JNI. No native binaries.** Powered entirely by the JDK Vector API and optimized for C2 loop fusion.

`parser-ng-simd` provides an architectural extension for ParserNG, enabling zero-dependency, tile-based bulk evaluations via `jdk.incubator.vector`. It compiles mathematical expression trees into vectorized loops optimized for L1/L2 cache residency, achieving near-native throughput directly on the JVM without the configuration overhead of complex native toolchains (such as ND4J/`libnd4j`).

---

### Performance Benchmarks

#### Bulk Execution Performance (2 Cores, JDK 24)

Executed on an Intel Core i5-1135G7 environment running a JDK 24 Early Access build.

| Operation | Matrix: $70 \times 70$ | Matrix: $100 \times 100$ | Matrix: $200 \times 200$ | Performance vs Janino |
| --- | --- | --- | --- | --- |
| **GELU** | 10.7 ns/elt ($52.5\,\mu\text{s}$) | 7.46 ns/elt ($74.6\,\mu\text{s}$) | 7.30 ns/elt ($292\,\mu\text{s}$) | **2.3x – 3.4x Faster** |
| **SwiGLU** | — | — | 13.30 ns/elt ($530\,\mu\text{s}$) | **3.0x – 4.0x Faster** |

> 📊 **Note on Peak Efficiency:** An execution speed of `7.3 ns/element` for GELU translates to roughly 26 CPU cycles. This safely hits the theoretical throughput ceiling of what AVX2 combined with hardware `vdexp` intrinsics can process within a managed runtime.

#### Ecosystem Comparison ($40\text{k}$ Elements, 2 Cores)

| Library / Engine | GELU Execution Profile | Architectural Mechanism |
| --- | --- | --- |
| **parser-ng-simd** | **292 µs (7.3 ns/elt)** | **Pure Java Vector API (Direct SIMD)** |
| Janino | ~1,000 µs (25.0 ns/elt) | Scalar Runtime Bytecode Compilation |
| ND4J (CPU) | ~1,600 µs (40.0 ns/elt) | JNI Bridging + Native `libnd4j` |
| Colt | ~2,000 µs (50.0 ns/elt) | Legacy Scalar Loops |

*Data Scaling Boundary:* At extreme workloads (e.g., 67M+ records), computation throughput bounds shift toward physical DRAM bandwidth, where execution stabilizes at approximately `1.3x` faster than optimized scalar bytecode compilers.

---

### Key Architectural Pillars

1. **Tile-Based Workload Dispatch:** Compute tasks are split into deterministic block matrices (from $70 \times 70$ up to $200 \times 200$), ensuring data tiles remain completely resident within L1/L2 CPU caches while minimizing loop-bound orchestration overhead.
2. **C2 Loop Fusion:** The engine collapses composite abstract syntax trees (ASTs) into unified native vector loops. Sequential steps like `exp + tanh + FMA` resolve into 3 to 4 hyper-optimized AVX2 machine instructions.
3. **Pure Java Intrinsic Pipelines:** Built directly on JDK Vector API concepts (`FloatVector`, `VectorMask`), utilizing hardware-level primitive acceleration without relying on unsafe memory access blocks or third-party wrappers.
4. **Zero-Allocation Hot Paths:** Execution tracks pre-allocated internal memory frames. Buffer reuse prevents heap churning, entirely insulating math execution paths from Garbage Collection (GC) pauses.

---

### Prerequisites & Requirements

* **Runtime Environment:** JDK 21+ (JDK 24+ is highly recommended for optimized C2 pipeline compilation; JDK 21 performance may measure up to 15% slower due to older optimization passes).
* **Hardware Architecture:** x86-64 processors supporting AVX2 or AVX-512 extensions, or ARM hardware with NEON/SVE vectors.
* **JVM Execution Flags:**
* **JDK 21:** `--enable-preview --add-modules jdk.incubator.vector`
* **JDK 22+:** `--add-modules jdk.incubator.vector`



---

### Quick Start

#### 1. Configure Maven Dependencies

Include both the core module and the SIMD engine extension in your `pom.xml`:

```xml
<dependencies>
    <dependency>
        <groupId>com.github.gbenroscience</groupId>
        <artifactId>parser-ng</artifactId>
        <version>2.0.0</version>
    </dependency>
    
    <dependency>
        <groupId>com.github.gbenroscience</groupId>
        <artifactId>parser-ng-simd</artifactId>
        <version>2.0.0</version>
    </dependency>
</dependencies>

```

#### 2. Vector Evaluation Implementation

Ensure your application launch configuration includes the relevant incubator modules before initializing the compiler engine.

```java
import com.github.gbenroscience.simd.VectorTurboEvaluator;

public class ExecutionProfile {
    public static void main(String[] args) {
        // Initialize evaluator with mandatory environment configurations
        VectorTurboEvaluator evaluator = new VectorTurboEvaluator(
            "--enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector"
        );

        // Compile the target mathematical formula once
        var vectorizedFunction = evaluator.compile("gelu(x)"); 

        // Prepare dataset array
        float[] inputData = new float[40000];
        // ... populate inputData

        // Execute hot path - tiles are automatically configured for your cache layout
        float[] outputData = vectorizedFunction.eval(inputData); 
    }
}

```

#### Supported Activation Kernels

Out-of-the-box support includes: `gelu`, `swiglu`, `silu`, `tanh`, `exp`, `log`, `relu`, `rmsnorm`, along with arbitrary expression trees. Custom vector kernels can be integrated directly by extending core evaluation interfaces.

---

### Local Verification & Verification Testing

To execute microbenchmarks and verify SIMD registration on your host architecture, use the following profile goal:

```bash
mvn clean test -Dtest=VectorTurboEvaluatorTest

```

The test harness evaluates throughput performance vectors across varying scales ($20 \times 20$, $70 \times 70$, $100 \times 100$, and $200 \times 200$).

> ⚠️ **Configuration Suggestion:** Avoid configuring raw matrix printouts at higher dimension limits ($200 \times 200$) within the terminal log to prevent console buffer IO bottlenecks from stalling the test cycle.

---

### Current Project Roadmap

* [ ] Native AVX-512 targeted code-generation paths
* [ ] Unified / Fused `rmsnorm + swiglu` computational kernel
* [ ] Comparative execution benchmarks against native PyTorch CPU engines
* [ ] Multi-threaded horizontal scaling analysis across 8 and 16-core layouts

---

### License

Distributed under the **Apache License 2.0**. Contributions, optimizations, and performance discussions regarding hardware intrinsic precision are always welcome.