# ParserNG SIMD/GPU Extension(Enterprise version now open-source!)

> **Hardware-Accelerated Mathematical Kernels for Enterprise JVM Ecosystems. Zero JNI. Zero Native Binaries. Zero Garbage Collection Churn.**

**ParserNG SIMD/GPU extension** provides an architectural extension for ParserNG via the proprietary `parser-ng-gpu-simd` module. It enables zero-dependency, ultra-high-throughput, tile-based bulk evaluations powered entirely by the native JDK Vector API(`jdk.incubator.vector`) and pure Java native GPU access.

By compiling mathematical expression trees into hyper-optimized, vectorized machine loops at runtime, this extension delivers native-C/C++ throughput directly within a secure, managed Java runtime. This completely eliminates the stability risks, memory leaks, security vulnerabilities, and deployment complexities of native toolchains (e.g., ND4J, `libnd4j`, or custom JNI bindings).

---

## Key Architectural Pillars

### 1. Tile-Based Workload Dispatch
Compute tasks are segmented into deterministic, block-aligned matrices (ranging from 70 x 70 up to 200 x 200). This structural orchestration guarantees that data tiles remain 100% resident within L1/L2 CPU caches during evaluation sweeps, maximizing hardware cache utilization and minimizing memory controller bus contention.

### 2. C2 Loop Fusion
The internal optimization layer collapses complex abstract syntax trees (ASTs) into single, unified native vector loops. Chained transformations (e.g., `exp + tanh + FMA`) do not generate intermediate array allocations or separate loop passes; the HotSpot C2 compiler fuses them into 3 to 4 sequential AVX2/AVX-512 machine instructions.

### 3. Pure Java Intrinsic Pipelines
Built on top-tier JVM semantics utilizing `FloatVector` and `VectorMask` primitives. The framework interacts directly with hardware vector registers without relying on `sun.misc.Unsafe`, out-of-heap off-heap hacks, or platform-dependent compilation chains. It is fully cross-platform and cross-architecture compliant out of the box.

### 4. Zero-Allocation Hot Paths
Execution paths utilize pre-allocated internal memory frames and Thread-Local Allocation Buffers (TLABs). Complete insulation from the Java Heap prevents GC promotion, ensuring perfectly predictable, real-time latency characteristics under sustained, multi-million-record production workloads.

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
| **ParserNG Enterprise** | **80 µs (2.00 ns/elt)** | **Pure Java Vector API (Direct SIMD)** | **None (Zero Allocation)** |
| Standard JIT Scalar | ~3,800 µs (95.0 ns/elt) | Unrolled C2 Scalar Loops | None |
| Typical Native-JNI Bridge | ~1,200 µs (30.0 ns/elt) | Off-Heap C++ Context Switch | High (JNI Boundary Overhead) |

### 3. Data Scaling & Core Linearity
* **Single-Worker Purity:** In strict single-threaded environments, ParserNG's contiguous memory alignment outperforms traditional scalar bytecode compilers (like Janino) by over **2× to 14.6×**.
* **Linear Parallel Scaling:** Unlike scalar compilers that suffer from "negative scaling" due to cache line thrashing across multiple threads, ParserNG’s Structure of Arrays (SoA) layout scales with **~80% parallel efficiency** when assigning multiple pinned hardware cores.
* **Bandwidth Saturation:** At extreme dataset limits (e.g., 67M+ elements), throughput safely stabilizes to track physical DRAM bandwidth caps, maintaining a consistent `1.3x` advantage over optimized scalar loops.

---

## Deep Comparison: ParserNG Enterprise vs. Janino JIT Compiler
To demonstrate the limitations of standard runtime runtime-compilation engines, the micro-benchmarks below contrast Janino's scalar bytecode output with ParserNG’s hardware-aligned SIMD execution architecture across 2,000,000 elements.

### Workload A: Transcendental Composition (21 Sines + 20 Arithmetic Steps)
When tasked with deep, complex mathematical curves containing intense instruction depths, traditional scalar JIT execution hits a hard processing wall.
* **Janino / Traditional Scalar Baseline:** ~590.00 ms
* **ParserNG Enterprise (Parallel Vector Engine):** **~104.00 ms** (A massive **5.6×** throughput explosion)

### Workload B: Multi-Variable Algebraic Complexity (x1^3+x2^3+x3^3+x4^3+x5^3+x6^3)
The performance gap scales exponentially when moving to a Structure of Arrays (SoA) layout across multiple independent variables. Because scalar compilers cannot vectorize execution loops, they force the CPU to cycle across separate memory streams, inducing devastating cache thrashing and register pressure.

#### Single-Thread Control (1 Pinned Worker)
* **Janino AoS (Array of Structures):** 260.31 ms
* **Janino SoA (Structure of Arrays):** 277.28 ms
* **ParserNG Enterprise (Pure SoA SIMD):** **19.03 ms** (Up to **14.6× faster** on a single core)

#### Multi-Thread Scaling Execution (e.g. 2 Active Workers)
* **Janino AoS (Array of Structures):** 271.40 ms *(Stagnated / Degraded by -4.2%)*
* **Janino SoA (Structure of Arrays):** 301.36 ms *(Choked / Degraded by -8.7%)*
* **ParserNG Enterprise (Pure SoA SIMD):** **11.86 ms** (**🚀 1.60× Near-Linear Speedup**)

> 💡 **Concurrency handling:** Adding threads to Janino is not straightforward, unlike in ParserNG. ParserNG's vector loop design tracks contiguous chunks smoothly, converting raw compute addition directly into clean, scalable throughput.
---

## Prerequisites & JVM Runtime Configurations

### Hardware Requirements
* **x86_64:** Processors with AVX2 or AVX-512 vector extensions.
* **ARM64:** Apple Silicon or Enterprise ARM servers supporting NEON or SVE vector extensions.

### Supported Environments
* **Target Runtime:** Java 22, 24, or 26. *(JDK 24+ is highly recommended for highly advanced C2 vector unrolling optimization passes).*

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
        <name>ParserNG Enterprise Private Artifact Registry</name>
        <url>[https://maven.pkg.github.com/gbenroscience/parserng-enterprise](https://maven.pkg.github.com/gbenroscience/parserng-enterprise)</url>
        <releases><enabled>true</enabled></releases>
    </repository>
</repositories>

<dependencies>
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
</dependencies>

```

### 2. Core Execution Paradigm

```java
import com.github.gbenroscience.parserng.core.MathExpression;
import com.github.gbenroscience.parserng.simd.VectorTurboEvaluator;

public class EnterpriseComputeRegistry {
    public static void main(String[] args) {
        // Define your deep neural activation or custom transformer loop
        String expression = "x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x^3)))"; // GELU
        
        // 1. Instantiation and Turbo Registration
        MathExpression mathExpr = new MathExpression(expression, true);
        SIMDEngineEvaluator.SIMDVectorCompositeExpression evaluator = SIMDEngineEvaluator.getEvaluator(mathExpr);
        
        // 2. Prepare Structure of Arrays (SoA) Buffers
        int dataSize = 2_000_000;
        double[] inputSource = new double[dataSize]; 
        double[] outputTarget = new double[dataSize];
        
        // Populate input data...
        java.util.Arrays.fill(inputSource, 1.5);
        
        // 3. Hardware-Accelerated High-Throughput Execution
        // Automatically leverages tile boundaries and registers
        evaluator.applyBulk(inputSource, outputTarget);
        
        System.out.println("Bulk processing complete. First result: " + outputTarget[0]);
    }
}

```

### Supported Activation Kernels

Out of the box, the enterprise kernel includes optimized presets for:

* Modern Neural layers: `gelu`, `swiglu`, `silu`, `rmsnorm`.
* Standard Core Transcendental operations: `tanh`, `exp`, `log`, `relu`.
* Fully customized, multivariant expression trees via custom user lambdas.

---

## Local Validation Testing

To run microbenchmarks, check hardware lane allocations, and verify vector compilation paths on your machine architecture, execute the included target test harness:

```bash
mvn clean test -Dtest=SIMDTurboEvaluatorTest

```

The validation tool tests across variable scale profiles (20 x 20, 70 x 70, 100 x 100, 200 x 200, 512 x 512, and 1024 x 1024).

> ⚠️ **Developer Note:** To prevent console I/O bottlenecks and virtual terminal terminal blocking from stalling the execution test cycle, do not pass debug variables to print raw matrices out into standard log files during 200 x 200 test iterations.

---

## Enterprise Support, SLAs, & Custom DSL Services

ParserNG Enterprise includes full support contracts designed for mission-critical banking, financial technology, and AI inference runtimes:

* **Commercial Support SLAs:** Guaranteed response times (24-hour response windows for tier-1 critical processing interruptions).
* **Custom Architecture Consultative Audits:** Direct assistance with cache layouts, hardware tuning, pinning worker threads to prevent bus contention, and maximizing mechanical sympathy profiles.
* **Custom Domain-Specific Language (DSL) Design:** Implementation services for domain-specific mathematical grammars, secure custom operations, and isolated macro processors tailored to your corporate financial modeling rules.

For enterprise license tokens, support inquiries, or custom engagement keys, open a ticket via the enterprise service portal or contact `gbenroscience@gmail.com`.

---

## License

ParserNG Enterprise is a commercial product and its source available components are governed strictly by the **ParserNG Enterprise Commercial License**.

Unauthorized copying, distribution, or reproduction of the code contained within the `parser-ng-simd` module via any medium is strictly prohibited. Continued use requires an active, paid commercial enterprise agreement token.

*Copyright © 2026 Gbemiro Jiboye / gbenroscience. All rights reserved.*

