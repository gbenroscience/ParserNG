# ParserNG 2.0.7 🧮⚡

[![Maven Central](https://img.shields.io/maven-central/v/com.github.gbenroscience/parser-ng.svg?style=flat-square&color=blue)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Downloads](https://img.shields.io/badge/Downloads-11k%2B-brightgreen?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![Growth](https://img.shields.io/badge/Growth-%2B250%25%20(90%20days)-orange?style=flat-square)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![JDK Compatibility](https://img.shields.io/badge/JDK-8%20to%2026%2B-red?style=flat-square)](https://www.oracle.com/java/)

> **The Fastest Pure-Java Expression Engine & Vector Math Kernels. Zero JNI. Zero Native Binaries. Zero Bytecode Safety Risks.**

**ParserNG** is a ultra-high-performance mathematical expression evaluation engine designed for modern JVM workloads, including complex real-time plotting pipelines, financial modeling, and deep learning neural network activations (e.g., Transformer architectures). 

By adopting a hardware-aligned, fast-interpreted memory model instead of risky dynamic bytecode generation, ParserNG eliminates classloader bloat, protects your runtime from native segmentation faults, and dramatically simplifies your Software Bill of Materials (SBOM) compliance posture.

---

### Pure-Java Real-Time Hardware Acceleration
<!--
Below are live renderings of a dynamic transcendental surface layer computed entirely within a secure Java runtime using the ParserNG Vector kernel. No JNI, no off-heap memory hacks, just pure mechanical sympathy.


<div align="center">
  <video src="https://github.com/user-attachments/assets/d1e2e1f8-4fd2-4be3-a778-33d8360d949d" width="70%" poster="./assets/video_poster.png"> </video>
</div>
-->


## 🚀 What's New in Version 2.0.7

Version 2.0.7 marks a complete architectural milestone, featuring the total migration of the bulk evaluation engine to the native JDK Vector API. 

* **The Vector Breakthrough:** Guarantees a **2x to 20x throughput explosion** for complex expressions containing intense transcendental operations.
* **Core Efficiency Saturation:** Single-core throughput per worker thread has been optimized so aggressively that computation speeds approach physical hardware memory bandwidth limits—making horizontal scaling bottlenecks trivial.
* **Enterprise Horizontal Scaling:** For high-scale parallel clustered JAR deployments (optimized for Linux environments), contact the author(`gbenroscience@gmail.com`) directly for specialized builds of **ParserNG Enterprise**.

---

## 📊 Performance & Throughput Profiles

### Throughput Horizons
| Execution Mode | Throughput Capacity (Evaluations / Sec) |
| :--- | :--- |
| **Standard Mode** | 3,000,000 – 10,000,000 |
| **Turbo Mode** | 10,000,000 – 90,000,000 |
| **Bulk Turbo (Single Core)** | **200,000,000+** |
| **Bulk Turbo (Parallel Workers)** | **Even Higher (Hardware Saturation)** |

### Amortized Activation Latencies (JDK 21+)
* **SwiGLU Kernel:** `1.8 ns / element`
* **GELU Kernel:** `2.1 ns / element`
* **Custom Expressions:** Highly scalable, consistently executing multiple times faster than standard Janino scalar compilations on single-core setups.

---

## ✨ Key Capabilities

* **Calculus & Advanced Algebra:** Built-in support for symbolic differentiation (`diff`), resilient numerical integration (`intg`), and rigorous matrix algebra operations (determinants `det`, eigenvalues `eigvalues`, and linear system solvers `linear_sys`).
* **Vectorized Orchestration:** Leverages hardware-level vector registers via the `jdk.incubator.vector` API, while maintaining a smooth Android-compliant scalar fallback layer.
* **Versatile Execution Architecture:** Fully handles complex logical expressions, user-defined functions (`@(x,y)...`), native equation root solvers (Quadratic up to Tartaglia cubic arrays), and real-time geometric graph plotting buffers.
* **Zero Dependencies:** Completely portable and self-contained across Java SE, Android ecosystems, and Legacy JavaME profiles.

---

## ⚡ Turbo Mode: Instant Execution

Turbo mode provides the flexible usability of an interpreter coupled with the absolute execution speed of a compiled execution loop. It creates a specialized, lightweight compiled execution path that bypasses classloading overhead entirely.

```java
// Compile the mathematical expression once
MathExpression me = new MathExpression("x*sin(x) + y*cos(y) + z^2", false);
FastCompositeExpression turbo = me.compileTurbo();

// Evaluate millions of coordinates with zero object allocations on the hot path
double[] frame = new double[3]; 
for (double t = 0; t < 10_000_000; t += 0.001) {
    frame[0] = t; 
    frame[1] = t * 1.5; 
    frame[2] = t / 2.0;
    
    double result = turbo.applyScalar(frame); 
}

```

---

## ⚡ Vectorized Processing: Under the Hood

When using the `SIMDVectorTurboEvaluator`, ParserNG maps your math tokens straight to CPU SIMD hardware registers (e.g., 256-bit AVX2 or 512-bit AVX-512 lanes). This enables the hardware to process large arrays of primitive data concurrently in a minimal number of CPU cycles.

```
[Input Array Buffers] ──► [SIMD Register Packing] ──► [Parallel Vector Ops] ──► [Direct Array Write]
     (Flat Layout)           (256 / 512-bit Lanes)         (Fused Loops)             (Zero Churn)

```

### Dynamic Execution Management

1. **Mechanical Sympathy:** The execution path relies on a flat memory layout (`double[][]` or sequential single arrays) to limit pointer chasing, maximize L1/L2 cache locality, and prevent cache line thrashing.
2. **Automatic Tail Masking:** If your total element scale is not a perfect multiple of your CPU's hardware vector register width (for example, processing exactly 400,017 items), the engine automatically fires a fast scalar fallback loop to clear the remainder elements without dropping execution density.
3. **Zero Allocation:** Memory matrices are reused inside the loop paths, preventing Garbage Collection (GC) pauses from interrupting real-time computational sweeps.

### Example: High-Throughput Vectorized Bulk Sweep

```java
public void bulkRun() throws Throwable {
    // Instantiate a highly transcendental probability density distribution function
    MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
    
    SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = 
        (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

    // 400,017 data points: Automatically coordinates SIMD lanes + tail scalar cleanup
    int totalElements = 400017;
    double[][] inputs = new double[3][totalElements]; 
    double[] outputVector = new double[totalElements];

    // Seed input data matrices
    for (int i = 0; i < totalElements; i++) {
        inputs[0][i] = 1.5 + (i * 0.1); // Variable: x1
        inputs[1][i] = 2.0 + (i * 0.5); // Variable: x2
        inputs[2][i] = 0.5;             // Variable: x3
    }

    // High-throughput vectorized execution on a single core
    evaluator.applyBulk(inputs, outputVector);

    // Architectural Note: Switch to the line below to distribute execution loops
    // across all available hardware cores simultaneously:
    // evaluator.applyBulkParallel(inputs, outputVector);
}

```

---

## 📦 Installation & Configuration

### For Standard Android or Legacy Pre-JDK 21 Runtimes

Add the standalone core dependencies to your project configurations:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.7</version>
</dependency>

```

### For Modern JDK 21+ SIMD Vector Acceleration Environments

Include the core engine alongside the specialized hardware accelerator modules:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.7</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>2.0.7</version>
</dependency>

```

### For Microbenchmarking and Profiling Verification Harnesses

To tap into local benchmarking configurations and explicit scalar fallback APIs:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-bench</artifactId>
    <version>2.0.7</version>
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

Production infrastructures requiring predictable performance, safety assurances, and expert engineering access can upgrade to **ParserNG Enterprise**:

* **Priority Operational Support:** 24/48-hour SLAs for immediate bug tracking, profiling, and compliance vulnerabilities.
* **GraalVM Native Image Deployment:** Turn-key integration configurations for compiling down to lightning-fast native binary footprints.
* **Direct Consultative Access:** Architectural reviews, customized functions (extension), and hand-tailored vector math kernel design directly from the author.

📧 **Contact Corporate Licensing:** `gbenroscience@gmail.com`

☕ **Support Open Source Development:** [GitHub Sponsors Corporate Tiers](https://buymeacoffee.com/gbenroscience/membership)

---

## 📚 Documentation & Technical Resources

* **Deep Dive Benchmarking Logs:** [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md) — Comprehensive execution breakdowns versus competitor runtimes.
* **High-Fidelity Graphical Plotting:** [GRAPHING.md](parser-ng/GRAPHING.md) — Render configuration rules for JavaFX, Swing, and Android surfaces.
* **Bulk Vectorization Blueprints:** [BULK.md](https://www.google.com/search?q=parser-ng/BULK.md) — Optimization techniques for massive array processing.
* **Release Artifact Logs:** [LATEST.md](LATEST.md) — Change logs and technical notes for v2.0.7.
* [MORE.md](MORE.md) — Even more to know
* [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md) — Original readme for pre-1.0 versions with a lot of, still valid, examples

---

*Developed with mathematical rigor and mechanical sympathy by **Gbemiro Jiboye** (@gbenroscience).*
