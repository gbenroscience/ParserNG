# ParserNG 2.0.6 🧮⚡

### The Fastest Pure-Java Expression Engine + Vector Math Kernels

**ParserNG** is a pure-Java expression evaluation engine designed for high-performance enterprise workloads, including complex mathematical pipelines and transformer activations. 
By utilizing a high-speed interpreted model instead of risky bytecode generation, it achieves near-native throughput with zero JNI, simplifying your SBOM and compliance posture. 
Built on the JDK Vector API and principles of mechanical sympathy, ParserNG is the secure, portable choice for modern JVM infrastructures.

**Benchmarks:** `SwiGLU: 1.8ns/elem` | `GeLU: 2.1ns/elem` | `Expr: scalable(generally faster than Janino on single core)` | `JDK 21+ |(SIMD) & maintains compatiblility till JDK8`

---

### 🏢 Trusted in Production 
**11,000+ Maven Central downloads** | **+250% growth in 90 days** | **618 organizations**

**Enterprise Support & Commercial Assurances**
Production teams choose ParserNG Enterprise for:
* **Priority SLAs:** 24/48h turnaround on bugs, perf, and security 
*  **Zero JNI**: Pure Java. No native crashes, easy SBOM, passes compliance.
*  **Direct Engineering:** Architectural review + custom kernel work from the author
*  **Vector API Speed**: SwiGLU `1.8ns/elem`, GeLU `2.1ns/elem` on JDK 21+.
*  **Enterprise Ready**: Private Maven, Graal Native Image, AVX512 builds, CVE/SBOM reports.

👉 **Contact:** `gbenroscience@gmail.com`  
👉 **Sponsor / SMB:** [GitHub Sponsors Corporate Tiers](https://buymeacoffee.com/gbenroscience/membership)


---

## 🚀 Performance & Throughput

* **Speed Champion:** Rivals and often outperforms established engines like Janino, and Parsii across bulk data processing tasks.

* **Throughput:** Standard mode delivers **3-10 million evaluations/sec**; Turbo mode reaches **10-90 million evaluations/sec**; Bulk Turbo: **Single core - up to 200million evaluations/sec, With workers - even more!**.

* **Turbo Mode:** A specialized compiled path for scalar, matrix operations and bulk operations that eliminates class-loading bottlenecks.



## ✨ Key Capabilities

* **Calculus & Algebra:** Symbolic differentiation (`diff`), resilient numerical integration (`intg`), and comprehensive matrix algebra (`det`, `eigvalues`, `linear_sys`).


* **Vectorized Processing:** Supports modern hardware acceleration via the `jdk.incubator.vector` API, with an Android-compatible scalar fallback.


* **Versatile Execution:** Handles logical expressions, user-defined functions (`@(x,y)…`), equation solvers (Quadratic, Tartaglia cubic), and 2D geometric plotting.


* **Zero-Dependency:** Portable across Java SE, Android, and JavaME.



---

## ⚡ Turbo Mode: Instant Execution

Turbo mode provides the flexibility of an interpreter with the throughput of a compiler.

```java
// Compile once
MathExpression me = new MathExpression("x*sin(x) + y*cos(y) + z^2", false);
FastCompositeExpression turbo = me.compileTurbo();

// Evaluate millions of times with zero allocation
double[] frame = new double[3]; 
for (double t = 0; t < 10_000_000; t += 0.001) {
    frame[0] = t; frame[1] = t * 1.5; frame[2] = t / 2.0;
    double result = turbo.applyScalar(frame); 
}

```

## ⚡ Vectorized Processing: Under the Hood
When using the SIMDVectorTurboEvaluator, ParserNG automatically maps your mathematical expressions to CPU-level SIMD (Single Instruction, Multiple Data) lanes. This allows the processor to compute multiple data points simultaneously within a single clock cycle, rather than iterating through them one by one.

### How it works
The applyBulk method handles the complex orchestration of data alignment and lane masking for you:

SIMD Lane Utilization: The engine packs your input arrays into registers (e.g., 256-bit or 512-bit), allowing it to process multiple double values per operation.

Tail Handling: When your dataset size is not a perfect multiple of the SIMD lane width (like the 400,017 elements in the example below), the engine automatically manages the "tail" using a highly optimized scalar fallback loop, ensuring no data is missed and performance remains consistent.

Mechanical Sympathy: The evaluator uses a flat memory layout for your inputs and outputVector to minimize CPU cache misses, keeping the data pipelines full.

Example: Vectorized Bulk Evaluation
This example illustrates how to process large datasets efficiently, triggering both the hardware-accelerated vector lanes and the necessary remainder logic:

```Java
public void bulkRun() throws Throwable {
    MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
    SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression evaluator = 
        (SIMDVectorTurboEvaluator.SIMDVectorCompositeExpression) new SIMDVectorTurboEvaluator(me).compile();

    // 400,017 datapoints: Automatically triggers SIMD lanes + tail scalar loop
    int totalElements = 400017;
    double[][] inputs = new double[3][totalElements]; 
    double[] outputVector = new double[totalElements];

    // Prepare data
    for (int i = 0; i < totalElements; i++) {
        inputs[0][i] = 1.5 + (i * 0.1); // x1
        inputs[1][i] = 2.0 + (i * 0.5); // x2
        inputs[2][i] = 0.5;             // x3
    }

    // High-throughput vectorized execution
    evaluator.applyBulk(inputs, outputVector);

   // Use evaluator.applyBulkParallel(inputs, outputVector); to automatically process using all your cores
    
    // Result is now stored in outputVector
}

```



---

## 📦 Installation

To use in your Android or pre-JDK21 environment, add the following to your `pom.xml`:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.6</version>
</dependency>
```

To use in JDK21+ environments, do:


```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.6</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>2.0.6</version>
</dependency>
```

To run the SIMD/scalar fallback APIs for BULK evaluation, add:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-bench</artifactId>
    <version>2.0.6</version>
</dependency>
```

---

## 📊 Feature Comparison

| Category | Supported Features | Turbo Support |
| --- | --- | --- |
| **Arithmetic** | `+ - * / ^ % and or == !=` | Full |
| **Trigonometry** | `sin cos tan asin … sinh` | Full |
| **Calculus** | `diff` (symbolic), `intg` (resilient) | Yes |
| **Matrices** | `det`, `eigvalues`, `eigvec`, `adjoint` | Excellent |
| **Statistics** | `avg`, `variance`, `rms`, `sort` | Yes |

---

## 📚 Documentation & Resources

* **Benchmarks:** [BENCHMARK_RESULTS.md](https://www.google.com/search?q=BENCHMARK_RESULTS.md)
* **Bulk Evaluation:** [Bulk Evaluation Guide](https://www.google.com/search?q=parser-ng/BULK.md)
* **Graphing:** [GRAPHING.md](https://www.google.com/search?q=GRAPHING.md)
* **Javadoc:** [Latest API Documentation](https://www.google.com/search?q=https://javadoc.io/doc/com.github.gbenroscience/parser-ng/2.0.6)


[jenkins-report-generic-chart-column](https://github.com/jenkinsci/report-generic-chart-column-plugin/) by Jiri Vanek, makes use of `ExpandingParser`, and his contributions are appreciated.


## 📚 More Documentation

- [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md) — full speed comparisons
- [GRAPHING.md](parser-ng/GRAPHING.md) — plotting on Swing / JavaFX / Android
- [LATEST.md](LATEST.md) — what’s new in 2.0.6
- [EXAMPLES.md](EXAMPLES.md) — what’s new in 2.0.6
- Javadoc: https://javadoc.io/doc/com.github.gbenroscience/parser-ng/2.0.6
- [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md) — Original readme for pre-1.0 versions with a lot of, still valid, examples
---

**Developed by GBENRO JIBOYE (@gbenroscience)**

*If ParserNG helps your project, please consider starring the repository or [buying me a coffee](https://buymeacoffee.com/gbenroscience).*
