# ParserNG 2.0.5 🧮⚡

### High-Performance, Interpreted Math Engine for Java

**ParserNG** is a blazing-fast, zero-native-dependency, pure Java math expression parser and evaluator designed for high-throughput applications—ranging from real-time simulations to scientific computing and Android apps.

Unlike compilers that introduce overhead via runtime bytecode generation (e.g., Janino), ParserNG achieves near-native speeds through **mechanical sympathy**, loop unrolling, and the **JDK Vector API**.

## 🏢 Enterprise Support & Commercial Assurances
ParserNG is used by top-tier global enterprises, including Fortune 500 financial institutions, telecom giants, and consulting firms. 
If your organization relies on ParserNG for production-critical math parsing, we offer formal commercial channels to ensure compliance and stability:

* **Priority SLAs:** 24/48-hour guaranteed turnaround on bug fixes, performance optimizations, and security queries.
* **Direct Engineering Support:** Direct email access to the author for architectural advice and custom library extension.
* **Supply Chain Security:** Direct validation of ParserNG's zero-dependency architecture for your internal compliance audits.


👉 **Need Enterprise Support?** Contact the maintainer directly at `gbenroscience@gmail.com` or explore our **[GitHub Sponsors Corporate Tiers](https://buymeacoffee.com/gbenroscience/membership)**.

---

## 🚀 Performance & Throughput

* **Speed Champion:** Rivals and often outperforms established engines like Janino, exp4J, and Parsii across bulk data processing tasks.


* **Throughput:** Standard mode delivers **3-10 million evaluations/sec**; Turbo mode reaches **10-90 million evaluations/sec**.


* **Turbo Mode:** A specialized compiled path for scalar and matrix operations that eliminates class-loading bottlenecks.



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

Add the following to your `pom.xml`:

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.5</version>
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
* **Javadoc:** [Latest API Documentation](https://www.google.com/search?q=https://javadoc.io/doc/com.github.gbenroscience/parser-ng/2.0.5)


[jenkins-report-generic-chart-column](https://github.com/jenkinsci/report-generic-chart-column-plugin/) by Jiri Vanek, makes use of `ExpandingParser`, and his contributions are appreciated.


## 📚 More Documentation

- [BENCHMARK_RESULTS.md](parser-ng/BENCHMARK_RESULTS.md) — full speed comparisons
- [GRAPHING.md](parser-ng/GRAPHING.md) — plotting on Swing / JavaFX / Android
- [LATEST.md](LATEST.md) — what’s new in 2.0.5
- [EXAMPLES.md](EXAMPLES.md) — what’s new in 2.0.5
- Javadoc: https://javadoc.io/doc/com.github.gbenroscience/parser-ng/2.0.5
- [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md) — Original readme for pre-1.0 versions with a lot of, still valid, examples
---

**Developed by GBENRO JIBOYE (@gbenroscience)**

*If ParserNG helps your project, please consider starring the repository or [buying me a coffee](https://buymeacoffee.com/gbenroscience).*
