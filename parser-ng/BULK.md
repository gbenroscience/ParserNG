# ParserNG 2.0.0

An ultra-high-performance mathematical expression parsing and evaluation engine for Java. Version 2.0.0 introduces a breakthrough **Block Vectorization Engine** (`VectorTurboEvaluator`) and `SIMDVectorTurboEvaluator` powered by the Java **Vector API** (`jdk.incubator.vector`). 

By processing data pipelines natively inside hardware registers, ParserNG completely eliminates runtime allocation overhead and flattens interpreter branches. It outpaces dynamic bytecode compilation frameworks (like Janino) on massive datasets by leveraging bare-metal SIMD hardware capabilities.

---

## Key Features

* **Hardware-Level SIMD Execution**: Automatically binds operations to modern CPU vector tracks (AVX-512, AVX2, or ARM Neon) via the Java Incubator Vector API.
* **Outperforming Bytecode Compilation**: Beats runtime bytecode compilation by eliminating pointer-chasing and maximizing instruction-level parallelism.
* **Zero Allocation on Hot Paths**: Eradicates heap-allocation overhead inside heavy processing loops, preventing Garbage Collection (GC) spikes in real-time or high-frequency calculation environments.
* **Advanced Memory Layout Topology**: Supports standard multi-dimensional matrices (`double[][]`) as well as linear high-performance, column-major stride-mapped arrays (`double[]`).
* **Flexible Execution Models**: Transparently shifts between standard sequential tiled loops, chunk-batched blocks, and asynchronous multi-threaded thread pools.
* **Vectorized User-Defined Functions (UDFs)**: Parse, compile, and execute multi-parameter custom functions efficiently across deep data lanes.

---

## Performance Benchmarks (JMH Results)

The following benchmark profile demonstrates evaluation throughput across massive data scales. As record pools increase, ParserNG's manual block vectorization scales aggressively, beating Janino by **up to 2x**.

* **Data Scales Evaluated**: $1,000,000$ and $67,108,864$ floating-point elements.
* **Topology Scaling**: `MathEvalBenchmarkForSIMD` utilizes 2 parallel execution worker threads.

```text
Benchmark                                     (dataSize)                                                             (expression)  Mode  Cnt            Score           Error  Units
MathEvalBenchmark.janino                          1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30     10187544.823 ±   752281.235  ns/op
MathEvalBenchmark.janino:asm                      1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmark.janino                          1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30     24072886.735 ±  1402925.987  ns/op
MathEvalBenchmark.janino:asm                      1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---
MathEvalBenchmark.janino                         67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30    652196391.200 ± 27609006.557  ns/op
MathEvalBenchmark.janino:asm                     67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmark.janino                         67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30   1578542916.300 ± 40349710.413  ns/op
MathEvalBenchmark.janino:asm                     67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---

MathEvalBenchmark.parserNG                        1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30      9360288.977 ±   760429.699  ns/op
MathEvalBenchmark.parserNG:asm                    1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmark.parserNG                        1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30     19257762.340 ±  1065339.581  ns/op
MathEvalBenchmark.parserNG:asm                    1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---
MathEvalBenchmark.parserNG                       67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30    570989708.567 ± 19005754.577  ns/op
MathEvalBenchmark.parserNG:asm                   67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmark.parserNG                       67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30   1284552658.233 ± 69148183.122  ns/op
MathEvalBenchmark.parserNG:asm                   67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---

MathEvalBenchmarkForSIMD.janino                   1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30      9856751.812 ±   535025.997  ns/op
MathEvalBenchmarkForSIMD.janino:asm               1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmarkForSIMD.janino                   1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30     29323096.988 ±  4035993.090  ns/op
MathEvalBenchmarkForSIMD.janino:asm               1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---
MathEvalBenchmarkForSIMD.janino                  67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30    838930604.167 ± 249394348.633  ns/op
MathEvalBenchmarkForSIMD.janino:asm              67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmarkForSIMD.janino                  67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30   1584398907.167 ± 40514372.979  ns/op
MathEvalBenchmarkForSIMD.janino:asm              67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---

MathEvalBenchmarkForSIMD.parserNG                 1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30      6346274.914 ±  1266473.757  ns/op
MathEvalBenchmarkForSIMD.parserNG:asm             1000000                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmarkForSIMD.parserNG                 1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30     13538502.496 ±  3115835.887  ns/op
MathEvalBenchmarkForSIMD.parserNG:asm             1000000  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---
MathEvalBenchmarkForSIMD.parserNG                67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt   30    371223062.917 ± 59975811.391  ns/op
MathEvalBenchmarkForSIMD.parserNG:asm            67108864                               12*x1 + 3*x2 - 4*x3 + 5*x1 - x2 - 4*x3 + 2*x1 + x2  avgt                   NaN                ---
MathEvalBenchmarkForSIMD.parserNG                67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt   30    749919491.967 ± 35895952.492  ns/op
MathEvalBenchmarkForSIMD.parserNG:asm            67108864  0.39894228 / x1 * exp(-((x2 - x3) * (x2 - x3)) / (2 * x1 * x1))  avgt                   NaN                ---

```

### Analysis

On an intensive mathematical workload executing $67.1\text{M}$ operations under complex transcendental configurations (`exp` models), a single-threaded **ParserNG** run registers at **$1284\text{ ms}$** vs Janino's **$1578\text{ ms}$**.

When utilizing multi-threaded parallel execution splits via `MathEvalBenchmarkForSIMD` (2 parallel workers), ParserNG drops computation time to **$749\text{ ms}$**—completely outstripping Janino's **$1584\text{ ms}$** bytecode path.

---

## API Usage Guide

### 1. Multi-Dimensional Array Processing (`applyBulk`)

Evaluate expressions across independent variables using standard data layouts:

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression;

MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");
BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

int totalElements = 2000;
double[][] inputs = new double[3][totalElements]; // x1, x2, x3 tracks
double[] outputVector = new double[totalElements];

// Execute with hardware-aligned loop tiling enabled
boolean tiledExecution = true;
evaluator.applyBulk(inputs, outputVector, tiledExecution);

```

### 2. Flattened Column-Major Stride Layouts

Eliminate spatial extraction jumps and optimize CPU L1/L2 data cache usage by linearizing your input parameters into a flat 1D segment buffer:

```java
int totalElements = 2000;
int varCount = 3;
double[] flatInputs = new double[varCount * totalElements];
double[] outputVector = new double[totalElements];

// Distribute tracking segments via column-major indexing strides
for (int i = 0; i < totalElements; i++) {
    flatInputs[(0 * totalElements) + i] = 1.5 + (i * 0.1); // x1 segment
    flatInputs[(1 * totalElements) + i] = 2.0 + (i * 0.5); // x2 segment
    flatInputs[(2 * totalElements) + i] = 0.5;             // x3 segment
}

evaluator.applyBulk(flatInputs, outputVector, true);

```

### 3. Chunk-Batched Layout Boundaries (`applyBulkBatched`)

Enforce fixed matrix block chunk allocations to tightly align loop boundaries with specific target sizes:

```java
int batchSize = 128;
evaluator.applyBulkBatched(flatInputs, outputVector, batchSize, true);

```

### 4. High-Throughput Thread-Pooled Scaling (`applyBulkParallel`)

Leverage an integrated multi-threaded background executor structure to divide large-scale workloads across multiple processor cores:

```java
evaluator.applyBulkParallel(flatInputs, outputVector);

```

### 5. Vectorized User-Defined Functions (UDF)

Compute custom nested algebraic function assignments natively inside the evaluation pass:

```java
MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(x+3,y-2,2*z-3)");
BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

double[] inputs = new double[]{5.0, 4.0, 1.0}; // x, y, z arrays
double[] output = new double[1];

evaluator.applyBulk(inputs, output, false);

```

---

## Technical Configuration Requirements

Because this module targets explicit low-level CPU vector registers, it requires **Java 21** or later, along with JVM incubator arguments to activate the platform layer.

### Mandated JVM Parameters

Ensure the following parameter flags are appended to your runtime run targets or test configurations:

```bash
--add-modules jdk.incubator.vector

```

### Maven Build Definitions

Add the matching compilation module allocations inside your engine `pom.xml`:

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-compiler-plugin</artifactId>
    <version>3.11.0</version>
    <configuration>
        <release>21</release>
        <compilerArgs>
            <arg>--add-modules</arg>
            <arg>jdk.incubator.vector</arg>
        </compilerArgs>
    </configuration>
</plugin>

```

```

```