# ParserNG 2.0.4

An ultra-high-performance mathematical expression parsing and evaluation engine for Java. Version 2.0.4 introduces a breakthrough **Block Vectorization Engine** (`VectorTurboEvaluator`) and `SIMDVectorTurboEvaluator` powered by the Java **Vector API** (`jdk.incubator.vector`). 

By processing data pipelines natively inside hardware registers, ParserNG completely eliminates runtime allocation overhead and flattens interpreter branches. It outpaces dynamic bytecode compilation frameworks (like Janino) on massive datasets by leveraging bare-metal SIMD hardware capabilities.

---




## Distinction - `SIMDVectorTurboEvaluator` vs `VectorTurboEvaluator`

These 2 classes are responsible independently for bulk evaluations in ParserNG.

### SIMDVectorTurboEvaluator

The **`SIMDVectorTurboEvaluator`** is the hardware-accelerated bulk processing powerhouse of ParserNG. It is specifically built to leverage the cutting-edge **JDK 21 Vector API** (`jdk.incubator.vector`), allowing the JVM to compile mathematical evaluation pipelines directly into native SIMD (Single Instruction, Multiple Data) CPU instructions such as AVX or ARM NEON.

Instead of processing array elements one by one, it binds operations across hardware vector lanes to process large chunks of data in a single CPU clock cycle.

* **Performance Profile:** Unmatched throughput on large mathematical datasets, executing bulk computations significantly faster than lightweight runtime bytecode compilers like Janino.
* **Environment Requirements:** Requires **Java 21+** and the explicit runtime configuration flag: `--add-modules jdk.incubator.vector`.
* **Best Used For:** High-frequency, massive data streams, heavy numerical integration, spatial data grids, and large-scale bulk vector transformations where hardware acceleration is fully available.

### VectorTurboEvaluator

The **`VectorTurboEvaluator`** is the standard, highly portable bulk evaluation engine embedded directly within the core architecture of ParserNG. It processes mathematical expressions across data arrays using highly optimized sequential loops and zero-allocation memory profiles powered by our foundational interpretation algorithms.

It achieves the absolute peak of pure interpreted execution speed without binding itself to specialized hardware architecture features.

* **Performance Profile:** Extremely fast with zero warm-up time or compilation overhead, outperforming standard loop-based expression evaluators on small to medium-sized data batches.
* **Environment Requirements:** Highly backward-compatible and portable. It runs anywhere the core library runs (**Java 8+**), requiring no special JVM incubator module flags or environment modifications.
* **Best Used For:** Cross-platform systems (including Android and legacy enterprise backend servers), environments where adding specialized JVM flags is restricted, or applications handling small-to-medium dataset arrays where CPU-level SIMD orchestration overhead isn't justified.


To use SIMDVectorTurboEvaluator(e.g. on modern servers(JDK 21+) and laptops(JDK21+)), add the 2 dependencies below to your application's `pom.xml`:
```XML
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.4</version>
</dependency>
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-simd</artifactId>
    <version>2.0.4</version>
</dependency>
```

To use VectorTurboEvaluator(e.g. on Android and legacy systems supporting <JDK21), add to your application's `pom.xml`:
```XML
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>2.0.4</version>
</dependency>
```

They benchmark at *almost* same speed and same allocation, and you can use workers to increase their throughput
---



### Comparison Matrix

| Feature | `SIMDVectorTurboEvaluator` | `VectorTurboEvaluator` |
| --- | --- | --- |
| **Execution Layer** | Hardware CPU Vector Lanes (SIMD) | Highly Optimized Sequential CPU Loops |
| **Minimum Java Version** | Java 21+ | Java 8+ |
| **JVM Configuration** | Requires `--add-modules jdk.incubator.vector` | Zero configuration (Works out of the box) |
| **Portability** | Limited to environments supporting JDK 21+ | Universal (Cross-platform, Android, Desktop) |
| **Optimal Data Scale** | Massive datasets (Thousands to millions of data points) | Same |
| **Warm-Up Overhead** | Negligible, but bound to vector lane size queries | Absolute zero |


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

MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");//Gaussian

        int stride = meLinear.getVariablesNames().length
BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

int totalElements = 2000;
double[][] inputs = new double[stride][totalElements]; // x1, x2, x3 tracks
double[] outputVector = new double[totalElements];

  for (int i = 0; i < dataSize; i++) {
            // 1. Populate the 2D SoA layout for the standard evaluator
            inputs[0][i] = 1.5 + rand.nextDouble() * 5.0;  // x1
            inputs[1][i] = rand.nextDouble() * 5.0;        // x2
            inputs[2][i] = rand.nextDouble() * 2.0;        // x3
        }

// Execute with hardware-aligned loop tiling enabled
boolean tiledExecution = true;
evaluator.applyBulk(inputs, outputVector, tiledExecution);

```


### 2. Flat Array Processing (`applyBulk`)

Evaluate expressions across independent variables using standard data layouts:

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator.BatchedVectorCompositeExpression;

MathExpression me = new MathExpression("(1 / (x1 * sqrt(2 * 3.14159))) * exp((-(x2 - x3)^2) / (2 * x1^2))");

        int stride = meLinear.getVariablesNames().length
BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();
  double[]flatVariables = new double[stride * dataSize];

int totalElements = 200000;
double[] outputVector = new double[totalElements];

  for (int i = 0; i < dataSize; i++) {
            // 2. Calculate the flat interleaved base address for timestep 'i'
            // For i=0, base=0. For i=1, base=3. For i=2, base=6...
            int base = i * stride;
            // 3. Populate the flat interleaved array perfectly
            for (int k = 0; k < stride; k++) {
                flatVariables[base + k] = 1.5 + rand.nextDouble() * 5.0;
            }
        }

// Execute with hardware-aligned loop tiling enabled
boolean tiledExecution = true;
evaluator.applyBulk(flatVariables, outputVector, tiledExecution);

```

### 3. Flattened Column-Major Stride Layouts

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

### 4. Chunk-Batched Layout Boundaries (`applyBulkBatched`)

Enforce fixed matrix block chunk allocations to tightly align loop boundaries with specific target sizes:

```java
int batchSize = 128;
evaluator.applyBulkBatched(flatInputs, outputVector, batchSize, true);

```

### 5. High-Throughput Thread-Pooled Scaling (`applyBulkParallel`)

Leverage an integrated multi-threaded background executor structure to divide large-scale workloads across multiple processor cores:

```java
evaluator.applyBulkParallel(flatInputs, outputVector);

```

### 6. Vectorized User-Defined Functions (UDF)

Compute custom nested algebraic function assignments natively inside the evaluation pass:

```java
MathExpression me = new MathExpression("f(x,y,z)=3*x+4*y+sin(z-2);f(x+3,y-2,2*z-3)");
BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

double[] inputs = new double[]{5.0, 4.0, 1.0}; // x, y, z arrays
double[] output = new double[1];

evaluator.applyBulk(inputs, output, false);

```

### Matrix operation (Gelu-200x200) (7.3ns per element on a single core-2.5GHz)
```

    @Test
    void testGelu() throws Throwable {

        MathExpression me = new MathExpression("x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))");

        BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

        int sz = 200;
        FlatMatrixF in1 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in1);

        FlatMatrixF in2 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in2);

        FlatMatrixF out = new FlatMatrixF(sz, sz);

        double n = 10000;
        double t = System.nanoTime();
        for (int i = 0; i < n; i++) {
            evaluator.applyMatrixKernel(new FlatMatrixF[]{in1, in2}, out, "gelu");
        }

        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + (t1/n) + "ns--- answer: ");
        //System.out.println("timed at = " + (t1/n) + "ns--- answer: " + out);

        assertTrue(true);

    }
```
### Matrix operation (Swiglu-200x200) (13.3ns per element on a single core-2.5GHz)


```
     @Test
    void testSwiglu() throws Throwable {

        MathExpression me = new MathExpression("x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))");

        BatchedVectorCompositeExpression evaluator = (BatchedVectorCompositeExpression) new VectorTurboEvaluator(me).compile();

        int sz = 200;
        FlatMatrixF in1 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in1);

        FlatMatrixF in2 = new FlatMatrixF(sz, sz);
        FlatMatrixF.randomFill(in2);

        FlatMatrixF out = new FlatMatrixF(sz, sz);

        double n = 10000;
        double t = System.nanoTime();
        for (int i = 0; i < n; i++) {
            evaluator.applyMatrixKernel(new FlatMatrixF[]{in1, in2}, out, "swiglu");
        }

        double t1 = System.nanoTime() - t;

        System.out.println("timed at = " + (t1/n) + "ns--- answer: ");
        //System.out.println("timed at = " + (t1/n) + "ns--- answer: " + out);

        assertTrue(true);

    }
```

The supported matrix fuctions at the moment are:
- matmul
- matmul_bias_gelu
- matmul_add_sin
- softmax
- relu
- gelu
- q8_quantize
- q8_dequant
- q8_absmax_quant
- rope_split
- accum_v
- matmul_1xn_axpy
- matmul_bias_relu
- rms_norm
- layer_norm
- matmul_bias_silu
- mha_attention
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