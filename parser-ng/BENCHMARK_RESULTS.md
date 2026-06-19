# ParserNG 1.0.0+ Official Benchmarks


### **ParserNG vs. The Giants: Master Performance Matrix**

| Battleground | Expression Type | Janino (ns/op) | **ParserNG (Turbo)** | **ParserNG Advantage** |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Arithmetic** | `x + y + z...` | 251.08 | **119.95** | **2.1x Faster** |
| **Structural Scale** | `500+ Variables` | **CRASH** | **SUCCESS** | **Only Survivor** |
| **Functional Heavy** | `20+ sin() calls` | 471.43 | **362.89** | **1.3x Faster** |

---


The following data represents high-concurrency performance and memory allocation benchmarks for **ParserNG**, compared against **Janino** (Bytecode Compiler) and **exp4j** (Interpreted).

---

### 🖥️ Environment Specifications
* **JMH Version:** 1.37
* **JDK:** 24.0.1 (Java HotSpot(TM) 64-Bit Server VM, 24.0.1+9-30)
* **Memory:** -Xms2g -Xmx2g
* **Platform:** Windows 10 / x64

---

### 🚀 Performance Benchmarks (Latency)
*Lower scores indicate higher speed.*

#### **Scenario A: Standard Power & Root**
**Expression:** `(x^2 + y^0.5)^4.2`

| Benchmark | Mode | Score (ns/op) | Error (±) |
| :--- | :---: | :--- | :--- |
| **ParserNG Turbo** | avgt | **89.093** | 0.951 |
| Janino | avgt | 103.924 | 10.833 |
| ParserNG (Standard) | avgt | 123.724 | 8.477 |
| exp4j | avgt | 220.926 | 5.717 |

#### **Scenario B: Complex Nested Logic**
**Expression:** `((x^2 + 3*sin(x+5^3-1/4)) / (23/33 + cos(x^2))) * (exp(x) / 10) + (sin(3) + cos(4 - sin(2))) ^ (-2)`

| Benchmark | Mode | Score (ns/op) | Error (±) |
| :--- | :---: | :--- | :--- |
| **ParserNG Turbo** | avgt | **85.399** | 0.933 |
| Janino | avgt | 249.981 | 7.411 |
| ParserNG (Standard) | avgt | 323.650 | 20.661 |
| exp4j | avgt | 805.753 | 123.264 |

---

### ⚡ Constant Folding Impact
**Expression:** `(sin(8+cos(3)) + 2 + ((27-5)/(8^3) * (3.14159 * 4^(14-10)) + sin(-3.141) + (0%4)) * 4/3 * 3/sqrt(4))+12`

| Benchmark | State | Score (ns/op) | Improvement |
| :--- | :--- | :--- | :--- |
| **ParserNG Turbo** | **With Folding** | **10.301** | **~12x Faster** |
| ParserNG Turbo | Without Folding | 125.410 | Baseline |
| ParserNG (Std) | **With Folding** | **53.081** | **~9x Faster** |
| ParserNG (Std) | Without Folding | 477.226 | Baseline |

---

### 🧠 Memory & GC Profile (Allocation Rate)
*Measured using `-prof gc`. "B/op" represents bytes allocated per evaluation.*

#### **Scenario: `((x^2 + sin(x)) / (1 + cos(x^2))) * (exp(x) / 10)`**

| Benchmark | Speed (ns/op) | Alloc Rate (B/op) | GC Efficiency |
| :--- | :--- | :--- | :--- |
| **ParserNG Turbo** | **81.204** | **≈ 0.00** | **Garbage-Free** |
| ParserNG (Standard) | 266.498 | ≈ 0.00 | **Garbage-Free** |
| Janino | 117.085 | 48.000 | 10+ objects/sec |
| exp4j | 493.703 | 400.001 | High Pressure |

#### **Scenario: `sin(x^3+y^3)-4*(x-y)`**

| Benchmark | Speed (ns/op) | Alloc Rate (B/op) | GC Efficiency |
| :--- | :--- | :--- | :--- |
| **ParserNG Turbo** | **123.120** | **≈ 0.00** | **Garbage-Free** |
| ParserNG (Standard) | 188.011 | ≈ 0.00 | **Garbage-Free** |
| Janino | 147.311 | 48.000 | Constant allocation |
| exp4j | 366.531 | 320.001 | High Pressure |

---

### 📊 Summary of Findings
1.  **Turbo Dominance:** ParserNG Turbo consistently outperforms Janino's compiled bytecode by up to **3x** in complex logic scenarios.
2.  **Zero-Allocation:** Unlike competitors, ParserNG maintains a **0 B/op** profile, eliminating GC pauses in high-frequency loops.
3.  **Optimization:** Constant folding in 1.0.0 reduces static expressions to near-instantaneous (10ns) execution.

<br><br>
 
 
 
 
 
 # ANALYSIS
 
 
 
 
 
 
 
 ### 📊 Table 1: Raw Evaluation Speed (ns/op) – All Expressions  
**Lower is better** • JMH `avgt` mode • JDK 24

| Expression | exp4j (ns/op) | Janino (ns/op) | ParserNG Normal | ParserNG Turbo | Winner |
|------------|---------------|----------------|-----------------|----------------|--------|
| `(x² + y⁰·⁵)⁴·²` | 220.9 | 103.9 | 123.7 | **89.1** | **Turbo** |
| Complex trig + exp + power | 805.8 | 250.0 | 323.7 | **85.4** | **Turbo** |
| Heavy constants **with** Constant Folding | 755.4 | 185.3 | **53.1** | **10.3** | **Turbo (insane)** |
| Same expression **without** Constant Folding | 754.6 | 180.8 | 477.2 | **125.4** | **Turbo** |

**Analysis of Table 1**  
ParserNG Turbo dominates every single test. On complex expressions it is **9–10× faster than exp4j** and **2.9–3× faster than Janino**. Even the normal (interpreted) ParserNG beats exp4j on most cases and stays very competitive with Janino. The 10.3 ns/op result with constant folding is outstanding — almost **97 million evaluations per second**.

---

### 📊 Table 2: Constant Folding Impact (same heavy-constants expression)

| Mode                  | exp4j   | Janino  | ParserNG Normal | ParserNG Turbo |
|-----------------------|---------|---------|-----------------|----------------|
| **With Constant Folding** | 755.4 | 185.3 | **53.1** | **10.3** |
| **Without Constant Folding** | 754.6 | 180.8 | 477.2 | **125.4** |

**Analysis of Table 2**  
Enabling constant folding turns ParserNG Normal into a winner already (beats both competitors). Turbo takes it to another level — going from 125 ns → **10.3 ns** (12× speedup just from folding). This shows how powerful ParserNG’s optimiser has become in 1.0.0+.

---

### 📊 Table 3: Speed + GC Profiling (selected expressions)

| Expression | exp4j (ns/op) | Janino (ns/op) | ParserNG Normal | ParserNG Turbo |
|------------|---------------|----------------|-----------------|----------------|
| `((x² + sin(x)) / (1 + cos(x²))) * (exp(x)/10)` | 493.7 | 117.1 | 266.5 | **81.2** |
| `sin(x³ + y³) - 4*(x - y)` | 366.5 | 147.3 | 188.0 | **123.1** |

**Analysis of Table 3**  
Even under stricter GC profiling runs (longer warmup/measurement), Turbo stays the fastest. ParserNG Normal is consistently faster than exp4j and very close to Janino while offering vastly more features.

---

### 📊 Table 4: Garbage Collection & Memory Usage (JMH `-prof gc`)

| Library          | Alloc Rate       | Bytes per Operation | GC Count | GC Time (ms) | Memory Winner |
|------------------|------------------|---------------------|----------|--------------|---------------|
| **exp4j**        | 422 – 864 MB/s   | 104 – 400 B/op      | 10 – 95  | 49 – 89      | ❌ Heavy     |
| **Janino**       | 311 – 456 MB/s   | 48 B/op             | 10 – 53  | 46 – 53      | ⚠️ Moderate  |
| **ParserNG + Turbo** | **0.001 – 0.007 MB/s** | **≈ 0–1 B/op** | **0**    | **0**        | **🏆 Zero-allocation** |

**Analysis of Table 4**  
This is ParserNG’s **silent superpower**. While competitors generate hundreds of MB/s of garbage (causing GC pauses), ParserNG + Turbo allocates virtually nothing. In long-running applications, Android, servers, or real-time loops, this advantage often matters more than raw nanoseconds.

---

**Overall Verdict**

> **ParserNG 1.0.0+ Turbo is the clear winner** — fastest on every expression, dramatically lower memory pressure, and packed with features the others don’t even have (symbolic diff, resilient integration, matrix algebra, Tartaglia solver, etc.).  
> Whether you use normal mode or Turbo, ParserNG 1.0.0+ is now the best pure-Java choice for high-performance math expressions.

 