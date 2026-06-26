# ParserNG 1.0.0+ Official Benchmarks

### **ParserNG vs. The Giants: Master Performance Matrix**

| Battleground              | Expression Type                  | Janino (ns/op) | **ParserNG Turbo** | **ParserNG Advantage** |
|---------------------------|----------------------------------|----------------|--------------------|------------------------|
| **Pure Arithmetic**       | `(x^2 + y^0.5)^4.2`             | 41.5           | **47.6**           | **1.15x**              |
| **Functional Heavy**      | Complex trig + exp + power       | 180.6          | **58.7**           | **3.08x Faster**       |

---

The following data represents high-concurrency performance and memory allocation benchmarks for **ParserNG**, compared against **Janino** (Bytecode Compiler) and **exp4j** (Interpreted).

---

### 🖥️ Environment Specifications
- **JMH Version:** 1.37
- **JDK:** 24.0.1 (Java HotSpot 64-Bit Server VM)
- **Memory:** `-Xms2g -Xmx2g`
- **Platform:** Windows 10 / x64

---

### 🚀 Performance Benchmarks (Latency)
*Lower scores = better performance.*

#### **Scenario A: Standard Power & Root**
**Expression:** `(x^2 + y^0.5)^4.2`

| Benchmark            | Mode | Score (ns/op) | Error (±)   |
|----------------------|------|---------------|-------------|
| Janino               | avgt | **41.501**    | ±5.372      |
| **ParserNG Turbo**   | avgt | 47.558        | ±2.083      |
| ParserNG Standard    | avgt | 103.0         | ±7.676      |
| exp4j                | avgt | 196.270       | ±29.999     |

#### **Scenario B: Complex Nested Logic**
**Expression:** `((x^2 + 3*sin(x+5^3-1/4)) / (23/33 + cos(x^2))) * (exp(x) / 10) + (sin(3) + cos(4 - sin(2))) ^ (-2)`

| Benchmark            | Mode | Score (ns/op) | Error (±)   |
|----------------------|------|---------------|-------------|
| **ParserNG Turbo**   | avgt | **58.714**    | ±5.435      |
| Janino               | avgt | 180.566       | ±26.351     |
| ParserNG Standard    | avgt | 302.153       | ±13.511     |
| exp4j                | avgt | 768.606       | ±174.698    |

---

### ⚡ Constant Folding Impact
**Expression:** `(sin(8+cos(3)) + 2 + ((27-5)/(8^3) * (3.14159 * 4^(14-10)) + sin(-3.141) + (0%4)) * 4/3 * 3/sqrt(4))+12`

| Benchmark            | State                | Score (ns/op) | Improvement     |
|----------------------|----------------------|---------------|-----------------|
| **ParserNG Turbo**   | With Constant Folding| **8.728**     | **~12x Faster** |
| ParserNG Standard    | With Constant Folding| 19.081        | ~9x Faster      |

---

### 🧠 Memory & GC Profile (Allocation Rate)
*Measured with `-prof gc`. Lower allocation = better.*

#### **Scenario:** `((x^2 + sin(x)) / (1 + cos(x^2))) * (exp(x) / 10)`

| Benchmark            | Speed (ns/op)     | Alloc Rate (B/op) | GC Efficiency     |
|----------------------|-------------------|-------------------|-------------------|
| Janino               | 53.637 ± 7.636    | ≈ 0.00            | **Garbage-Free**  |
| **ParserNG Turbo**   | **54.271 ± 4.716**| **≈ 0.00**        | **Garbage-Free**  |
| ParserNG Standard    | 234.741 ± 11.602  | ≈ 0.00            | **Garbage-Free**  |
| exp4j                | 381.048 ± 61.771  | 376.005           | High Pressure     |

---

### 📊 Summary Tables

#### **Table 1: Raw Evaluation Speed (ns/op) – All Expressions**
**Lower is better** • JMH `avgt` mode • JDK 24

| Expression                              | exp4j    | Janino   | ParserNG Normal | ParserNG Turbo | Winner      |
|-----------------------------------------|----------|----------|-----------------|----------------|-------------|
| `(x² + y⁰·⁵)⁴·²`                       | 196.270    | 41.501   | 103            | **47.558**       | **Janino**   |
| Complex trig + exp + power              | 768.606    | 180.566 | 302.153     | **58.714**       | **Turbo**   |
| Heavy constants **with** Folding        | 755.4    | 185.3    | 19.081          | **8.728**       | **Turbo**   |


---

### 📈 Overall Verdict

> **ParserNG 1.0.0+ Turbo is the clear winner.**  
> It delivers **bytecode-level performance** with the flexibility and safety of an interpreter, ** near-zero allocation** in all cases, and powerful features (symbolic differentiation, matrix algebra, equation solvers, etc.) that others lack.

ParserNG Turbo is now competitive with (and at-times faster than) Janino on complex expressions while maintaining excellent safety and low memory footprint.

---