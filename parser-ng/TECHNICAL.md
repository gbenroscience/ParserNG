# **ParserNG: Performance & Memory Manifesto**

ParserNG is a high-performance mathematical expression engine for Java, designed for systems where **nanoseconds matter** and **Garbage Collection (GC) is unacceptable.**


### **ParserNG vs. The Giants: Master Performance Matrix**

| Battleground | Expression Type | Janino (ns/op) | **ParserNG (Turbo)** | **ParserNG Advantage** |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Arithmetic** | `x + y + z...` | 251.08 | **119.95** | **2.1x Faster** |
| **Structural Scale** | `500+ Variables` | **CRASH** | **SUCCESS** | **Only Survivor** |
| **Functional Heavy** | `20+ sin() calls` | 471.43 | **362.89** | **1.3x Faster** |

---


## **1. Performance: Beyond the "Gold Standard"**
For years, Janino has been considered the speed leader because it compiles expressions to bytecode. ParserNG’s **Turbo** engine shatters this benchmark.

In head-to-head testing using a complex trigonometric and power-based expression, ParserNG consistently beats both Janino and Parsii.

### **Benchmark Results (Complex Expression)**
*Expression: `(sin(x) + 2 + ((7-5) * (3.14159 * x^(14-10)) + sin(-3.141) + (0%x)) * x/3 * 3/sqrt(x+12))`*

| Library | Speed (ns/op) | Performance Gap |
| :--- | :--- | :--- |
| **ParserNG Turbo (Widening)** | **119.95 ns** | **1.0x (Winner)** |
| **ParserNG Turbo (Array)** | **133.26 ns** | 1.1x Slower |
| **Janino** | 251.08 ns | **2.1x Slower** |
| **Parsii** | 370.29 ns | 3.1x Slower |

**The Verdict:** ParserNG is **2x faster than Janino** and **3x faster than Parsii**, all while operating under the same JVM constraints.

---

## **2. Memory Profile: Zero Allocation, Zero Disturbance**
High-performance Java isn't just about raw speed; it's about **predictability**. Most parsers create object "noise" that forces the Garbage Collector to pause your application.

### **The "Zero-Allocation" Guarantee**
ParserNG achieves a steady-state execution with **zero object churn**. 
* **GC Allocation Rate:** **0.007 MB/sec** (effectively zero).
* **GC Count:** **0** during the entire measurement period.

### **Metaspace Safety**
Unlike Janino, which creates a new Java class for every expression (risking **Metaspace OutOfMemoryErrors**), ParserNG uses a reusable, high-density architecture. You can compile millions of unique expressions without bloating the JVM's permanent memory or requiring a restart.

---

## **3. Architecture: Hybrid Variable Passing**
ParserNG intelligently optimizes how data reaches the CPU.

### **Widening Strategy (The "Register" Path)**
For expressions within the JVM’s method slot limits (up to 63 variables), ParserNG uses **Widening**. This passes values directly through registers and the stack, achieving the absolute lowest possible latency.
* **Winner for:** Standard formulas, physics engines, and real-time signals.

### **Array-Based Strategy (The "Unlimited" Path)**
When expressions grow into hundreds or thousands of variables, ParserNG seamlessly switches to an **Array-Based** approach. This bypasses the JVM’s 255-slot limit, allowing for massive high-dimensional data processing that would crash a standard compiler.

---

## **4. Scalability: Linear Performance**
ParserNG scales predictably. As you increase variable counts, the "tax" per variable remains under **1 nanosecond** per op, allowing you to build complex models without hitting a performance wall.

| Variables | Array-Based | Widening |
| :--- | :--- | :--- |
| **1 Var** | 5.65 ns | 6.17 ns |
| **20 Vars** | 15.11 ns | 15.34 ns |
| **63 Vars** | 52.07 ns | **51.60 ns** |

---

## **Summary**
If you are building high-frequency trading platforms, real-time simulation software, or cloud-native microservices with tight memory budgets, **ParserNG** provides the deterministic speed you need without the Garbage Collection or Metaspace risks of traditional tools.

---
 