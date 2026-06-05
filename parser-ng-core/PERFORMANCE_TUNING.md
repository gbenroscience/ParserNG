This section is designed to help you squeeze every last nanosecond out of **ParserNG 1.0.3+**. Because the engine utilizes a JIT-native architecture via `MethodHandle` trees, its performance characteristics differ significantly from traditional interpreted parsers.

---

## 🚀 Performance Tuning Guide

### 1. Choosing the Right Mode
ParserNG offers two primary execution paths. Choosing the right one depends on your specific use case:

| Mode | Best For | Technical Profile |
| :--- | :--- | :--- |
| **Standard** | One-off evaluations, dynamic formulas, low-memory environments. | High-speed interpreted postfix traversal. **Zero Allocation.** |
| **Turbo** | High-frequency loops, real-time streaming, fintech, physics simulations. | Compiled `MethodHandle` tree. **Zero Allocation + JIT Inlining.** |

**Recommendation:** If you are evaluating the same expression more than 1,000 times, always use **Turbo Mode**.

---

### 2. The Power of Constant Folding
Version 1.0.3+ has aggressive **Constant Folding**. This optimization happens during the compilation phase, where the parser identifies sub-expressions that result in a constant value and "pre-calculates" them.

* **Static Expression:** `sin(3.14159 / 2) + x`
* **Folded Expression:** `1.0 + x`

By folding constants, you eliminate unnecessary mathematical calls (like `Math.sin`) from the runtime execution path.



---

### 3. JVM Warm-up (The "JIT" Factor)
Because **Turbo Mode** builds a `MethodHandle` tree, the JVM's HotSpot compiler needs a small "warm-up" period to identify the expression as a "hot path" and inline the code.

* **Cold Start:** ~500–1,000 ns per op.
* **Warmed Up:** ~80–90 ns per op.

**Tip:** In production environments, run a few thousand "dummy" evaluations during application startup to ensure the JVM has fully optimized the execution tree before the first real request arrives.



---

### 4. Avoiding Boxing Penalties
To maintain **0 B/op** (Garbage-Free) performance, always prefer primitive signatures. 

When using `FastCompositeExpression`, use the `applyScalar` method instead of the generic `apply` method. The generic `apply` method returns an `EvalResult` object, which—while convenient—triggers a small allocation. `applyScalar` stays entirely within the primitive `double` domain.

```java
// ❌ Slower (Allocates EvalResult)
MathExpression.EvalResult result = fastExpr.apply(variables);

// ✅ Faster (Zero Allocation, Direct Primitive)
double result = fastExpr.applyScalar(variables);
```

---

### 5. Multi-Variable Optimization
When working with multiple variables ($x, y, z$), ensure your variable array matches the order defined in the expression to avoid index-lookup overhead. ParserNG is optimized to read directly from the `double[]` data frame provided to the execution bridge.

```java
// Pre-allocate your data frame to avoid array creation in the loop
double[] vars = new double[2]; 

for (int i = 0; i < 1_000_000; i++) {
    vars[0] = i; // x
    vars[1] = Math.sqrt(i); // y
    double val = fastExpr.applyScalar(vars);
}
```

---

### 6. JDK Version Matters
ParserNG 1.0.3+ is optimized for **modern JDKs (17, 21, and 24)**. Improvements in the `java.lang.invoke` package in later versions directly translate to faster "Turbo" execution. If you are running on JDK 8 or 11, you may see slightly higher latencies due to less efficient `MethodHandle` inlining.

---

