# ParserNG 🧮⚡

**ParserNG** is a **fast**, **pure Java**, **no-native-dependencies** math expression parser and evaluator.

If you need a rich, fully featured parser that can do 3 million to 10 million evaluations per second, ParserNG v0.2.5 is the one for you.

It goes far beyond basic expression parsing — offering **symbolic differentiation**, **numerical integration**, **matrix operations**, **statistics**, **equation solving**, **user-defined functions**, **2D graphing**, and more — all in a lightweight, cross-platform library.

Perfect for scientific computing, education, engineering tools, Android apps, financial models, scripting engines, and anywhere you need powerful math without heavy dependencies.

[![Maven Central](https://img.shields.io/maven-central/v/com.github.gbenroscience/parser-ng.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![License](https://img.shields.io/github/license/gbenroscience/ParserNG?color=blue)](https://github.com/gbenroscience/ParserNG/blob/master/LICENSE)
![Java](https://img.shields.io/badge/Java-8%2B-orange)
![Latest Version](https://img.shields.io/badge/version-0.2.5-success)

> v0.2.5 brings **strength reduction**, **constant folding**, and **O(1) argument passing** via execution frames → significantly faster evaluation.

## ✨ Highlights

- Extremely fast evaluation (one of the fastest pure-Java parsers)
- Symbolic differentiation (`diff`) with high accuracy
- Numerical integration (`intg`)
- Built-in matrix operations (`det`, `eigvalues`, `matrix_mul`, …)
- Statistics (`avg`, `variance`, `cov`, `min`, `max`, …)
- Equation solvers: quadratic, cubic (Tartaglia), numerical root finding
- User-defined functions (`f(x) = …` or lambda `@(x) …`)
- Variables & persistent scope (`VariableManager`, `FunctionManager`)
- 2D function & geometric plotting support
- Logical expression parsing (`and`, `or`, `==`, …)
- No external dependencies — runs on Java SE, Android, JavaME, …

## 🚀 Installation (Maven)


```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>0.2.5</version>
</dependency>
```

Also available on **Maven Central**:  
https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng/0.2.5

## Quick Start

### 1. Simple expression (as library)

```java
import net.sourceforge.parserng.MathExpression;

public class QuickStart {
    public static void main(String[] args) {
        MathExpression expr = new MathExpression("r = 4; 2 * pi * r");
        double result = expr.solve();
        System.out.printf("Circumference ≈ %.4f%n", result);   // ≈ 25.1327
    }
}
```

### 2. Reuse for high-performance loops

```java
MathExpression expr = new MathExpression("quadratic(@(x)x^2 + 5*x + sin(x))");
int xSlot = expr.getVariable('x').getFrameIndex();
for (double x = 0; x < 100000; x += 0.1) {
    expr.updateSlot(xSlot, x);
    double y = expr.solveGeneric().scalar;   // very fast — parsing done only once
    // ... plot or process y
}
```

### 3. Symbolic derivative & evaluation

```java
MathExpression expr = new MathExpression("f(x) = x^3 * ln(x); diff(f, 2, 1)");
System.out.println(expr.solveGeneric().scalar);   // derivative of f at x=2
```

### 4. Numerical integration

```java
MathExpression expr = new MathExpression("intg(@(x) sin(x^2), 0, 1.5)");
System.out.println("∫ ≈ " + expr.solve());
```

### 5. Matrix example

```java
MathExpression expr = new MathExpression("""
    M = @(3,3)(1,2,3, 4,5,6, 7,8,9);
    det(M)
""");
System.out.println("Determinant = " + expr.solve());
```

## ⌨️ Command-line tool (REPL-like)

```bash
# Simple calculation
java -jar parser-ng-0.2.5.jar "2+3*4^2"

# With variables & functions
java -jar parser-ng-0.2.5.jar "f(x)=x^2+1; f(5)"

# Symbolic derivative
java -jar parser-ng-0.2.5.jar "diff(@(x)x^3 + 2x, x)"

# Show all built-in functions
java -jar parser-ng-0.2.5.jar help

# Interactive mode
java -jar parser-ng-0.2.5.jar -i
```

## 📊 Supported Features at a Glance

| Category              | Examples                                      | Notes                              |
|-----------------------|-----------------------------------------------|------------------------------------|
| Arithmetic            | `+ - * / ^ %`                                 | Standard precedence                |
| Trigonometry          | `sin cos tan asin acos atan sinh …`           | `RAD`/`DEG`/`GRAD` modes           |
| Statistics            | `sum avg variance cov min max rms …`          | Vector-aware                       |
| Combinatorics         | `fact comb perm`                              |                                    |
| Calculus              | `diff` (symbolic), `intg` (numerical), `root` | Symbolic diff is exact where possible |
| Equations             | Quadratic, cubic (Tartaglia), `linear_sys`    | Numerical root finding too         |
| Matrices              | `det matrix_mul transpose eigvalues eigvec …` | Full linear algebra basics         |
| Logic                 | `and or xor == != > < <= >= !`                | Use `-l` flag or logical mode      |
| Plotting              | `plot` (see GRAPHING.md)                      | 2D function & geometric plots      |
| Custom functions      | `f(x)=…` or `@(x,y)…`                         | Lambda & named syntax              |

Full function list: run `help` in the parser or call `new MathExpression("help").solve()`.

## 📚 More Documentation

- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) — performance comparisons
- [GRAPHING.md](GRAPHING.md) — how to use the plotting features
- [API Javadoc](https://javadoc.io/doc/com.github.gbenroscience/parser-ng/latest/index.html) (hosted on javadoc.io)

## ❤️ Support the Project

ParserNG is maintained in free time. If you find it useful:

- ⭐ Star the repository
- 🐞 Report bugs or suggest features
- 💡 Share how you use it
- ☕ [Buy me a coffee](https://buymeacoffee.com/gbenroscience)

## 📄 License

**Apache License 2.0**

Happy parsing! 🚀  
— GBENRO JIBOYE (@gbenroscience)

