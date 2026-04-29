# ParserNG 🧮⚡

**ParserNG 1.1.1** is a **blazing-fast**, nigh zero allocation(memory wise), **pure Java**, **zero-native-dependencies** math expression parser and evaluator.

It **beats exp4J, and com.expression.parser on evaluation speed** across every kind of expression — from simple algebra to heavy trig, matrices, and calculus; and manages to beat Janino, the gold standard on some, while rivalling it on a host of other expressions  
The normal mode routinely does about **3-10 million evaluations per second** while the new Turbo mode easily peaks at about **10 million to 90 million evaluations per second**.

It goes far beyond basic parsing — offering **symbolic differentiation**, **resilient numerical integration**, **full matrix algebra**, **statistics**, **equation solving**, **user-defined functions**, **2D graphing**, and more — all in one lightweight, cross-platform library.

Perfect for scientific computing, simulations, real-time systems, education tools, Android apps, financial models, and high-performance scripting.

[![Maven Central](https://img.shields.io/maven-central/v/com.github.gbenroscience/parser-ng.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng)
[![License](https://img.shields.io/github/license/gbenroscience/ParserNG?color=blue)](https://github.com/gbenroscience/ParserNG/blob/master/LICENSE)
![Java](https://img.shields.io/badge/Java-8%2B-orange)
![Latest Version](https://img.shields.io/badge/version-1.1.1-success)

> **1.1.1** introduces **Turbo Scalar** and **Turbo Matrix** compiled paths + massive speed improvements via strength reduction, constant folding, and O(1) frame-based argument passing.

## ✨ Highlights (v1.1.1)

- **Speed champion** — rivals Janino in most benchmarks, and beats exp4J, com.expression.parser and Parsii in every benchmark (see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md))
- **Turbo Mode** — compile once, evaluate millions of times per second (Scalar + Matrix paths)
- Symbolic differentiation (`diff`) + resilient numerical integration (`intg`) that handles difficult expressions
- Full matrix algebra (`det`, `eigvalues`, `eigvec`, `adjoint`, `cofactor`, matrix division, `linear_sys`, …)
- Statistics (`avg`, `variance`, `cov`, `min`, `max`, `rms`, `listsum`, `sort`, …)
- Equation solvers: quadratic, **Tartaglia cubic**, numerical roots, linear systems
- User-defined functions (`f(x)=…` or lambda `@(x,y)…`) + persistent `FunctionManager` / `VariableManager`
- Variables with execution frames for ultra-fast loops
- 2D function & geometric plotting support
- Logical expressions (`and`, `or`, `==`, …)
- No external dependencies — runs on Java SE, Android, JavaME, …

## 🚀 Installation (Maven)

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng</artifactId>
    <version>1.1.1</version>
</dependency>
```

Also available on **Maven Central**:  
https://central.sonatype.com/artifact/com.github.gbenroscience/parser-ng/1.1.1

## 🧮 Standard Mode — The old way

```java
import com.github.gbenroscience.parser.MathExpression;

MathExpression me = new MathExpression("x=2;sin(3*x+2)+sec(5*x/4)");
double res = me.solveGeneric().scalar;
```


## ⚡ Turbo Mode — The Game Changer

```java
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
```

### Turbo Scalar (tight loops / millions per second)

```java
String expr = "x*sin(x) + y*cos(y) + z^2";
MathExpression me = new MathExpression(expr, false);      // prepare for turbo
FastCompositeExpression turbo = me.compileTurbo();        // compile once

int xSlot = me.getVariable("x").getFrameIndex();
int ySlot = me.getVariable("y").getFrameIndex();
int zSlot = me.getVariable("z").getFrameIndex();

double[] frame = new double[3];

for (double t = 0; t < 10_000_000; t += 0.001) {
    frame[xSlot] = t;
    frame[ySlot] = t * 1.5;
    frame[zSlot] = t / 2.0;
    double result = turbo.applyScalar(frame);   // ← ultra-fast!
}
```

### Turbo Matrix (eigvalues, linear systems, etc.)

```java
MathExpression me = new MathExpression("eigvalues(R)");
FastCompositeExpression turbo = TurboEvaluatorFactory.getCompiler(me).compile();

Matrix result = turbo.applyMatrix(new double[0]);   // works for: linear_sys, adjoint, cofactor, A/B, etc.
```

## Quick Start (Normal Mode + Turbo)

### 1. Simple expression

```java
MathExpression expr = new MathExpression("r = 5; 2 * pi * r");
System.out.println(expr.solve());   // ≈ 31.4159
```

### 2. High-performance loop (Turbo recommended)

```java
MathExpression expr = new MathExpression("x^2 + 5*x + sin(x)", false);
FastCompositeExpression turbo = expr.compileTurbo();
int xSlot = expr.getVariable("x").getFrameIndex();
double[] frame = new double[1];

for (double x = 0; x < 100_000; x += 0.1) {
    frame[xSlot] = x;
    double y = turbo.applyScalar(frame);
    // plot or process y
}
```

### 3. Symbolic derivative

```java
MathExpression expr = new MathExpression("f(x) = x^3 * ln(x); diff(f, 2, 1)");
System.out.println(expr.solveGeneric().scalar);
```

### 4. Numerical integration (even difficult ones work in Turbo)

```java
MathExpression expr = new MathExpression("intg(@(x) 1/(x*sin(x)+3*x*cos(x)), 0.5, 1.8)");
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


### 6. ROTOR
You may use the rot function to rotate functions, surfaces(plane or curved), lines and even raw points in 3D space.

To rotate any of these, you need the orbital center, the coordinates of the direction vector(a,b,c) and the angle of rotation.

The example below shows two ways to use the ParserNG library to rotate the point `p` and `q` about the orbital center (1,0,1)
with the directio vector,(1,1,0). The angle of rotation is pi radians.

```Java
 String expression = "p=@(1,3)(4,2,5);q=@(1,3)(12,3,-1);rot(p,q, pi, @(1,3)(1,0,1),@(1,3)(1,1,0))";
        MathExpression interpreted = new MathExpression(expression);

        MathExpression.EvalResult ev = interpreted.solveGeneric(); 
        System.out.printf("Expression: %s%n", expression);
        System.out.println("interpreted: " + ev);

        // Compile to turbo
        FastCompositeExpression compiled = new ScalarTurboEvaluator(interpreted, true).compile();
        // Warm up turbo JIT
        double[] vars = new double[0]; 
        MathExpression.EvalResult evr = compiled.apply(vars);
        System.out.println("turbo: " + evr);
```


 
## ⌨️ Command-line tool (REPL)

```bash
java -jar parser-ng-1.1.1.jar "sin(x) + cos(x)"
java -jar parser-ng-1.1.1.jar "eigvalues(R=@(5,5)(...))"
java -jar parser-ng-1.1.1.jar help
java -jar parser-ng-1.1.1.jar -i          # interactive mode
```

## 📊 Supported Features at a Glance

| Category          | Key Functions                                      | Turbo Support |
|-------------------|----------------------------------------------------|---------------|
| Arithmetic & Logic| `+ - * / ^ % and or == !=`                         | Full          |
| Trigonometry      | `sin cos tan asin … sinh`                          | Full          |
| Calculus          | `diff` (symbolic), `intg` (resilient)              | Yes           |
| Equations         | Quadratic, Tartaglia cubic, `root`, `linear_sys`   | Yes           |
| Matrices          | `det`, `eigvalues`, `eigvec`, `adjoint`, `cofactor`, `A / B` | Excellent (Turbo Matrix) |
| Statistics        | `avg variance cov min max rms listsum sort`        | Yes           |
| Custom            | `@(x,y)…` or named functions                       | Full          |

Full list: run `help` or `new MathExpression("help").solve()`.

## 📚 More Documentation

- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) — full speed comparisons
- [GRAPHING.md](GRAPHING.md) — plotting on Swing / JavaFX / Android
- [LATEST.md](LATEST.md) — what’s new in 1.1.1
- Javadoc: https://javadoc.io/doc/com.github.gbenroscience/parser-ng/1.1.1
- [Hello world and original readme](src/main/java/com/github/gbenroscience/README.md) — Original readme for pre-1.0 versions with a lot of, still valid, examples

## ❤️ Support the Project

ParserNG is built with love in my free time. If it helps you:

- ⭐ Star the repository  
- 🐞 Report bugs or suggest features  
- 💡 Share what you built with it  
- ☕ [Buy me a coffee](https://buymeacoffee.com/gbenroscience)

## 📄 License

**Apache License 2.0**

---

**ParserNG 1.1.1** — faster than the competition, stronger on matrices, and now with real Turbo Scalar + Turbo Matrix compiled power.

Happy parsing! 🚀  
— **GBENRO JIBOYE** (@gbenroscience)