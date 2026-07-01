# ParserNG


### ParserNG 2.0.6 is out on maven-central! 
Very stable release. Lots of bug fixes in bulk gelu, swiglu etc

### ParserNG 2.0.5 is out on maven-central! 
More stable. Cleaned up API.


### ParserNG 2.0.3 is out on maven-central! 
More stable. Indexing bugs fixed in bulk processors

### ParserNG 2.0.0 is out on maven-central! 
Guess who the kid on the block is? Vector API bulk evaluator(SIMDVectorTurboEvaluator)and its compatibility partner, VectorTurboEvaluator.
Both run bulk evaluations at roughly same speed(faster than Janino), and come with workers out of the box!
`VectorTurboEvaluator` works because its code is mechanically sympathetic to the hardware running it,so auto-vectorization occurs.



### Parser 1.2.1 has been released on maven-central!
Version 1.2.1 fixes bugs in Matrix Algebra in the standard mode.


### Parser 1.2.0 has been released on maven-central!
Version 1.2.0 introduces pure Matrix algebra into ParserNG standard. Its great performance is not at par with what
ParserNG Turbo can do( with MatrixTurboEvaluator), in terms of memory allocation optimization and matrix evaluation speed, but it ensures that
Matrix Algebra is fully available alongside matrix functionality on ParserNG Standard also.

### Parser 1.1.5 has been released on maven-central!
#### ParserNG Turbo: Zero-Allocation Optimization Pass
We have completely refactored the runtime variable mapping layer inside `ScalarTurboEvaluator1` and `ScalarTurboEvaluator2`. 

* **Runtime Remapping Eliminated:** Variable array positions are now baked directly into the `MethodHandle` topology at compile-time. 
The runtime engine now evaluates expressions by reading straight from the user's input arrays.
* **30%+ Evaluation Speed Burst:** Microbenchmarks show arithmetic evaluation speeds dropping from ~18ns down to **~12.2ns**, pulling within arm's reach of raw native Java performance (~6.4ns).
* **Flat Memory Profile:** GC allocation churn on hot evaluation paths remains at **absolute zero**. 
This guarantees stutter-free performance during heavy graph plotting or the soon-coming multi-million step differential equation loops.


### Parser 1.1.4 has been released on maven-central!
Version 1.1.4 squashes a bug where an over active syntax checker disables nested stats functions e.g. sort(3,1,5,listsum(4,12,18,-9),5,2,31,4) returns a syntax error Added tests


### Parser 1.1.3 has been released on maven-central!
Fixed validation bugs in parser, made relevant matrix maethods support algebraic operations, like A*invert(B) etc, fixed bad bugs in flat matrix turbo implementations and optimized them further. Added tests

### Parser 1.1.2 has been released on maven-central!
1. Fixes bugs and makes `MatrixTurboEvaluator` natively support turbo execution of the rot function. Note that the `ScalarTurboEvaluator`s already support it.
2. Also `FastCompositeExpression` is now aware of its compiler as it now sports a `getCompiler` default method(which can be overriden to specify the turbo class that compiled it) 

### Parser 1.1.1 has been released on maven-central!
Implemented version retrieval for ParserNG

### Parser 1.1.0 has been released on maven-central!
Bug fixes in Rotor and ErrorLog. Matrix of Points upgrade for Rotor

### Parser 1.0.8 has been released on maven-central!
Bug fixes and more Android compatibility issues resolved.

### Parser 1.0.6 has been released on maven-central!
Bug fixes and Android compatibility issues resolved.


### Parser 1.0.5 has been released on maven-central!
Features bug fixes and optimizations in the scanning/semantic analysis stages.

### Parser 1.0.4 has been released on maven-central!
This version features various optimizations and turbo capability for the Function
class.

Parser 1.0.3 has been released on maven-central!
Maintaining the industry standard besting speeds of v1.0.x, it adds the functionality of rotational geometry.
In v1.0.3, you can use the rotor function, `rot` to rotate raw points in 3D space and other functions such as curves, lines, surfaces(both plane and curved) and 3D equations of all sorts.

### ParserNG 1.0.2 has been released on maven-central!
Version 1.0.2 retains the wild speeds of Version 1.0.1. Adds an extra widening technique of variable passing to the Turbo mode,
In addition to the current method of array based passing. The widening technique can be sometimes faster than the array based methods,
but their speed profiles and memory profiles are similar. Its weakness though is that it cannot use more than 63 variables per expression, whereas the array based approach allows in theory any number up to the max integer size.


### ParserNG 1.0.1 has been released on maven-central!
This close update ensures that Turbo mode's  memory profile stays close to that of the normal mode, which is, nigh zero.


### ParserNG 1.0.0 has been released on maven-central!
The library has finally come of age with the introduction of its Turbo mode, which offers a massive speed boost over its normal mode.
The nomal mode already beats famous libraries like exp4J, and rivals Janino, the widely acclaimed Gold Standard of Java math parser speed measurements, very closely

 
### ParserNG 0.2.5 has been released on maven-central

1. Functions like `intg`, `root`, `t_root` and `quadratic` have been fixed and are working well. .

2. Frame based args passing is used to milk to milk the last drops of performance during iterations.

3. Constant folding and strength reduction make the evaluation process feel much faster.

4. The `print` function that can be used to view the contents of an `EvalResult` which the `solveGeneric()` method returns is also functional.

5. If you need a rich, fully featured parser that can do 3 million to 10 million evaluations per second, ParserNG v0.2.5 is the one for you.

### ParserNG 0.2.4 has been released on maven-central!

### ParserNG 0.2.4 drives the limits of expression interpretation velocity even further than all previous versions, beating many lighterweight and fast Java math parsers(interpreted) in many benchmarks.
Check [ParserNG-Wars](https://github.com/gbenroscience/ParserNG) for some shootouts between ParserNG and other parsers, both handrafted benchmarks and JMH based ones

ParserNG evaluates expressions at almost the speed at which the expressions would run if they were compiled statements in Java code. Typical values for moderate expression evaluation speeds are between `85ns`(algebraic expressions e.g.`((12+5)*3 - (45/9))^2` to `176ns`(methods with trig. functions, e.g. `(sin(3) + cos(4 - sin(2))) ^ (-2))`.


Applications that need 5 million to 10 million points generated per second would benefit from `ParserNG v0.2.4`
 

### **ParserNG 0.2.3** has been released on maven central!<br>
comes with couple microsecond (on decent hardware and) expression solving ability while maintaining its full feature stack. Graphing feels butter-smooth and iterations shouldn't feel so iterative. 

At 5 microsecond, moderately complex expressions such as
```Java
String s6 = "5*sin(3+2)/(4*3-2)";
```
 can be evaluated almost 200 thousand times per second.

 The model of a `Matrix` has also been optimized to use a 1D array internally. This makes it faster due to memory locality of Matrix data.

 Also, we support eigenvalues and eigenvectors as inbuilt methods, so enjoy! 

### ParserNG 0.2.2 has been released on maven-central!**


## What's new?

1. Breaking change!!! Package name change. **ParserNG** now has a proper root package name, com.github.gbenroscience
2. Really cool!!! -> Speed upgrades of up to 10x to 40x in the base algebraic expression parser. This means more speedy and energy efficient iterative calculations and graphing!
3. Very nice to have -> **ParserNG** now comes with a Java platform agnostic graphing capability. All the developer needs to do is to implement 2 interfaces: 
```java
com.github.gbenroscience.math.graph.DrawingContext
``` 
```java
com.github.gbenroscience.math.graph.AbstractView
``` 

and pass the instance to  
```java
com.github.gbenroscience.math.graph.Grid
```

So you can use the same codebase to plot graphs on Android, JavaFX, Swing and other Java platforms. We have made the required implementations of DrawingContext(so called adapters) for the most popular Java platforms(Android, Swing and JavaFX) available in [GRAPHING.md](./GRAPHING.md)
If you have more platforms in mind, we will be excited to have you contribute the code and make **ParserNG** even more versatile.
Enjoy!

Fun fact: here are some interactions with ParserNG hosted on maven-central within Dec 2025 and Feb 11 2026:
We had 970 total downloads from 193 unique sources across 82 companies
[See here](./maven-central-3-month-data.png)
