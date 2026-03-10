# ParserNG
 
 
ParserNG 0.2.5 has been released on maven-central

1. Functions like `intg`, `root`, `t_root` and `quadratic` have been fixed and are working well. .

2. Frame based args passing is used to milk to milk the last drops of performance during iterations.

3. Constant folding and strength reduction make the evaluation process feel much faster.

4. The `print` function that can be used to view the contents of an `EvalResult` which the `solveGeneric()` method returns is also functional.

5. If you need a rich, fully featured parser that can do 3 million to 10 million evaluations per second, ParserNG v0.2.5 is the one for you.

ParserNG 0.2.4 has been released on maven-central!

ParserNG 0.2.4 drives the limits of expression interpretation velocity even further than all previous versions, beating many lighterweight and fast Java math parsers(interpreted) in many benchmarks.
Check [ParserNG-Wars](https://github.com/gbenroscience/ParserNG) for some shootouts between ParserNG and other parsers, both handrafted benchmarks and JMH based ones

ParserNG evaluates expressions at almost the speed at which the expressions would run if they were compiled statements in Java code. Typical values for moderate expression evaluation speeds are between `85ns`(algebraic expressions e.g.`((12+5)*3 - (45/9))^2` to `176ns`(methods with trig. functions, e.g. `(sin(3) + cos(4 - sin(2))) ^ (-2))`.


Applications that need 5 million to 10 million points generated per second would benefit from `ParserNG v0.2.4`
 
0.2.3
**ParserNG 0.2.3** has been released on maven central!<br>
comes with couple microsecond (on decent hardware and) expression solving ability while maintaining its full feature stack. Graphing feels butter-smooth and iterations shouldn't feel so iterative. 

At 5 microsecond, moderately complex expressions such as
```Java
String s6 = "5*sin(3+2)/(4*3-2)";
```
 can be evaluated almost 200 thousand times per second.

 The model of a `Matrix` has also been optimized to use a 1D array internally. This makes it faster due to memory locality of Matrix data.

 Also, we support eigenvalues and eigenvectors as inbuilt methods, so enjoy! 

ParserNG 0.2.2 has been released on maven-central!**


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