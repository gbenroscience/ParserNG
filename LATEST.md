# ParserNG

 
ParserNG 0.2.4 has been released on maven-central!

ParserNG 0.2.4 drives the limits of expression interpretation velocity even further than all previous versions, beating much lighterweight and fast Java math parsers in many benchmarks.
Check [ParserNG-Wars](https://github.com/gbenroscience/tinymath) for some shootouts between ParserNG and other parsers, both handrafted benchmarks and JMH based ones

ParserNG evaluates expressions at almost the speed at which the expressions would run if they were compiled statements in Java code. Typical values for moderate expression evaluation speeds are between `85ns`(algebraic expressions e.g.`((12+5)*3 - (45/9))^2` to `176ns`(methods with trig. functions, e.g. `(sin(3) + cos(4 - sin(2))) ^ (-2))`.


Applications that need 5 million to 10 million points generated per second would benefit from `ParserNG v0.2.4`


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
If you have more platforms in mind, we will be excited to have you contribute the code and make the **ParserNG** even more versatile.
Enjoy!

Fun fact: here are some interactions with ParserNG hosted on maven-central within Dec 2025 and Feb 11 2026:
We had 533 total downloads from 122 unique sources across 51 companies
[See here](./maven-central-3-month-data.png)