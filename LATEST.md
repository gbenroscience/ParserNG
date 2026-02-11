# ParserNG


Back with a bang!
ParserNG 0.2.2 has been released on maven-central!


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