# ParserNG
ParserNG is a powerful open-source math tool that parses and evaluates algebraic expressions. 

## NOTE:

<i>If you need to use the parser directly in your Android project, go to:
[parserng-android](https://github.com/gbenroscience/parserng-android) by the same author
</i><br>

If you need to access this library via Maven Central, do:
      
```Java
        <dependency>
            <groupId>com.github.gbenroscience</groupId>
            <artifactId>parser-ng</artifactId>
            <version>0.1.0</version>
        </dependency>
 ```      

This library was created in 2009 and later used by the author as part of a critical part of his University final year project
at the Department of Computer Science and Engineering, Obafemi Awolowo University,Ile-Ife, Osun State, Nigeria.

The design goal of this library was to create a simple, yet powerful, not too bogus math tool that scientists and developers could deploy with their
work to solve mathematical problems both simple and complex.

ParserNG is written completely in (pure) Java and so is as cross-platform as Java can be. It has been used to design math platforms for desktop Java, Java MicroEdition devices(as far back as 2010-2011) , Android,  and by porting the whole platform using J2OBJC from Gooogle; Swift also. The performance has been exceptionally acceptable in all cases.


<p><b>FEATURES</b></p>
<ol>
<li>Arithmetic operations.</li>
<li>Statistical operations</li>
<li>Trigonometric operations</li>
<li>Permutations and Combinations</li>
<li>Basic matrix operations</li>
<li>Differential Calculus(Exact numerical accuracy achieved using symbolic differentiation)</li>
<li>Integral Calculus(Numerical)</li>
<li>Quadratic Equations</li>
<li>Tartaglia's Equations( or generally: <code>a.x<sup>3</sup>+b.x+c = 0</code> )</li>
<li>Numerical (Iterative) solution for roots of equations</li>
<li>Simultaneous Linear Equations</li>
<li>Amongst others</li>
<li>Variables creation and usage in math expressions</li>
<li>Function creation and usage in math expressions</li>
</ol>


<p><b>Using ParserNG</b></p>


The simplest way to evaluate an expression in ParserNG is to use the <code>MathExpression</code> class.
<code>MathExpression</code> is the class responsible for basic expression parsing and evaluation.

Do:<br>
<code>MathExpression expr = new MathExpression("r=4;r*5");</code>
<br>
<code>System.out.println("result: " + expr.solve());</code>

<span>What does this do?</span>

It creates a variable called <code>r</code>and sets its value to 4. Then it goes ahead to evaluate the expression
<code>r*5</code> and returns its value when expr.solve() is called. <br>The print statement would give
<br><br><code>solution: 20.0</code><br><br>
at the console.


Some key applications of parsers involve repeated iterations of a given expression at different values of the variables involved. Iteratively determining the roots of an equation, graphing etc.

For repeated iterations of an expression over a value range, say 'x^2+5*x+1', the wrong usage would be:<br>

<pre><code>
for(int i=0;i<10000;i++){

double x = i;
MathExpression expression = new MathExpression("x="+i+";x^2+5*x+1");<br>

expression.solve();<br>

}
</code></pre>
<br>

The MathExpression constructor basically does all the operations of scanning and interpreting of the input expression. This is a very expensive operation. It is better to do it just once and then run the solve() method over and over again at various values of the variables.
  

For example:

<pre><code>
MathExpression expression = new MathExpression("x^2+5*x+1");

for(int i=0; i<100000; i++){
expression.setValue("x", String.valueOf(i) );
expression.solve();
}
</code></pre>
<br>
This ensures that the expression is parsed once(expensive operation) and then evaluated at various values of the variables. This second step is an high speed one, sometimes taking barely 3 microseconds on some machines.<br><br>


<b>Inbuilt Functions</b><br>
The parser has its own set of built-in functions. They are:
<code>
sin,cos,tan,sinh,cosh,tanh,sin-¹,cos-¹,tan-¹,sinh-¹,cosh-¹,tanh-¹,sec,csc,cot,sech,csch,coth,sec-¹,csc-¹,cot-¹,sech-¹,csch-¹,coth-¹,exp,ln,lg,log,ln-¹,lg-¹,log-¹,asin,acos,atan,asinh,acosh,atanh,asec,acsc,acot,asech,acsch,acoth,aln,alg,alog,floor,ceil,sqrt,cbrt,inverse,square,cube,pow,fact,comb,perm,sum,prod,avg,med,mode,rng,mrng,rms,cov,min,max,s_d,variance,st_err,rnd,sort,plot,diff,intg,quad,t_root,root,linear_sys,det,invert,tri_mat,echelon,matrix_mul,matrix_div,matrix_add,matrix_sub,matrix_pow,transpose,matrix_edit,
</code>

Note that alternatives to many functions having the inverse operator are provided in the form of an 'a' prefix.
For example the inverse <code>sin</code> function is available both as <code>sin-¹</code> and as <code>asin</code>

<b>User defined functions</b><br>
You can also define your own functions and use them in your math expressions.
This is done in one of 2 ways:
<ol>
  <li>f(x,a,b,c,...)= expr_in_said_variables<br> For example: f(x,y)=3*x^2+4*x*y+8</li>
  <li>f = @(x,a,b,c,...)expr_in_said_variables<br> For example: f= @(x,y)3*x^2+4*x*y+8</li>  
</ol>

Your defined functions are volatile and will be forgotten once the current parser session is over. The only way to have the parser remember them always is to introduce some form of persistence.

So for instance, you could pass the following to a MathExpression constructor:

f(x)=sin(x)+cos(x-1)<br>
Then do: f(2)....the parser automatically calculates sin(2)+cos(2-1) behind the scenes.<br><br>

<b>Differential Calculus</b><br>

<b>ParserNG</b> makes differentiating Math Expressions really easy.

<p>
ParserNG uses its very own implementation of a symbolic differentiator.
  
  It performs symbolic differentiation of expressions behind the scenes and then computes the differential coefficient
  of the function at some supplied x-value.
  
  <b>To differentiate a function, do:</b>
  
  <pre><code>
  MathExpression expr = new MathExpression("diff(@(x)x^3,3,1)"); 
  
  System.out.println(ex.solve());
  </code></pre>
  
  This will print:
  
  <code>
  27.0
  </code>

</p>

## More Examples

Evaluating an expression is as simple as: 

```java 

MathExpression expr = new MathExpression("(34+32)-44/(8+9(3+2))-22"); 

System.out.println("result: " + expr.solve()); 
``` 
This gives: 43.16981132075472 

#### Or using variables and calculating simple expressions: 

```java 
MathExpression expr = new MathExpression("r=3;P=2*pi*r;"); 

System.out.println("result: " + expr.getValue("P")); 
```
#### Or using functions: 

```java
MathExpression expr = new MathExpression("f(x)=39*sin(x^2)+x^3*cos(x);f(3)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: -10.65717648378352 

#### Derivatives (Differential Calculus)

To evaluate the derivative at a given point(Note it does symbolic differentiation(not numerical) behind the scenes, so the accuracy is not limited by the errors of numerical approximations): 
```java
MathExpression expr = new MathExpression("f(x)=x^3*ln(x); diff(f,3,1)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: 38.66253179403897 

The above differentiates x^3 * ln(x) once at x=3. 
The number of times you can differentiate is 1 for now. 

#### For Numerical Integration: 

```java 

MathExpression expr = new MathExpression("f(x)=2*x; intg(f,1,3)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: 7.999999999998261... approx: 8 ...


I will talk about other functionalities of the library, such as numerical integration later on! Thanks.
