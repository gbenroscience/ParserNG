# ParserNG
<b>ParserNG</b> is a powerful open-source math tool that parses and evaluates algebraic expressions and also knows how to handle a lot of mathematical expressions. 

* [ParserNG](#ParserNG)
    * [Usage and note](#usage-and-note)
    * [FEATURES](#features)
* [Using ParserNG as commandline tool](#using-parserng-as-commandline-tool)
    * [cmdline examples](#cmdline-examples)
* [Using ParserNG as library](#using-parserng-as-library)
    * [Inbuilt Functions](#inbuilt-functions)
    * [User defined functions](#user-defined-functions)
    * [User hardcoded functions](#user-hardcoded-functions)
    * [Differential Calculus](#differential-Calculus)
* [More Examples](#more-examples)
    * [Or using variables and calculating simple expressions](#or-using-variables-and-calculating-simple-expressions)
    * [Or using functions](#or-using-functions)
    * [Derivatives - Differential Calculus](#derivatives---differential-calculus)
    * [For Numerical Integration](#for-numerical-integration)
* [Functions and FunctionManager, Variables and VariableManager](#functions-and-functionmanager-variables-and-variablemanager)
* [Matrices](#matrices)
    * [Parser manipulation of matrices](#parser-manipulation-of-matrices)
       * [1. Create a matrix](#1-create-a-matrix)
       * [2. Determinants](#2-determinants)
       * [3. Solving simultaneous linear equations](#3-solving-simultaneous-linear-equations)
       * [4. Building triangular matrices](#4-building-triangular-matrices)
       * [5. Echelon form of a matrix](#5-echelon-form-of-a-matrix)
       * [6. Matrix multiplication](#6-matrix-multiplication)
       * [7. Matrix addition](#7-matrix-addition)
       * [8. Matrix subtraction](#8-matrix-subtraction)
       * [9. Powers of a Matrix](#9-powers-of-a-matrix)
       * [10. Transpose of a Matrix](#10-transpose-of-a-matrix)
       * [11. Editing a Matrix](#11-editing-a-matrix)
          * [Editing a Matrix example](#editing-a-matrix-example)
       * [12. Finding the characteristic polynomial of a Matrix](#12-finding-the-characteristic-polynomial-of-a-matrix)
* [Logical Calculus](#logical-calculus)
* [Expanding Calculus](#expanding-calculus)
* [TO BE CONTINUED](#to-be-continued)


## Usage and note

If you need to use the parser directly in your Android project, go to:
[parserng-android](https://github.com/gbenroscience/parserng-android) by the same author
<br>

If you need to access this library via Maven Central, do:
      
 
        <dependency>
            <groupId>com.github.gbenroscience</groupId>
            <artifactId>parser-ng</artifactId>
            <version>0.1.9</version>
        </dependency>
       

This library was created in 2009 and later used by the author as part of a critical part of his University final year project
at the Department of Computer Science and Engineering, Obafemi Awolowo University,Ile-Ife, Osun State, Nigeria.

The design goal of this library was to create a simple, yet powerful, not too bogus math tool that scientists and developers could deploy with their
work to solve mathematical problems both simple and complex.

ParserNG is written completely in (pure) Java and so is as cross-platform as Java can be. It has been used to design math platforms for desktop Java, Java MicroEdition devices(as far back as 2010-2011) , Android,  and by porting the whole platform using J2OBJC from Google; Swift also. The performance has been exceptionally acceptable in all cases.


## FEATURES
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

## Using ParserNG as commandline tool
You can use jar directly as commandline calculus. Unless the tool is packed to your distribution:
```
java -jar parser-ng-0.1.8.jar  1+1
2.0
```
Or as logical parser
```
java -jar parser-ng-0.1.8.jar -l true and true
true
java -jar parser-ng-0.1.8.jar -l "2 == (4-2)"
true
```
You can get help by 
```
java -jar parser-ng-0.1.8.jar  -h
  ParserNG 0.1.8 math.Main
-h/-H/--help         this text; do not change for help (witout dashes), which lists functions
-v/-V/--verbose      output is reprinted to stderr with some inter-steps
-l/-L/--logic        will add logical expression wrapper around the expression
                     Logical expression parser is much less evolved and slow. Do not use it if you don't must
                     If you use logical parse, result is always true/false. If it is not understood, it reuslts to false
-t/-T/--trim         by default, each line is one expression,
                     however for better redability, sometimes it is worthy to
                     to split the expression to multiple lines. and evaluate as one.
-i/-I/--interactive  instead of evaluating any input, interactive prompt is opened
                     If you lunch interactive mode wit TRIM, the expression is
                     evaluated once you exit (done, quit, exit...)
                     it is the same as launching parser.cmd.ParserCmd main class
           To read stdin, you have to set INTERACTIVE mode on
           To list all known functions,  type `help` as MathExpression
  Without any parameter, input is considered as math expression and calculated
  without trim, it would be the same as launching parser.MathExpression main class
  run help in verbose mode (-h -v) to get examples

```
You  can get examples by verbose help:
```
java -jar parser-ng-0.1.8.jar  -h -v
```
you can list functions:
```
java -jar parser-ng-0.1.8.jar  help
List of currently known methods:
acos        - help not yet written. See https://github.com/gbenroscience/ParserNG
...
variance    - help not yet written. See https://github.com/gbenroscience/ParserNG
List of functions is just tip of iceberg, see: https://github.com/gbenroscience/ParserNG for all features
```
you can list logical operators:
```
java -jar parser-ng-0.1.8.jar  -l help
Comparing operators: !=, ==, >=, <=, le, ge, lt, gt, <, >
Logical operators: impl, xor, imp, eq, or, and, |, &
As Mathematical parts are using () as brackets, Logical parts must be grouped by [] eg.
Negation can be done by single ! strictly close attached to [; eg ![true]  is ... false. Some spaces like ! [ are actually ok to
...
```
  Note, that parser.MathExpression nor parser.LogicalExpression classes do not take any parameters except expressions
  Note, that parser.cmd.ParserCmd class takes single parameter -l/-L/--logic to contorl its evaluation

Program can work with stdin, out and err properly. Can work with multiline input - see `-t` switch. If you ned to work with stdin, use `-i` which is otherwise interactive mode

### cmdline examples
Following lines describes, how stdin/arguments are processed, and how different is input/output with `-t` on/off
```
   java -jar parser-ng-0.1.8.jar -h
    this help
  java -jar parser-ng-0.1.8.jar 1+1
    2.0
  java -jar parser-ng-0.1.8.jar "1+1
                                 +2+2"
    2.0
    4.0
  java -jar parser-ng-0.1.8.jar -t "1+1
                                    +2+2"
    6.0
  java -jar parser-ng-0.1.8.jar -i  1+1
    nothing, will expect manual output, and calculate line by line
  java -jar parser-ng-0.1.8.jar -i -t  1+1
    nothing, will expect manual output and calcualte it all as one expression
  echo 2+2 | java -jar parser-ng-0.1.8.jar  1+1
    2.0
  echo "1+1 
        +2+2 | java -jar parser-ng-0.1.8.jar -i
    2.0
    4.0
  echo "1+1 
        +2+2 | java -jar parser-ng-0.1.8.jar -i -t
    6.0
  java -cp parser-ng-0.1.8.jar parser.cmd.ParserCmd "1+1
    will ask for manual imput en evaluate per line
  echo "1+1 
        +2+2 | java -cp parser-ng-0.1.8.jar parser.cmd.ParserCmd 2>/dev/null
    2.0
    4.0
  java -cp parser-ng-0.1.8.jar parser.MathExpression "1+1
                                                      +2+2"
    6.0
  java -cp parser-ng-0.1.8.jar parser.LogicalExpression "true or false"
    true

```

## Using ParserNG as library
The simplest way to evaluate an expression in ParserNG is to use the <code>MathExpression</code> class.
<code>MathExpression</code> is the class responsible for basic expression parsing and evaluation.

Do:<br>
`MathExpression expr = new MathExpression("r=4;r*5");`
<br>
`System.out.println("result: " + expr.solve());`

<span>What does this do?</span>

It creates a variable called <code>r</code>and sets its value to `4`. Then it goes ahead to evaluate the expression
`r*5` and returns its value when `expr.solve()` is called. <br>The print statement would give
<br><br>`solution: 20.0`<br><br>
at the console.


Some key applications of parsers involve repeated iterations of a given expression at different values of the variables involved. Iteratively determining the roots of an equation, graphing etc.

For repeated iterations of an expression over a value range, say 'x^2+5*x+1', the wrong usage would be:<br>

```java
for(int i=0;i<10000;i++){

double x = i;
MathExpression expression = new MathExpression("x="+i+";x^2+5*x+1");
expression.solve();

}
```
<br>

The `MathExpression` constructor basically does all the operations of scanning and interpreting of the input expression. This is a very expensive operation. It is better to do it just once and then run the `solve()` method over and over again at various values of the variables.
  

For example:


```java
MathExpression expression = new MathExpression("x=0;x^2+5*x+1");

for(int i=0; i<100000; i++){
expression.setValue("x", String.valueOf(i) );
expression.solve();//Use the value from here according to your iterative needs...e.g plot a graph , do some summation etc..
}
```
<br>
This ensures that the expression is parsed once(expensive operation) and then evaluated at various values of the variables. This second step is an high speed one, sometimes taking barely 3 microseconds on some machines.<br><br>


#### Inbuilt Functions
The parser has its own set of built-in functions. They are:

    sin,cos,tan,sinh,cosh,tanh,sin-¹,cos-¹,tan-¹,sinh-¹,cosh-¹,tanh-¹,sec,csc,cot,
    sech,csch,coth,sec-¹,csc-¹,cot-¹,sech-¹,csch-¹,coth-¹,exp,ln,lg,log,ln-¹,lg-¹,log-¹,
    asin,acos,atan,asinh,acosh,atanh,asec,acsc,acot,asech,acsch,acoth,aln,alg,alog,
    round,roundN,roundDigitsN,floor,floorN,floorDigitsN,ceil,ceilN,ceilDigitsN,length
    abs,sqrt,cbrt,inverse,square,cube,pow,fact,comb,perm,
    sum,prod,avg,med,mode,geom,gsum,count,avgN,geomN
    rng,mrng,rms,cov,min,max,s_d,variance,st_err,rnd,sort,plot,diff,intg,quad,t_root,
    root,linear_sys,det,invert,tri_mat,echelon,matrix_mul,matrix_div,matrix_add,matrix_sub,matrix_pow,transpose,matrix_edit
    
<br>
For runtime loaded list of all functions (with description, even in-runtime-added - see User hardcoded functions), and environment variables, run <code>help</code> as MathExpression's value<br>

  ```
MathExpression expression = new MathExpression("help");
expression.solve();
  ``` 
  
##### Environment variables/java properties (so setup-able) in runtime).
See <code>help</code> for actual version-specific, up-to date, list<br>
<li> RADDEGDRAD_PNG - DEG/RAD/GRAD - allows to change units for trigonometric operations. Default is RAD. It is same as <code>MathExpression().setDRG(...)</code>
  <br>


Note that alternatives to many functions having the inverse operator are provided in the form of an 'a' prefix.
For example the inverse <code>sin</code> function is available both as <code>sin-¹</code> and as <code>asin</code>

#### User defined functions
You can also define your own functions and use them in your math expressions.
This is done in one of 2 ways:
<ol>
  <li>
 
        f(x,a,b,c,...) = expr_in_said_variables
        
   <br> 
        For example: 
 
       f(x,y)=3*x^2+4*x*y+8

  </li>
  <li>
 
       f = @(x,a,b,c,...)expr_in_said_variables
 <br> For example: 
 
      f = @(x,y)3*x^2+4*x*y+8

</li>  
</ol>

<i>
Your defined functions are volatile and will be forgotten once the current parser session is over. The only way to have the parser remember them always is to introduce some form of persistence.
</i>
<br>
So for instance, you could pass the following to a MathExpression constructor:

    f(x)=sin(x)+cos(x-1)
   <br>
And then do: 
     
     f(2)
 
 the parser automatically calculates 
    
     sin(2)+cos(2-1)
  
  behind the scenes.<br><br>
Note, that such functions do not propagate to help.

#### User hardcoded functions
if you need more complex function, it is best to hardcode it and contribute it.
However sometimes the mehtod may be to dummy, or review to slow, so for such cases you can implement `BasicNumericalMethod` interface and `Declarations.registerBasicNumericalMethod` it.
Such method will be used as any other hardcoded function. See `MathExpressionTest.customUserFunctionTest` for basic example.
Note, that current implementation is stateless. It may be changed in future if needed. Unlike `User defined functions` those methods propagate to help.

#### Differential Calculus

<b>ParserNG</b> makes differentiating math expressions really easy.

 

It uses its very own implementation of a symbolic differentiator.
  
  It performs symbolic differentiation of expressions behind the scenes and then computes the differential coefficient
  of the function at some supplied x-value.
  <br><br>
  
  
  
  To differentiate a function, do: <br><br>
  
    
       
 ```java
 MathExpression expr = new MathExpression("diff(@(x)x^3,3,1)");
  
 System.out.println(ex.solve());
 ```
  
  This will print:
 
        27.0


 

## More Examples

Evaluating an expression is as simple as: 

```java 

MathExpression expr = new MathExpression("(34+32)-44/(8+9(3+2))-22"); 

System.out.println("result: " + expr.solve()); 
``` 
This gives: `43.16981132075472`

#### Or using variables and calculating simple expressions

```java 
MathExpression expr = new MathExpression("r=3;P=2*pi*r;"); 

System.out.println("result: " + expr.getValue("P")); 
```
#### Or using functions

```java
MathExpression expr = new MathExpression("f(x)=39*sin(x^2)+x^3*cos(x);f(3)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: `-10.65717648378352` 

#### Derivatives - Differential Calculus

To evaluate the derivative at a given point(Note it does symbolic differentiation(not numerical) behind the scenes, so the accuracy is not limited by the errors of numerical approximations): 
```java
MathExpression expr = new MathExpression("f(x)=x^3*ln(x); diff(f,3,1)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: `38.66253179403897` 

The above differentiates x<sup>3</sup> * ln(x) once at x=3. 
The number of times you can differentiate is 1 for now. 

#### For Numerical Integration

```java 

MathExpression expr = new MathExpression("f(x)=2*x; intg(f,1,3)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: `7.999999999998261...` approximately: `8` ...


## Functions and FunctionManager, Variables and VariableManager


ParserNG comes with a FunctionManager class that allows users persist store functions for the duration of the session(JVM run).

You may create and store a function directly by doing:

    FunctionManager.add("f(x,y) = x-x/y");
And then retrieve and use the function like this:

    Function fxy = FunctionManager.lookUp("f");

Or you may create the function directly and store it, like:

    Function fxy = new Function("f(x,y) = x-x/y");

And then store it using:

     FunctionManager.add(fxy);

The same applies to variables.

The variables that you create go into the VariableManager.
So if you do:

`MathExpression me = new MathExpression("a=5;4*a");`

The parser immediately creates a variable called `a` , stores 5 in it, and saves the variable in the VariableManager.
This variable can be used within other `MathExpression`s that you create within the current parser session.

## Matrices

ParserNG deals with matrices; howbeit on a functional level. On the way though is a pure Matrix Algebra parser component which is one of the targets set for the platform.

Currently you can define matrices and even store them like functions...

For example to define and store a matrix <b>M</b>

     FunctionManager.add("M=@(3,3)(3,4,1,2,4,7,9,1,-2)"); 

This can be extracted as a function by doing a simple lookup:

    Function matrixFun = FunctionManager.lookUp("M");
To find its determinant, do something like:

    double det = matrixFun.calcDet();

You can do more by getting the underlying Matrix object, i.e do:

    Matrix m = matrixFun.getMatrix();

But I digress. Let us look at the matrix functionality runnable from within the parser.

### Parser manipulation of matrices

The parser comes with inbuilt matrix manipulating functions.


#### 1. Create a matrix

        MathExpression expr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");

This expression creates a new matrix function , `M` and stores it in the FunctionManager.

Or the more direct form:

        FunctionManager.add("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        
#### 2. Determinants

To calculate the determinant of the matrix `M`, above do:

        MathExpression expr = new MathExpression("det(M)");
        System.out.println(m.solve());

This gives:
         
         188.99999999999997
         
         
#### 3. Solving simultaneous linear equations

The function that does this is `linear_sys`

To represent the linear system:

    2x + 3y = -5
    3x - 4y = 20
 
 in ParserNG, do:
 
 
         MathExpression linear = new MathExpression("linear_sys(2,3,-5,3,-4,20)");
         System.out.println("soln: "+linear.solve());
         
This prints:

     soln: 
     2.3529411764705888            
     -3.235294117647059`

#### 4. Building triangular matrices

Say you have defined a matrix `M` as in past examples, to decompose it into a triangular matrix, do:

     MathExpression expr = new MathExpression("tri_mat(M)");
        System.out.println(expr.solve());
        
For the matrix above, this would give:


    1.0  ,1.3333333333333333  ,0.3333333333333333            
    0.0  ,    1.0  ,4.749999999999999            
    0.0  ,    0.0  ,    1.0            
        
         
#### 5. Echelon form of a matrix

To find the echelon of the matrix `M` defined in 1. do,

     MathExpression expr = new MathExpression("echelon(M)");
     System.out.println(expr.solve());
     
This would give:

     3.0  ,    4.0  ,    1.0            
     0.0  ,    4.0  ,   19.0            
     0.0  ,    0.0  ,  567.0     
         


#### 6. Matrix multiplication

ParserNG of course allows matrix multiplication with ease.

To multiply 2 matrices in 1 step: Do,

    MathExpression mulExpr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    P=matrix_mul(M,N);P;");
    System.out.println("soln: "+mulExpr.solve());
    
      
   Or: 
   
    MathExpression mulExpr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    matrix_mul(M,N);");
    System.out.println("soln: "+mulExpr.solve());
    
         
This would give:

    25.0  ,    8.0  ,   45.0            
    51.0  ,   13.0  ,   91.0            
    28.0  ,    8.0  ,   57.0   


#### 7. Matrix addition

ParserNG allows easy addition of matrices.

To add 2 matrices in 1 step: Do,

    MathExpression addMat = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    P=matrix_add(M,N);P;");
    System.out.println("soln: "+ addMat.solve());
    
      
   Or: 
   
    MathExpression addMat = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    matrix_add(M,N);");
    System.out.println("soln: "+addMat.solve());
    
         
This would give:

    7.0  ,    5.0  ,    9.0            
    4.0  ,    5.0  ,   10.0            
    14.0  ,    2.0  ,    7.0   



#### 8. Matrix subtraction

ParserNG also allows matrix subtraction.

To find the difference of 2 matrices in 1 step: Do,

    MathExpression subMat = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    P=matrix_sub(M,N);P;");
    System.out.println("soln: "+ subMat.solve());
    
      
   Or: 
   
    MathExpression subMat = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    matrix_sub(M,N);");
    System.out.println("soln: "+ subMat.solve());
    
         
This would give:

    -1.0  ,    3.0  ,   -7.0            
     0.0  ,    3.0  ,    4.0            
     4.0  ,    0.0  ,  -11.0 



#### 9. Powers of a Matrix

ParserNG also allows quick computation of powers of a matrix.

Here, given a matrix `M` , M<sup>2</sup> is defined as `MxM` and M<sup>n</sup> is defined as `MxMxM...(n times)`

To find the power of a matrix, say M<sup>4</sup>,  do:

    MathExpression mpow = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    P=matrix_pow(M,4);P;");
    System.out.println("soln: "+mpow.solve());
    
      
   Or: 
   
    MathExpression mpow = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    matrix_pow(M,4);");
    System.out.println("soln: "+ mpow.solve());
    
         
This would give:

    3228.0  , 2755.0  , 1798.0            
    4565.0  , 3802.0  , 3049.0            
    3432.0  , 2257.0  , 1327.0            



#### 10. Transpose of a Matrix

ParserNG also allows quick computation of the transpose of a matrix.


To find the transpose of a matrix, `M`, do:

    MathExpression trexp = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);P=transpose(M);P;");
    System.out.println("soln: "+ trexp.solve());
    
      
   Or: 
   
    MathExpression trexp = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);transpose(M);");
    System.out.println("soln: "+ trexp.solve());
    
         
This would give:

    3.0  ,    2.0  ,    9.0            
    4.0  ,    4.0  ,    1.0            
    1.0  ,    7.0  ,   -2.0            




#### 11. Editing a Matrix

ParserNG also allows the entries in a matrix to be edited.

The command for this is: `matrix_edit(M,2,2,-90)`

The first argument is the Matrix object which we wish to edit.
The second and the third arguments respectively represent the row and column that we wish to edit in the Matrix.

The last entry represents the value to store in the specified location(entry) in the Matrix.

##### Editing a Matrix example

To edit the contents of a matrix, `M`, do:

    MathExpression trexp = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);P=matrix_edit(M,2,2,-90);P;");
    System.out.println("soln: "+ trexp.solve());
    
      
   Or: 
   
    MathExpression trexp = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);matrix_edit(M,2,2,-90);");
    System.out.println("soln: "+ trexp.solve());
    
         
This would give:

    3.0  ,    4.0  ,    1.0            
    2.0  ,    4.0  ,    7.0            
    9.0  ,    1.0  ,  -90.0           


Note that matrix indexes in ParserNG are zero-based, so be advised accordingly as entering an invalid row/column combination will throw an error in your code.


#### 12. Finding the characteristic polynomial of a Matrix

ParserNG allows the quick evaluation of the characteristic polynomial of a square matrix; this polynomial can then be solved to find the eigenvalues, and hence the eigenvector of the Matrix.

The function is called `eigpoly`

<p style="font-weight:bold;color:brown;font-style:italic;font-size:1.3em">
Actually, there is a function called `eigvec`, which in the future will allow the user to automatically generate the eigenvector from the Matrix; but at the moment, we cannot completely solve all generated polynomials completely, so the `eigvec` function is still in the works.
</p>

To generate the characteristic polynomial, do:

    MathExpression expression = new MathExpression("eigpoly(@(3,3)(4,2,1,3,1,8,-5,6,12))");
    System.out.println("soln: "+ expression.solve());

This will give:

    anon2=@(n)(-273.0*n^0.0-15.0*n^1.0+17.0*n^2.0-1.0*n^3.0)
    
The `anon2` may be `anon` anything.<br><br>
`anon` signifies an automatically generated anonymous function created to hold a function value that no variable was created for by the user.

So the parser keeps records of them by using the prefixed variable name, `anon`, alongside a digit which indicates the position of the referenced anonymous function in memory.

Note that the anonymous function is a valid function in `n`, and so if you do: anon2(12) it will evaluate the eigen polynomial (the characteristic polynomial) at `n=12`

If you did:

    MathExpression expression = new MathExpression("eigpoly(@(5,5)(12,1,4,2,9,3,1,8,-5,6,13,9,7,3,5,7,3,5,4,9,13,2,4,8,6))");
    System.out.println("soln: "+ expression.solve());
    
This would give:

     anon3=@(n)(20883.0*n^0.0+1155.0*n^1.0-1667.0*n^2.0+30.0*n^4.0-1.0*n^5.0+35.0*n^3.0)

## Logical Calculus

The logical expressions in math engine have theirs intentional limitations. Thus allmighty logical expression parser was added around individually evaluated Mathematical expressions which results can be later compared, and logically grouped.  The simplest way to evaluate an logical  expression in ParserNG is to use the <code>LogicalExpression</code> class.
<code>LogicalExpression</code> is the class responsible for basic comaprsions and logica  expression parsing and evaluation. It calls <code>MathExpression</code> to ech of its basic non-logical parts. The default <code>MathExpression</code> can be repalced by any custom implementation of <code>Solvable</code>, but it is only for highly specialized usages. Highlight, where MathExpression is using <code>()</code> for mathematical bracketing, LogicalExpression - as () can be part of underlying comapred mathematical expressiosn  uses <code>[]</code> brackets.<br>
<br>
In CLI, you can use -l/-L/--logic switch to work with LogicalExpression. Although it is fully compatible with MathExpression you may face unknown issue<br>
<br>
Do:<br>
`LogicalExpression expr = new LogicalExpression("[1+1 < (2+0)*1 impl [ [5 == 6 || 33<(22-20)*2 ]xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]]] eq [ true && false ] ");`
<br>
`System.out.println("result: " + expr.solve());`
<br>
`true`

<span>What does this do? it will call new MathExpression(...) to each math expression, then comapre them all and then do logicla operations above resulting booleans</span> 
```
brackets: [1+1 < (2+0)*1 impl [ [5 == 6 || 33<(22-20)*2 ]xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]]] eq [ true && false ] 
  brackets: 1+1 < (2+0)*1 impl [ [5 == 6 || 33<(22-20)*2 ]xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]]
    brackets:  [5 == 6 || 33<(22-20)*2 ]xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]
        evaluating: 5 == 6 || 33<(22-20)*2 
          evaluating: 5 == 6
            evaluating: 5
            is: 5
            evaluating: 6
            is: 6
          ... 5 == 6
          is: false
          evaluating: 33<(22-20)*2 
            evaluating: 33
            is: 33
            evaluating: (22-20)*2 
            is: 4.0
          ... 33 < 4.0
          is: false
        ... false | false
        is: false
    to:   false xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]
      brackets:  [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2
          evaluating:   5-3 < 2 or 7*(5+2)<=5 
            evaluating:   5-3 < 2
              evaluating:   5-3
              is: 2.0
              evaluating: 2
              is: 2
            ... 2.0 < 2
            is: false
            evaluating: 7*(5+2)<=5 
              evaluating: 7*(5+2)
              is: 49.0
              evaluating: 5 
              is: 5
            ... 49.0 <= 5
            is: false
          ... false or false
          is: false
      to:   false  and 1+1 == 2
          evaluating:   false  and 1+1 == 2
            evaluating:   false
            is: false
            evaluating: 1+1 == 2
              evaluating: 1+1
              is: 2.0
              evaluating: 2
              is: 2
            ... 2.0 == 2
            is: true
          ... false and true
          is: false
    to:   false xor  false 
        evaluating:   false xor  false 
          evaluating:   false
          is: false
          evaluating: false 
          is: false
        ... false xor false
        is: false
  to: 1+1 < (2+0)*1 impl  false 
      evaluating: 1+1 < (2+0)*1 impl  false 
        evaluating: 1+1 < (2+0)*1
          evaluating: 1+1
          is: 2.0
          evaluating: (2+0)*1
          is: 2.0
        ... 2.0 < 2.0
        is: false
        evaluating: false 
        is: false
      ... false impl false
to:  true  eq [ true && false ] 
    evaluating:  true && false 
      evaluating:  true
      is: true
      evaluating: false 
      is: false
    ... true & false
    is: false
to:  true  eq  false  
    evaluating:  true  eq  false  
      evaluating:  true
      is: true
      evaluating: false  
      is: false
    ... true eq false
    is: false
false
```
Note, that logical parsser's comparsions supports only dual operators, so where true|false|true is valid, 1<2<3  is invalid!
Thus:  [1<2]<3   is necessary and  even  [[true|false]|true]is recomeded to be used, For 1<2<3  exception is thrown.
Single letter can logical operands can be used in row. So eg | have same meaning as ||. But also unluckily also eg < is same as <<
Negation can be done by single ! strictly close attached to [; eg ![true]  is ... false. Some spaces like ! [ are actually ok to
Note, that variables works, but must be included in first evaluated expression. Which is obvious for "r=3;r<r+1"
But much less for [r=3;r<r+1 || [r<5]]", which fails and must be declared as "[r<r+1 || [r=3;r<5]]"
To avoid this, you can declare all in first dummy expression: "[r=3;r<1] || [r<r+1 || [r<5]]" which ensure theirs allocation ahead of time and do not affect the rest
If you modify the variables, in the subseqet calls, results maybe funny. Use verbose mode to debug order

## Expanding Calculus
Very often an expressions, or CLI is called above known, huge (generated) array of values. Such can be processed via <code>ExpandingExpression</code>. Unlike  other Expressins, this one have List<String> as aditional parameters, where each member is a number. THose numbers can thenbe accessed as L0, L1...Ln. Size of the list is held in special MN variable. The index can be calucalted dynamically, like L{MN/2} - in example of four items, will expand to L2. Although `{}` and `MN` notations are powerfull, the main power is in *slices*:
```
Instead of numbers, you can use literalls L0, L1...L99, which you can then call by:
Ln - vlaue of Nth number
L2..L4 - will expand to values of L2,L3,L4 - order is hnoured
L2.. - will expand to values of L2,L3,..Ln-1,Ln
..L5 - will expand to values of  L0,L1...L4,L5
where ..L5 or L2.. are order sensitive, the L{MN}..L0 or L0..L{MN} is not. But requires dynamic index evaluation
When used as standalone, VALUES_PNG xor VALUES_IPNG  are used to pass in the space separated numbers (the I is inverted order)
Assume VALUES_PNG='5 9 3 8', then it is the same as VALUES_IPNG='8 3 9 5'; BUt be aware, with I the L.. and ..L are a bit oposite then expected
L0 then expand to 8; L2.. expands to 9,3,8; ' ..L2 expands to 5,9 
L2..L4 expands to 9,5; L4..L2 expands to 5,9
```
ExpandingExpression calls <code>LogicalExpression</code> inside, and yet again the underlying Math evaluator is - defaulting as  <code>MathExpression</code> can be repalced by any custom implementation of <code>Solvable</code>. Highlight, where MathExpression is using <code>()</code> for mathematical bracketing, LogicalExpression ses <code>[]</code> brackets. The dynamic indexes in <code>ExpandingExpression</code> uses are wrapped in `{}`  <br>
<br>
In CLI, you can use -e/-E/--expanding switch to work with Expanding expressions. The array of numbers goes in via VALUES_PNG xor VALUES_IPNG variable. Although it is fully compatible with MathExpression and LogicalExpression you may face unknown issue<br>
<br>
Example:<br>
```
VALUES_PNG="1 8 5 2" java -jar target/parser-ng-0.1.9.jar -e "avg(..L{MN/2})*1.1-MN <  L0 | (L1+L{MN-1})*1.3 + MN<  L0" -v
avg(..L{MN/2})*1.1-MN <  L0 | (L1+L{MN-1})*1.3 + MN<  L0 
Expression : avg(..L{MN/2})*1.1-MN <L0 | (L1+L{MN-1})*1.3 + MN<L0 
Upon       : 1,8,5,2
As         : Ln...L1,L0
MN         = 4
  L indexes brackets: avg(..L{4/2})*1.1-4 <  L0 | (L1+L{4-1})*1.3 + 4<  L0 
    Expression : 4/2
    Expanded as: 4/2
    is: 2.0
    4/2 = 2 (2.0)
  to: avg(..L 2 )*1.1-4 <  L0 | (L1+L{4-1})*1.3 + 4<  L0 
    Expression : 4-1
    Expanded as: 4-1
    is: 3.0
    4-1 = 3 (3.0)
  to: avg(..L 2 )*1.1-4 <  L0 | (L1+L 3 )*1.3 + 4<  L0 
Expanded as: avg(1,8 )*1.1-4 <  2 | (5+1 )*1.3 + 4<  2 
avg(1,8 )*1.1-4 <  2 | (5+1 )*1.3 + 4<  2 
  brackets: avg(1,8 )*1.1-4 <  2 | (5+1 )*1.3 + 4<  2 
      evaluating logical: avg(1,8 )*1.1-4 <  2 | (5+1 )*1.3 + 4<  2 
        evaluating comparison: avg(1,8 )*1.1-4 <  2
          evaluating math: avg(1,8 )*1.1-4
          is: 0.9500000000000002
          evaluating math: 2
          is: 2
        ... 0.9500000000000002 < 2
        is: true
        evaluating comparison: (5+1 )*1.3 + 4<  2 
          evaluating math: (5+1 )*1.3 + 4
          is: 11.8
          evaluating math: 2 
          is: 2
        ... 11.8 < 2
        is: false
      ... true | false
      is: true
  true
is: true
true
```

## TO BE CONTINUED
And much more!

