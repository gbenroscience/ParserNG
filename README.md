# ParserNG
<b>ParserNG</b> is a powerful open-source math tool that parses and evaluates algebraic expressions and also knows how to handle a lot of mathematical expressions. 

## NOTE:

If you need to use the parser directly in your Android project, go to:
[parserng-android](https://github.com/gbenroscience/parserng-android) by the same author
<br>

If you need to access this library via Maven Central, do:
      
 
        <dependency>
            <groupId>com.github.gbenroscience</groupId>
            <artifactId>parser-ng</artifactId>
            <version>0.1.5</version>
        </dependency>
       

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
MathExpression expression = new MathExpression("x^2+5*x+1");

for(int i=0; i<100000; i++){
expression.setValue("x", String.valueOf(i) );
expression.solve();//Use the value from here according to your iterative needs...e.g plot a graph , do some summation etc..
}
```
<br>
This ensures that the expression is parsed once(expensive operation) and then evaluated at various values of the variables. This second step is an high speed one, sometimes taking barely 3 microseconds on some machines.<br><br>


<b>Inbuilt Functions</b><br>
The parser has its own set of built-in functions. They are:

    sin,cos,tan,sinh,cosh,tanh,sin-¹,cos-¹,tan-¹,sinh-¹,cosh-¹,tanh-¹,sec,csc,cot,
    sech,csch,coth,sec-¹,csc-¹,cot-¹,sech-¹,csch-¹,coth-¹,exp,ln,lg,log,ln-¹,lg-¹,log-¹,
    asin,acos,atan,asinh,acosh,atanh,asec,acsc,acot,asech,acsch,acoth,aln,alg,alog,
    floor,ceil,sqrt,cbrt,inverse,square,cube,pow,fact,comb,perm,sum,prod,avg,med,mode,
    rng,mrng,rms,cov,min,max,s_d,variance,st_err,rnd,sort,plot,diff,intg,quad,t_root,
    root,linear_sys,det,invert,tri_mat,echelon,matrix_mul,matrix_div,matrix_add,matrix_sub,matrix_pow,transpose,matrix_edit
    
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
This gives: `-10.65717648378352` 

#### Derivatives (Differential Calculus)

To evaluate the derivative at a given point(Note it does symbolic differentiation(not numerical) behind the scenes, so the accuracy is not limited by the errors of numerical approximations): 
```java
MathExpression expr = new MathExpression("f(x)=x^3*ln(x); diff(f,3,1)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: `38.66253179403897` 

The above differentiates x<sup>3</sup> * ln(x) once at x=3. 
The number of times you can differentiate is 1 for now. 

#### For Numerical Integration: 

```java 

MathExpression expr = new MathExpression("f(x)=2*x; intg(f,1,3)"); 
System.out.println("result: " + expr.solve()); 
```
This gives: `7.999999999998261...` approximately: `8` ...


## Functions and FunctionManager , Variables and VariableManager


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

### Parser manipulation of matrices.

The parser comes with inbuilt matrix manipulating functions.


#### 1. Create a matrix:

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
         
         
#### 3. Solving simultaneous linear equations.

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

To multiply 2 matrices in 1 step: Do,

    MathExpression mulExpr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    P=matrix_add(M,N);P;");
    System.out.println("soln: "+mulExpr.solve());
    
      
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

To multiply 2 matrices in 1 step: Do,

    MathExpression mulExpr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    P=matrix_sub(M,N);P;");
    System.out.println("soln: "+mulExpr.solve());
    
      
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
   
    MathExpression mulExpr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);
    matrix_pow(M,4);");
    System.out.println("soln: "+mulExpr.solve());
    
         
This would give:

    3228.0  , 2755.0  , 1798.0            
    4565.0  , 3802.0  , 3049.0            
    3432.0  , 2257.0  , 1327.0            



#### 9. Transpose of a Matrix

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




#### 10. Editing a Matrix

ParserNG also allows the entries in a matrix to be edited.

The command for this is: `matrix_edit(M,2,2,-90)`

The first argument is the Matrix object which we wish to edit.
The second and the third arguments respectively represent the row and column that we wish to edit in the Matrix.

The last entry represents the value to store in the specified location(entry) in the Matrix.

##### For example

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




## TO BE CONTINUED

