# ParserNG
ParserNG is a powerful open-source math tool that parses and evaluates algebraic expressions. 

It was created in 2009 by me, Gbemiro Jiboye and later used as a critical part of my final year project
at the Department of Computer Science and Engineering, Obafemi Awolowo University,Ile-Ife, Osun State, Nigeria.

My goal was to create a simple, yet powerful, not too bogus math tool that scientists and developers can deploy with their
work to solve problems of all ranges--from simple to complex.


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


for(int i=0;i<10000;i++){

double x = i;
MathExpression expression = new MathExpression("x="+i+";x^2+5*x+1");<br>

expression.solve();<br>

}<br>

The MathExpression constructor basically does all the operations of scanning and interpreting of the input expression. This is a very expensive operation. It is better to do it just once and then run the solve() method over and over again at various values of the variables.

For example:


 
      MathExpression expression = new MathExpression("x^2+5*x+1");
 
        for(int i=0;i<100000;i++){
            expression.setValue("x" , i+"");
           expression.solve();
        }
        
        This ensures that the expression is parsed once (expensive operation) and then evaluated at various values of the variables. This second step is an high speed one, sometimes taking barely 3 microseconds on some machines.


 




<b>Inbuilt Functions</b><br>
The parser has its own set of inbuilt functions. They are:
<code>
sin,cos,tan,sinh,cosh,tanh,sin-¹,cos-¹,tan-¹,sinh-¹,cosh-¹,tanh-¹,sec,csc,cot,sech,csch,coth,sec-¹,csc-¹,cot-¹,sech-¹,csch-¹,coth-¹,exp,ln,lg,log,ln-¹,lg-¹,log-¹,asin,acos,atan,asinh,acosh,atanh,asec,acsc,acot,asech,acsch,acoth,aln,alg,alog,floor,ceil,sqrt,cbrt,inverse,square,cube,pow,fact,comb,perm,sum,prod,avg,med,mode,rng,mrng,rms,cov,min,max,s_d,variance,st_err,rnd,sort,plot,diff,intg,quad,t_root,root,linear_sys,det,invert,tri_mat,echelon,matrix_mul,matrix_div,matrix_add,matrix_sub,matrix_pow,transpose,matrix_edit,
</code>

Note that alternatives to many functions having the inverse operator are provided in the form of an 'a' prefix.
For example the inverse <code>sin</code> function is available both as <code>sin-¹</code> and as <code>asin</code>

<b>User defined functions</b><br>
You can also define your own functions and use them in your math expressions.
This is done in 2 ways:
<ol>
  <li>f(x,a,b,c,...)= expr_in_said_variables<br> For example: f(x,y)=3*x^2+4*x*y+8</li>
  <li>f = @(x,a,b,c,...)expr_in_said_variables<br> For example: f(x,y)=3*x^2+4*x*y+8</li>  
</ol>

Your defined functions are volatile and will be forgotten once the current parser session is over. The only way to have the parser remember them always is to introduce some form of persistence.

So for instance, you could pass the following to a MathExpression constructor:

f(x)=sin(x)+cos(x-1)<br>
Then do: f(2)....the parser automatically calculates sin(2)+cos(2-1) behind the scenes.






<b>Differential Calculus</b><br>

<b>ParserNG</b> makes differentiating Math Expressions really easy.
ParserNG uses its very own implementation of a differentiator.

For








