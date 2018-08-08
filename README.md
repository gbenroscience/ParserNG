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
at the console



