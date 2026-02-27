## Logical Calculus

The logical expressions in math engine have theirs intentional limitations. Thus allmighty logical expression parser was added around individually evaluated Mathematical expressions which results can be later compared, and logically grouped.  The simplest way to evaluate an logical  expression in **ParserNG** is to use the <code>LogicalExpression</code> class.
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
VALUES_PNG="1 8 5 2" java -jar target/parser-ng-0.2.2.jar -e "avg(..L{MN/2})*1.1-MN <  L0 | (L1+L{MN-1})*1.3 + MN<  L0" -v
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

See [jenkins-report-generic-chart-column](https://github.com/judovana/jenkins-report-generic-chart-column#most-common-expressions) as real-world user of `ExpandingParser`
