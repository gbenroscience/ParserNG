/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.Serializer;
import com.github.gbenroscience.util.VariableManager;
import com.github.gbenroscience.interfaces.Savable;
import com.github.gbenroscience.interfaces.Solvable;
import java.util.ArrayDeque;
import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.math.Main;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.Help;
import com.github.gbenroscience.parser.methods.Method;

import com.github.gbenroscience.math.Maths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.InputMismatchException;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import static com.github.gbenroscience.parser.Variable.*;
import static com.github.gbenroscience.parser.Number.*;
import static com.github.gbenroscience.parser.Operator.*;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.benchmarks.Gemini;
import com.github.gbenroscience.parser.benchmarks.Gemini2;
import com.github.gbenroscience.parser.benchmarks.GrokMBP;
import com.github.gbenroscience.parser.methods.MethodRegistry;
import java.util.Stack;

/**
 *
 * This class models a mathematical parser. It is designed to handle div
 * operators, methods(inbuilt and user-defined.)
 *
 * The parser is designed to work with a VariableManager and a FunctionManager.
 * To work with expressions that contain variables,the parser looks up the value
 * of variables stored in its VariableManager and sets those values in the
 * expression. To work with expressions containing user defined functions, the
 * parser looks up the underlying expression of the function in the
 * FunctionManager and employs that value in evaluating the function.
 *
 * <p style="font-weight:'bold';color:'red'">
 * NOTE: The parser operation is divided into: Step 1. Expression
 * Processing...This step takes time. Step 2. Expression Evaluation....Is an
 * high speed one.
 *
 * For expressions that contain either user defined functions or statistical
 * functions however, The second part is self-referentially mixed with the first
 * and so this gives rise to a problem. For iterative processes, the parser only
 * needs parse (Step 1.)the expression once(which takes the bulk of the time)
 * and then it can evaluate it many times over iteratively in a loop. Step 2 is
 * an high speed one. But if the expression contains statistical functions or
 * user defined ones, the self-referential processes modify the scanner output
 * and so this scanner output cannot be reliably referred to later on by
 * iterative processes or any process that seeks to reuse the scanner's output.
 *
 * </p>
 *
 * @author GBENRO
 */
public class MathExpression implements Savable, Solvable {

    public Parser_Result parser_Result = Parser_Result.VALID;
    //determines the mode in which trig operations will be carried out on numbers.if DRG==0,it is done in degrees
//if DRG==1, it is done in radians and if it is 2, it is done in grads.
    private DRG_MODE DRG = Declarations.degGradRadFromVariable();
    public static String lastResult = "0.0";
    private ArrayList<String> whitespaceremover = new ArrayList<>();//used to remove white spaces from the ArrayList
    /**
     * The expression to evaluate.
     */
    private String expression;
    protected boolean correctFunction = true;//checks if the function is valid.
    protected int noOfListReturningOperators;
    protected List<String> scanner = new ArrayList<>();//the ArrayList that stores the scanner input function
    private boolean optimizable;

    protected boolean hasListReturningOperators;
    private boolean hasNumberReturningStatsOperators;
    private boolean hasPlusOrMinusOperators;
    private boolean hasMulOrDivOperators;
    private boolean hasPowerOperators;
    private boolean hasPostNumberOperators;
    private boolean hasPreNumberOperators;
    private boolean hasRemainderOperators;
    private boolean hasPermOrCombOperators;
    private boolean hasLogicOperators;

    private ExpressionSolver expressionSolver = new ExpressionSolver();
    /**
     * If set to true, MathExpression objects will automatically initialize
     * undeclared variables to zero and store them.
     */
    private static boolean autoInitOn;
    /**
     * <b><i>
     * Set this attribute to true, if and only if the input into this system is
     * meant to initialize or change the value of variable data alone and is not
     * calculating a standalone math expression. For example, var a =2;var
     * b=3;var c=cos(a-b);f(x)=3*x^2; d=3cos(c);
     * </i></b>
     * <b><strong>where c has been declared before!</b></strong>
     */
    private boolean hasFunctionOrVariableInitStatement;

    /**
     * The VariableManager object that allows an object of this class to
     * remember its variables.
     */
    private VariableManager variableManager;

    // A simple pre-allocated array of results to act as a stack
    private final EvalResult[] pool = new EvalResult[64];
    private int poolPointer = 0;

    /**
     * The kind of output returned by the parser.
     */
    TYPE returnType = TYPE.NUMBER;

    // Token kinds
    public static final int NUMBER = 0, OPERATOR = 1, FUNCTION = 2, METHOD = 3, LPAREN = 4, RPAREN = 5;

    // Precedence levels
    private static final int PREC_POSTFIX = 5;  // !, ², ³, √, ³√, -¹
    private static final int PREC_POWER = 4;    // ^
    private static final int PREC_MULDIV = 3;   // *, /, %, Р, Č
    private static final int PREC_ADDSUB = 2;   // +, -
    private static final int PREC_UNARY = 100;  // Unary minus
    private Token[] cachedPostfix = null;  // Cache the compiled postfix

    /**
     * Sometimes, after evaluation the evaluation list which is a local
     * variable, is reduced to a function name(or other object as time goes on)
     * instead of a number of other list. The parser unfortunately will not
     * return this variable, but instead returns the data which it refers
     * to..e.g a {@link Matrix} function or other. But we atimes need that
     * function name...So we cache this value here.
     */
    private String returnObjectName;
    public static final String SYNTAX_ERROR = "SYNTAX ERROR";

    // Updated Token class (from your provided)
    static class Token {

        public static final int NUMBER = 0, OPERATOR = 1, FUNCTION = 2, METHOD = 3, LPAREN = 4, RPAREN = 5;
        public int kind;
        public double value;
        public String name; // REQUIRED for functions/methods/variables
        public int id;
        public char opChar;
        public int precedence;
        public boolean isRightAssoc;
        public boolean isPostfix;
        public int arity;

        // Constructor for Numbers
        public Token(double value) {
            this.kind = NUMBER;
            this.value = value;
            this.precedence = -1;
            this.isRightAssoc = false;
        }

        // Constructor for Operators
        public Token(char opChar, int precedence, boolean isRightAssoc, boolean isPostfix) {
            this.kind = OPERATOR;
            this.opChar = opChar;
            this.precedence = precedence;
            this.isRightAssoc = isRightAssoc;
            this.isPostfix = isPostfix;
            this.arity = isPostfix ? 1 : 2;
        }

        // Constructor for Functions/Methods
        public Token(int kind, String name, int arity, int id) {
            this.kind = kind;
            this.name = name;
            this.arity = arity;
            this.id = id;
            this.precedence = PREC_POSTFIX + 1;  // Bind tighter than postfix
            this.isRightAssoc = false;
        }

        // Constructor for Parens
        public Token(int kind) {
            this.kind = kind;
            this.precedence = -1;
            this.isRightAssoc = false;
        }

        // Helper to get precedence for opChar
        public static int getPrec(char op) {
            switch (op) {
                case '!':
                case '²':
                case '³':
                case '√':
                case 'R':  // Internal for ³√
                    return PREC_POSTFIX;
                case '^':
                    return PREC_POWER;
                case '*':
                case '/':
                case '%':
                case 'Р':
                case 'Č':
                    return PREC_MULDIV;
                case '+':
                case '-':
                    return PREC_ADDSUB;
                default:
                    return 0;
            }
        }

        public static boolean isRightAssociative(char op) {
            return op == '^';
        }

        public String toJsonString() {
            return "{\n"
                    + "\"kind\": " + kind + ",\n"
                    + "\"value\": " + value + ",\n"
                    + "\"name\": " + name + ",\n"
                    + "\"id\": " + id + ",\n"
                    + "\"opChar\": " + opChar + ",\n"
                    + "\"precedence\": " + precedence + ",\n"
                    + "\"isRightAssoc\": " + isRightAssoc + ",\n"
                    + "\"isPostfix\": " + isPostfix + ",\n"
                    + "\"arity\": " + arity + "\n"
                    + "}\n";
        }

        @Override
        public String toString() {
            return "{"
                    + "\"kind\": " + kind + ","
                    + "\"value\": " + value + ","
                    + "\"name\": " + name + ","
                    + "\"id\": " + id + ","
                    + "\"opChar\": " + opChar + ","
                    + "\"precedence\": " + precedence + ","
                    + "\"isRightAssoc\": " + isRightAssoc + ","
                    + "\"isPostfix\": " + isPostfix + ","
                    + "\"arity\": " + arity
                    + "}";
        }

    }

    /**
     * default no argument constructor for class MathExpression. It creates a
     * function which has the value 0;
     *
     */
    public MathExpression() {
        this("(0.0)");
    }

    /**
     *
     * @param input The function to be evaluated. The general format contains
     * variable, constant and function declarations for variables, constants and
     * functions that are not yet initialized, assignment expressions for those
     * that have been initialized and then an expression to evaluate. e.g. x =
     * -12; y =x+1/12; const x1,x2,x3=10; z =sin(3x-1)+2.98cos(4x);cos(3x+12);
     * The last expression is a function to be evaluated and it is always
     * without any equals sign and may or may not end with a semicolon.
     *
     */
    public MathExpression(String input) {
        this(input, new VariableManager());
    }//end constructor MathExpression

    public MathExpression(String input, VariableManager variableManager) {
        for (int i = 0; i < 64; i++) {
            pool[i] = new EvalResult();
        }
        setAutoInitOn(false);
        this.variableManager = variableManager;

        Scanner cs = new Scanner(STRING.purifier(input), false, VariableManager.endOfLine);

        List<String> scanned = cs.scan();

        String mathExpr = null;
        int exprCount = 0;

        for (String code : scanned) {
            if (code.contains("=")) {
                boolean success = Function.assignObject(code + ";");
                if (!success) {
                    correctFunction = success;
                    parser_Result = Parser_Result.SYNTAX_ERROR;
                    input = null;
                    return;
                }
            } else {
                mathExpr = "(" + code + ")";
                ++exprCount;
            }
        }

        if (mathExpr != null && !mathExpr.isEmpty() && exprCount == 1) {
            setExpression(mathExpr);
        }//end if
        else {
            setExpression("(0.0)");
        }

    }

    public String getExpression() {
        return expression;
    }

    /**
     * @param expression The expression
     */
    public final void setExpression(String expression) {
        if (!expression.equals(this.expression)) {
            scanner.clear();
            this.cachedPostfix = null;  // Force recompile
            setCorrectFunction(true);
            this.expression = expression;
            initializing(expression);
        } else {
            this.expression = expression;
        }
    }

    /**
     *
     * @return true if this object has been scanned and is found valid. In this
     * state, objects of this class are optimized to run at very high speeds.
     */
    public boolean isScannedAndOptimized() {
        try {
            return !scanner.isEmpty() && correctFunction && this != null;
        }//end try
        catch (NullPointerException nol) {
            return false;
        }//end catch
    }//end method

    public static void setAutoInitOn(boolean autoInitOn) {
        MathExpression.autoInitOn = autoInitOn;
    }

    public static boolean isAutoInitOn() {
        return autoInitOn;
    }

    private void initializing(String expression) {

        setCorrectFunction(true);
        setHasListReturningOperators(false);
        setNoOfListReturningOperators(0);
        whitespaceremover.add("");
        //Scanner operation

        MathScanner opScanner = new MathScanner(expression);
        scanner = opScanner.scanner(variableManager);

        correctFunction = opScanner.isRunnable();

        parser_Result = opScanner.parser_Result;
        if (parser_Result == Parser_Result.VALID) {

            statsVerifier();
            codeModifier();

            removeCommas();

            mapBrackets();

            functionComponentsAssociation();

        }//end if

    }//end method initializing(args)

    private void removeCommas() {
        List<String> commaList = new ArrayList<>();
        commaList.add(",");
        commaList.add("");
        scanner.removeAll(commaList);
    }

    /**
     *
     * @return the DRG value:0 for degrees, 1 for rads, 2 for grads
     */
    public DRG_MODE getDRG() {
        return DRG;
    }

    /**
     * sets the DRG property
     *
     * @param DRG
     */
    public void setDRG(DRG_MODE DRG) {
        this.DRG = DRG;
    }

    public void setDRG(int mode) {
        switch (mode) {
            case 0:
                this.DRG = DRG_MODE.DEG;
                break;
            case 1:
                this.DRG = DRG_MODE.RAD;
                break;
            case 2:
                this.DRG = DRG_MODE.GRAD;
                break;
            default:
                break;
        }
    }

    /**
     *
     * @return true if the input can be evaluated.
     */
    public boolean isCorrectFunction() {
        return correctFunction;
    }

    /**
     *
     * @param correctFunction sets if the input is valid and can be evaluated or
     * not.
     */
    public void setCorrectFunction(boolean correctFunction) {
        this.correctFunction = correctFunction;
    }

    /**
     *
     * @return the number of list returning operators found in the input.
     */
    public int getNoOfListReturningOperators() {
        return noOfListReturningOperators;
    }

    /**
     *
     * @param noOfListReturningOperators sets the number of list returning
     * operators found in the input.
     */
    public void setNoOfListReturningOperators(int noOfListReturningOperators) {
        this.noOfListReturningOperators = noOfListReturningOperators;
    }

    /**
     *
     * @return the ArrayList object that the input is scanned into.
     */
    public List<String> getScanner() {
        return scanner;
    }

    /**
     *
     * @param scanner sets the ArrayList object that the input is scanned into.
     */
    public void setScanner(ArrayList<String> scanner) {
        this.scanner = scanner;
    }

    /**
     *
     * @return the white space remover ArrayList object.
     */
    public ArrayList<String> getWhitespaceremover() {
        return whitespaceremover;
    }

    /**
     *
     * @param whitespaceremover sets the white space remover ArrayList object.
     */
    public void setWhitespaceremover(ArrayList<String> whitespaceremover) {
        this.whitespaceremover = whitespaceremover;
    }

    /**
     *
     * @return true if there are list-returning operators
     */
    public boolean isHasListReturningOperators() {
        return hasListReturningOperators;
    }

    /**
     *
     * @param hasListReturningOperators sets the number of list returning
     * operators.
     */
    public void setHasListReturningOperators(boolean hasListReturningOperators) {
        this.hasListReturningOperators = hasListReturningOperators;
    }

    /**
     *
     * @param optimizable sets whether this input can be optimized.
     */
    public void setOptimizable(boolean optimizable) {
        this.optimizable = optimizable;
    }

    /**
     *
     * @return whether or not this input can be optimized
     */
    public boolean isOptimizable() {
        return optimizable;
    }

    /**
     *
     * @param lastResult sets the last answer gotten by this parser
     */
    public static void setLastResult(String lastResult) {
        MathExpression.lastResult = lastResult;
    }

    /**
     *
     * @return the last answer calculated by this tool
     */
    public static String getLastResult() {
        return lastResult;
    }

    /**
     *
     * @param hasPreNumberOperators sets whether the input has pre-number
     * operators or not
     */
    public void setHasPreNumberOperators(boolean hasPreNumberOperators) {
        this.hasPreNumberOperators = hasPreNumberOperators;
    }

    /**
     *
     * @return true if the input has pre number operators
     */
    public boolean isHasPreNumberOperators() {
        return hasPreNumberOperators;
    }

    /**
     *
     * @param hasLogicOperators sets whether the input has logic operators or
     * not.
     */
    public void setHasLogicOperators(boolean hasLogicOperators) {
        this.hasLogicOperators = hasLogicOperators;
    }

    /**
     *
     * @return true if the input has logic operators
     */
    public boolean isHasLogicOperators() {
        return hasLogicOperators;
    }

    /**
     *
     * @param hasPostNumberOperators sets whether the input has post number
     * operators
     */
    public void setHasPostNumberOperators(boolean hasPostNumberOperators) {
        this.hasPostNumberOperators = hasPostNumberOperators;
    }

    /**
     *
     * @return true if post number operators like factorial, inverse e.t.c
     */
    public boolean isHasPostNumberOperators() {
        return hasPostNumberOperators;
    }

    /**
     *
     * @param hasPowerOperators sets whether or not the input has the power
     * operator
     */
    public void setHasPowerOperators(boolean hasPowerOperators) {
        this.hasPowerOperators = hasPowerOperators;
    }

    /**
     *
     * @return true if the input has the power operator
     */
    public boolean isHasPowerOperators() {
        return hasPowerOperators;
    }

    /**
     *
     * @param hasMulOrDivOperators sets whether the input has multiplication or
     * division operators
     */
    public void setHasMulOrDivOperators(boolean hasMulOrDivOperators) {
        this.hasMulOrDivOperators = hasMulOrDivOperators;
    }

    /**
     *
     * @return true if the input has multiplication or division operators
     */
    public boolean isHasMulOrDivOperators() {
        return hasMulOrDivOperators;
    }

    /**
     * Sometimes, after evaluation the evaluation list which is a local
     * variable, is reduced to a function name(or other object as time goes on)
     * instead of a number of other list. The parser unfortunately will not
     * return this variable, but instead returns the data which it refers
     * to..e.g a {@link Matrix} function or other. But we atimes need that
     * function name...So we cache this value here.
     */
    public String getReturnObjectName() {
        return returnObjectName;
    }

    /**
     *
     * @param hasPlusOrMinusOperators sets whether or not the input contains
     * plus or minus operators
     */
    public void setHasPlusOrMinusOperators(boolean hasPlusOrMinusOperators) {
        this.hasPlusOrMinusOperators = hasPlusOrMinusOperators;
    }

    /**
     *
     * @return true if plus or minus operators are found in the input
     */
    public boolean isHasPlusOrMinusOperators() {
        return hasPlusOrMinusOperators;
    }

    /**
     *
     * @param hasRemainderOperators sets whether or not remainder operators are
     * found in the input
     */
    public void setHasRemainderOperators(boolean hasRemainderOperators) {
        this.hasRemainderOperators = hasRemainderOperators;
    }

    /**
     *
     * @return true if remainder operators are found in the input
     */
    public boolean isHasRemainderOperators() {
        return hasRemainderOperators;
    }

    /**
     *
     * @param hasPermOrCombOperators sets whether permutation and combination
     * operators are found in the input
     */
    public void setHasPermOrCombOperators(boolean hasPermOrCombOperators) {
        this.hasPermOrCombOperators = hasPermOrCombOperators;
    }

    /**
     *
     * @return true if permutation and combination operators are found in the
     * input
     */
    public boolean isHasPermOrCombOperators() {
        return hasPermOrCombOperators;
    }

    /**
     *
     * @param hasNumberReturningStatsOperators sets whether or not the input
     * contains a data set that will evaluate to a number
     */
    public void setHasNumberReturningStatsOperators(boolean hasNumberReturningStatsOperators) {
        this.hasNumberReturningStatsOperators = hasNumberReturningStatsOperators;
    }

    /**
     *
     * @return true if the input contains a data set that will evaluate to a
     * number
     */
    public boolean isHasNumberReturningStatsOperators() {
        return hasNumberReturningStatsOperators;
    }

    public void setVariableHandlerOnly(boolean variableHandlerOnly) {
        this.hasFunctionOrVariableInitStatement = variableHandlerOnly;
    }

    public boolean isVariableHandlerOnly() {
        return hasFunctionOrVariableInitStatement;
    }

    public VariableManager getVariableManager() {
        return variableManager;
    }

    public void setVariableManager(VariableManager variableManager) {
        this.variableManager = variableManager;
    }

    /**
     *
     * @return an ArrayList object containing all Variable objects found in the
     * current input expression. This is only a subset of all Variable objects
     * used in the workspace of operation of this MathExpression object.
     */
    public ArrayList<Variable> getVars() {

        ArrayList<Variable> usedVars = new ArrayList<>();

        for (int i = 0; i < scanner.size(); i++) {
            if (isVariableString(scanner.get(i)) && !isOpeningBracket(scanner.get(i + 1))) {
                String str = scanner.get(i);
                Variable v = VariableManager.lookUp(str);
                //Variable does not exist
                if (v == null) {
                    setCorrectFunction(false);
                    throw new NullPointerException("Variable " + str + " Was Never Initialized!!");
                } //Variable exists
                else {
                    //  if var is in workspace but not yet recognized in this expression, add to usedVars
                    if (!usedVars.contains(v)) {
                        usedVars.add(v);
                    }
                }
            }//end if

        }//end for loop
        return usedVars;

    }

    public void statsVerifier() {
        scanner.removeAll(whitespaceremover);

        //determine the presence of list returning statistical operators
        for (int i = 0; i < scanner.size(); i++) {
            if (Method.isListReturningStatsMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                noOfListReturningOperators++;
            }//end if

        }//end for

        correctFunction = ListReturningStatsMethod.validateFunction(this.scanner);
        parser_Result = correctFunction ? Parser_Result.VALID : Parser_Result.SYNTAX_ERROR;
        //processLogger.writeLog(ListReturningStatsMethod.getErrorMessage());

        if (noOfListReturningOperators > 0 && correctFunction) {
            setHasListReturningOperators(true);
            //disable usage of multiple list operators in function here by un-commenting the statement below
            // correctFunction=false;

            if (isHasListReturningOperators()) {

                scanner.remove(0);//temporarily remove the starting brackets
                scanner.remove(scanner.size() - 1);//temporarily remove the ending brackets

                for (int i = 0; i < scanner.size(); i++) {
                    try {
                        if (Method.isListReturningStatsMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                            if (isBinaryOperator(scanner.get(i - 1))) {
                                //processLogger.writeLog("Invalid Association Discovered For: \""+scanner.get(i-1)+"\" And \""+scanner.get(i)+"\".\n");
                                correctFunction = false;
                                break;
                            }
                            if (isOpeningBracket(scanner.get(i - 1)) && !Method.isNumberReturningStatsMethod(scanner.get(i - 2))
                                    && !Method.isListReturningStatsMethod(scanner.get(i - 2))) {
                                //processLogger.writeLog("Invalid Association Discovered For: \"(\" And "+scanner.get(i-2)+" And \""+scanner.get(i-1)+"\" And \""+scanner.get(i)+"\"\n ");
                                correctFunction = false;
                                break;
                            }

                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                    }

                }//end for

                scanner.add(0, "(");
                scanner.add(")");

                if (!correctFunction) {
                    parser_Result = Parser_Result.SYNTAX_ERROR;
                    //processLogger.writeLog("Verifier discovers invalid association between data set returning operators:");
                }
            }//end if isHasListReturningOperator

        }

    }//end method statsVerifier

    /**
     * The method establishes meaning to some shorthand techniques in math that
     * the average mathematician might expect to see in a math device.
     * :e.g(3+4)(1+2) will become (3+4)*(1+2).It is essentially a stage that
     * generates code.
     */
    public void codeModifier() {

        if (correctFunction) {
            StringBuilder utility = new StringBuilder();

            /**
             * This stage serves for negative number detection. Prior to this
             * stage, all numbers are seen as positive ones. For example: turns
             * [12,/,-,5] to [12,/,-5].
             *
             * It also counts the number of list-returning operators in the
             * system.
             */
            for (int i = 0; i < scanner.size(); i++) {
                try {
                    if ((isBinaryOperator(scanner.get(i)) || Method.isUnaryPreOperatorORDefinedMethod(scanner.get(i))
                            || isOpeningBracket(scanner.get(i))
                            || isLogicOperator(scanner.get(i)) || isAssignmentOperator(scanner.get(i))
                            || isComma(scanner.get(i)) || Method.isStatsMethod(scanner.get(i)))
                            && Operator.isPlusOrMinus(scanner.get(i + 1)) && isNumber(scanner.get(i + 2))) {
                        utility.append(scanner.get(i + 1));
                        utility.append(scanner.get(i + 2));
                        scanner.set(i + 1, utility.toString());
                        scanner.set(i + 2, "");
                        utility.delete(0, utility.length());//clear the builder.
                    }//end if

                }//end try
                catch (IndexOutOfBoundsException ind) {
                }//end catch

            }//end for

            scanner.removeAll(whitespaceremover);

        } else if (!correctFunction) {
            //processLogger.writeLog("Beginning Parser Shutdown Tasks Due To Errors In User Input.");
        }

    }//end method codeModifier

    /**
     *
     * @param scanner The ArrayList object that holds the string values in the
     * scanned function.
     * @return a Bracket array that holds related brackets pairs.
     */
    public static Bracket[] mapBrackets(List<String> scanner) {
        for (Iterator<String> it = scanner.iterator(); it.hasNext();) {
            if (" ".equals(it.next())) {
                it.remove();
            }
        }

        ArrayList<Bracket> bracs = new ArrayList<>();
        Deque<Integer> stack = new ArrayDeque<>();

        for (int i = 0; i < scanner.size(); i++) {
            String token = scanner.get(i);

            if ("(".equals(token)) {
                stack.push(i);
            } else if (")".equals(token)) {
                if (stack.isEmpty()) {
                    throw new InputMismatchException("Unmatched closing bracket at index " + i);
                }
                int openIndex = stack.pop();

                Bracket openBrac = new Bracket("(");
                Bracket closeBrac = new Bracket(")");
                openBrac.setIndex(openIndex);
                closeBrac.setIndex(i);
                openBrac.setComplement(closeBrac);
                closeBrac.setComplement(openBrac);

                bracs.add(openBrac);
                bracs.add(closeBrac);
            }
        }

        if (!stack.isEmpty()) {
            throw new InputMismatchException("Unmatched opening bracket(s) remain");
        }

        return bracs.toArray(new Bracket[0]);
    }

    /**
     * Method mapBrackets goes over an input equation and maps all positions
     * that have corresponding brackets
     */
    public void mapBrackets() {
        try {
            mapBrackets(scanner);
        }//end method//end method
        catch (InputMismatchException ime) {
            parser_Result = Parser_Result.PARENTHESES_ERROR;
            setCorrectFunction(false);
            scanner.clear();
        }//end catch

    }//end method

    /**
     *
     * method functionComponentsAssociation does final adjustments to the
     * scanner function e.g it will check for errors in operator combination in
     * the scanner function and so on
     */
    public void functionComponentsAssociation() {

        if (correctFunction) {
            scanner.removeAll(whitespaceremover);//remove white spaces that may result from past parser actions
//check for good combinations of operators and numbers and dis-allow any other.

            int sz = scanner.size();
            for (int i = 0; i < scanner.size(); i++) {
//check for the various valid arrangements for all members of the function.
                String token = scanner.get(i);
                //Variables
                if (isVariableString(scanner.get(i)) && !Method.isUserDefinedFunction(token) && !Method.isDefinedMethod(token)) {
                    try {
                        //specify valid tokens that can come before a variable
                        if (i - 1 >= 0 && !isOpeningBracket(scanner.get(i - 1))
                                && !isLogicOperator(scanner.get(i - 1)) && !isUnaryPreOperator(scanner.get(i - 1))
                                && !isBinaryOperator(scanner.get(i - 1)) && !isAssignmentOperator(scanner.get(i - 1)) && !isNumber(scanner.get(i - 1))
                                && !isVariableString(scanner.get(i - 1)) && !isComma(scanner.get(i - 1))) {
                            //processLogger.writeLog("ParserNG Does Not Allow "+expression+" To Combine The MathExpression Members \""+scanner.get(i-1)+"\" And \""+scanner.get(i)+"\"\n");
                            correctFunction = false;
                            scanner.clear();
                            break;
                        }//end if
                        //specify valid tokens that can come after a variable
                        if (i + 1 < sz && !isBracket(scanner.get(i + 1)) && !isBinaryOperator(scanner.get(i + 1))
                                && !isUnaryPostOperator(scanner.get(i + 1)) && !Method.isNumberReturningStatsMethod(scanner.get(i + 1))
                                && !isLogicOperator(scanner.get(i + 1)) && !isAssignmentOperator(scanner.get(i + 1))
                                && !isUnaryPreOperator(scanner.get(i + 1)) && !Method.isNumberReturningStatsMethod(scanner.get(i + 1))
                                && !Method.isLogToAnyBase(scanner.get(i + 1)) && !Method.isAntiLogToAnyBase(scanner.get(i + 1)) && !isNumber(scanner.get(i + 1))
                                && !isVariableString(scanner.get(i + 1)) && !isComma(scanner.get(i + 1))) {
                            //processLogger.writeLog("ParserNG Does Not Allow "+expression+" To Combine The MathExpression Members \""+scanner.get(i)+"\" And \""+scanner.get(i+1)+"\"PLUS As You Have Done.\n");
                            correctFunction = false;
                            scanner.clear();
                            break;
                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                        ind.printStackTrace();
                    }//end catch
                }//end else if

            }//end for
            if (correctFunction) {
                setCorrectFunction(validateAll(scanner));
            }

            if (correctFunction) {
                scanner.removeAll(whitespaceremover);
            }//end if
            else {
                scanner.clear();
                parser_Result = Parser_Result.SYNTAX_ERROR;
            }//end else

        }//end if
        else {
            scanner.clear();
            parser_Result = Parser_Result.SYNTAX_ERROR;
        }

    }//end method functionComponentAssociation

    /**
     * An important process that must occur before the function is solved.
     * Variables must be replaced by their values. The method checks the
     * variable store and assumes that between function input time and function
     * solution time, the user would have modified the value attribute stored in
     * the variables. So it gets the values there and fixes them in the
     * appropriate points in the equation. Ensure that no shift has occurred in
     * Variable object position during the time that the record was taken and
     * the time when the position is about to be referenced.
     *
     * @param scan the data it is to process
     */
    public void setVariableValuesInFunction(List<String> scan) {

        int sz = scan.size();
        for (int i = 0; i < sz; i++) {
            String varName = scan.get(i);

            try {
                if (i + 1 < sz) {//a next token exists (at i+1) so check the next token
                    if (isVariableString(scan.get(i)) && !isOpeningBracket(scan.get(i + 1))) {
                        Variable v = VariableManager.lookUp(varName);
                        if (v != null) {
                            scan.set(i, String.valueOf(v.getValue()));
                        }

                    }//end if
                } else {//no next token exists
                    if (isVariableString(scan.get(i))) {
                        Variable v = VariableManager.lookUp(varName);
                        if (v != null) {
                            scan.set(i, String.valueOf(v.getValue()));
                        }

                    }//end if
                }

            }//end try
            catch (IndexOutOfBoundsException ind) {
            }//end catch

        }//end for

    }// end method

    /**
     *
     * @param name The name of the variable or constant.
     * @return the value of the named variable or constant
     * @throws NullPointerException if a Variable object that has that name id
     * not found.
     */
    public double getValue(String name) throws NullPointerException {
        Variable var = variableManager.lookUp(name);
        return var == null ? Double.NaN : var.getValue();
    }

    /**
     *
     * @param name The name of the variable or constant.
     * @param value The value to set to the variable
     * @throws NullPointerException if a Variable object that has that name id
     * not found.
     */
    public void setValue(String name, double value) throws NullPointerException, NumberFormatException {
        Variable v = VariableManager.lookUp(name);
        if (v != null) {
            v.setValue(value);
        }

    }

    /**
     * Utility method used to dynamically change the indices of brackets in the
     * governing bracket map of the scanner function.
     *
     * When method solve is performing its task,it uses the bracket ArrayList to
     * know the next portion to evaluate in the scanner function. However when
     * it evaluates this portion,the size of the scanner function
     * changes(reduces) and this means that the bracket ArrayList is no longer
     * relevant in determining the next point to evaluate in the scanner
     * function. So this method is called to automatically re-configure the
     * bracket ArrayList so that it can continue to be relevant to the solution
     * process.
     *
     * The process occurs during method solve and is a sort of automatic
     * compression in response to the solution process which brings about
     * changes in the number of elements in the scanner function.
     *
     * LOGIC: Bearing in mind the fact that each bracket stores its current
     * index in the MathExpression object's ArrayList,scanner; we traverse the
     * bracket list starting from the bracket from which evaluation is to begin
     * in the client method solve and looping on to the end of the bracket list.
     * We check for the current index (say A) ( as in its position in the
     * ArrayList object) stored by each bracket we meet during this scan and
     * compare it with the index (say B)stored by the bracket from which we
     * began the loop the last time this method was called. This is the bracket
     * at index startPosition-2. If A&lt;B then we apply the decrement or
     * shrinking factor to it.Else we continue the scan.
     *
     *
     * @param brac the Bracket store to modify
     * @param startPosition the index in the ArrayList where the modification is
     * to start
     * @param increment the amount by which each brackets index is to be
     * decreased
     * @param run will run this method if given the sign to do so.
     */
    protected void modifyBracketIndices(Bracket[] brac, int startPosition, int increment, boolean run) {
        if (run) {
            int valAtBracIndex = 0;
            int valAtLastEvaluatedBracIndex = 0;
            if (startPosition > 0) {
                valAtLastEvaluatedBracIndex = brac[startPosition - 2].getIndex();

                for (int i = startPosition; i < brac.length; i++) {
                    valAtBracIndex = brac[i].getIndex();
//values greater than the value stored by the brackets from which looping was started the
// last time represent the indices of brackets in the function list that the compression function
// will affect.So apply the decrement to them.
                    try {
                        if (valAtBracIndex > valAtLastEvaluatedBracIndex) {
                            brac[i].setIndex(brac[i].getIndex() + increment);
                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException indexErr) {
                    }//end catch
                }//end for

                ArrayList<Integer> arr = new ArrayList<Integer>();
                for (int i = 0; i < brac.length; i++) {
                    arr.add(brac[i].getIndex());
                }
                //displayIndicesStoredInBrackets();

            }
        }

    }//end method reduceBracketIndices
//(1+1+1+1+1+1+1+1+1+1)(1+1+1+1+1+1+1+1+1+1)(1+1+1+1+1+1+1+1+1+1)(1+1+1+1+1+1+1+1+1+1)(1+1+1+1+1+1+1+1+1+1)(1+1+1+1+1+1+1+1+1+1)

    /**
     * Display the indices of all brackets in the function,brackets pair by
     * brackets pair
     */
    /**
     *
     * @param scan The ArrayList object.
     * @return the string version of the ArrayList and removes the braces i.e.
     * []
     */
    protected String listToString(List<String> scan) {
        if (scan == null || scan.isEmpty()) {
            return "";
        }

        // Optimization 1: Pre-size the builder to avoid internal array copying
        // We estimate: avg string length (e.g., 5) + comma/space (2) * size
        StringBuilder sb = new StringBuilder(scan.size() * 7);

        // Optimization 2: Use a standard for-loop (slightly faster than Iterator in some JVMs)
        for (int i = 0; i < scan.size(); i++) {
            String s = scan.get(i);

            // Skip nulls or empties if necessary for your parser logic
            if (s != null && !s.isEmpty()) {
                if (sb.length() > 0) {
                    sb.append(", ");
                }
                sb.append(s);
            }
        }

        return sb.toString();
    }

    /**
     * [-6, -4, 2, 3, 4, 5, 5, 13, 62, 2024]
     *
     * @param text
     * @return
     */
    private List<String> readNumbersFromCommaSeparatedList(String text) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }

        // Pre-size ArrayList if possible to avoid resizing overhead
        List<String> numbers = new ArrayList<>(16);
        int len = text.length();
        int start = -1;

        for (int i = 0; i < len; i++) {
            char c = text.charAt(i);

            // Check if character is part of a valid number
            if ((c >= '0' && c <= '9') || c == '-' || c == '.'
                    || c == 'e' || c == 'E' || c == '+') {
                if (start == -1) {
                    start = i; // Mark the beginning of a number
                }
            } else {
                // We hit a delimiter (comma, space, brackets, etc.)
                if (start != -1) {
                    numbers.add(text.substring(start, i));
                    start = -1;
                }

                // Validation for '[' only at start
                if (c == '[' && i != 0) {
                    throw new InputMismatchException("Invalid character [ at index " + i);
                }
            }
        }

        // CRITICAL: Catch the last number if the string doesn't end with a delimiter
        if (start != -1) {
            // If the last char is ']', exclude it from the substring
            int end = text.charAt(len - 1) == ']' ? len - 1 : len;
            if (end > start) {
                numbers.add(text.substring(start, end));
            }
        }

        return numbers;
    }

    public void setReturnType(TYPE returnType) {
        this.returnType = returnType;
    }

    public TYPE getReturnType() {
        return returnType;
    }

    @Override
    public String solve() {
        if (expression.equalsIgnoreCase("(" + Declarations.HELP + ")")) {
            return Help.getHelp();
        }
        System.out.println("scannER: " + scanner);
        compileToPostfix();  // Compile once if not already done
        System.out.println("check postfix: " + Arrays.toString(cachedPostfix));
        return expressionSolver.evaluate(cachedPostfix).scalar + "";  // Just evaluate
    }//end method solve()

    protected List<String> solve(List<String> list) {
        return Arrays.asList(String.valueOf(GG.evaluate(list)));
    }

    // Your translate method (with small updates for Java 8)
    public Token translate(String s) {
        if (s == null || s.isEmpty()) {
            return null;
        }
        char c0 = s.charAt(0);
        int len = s.length();
        // 1. Identify Numbers
        // (Assuming isNumber() handles negative signs and decimals correctly)
        if (isNumber(s)) {
            return new Token(fastParseDouble(s));
        }

        // 2. Identify Brackets (Commas completely removed)
        if (len == 1) {
            switch (c0) {
                case '(':
                    return new Token(LPAREN);
                case ')':
                    return new Token(RPAREN);
            }
        }

        // 3. Identify Operators
        if (isOperator(s)) {
            boolean isPostfix = (c0 == '!' || c0 == '²' || c0 == '³');
            // Map "³√" to 'R' for fast switch
            char internalOp = (len > 1 && s.equals("³√")) ? 'R' : c0;
            return new Token(internalOp, Token.getPrec(internalOp), Token.isRightAssociative(internalOp), isPostfix);
        }

        // 4. Identify Functions (Built-in or User-Defined)
        if (FunctionManager.FUNCTIONS.containsKey(s)) {
            return new Token(FUNCTION, s, FunctionManager.getFunction(s).getArity(), -1);
        }

        // 5. Identify Methods
        if (Method.isInBuiltMethod(s)) {
            int methodId = MethodRegistry.getMethodID(s);
            return new Token(METHOD, s, -1, methodId); // Arity set during compile
        }

        // 6. Fallback: Treat as Variable/Constant
        Token t = new Token(0.0);
        t.name = s;
        t.kind = NUMBER;

        return t;

    }

    // Helper for isOperator (your custom ops)
    private boolean isOperator(String s) {
        if (s.length() == 1) {
            char c = s.charAt(0);
            return "+-*/%^√!²³ČР".indexOf(c) != -1;
        }
        return s.equals("³√");
    }

    private boolean isABinaryOperator(char opChar) {
        return opChar == '+' || opChar == '-' || opChar == '*'
                || opChar == '/' || opChar == '%' || opChar == '^'
                || opChar == 'Č' || opChar == 'Р';
    }

    public final void compileToPostfix() {
        if (cachedPostfix != null) {
            return;   // already compiled
        }
        Stack<Token> opStack = new Stack<>();
        Token[] postfix = new Token[scanner.size() * 2];
        int p = 0;

        int[] argCount = new int[64];
        int depth = 0;
        argCount[0] = 0;

        for (String s : scanner) {
            Token t = translate(s);
            if (t == null) {
                continue;
            }

            switch (t.kind) {
                case Token.NUMBER:
                    postfix[p++] = t;
                    argCount[depth]++;
                    break;

                case Token.FUNCTION:
                case Token.METHOD:
                    opStack.push(t);
                    argCount[++depth] = 0;           // start counting arguments for this function
                    break;

                case Token.LPAREN:
                    opStack.push(t);
                    break;

                case Token.RPAREN:
                    // Pop operators until matching '('
                    while (!opStack.isEmpty() && opStack.peek().kind != Token.LPAREN) {
                        postfix[p++] = opStack.pop();
                    }
                    if (!opStack.isEmpty()) {
                        opStack.pop(); // discard '('
                    }

                    // === KEY FIX: Check what this closing ) completed ===
                    if (!opStack.isEmpty()
                            && (opStack.peek().kind == Token.FUNCTION || opStack.peek().kind == Token.METHOD)) {

                        // This ) completed a function call
                        Token callable = opStack.pop();
                        callable.arity = argCount[depth];
                        postfix[p++] = callable;

                        depth--;
                        argCount[depth]++;        // one complete value for outer function
                    } else {
                        // This ) completed a grouped argument like (sin(3)) or (12)
                        // → it counts as ONE argument for the outer function
                        argCount[depth]++;
                    }
                    break;

                case Token.OPERATOR:
                    if (t.isPostfix) {
                        postfix[p++] = t;
                    } else {
                        while (!opStack.isEmpty() && opStack.peek().kind == Token.OPERATOR) {
                            Token top = opStack.peek();
                            if ((!t.isRightAssoc && t.precedence <= top.precedence)
                                    || (t.isRightAssoc && t.precedence < top.precedence)) {
                                postfix[p++] = opStack.pop();
                            } else {
                                break;
                            }
                        }
                        opStack.push(t);
                    }
                    break;
            }
        }

        // Flush remaining operators
        while (!opStack.isEmpty()) {
            postfix[p++] = opStack.pop();
        }

        // Trim and cache
        cachedPostfix = new Token[p];
        System.arraycopy(postfix, 0, cachedPostfix, 0, p);

        //System.out.println("cachedPostfix: " + Arrays.toString(cachedPostfix));
    }

    public final class ExpressionSolver {

        public EvalResult evaluate(Token[] postfix) {
            final EvalResult[] stack = new EvalResult[postfix.length];
            int ptr = -1;

            for (Token t : postfix) {
                switch (t.kind) {
                    case Token.NUMBER:
                        stack[++ptr] = getNextResult().wrap(t.name != null
                                ? VariableManager.getVariable(t.name).getValue() : t.value);
                        break;

                    case Token.OPERATOR:
                        if (t.isPostfix || t.opChar == '√' || t.opChar == 'R') {
                            // Unary: Modify the top of stack in-place
                            applyUnary(t.opChar, stack[ptr]);
                        } else {
                            // Binary: Pop 'b', modify 'a' in-place, reclaim 'b' from pool
                            double bVal = stack[ptr--].scalar;
                            applyBinary(t.opChar, stack[ptr], bVal);
                            poolPointer--;
                        }
                        break;

                    case Token.METHOD:
                    case Token.FUNCTION:
                        int arity = t.arity;
                        System.out.println("arity of " + t.name + " = " + t.arity);
                        EvalResult[] args = new EvalResult[arity];
                        for (int j = arity - 1; j >= 0; j--) {
                            args[j] = stack[ptr--];
                        }

                        // 2. CRITICAL: Move pointer back BEFORE execution.
                        // This allows the method to use the argument slots for its result.
                        poolPointer -= arity;
                        System.out.println("--------------------------------Method Input: method-name=" + t.name + ", type: " + t.kind + ", args=" + Arrays.toString(args));
                        EvalResult result = (t.kind == Token.METHOD)
                                ? MethodRegistry.getAction(t.id).execute(MathExpression.this, t.name, arity, args)
                                : FunctionManager.lookUp(t.name).calc(args);
                        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>Method output " + result.toString());
                        stack[++ptr] = result;
                        // 3. REMOVE the old release(arity) call here.
                        break;

                }
            }
            return stack[0];

        }

        private void applyUnary(char op, EvalResult res) {
            double val = res.scalar;
            switch (op) {
                case '√':
                    res.scalar = Math.sqrt(val);
                    break;
                case 'R': // Internal mapping for ³√
                    res.scalar = Math.cbrt(val);
                    break;
                case '!':
                    res.scalar = Maths.fact(val); // Using your Maths utility
                    break;
                case '²':
                    res.scalar = val * val;
                    break;
                case '³':
                    res.scalar = val * val * val;
                    break;
                case '-': // Unary negation (if supported by your scanner)
                    res.scalar = -val;
                    break;
                case 'i': // Reciprocal/Inverse x⁻¹
                    res.scalar = 1.0 / val;
                    break;
            }
            // Note: type remains TYPE_SCALAR, so no need to call wrap()
        }

        private void applyBinary(char op, EvalResult aRes, double b) {
            double a = aRes.scalar;
            switch (op) {
                case '+':
                    aRes.scalar = a + b;
                    break;
                case '-':
                    aRes.scalar = a - b;
                    break;
                case '*':
                    aRes.scalar = a * b;
                    break;
                case '/':
                    aRes.scalar = a / b;
                    break;
                case '%':
                    aRes.scalar = a % b;
                    break;
                case '^':
                    aRes.scalar = Math.pow(a, b);
                    break;
                case 'Č':
                    aRes.scalar = Maths.combination(a, b);
                    break;
                case 'Р':
                    aRes.scalar = Maths.permutation(a, b);
                    break;
            }
        }
    }

    public static final class EvalResult {

        public static final int TYPE_SCALAR = 0;
        public static final int TYPE_VECTOR = 1;
        public static final int TYPE_MATRICES = 2;
        public static final int TYPE_STRING = 3;

        public double scalar;      // For single numbers
        public double[] vector;    // For stats/lists
        public Matrix matrix;  // For matrices
        public int type;           // 0=Scalar, 1=Vector, 2=Matrix, 3=String
        /**
         * Use this to return string results, e.g. formulas, function results,
         * etc.
         */
        public String textRes;

        // Helper to reset the object for reuse
        public EvalResult wrap(double s) {
            this.scalar = s;
            this.type = TYPE_SCALAR;
            return this;
        }

        public EvalResult wrap(double[] v) {
            this.vector = v;
            this.type = TYPE_VECTOR;
            return this;
        }

        public EvalResult wrap(Matrix m) {
            this.matrix = m;
            this.type = TYPE_MATRICES;
            return this;
        }

        public EvalResult wrap(String s) {
            this.textRes = s;
            this.type = TYPE_STRING;
            return this;
        }

        private static String fastToString(double[] arr) {
            if (arr == null) {
                return "null";
            }
            // Pre-size estimate: arr length * average double length + separators
            StringBuilder sb = new StringBuilder(arr.length * 10);
            sb.append('[');
            for (int i = 0; i < arr.length; i++) {
                sb.append(arr[i]);
                if (i < arr.length - 1) {
                    sb.append(", ");
                }
            }
            return sb.append(']').toString();
        }

        @Override
        public String toString() {
            switch (type) {
                case TYPE_SCALAR:
                    return String.valueOf(scalar);
                case TYPE_VECTOR:
                    return fastToString(vector);
                case TYPE_MATRICES:
                    return matrix.toString();
                case TYPE_STRING:
                    return textRes;
                default:
                    return "0.0";
            }

        }

    }

    public EvalResult getNextResult() {
        return pool[poolPointer++];
    }

    public void release(int count) {
        poolPointer -= count;
    }

    public final class ExpressionSolver1 {

        public double evaluate(List<String> tokens) {
            final String[] ts = tokens.toArray(new String[0]);
            final int n = ts.length;

            final double[] valStack = new double[n];
            final int[] opStack = new int[n];
            int vIdx = -1;
            int oIdx = -1;

            for (int i = 0; i < n; i++) {
                final String s = ts[i];
                final char c0 = s.charAt(0);
                final int sLen = s.length();

                // 1. NUMERIC CHECK
                if ((c0 >= '0' && c0 <= '9') || (c0 == '-' && sLen > 1 && s.charAt(1) >= '0' && s.charAt(1) <= '9')) {
                    valStack[++vIdx] = fastParseDouble(s);
                    continue;
                }

                // 2. UNARY PREFIX (ROOTS)
                // We treat ³√ as a single internal char 'R' for the switch speed
                if (c0 == '√' || s.equals("³√")) {
                    opStack[++oIdx] = (c0 == '√') ? '√' : 'R';
                    continue;
                }

                // 3. UNARY POSTFIX (FACTORIAL/POWERS)
                if (c0 == '!' || c0 == '²' || c0 == '³' || (c0 == '-' && sLen > 1 && s.charAt(1) == '¹')) {
                    final double a = valStack[vIdx];
                    if (c0 == '!') {
                        valStack[vIdx] = Maths.fact(a);
                    } else if (c0 == '²') {
                        valStack[vIdx] = a * a;
                    } else if (c0 == '³') {
                        valStack[vIdx] = a * a * a;
                    } else {
                        valStack[vIdx] = 1.0 / a;
                    }
                    continue;
                }

                // 4. BINARY OPERATORS
                final int currentPrec = getPrec(c0);
                while (oIdx >= 0) {
                    final char topOp = (char) opStack[oIdx];
                    final int topPrec = getPrec(topOp);

                    // Right-associative logic for '^'
                    if (c0 == '^' ? topPrec > currentPrec : topPrec >= currentPrec) {
                        vIdx = applyOp((char) opStack[oIdx--], valStack, vIdx);
                    } else {
                        break;
                    }
                }
                opStack[++oIdx] = c0;
            }

            // Final Stack Reduction
            while (oIdx >= 0) {
                vIdx = applyOp((char) opStack[oIdx--], valStack, vIdx);
            }
            return valStack[0];
        }

        private int getPrec(char op) {
            switch (op) {
                case '+':
                case '-':
                    return 1;
                case '*':
                case '/':
                case '%':
                case 'Р':
                case 'Č':
                    return 2;
                case '^':
                    return 3;
                case '√':
                case 'R':
                    return 4; // Highest precedence for prefix roots
                default:
                    return 0;
            }
        }

        private int applyOp(char op, double[] valStack, int vIdx) {
            // Handle Unary Prefix first (takes only one operand)
            if (op == '√') {
                valStack[vIdx] = Math.sqrt(valStack[vIdx]);
                return vIdx;
            }
            if (op == 'R') { // Internal ID for ³√
                valStack[vIdx] = Math.cbrt(valStack[vIdx]);
                return vIdx;
            }

            // Handle Binary (takes two operands)
            double b = valStack[vIdx--];
            double a = valStack[vIdx];
            switch (op) {
                case '+':
                    valStack[vIdx] = a + b;
                    break;
                case '-':
                    valStack[vIdx] = a - b;
                    break;
                case '*':
                    valStack[vIdx] = a * b;
                    break;
                case '/':
                    valStack[vIdx] = a / b;
                    break;
                case '%':
                    valStack[vIdx] = a % b;
                    break;
                case '^':
                    valStack[vIdx] = Math.pow(a, b);
                    break;
                case 'Р':
                    valStack[vIdx] = Maths.fact(a) / Maths.fact(a - b);
                    break;
                case 'Č':
                    valStack[vIdx] = Maths.fact(a) / (Maths.fact(b) * Maths.fact(a - b));
                    break;
            }
            return vIdx;
        }

    }

    /**
     * used by the main parser solve to figure out SBP portions of a
     * multi-bracketed expression (MBP)
     *
     * @param list a list of scanner tokens of a maths expression
     * @return the solution to a SBP maths expression
     */
    protected List<String> solve1(List<String> list) {

//correct the anomaly: [ (,-,number....,)  ]
        //   turn it into: [ (,,-number........,)     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("(") && list.get(1).equals(MINUS) && isNumber(list.get(2))) {
            list.set(1, "");
            //if the number is negative,make it positive
            if (list.get(2).charAt(0) == MINUS.charAt(0)) {
                list.set(2, list.get(2).substring(1));
            } //if the number is positive,make it negative
            else {
                list.set(2, MINUS + list.get(2));
            }
        }
//Create a collection to serve as a garbage collector for the empty memory
//locations and other unwanted locations created in the processing collection
        ArrayList<String> garbage = new ArrayList<>();
//insert an empty string in it so that we can use it to remove empty spaces from the processing collection.
        garbage.add("");
        garbage.add("(");
        garbage.add(")");

        list.removeAll(garbage);

//solves the factorial component of the input|[²]|[³]|[-¹]²³-¹
        if (isHasPostNumberOperators()) {
            for (int i = 0; i < list.size(); i++) {
                try {

                    if (isFactorial(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, Maths.fact(list.get(i)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else

                    } else if (isSquare(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.parseDouble(list.get(i)), 2)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else
                    } else if (isCube(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.parseDouble(list.get(i)), 3)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else
                    } else if (isInverse(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(1 / Double.parseDouble(list.get(i))));
                            list.set(i, "");
                        } else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "0.0");
                            list.set(i, "");
                        }//end else
                    }
                }//end try
                catch (NumberFormatException numerror) {
                } catch (NullPointerException nullerror) {
                } catch (IndexOutOfBoundsException inderror) {
                }
            }//end for
            list.removeAll(garbage);
        }//end if

        if (isHasPowerOperators()) {

            /*Deals with powers.
            Handles the  primary power operator e.g in 3^sin3^4.This is necessary at this stage to dis-allow operations like sinA^Bfrom giving the result:(sinA)^B
             instead of sin(A^B).
             Also instructs the software to multiply any 2 numbers in consecutive positions in the list.
             This is important in distinguishing between functions such as sinAB and sinA*B.Note:sinAB=sin(A*B),while sinA*B=B*sinA.
             */
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (list.get(i).equals(POWER) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.parseDouble(list.get(i - 1)), Double.parseDouble(list.get(i + 1)))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                    }//end if
                    else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (Double.parseDouble(list.get(i + 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) == 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) < 1 && Double.parseDouble(list.get(i + 1)) > 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) < 1 && Double.parseDouble(list.get(i + 1)) == 0) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) < 1 && Double.parseDouble(list.get(i + 1)) < 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        if (Double.parseDouble(list.get(i - 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) == 1) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) < 1 && Double.parseDouble(list.get(i - 1)) > 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) < 1 && Double.parseDouble(list.get(i - 1)) == 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) < 1 && Double.parseDouble(list.get(i - 1)) < 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end else if
                    }//end else if
                    else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        list.set(i + 1, "Infinity");
                        list.set(i - 1, "");
                        list.set(i, "");
                    }

                }//end try
                catch (NumberFormatException numerror) {
                } catch (NullPointerException nullerror) {
                } catch (IndexOutOfBoundsException inderror) {
                }
            }//end for

            list.removeAll(garbage);

        }//end if
        //Handles the pre-number operators.

        if (isHasPreNumberOperators()) {
            for (int i = list.size() - 1; i >= 0; i--) {
                try {
                    if (!list.get(i + 1).equals("Infinity") && isNumber(list.get(i + 1))) {
                        if (list.get(i).equals(ROOT)) {
                            list.set(i, String.valueOf(Math.sqrt(Double.parseDouble(list.get(i + 1)))));
                            list.set(i + 1, "");
                        }//end if
                        if (list.get(i).equals(CUBE_ROOT)) {
                            list.set(i, String.valueOf(Math.cbrt(Double.parseDouble(list.get(i + 1)))));
                            list.set(i + 1, "");
                        }//end if
//add more pre-number functions here...
                    }//end if
                    else if (list.get(i + 1).equals("Infinity")) {
                        list.set(i, "Infinity");
                        list.set(i + 1, "");
                    }//end else if
                }//end try
                catch (NumberFormatException numerror) {
                } catch (NullPointerException nullerror) {
                } catch (IndexOutOfBoundsException inderror) {
                }
            }//end for
        }//end if

        list.removeAll(garbage);

        if (isHasPowerOperators()) {
            //do the in between operators

            /*Deals with powers.Handles the  primary power operator e.g in 3^sin3^4.This is necessary at this stage to dis-allow operations like sinA^Bfrom giving the result:(sinA)^B
             instead of sin(A^B).
             Also instructs the software to multiply any 2 numbers in consecutive positions in the vector.
             This is important in distinguishing between functions such as sinAB and sinA*B.Note:sinAB=sin(A*B),while sinA*B=B*sinA.
             */
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (list.get(i).equals(POWER) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.parseDouble(list.get(i - 1)), Double.parseDouble(list.get(i + 1)))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                    }//end if
                    else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (Double.parseDouble(list.get(i + 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) == 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) < 1 && Double.parseDouble(list.get(i + 1)) > 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) < 1 && Double.parseDouble(list.get(i + 1)) == 0) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i + 1)) < 1 && Double.parseDouble(list.get(i + 1)) < 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        if (Double.parseDouble(list.get(i - 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) == 1) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) < 1 && Double.parseDouble(list.get(i - 1)) > 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) < 1 && Double.parseDouble(list.get(i - 1)) == 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.parseDouble(list.get(i - 1)) < 1 && Double.parseDouble(list.get(i - 1)) < 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end else if
                    }//end else if
                    else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        list.set(i + 1, "Infinity");
                        list.set(i - 1, "");
                        list.set(i, "");
                    }

                }//end try
                catch (NumberFormatException numerror) {
                } catch (NullPointerException nullerror) {
                } catch (IndexOutOfBoundsException inderror) {
                }
            }//end for

            list.removeAll(garbage);

        }//end if

        list.removeAll(garbage);

        if (isHasPermOrCombOperators()) {
            //do the lower precedence in between operators

            for (int i = 0; i < list.size(); i++) {
                try {
                    if (list.get(i).equals(Operator.PERMUTATION)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.parseDouble(Maths.fact(list.get(i - 1))) / (Double.parseDouble(Maths.fact(String.valueOf(Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))))))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                        else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }

                    }//end if
                    else if (list.get(i).equals(Operator.COMBINATION)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.parseDouble(Maths.fact(list.get(i - 1))) / (Double.parseDouble(Maths.fact(String.valueOf(Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))))) * Double.parseDouble(Maths.fact(list.get(i + 1))))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                        else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }

                    }//end else if

                }//end try
                catch (NullPointerException nolan) {
                }//end catch
                catch (NumberFormatException numerr) {
                }//end catch
                catch (IndexOutOfBoundsException inderr) {
                }//end catch

            }//end for
            list.removeAll(garbage);
        }//end if
        boolean skip = false;
        if (isHasMulOrDivOperators() || isHasRemainderOperators() || isHasLogicOperators()) {
            for (int i = 0; i < list.size(); i++) {

                try {

                    if (list.get(i).equals(MULTIPLY)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.parseDouble(list.get(i - 1)) * Double.parseDouble(list.get(i + 1))));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end if
                    else if (list.get(i).equals(DIVIDE)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.parseDouble(list.get(i - 1)) / Double.parseDouble(list.get(i + 1))));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end else if
                    else if (list.get(i).equals(REMAINDER)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, String.valueOf(Double.parseDouble(list.get(i - 1)) % Double.parseDouble(list.get(i + 1))));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, list.get(i - 1));
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                            skip = true;
                        }

                    }//end else if
                }//end try
                catch (NullPointerException nolan) {
                }//end catch
                catch (NumberFormatException numerr) {
                }//end catch
                catch (IndexOutOfBoundsException inderr) {
                }//end catch

                if (!skip) {
                    try {
                        if (list.get(i).equals(EQUALS)) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))) == 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }
                        }
                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch

                    try {
                        if (list.get(i).equals(GREATER_THAN)) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))) > 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }
                        }
                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch

                    try {
                        if (list.get(i).equals(GREATER_OR_EQUALS)) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))) >= 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }
                        }
                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch

                    try {
                        if (list.get(i).equals(LESS_THAN)) {

                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, String.valueOf((Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))) < 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }

                        }//end if
                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch

                    try {
                        if (list.get(i).equals(LESS_OR_EQUALS)) {
                            if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {

                                list.set(i + 1, String.valueOf((Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))) <= 0));
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "false");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            } else if (list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                                list.set(i + 1, "true");
                                list.set(i - 1, "");
                                list.set(i, "");
                            }
                        }//end if

                    }//end try
                    catch (NullPointerException nolerr) {
                    }//end catch
                    catch (NumberFormatException numerr) {
                    }//end catch
                    catch (IndexOutOfBoundsException inderr) {
                    }//end catch

                }//end if (skip)
            }//end for

            list.removeAll(garbage);

        }//end if
        if (isHasPlusOrMinusOperators()) {
            //Handles the subtraction and addition operators
            for (int i = 0; i < list.size(); i++) {
                try {
                    if (list.get(i).equals(PLUS) || list.get(i).equals(MINUS)) {
                        if (!list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                            if (list.get(i).equals(PLUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                list.set(i + 1, String.valueOf(Double.parseDouble(list.get(i - 1)) + Double.parseDouble(list.get(i + 1))));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else
                            else if (list.get(i).equals(MINUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                list.set(i + 1, String.valueOf(Double.parseDouble(list.get(i - 1)) - Double.parseDouble(list.get(i + 1))));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else if
                        }//end if
                        else {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    }

                }//end try
                catch (NullPointerException nolerr) {
                }//end catch
                catch (NumberFormatException numerr) {
                }//end catch
                catch (IndexOutOfBoundsException inderr) {
                }//end catch
            }//end for
        }//end if

        garbage.add("(");
        garbage.add(")");

        list.removeAll(garbage);
        if (list.size() != 1) {
            this.parser_Result = this.parser_Result == Parser_Result.VALID ? Parser_Result.SYNTAX_ERROR : this.parser_Result;
            this.correctFunction = false;
            list.clear();
            list.add(parser_Result.name());
        }//end if

//Now de-list or un-package the input.If all goes well the list should have only its first memory location occupied.
        return list;

    }//end method solve

    /**
     * While traversing a list of tokens, sometimes its good to know if a token
     * is existing in the context of a stat function, or is a free token within
     * an algebraic expression etc Represents a scope within a mathematical
     * expression. Used to track if we are inside a statistical function,
     * algebraic group, or standard function.
     */
    static class ParseContext {

        public static final int CONTEXT_IS_STAT_METHOD = 1;
        public static final int CONTEXT_IS_ALGEBRAIC = 2;
        public static final int CONTEXT_IS_NON_STAT_METHOD = 3;

        /**
         * The index of the start of the context identifier(the token that
         * identifies the context -> e.g. sum for sum(...)
         */
        int start;
        /**
         * The token that identifies the context. e.g. sum for sum(...) or ( for
         * a normal algebraic expression
         */
        private String token;
        /**
         * The kind of the context
         */
        private int type;
        /**
         * A context encountered while in another context was executing e.g.
         * sum,(,3,4,sort,(,-3,2,1,-4,),)... The scan got to sum and changed its
         * context to {@link ParseContext#CONTEXT_IS_STAT_METHOD}, while
         * traversing sum's elements, it came across sort, which is a nested
         * context within sum, so it changes its context to
         * {@link ParseContext#CONTEXT_IS_STAT_METHOD} within and sets the
         * {@link ParseContext#next} field of the original context. When it
         * emerges from the close brackets of the sort(, it must null the next
         * field and then make the current context the original or the parent of
         * {@link ParseContext#next}
         */
        private ParseContext child;

        private ParseContext parent;

        public ParseContext(int start, String token, int type) {
            this.start = start;
            this.token = token;
            this.type = type;
        }

        /**
         * Finds the root context (the very first opening scope).
         */
        public ParseContext getFirst() {
            ParseContext curr = this;
            while (curr.parent != null) {
                curr = curr.parent;
            }
            return curr;
        }

        /**
         * Finds the current deepest nested context.
         */
        public ParseContext getLast() {
            ParseContext curr = this;
            while (curr.child != null) {
                curr = curr.child;
            }
            return curr;
        }

        /**
         * Correctly removes the deepest context and returns the new "last"
         * context. This is vital for "emerging" from a close bracket.
         */
        public ParseContext removeLast() {
            ParseContext last = getLast();
            ParseContext newLast = last.parent;
            if (newLast != null) {
                newLast.child = null; // Sever the link to the child
                last.parent = null;   // Clean up references for GC
            }
            return newLast;
        }

        /**
         * Adds a new nested context to the current chain.
         */
        public void addChild(ParseContext newChild) {
            ParseContext last = getLast();
            last.child = newChild;
            newChild.parent = last;
        }

        // --- Getters and Setters ---
        public int getType() {
            return type;
        }

        public void setType(int type) {
            this.type = type;
        }

        public int getStart() {
            return start;
        }

        public void setStart(int start) {
            this.start = start;
        }

        public String getToken() {
            return token;
        }

        public void setToken(String token) {
            this.token = token;
        }

        public ParseContext getChild() {
            return child;
        }

        public ParseContext getParent() {
            return parent;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            buildString(sb, 0);
            return sb.toString();
        }

        private void buildString(StringBuilder sb, int depth) {
            // Create indentation based on nesting depth
            for (int i = 0; i < depth; i++) {
                sb.append("  ");
            }

            // Append context details
            sb.append("Context{")
                    .append("type=").append(getTypeName(type))
                    .append(", token='").append(token).append('\'')
                    .append(", start=").append(start)
                    .append("}\n");

            // Recurse into the child if it exists
            if (child != null) {
                child.buildString(sb, depth + 1);
            }
        }

        private String getTypeName(int type) {
            switch (type) {
                case CONTEXT_IS_STAT_METHOD:
                    return "STAT";
                case CONTEXT_IS_ALGEBRAIC:
                    return "ALGEBRAIC";
                case CONTEXT_IS_NON_STAT_METHOD:
                    return "FUNCTION";
                default:
                    return "UNKNOWN";
            }
        }
    }

    public final class ExpressionSolverOld {

        // Operator constants (adjust to match your project's constants if needed)
        private static final String INFINITY = "Infinity";

        // -------------------------
        // Public entry point
        // -------------------------
        protected List<String> solve(List<String> list) {
            if (list == null) {
                return null;
            }

            // 1) Fix leading "( - number" anomaly -> mark the minus slot for removal and
            //    convert the number to negative (or remove leading '-' if already negative).
            fixLeadingParenthesisMinus(list);

            // 2) Remove garbage tokens quickly (single pass)
            cleanList(list);

            // 3) Post-number operators (factorial, square, cube, inverse)
            if (isHasPostNumberOperators()) {
                applyPostNumberOperators(list);
                cleanList(list);
            }

            // 4) Primary power pass (handles numeric ^ numeric)
            if (isHasPowerOperators()) {
                applyPowerOperators(list);
                cleanList(list);
            }

            // 5) Pre-number operators (root, cube root, etc.)
            if (isHasPreNumberOperators()) {
                applyPreNumberOperators(list);
                cleanList(list);
            }

            // 6) Secondary power pass (if original logic required a second pass)
            //    The original code ran power handling in two places; keep a second pass
            //    to preserve semantics where functions and powers interact.
            if (isHasPowerOperators()) {
                applyPowerOperators(list);
                cleanList(list);
            }

            // 7) Permutation / Combination
            if (isHasPermOrCombOperators()) {
                applyPermCombOperators(list);
                cleanList(list);
            }

            // 8) Multiplication / Division / Remainder and logical numeric operators
            if (isHasMulOrDivOperators() || isHasRemainderOperators() || isHasLogicOperators()) {
                applyMulDivRemainderAndLogic(list);
                cleanList(list);
            }

            // 9) Plus / Minus (final pass)
            if (isHasPlusOrMinusOperators()) {
                applyAddSubOperators(list);
                cleanList(list);
            }

            // Final packaging: if more than one token remains, mark syntax error
            if (list.size() != 1) {
                MathExpression.this.parser_Result = (MathExpression.this.parser_Result == Parser_Result.VALID)
                        ? Parser_Result.SYNTAX_ERROR
                        : MathExpression.this.parser_Result;
                MathExpression.this.correctFunction = false;
                list.clear();
                list.add(MathExpression.this.parser_Result.name());
            }

            return list;
        }

        // -------------------------
        // Helper: fix leading "( - number" anomaly
        // -------------------------
        private void fixLeadingParenthesisMinus(List<String> list) {
            if (list.size() >= 3 && "(".equals(list.get(0)) && MINUS.equals(list.get(1))) {
                String third = list.get(2);
                if (isNumber(third)) {
                    // mark the minus slot for removal and flip sign on the number
                    list.set(1, "");
                    if (third.startsWith(MINUS)) {
                        list.set(2, third.substring(1)); // "-5" -> "5"
                    } else {
                        list.set(2, MINUS + third); // "5" -> "-5"
                    }
                }
            }
        }

        // -------------------------
        // Helper: remove garbage tokens quickly
        // -------------------------
        private void cleanList(List<String> list) {
            // Remove empty strings and parentheses in one pass using removeIf (O(n))
            Predicate<String> garbage = s -> s == null || s.isEmpty() || "(".equals(s) || ")".equals(s);
            list.removeIf(garbage);
        }

        // -------------------------
        // Post-number operators: factorial, square, cube, inverse
        // -------------------------
        private void applyPostNumberOperators(List<String> list) {
            // iterate left-to-right; we examine token i (value) and i+1 (post-operator)
            for (int i = 0; i + 1 < list.size(); i++) {
                String val = list.get(i);
                String op = list.get(i + 1);
                if (op == null) {
                    continue;
                }

                if (isFactorial(op)) {
                    if (isNumber(val)) {
                        list.set(i + 1, Maths.fact(val));
                        list.set(i, "");
                    } else if (INFINITY.equals(val)) {
                        list.set(i + 1, INFINITY);
                        list.set(i, "");
                    }
                } else if (isSquare(op)) {
                    handlePowerEffect(list, i, val, 2.0);
                } else if (isCube(op)) {
                    handlePowerEffect(list, i, val, 3.0);
                } else if (isInverse(op)) {
                    if (isNumber(val)) {
                        double d = parseDoubleSafe(val);
                        list.set(i + 1, String.valueOf(1.0 / d));
                        list.set(i, "");
                    } else if (INFINITY.equals(val)) {
                        list.set(i + 1, "0.0");
                        list.set(i, "");
                    }
                }
            }
        }

        // Small helper used by post-number operators to avoid duplication
        private void handlePowerEffect(List<String> list, int i, String valStr, double pow) {
            if (isNumber(valStr)) {
                double base = parseDoubleSafe(valStr);
                list.set(i + 1, String.valueOf(Math.pow(base, pow)));
                list.set(i, "");
            } else if (INFINITY.equals(valStr)) {
                list.set(i + 1, INFINITY);
                list.set(i, "");
            }
        }

        // -------------------------
        // Power operators (binary ^)
        // -------------------------
        private void applyPowerOperators(List<String> list) {
            // iterate left-to-right; evaluate only when both neighbors are numbers or Infinity
            // We iterate from left to right and collapse triples [left, ^, right] into right
            for (int i = 1; i + 1 < list.size(); i++) {
                String op = list.get(i);
                if (!POWER.equals(op)) {
                    continue;
                }

                String left = safeGet(list, i - 1);
                String right = safeGet(list, i + 1);

                // If either neighbor missing, skip
                if (left == null || right == null) {
                    continue;
                }

                // If both numeric
                if (isNumber(left) && isNumber(right)) {
                    double base = parseDoubleSafe(left);
                    double exponent = parseDoubleSafe(right);
                    double result = Math.pow(base, exponent);
                    list.set(i + 1, String.valueOf(result));
                    list.set(i - 1, "");
                    list.set(i, "");
                    i++; // skip past the replaced token
                    continue;
                }

                // Handle Infinity cases with simplified but consistent rules:
                // - Infinity ^ positive -> Infinity
                // - Infinity ^ 0 -> 1.0
                // - Infinity ^ negative -> 0.0
                // - number ^ Infinity -> depends on base: >1 -> Infinity, ==1 -> 1.0, 0<base<1 -> 0.0, base<0 -> Infinity (original had many branches)
                if (INFINITY.equals(left) && INFINITY.equals(right)) {
                    list.set(i + 1, INFINITY);
                    list.set(i - 1, "");
                    list.set(i, "");
                    continue;
                }

                if (INFINITY.equals(left) && isNumber(right)) {
                    double r = parseDoubleSafe(right);
                    if (r > 0) {
                        list.set(i + 1, INFINITY);
                    } else if (r == 0) {
                        list.set(i + 1, "1.0");
                    } else {
                        list.set(i + 1, "0.0");
                    }
                    list.set(i - 1, "");
                    list.set(i, "");
                    continue;
                }

                if (isNumber(left) && INFINITY.equals(right)) {
                    double l = parseDoubleSafe(left);
                    if (l > 1) {
                        list.set(i + 1, INFINITY);
                    } else if (l == 1) {
                        list.set(i + 1, "1.0");
                    } else if (l > 0) {
                        list.set(i + 1, "0.0");
                    } else {
                        list.set(i + 1, INFINITY);
                    }
                    list.set(i - 1, "");
                    list.set(i, "");
                }
            }
        }

        // -------------------------
        // Pre-number operators (ROOT, CUBE_ROOT)
        // -------------------------
        private void applyPreNumberOperators(List<String> list) {
            // iterate right-to-left to allow replacing operator with computed value
            for (int i = list.size() - 2; i >= 0; i--) {
                String op = list.get(i);
                String next = safeGet(list, i + 1);
                if (op == null || next == null) {
                    continue;
                }

                if (INFINITY.equals(next)) {
                    list.set(i, INFINITY);
                    list.set(i + 1, "");
                    continue;
                }

                if (isNumber(next)) {
                    double val = parseDoubleSafe(next);
                    if (ROOT.equals(op)) {
                        list.set(i, String.valueOf(Math.sqrt(val)));
                        list.set(i + 1, "");
                    } else if (CUBE_ROOT.equals(op)) {
                        list.set(i, String.valueOf(Math.cbrt(val)));
                        list.set(i + 1, "");
                    }
                }
            }
        }

        // -------------------------
        // Permutation / Combination
        // -------------------------
        private void applyPermCombOperators(List<String> list) {
            for (int i = 1; i + 1 < list.size(); i++) {
                String op = list.get(i);
                if (PERMUTATION.equals(op) || COMBINATION.equals(op)) {
                    String left = safeGet(list, i - 1);
                    String right = safeGet(list, i + 1);
                    if (left == null || right == null) {
                        continue;
                    }

                    if (INFINITY.equals(left) || INFINITY.equals(right)) {
                        // If either side is Infinity, result is Infinity (original used many branches)
                        list.set(i + 1, INFINITY);
                        list.set(i - 1, "");
                        list.set(i, "");
                        continue;
                    }

                    if (isNumber(left) && isNumber(right)) {
                        // Use factorial-based formulas; guard against large factorials by parsing to integer where appropriate
                        double n = parseDoubleSafe(left);
                        double r = parseDoubleSafe(right);
                        if (PERMUTATION.equals(op)) {
                            // nPr = n! / (n-r)!
                            String res = safePermutation(n, r);
                            list.set(i + 1, res);
                        } else {
                            // nCr = n! / ((n-r)! * r!)
                            String res = safeCombination(n, r);
                            list.set(i + 1, res);
                        }
                        list.set(i - 1, "");
                        list.set(i, "");
                    }
                }
            }
        }

        // Safe permutation using Maths.fact if available; fallback to 0.0 on error
        private String safePermutation(double n, double r) {
            try {
                // Prefer integer factorial if values are integral and small
                if (isIntegral(n) && isIntegral(r) && n >= r && n >= 0 && r >= 0) {
                    return String.valueOf(Double.parseDouble(Maths.fact(String.valueOf((long) n)))
                            / Double.parseDouble(Maths.fact(String.valueOf((long) (n - r)))));
                }
            } catch (Exception ignored) {
            }
            return "0.0";
        }

        private String safeCombination(double n, double r) {
            try {
                if (isIntegral(n) && isIntegral(r) && n >= r && n >= 0 && r >= 0) {
                    double numerator = Double.parseDouble(Maths.fact(String.valueOf((long) n)));
                    double denom = Double.parseDouble(Maths.fact(String.valueOf((long) (n - r))))
                            * Double.parseDouble(Maths.fact(String.valueOf((long) r)));
                    return String.valueOf(numerator / denom);
                }
            } catch (Exception ignored) {
            }
            return "0.0";
        }

        // -------------------------
        // Multiplication / Division / Remainder and numeric logic (==, >, >=, <, <=)
        // -------------------------
        private void applyMulDivRemainderAndLogic(List<String> list) {
            // We will do a single pass and handle multiply/divide/remainder first,
            // then comparisons in the same pass to preserve precedence similar to original.
            for (int i = 1; i + 1 < list.size(); i++) {
                String op = list.get(i);
                String left = safeGet(list, i - 1);
                String right = safeGet(list, i + 1);
                if (op == null || left == null || right == null) {
                    continue;
                }

                // Multiplicative operators
                if (MULTIPLY.equals(op) || DIVIDE.equals(op) || REMAINDER.equals(op)) {
                    // Handle Infinity cases and numeric cases
                    String result = computeMulDivRem(left, right, op);
                    if (result != null) {
                        list.set(i + 1, result);
                        list.set(i - 1, "");
                        list.set(i, "");
                        i++; // skip past replaced token
                    }
                    continue;
                }

                // Comparison / logical operators (==, >, >=, <, <=)
                if (EQUALS.equals(op) || GREATER_THAN.equals(op) || GREATER_OR_EQUALS.equals(op)
                        || LESS_THAN.equals(op) || LESS_OR_EQUALS.equals(op)) {
                    String result = computeComparison(left, right, op);
                    if (result != null) {
                        list.set(i + 1, result);
                        list.set(i - 1, "");
                        list.set(i, "");
                        i++;
                    }
                }
            }
        }

        private String computeMulDivRem(String left, String right, String op) {
            // Handle Infinity strings explicitly
            boolean leftInf = INFINITY.equals(left);
            boolean rightInf = INFINITY.equals(right);

            try {
                if (leftInf && rightInf) {
                    if (MULTIPLY.equals(op) || REMAINDER.equals(op)) {
                        return INFINITY;
                    }
                    if (DIVIDE.equals(op)) {
                        return INFINITY;
                    }
                }
                if (leftInf && !rightInf) {
                    double r = parseDoubleSafe(right);
                    if (MULTIPLY.equals(op)) {
                        return INFINITY;
                    }
                    if (DIVIDE.equals(op)) {
                        return INFINITY;
                    }
                    if (REMAINDER.equals(op)) {
                        return INFINITY;
                    }
                }
                if (!leftInf && rightInf) {
                    double l = parseDoubleSafe(left);
                    if (MULTIPLY.equals(op)) {
                        return INFINITY;
                    }
                    if (DIVIDE.equals(op)) {
                        return "0.0";
                    }
                    if (REMAINDER.equals(op)) {
                        return left; // x % Infinity -> x
                    }
                }

                // Both numeric
                if (isNumber(left) && isNumber(right)) {
                    double l = parseDoubleSafe(left);
                    double r = parseDoubleSafe(right);
                    if (op.charAt(0) == MULTIPLY.charAt(0)) {
                        return String.valueOf(l * r);
                    } else if (op.charAt(0) == DIVIDE.charAt(0)) {
                        return String.valueOf(l / r);
                    } else if (op.charAt(0) == REMAINDER.charAt(0)) {
                        return String.valueOf(l % r);
                    }
                }
            } catch (Exception ignored) {
            }
            return null;
        }

        private String computeComparison(String left, String right, String op) {
            boolean leftInf = INFINITY.equals(left);
            boolean rightInf = INFINITY.equals(right);

            try {
                if (leftInf && rightInf) {
                    // Infinity compared to Infinity: == true, > true, >= true, < false, <= true
                    if (isEqualsOperator(op)) {
                        return "true";
                    } else if (isGreaterThanOperator(op)) {
                        return "true";
                    } else if (isGreaterOrEqualsOperator(op)) {
                        return "true";
                    } else if (isLessThanOperator(op)) {
                        return "false";
                    } else if (isLessThanOrEqualsOperator(op)) {
                        return "true";
                    }
                }
                if (leftInf && !rightInf) {
                    if (isEqualsOperator(op)) {
                        return "false";
                    } else if (isGreaterThanOperator(op)) {
                        return "true";
                    } else if (isGreaterOrEqualsOperator(op)) {
                        return "true";
                    } else if (isLessThanOperator(op)) {
                        return "false";
                    } else if (isLessThanOrEqualsOperator(op)) {
                        return "false";
                    }
                }
                if (!leftInf && rightInf) {
                    if (isEqualsOperator(op)) {
                        return "false";
                    } else if (isGreaterThanOperator(op)) {
                        return "false";
                    } else if (isGreaterOrEqualsOperator(op)) {
                        return "false";
                    } else if (isLessThanOperator(op)) {
                        return "true";
                    } else if (isLessThanOrEqualsOperator(op)) {
                        return "false";
                    }
                }

                if (isNumber(left) && isNumber(right)) {
                    double l = parseDoubleSafe(left);
                    double r = parseDoubleSafe(right);

                    if (isEqualsOperator(op)) {
                        return String.valueOf((l - r) == 0);
                    } else if (isGreaterThanOperator(op)) {
                        return String.valueOf((l - r) > 0);
                    } else if (isGreaterOrEqualsOperator(op)) {
                        return String.valueOf((l - r) >= 0);
                    } else if (isLessThanOperator(op)) {
                        return String.valueOf((l - r) < 0);
                    } else if (isLessThanOrEqualsOperator(op)) {
                        return String.valueOf((l - r) <= 0);
                    }
                }
            } catch (Exception ignored) {
            }
            return null;
        }

        // -------------------------
        // Plus / Minus (final pass)
        // -------------------------
        private void applyAddSubOperators(List<String> list) {
            for (int i = 1; i + 1 < list.size(); i++) {
                String op = list.get(i);
                if (!PLUS.equals(op) && !MINUS.equals(op)) {
                    continue;
                }

                String left = safeGet(list, i - 1);
                String right = safeGet(list, i + 1);
                if (left == null || right == null) {
                    continue;
                }

                // Handle Infinity cases
                if (INFINITY.equals(left) || INFINITY.equals(right)) {
                    String res = computeAddSubInfinity(left, right, op);
                    list.set(i + 1, res);
                    list.set(i - 1, "");
                    list.set(i, "");
                    i++;
                    continue;
                }

                if (isNumber(left) && isNumber(right)) {
                    double l = parseDoubleSafe(left);
                    double r = parseDoubleSafe(right);
                    double result = PLUS.equals(op) ? (l + r) : (l - r);
                    list.set(i + 1, String.valueOf(result));
                    list.set(i - 1, "");
                    list.set(i, "");
                    i++;
                }
            }
        }

        private String computeAddSubInfinity(String left, String right, String op) {
            // Simplified consistent rules:
            if (INFINITY.equals(left) && INFINITY.equals(right)) {
                if (PLUS.equals(op)) {
                    return INFINITY;
                }
                if (MINUS.equals(op)) {
                    return "NaN"; // Infinity - Infinity is undefined
                }
            }
            if (INFINITY.equals(left) && !INFINITY.equals(right)) {
                return INFINITY;
            }
            if (!INFINITY.equals(left) && INFINITY.equals(right)) {
                return INFINITY;
            }
            return "0.0";
        }

        // -------------------------
        // Utility helpers
        // -------------------------
        private String safeGet(List<String> list, int idx) {
            if (idx < 0 || idx >= list.size()) {
                return null;
            }
            return list.get(idx);
        }

        private boolean isIntegral(double d) {
            return Math.floor(d) == d;
        }

        private double parseDoubleSafe(String s) {
            try {
                return Double.parseDouble(s);
            } catch (Exception e) {
                return Double.NaN;
            }
        }

        // -------------------------
        // Placeholder predicates and external calls
        // -------------------------
        // The original code referenced many helper methods (isNumber, isFactorial, etc.)
        // which are assumed to exist in the original codebase. We keep the same names
        // so this refactor can be dropped into the project and wired to the existing
        // implementations.
        protected boolean isNumber(String s) {
            // Delegate to existing implementation in your codebase.
            // Placeholder: treat numeric strings and "Infinity" as numbers.
            if (s == null) {
                return false;
            }
            if (INFINITY.equals(s)) {
                return true;
            }
            try {
                Double.parseDouble(s);
                return true;
            } catch (Exception e) {
                return false;
            }
        }

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     *
     * @param scanner is a list of scanner functions, gotten during the
     * evaluation of sets of data that contain functions that need to be
     * evaluated instead of numbers.If the data set does not contain functions
     * e.g avg(2,3,7,1,0,9,5), then method solve will easily solve it. But if it
     * does e.g avg(2,sin,3,5,cos,(,5,) ), then we invoke this method in class
     * Set's constructor before we evaluate the data set. Note this is method is
     * not called directly by MathExpression objects but by objects of class Set
     * invoked by a MathExpression object.
     * @return the solution to the scanner function
     */
    public List<String> solveSubPortions(List<String> scanner) {

        scanner.add(0, "(");
        scanner.add(")");
//[3,4,2,sin,2,sin,(,3,),3,*,cos,5,(,3,+,(,6,-,78,),)]
        int passes = 0;
        int i = 0;
        int j = 0;
        while (scanner.size() > 1) {
            passes++;
            try {
                i = scanner.indexOf(")");//index of first )
                j = LISTS.prevIndexOf(scanner, i, "("); //index of enclosing brackets of ) above
                List<String> sub = scanner.subList(j, i + 1);
                solve(sub);

            }//end try
            catch (IndexOutOfBoundsException indexerr) {
                break;
            }//end catch
        }//end while

        return scanner;
    }//end method solveSubPortions()

    public static void main1(String... args) {
        String in = Main.joinArgs(Arrays.asList(args), true);
        if (Main.isVerbose()) {
            System.err.println(in);
        }
        System.out.println(new MathExpression(in).solve());
    }//end method

    public static void main(String... args) {
        System.out.println(new MathExpression("x=20;sin(ln(x));").solve());
        MathExpression ml = new MathExpression("D=@(3,3)(3,4,1,2,4,7,9,1,-2);tri_mat(D)");
        System.out.println("ml.solve():" + ml.solve());
        MathExpression linear = new MathExpression("a=4;a11=3.14159265357;b=2.718281828;M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);C=matrix_sub(M,N);C;");
        System.out.println("VARIABLES: " + VariableManager.VARIABLES);
        System.out.println("M:" + FunctionManager.lookUp("M").getMatrix());
        System.out.println("N:" + FunctionManager.lookUp("N").getMatrix());
        System.out.println("C:" + FunctionManager.lookUp("C").getMatrix());

        System.out.println("FUNCTIONS: " + FunctionManager.FUNCTIONS);
        String ls = linear.solve();
        System.out.println("ls = " + ls);
        String s = "listsum(listsum(5),sin(5))";
        MathExpression ms = new MathExpression(s);
        System.out.println("ms.scanner: " + ms.scanner);
        System.out.println("s--->" + ms.solve());
        String ss = "listsum(sin(5),sin(5))";
        System.out.println("ss--->" + new MathExpression(ss).solve());
        String s1 = "x=9;g(x)=((sin(x)-tan(x)));v(x)=((ln(x)/tan(x)));f(x)=g(x);sin(1.75)+cos(1.23)+tan(1.86-0.26)+log(10,3)+sqrt(16)+exp(1)+pow(2,8)+abs(-42)+listsum(1,2,3,4,5)+sin(3*12+cos(55))-(4+5)*(2*(9-2)+12*(4-7));v(3);";
        String s2 = "v(x)=((ln(x)/tan(x)));2+v(3);";
        String s3 = "2+sin(3)-ln(12)+1/3.689";
        String s4 = "sin(2)+cos(3)-sum(2,3,4,sin(2),ln(42),3,4,5,6,1,2,3,45,2)+12";
        String s5 = "listsum(sin(3),cos(3),ln(345),sort(3,-4,5,-6,13,2,4,5,listsum(3,4,5,6,9,12,23), 12, listsum(3,4,8,9, 2000)), 12000, mode(3,2,2,1),32.897, mode(1,5,7,7,1,1,7))";

        MathExpression m = new MathExpression(s5);
        System.out.println("scanner:\n" + m.scanner);
        System.out.println("m.solve(): " + m.solve());

        System.out.println(new MathExpression("sort(0,4+0,2+0)").solve());

        //   double N = 100; 
        //   Shootouts.benchmark(s2, (int) N);
        //   Shootouts.benchmark(s3, (int) N);
    }//end method

    @Override
    public String serialize() {
        return Serializer.serialize(this);
    }

    public MathExpression parse(String enc) {
        return (MathExpression) Serializer.deserialize(enc);
    }

}//end class MathExpression
