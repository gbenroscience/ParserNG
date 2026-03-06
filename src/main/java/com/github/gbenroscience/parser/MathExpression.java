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

import static com.github.gbenroscience.parser.Variable.*;
import static com.github.gbenroscience.parser.Number.*;
import static com.github.gbenroscience.parser.Operator.*;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import static com.github.gbenroscience.parser.TYPE.ALGEBRAIC_EXPRESSION;
import static com.github.gbenroscience.parser.TYPE.LIST;
import static com.github.gbenroscience.parser.TYPE.MATRIX;
import com.github.gbenroscience.parser.benchmarks.GG;
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

    /**
     * Backup the alias from the scanner here
     */
    private String commaAlias;
    public ParserResult parser_Result = ParserResult.VALID;
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

    private static final int INIT_POOL_SIZE = 64;
    // A simple pre-allocated array of results to act as a stack
    private EvalResult[] pool = new EvalResult[INIT_POOL_SIZE];
    private int poolPointer = 0;

    private boolean help;

    /**
     * The kind of output returned by the parser.
     */
    TYPE returnType = TYPE.NUMBER;

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

        public static final int NUMBER = 0, OPERATOR = 1, FUNCTION = 2, METHOD = 3, LPAREN = 5, RPAREN = 6, COMMA = 7;
        public int kind;
        public double value;
        public String name; // REQUIRED for functions/methods/variables
        public int id;
        public char opChar;
        public int precedence;
        public boolean isRightAssoc;
        public boolean isPostfix;
        public int arity;
        private MethodRegistry.MethodAction action;

        // NEW FIELDS FOR FUNCTION ASSIGNMENT
        public String assignToName;  // The variable to assign result to (e.g., "vw")
        public boolean isAssignmentTarget = false;

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
            //This is an inbuilt method call
            if (id != -1) {
                this.action = MethodRegistry.getAction(id);
            } else {
                this.action = FunctionManager.lookUp(this.name);
                if (this.action == null) {
                    throw new RuntimeException("Function " + this.name + " not found");
                }
            }
        }

        // Constructor for Parens
        public Token(int kind) {
            this.kind = kind;
            this.precedence = -1;
            this.isRightAssoc = false;
        }

        // Add constructor for assignment
        public Token(int kind, String name, int arity, int id, String assignToName) {
            this(kind, name, arity, id);
            this.assignToName = assignToName;
            this.isAssignmentTarget = (assignToName != null);
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
                    + "\"arity\": " + arity + ",\n"
                    + "\"assignToName\": " + assignToName + ",\n"
                    + "\"isAssignmentTarget\": " + isAssignmentTarget + "\n"
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
                    + "\"arity\": " + arity + ","
                    + "\"assignToName\": " + assignToName + ","
                    + "\"isAssignmentTarget\": " + isAssignmentTarget
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
        this.help = input.equals(Declarations.HELP);
        for (int i = 0; i < INIT_POOL_SIZE; i++) {
            pool[i] = new EvalResult();
        }

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
                    parser_Result = ParserResult.SYNTAX_ERROR;
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
            this.poolPointer = 0;
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
        return scanner != null && !scanner.isEmpty() && correctFunction && this != null;
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
        opScanner.scanner(variableManager);
        this.commaAlias = opScanner.commaAlias;

        scanner = opScanner.getScanner();

        correctFunction = opScanner.isRunnable();

        parser_Result = opScanner.parser_Result;
        if (parser_Result == ParserResult.VALID) {
            statsVerifier();
            codeModifier();
            refixCommas();
            mapBrackets();
            functionComponentsAssociation();
            compileToPostfix();  // Compile once if not already done
        }//end if
    }//end method initializing(args)

    private void removeCommas() {
        scanner.replaceAll((String t) -> isComma(t) ? this.commaAlias : t);
    }

    private void refixCommas() {
        scanner.replaceAll((String t) -> this.commaAlias.equals(t) ? "," : t);
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
        if (DRG != this.DRG) {
            for (int i = 0; i < cachedPostfix.length; i++) {
                Token t = cachedPostfix[i];
                if (t.kind == Token.METHOD) {
                    if (t.name.endsWith("_deg") || t.name.endsWith("_rad") || t.name.endsWith("_grad")) {
                        String name = t.name.substring(0, t.name.lastIndexOf("_"));
                        t.name = Declarations.getTrigFuncDRGVariant(name, DRG);
                        t.id = MethodRegistry.getMethodID(t.name);
                        t.action = MethodRegistry.getAction(t.id);
                    }
                }
            }
        }
        this.DRG = DRG;
    }

    public void setDRG(int mode) {
        switch (mode) {
            case 0:
                setDRG(DRG_MODE.DEG);
                break;
            case 1:
                setDRG(DRG_MODE.RAD);
                break;
            case 2:
                setDRG(DRG_MODE.GRAD);
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
     * to..e.g a {@link Matrix} function or other. But we at times need that
     * function name...So we cache this value here.
     *
     * @return the object returned via its string reference name
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

    private void statsVerifier() {
        scanner.removeAll(whitespaceremover);

        //determine the presence of list returning statistical operators
        for (int i = 0; i < scanner.size(); i++) {
            if (Method.isListReturningStatsMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                noOfListReturningOperators++;
            }//end if

        }//end for

        correctFunction = ListReturningStatsMethod.validateFunction(this.scanner);
        parser_Result = correctFunction ? ParserResult.VALID : ParserResult.SYNTAX_ERROR;
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
                    parser_Result = ParserResult.SYNTAX_ERROR;
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
    private void codeModifier() {

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
    private static Bracket[] mapBrackets(List<String> scanner) {
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
    private void mapBrackets() {
        try {
            mapBrackets(scanner);
        }//end method//end method
        catch (InputMismatchException ime) {
            parser_Result = ParserResult.PARENTHESES_ERROR;
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
    private void functionComponentsAssociation() {

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
                parser_Result = ParserResult.SYNTAX_ERROR;
            }//end else

        }//end if
        else {
            scanner.clear();
            parser_Result = ParserResult.SYNTAX_ERROR;
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
        Variable var = VariableManager.lookUp(name);
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

    public void setReturnType(TYPE returnType) {
        this.returnType = returnType;
    }

    public TYPE getReturnType() {
        return returnType;
    }

    public EvalResult solveGeneric() {
        if (help) {
            EvalResult res = new EvalResult();
            res.wrap(Help.getHelp());
            return res;
        }
        if (cachedPostfix != null) {
            resetPool();
            EvalResult r = expressionSolver.evaluate();
            returnType = r.getType();
            return r;
        }
        if (scanner == null || scanner.isEmpty() || !correctFunction || parser_Result != ParserResult.VALID) {
            return EvalResult.ERROR;
        }

        return EvalResult.ERROR;
    }

    @Override
    public String solve() {
        return solveGeneric().toString();
    }//end method solve()

    protected List<String> solve(List<String> list) {
        return Arrays.asList(String.valueOf(GG.evaluate(list)));
    }

    private Token translate(String s, String next) {
        if (s == null || s.isEmpty()) {
            return null;
        }
        char c0 = s.charAt(0);
        int len = s.length();

        // 1. Identify Numbers
        if (isNumber(s)) {
            return new Token(fastParseDouble(s));
        }

        // 2. Identify Brackets
        if (len == 1) {
            switch (c0) {
                case '(':
                    return new Token(Token.LPAREN);
                case ')':
                    return new Token(Token.RPAREN);
            }
        }

        // 3. Identify Operators
        if (isOperator(s)) {
            boolean isPostfix = (c0 == '!' || c0 == '²' || c0 == '³') && !s.equals("³√");
            char internalOp = (len > 1 && s.equals("³√")) ? 'R' : c0;
            return new Token(internalOp, Token.getPrec(internalOp), Token.isRightAssociative(internalOp), isPostfix);
        }

        // Handle commas (function argument separators)
        if (s.equals(",")) {
            Token t = new Token(0.0);
            t.kind = Token.COMMA;
            return t;
        }

        // 4. Identify Functions/Anonymous Functions (ONLY if followed by "(")
        if (FunctionManager.FUNCTIONS.containsKey(s)) {
            // CRITICAL: Only treat as FUNCTION if it's being CALLED (has "(" after it)
            if (next != null && next.equals("(")) {
                return new Token(Token.FUNCTION, s, -1, -1); // Arity set during compile
            }

            // If NOT followed by "(", treat as a FUNCTION REFERENCE/POINTER
            // This allows passing function names as arguments (like to diff)
            Token t = new Token(0.0);
            t.kind = Token.NUMBER;  // Treat as a "value" that can be passed as argument
            t.name = s;  // Store the function name
            return t;
        }

        // 5. Identify Methods
        if (Method.isInBuiltMethod(s)) {
            String transformedMethodName = Declarations.getTrigFuncDRGVariant(s, DRG);
            int methodId = MethodRegistry.getMethodID(transformedMethodName);
            return new Token(Token.METHOD, transformedMethodName, -1, methodId); // Arity set during compile
        }

        // 6. Fallback: Treat as Variable/Constant
        Token t = new Token(0.0);
        t.name = s;
        t.kind = Token.NUMBER;

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

    private void compileToPostfix() {
        if (cachedPostfix != null) {
            return;
        }

        Stack<Token> opStack = new Stack<>();
        Stack<Integer> argCounts = new Stack<>();
        Stack<Boolean> lastWasComma = new Stack<>();
        Stack<Boolean> isGrouping = new Stack<>();  // Track if this paren is grouping

        Token[] postfix = new Token[scanner.size() * 2];
        int p = 0;

        int depth = 0;
        argCounts.push(0);
        lastWasComma.push(true);

        int len = scanner.size();
        for (int idx = 0; idx < len; idx++) {
            String s = scanner.get(idx);
            String next = idx + 1 < len ? scanner.get(idx + 1) : null;

            Token t = translate(s, next);
            if (t == null) {
                continue;
            }

            switch (t.kind) {
                case Token.NUMBER:
                    postfix[p++] = t;
                    if (depth > 0 && lastWasComma.peek()) {
                        int currentCount = argCounts.pop();
                        argCounts.push(currentCount + 1);
                        lastWasComma.pop();
                        lastWasComma.push(false);
                    }
                    break;

                case Token.FUNCTION:
                case Token.METHOD:
                    opStack.push(t);
                    break;

                case Token.LPAREN:
                    boolean isFuncParen = false;
                    if (!opStack.isEmpty()) {
                        Token lastOp = opStack.peek();
                        if (lastOp.kind == Token.FUNCTION || lastOp.kind == Token.METHOD) {
                            isFuncParen = true;
                        }
                    }

                    opStack.push(t);

                    if (isFuncParen) {
                        depth++;
                        argCounts.push(0);
                        lastWasComma.push(true);
                        isGrouping.push(false);
                    } else {
                        // Grouping paren - still track if we're in a function
                        isGrouping.push(true);
                    }
                    break;

                case Token.RPAREN:
                    // Pop operators until matching '('
                    while (!opStack.isEmpty() && opStack.peek().kind != Token.LPAREN) {
                        postfix[p++] = opStack.pop();
                    }

                    if (!opStack.isEmpty()) {
                        opStack.pop(); // discard the '('
                    }

                    boolean wasGrouping = !isGrouping.isEmpty() && isGrouping.pop();

                    if (wasGrouping) {
                        // Closing a GROUPING paren
                        // This completes a value in the function's arg list
                        if (depth > 0 && lastWasComma.peek()) {
                            int currentCount = argCounts.pop();
                            argCounts.push(currentCount + 1);
                            lastWasComma.pop();
                            lastWasComma.push(false);
                        }
                    } else {
                        // Closing a FUNCTION CALL paren
                        if (!opStack.isEmpty()) {
                            Token callable = opStack.pop();

                            int actualArgCount = argCounts.pop();
                            lastWasComma.pop();

                            callable.arity = Math.max(1, actualArgCount);
                            postfix[p++] = callable;

                            depth--;

                            // Function result is a value in parent
                            if (depth > 0 && lastWasComma.peek()) {
                                int parentCount = argCounts.pop();
                                argCounts.push(parentCount + 1);
                                lastWasComma.pop();
                                lastWasComma.push(false);
                            }
                        }
                    }
                    break;

                case Token.COMMA:
                    while (!opStack.isEmpty() && opStack.peek().kind != Token.LPAREN) {
                        postfix[p++] = opStack.pop();
                    }
                    if (depth > 0 && !lastWasComma.isEmpty()) {
                        lastWasComma.pop();
                        lastWasComma.push(true);
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

        while (!opStack.isEmpty()) {
            Token top = opStack.pop();
            if (top.kind != Token.LPAREN) {
                postfix[p++] = top;
            }
        }

        cachedPostfix = new Token[p];
        System.arraycopy(postfix, 0, cachedPostfix, 0, p);
    }

    private final class ExpressionSolver {

        private static final int MAX_ARITY = 32;
        private static final int ABSOLUTE_MAX_ARITY = 100_000;
        private EvalResult[][] argCache = new EvalResult[MAX_ARITY + 1][];

        public ExpressionSolver() {
            // Pre-allocate ONE array per arity that we reuse
            for (int i = 0; i <= MAX_ARITY; i++) {
                argCache[i] = new EvalResult[i];
            }
        }

        public EvalResult evaluate() {
            final EvalResult[] stack = new EvalResult[Math.max(cachedPostfix.length * 2, 64)];
            int ptr = -1;

            for (int i = 0; i < cachedPostfix.length; i++) {
                Token t = cachedPostfix[i];
                /*              System.out.println("\n=== Evaluating token: "
                        + (t.kind == Token.NUMBER ? "NUM(" + t.value + ")"
                                : t.kind == Token.OPERATOR ? "OP(" + t.opChar + ")"
                                        : t.kind == Token.FUNCTION ? "FUNC(" + t.name + ",arity=" + t.arity + ")"
                                                : "METHOD(" + t.name + ",arity=" + t.arity + ")")
                        + " | Stack ptr before = " + ptr);
                 */
                switch (t.kind) {
                    case Token.NUMBER:
                        if (t.name != null && !t.name.isEmpty()) {
                            // Could be a variable OR a function reference (like anon1)
                            Variable var = VariableManager.lookUp(t.name);
                            if (var != null) {
                                // It's a variable
                                stack[++ptr] = getNextResult().wrap(var.getValue());
                            } else if (FunctionManager.FUNCTIONS.containsKey(t.name)) {
                                // It's a function reference - wrap it as a special object
                                // The diff method will know how to handle it
                                stack[++ptr] = getNextResult().wrap(t.name);  // Store the function name as string
                            } else {
                                String prevToken = stack[ptr].toString();
                                if (FunctionManager.lookUp(prevToken) != null) {
                                    //e.g diff(F,v) or diff(F,v,n) where F is the prevToken and v is the undefined variable to be used to create
                                    //a function that will hold the return value of diff(F)
                                    stack[++ptr] = getNextResult().wrap(t.name); // Store pointer to the to-be-created function as string
                                } else {
                                    throw new RuntimeException("Undefined variable or function: " + t.name);
                                }
                            }
                        } else {
                            // Direct number
                            stack[++ptr] = getNextResult().wrap(t.value);
                        }
                        break;
                    case Token.OPERATOR:
                        if (t.isPostfix || t.opChar == '√' || t.opChar == 'R') {
                            // Unary operator
                            if (ptr < 0) {
                                throw new RuntimeException("Insufficient operands for unary operator: " + t.opChar);
                            }
                            applyUnary(t.opChar, stack[ptr]);
                        } else {
                            // Binary operator
                            if (ptr < 1) {
                                throw new RuntimeException("Insufficient operands for binary operator: " + t.opChar
                                        + " (stack ptr=" + ptr + ")");
                            }
                            double bVal = stack[ptr--].scalar;
                            applyBinary(t.opChar, stack[ptr], bVal);
                        }
                        break;

                    case Token.METHOD:
                    case Token.FUNCTION:
                        int arity = t.arity;

                        if (arity > ABSOLUTE_MAX_ARITY) {
                            throw new RuntimeException(
                                    "Function " + t.name + " has too many arguments (" + arity
                                    + "). Maximum supported: " + ABSOLUTE_MAX_ARITY);
                        }

                        int valuesOnStack = ptr + 1;

                        if (arity == 0) {
                            EvalResult result = t.action.calc(getNextResult(), arity);
                            stack[++ptr] = result;
                            break;
                        }

                        if (valuesOnStack < arity) {
                            throw new RuntimeException("Function " + t.name + " requires " + arity
                                    + " arguments but only " + valuesOnStack + " values available on stack");
                        }

                        // ULTRA FAST: Reuse cached array for this arity
                        EvalResult[] args = (arity <= MAX_ARITY) ? argCache[arity] : new EvalResult[arity];

                        // Direct loop (fastest for small arrays)
                        for (int j = arity - 1; j >= 0; j--) {
                            args[j] = stack[ptr--];
                        }

                        EvalResult result;
                        try {
                            result = t.action.calc(getNextResult(), arity, args);
                        } catch (Exception e) {
                            throw new RuntimeException("Error executing " + t.name + ": " + e.getMessage(), e);
                        }

                        stack[++ptr] = result;
                        break;
                }
            }

            // Final validation
            if (ptr < 0) {
                EvalResult r = new EvalResult();
                r.wrap(ParserResult.SYNTAX_ERROR);
                return r;
            }

            if (ptr > 0) {
                System.out.println("WARNING: Evalaution stack has " + (ptr + 1) + " values at end, returning top");
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
                    res.scalar = Maths.fact(val);
                    break;
                case '²':
                    res.scalar = val * val;
                    break;
                case '³':
                    res.scalar = val * val * val;
                    break;
                case '-': // Unary negation (if supported by my scanner, actually my scanner has fixed it before this stage)
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
                    if (b == 0) {
                        throw new ArithmeticException("Division by zero");
                    }
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
        public static final int TYPE_MATRIX = 2;
        public static final int TYPE_STRING = 3;
        public static final int TYPE_BOOLEAN = 4;
        public static final int TYPE_ERROR = 5;

        public static final EvalResult ERROR = new EvalResult();

        static {
            ERROR.wrap(ParserResult.SYNTAX_ERROR);
        }

        public int type;           // 0=Scalar, 1=Vector, 2=Matrix, 3=String

        public double scalar;      // For single numbers
        public double[] vector;    // For stats/lists
        public Matrix matrix;  // For matrices
        public ParserResult error;
        public boolean boolVal;
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
            this.type = TYPE_MATRIX;
            return this;
        }

        public EvalResult wrap(String s) {
            this.textRes = s;
            this.type = TYPE_STRING;
            return this;
        }

        public EvalResult wrap(boolean s) {
            this.boolVal = s;
            this.type = TYPE_BOOLEAN;
            return this;
        }

        public EvalResult wrap(ParserResult s) {
            this.error = s;
            this.type = TYPE_ERROR;
            return this;
        }

        public EvalResult wrap(EvalResult evr) {
            switch (evr.type) {
                case TYPE_ERROR:
                    wrap(evr.error);
                    break;
                case TYPE_BOOLEAN:
                    wrap(evr.boolVal);
                    break;
                case TYPE_SCALAR:
                    wrap(evr.scalar);
                    break;
                case TYPE_MATRIX:
                    wrap(evr.matrix);
                    break;
                case TYPE_STRING:
                    wrap(evr.textRes);
                    break;
                case TYPE_VECTOR:
                    wrap(evr.vector);
                    break;
                default:
                    throw new AssertionError();
            }
            this.type = evr.type;
            return this;
        }

// In EvalResult class:
        public void reset() {
            this.scalar = 0.0;
            this.vector = null;
            this.matrix = null;
            this.textRes = null;
            this.boolVal = false;
            this.error = null;
            this.type = TYPE_SCALAR;
        }

        @Override
        public String toString() {
            switch (type) {
                case TYPE_SCALAR:
                    return String.valueOf(scalar);
                case TYPE_VECTOR:
                    return Arrays.toString(vector);
                case TYPE_MATRIX:
                    return matrix.toString();
                case TYPE_STRING:
                    return textRes;
                case TYPE_BOOLEAN:
                    return String.valueOf(boolVal);
                case TYPE_ERROR:
                    return error == ParserResult.SYNTAX_ERROR ? SYNTAX_ERROR : error.toString();
                default:
                    return "0.0";
            }
        }

        public String getTypeName() {
            switch (type) {
                case TYPE_SCALAR:
                    return "TYPE_SCALAR(double)";
                case TYPE_STRING:
                    return "TYPE_STRING(text)";
                case TYPE_VECTOR:
                    return "TYPE_VECTOR(an array of doubles)";
                case TYPE_MATRIX:
                    return "TYPE_MATRIX(Matrix)";
                case TYPE_BOOLEAN:
                    return "TYPE_BOOLEAN(Boolean)";
                case TYPE_ERROR:
                    return "TYPE_ERROR(" + error.name() + ")";
                default:
                    return "TYPE_UNKNOWN";
            }
        }

        public TYPE getType() {
            switch (type) {
                case TYPE_SCALAR:
                    return TYPE.NUMBER;
                case TYPE_STRING:
                    return TYPE.STRING;
                case TYPE_VECTOR:
                    return TYPE.LIST;
                case TYPE_MATRIX:
                    return TYPE.MATRIX;
                case TYPE_BOOLEAN:
                    return TYPE.BOOLEAN;
                case TYPE_ERROR:
                    return TYPE.ERROR;
                default:
                    return TYPE.ALGEBRAIC_EXPRESSION;
            }
        }

        public EvalResult absorb(Function f) {
            if (f != null) {
                TYPE tt = f.getType();
                switch (tt) {
                    case MATRIX:
                        wrap(f.getMatrix());
                        break;
                    case ALGEBRAIC_EXPRESSION:
                        wrap(f.getMathExpression().getExpression());
                        break;
                    case LIST:
                        wrap(f.getMatrix().getFlatArray());
                        break;
                    default:
                        break;
                }
                return this;
            }
            return null;
        }
    }

// In ExpressionSolver.getNextResult():
    public EvalResult getNextResult() {
        if (poolPointer >= pool.length) {
            // Expand pool if needed
            EvalResult[] newPool = new EvalResult[pool.length * 2];
            System.arraycopy(pool, 0, newPool, 0, pool.length);
            for (int i = pool.length; i < newPool.length; i++) {
                newPool[i] = new EvalResult();
            }
            // Note: You'd need to make pool non-final to do this
        }
        EvalResult result = pool[poolPointer++];
        result.reset();
        return result;
    }

// Add a reset method to clear the pool between evaluations
    private void resetPool() {
        poolPointer = 0;
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

    static class Test {

        public static void main(String... args) {
            System.out.println("=".repeat(80));
            System.out.println("POSTFIX EVALUATION ACCURACY TESTS");
            System.out.println("=".repeat(80));

            // Test 1: Basic arithmetic
            testExpression("2+3", 5.0, "Basic addition");
            testExpression("10-4", 6.0, "Basic subtraction");
            testExpression("3*4", 12.0, "Basic multiplication");
            testExpression("12/3", 4.0, "Basic division");
            testExpression("2^3", 8.0, "Basic exponentiation");

            // Test 2: Operator precedence
            testExpression("2+3*4", 14.0, "Precedence: multiply before add");
            testExpression("2*3+4", 10.0, "Precedence: multiply before add (reversed)");
            testExpression("2^3*4", 32.0, "Precedence: power before multiply");
            testExpression("2+3^2", 11.0, "Precedence: power before add");

            // Test 3: Parentheses
            testExpression("(2+3)*4", 20.0, "Parentheses override precedence");
            testExpression("2*(3+4)", 14.0, "Parentheses with multiplication");
            testExpression("((2+3)*4)+1", 21.0, "Nested parentheses");

            // Test 4: Single argument functions
            testExpression("sin(0)", 0.0, "sin(0) = 0", 1e-10);
            testExpression("cos(0)", 1.0, "cos(0) = 1", 1e-10);
            testExpression("sqrt(16)", 4.0, "sqrt(16) = 4", 1e-10);
            testExpression("abs(-5)", 5.0, "abs(-5) = 5", 1e-10);

            // Test 5: Nested functions
            testExpression("sin(cos(0))", Math.sin(Math.cos(0)), "Nested: sin(cos(0))", 1e-10);
            testExpression("sqrt(abs(-16))", 4.0, "Nested: sqrt(abs(-16))", 1e-10);

            // Test 6: Variable substitution
            MathExpression me = new MathExpression("x=5;2*x+3");
            double result = Double.parseDouble(me.solve());
            testValue(result, 13.0, "Variable: x=5; 2*x+3", 1e-10);

            // Test 7: Multi-argument functions
            testExpression("listsum(1,2,3,4,5)", 15.0, "listsum(1,2,3,4,5) = 15", 1e-10);
            testExpression("listsum(10,20,30)", 60.0, "listsum(10,20,30) = 60", 1e-10);

            // Test 8: Functions with grouped expressions
            testExpression("listsum((2+3),(4+1),5)", 15.0, "listsum with grouped args", 1e-10);
            testExpression("sort(3,-1,2,0,1)", "sorted array", "sort function");

            // Test 9: Complex nested expressions
            testExpression("sin(3.14159/2)", 1.0, "sin(π/2) ≈ 1", 0.001);
            testExpression("sqrt(2^2 + 3^2)", 3.605551275, "sqrt(4+9) = sqrt(13)", 1e-6);

            // Test 10: Mixed operators and functions
            testExpression("2*sin(0)+3*cos(0)", 3.0, "2*sin(0) + 3*cos(0) = 3", 1e-10);
            testExpression("sqrt(9)+2^3-1", 12.0, "sqrt(9) + 2^3 - 1 = 12", 1e-10);

            // Test 11: Edge cases
            testExpression("0+0", 0.0, "Zero addition");
            testExpression("1*1", 1.0, "One multiplication");
            testExpression("5-5", 0.0, "Subtraction to zero");

            System.out.println("\n" + "=".repeat(80));
            System.out.println("ACCURACY TEST SUMMARY");
            System.out.println("=".repeat(80));
        }

        private static void testExpression(String expr, double expected, String description) {
            testExpression(expr, expected, description, 1e-9);
        }

        private static void testExpression(String expr, double expected, String description, double tolerance) {
            try {
                MathExpression me = new MathExpression(expr);
                String resultStr = me.solve();
                double result = Double.parseDouble(resultStr);
                testValue(result, expected, description, tolerance);
            } catch (Exception e) {
                System.out.printf("❌ %s: EXCEPTION - %s%n", description, e.getMessage());
            }
        }

        private static void testExpression(String expr, String expectedType, String description) {
            try {
                MathExpression me = new MathExpression(expr);
                String result = me.solve();
                System.out.printf("✓ %s: %s%n", description, result);
            } catch (Exception e) {
                System.out.printf("❌ %s: EXCEPTION - %s%n", description, e.getMessage());
            }
        }

        private static void testValue(double actual, double expected, String description, double tolerance) {
            if (Math.abs(actual - expected) <= tolerance) {
                System.out.printf("✓ %s: %.10f (expected %.10f)%n", description, actual, expected);
            } else {
                System.out.printf("❌ %s: got %.10f, expected %.10f (diff: %.2e)%n",
                        description, actual, expected, Math.abs(actual - expected));
            }
        }
    }

    public static void main(String... args) {

        Test.main(args);

        System.out.println("________________________________________________________________TESTS-DONE________________________________________________________________");
        System.out.println("sum(32,12,10,18,1,3,2,5,6): -->> " + new MathExpression("sum(32,12,10,18,1,3,2,5,6)").solve());
        MathExpression lin = new MathExpression("linear_sys(2,3,-5,3,-4,20)");
        System.out.println("linear_sys(2,3,-5,3,-4,20): -->> " + lin.solve());
        MathExpression lin1 = new MathExpression("linear_sys(2,3,-5,12,3,-4,8,20,7,-6,2,18)");
        System.out.println("linear_sys(2,3,-5,12,3,-4,8,20,7,-6,2,18): -->> " + lin1.solve());
        System.out.println("linear_sys(12,16, 24,15,21, 24): -->> " + new MathExpression("linear_sys(@(2,3)(12,16, 24,15,21, 32))").solve());
        System.out.println("det(2,3,-5,12,3,-4,8,20,7,-6,2,18,4,2,18,5): -->> " + new MathExpression("det(@(4,4)(2,3,-5,12,3,-4,8,20,7,-6,2,18,4,2,18,5))").solve());
        System.out.println("det(@(4,4)(2,3,-5,12,3,-4,8,20,7,-6,2,18,4,2,18,5)): -->> " + new MathExpression("det(@(4,4)(2,3,-5,12,3,-4,8,20,7,-6,2,18,4,2,18,5))").solve());
        System.out.println("sin(5,6)-->>" + new MathExpression("sin(5,6)").solve());
        System.out.println("√81 + ³√(27) + 2^10-->>" + new MathExpression("√81 + ³√(27) + 2^10").solve());
        System.out.println(new MathExpression("5! + 9Р3 + 6Č5").solve());
        System.out.println(new MathExpression("f(x,y)=2*x*y;f(3,4);").solve());
        System.out.println("sum(sum(5),sum(6)): " + new MathExpression("sum(sum(5),sum(6))").solve());
        System.out.println("sum(sum(5,3,4,5),sum(6,12,14,1,2,1)): " + new MathExpression("sum(sum(5,3,4,5),sum(6,12,14,1,2,1))").solve());
        System.out.println("prod(sin(5),sin(5)): " + new MathExpression("prod(sin(5),sin(5))").solve());

        System.out.println("differential calculus:>>1 " + new MathExpression("diff(@(x)sin(ln(x)), 1);").solve());
        System.out.println("differential calculus:>>2 " + new MathExpression("diff(@(x)x^10, m);").solve());
        System.out.println("differential calculus:>>2a " + new MathExpression("diff(@(x)x^10, n,1);").solve());
        MathExpression meDiff = new MathExpression("x=3;diff(@(x)sin(ln(x)),vw);");
        System.out.println("--------------------------" + FunctionManager.FUNCTIONS);
        System.out.println("differential calculus:>>3 " + meDiff.solve());
        System.out.println("differential calculus:>>4 " + new MathExpression("diff(@(x)sin(ln(x)), b);").solve());
        System.out.println(new MathExpression("sin(ln(x));").solve());
        System.out.println("FUNCTIONS: " + FunctionManager.FUNCTIONS);
// Expected: 11

        System.out.println("sort(-3,8,3,2,6,-7,9,1,0,-1): " + new MathExpression("sort(-3,8,3,2,6,-7,9,1,0,-1)").solve());
        System.out.println("sort(0,4+0,2+0): " + new MathExpression("sort(0,4+0,2+0)").solve());
        System.out.println("sort(3+1,-3): " + new MathExpression("sort(3+1, -3)").solve());
        System.out.println("sort(4+2): " + new MathExpression("sort(4+2)").solve());
        System.out.println(new MathExpression("x=0.9;sqrt(0.64-x^2)").solve());

        MathExpression mex = new MathExpression("A=@(3,3)(3,4,2,9,12,5,4,1,2);B=eigvalues(A);C=eigvec(A);D=eigpoly(A);eigpoly(A);");
        System.out.println(FunctionManager.FUNCTIONS);
        System.out.println("----------" + mex.solve());

        Function f = FunctionManager.add("f(x,y) = x - x/y");
        double r = f.calc(2, 3);
        System.out.println("r = " + r);
        int iterations = 1;
        long start = System.nanoTime();
        double vvv[] = new double[1];
        for (int i = 1; i <= iterations; i++) {
            vvv[0] = f.calc(2, 3);
        }
        double duration = (System.nanoTime() - start) / iterations;
        System.out.println("dur = " + (duration / 1000) + " microns, ans = " + vvv[0]);//μμμ

        String geom = "geom((2,8,4))+geom(((2,8,4)))";
        System.out.println(geom + ": " + new MathExpression(geom).solve());
        String sum = "sum(3,1,4,5,9)";
        System.out.println(sum + "--->>" + new MathExpression(sum).solve());
        sum = "sum(3+3)";
        System.out.println(sum + "--->>" + new MathExpression(sum).solve());
        sum = "sum(sum(5),sum(6))";
        System.out.println(sum + "--->>" + new MathExpression(sum).solve());

        sum = "sum(sin(5),sin(5))";
        System.out.println(sum + "--->>" + new MathExpression(sum).solve());

        sum = "listsum(listsum(5))";
        System.out.println(sum + "--->>" + new MathExpression(sum).solve());
        System.out.println("____________________________________________________________________________________________________________________________________________");

        String prod = "prod(5)";
        System.out.println(prod + "--->>" + new MathExpression(prod).solve());
        prod = "prod(5+5)";
        System.out.println(prod + "--->>" + new MathExpression(prod).solve());
        prod = "prod(5+5)";
        System.out.println(prod + "--->>" + new MathExpression(prod).solve());
        System.out.println(new MathExpression("sort(3,4,-2,-13,21,34,-99,1,-1.498,sin(28))").solve());
        System.out.println(new MathExpression("3/2").solve());
        System.out.println(new MathExpression("x=20;sin(ln(x));").solve());
        System.out.println(new MathExpression("x=20;sin(listsum(3,9,sin(19),cos(21),4,13,2))").solve());
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
        System.out.println("s--->" + ms.solve());
        String ss = "listsum(sin(5),sin(5))";
        System.out.println("ss--->" + new MathExpression(ss).solve());
        String s1 = "x=9;g(x)=((sin(x)-tan(x)));v(x)=((ln(x)/tan(x)));f(x)=g(x);sin(1.75)+cos(1.23)+tan(1.86-0.26)+log(10,3)+sqrt(16)+exp(1)+pow(2,8)+abs(-42)+listsum(1,2,3,4,5)+sin(3*12+cos(55))-(4+5)*(2*(9-2)+12*(4-7));v(3);";
        String s2 = "v(x)=((ln(x)/tan(x)));2+v(3);";
        String s3 = "2+sin(3)-ln(12)+1/3.689";
        String s4 = "sin(2)+cos(3)-listsum(2,3,4,sin(2),ln(42),3,4,5,6,1,2,3,45,2)+12";
        String s5 = "listsum(sin(3),cos(3),ln(345),sort(3,-4,5,-6,13,2,4,5,listsum(3,4,5,6,9,12,23), 12, listsum(3,4,8,9, 2000)), 12000, mode(3,2,2,1),32.897, mode(1,5,7,7,1,1,7))";

        MathExpression m = new MathExpression(s5);
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
