/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.InputMismatchException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.github.gbenroscience.interfaces.Savable;
import com.github.gbenroscience.interfaces.Solvable;
import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.math.Main;
import com.github.gbenroscience.math.Maths;
import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import static com.github.gbenroscience.parser.Number.fastParseDouble;
import static com.github.gbenroscience.parser.Number.isNumber;
import static com.github.gbenroscience.parser.Operator.isAssignmentOperator;
import static com.github.gbenroscience.parser.Operator.isBinaryOperator;
import static com.github.gbenroscience.parser.Operator.isBracket;
import static com.github.gbenroscience.parser.Operator.isComma;
import static com.github.gbenroscience.parser.Operator.isLogicOperator;
import static com.github.gbenroscience.parser.Operator.isOpeningBracket;
import static com.github.gbenroscience.parser.Operator.isUnaryPostOperator;
import static com.github.gbenroscience.parser.Operator.isUnaryPreOperator;
import static com.github.gbenroscience.parser.Operator.validateAll;
import static com.github.gbenroscience.parser.TYPE.ALGEBRAIC_EXPRESSION;
import static com.github.gbenroscience.parser.TYPE.MATRIX;
import static com.github.gbenroscience.parser.TYPE.VECTOR;
import static com.github.gbenroscience.parser.Variable.isVariableString;
import com.github.gbenroscience.parser.benchmarks.GG;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.Help;
import com.github.gbenroscience.parser.methods.Method;
import com.github.gbenroscience.parser.methods.MethodRegistry;
import com.github.gbenroscience.parser.turbo.tools.FastCompositeExpression;
import com.github.gbenroscience.parser.turbo.tools.MatrixTurboEvaluator;
import com.github.gbenroscience.parser.turbo.tools.ScalarTurboEvaluator;
import com.github.gbenroscience.parser.turbo.tools.TurboExpressionEvaluator;
import com.github.gbenroscience.util.ErrorLog;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.Serializer;
import com.github.gbenroscience.util.VariableManager;
import java.util.LinkedHashMap;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * <p style="font-weight:'bold';color:'red'; font-size:'2em';">
 * This class is NOT thread-safe. If multiple threads access a MathExpression
 * concurrently, they must synchronize externally or use separate instances.
 *
 * </p>
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
 * an high speed one. But if the expression containsAlgebraicFunction
 * statistical functions or user defined ones, the self-referential processes
 * modify the scanner output and so this scanner output cannot be reliably
 * referred to later on by iterative processes or any process that seeks to
 * reuse the scanner's output.
 *
 * </p>
 *
 * @author GBENRO
 */
public class MathExpression implements Savable, Solvable {

    private final ErrorLog errorLog = new ErrorLog();
    private static final long serialVersionUID = 1L;
    /**
     * Backup the alias from the scanner here
     */
    private String commaAlias;
    public ParserResult parser_Result = ParserResult.VALID;
    //determines the mode in which trig operations will be carried print on numbers.if DRG==0,it is done in degrees
//if DRG==1, it is done in radians and if it is 2, it is done in grads.
    private DRG_MODE DRG = Declarations.degGradRadFromVariable();
    private ArrayList<String> whitespaceremover = new ArrayList<>();//used to remove white spaces from the ArrayList
    /**
     * The expression to evaluate.
     */
    private String expression;
    private MathExpressionTreeDepth.Result treeStats;
    protected boolean correctFunction = true;//checks if the function is valid.
    protected int noOfListReturningOperators;
    protected List<String> scanner = new ArrayList<>();//the ArrayList that stores the scanner input function
    private boolean optimizable;

    protected boolean hasListReturningOperators;
    /**
     * If set to true, the constants folding algorithm will be run to further
     * optimize the compiled postfix and so make the speed of evaluation faster.
     */
    private boolean willFoldConstants;

    private ExpressionSolver expressionSolver;

    private FastCompositeExpression compiledTurbo = null;
    private boolean turboCompiled = false;

    /**
     * If set to true, MathExpression objects will automatically initialize
     * undeclared variables to zero and store them.
     */
    private static boolean autoInitOn = true;
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
    private transient VariableManager variableManager;

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

    private int[] slots;
    private double[] executionFrame;
    VariableRegistry registry = new VariableRegistry();

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
    public static final class Token implements Savable {

        private static final long serialVersionUID = 1L;
        public static final int NUMBER = 0, OPERATOR = 1, FUNCTION = 2, METHOD = 3, LPAREN = 5, RPAREN = 6, COMMA = 7;
        public int kind;
        public double value;
        public String name; // REQUIRED for functions/methods/variables
        /**
         * The slot in the execution frame where this token's value lives. Used
         * only when kind == NUMBER and it represents a variable.
         */
        public int frameIndex = -1;
        public Variable v;//cache the variable here

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
        private String[] rawArgs;

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
        public Token(int kind, String name, int arity, int id, ErrorLog log) {
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
                    String err = "Function " + this.name + " not found";
                    log.info(err);
                    throw new RuntimeException(err);
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
        public Token(int kind, String name, int arity, int id, String assignToName, ErrorLog log) {
            this(kind, name, arity, id, log);
            this.assignToName = assignToName;
            this.isAssignmentTarget = (assignToName != null);
        }

        public Token clone() {
            Token t = new Token(kind);
            t.action = action;
            t.arity = arity;
            t.assignToName = assignToName;
            t.frameIndex = frameIndex;
            t.id = id;
            t.isAssignmentTarget = isAssignmentTarget;
            t.isPostfix = isPostfix;
            t.isRightAssoc = isRightAssoc;
            t.name = name;
            t.opChar = opChar;
            t.precedence = precedence;
            if (rawArgs != null) {
                t.rawArgs = new String[rawArgs.length];
                for (int i = 0; i < rawArgs.length; i++) {
                    t.rawArgs[i] = rawArgs[i];
                }
            }
            if (v != null) {
                t.v = new Variable(v.getName(), v.getValue());
            }
            t.value = value;
            return t;
        }

        public String[] getRawArgs() {
            return rawArgs;
        }

        public boolean isVariable() {
            return this.name != null && !this.name.isEmpty() && this.v != null && this.frameIndex >= 0;
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
                    + "\"kind\": \"" + kind + "\",\n"
                    + "\"value\": " + value + ",\n"
                    + "\"name\": \"" + name + "\",\n"
                    + "\"id\": " + id + ",\n"
                    + "\"opChar\": " + opChar + ",\n"
                    + "\"precedence\": " + precedence + ",\n"
                    + "\"isRightAssoc\": " + isRightAssoc + ",\n"
                    + "\"isPostfix\": " + isPostfix + ",\n"
                    + "\"arity\": " + arity + ",\n"
                    + "\"rawArgs\": " + Arrays.toString(rawArgs) + ",\n"
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
                    + "\"rawArgs\": " + Arrays.toString(rawArgs) + ",\n"
                    + "\"assignToName\": " + assignToName + ","
                    + "\"isAssignmentTarget\": " + isAssignmentTarget + ","
                    + "\n\"v\": " + (v == null ? "null" : v.toJSON()) + "\n"
                    + "}";
        }

    }

    /**
     * Used internally to have a a cheap way of accessing an empty
     * {@linkplain MathExpression} object
     *
     * @param plain
     */
    private MathExpression(boolean plain) {

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
     * @param input The function to be evaluated. The general format
     * containsAlgebraicFunction variable, constant and function declarations
     * for variables, constants and functions that are not yet initialized,
     * assignment expressions for those that have been initialized and then an
     * expression to evaluate. e.g. x = -12; y =x+1/12; const x1,x2,x3=10; z
     * =sin(3x-1)+2.98cos(4x);cos(3x+12); The last expression is a function to
     * be evaluated and it is always without any equals sign and may or may not
     * end with a semicolon.
     *
     */
    public MathExpression(String input) {
        this(input, new VariableManager(), true);
    }//end constructor MathExpression

    public MathExpression(String input, boolean foldConstants) {
        this(input, new VariableManager(), foldConstants);
    }//end constructor MathExpression

    /**
     * MathExpression me = new
     * MathExpression("rot(f=@(x)sin(x),pi/2,@(1,3)(0,0,0),@(1,3)(1,1,0))");
     * scanner-output: [rot, (, (, f, =, anon1, ), ,, (, pi, /, 2, ), ,, anon2,
     * ,, anon3, )] MathExpression ma = new
     * MathExpression("rot(t(x)=ln(x),pi/2,@(1,3)(0,0,0),@(1,3)(1,1,0))");
     * scanner-output: [rot, (, (, t, (, x, ), =, ln, (, x, ), ), ,, (, pi, /,2,
     * ), ,, anon1, ,, anon2, )] MathExpression m1 = new
     * MathExpression("rot(@(1,3)(4,2,0),pi/2,@(1,3)(0,0,0),@(1,3)(1,1,0))");
     * scanner-output: [rot, (, anon1, ,, (, pi, /, 2, ), ,, anon2, ,, anon3, )]
     * MathExpression m = new
     * MathExpression("rot(@(x)2*x+1,pi/2,@(1,3)(0,0,0),@(1,3)(1,1,0))");
     * scanner-output: [rot, (, anon1, ,, (, pi, /, 2, ), ,, anon2, ,, anon3, )]
     *
     * @param data
     */
    private String processRotScenarios(List<String> data) {
        if (whitespaceremover.isEmpty()) {
            whitespaceremover.add("");
        }
        int eqIdx = data.indexOf("=");
        if (eqIdx != -1) {
            String prev = data.get(eqIdx - 1);
            String next = data.get(eqIdx + 1);
            if (prev.equals(")")) {// [rot, (, (, t, (, x, ), =, ln, (, x, ), ), ,, (, pi, /,2, ), ,, anon1, ,, anon2, )] 
                int fnameIndex = Bracket.getComplementIndex(false, eqIdx - 1, data) - 1;
                String fnNameToLeftOfEquals = data.get(fnameIndex);
                int openBracIdxForFnAssignExpression = fnameIndex - 1;
                int closeBracIdxForFnAssignExpression = Bracket.getComplementIndex(true, openBracIdxForFnAssignExpression, data);
                List<String> l = data.subList(openBracIdxForFnAssignExpression, closeBracIdxForFnAssignExpression + 1);
                final StringBuilder s = new StringBuilder();
                l.forEach(new Consumer<String>() {
                    @Override
                    public void accept(String t) {
                        s.append(t);
                    }
                });

                Function f = FunctionManager.add(s.toString());
                l.clear();
                l.add(fnNameToLeftOfEquals);
            } else {
                if (isVariableString(prev) && FunctionManager.containsAny(next)) {//e.g f, = , anon1
                    Function.assignAnonymousToNamedFunction(prev, next);
                    data.set(eqIdx, "");//squash consumed token
                    data.set(eqIdx + 1, "");//squash consumed token

                    if (data.get(eqIdx - 2).equals("(")) {// (, f, =, anon1
                        int closeBracIdx = Bracket.getComplementIndex(true, eqIdx - 2, data);
                        if (closeBracIdx == eqIdx + 2) {// (, f, =, anon1, )
                            data.set(eqIdx - 2, "");//squash consumed token
                            data.set(eqIdx + 2, "");//squash consumed token
                        }
                    }
                    data.removeAll(whitespaceremover);
                }
            }
        }

        final StringBuilder s = new StringBuilder();
        data.forEach(new Consumer<String>() {
            @Override
            public void accept(String t) {
                s.append(t);
            }
        });

        return s.toString();
    }

    public MathExpression(String input, VariableManager variableManager, boolean foldConstants) {
        this.willFoldConstants = foldConstants;
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
            int indexOfRot = code.indexOf("rot(");
            if (indexOfRot != -1) {//rot(f(x)=sin(x),ang,P1,P2) or rot(f=@(x)sin(x),ang,P1,P2)
                //process rot function if it contains =

                int openBracIdx = indexOfRot + "rot".length();
                int idxOfCloseBracForRot = Bracket.getComplementIndex(true, openBracIdx, code);
                int eqIdx = code.indexOf("=", indexOfRot);
                if (eqIdx != -1 && eqIdx < idxOfCloseBracForRot) {//rot(f,=,jjj,....) e.g equals index lies inside rot's bracket contents
                    MathScanner sc = new MathScanner(code.substring(indexOfRot, idxOfCloseBracForRot + 1));
                    sc.scanner();
                    errorLog.copyFrom(sc.errorLog);
                    sc.unmaskCommas();
                    code = processRotScenarios(sc.getScanner());
                }

            }
            if (code.contains("=")) {
                boolean success = Function.assignObject(code + ";", errorLog);
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
        this.slots = registry.getSlots();
    }

    public double[] getExecutionFrame() {
        return executionFrame;
    }

    public static final VariableRegistry createNewVariableRegistry() {
        return new VariableRegistry();
    }

    public String getExpression() {
        return expression;
    }

    /**
     * @param expression The expression
     */
    public final void setExpression(String expression) {
        if (!expression.equals(this.expression)) {
            invalidateTurbo();  // Clear turbo cache
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

    public FastCompositeExpression getCompiledTurbo() {
        return compiledTurbo;
    }

    public void copyErrorLogTo(ErrorLog log) {
        log.copyFrom(errorLog);
    }

    private void initializing(String expression) {
        computeTreeDepth();
        setCorrectFunction(true);
        setHasListReturningOperators(false);
        setNoOfListReturningOperators(0);
        whitespaceremover.add("");
        //Scanner operation

        MathScanner opScanner = new MathScanner(expression);
        opScanner.scanner(variableManager);
        errorLog.copyFrom(opScanner.errorLog);
        for (Variable v : opScanner.foundVariables) {
            v.setFrameIndex(registry.getSlot(v.getName()));
        }

        this.commaAlias = opScanner.commaAlias;
        scanner = opScanner.getScanner();

        correctFunction = opScanner.isRunnable();
        parser_Result = opScanner.parser_Result;

        if (parser_Result == ParserResult.VALID) {
            //refixCommas(); 
            statsVerifier();
            refixCommas();
            mapBrackets();
            functionComponentsAssociation();
            compileToPostfix();  // Compile once if not already done 
        }//end if

    }//end method initializing(args)

    public void setWillFoldConstants(boolean willFoldConstants) {
        boolean changed = willFoldConstants != this.willFoldConstants;
        this.willFoldConstants = willFoldConstants;
        if (changed) {
            compileToPostfix();
        }
    }

    public boolean isWillFoldConstants() {
        return willFoldConstants;
    }

    private void removeCommas() {
        scanner.replaceAll((String t) -> isComma(t) ? this.commaAlias : t);
    }

    private void refixCommas() {
        scanner.replaceAll((String t) -> this.commaAlias.equals(t) ? "," : t);
    }

    /**
     * Check if turbo mode is available.
     */
    public boolean isTurboCompiled() {
        return turboCompiled && compiledTurbo != null;
    }

    /**
     * Invalidate turbo compilation when expression changes.
     */
    private void invalidateTurbo() {
        compiledTurbo = null;
        turboCompiled = false;
    }

    /**
     * Compile to Turbo mode using adaptive compiler selection. Auto-selects
     * scalar or matrix compiler based on expression analysis.
     *
     * Performance: - Pure scalar: ~5-10 ns - Matrix ops: ~50-1000 ns (vs 5-100
     * μs interpreted) - First call: ~5-10 ms (compilation), cached thereafter
     *
     * @return FastCompositeExpression ready for high-speed evaluation
     * @throws IllegalStateException if expression not properly compiled
     * @throws Throwable if turbo compilation fails
     */
    public FastCompositeExpression compileTurbo() throws Throwable {
        if (turboCompiled && compiledTurbo != null) {
            return compiledTurbo;
        }

        if (!isScannedAndOptimized() || cachedPostfix == null) {
            String err = "Expression not properly compiled. " + (cachedPostfix == null ? "RPN tokens array empty" : " Expression not scanned and optimized");
            errorLog.info(err);
            throw new IllegalStateException(err);
        }

        try {
            // Analyze expression to determine best compiler
            boolean hasMatrixOps = hasMatrixOperations(cachedPostfix);

            TurboExpressionEvaluator compiler;
            if (!hasMatrixOps) {
                //  System.print.println("SELECTED ScalarTurboCompiler");
                // Pure scalar expressions: use ultra-fast scalar compiler (~5ns)
                compiler = new ScalarTurboEvaluator(this);
            } else {
                //   System.print.println("SELECTED FlatMatrixTurboCompiler");
                // Any matrix operations: use flat-array optimized compiler (~50-1000ns)
                compiler = new MatrixTurboEvaluator(this);
            }

            compiledTurbo = compiler.compile();
            turboCompiled = true;
            return compiledTurbo;

        } catch (Throwable e) {
            String err = "Failed to compile expression to turbo: " + e.getMessage();
            errorLog.info(err);
            throw new RuntimeException(
                    err, e);
        }
    }

    /**
     * Detect if postfix contains matrix operations. Used to select appropriate
     * turbo compiler.
     */
    private boolean hasMatrixOperations(Token[] postfix) {
        for (Token t : postfix) {
            // 1. Check for explicit matrix functions (Your existing logic)
            if (t.kind == Token.FUNCTION || t.kind == Token.METHOD) {
                String name = t.name.toLowerCase();
                if (name.contains("matrix") || name.equals("det") || name.equals("invert")
                        || name.equals("inverse") || name.equals("transpose") || Method.isMatrixMethod(t.name)) {
                    return true;
                }
            }

            Function func = FunctionManager.lookUp(t.name);
            // 2. Check for Matrix Literals (@ notation)
            if (func != null && func.getType() == TYPE.MATRIX) {
                return true;
            }

            // 3. THE CRITICAL FIX: Check if a named variable is actually a Matrix
            if (t.name != null && !t.name.isEmpty()) {
                if (func != null && func.getType() == TYPE.MATRIX) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     *
     * @return the DRG value:0 for degrees, 1 for rads, 2 for grads
     */
    public DRG_MODE getDRG() {
        return DRG;
    }

    public void setDRG(DRG_MODE DRG) {
        if (DRG != this.DRG) {
            invalidateTurbo();  // Clear turbo cache when mode changes
            this.DRG = DRG;
            this.cachedPostfix = null;  // Invalidate cache
            this.poolPointer = 0;

            // Recompile for new mode if we have a valid expression
            if (scanner != null && !scanner.isEmpty() && correctFunction) {
                compileToPostfix();
            }
        }
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

    private void computeTreeDepth() {
        treeStats = new MathExpressionTreeDepth(expression).calculate();
    }

    public MathExpressionTreeDepth.Result getTreeStats() {
        return treeStats;
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

    public VariableRegistry getRegistry() {
        return registry;
    }

    public boolean hasVariable(String var) {
        return registry.hasVariable(var);
    }

    public String[] getVariablesNames() {
        return registry.getVariables();
    }

    public int[] getSlots() {
        return slots;
    }

    /**
     *
     * @return an array of {@link Slot} objects(name of variable and frame index
     * of variables in the expression)
     */
    public Slot[] getSlotItems() {
        ArrayList<Slot> slots = new ArrayList<>();
        for (int i = 0; i < cachedPostfix.length; i++) {
            Token t = cachedPostfix[i];
            if (t.v != null) {
                Slot s = new Slot(t.v.getName(), t.frameIndex);
                if (!slots.contains(s)) {
                    slots.add(s);
                }
            }//end if
        }//end for loop
        return slots.toArray(new Slot[0]);
    }

    public Pair<String[], Integer[]> getVariables() {
        return registry.getVarsAndSlots();
    }

    /**
     * Retrieves a Variable handle from the expression's registry. This handle
     * is "Pre-Bound" to the correct slot in the execution frame.
     *
     * @param name
     * @return {@link Variable}
     */
    public Variable getVariable(String name) {
        // 1. Check if this variable exists in our compiled registry
        int slot = this.registry.getSlot(name);
        // 2. Look up the variable in our existing token list or VariableManager
        Variable v = VariableManager.lookUp(name);
        if (v == null) {
            // Fallback: Create a new one if it's a dynamic variable
            v = new Variable(name);
        }
        // 3. IMPORTANT: Sync the index so the Handle knows where to error
        v.setFrameIndex(slot);
        return v;
    }

    public static final class Slot implements Savable {

        private static final long serialVersionUID = 1L;
        /**
         * The slot or frame index
         */
        private final int slot;
        /**
         * The name of the Variable that owns this slot
         */
        private final String name;

        public Slot(String name, int slot) {
            this.slot = slot;
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public int getSlot() {
            return slot;
        }

        @Override
        public String toString() {
            return "name=" + name + ", slot=" + slot;
        }

        public String toJSON() {
            return "{\n"
                    + "\"name\": \"" + name + "\",\n"
                    + "\"slot\": \"" + slot + "\"\n"
                    + "}";
        }

    }

    public String checkErrorLogs() {
        String logs = errorLog.getLogs();
        errorLog.print();
        return logs;
    }

    private boolean isCommaAlias(String token) {
        return token.equals(commaAlias);
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
        errorLog.info(ListReturningStatsMethod.getErrorMessage());

        if (noOfListReturningOperators > 0 && correctFunction) {
            setHasListReturningOperators(true);
            //disable usage of multiple list operators in function here by un-commenting the statement below
            // correctFunction=false;

            if (isHasListReturningOperators()) {

                scanner.remove(0);//temporarily remove the starting brackets
                scanner.remove(scanner.size() - 1);//temporarily remove the ending brackets

                for (int i = 0; i < scanner.size(); i++) {
                    try {
                        if (i + 1 < scanner.size() && !Method.isListReturningStatsMethodThatAllowsAlgebraicOps(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                            if (isBinaryOperator(scanner.get(i - 1))) {
                                errorLog.info("1. Invalid Association Discovered For: \"" + scanner.get(i - 1) + "\" And \"" + scanner.get(i) + "\".\n");
                                correctFunction = false;
                                break;
                            }
                            if (isOpeningBracket(scanner.get(i - 1))
                                    && !Method.isNumberReturningStatsMethod(scanner.get(i - 2))
                                    && !Method.isListReturningStatsMethod(scanner.get(i - 2))
                                    && !isCommaAlias(scanner.get(i - 2))) { 
                                errorLog.info("2. Invalid Association Discovered For: \"(\" And " + scanner.get(i - 2) + " And \"" + scanner.get(i - 1) + "\" And \"" + scanner.get(i) + "\"\n ");
                                correctFunction = false;
                                break;
                            }

                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                        errorLog.info("Let developer handle the indexing of this checks properly!");
                    }

                }//end for

                scanner.add(0, "(");
                scanner.add(")");

                if (!correctFunction) {
                    parser_Result = ParserResult.SYNTAX_ERROR;
                    errorLog.info("Verifier discovers invalid association between data set returning operators:");
                }
            }//end if isHasListReturningOperator
        }
    }//end method statsVerifier

    public Token[] getCachedPostfix() {
        return cachedPostfix;
    }

    /**
     *
     * @param scanner The ArrayList object that holds the string values in the
     * scanned function.
     * @return a Bracket array that holds related brackets pairs.
     */
    private Bracket[] mapBrackets(List<String> scanner) {
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
                    String err = "Unmatched closing bracket at index " + i;
                    errorLog.info(err);
                    throw new InputMismatchException(err);
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
            String err = "Unmatched opening bracket(s) remain";
            errorLog.info(err);
            throw new InputMismatchException(err);
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
            errorLog.error(ime);
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
                            errorLog.info("ParserNG Does Not Allow " + expression + " To Combine The MathExpression Members \"" + scanner.get(i - 1) + "\" And \"" + scanner.get(i) + "\"\n");
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
                            errorLog.info("ParserNG Does Not Allow " + expression + " To Combine The MathExpression Members \"" + scanner.get(i) + "\" And \"" + scanner.get(i + 1) + "\"PLUS As You Have Done.\n");
                            correctFunction = false;
                            scanner.clear();
                            break;
                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                        errorLog.error(ind);
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
                    if (isVariableString(varName) && !isOpeningBracket(scan.get(i + 1))) {
                        Variable v = VariableManager.lookUp(varName);
                        if (v != null) {
                            scan.set(i, String.valueOf(v.getValue()));
                        }

                    }//end if
                } else {//no next token exists
                    if (isVariableString(varName)) {
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
     *
     * public void setValue(String name, double value) throws
     * NullPointerException, NumberFormatException { Variable v =
     * VariableManager.lookUp(name); if (v != null) { v.setValue(value); } }
     */
    public void setValue(String name, double value) {
        // 1. Ask the Registry where this name lives (O(1) Map lookup)
        // This replaces the entire for-loop and String.equals()
        int slot = this.registry.getSlot(name);
        if (slot != -1 && slot < executionFrame.length) {
            // 2. Write directly to the frame
            this.executionFrame[slot] = value;//.v.setValue(value);
        }
    }

    /**
     * Updates specific slots with specific values. Usage: updateArgs(new
     * int[]{0, 2}, new double[]{valX, valZ});
     *
     * @param values
     */
    public void updateArgs(double... values) {
        // 1. Guard check outside the loop to prevent bounds checks inside the loop
        int limit = Math.min(slots.length, values.length);
        // 2. Trust your compiler. If you mapped slots correctly at compile time,
        // you don't need to check executionFrame.length on every iteration.
        for (int i = 0; i < limit; i++) {
            this.executionFrame[slots[i]] = values[i];
        }
    }
    //  parserNG.updateSlot(slots[i], xValues[i]);
    // Or, for a single variable make (the most common benchmark case):

    public void updateSlot(int slot, double value) {
        if (slot >= 0 && slot < this.executionFrame.length) {
            this.executionFrame[slot] = value;
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
            EvalResult r = Variable.lastResult = expressionSolver.evaluate();
            returnType = r.getType();
            return r;
        }
        if (scanner == null || scanner.isEmpty() || !correctFunction || parser_Result != ParserResult.VALID) {
            return EvalResult.ERROR;
        }
        return EvalResult.ERROR;
    }

    public EvalResult solveGeneric(double... args) {
        this.executionFrame = args;
        return solveGeneric();
    }

    @Override
    public String solve() {
        return solveGeneric().toString();
    }//end method solve()

    public String solve(double... args) {
        return solveGeneric(args).toString();
    }//end method solve()

    protected List<String> solve(List<String> list) {
        return Arrays.asList(String.valueOf(GG.evaluate(list)));
    }

    // Automatically strips unbalanced grouping parentheses from captured arguments
    private static String cleanArgument(String arg) {
        if (arg == null) {
            return "";
        }
        arg = arg.trim();
        int open = 0, close = 0;

        for (int i = 0; i < arg.length(); i++) {
            if (arg.charAt(i) == '(') {
                open++;
            } else if (arg.charAt(i) == ')') {
                close++;
            }
        }

        // Strip leading redundant '('
        while (open > close && arg.startsWith("(")) {
            arg = arg.substring(1).trim();
            open--;
        }

        // Strip trailing redundant ')'
        while (close > open && arg.endsWith(")")) {
            arg = arg.substring(0, arg.length() - 1).trim();
            close--;
        }

        return arg;
    }

    private Token translate(String s, String next) {
        if (s == null || s.isEmpty()) {
            return null;
        }
        char c0 = s.charAt(0);
        int len = s.length();
        Variable vv = null;
        boolean isNumber = false;
        // 1. Identify Numbers and constants here
        if ((isNumber = isNumber(s)) || ((vv = VariableManager.lookUp(s)) != null && vv.isConstant())) {
            return new Token(isNumber ? fastParseDouble(s) : vv.getValue());
        }

        if ((isNumber = isNumber(s))) {
            return new Token(fastParseDouble(s));
        }
        if (isVariableString(s)) {
            vv = VariableManager.lookUp(s);
            if (vv != null) {
                if (vv.isConstant()) {
                    return new Token(vv.getValue());
                } else {
                    Token t = new Token(0.0);
                    t.name = s;
                    t.v = vv;
                    t.frameIndex = registry.getSlot(s);
                    t.v.setFrameIndex(t.frameIndex);
                    t.kind = Token.NUMBER;
                    return t;
                }
            }
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
                return new Token(Token.FUNCTION, s, -1, -1, errorLog); // Arity set during compile
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
            return new Token(Token.METHOD, transformedMethodName, -1, methodId, errorLog); // Arity set during compile
        }

        // 6. Fallback: Treat as Variable/Constant
        Token t = new Token(0.0);
        t.name = s;
        if (!FunctionManager.containsAlgebraicFunction(s) && isVariableString(s)) {
            if (vv == null) {
                vv = new Variable(s, 0.0);
            }
            t.v = vv;
            t.frameIndex = registry.getSlot(s);
            t.v.setFrameIndex(t.frameIndex);

        }
        t.kind = Token.NUMBER;
        return t;
    }

    public void readErrorLog() {
        errorLog.print();
    }

    private void compileToPostfix() {

        if (cachedPostfix != null) {
            return;
        }

        // --- 1. THE FIX: Passive Listeners for Function Arguments ---
        class FuncArgTracker {

            Token funcToken;
            int depthLevel; // The exact paren depth this function operates at
            List<String> args = new ArrayList<>();
            StringBuilder currentArg = new StringBuilder();

            FuncArgTracker(Token funcToken, int depthLevel) {
                this.funcToken = funcToken;
                this.depthLevel = depthLevel;
            }
        }

        // Using a List so we can broadcast token strings to all open functions simultaneously
        List<FuncArgTracker> trackers = new ArrayList<>();
        int currentParenDepth = 0;

        // --- 2. STANDARD SHUNTING YARD STACKS ---
        Stack<Token> opStack = new Stack<>();
        Stack<Boolean> isFuncParenStack = new Stack<>(); // Tracks if a '(' belongs to a function

        Token[] postfix = new Token[scanner.size() * 2];
        int p = 0;

        int len = scanner.size();
        for (int idx = 0; idx < len; idx++) {
            String s = scanner.get(idx);
            String next = idx + 1 < len ? scanner.get(idx + 1) : null;

            Token t = translate(s, next);
            if (t == null) {
                continue;
            }

// ==========================================
            // PHASE A: PASSIVE STRING TRACKING
            // ==========================================
            boolean isFuncParen = false;

            if (t.kind == Token.LPAREN) {
                currentParenDepth++;
                if (!opStack.isEmpty() && (opStack.peek().kind == Token.FUNCTION || opStack.peek().kind == Token.METHOD)) {
                    isFuncParen = true;
                    // Turn on a new microphone for this function at the current depth
                    trackers.add(new FuncArgTracker(opStack.peek(), currentParenDepth));
                }
            }

            // Broadcast the current token string 's' to all open trackers
            for (int i = 0; i < trackers.size(); i++) {
                FuncArgTracker tracker = trackers.get(i);

                // FIX: A comma belongs to this function if it is the innermost open function
                // This allows the comma to penetrate redundant grouping parentheses!
                boolean isOwningComma = (t.kind == Token.COMMA && i == trackers.size() - 1);

                // The function still closes exactly when its base parenthesis depth resolves
                boolean isOwningRParen = (t.kind == Token.RPAREN && currentParenDepth == tracker.depthLevel);

                // Prevent the function from capturing its own opening parenthesis
                boolean isStartingParen = (t.kind == Token.LPAREN && isFuncParen && i == trackers.size() - 1);

                if (isOwningComma) {
                    // Lock in the finished argument and auto-balance any stripped parens
                    tracker.args.add(cleanArgument(tracker.currentArg.toString()));
                    tracker.currentArg.setLength(0);
                } else if (isOwningRParen) {
                    // Function is closing. Lock in the final argument.
                    String lastArg = cleanArgument(tracker.currentArg.toString());
                    if (!lastArg.isEmpty() || !tracker.args.isEmpty()) {
                        if (!lastArg.isEmpty()) {
                            tracker.args.add(lastArg);
                        }
                    }

                    // Wire up the perfectly parsed arguments directly to the Token
                    tracker.funcToken.rawArgs = tracker.args.toArray(new String[0]);
                    tracker.funcToken.arity = tracker.args.size(); // Perfect arity counting

                } else if (!isStartingParen) {
                    // It's a regular token inside an argument, just append it!
                    tracker.currentArg.append(s);
                }
            }

            // ==========================================
            // PHASE B: STANDARD SHUNTING YARD
            // ==========================================
            switch (t.kind) {
                case Token.NUMBER:
                    postfix[p++] = t;
                    if (t.v != null) {
                        int slot = registry.getSlot(t.name);
                        t.frameIndex = slot;
                        t.v.setFrameIndex(slot);
                    }
                    break;

                case Token.FUNCTION:
                case Token.METHOD:
                    opStack.push(t);
                    break;

                case Token.LPAREN:
                    opStack.push(t);
                    isFuncParenStack.push(isFuncParen);
                    break;

                case Token.RPAREN:
                    while (!opStack.isEmpty() && opStack.peek().kind != Token.LPAREN) {
                        postfix[p++] = opStack.pop();
                    }

                    if (!opStack.isEmpty()) {
                        opStack.pop(); // Discard the '('
                    }

                    boolean wasFunc = !isFuncParenStack.isEmpty() && isFuncParenStack.pop();
                    if (wasFunc && !opStack.isEmpty()) {
                        postfix[p++] = opStack.pop(); // Move function to output
                    }

                    // Turn off the tracker if a function just closed
                    if (!trackers.isEmpty() && currentParenDepth == trackers.get(trackers.size() - 1).depthLevel) {
                        trackers.remove(trackers.size() - 1);
                    }

                    currentParenDepth--;
                    break;

                case Token.COMMA:
                    while (!opStack.isEmpty() && opStack.peek().kind != Token.LPAREN) {
                        postfix[p++] = opStack.pop();
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

        // Clean up remaining operators
        while (!opStack.isEmpty()) {
            Token top = opStack.pop();
            if (top.kind != Token.LPAREN) {
                postfix[p++] = top;
            }
        }

        cachedPostfix = new Token[p];
        System.arraycopy(postfix, 0, cachedPostfix, 0, p);

        if (willFoldConstants) {
            foldConstantsWithSafetyGuards();
        }

        this.executionFrame = new double[registry.size()];
        for (Token t : cachedPostfix) {
            if (t.v != null) {
                executionFrame[t.frameIndex] = t.v.getValue();
            }
        }

        expressionSolver = new ExpressionSolver();
    }

    // Helper for isOperator (your custom ops)
    private boolean isOperator(String s) {
        if (s.length() == 1) {
            char c = s.charAt(0);
            return "+-*/%^√!²³ČР".indexOf(c) != -1;
        }
        return s.equals("³√");
    }

    private final class ExpressionSolver implements Savable {

        private static final long serialVersionUID = 1L;
        private static final int MAX_ARITY = 32;
        private static final int ABSOLUTE_MAX_ARITY = 100_000;

        // Pre-allocate and reuse
        private final EvalResult[] stack;
        private final EvalResult[][] argCache;

        public ExpressionSolver clone() {
            ExpressionSolver e = new ExpressionSolver();
            for (int i = 0; i < stack.length; i++) {
                e.stack[i] = stack[i].clone();
            }

            for (int i = 0; i <= MAX_ARITY; i++) {
                e.argCache[i] = new EvalResult[i];
                for (int j = 0; j < i; j++) {
                    e.argCache[i][j] = argCache[i][j].clone();  // Initialize
                }
            }
            return e;
        }

        public ExpressionSolver() {
            // Allocate stack ONCE, with sufficient capacity
            // Max stack depth is usually cachedPostfix.length, but we pad for safety
            int maxStackDepth = Math.max(cachedPostfix.length * 2, 256);
            this.stack = new EvalResult[maxStackDepth];

            // CRITICAL: Initialize all stack slots with EvalResult objects
            for (int i = 0; i < stack.length; i++) {
                stack[i] = new EvalResult();
            }

            // Pre-allocate argument arrays
            this.argCache = new EvalResult[MAX_ARITY + 1][];
            for (int i = 0; i <= MAX_ARITY; i++) {
                argCache[i] = new EvalResult[i];
                for (int j = 0; j < i; j++) {
                    argCache[i][j] = new EvalResult();  // Initialize
                }
            }
        }

        public EvalResult evaluate() {
            // Just use the pre-allocated stack - no allocation per call
            int ptr = -1;

            for (int i = 0; i < cachedPostfix.length; i++) {
                Token t = cachedPostfix[i];
                /*          System.print.println("\n=== Evaluating token: "
                        + (t.kind == Token.NUMBER ? "NUM(" + t.value + ") OR VAR("+t.name+"), "
                                : t.kind == Token.OPERATOR ? "OP(" + t.opChar + ")"
                                        : t.kind == Token.FUNCTION ? "FUNC(" + t.name + ",arity=" + t.arity + ")"
                                                : "METHOD(" + t.name + ",arity=" + t.arity + ")")
                        + " | Stack ptr before = " + ptr);*/

                switch (t.kind) {
                    case Token.NUMBER:
                        if (t.name != null && !t.name.isEmpty()) {

                            // Could be a variable OR a function reference (like anon1)
                            if (t.v != null) {
                                // It's a variable
                                double v = executionFrame[t.frameIndex];
                                stack[++ptr] = getNextResult().wrap(v);
                                t.v.setValue(v);
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
                                    String err = "Undefined variable or function: " + t.name;
                                    errorLog.info(err);
                                    throw new RuntimeException(err);
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
                                String err = "Insufficient operands for unary operator: " + t.opChar;
                                errorLog.info(err);
                                throw new RuntimeException("Insufficient operands for unary operator: " + t.opChar);
                            }
                            applyUnary(t.opChar, stack[ptr]);
                        } else {
                            // Binary operator
                            if (ptr < 1) {
                                String err = "Insufficient operands for binary operator: " + t.opChar
                                        + " (stack ptr=" + ptr + ")";
                                errorLog.info(err);
                                throw new RuntimeException(err);
                            }
                            double bVal = stack[ptr--].scalar;
                            applyBinary(t.opChar, stack[ptr], bVal);
                        }
                        break;

                    case Token.METHOD:
                    case Token.FUNCTION:
                        int arity = t.arity;

                        if (arity > ABSOLUTE_MAX_ARITY) {
                            String err
                                    = "Function " + t.name + " has too many arguments (" + arity
                                    + "). Maximum supported: " + ABSOLUTE_MAX_ARITY;
                            errorLog.info(err);
                            throw new RuntimeException(err);
                        }

                        int valuesOnStack = ptr + 1;

                        if (arity == 0) {
                            EvalResult result = t.action.calc(getNextResult(), arity, argCache[arity]);
                            stack[++ptr] = result;
                            break;
                        }

                        if (valuesOnStack < arity) {
                            String err
                                    = "Function " + t.name + " requires " + arity
                                    + " arguments but only " + valuesOnStack + " values available on stack";
                            errorLog.info(err);
                            throw new RuntimeException(err);
                        }

                        // ULTRA FAST: Reuse cached array for this arity
                        EvalResult[] args = (arity <= MAX_ARITY) ? argCache[arity] : new EvalResult[arity];

                        // This eliminates loop overhead for the most common cases
                        switch (arity) {
                            case 1:
                                args[0] = stack[ptr--];
                                break;
                            case 2:
                                args[1] = stack[ptr--];
                                args[0] = stack[ptr--];
                                break;
                            case 3:
                                args[2] = stack[ptr--];
                                args[1] = stack[ptr--];
                                args[0] = stack[ptr--];
                                break;
                            case 4:
                                args[3] = stack[ptr--];
                                args[2] = stack[ptr--];
                                args[1] = stack[ptr--];
                                args[0] = stack[ptr--];
                                break;
                            case 5:
                                args[4] = stack[ptr--];
                                args[3] = stack[ptr--];
                                args[2] = stack[ptr--];
                                args[1] = stack[ptr--];
                                args[0] = stack[ptr--];
                                break;
                            case 6:
                                args[5] = stack[ptr--];
                                args[4] = stack[ptr--];
                                args[3] = stack[ptr--];
                                args[2] = stack[ptr--];
                                args[1] = stack[ptr--];
                                args[0] = stack[ptr--];
                                break;
                            case 7:
                                args[6] = stack[ptr--];
                                args[5] = stack[ptr--];
                                args[4] = stack[ptr--];
                                args[3] = stack[ptr--];
                                args[2] = stack[ptr--];
                                args[1] = stack[ptr--];
                                args[0] = stack[ptr--];
                                break;
                            default:
                                // General case for arity > 7
                                for (int j = arity - 1; j >= 0; j--) {
                                    args[j] = stack[ptr--];
                                }
                                break;
                        }

                        if (t.name.equals(Declarations.DIFFERENTIATION)) {
                            for (int k = 0; k < t.rawArgs.length; k++) {
                                args[k] = getNextResult().wrap(t.rawArgs[k]);
                            }
                        }

                        EvalResult result = t.action.calc(getNextResult(), arity, args);
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
                String err = "WARNING: Evaluation stack has " + (ptr + 1) + " values at end, returning top";
                errorLog.info(err);
                System.out.println(err);
            }

            return stack[0];
        }

        public EvalResult evaluate(double... arguments) {
            MathExpression.this.executionFrame = arguments;
            return evaluate();
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
                        String err = "Division by zero---op=" + op + ", b=" + b + ", aRes=" + aRes + ", aRes.type = " + aRes.getTypeName();
                        errorLog.info(err);
                        aRes.scalar = Double.POSITIVE_INFINITY;
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

    /**
     * PRODUCTION-GRADE constant folding with DRG mode awareness.
     *
     * CRITICAL CHANGE: Trigonometric functions are NEVER folded, because
     * setDRG() must be able to change them after compilation.
     */
    private void foldConstantsWithSafetyGuards() {
        if (cachedPostfix == null || cachedPostfix.length < 2) {
            return;
        }

        boolean changed = true;
        int passes = 0;
        final int MAX_PASSES = 5;

        while (changed && passes < MAX_PASSES) {
            changed = false;
            passes++;

            Token[] original = cachedPostfix;
            Token[] folded = new Token[original.length];
            int writePtr = 0;

            for (int i = 0; i < original.length; i++) {
                Token t = original[i];

                // ========== BINARY OPERATORS ==========
                if (t.kind == Token.OPERATOR && !t.isPostfix && writePtr >= 2) {
                    Token right = folded[writePtr - 1];
                    Token left = folded[writePtr - 2];

                    if (isLiteralConstant(left) && isLiteralConstant(right)) {
                        double result = evaluateBinaryOpWithIEEECompliance(t.opChar, left.value, right.value);

                        if (isFoldableResult(result)) {
                            writePtr -= 2;
                            folded[writePtr++] = new Token(result);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ========== UNARY OPERATORS ==========
                if (t.kind == Token.OPERATOR && t.isPostfix && writePtr >= 1) {
                    Token operand = folded[writePtr - 1];

                    if (isLiteralConstant(operand)) {
                        double result = evaluateUnaryOpWithIEEECompliance(t.opChar, operand.value);

                        if (isFoldableResult(result)) {
                            writePtr--;
                            folded[writePtr++] = new Token(result);
                            changed = true;
                            continue;
                        }
                    }
                }

                // ========== PURE FUNCTIONS (WITH DRG AWARENESS) ==========
                if ((t.kind == Token.FUNCTION || t.kind == Token.METHOD) && writePtr >= t.arity) {

                    // CRITICAL: Never fold trigonometric functions!
                    /* if (isTrigonometricFunction(t)) {
                        folded[writePtr++] = t;  // Keep the function token as-is
                        continue;
                    }*/
                    if (!isConstantFoldableFunction(t)) {
                        folded[writePtr++] = t;
                        continue;
                    }

                    boolean allConstant = true;
                    for (int j = 0; j < t.arity; j++) {
                        if (!isLiteralConstant(folded[writePtr - t.arity + j])) {
                            allConstant = false;
                            break;
                        }
                    }

                    if (allConstant) {
                        EvalResult[] args = new EvalResult[t.arity];
                        for (int j = 0; j < t.arity; j++) {
                            args[j] = new EvalResult().wrap(folded[writePtr - t.arity + j].value);
                        }

                        try {
                            EvalResult evalResult = t.action.calc(new EvalResult(), t.arity, args);

                            if (evalResult.type == EvalResult.TYPE_SCALAR && isFoldableResult(evalResult.scalar)) {
                                writePtr -= t.arity;
                                folded[writePtr++] = new Token(evalResult.scalar);
                                changed = true;
                                continue;
                            }
                        } catch (Exception e) {
                            errorLog.error(e);
                            // Don't fold if runtime error occurs
                        }
                    }
                }

                folded[writePtr++] = t;
            }

            cachedPostfix = new Token[writePtr];
            System.arraycopy(folded, 0, cachedPostfix, 0, writePtr);
        }
    }

    /**
     * Whitelist of functions that ARE safe to constant-fold. Note: Trig
     * functions are handled separately (never folded).
     */
    private boolean isConstantFoldableFunction(Token t) {
        if (t.name == null) {
            return false;
        }

        String name = t.name.toLowerCase();

        // ===== BLACKLIST: NON-FOLDABLE (Side effects, randomness, I/O, user-defined) =====
        java.util.Set<String> nonFoldable = new java.util.HashSet<>(
                Arrays.asList(
                        // Stochastic - different result each time
                        "rand", "random", "randbetween", "randi", "randint", "randnormal", "rnd",
                        // Time-dependent
                        "now", "time", "today", "clock", "timestamp", "millis",
                        // Side effects (I/O, output)
                        "print", "println", "debug", "log", "plot",
                        // System-dependent
                        "random_seed", "env", "getenv",
                        // I/O operations
                        "read", "readline", "input", "scan",
                        // User-defined or dynamic
                        "diff", "intg", "differentiation", "integration",
                        // Matrix operations (complex state-dependent results)
                        "linear_sys", "det", "determinant", "invert", "inverse_matrix",
                        "tri_mat", "triangular_matrix", "echelon", "echelon_matrix",
                        "matrix_mul", "matrix_multiply", "matrix_div", "matrix_divide",
                        "matrix_add", "matrix_subtract", "matrix_sub",
                        "matrix_pow", "matrix_power", "transpose", "matrix_edit",
                        "matrix_cofactors", "cofactor", "matrix_adjoint", "adjoint",
                        "matrix_eigenvec", "eigvec", "matrix_eigenvalues", "eigvalues",
                        "matrix_eigenpoly", "eigpoly",
                        // List sorting (order-dependent, may depend on implementation)
                        "sort",
                        // Help
                        "help",
                        // Quadratic/root solving (may have multiple solutions or complex numbers)
                        "quadratic", "tartaglia_roots", "t_root", "general_root", "root")
        );

        if (nonFoldable.contains(name)) {
            return false;
        }

        // ===== SPECIAL HANDLING: Trigonometric functions =====
        // These ARE foldable, but ONLY if NOT being used with setDRG() changes
        // For safety, we fold them ONLY when the expression containsAlgebraicFunction constants
        // Since we're in compile-time, we CAN fold them with current DRG mode
        java.util.Set<String> trigonometric = new java.util.HashSet<>(
                Arrays.asList(MethodRegistry.expandedTrigAndHypMethodNames)
        );

        // ===== WHITELIST: DEFINITELY FOLDABLE =====
        java.util.Set<String> definitelyFoldable = new java.util.HashSet<>(
                Arrays.asList(
                        // ===== EXPONENTIAL & LOGARITHMIC =====
                        "exp", "ln", "lg", "log", "log10", "log2",
                        "ln-¹", "lg-¹", "log-¹", "aln", "alg", "alog",
                        // ===== POWER & ROOT =====
                        "sqrt", "cbrt", "pow", "hypot",
                        "inverse", "square", "cube",
                        "fact",
                        // ===== COMBINATORICS =====
                        "comb", "perm",
                        // ===== ROUNDING & ABSOLUTE =====
                        "abs", "floor", "ceil", "round", "trunc", "sign",
                        // ===== VECTOR STATISTICS (pure functions - no state) =====
                        "listsum", "sum", "prod", "product",
                        "mean", "listavg", "avg", "average",
                        "median", "med",
                        "min", "max",
                        "mode", "sort",
                        "range", "rng", "mid_range", "mrng",
                        "rms",
                        "s_d", "std_dev",
                        "variance", "st_err",
                        "cov"
                ));
        definitelyFoldable.addAll(trigonometric);

        return definitelyFoldable.contains(name);
    }

    /**
     * CRITICAL: IEEE 754 compliance check. Results that are valid but
     * exceptional should NOT be folded—let them fail/succeed at runtime where
     * they belong.
     */
    private boolean isFoldableResult(double result) {
        // FOLD: Normal numbers (including very small ones, subnormals, etc.)
        if (Double.isFinite(result)) {
            return true;
        }

        // DO NOT FOLD: Infinities or NaN
        // These are EXCEPTIONAL but defined by IEEE 754.
        // Folding them risks hiding:
        // - Overflow conditions that should be caught
        // - Invalid operation exceptions (sqrt(-1) in some strict modes)
        // - Rounding edge cases
        if (Double.isInfinite(result) || Double.isNaN(result)) {
            return false;
        }

        return false;
    }

    /**
     * Unary operator evaluation with IEEE 754 semantics preserved.
     */
    private double evaluateUnaryOpWithIEEECompliance(char op, double val) {
        switch (op) {
            case '√':
                return Math.sqrt(val);  // IEEE 754: sqrt(-1) = NaN, sqrt(Inf) = Inf

            case 'R':  // Cube root
                return Math.cbrt(val);  // IEEE 754 compliant

            case '!':
                return Maths.fact(val);  // Watch: factorial of negative or non-integer is NaN

            case '²':
                return val * val;  // IEEE 754: Inf^2 = Inf, NaN^2 = NaN

            case '³':
                return val * val * val;

            case 'i':  // Reciprocal (x^-1)
                return 1.0 / val;  // IEEE 754: 1/0 = Inf, 1/Inf = 0.0

            default:
                return Double.NaN;
        }
    }

    /**
     * Binary operator evaluation with IEEE 754 semantics preserved. This is
     * evaluated at compile-time, so it uses the JVM's double precision, which
     * should match runtime evaluation.
     */
    private double evaluateBinaryOpWithIEEECompliance(char op, double a, double b) {
        // Just delegate to the strength-reduced version
        return evaluateBinaryOpWithStrengthReduction(op, a, b);
    }

    /**
     * Identify literal constants (not variables, not function references).
     */
    private boolean isLiteralConstant(Token t) {
        // Must be a NUMBER token with no variable or function name attached
        return t.kind == Token.NUMBER
                && t.v == null
                && (t.name == null || t.name.isEmpty());
    }

    ///////////////////////////////////////////////////////
   /**
 * SAFE strength reduction: Only optimize constant expressions.
 * E.g., 2^3 → 8.0 (computed at compile time)
 * This is proven safe because it's just constant folding.
 */
private double evaluateBinaryOpWithStrengthReduction(char op, double a, double b) {
        switch (op) {
            case '+':
                return a + b;

            case '-':
                return a - b;

            case '*':
                return a * b;

            case '/':
                return a / b;

            case '%':
                return a % b;

            case '^':
                // STRENGTH REDUCTION for constant folding only
                return evaluatePowerWithStrengthReduction(a, b);

            case 'Р':  // Permutation
                return Maths.permutation(a, b);

            case 'Č':  // Combination
                return Maths.combination(a, b);

            default:
                return Double.NaN;
        }
    }

    /**
     * Evaluate a^b with strength reduction for common cases. This is safe
     * because we only do it for CONSTANT folding.
     *
     * CRITICAL FIX: Actually check if we matched a special fraction
     */
    private double evaluatePowerWithStrengthReduction(double base, double exponent) {

        // ===== SPECIAL FRACTIONS (do this FIRST, before integer check) =====
        if (isSpecialFraction(exponent)) {
            // Now figure print WHICH fraction it is
            final double TOLERANCE = 1e-9;

            // 1/2 → sqrt
            if (Math.abs(exponent - 0.5) < TOLERANCE) {
                return Math.sqrt(base);
            }

            // 1/3 → cbrt
            if (Math.abs(exponent - 1.0 / 3.0) < TOLERANCE) {
                return Math.cbrt(base);
            }

            // -0.5 → 1/sqrt
            if (Math.abs(exponent - (-0.5)) < TOLERANCE) {
                return 1.0 / Math.sqrt(base);
            }

            // -1/3 → 1/cbrt
            if (Math.abs(exponent - (-1.0 / 3.0)) < TOLERANCE) {
                return 1.0 / Math.cbrt(base);
            }

            // 2/3
            if (Math.abs(exponent - 2.0 / 3.0) < TOLERANCE) {
                return Math.cbrt(base * base);
            }

            // -2/3
            if (Math.abs(exponent - (-2.0 / 3.0)) < TOLERANCE) {
                return 1.0 / Math.cbrt(base * base);
            }

            // 3/4
            if (Math.abs(exponent - 3.0 / 4.0) < TOLERANCE) {
                double sqrtRoot = Math.sqrt(base);
                return Math.sqrt(sqrtRoot * sqrtRoot * sqrtRoot);
            }

            // If it's a special fraction but we didn't match it, fall through
        }

        // ===== INTEGER EXPONENTS =====
        if (isIntegerValue(exponent)) {
            int exp = (int) Math.round(exponent);  // Use round() not cast

            // POSITIVE INTEGER EXPONENTS
            if (exp > 1 && exp <= 10) {
                double result = base;
                for (int i = 1; i < exp; i++) {
                    result *= base;
                }
                return result;
            }

            // ZERO EXPONENT
            if (exp == 0) {
                return 1.0;
            }

            // NEGATIVE INTEGER EXPONENTS
            if (exp < 0) {
                int absExp = Math.abs(exp);
                double result = base;
                for (int i = 1; i < absExp; i++) {
                    result *= base;
                }
                return 1.0 / result;
            }
        }

        // Fall back to Math.pow() for everything else
        return Math.pow(base, exponent);
    }
// ========== HELPER METHODS ==========

    /**
     * Check if a double value is an integer
     */
    private boolean isIntegerValue(double value) {
        return value == Math.floor(value) && !Double.isInfinite(value);
    }

    /**
     * Check if value is a commonly-reduced fraction (1/2, 1/3, -1/2, -1/3,
     * etc.)
     */
    private boolean isSpecialFraction(double value) {
        // Check 1/2, 1/3, ..., 1/8 and their negatives
        for (int i = 2; i <= 8; i++) {
            if (Math.abs(value - 1.0 / i) < 1e-15) {
                return true;
            }
            if (Math.abs(value + 1.0 / i) < 1e-15) {
                return true;
            }
        }
        // Check 2/3, 3/4, etc. (common cases)
        if (Math.abs(value - 2.0 / 3.0) < 1e-15) {
            return true;
        }
        if (Math.abs(value - 3.0 / 4.0) < 1e-15) {
            return true;
        }

        return false;
    }

    public static final class EvalResult implements Savable {

        private static final long serialVersionUID = 1L;
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
        /**
         * Use this to say what happened when this Object is of TYPE_ERROR
         */
        private String errorText;

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

        public EvalResult clone() {
            EvalResult e = new EvalResult();
            e.boolVal = boolVal;
            e.error = error;
            e.errorText = errorText;
            if (matrix != null) {
                double[] flatArr = matrix.getFlatArray();
                double[] arr = new double[flatArr.length];
                System.arraycopy(flatArr, 0, arr, 0, flatArr.length);
                Matrix m = new Matrix(flatArr, matrix.getRows(), matrix.getCols());
                e.matrix = m;
                e.matrix.setName(matrix.getName());
            } else {
                e.matrix = null;
            }
            e.scalar = scalar;
            e.textRes = textRes;
            e.type = type;
            if (vector == null) {
                e.vector = null;
            } else {
                double[] v = new double[vector.length];
                System.arraycopy(e.vector, 0, v, 0, vector.length);
                e.vector = v;

            }

            return e;

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

        public void setErrorText(String errorText) {
            this.errorText = errorText;
        }

        public String getErrorText() {
            return errorText;
        }

// In EvalResult class:
        public void reset() {
            this.type = TYPE_SCALAR;
            this.error = null;
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
                    return TYPE.VECTOR;
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
                    case VECTOR:
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
        // 1. Check if we need to expand (using a local copy for thread safety)
        EvalResult[] currentPool = this.pool;
        if (poolPointer >= currentPool.length) {
            synchronized (this) {
                // Double-check pattern to prevent multi-thread OOM
                if (poolPointer >= pool.length) {
                    int newSize = pool.length * 2;
                    EvalResult[] newPool = new EvalResult[newSize];
                    System.arraycopy(pool, 0, newPool, 0, pool.length);
                    for (int i = pool.length; i < newSize; i++) {
                        newPool[i] = new EvalResult();
                    }
                    this.pool = newPool; // Now the pointer won't OOM
                }
            }
        }

        // 2. Fetch and Reset
        EvalResult result = pool[poolPointer++];
        result.reset(); // Clear old state (crucial for complex objects!)
        return result;
    }

// Add a reset method to clear the pool between evaluations
    private void resetPool() {
        poolPointer = 0;
    }
// Simple generic Pair implementation

    public static final class Pair<K, V> {

        private final K key;
        private final V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }

    /**
     * Manages the mapping of variable names to frame slots. Use one instance
     * per MathExpression compilation.
     */
    public static final class VariableRegistry implements Savable {

        private static final long serialVersionUID = 1L;
        private final Map<String, Integer> nameToSlot = new LinkedHashMap<>();
        private int nextAvailableSlot = 0;

        /**
         * Returns the slot index for a variable name. If the name is new, it
         * assigns a new slot.
         *
         * @param name The name of the Variable
         */
        public int getSlot(String name) {
            // We use computeIfAbsent to keep it atomic and clean
            return nameToSlot.computeIfAbsent(name, k -> nextAvailableSlot++);
        }

        public boolean hasVariable(String name) {
            return this.nameToSlot.containsKey(name);
        }

        /**
         * Returns the total number of slots required for the execution frame.
         */
        public int size() {
            return nextAvailableSlot;
        }

        private String[] getVariables() {
            String[] vars = new String[nameToSlot.size()];
            int i = 0;
            for (Map.Entry<String, Integer> entry : nameToSlot.entrySet()) {
                vars[i++] = entry.getKey();
            }
            return vars;
        }

        private int[] getSlots() {
            int[] slots = new int[nameToSlot.size()];
            int i = 0;
            for (Map.Entry<String, Integer> entry : nameToSlot.entrySet()) {
                slots[i++] = entry.getValue();
            }
            return slots;
        }

        private Pair<String[], Integer[]> getVarsAndSlots() {
            String[] vars = new String[nameToSlot.size()];
            Integer[] slots = new Integer[nameToSlot.size()];
            int i = 0;
            for (Map.Entry<String, Integer> entry : nameToSlot.entrySet()) {
                vars[i] = entry.getKey();
                slots[i] = entry.getValue();
                i++;
            }
            return new Pair<>(vars, slots);
        }

        public void reset() {
            nameToSlot.clear();
            nextAvailableSlot = 0;
        }

        protected VariableRegistry clone() {
            VariableRegistry vr = new VariableRegistry();
            vr.nextAvailableSlot = nextAvailableSlot;
            for (Map.Entry<String, Integer> entry : nameToSlot.entrySet()) {
                vr.nameToSlot.put(entry.getKey(), entry.getValue());
            }

            return vr;
        }
    }

    /**
     * Manual deep copy method that will most certainly outperform the clone
     * method. But needs to be updated whenever the MathExpression object's
     * field structure changes. The clone method does not necessarily need this,
     * except for its need for all fields of the {@link MathExpression} to be
     * {@link Savable}
     *
     * @return
     */
    public MathExpression copy() {
        MathExpression me = new MathExpression(true);
        me.DRG = DRG;
        Token tokens[] = new Token[cachedPostfix.length];
        int i = 0;
        for (Token t : cachedPostfix) {
            tokens[i] = t.clone();
            i++;
        }
        me.cachedPostfix = tokens;
        me.commaAlias = commaAlias;
        errorLog.copyTo(me.errorLog);

        me.correctFunction = correctFunction;
        me.executionFrame = new double[executionFrame.length];

        System.arraycopy(executionFrame, 0, me.executionFrame, 0, executionFrame.length);
        me.expression = expression;
        if (expressionSolver != null) {
            me.expressionSolver = expressionSolver.clone();
        }
        me.hasFunctionOrVariableInitStatement = hasFunctionOrVariableInitStatement;
        me.hasListReturningOperators = hasListReturningOperators;
        me.help = help;
        me.noOfListReturningOperators = noOfListReturningOperators;
        me.optimizable = optimizable;
        me.parser_Result = ParserResult.valueOf(parser_Result.name());
        me.pool = new EvalResult[pool.length];
        for (i = 0; i < me.pool.length; i++) {
            me.pool[i] = pool[i].clone();
        }
        me.poolPointer = poolPointer;
        if (registry != null) {
            me.registry = registry.clone();
        }
        me.returnObjectName = returnObjectName;
        me.returnType = TYPE.valueOf(returnType.name());
        me.scanner = new ArrayList<>(scanner);
        me.slots = new int[slots.length];
        System.arraycopy(slots, 0, me.slots, 0, slots.length);
        me.treeStats = new MathExpressionTreeDepth.Result(treeStats.depth, treeStats.binaryOperators, treeStats.divOperators, treeStats.unaryOperators, treeStats.functions);
        me.turboCompiled = turboCompiled;
        me.variableManager = new VariableManager();
        me.whitespaceremover = new ArrayList<>(whitespaceremover);
        me.willFoldConstants = willFoldConstants;
//        me.compiledTurbo = new FastCompositeExpression() {
//            @Override
//            public EvalResult apply(double[] variables) {
//                return compiledTurbo.apply(variables);
//            }
//
//            @Override
//            public double applyScalar(double[] variables) {
//                return compiledTurbo.applyScalar(variables);
//            }
//        };//clone this
        if (compiledTurbo != null) {
            try {
                me.compiledTurbo = me.compileTurbo();
            } catch (Throwable ex) {
                System.out.println("Failed to compile turbo for cloned MathExpression");
                Logger.getLogger(MathExpression.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return me;
    }

    @Override
    public MathExpression clone() throws CloneNotSupportedException {
        return Serializer.deepClone(this);
    }

    public static void main1(String... args) {
        String in = Main.joinArgs(Arrays.asList(args), true);
        if (Main.isVerbose()) {
            System.err.println(in);
        }
        System.out.println(new MathExpression(in).solve());
    }//end method

    static class Test_Strength_Reduction {

        public static void main(String... args) {
            String rpt = STRING.repeating("=", 80);
            System.out.println(rpt);
            System.out.println("SAFE CONSTANT FOLDING WITH STRENGTH REDUCTION");
            System.out.println(rpt);

            testConstantFolding();
            testWithVariables();
            testComplexExpression();
        }

        private static void testConstantFolding() {
            System.out.println("\n=== CONSTANT FOLDING (Strength Reduced) ===\n");

            String[] exprs = {
                "2^3", // 8.0
                "3^2", // 9.0
                "2^4", // 16.0
                "4^0.5", // 2.0 (sqrt)
                "8^(1/3)", // 2.0 (cbrt)
                "2^3 + 3^2", // 17.0
                "10^-2", // 0.01
                "sin(3)+cos(5)",
                "2*sinh(5)"
            };

            double[] expected = {8.0, 9.0, 16.0, 2.0, 2.0, 17.0, 0.01, 0.42478219352309348656738397432167, 5 * 148.40642115557751795401894399213};

            for (int i = 0; i < exprs.length; i++) {
                try {
                    MathExpression me = new MathExpression(exprs[i]);
                    double result = Double.parseDouble(me.solve());

                    String status = Math.abs(result - expected[i]) < 1e-9 ? "✓" : "✗";
                    System.out.printf("%s %s = %.6f (expected: %.6f)%n",
                            status, exprs[i], result, expected[i]);
                } catch (Exception e) {
                    System.out.printf("✗ %s: %s%n", exprs[i], e.getMessage());
                }
            }
        }

        private static void testWithVariables() {
            System.out.println("\n=== WITH VARIABLES (No folding) ===\n");

            String[] exprs = {
                "x^2",
                "x^3 + x^2",
                "sqrt(x^2 + 1)",
                "1/x^2",};

            for (String expr : exprs) {
                try {
                    MathExpression me = new MathExpression(expr);
                    me.updateSlot(me.registry.getSlot("x"), 2.0);
                    double result = Double.parseDouble(me.solve());

                    System.out.printf("✓ %s (at x=2) = %.6f%n", expr, result);
                } catch (Exception e) {
                    System.out.printf("✗ %s: %s%n", expr, e.getMessage());
                }
            }
        }

        private static void testComplexExpression() {
            System.out.println("\n=== COMPLEX EXPRESSION (The failing case) ===\n");

            try {
                MathExpression me = new MathExpression("sqrt(0.64-x^2)");
                me.updateSlot(me.registry.getSlot("x"), 0.9);
                double result = Double.parseDouble(me.solve());

                System.out.printf("✓ sqrt(0.64-x^2) at x=0.9 = %.6f%n", result);
            } catch (Exception e) {
                System.out.printf("✗ sqrt(0.64-x^2): %s%n", e.getMessage());
                e.printStackTrace();
            }
        }
    }

    static class Test {

        public static void main(String... args) {
            String rpt = STRING.repeating("=", 80);
            System.out.println(rpt);
            System.out.println("POSTFIX EVALUATION ACCURACY TESTS");
            System.out.println(rpt);

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

            System.out.println("\n" + rpt);
            System.out.println("ACCURACY TEST SUMMARY");
            System.out.println(rpt);
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
        Test_Strength_Reduction.main(args);
        MathExpression mathy = new MathExpression("x=12;sin(x-2)+cos(x-2)");
        System.out.println("x=12;sin(x-2)+cos(x-2)---->" + mathy.solveGeneric().scalar);
        System.out.println("VARIABLES--A = " + VariableManager.VARIABLES);
        for (int i = 0; i < 10; i++) {
            mathy.setValue("x", i);
            System.out.println("x=" + i + ";sin(x-2)+cos(x-2)---->" + mathy.solveGeneric().scalar);
        }
        System.out.println("VARIABLES--B = " + VariableManager.VARIABLES);
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
        System.out.println("FUNCTIONS: " + FunctionManager.FUNCTIONS);
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
        System.out.println("differential calculus:>>5 " + new MathExpression("diff(@(x)sin(ln(x)), 2,1);").solve());
        System.out.println(new MathExpression("sin(ln(x));").solve());
        System.out.println("FUNCTIONS: " + FunctionManager.FUNCTIONS);
// Expected: 11

        System.out.println("sort(-3,8,3,2,6,-7,9,1,0,-1): " + new MathExpression("sort(-3,8,3,2,6,-7,9,1,0,-1)").solve());
        System.out.println("sort(0,4+1,2+0): " + new MathExpression("sort(0,4+1,2+0)").solve());
        System.out.println("sort(3+1,-3): " + new MathExpression("sort(3+1, -3)").solve());
        System.out.println("sort(4+2): " + new MathExpression("sort(4+2)").solve());
        System.out.println(new MathExpression("x=0.9;sqrt(0.64-x^2)").solve());

        MathExpression mex = new MathExpression("A=@(3,3)(3,4,2,9,12,5,4,1,2);B=eigvalues(A);C=eigvec(A);D=eigpoly(A);eigpoly(A);");
        System.out.println(FunctionManager.FUNCTIONS);
        System.out.println("----------" + mex.solve());
        System.out.println("VARIABLES--1 = " + VariableManager.VARIABLES);
        Function f = FunctionManager.add("f(x,y) = x - x/y");
        System.out.println("VARIABLES--2 = " + VariableManager.VARIABLES);

        f.updateArgs(2, 3);
        double r = f.calc();
        System.out.println("VARIABLES--3 = " + VariableManager.VARIABLES);
        System.out.println("f(x,y) = x - x/y__________________r = " + r);
        int iterations = 1;
        double vvv[] = new double[1];
        long start = System.nanoTime();
        f.updateArgs(2, 3);
        for (int i = 1; i <= iterations; i++) {
            vvv[0] = f.calc();
        }
        double duration = (System.nanoTime() - start) / iterations;
        System.out.println("dur = " + (duration / 1000) + " microns, ans = " + vvv[0]);//μμμ

        System.out.println("VARIABLES---4: " + VariableManager.VARIABLES);
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
        System.out.println("m.solve(): " + new MathExpression("avgN(0,4+0,2+3,-2)").solve());
        System.out.println("cos(pi): " + new MathExpression("cos(pi)").solve());

        System.out.println("VARIABLES--1 = " + VariableManager.VARIABLES);
        System.out.println(new MathExpression("sort(0,4+0,2+0)").solve());
        System.out.println(new MathExpression("F=@(x)sin(x);intg(F,2,3)").solve());
        System.out.println(new MathExpression("F=@(x)sin(x);intg(F,2,3,100000)").solve());
        System.out.println(new MathExpression("G=@(x)sin(x)-cos(x);intg(G,2,30,1000)").solve());
        MathExpression meRoots = new MathExpression("root(@(x)x^2+2*x+1, 3)");
        System.out.println(meRoots.solve());
        MathExpression quadRoots = new MathExpression("quadratic(@(x)3*x^2-8*x+3)");
        System.out.println(quadRoots.solve());
        MathExpression quadRoots1 = new MathExpression("quadratic(@(x)3*x^2-8*x+203)");
        System.out.println(quadRoots1.solve());
        MathExpression tartRoots = new MathExpression("t_root(@(x)5*x^3-12*x+120)");
        System.out.println(tartRoots.solve());

        System.out.println(new MathExpression("M=@(x)7*x^2;M(2)").solve());
        System.out.println("FUNCTIONS: " + FunctionManager.FUNCTIONS);
        MathExpression printer = new MathExpression("print(anon9,C)");
        System.out.println("anon9: " + FunctionManager.lookUp("anon9"));
        MathExpression mee = new MathExpression("sum(2,3,sort(3,0,1,2))");

        System.out.println("listsum(2,3,sort(3,0,1,2)): " + mee.solve());
        System.out.println(printer.solve());

        MathExpression me = new MathExpression("A=@(x,y)sin(x)+cos(y-x);B=@(x,y)cos(x*y);C(x,y)=A(x,2*y)+B(3*x,2*y);D=C;print(D)");
        System.out.println("solve: " + me.solveGeneric());
        System.out.println(FunctionManager.FUNCTIONS);

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
