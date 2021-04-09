/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import interfaces.Savable;
import parser.methods.Method;

import math.Maths;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import util.*;

import static parser.Variable.*;
import static parser.Number.*;
import static parser.Operator.*;

import math.matrix.expressParser.Matrix;

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
 * <b>
 * <font color = 'red'>
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
 *
 *
 * </font>
 * </b>
 *
 * @author GBENRO
 */
public class MathExpression implements Savable {

    public Parser_Result parser_Result = Parser_Result.VALID;
    //determines the mode in which trig operations will be carried out on numbers.if DRG==0,it is done in degrees
//if DRG==1, it is done in radians and if it is 2, it is done in grads.
    protected int DRG = 1;
    public static String lastResult = "0.0";
    private ArrayList<String> whitespaceremover = new ArrayList<>();//used to remove white spaces from the ArrayList
    /**
     * The expression to evaluate.
     */
    private String expression;
    protected boolean correctFunction = true;//checks if the function is valid.
    protected int noOfListReturningOperators;
    protected ArrayList<String> scanner = new ArrayList<>();//the ArrayList that stores the scanner input function
    private boolean optimizable;
    private Bracket[] bracket;
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

    /**
     * Utility attribute used throughout the class for string appending
     * operations.
     */
    StringBuilder utility = new StringBuilder("");
    /**
     * The type of output returned by the parser.
     */
    private TYPE returnType = TYPE.NUMBER;

    /**
     * Sometimes, after evaluation the evaluation list which is a local
     * variable, is reduced to a function name(or other object as time goes on)
     * instead of a number of other list. The parser unfortunately will not
     * return this variable, but instead returns the data which it refers
     * to..e.g a {@link Matrix} function or other. But we atimes need that
     * function name...So we cache this value here.
     */
    private String returnObjectName;

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
     * variable, constant and function declarations for variables, constants and functions that are
     * not yet initialized, assignment expressions for those that have been
     * initialized and then an expression to evaluate. e.g. x = -12; y
     * =x+1/12; const x1,x2,x3=10; z =sin(3x-1)+2.98cos(4x);cos(3x+12); The last
     * expression is a function to be evaluated and it is always without any
     * equals sign and may or may not end with a semicolon.
     *
     */
    public MathExpression(String input) {
        setAutoInitOn(false);
        this.variableManager = new VariableManager();

        CustomScanner cs = new CustomScanner(STRING.purifier(input), false, VariableManager.endOfLine);

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

    }//end constructor MathExpression

    public String getExpression() {
        return expression;
    }

    /**
     * @param expression The expression
     */
    public final void setExpression(String expression) {
        if (!expression.equals(this.expression)) {
            scanner.clear();
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
            bracket = null;
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
    public int getDRG() {
        return DRG;
    }

    /**
     * sets the DRG property
     *
     * @param DRG
     */
    public void setDRG(int DRG) {
        this.DRG = (DRG == 0 || DRG == 1 || DRG == 2) ? DRG : 0;
    }

    /**
     *
     * @return the Brackets ArrayList containing all Bracket objects found in
     * the input.
     */
    public Bracket[] getBracket() {
        return bracket;
    }

    /**
     *
     * @param bracket the Brackets ArrayList containing all Bracket objects
     * found in the input.
     */
    public void setBracket(Bracket[] bracket) {
        this.bracket = bracket;
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
    public ArrayList<String> getScanner() {
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

    /**
     * Removes encapsulating brackets from data set returning statistical
     * operators. For example, (((sort(3,sin(4),5,-1))) does not need the two
     * enclosing bracket pairs so turn it into... sort(3,sin(4),5,-1).
     *
     *
     */
    public void unBracketDataSetReturningStatsOperators() {

        for (int i = 0; i < scanner.size(); i++) {

            boolean exitForLoop = false;
            if (Method.isListReturningStatsMethod(scanner.get(i))) {
                int I = i - 1;
                /**
                 * Loop backwards from the operator itself and if the object at
                 * the index is not the open bracket of a
                 * ListReturningStatsOperator,remove it.Keep doing this till the
                 * test fails. If the test keeps succeeding, then watch out for
                 * the starting index,0. If the element at the starting index is
                 * also an open bracket, remove it and exit the loop.
                 *
                 */
                while (isOpeningBracket(scanner.get(I))) {
                    int closeBracIndex = Bracket.getComplementIndex(true, I, scanner);
                    if (I > 0 && !Method.isListReturningStatsMethod(scanner.get(I - 1))) {
                        scanner.set(closeBracIndex, "");
                        scanner.set(I, "");
                        --I;
                        exitForLoop = closeBracIndex <= scanner.size();
                    } else if (I == 0 && isOpeningBracket(scanner.get(0))) {
                        scanner.set(closeBracIndex, "");
                        scanner.set(0, "");
                        exitForLoop = closeBracIndex <= scanner.size();
                        break;
                    }
                }//end while

            }//end if

            if (exitForLoop) {
                break;
            }

        }//end for

        scanner.removeAll(whitespaceremover);

    }//end method

    public void statsVerifier() {
        scanner.removeAll(whitespaceremover);

        //determine the presence of list returning statistical operators
        for (int i = 0; i < scanner.size(); i++) {
            if (Method.isListReturningStatsMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                noOfListReturningOperators++;
            }//end if

        }//end for

        correctFunction = ListReturningStatsOperator.validateFunction(this.scanner);
        parser_Result = correctFunction ? Parser_Result.VALID : Parser_Result.SYNTAX_ERROR;
        //processLogger.writeLog(ListReturningStatsOperator.getErrorMessage());

        if (noOfListReturningOperators > 0 && correctFunction) {
            setHasListReturningOperators(true);
            //disable usage of multiple list operators in function here by un-commenting the statement below
            // correctFunction=false;

            if (isHasListReturningOperators()) {

                scanner.remove(0);//temporarily remove the starting bracket
                scanner.remove(scanner.size() - 1);//temporarily remove the ending bracket

                for (int i = 0; i < scanner.size(); i++) {
                    try {
                        if (Method.isListReturningStatsMethod(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                            if (isBinaryOperator(scanner.get(i - 1))) {
                                //processLogger.writeLog("Invalid Association Discovered For: \""+scanner.get(i-1)+"\" And \""+scanner.get(i)+"\".\n");
                                correctFunction = false;
                                break;
                            }
                            if (isBracket(scanner.get(i - 1)) && !Method.isNumberReturningStatsMethod(scanner.get(i - 2))
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
     * e.g in structures like sort(3,sin2,2sin3,5,3,2,sin(4+5),4!...) This
     * method will help to reduce the input to the simple form i.e
     * sort(num1,num2,num3....)
     */
    private void evaluateCompoundStructuresInStatisticalInput() {
        if (hasListReturningOperators || hasNumberReturningStatsOperators) {

            for (int i = 0; i < scanner.size(); i++) {
                try {
                    if (Operator.isClosingBracket(scanner.get(i))) {
                        int open = Bracket.getComplementIndex(false, i, scanner);

                        if (!Method.isMethodName(scanner.get(open - 1))) {
                            Bracket opener = new Bracket("(");
                            opener.setIndex(open);
                            Bracket closer = new Bracket(")");
                            closer.setIndex(i);
                            opener.setComplement(closer);
                            closer.setComplement(opener);

                            if (opener.simpleBracketPairHasVariables(scanner)) {
                                continue;
                            }

                            String fun = opener.getDomainContents(scanner);
                            while (fun.startsWith("(") && fun.endsWith(")")) {
                                fun = fun.substring(1);
                                fun = fun.substring(0, fun.length() - 1);
                                fun = fun.trim();
                            }//end while
                            MathExpression f = new MathExpression(fun);
                            String val = f.solve();
                            scanner.add(open, val);
                            scanner.subList(open + 1, i + 2).clear();
                        }//end if
                        else if (Method.isDefinedMethod(scanner.get(open - 1))) {
                            int ind = open - 2;

                            while (Operator.isOpeningBracket(scanner.get(ind))) {
                                --ind;
                            }//end while
                            if (scanner.get(ind).equals("intg") || scanner.get(ind).equals("quad") || scanner.get(ind).equals("diff") || scanner.get(ind).equals("root")) {
                                Bracket opener = new Bracket("(");
                                opener.setIndex(open);
                                Bracket closer = new Bracket(")");
                                closer.setIndex(i);
                                opener.setComplement(closer);
                                closer.setComplement(opener);
                                if (opener.simpleBracketPairHasVariables(scanner)) {
                                    continue;
                                }
                                String fun = opener.getDomainContents(scanner);
                                while (fun.startsWith("(") && fun.endsWith(")")) {
                                    fun = fun.substring(1);
                                    fun = fun.substring(0, fun.length() - 1);
                                    fun = fun.trim();
                                }
                                MathExpression f = new MathExpression(fun);
                                String val = f.solve();
                                scanner.add(open, val);
                                scanner.subList(open + 1, i + 2).clear();
                            }//end if

                        }//end else if
                    }//end if
                }//end try
                catch (IndexOutOfBoundsException boundsException) {

                }
            }//end for
        }//end if
    }//end method

    /**
     * The method establishes meaning to some shorthand techniques in math that
     * the average mathematician might expect to see in a math device.
     * :e.g(3+4)(1+2) will become (3+4)*(1+2).It is essentially a stage that
     * generates code.
     */
    public void codeModifier() {

        detectKeyOperators();
        statsVerifier();
        evaluateCompoundStructuresInStatisticalInput();
        unBracketDataSetReturningStatsOperators();
        UnaryPostOperator.assignCompoundTokens(this.scanner);
        PowerOperator.assignCompoundTokens(this.scanner);

        if (correctFunction) {
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

            /**
             * The theory behind this is that any sub-expression that lies
             * between any 2 inBetweenOperators or between an in between
             * operator and the enclosing bracket: is a compound token and as
             * such must be evaluated first before execution of other parts of
             * the expression can proceed. For example: 3/2sin4+5 is scanned as:
             * [3,/,(,2,sin,4,),+,5] COME HEREEEEEEEEEEEEEE!!!!FOR COMPOUND
             * TOKENS IF NEEDED!!!!!!!!!!!!!!!!
             */
            scanner.removeAll(whitespaceremover);
            for (int i = 0; i < scanner.size(); i++) {

                /**
                 * Remember to do for the format:
                 * 3^sin2^sin3===3^sin(2^sin3)...correct not..(3^sin2)^sin3
                 * change things like 2sin3 to ( 2 * sin 3 ) Prevent the logic
                 * error: 3/4sin2==3/4*sin2 The LHS == 3/(4*sin2) THE RHS ==
                 * (3/4)*sin2
                 */
                try {
                    if ((isNumber(scanner.get(i)) || (isVariableString(scanner.get(i)) && !Method.isDefinedMethod(scanner.get(i)))) && (Method.isUnaryPreOperatorORDefinedMethod(scanner.get(i + 1))
                            || Method.isNumberReturningStatsMethod(scanner.get(i + 1)) || Method.isLogOrAntiLogToAnyBase(scanner.get(i + 1)))) {

                        //Determine the placement of the close bracket
                        int j = i + 1;
                        while (Method.isUnaryPreOperatorORDefinedMethod(scanner.get(j)) || Method.isNumberReturningStatsMethod(scanner.get(j)) || Method.isLogOrAntiLogToAnyBase(scanner.get(j))) {
                            ++j;
                        }
                        if (isNumber(scanner.get(j)) || isVariableString(scanner.get(j))) {
                            scanner.add(j + 1, ")");
                            scanner.add(i, "(");
                            scanner.add(i + 2, MULTIPLY);
                        } else if (isOpeningBracket(scanner.get(j))) {
                            int ind = Bracket.getComplementIndex(true, j, scanner);
                            scanner.add(ind + 1, ")");
                            scanner.add(i, "(");
                            scanner.add(i + 2, MULTIPLY);

                        }

                    }//end if
                }//end try.
                catch (IndexOutOfBoundsException indErr) {
                }//end catch

                try {
                    /**
                     * tackles situations like sin-(2+3) by converting them into
                     * sin(-1*(2+3)). The generic situation is
                     * "preNumberOperator-(expr)"
                     */
                    if (Method.isUnaryPreOperatorORDefinedMethod(scanner.get(i - 1)) && Operator.isPlusOrMinus(scanner.get(i)) && isOpeningBracket(scanner.get(i + 1))) {
                        if (scanner.get(i).equals(MINUS)) {
                            List<String> subList = scanner.subList(i - 1, Bracket.getComplementIndex(true, i + 1, scanner) + 1);

                            subList.set(1, "(");
                            subList.add(2, "-1");
                            subList.add(3, MULTIPLY);
                            subList.add(")");
                        }//end if
                        else if (scanner.get(i).equals(PLUS)) {
                            scanner.set(i, "");
                        }

                    }//end if
                }//end try
                catch (IndexOutOfBoundsException ind) {
                }//end catch

                try {
                    /**
                     * tackles situations like (-sin2+3) which it converts into
                     * -1*sin2+3 if this is not done, it would be evaluated as
                     * -(sin2+3)
                     */
                    if (isOpeningBracket(scanner.get(i - 1)) && Operator.isPlusOrMinus(scanner.get(i)) && Method.isUnaryPreOperatorORDefinedMethod(scanner.get(i + 1))) {
                        if (scanner.get(i).equals(MINUS)) {
                            scanner.set(i, "-1");
                            scanner.add(i + 1, MULTIPLY);
                        } else if (scanner.get(i).equals(PLUS)) {
                            scanner.set(i, "");
                        }
                    }
                }//end try
                catch (IndexOutOfBoundsException ind) {
                }//end catch

            }//end for

            scanner.removeAll(whitespaceremover);

        } else if (!correctFunction) {
            //processLogger.writeLog("Beginning Parser Shutdown Tasks Due To Errors In User Input.");
        }

        detectKeyOperators();
    }//end method codeModifier

    /**
     * Serves as a powerful optimizer of the evaluation section as it can govern
     * what sections of code will be executed and which ones will be ignored.
     */
    public void detectKeyOperators() {
        for (int i = 0; i < scanner.size(); i++) {
            if (isPlusOrMinus(scanner.get(i))) {
                setHasPlusOrMinusOperators(true);
            } else if (isUnaryPreOperator(scanner.get(i))) {
                setHasPreNumberOperators(true);
            } else if (isUnaryPostOperator(scanner.get(i))) {
                setHasPostNumberOperators(true);
            } else if (isMulOrDiv(scanner.get(i))) {
                setHasMulOrDivOperators(true);
            } else if (isPermOrComb(scanner.get(i))) {
                setHasPermOrCombOperators(true);
            } else if (isRemainder(scanner.get(i))) {
                setHasRemainderOperators(true);
            } else if (Method.isNumberReturningStatsMethod(scanner.get(i))) {
                setHasNumberReturningStatsOperators(true);
            } else if (isPower(scanner.get(i))) {
                setHasPowerOperators(true);
            } else if (isLogicOperator(scanner.get(i))) {
                setHasLogicOperators(true);
            }
        }//end for

    }

    /**
     *
     * @param scanner The ArrayList object that holds the string values in the
     * scanned function.
     * @return a Bracket array that holds related brackets pairs.
     */
    public static Bracket[] mapBrackets(ArrayList<String> scanner) {

        ArrayList<String> whiteSpaceRemover = new ArrayList<>();
        whiteSpaceRemover.add(" ");
        scanner.removeAll(whiteSpaceRemover);
        ArrayList<Bracket> bracs = new ArrayList<>();

        ArrayList<String> scan = new ArrayList<>();
        scan.addAll(scanner);
        int open = 0;//tracks the index of an opening bracket
        int close = scan.indexOf(")");//tracks the index of a closing bracket
        int i = 0;
        while (close != -1) {
            try {
                open = LISTS.prevIndexOf(scan, close, "(");
                Bracket openBrac = new Bracket("(");
                Bracket closeBrac = new Bracket(")");
                openBrac.setIndex(open);
                closeBrac.setIndex(close);
                openBrac.setComplement(closeBrac);
                closeBrac.setComplement(openBrac);

                bracs.add(openBrac);
                bracs.add(closeBrac);

                scan.set(open, "");
                scan.set(close, "");

                close = scan.indexOf(")");
                ++i;
            }//end try
            catch (IndexOutOfBoundsException ind) {
                break;
            }
        }//end while

//after the mapping the algorithm demands that all ( and ) should have been used up in the function
        if (scan.indexOf("(") == -1 && scan.indexOf(")") == -1) {
            int size = bracs.size();
            Bracket[] bracket = new Bracket[size];
            return bracs.toArray(bracket);
        } else {
            throw new InputMismatchException("SYNTAX ERROR!");
        }

    }//end method

    /**
     * Method mapBrackets goes over an input equation and maps all positions
     * that have corresponding brackets
     */
    public void mapBrackets() {
        try {
            setBracket(mapBrackets(scanner));
        }//end method
        catch (InputMismatchException ime) {
            parser_Result = Parser_Result.PARENTHESES_ERROR;
            setCorrectFunction(false);
            scanner.clear();
        }//end catch

    }//end method

    /**
     *
     * method finishUpScanner does final adjustments to the scanner function e.g
     * it will check for errors in operator combination in the scanner function
     * and so on
     */
    public void functionComponentsAssociation() {

        if (correctFunction) {
            scanner.removeAll(whitespaceremover);//remove white spaces that may result from past parser actions
//check for good combinations of operators and numbers and dis-allow any other.

            for (int i = 0; i < scanner.size(); i++) {
//check for the various valid arrangements for all members of the function.
                String token = scanner.get(i);
                //Variables
                if (isVariableString(scanner.get(i)) && !Method.isUserDefinedFunction(token)) {
                    try {
                        //specify valid tokens that can come before a variable
                        if (!isOpeningBracket(scanner.get(i - 1))
                                && !isLogicOperator(scanner.get(i - 1)) && !isUnaryPreOperator(scanner.get(i - 1))
                                && !isBinaryOperator(scanner.get(i - 1)) && !isAssignmentOperator(scanner.get(i - 1)) && !isNumber(scanner.get(i - 1))
                                && !isVariableString(scanner.get(i - 1))) {
                            //processLogger.writeLog("ParserNG Does Not Allow "+expression+" To Combine The MathExpression Members \""+scanner.get(i-1)+"\" And \""+scanner.get(i)+"\"\n");
                            correctFunction = false;
                            scanner.clear();
                            break;
                        }//end if
                        //specify valid tokens that can come after a variable
                        if (!isBracket(scanner.get(i + 1)) && !isBinaryOperator(scanner.get(i + 1))
                                && !isUnaryPostOperator(scanner.get(i + 1)) && !Method.isNumberReturningStatsMethod(scanner.get(i + 1))
                                && !isLogicOperator(scanner.get(i + 1)) && !isAssignmentOperator(scanner.get(i + 1))
                                && !isUnaryPreOperator(scanner.get(i + 1)) && !Method.isNumberReturningStatsMethod(scanner.get(i + 1))
                                && !Method.isLogToAnyBase(scanner.get(i + 1)) && !Method.isAntiLogToAnyBase(scanner.get(i + 1)) && !isNumber(scanner.get(i + 1))
                                && !isVariableString(scanner.get(i + 1))) {
                            //processLogger.writeLog("ParserNG Does Not Allow "+expression+" To Combine The MathExpression Members \""+scanner.get(i)+"\" And \""+scanner.get(i+1)+"\"PLUS As You Have Done.\n");

                            correctFunction = false;
                            scanner.clear();
                            break;
                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
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
                bracket = null;
                parser_Result = Parser_Result.SYNTAX_ERROR;
            }//end else

        }//end if
        else {
            scanner.clear();
            bracket = null;
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
    public void setVariableValuesInFunction(ArrayList<String> scan) {

        int sz = scan.size();
        for (int i = 0; i < sz; i++) {
            String varName = scan.get(i);

            try {
                if (i + 1 < sz) {//a next token exists (at i+1) so check the next token
                    if (isVariableString(scan.get(i)) && !isOpeningBracket(scan.get(i + 1))) {
                        Variable v = VariableManager.lookUp(varName);
                        if (v != null) {
                            scan.set(i, v.getValue());
                        }

                    }//end if
                } else {//no next token exists
                    if (isVariableString(scan.get(i))) {
                        Variable v = VariableManager.lookUp(varName);
                        if (v != null) {
                            scan.set(i, v.getValue());
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
    public String getValue(String name) throws NullPointerException {
        Variable var = variableManager.lookUp(name);
        return var == null ? null : var.getValue();
    }

    /**
     *
     * @param name The name of the variable or constant.
     * @param value The value to set to the variable
     * @throws NullPointerException if a Variable object that has that name id
     * not found.
     */
    public void setValue(String name, String value) throws NullPointerException, NumberFormatException {
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
     * @param increment the amount by which each bracket index is to be
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
//values greater than the value stored by the bracket from which looping was started the
// last time represent the indices of bracket in the function list that the compression function
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
     * Display the indices of all brackets in the function,bracket pair by
     * bracket pair
     */
    private void displayIndicesStoredInBrackets() {
        ArrayList<Integer> arr = new ArrayList<Integer>();
        int size = bracket.length;
        for (int i = 0; i < size; i++) {
            arr.add(bracket[i].getIndex());
        }
        /**
         * This method is not a useful one in the overall running. It is only
         * used by the coder to make sure that the system is parsing the
         * brackets properly.
         */
    }

    /**
     *
     * @param scan The ArrayList object.
     * @return the string version of the ArrayList and removes the braces i.e.
     * []
     */
    protected String listToString(ArrayList<String> scan) {
        String str = String.valueOf(scan);
        str = str.substring(1);
        str = str.substring(0, str.length() - 1);
        return str;
    }

    /**
     *
     * @return an Array object containing duplicate contents of the List object
     * alone
     */
    protected Bracket[] copyArrayToArray() {
        int size = bracket.length;
        Bracket[] obj = new Bracket[size];
        if (bracket != null) {
            for (int i = 0; i < size; i++) {
                obj[i] = bracket[i].createTwinBracket();
            }
        } else {
            throw new NullPointerException("Null List Cannot Be Copied");
        }
        return obj;
    }

    public void setReturnType(TYPE returnType) {
        this.returnType = returnType;
    }

    public TYPE getReturnType() {
        return returnType;
    }

    /**
     * Method solve is the main parser used to evaluate the input multi-bracket
     * pair (MBP) expressions used to initialize the constructor of class
     * MathExpression
     *
     * @return the result of the evaluation
     */
    public String solve() {

        if (correctFunction && !hasFunctionOrVariableInitStatement) {
            final ArrayList<String> myScan = new ArrayList<String>();

            myScan.addAll(scanner);
            Bracket[] brac = mapBrackets(myScan);
            int indexOpenInMyScan = 0;
            int indexCloseInMyScan = 0;

            setVariableValuesInFunction(myScan);

            String listAppender = "";

            while (brac.length > 0) {

                if (!correctFunction) {
                    break;
                }//end if
                else {
                    try {
                        indexOpenInMyScan = brac[0].getIndex();
                        indexCloseInMyScan = brac[1].getIndex();

                        boolean isMethod = false;//only list returning data sets e.g sort,rnd...

                        List<String> executable = null;
                        try {
                            isMethod = Method.isMethodName(myScan.get(indexOpenInMyScan - 1));
                        }//end try
                        catch (IndexOutOfBoundsException indexErr) {
                            isMethod = false;
                        }

                        try {
                            executable = myScan.subList(indexOpenInMyScan, indexCloseInMyScan + 1);//
                        }//end try
                        catch (IndexOutOfBoundsException indexErr) {
                        }
                        if (!isMethod) {
                            solve(executable);
                        }//end if
                        else if (isMethod) {
            
                            try {
                                /**
                                 * Get the view of the scanner between the
                                 * method and the closing bracket.. e.g.
                                 * [....sin,(,3,),....] stores [sin,(,3,)] in
                                 * executable.
                                 */
                                executable = myScan.subList(indexOpenInMyScan - 1, indexCloseInMyScan + 1);
                                if (Method.isStatsMethod(executable.get(0))) {

                                    if (!Method.isUserDefinedFunction(executable.get(0))) {
                                        Method.run(executable, DRG);
                                    }//end if
                                    else if (Method.isUserDefinedFunction(executable.get(0))) {
                                        Function f = FunctionManager.lookUp(executable.get(0));

                                        if (f.getIndependentVariables().size() <= 1) {
                                            solve(executable.subList(1, executable.size()));
                                            executable.add(")");
                                            executable.add(1, "(");
                                            Method.run(executable, DRG);
                                        } else {
                                            Method.run(executable, DRG);
                                        }//end else

                                    }//end else if

                                }//end if
                                else {
                                    
                                    solve(executable.subList(1, executable.size()));
                                    executable.add(")");
                                    executable.add(1, "(");
                                    Method.run(executable, DRG);

                                }

                            } catch (IndexOutOfBoundsException indexErr) {
                                break;
                            } catch (IllegalArgumentException ill) {
                                return "SYNTAX ERROR";
                            } catch (NullPointerException nolException) {
                                break;
                            }
                        }//end else if
                        brac = mapBrackets(myScan);
                    }//end try
                    catch (IndexOutOfBoundsException indexErr) {
                        indexErr.printStackTrace();
                        return "SYNTAX ERROR";
                    }//end catch
                    catch (NumberFormatException numErr) {
                        numErr.printStackTrace();
                        return "SYNTAX ERROR";
                    }//end catch
                    catch (InputMismatchException exception) {
                        exception.printStackTrace();
                        return "SYNTAX ERROR";
                    }
                }//end else

            }//end while

            listAppender = listToString(myScan);
            if (listAppender.startsWith("(")) {
                listAppender = listAppender.substring(1);
                listAppender = listAppender.substring(0, listAppender.length() - 1);

                if (isComma(listAppender.substring(0, 1))) {
                    listAppender = listAppender.replace(",", "");
                }

            }

            if (validNumber(listAppender)) {
                returnType = TYPE.NUMBER;
            } else if (listAppender.contains(",")) {
                returnType = TYPE.STRING;
            }
            Function f;
            if ((f = FunctionManager.lookUp(listAppender)) != null) {
                if (f.getType() == Function.MATRIX) {
                    listAppender = f.getMatrix().toString();
                    returnType = TYPE.MATRIX;
                } else if (f.getType() == Function.LIST) {
                    listAppender = f.toString();
                    returnType = TYPE.LIST;
                } else if (f.getType() == Function.ALGEBRAIC) {
                    returnType = TYPE.ALGEBRAIC_EXPRESSION;
                    listAppender = f.toString();
                }

            }

            //designed to deduce if or not the evaluating loop executed normally.
            //If it didn't the statements in the else will execute
            if (correctFunction) {
                lastResult = listAppender;

            }//end if
            //give an error statement and then reset correctFunction to true;
            else {
                listAppender = "A SYNTAX ERROR OCCURRED";
                correctFunction = true;
            }
            if (myScan.size() == 1) {
                returnObjectName = myScan.get(0);
            }
            return listAppender;
        }//end if
        else if (hasFunctionOrVariableInitStatement) {
            return "Variable Storage Process Finished!";
        } else {
            return "SYNTAX ERROR";
        }

    }//end method solve()

    /**
     * used by the main parser solve to figure out SBP portions of a
     * multi-bracketed expression (MBP)
     *
     * @param list a list of scanner tokens of a maths expression
     * @return the solution to a SBP maths expression
     */
    public List<String> solve(List<String> list) {

//correct the anomaly: [ (,-,number....,)  ]
        //   turn it into: [ (,,-number........,)     ]
        //The double commas show that there exists an empty location in between the 2 commas
        if (list.get(0).equals("(") && list.get(1).equals(MINUS) && isNumber(list.get(2))) {
            list.set(1, "");
            //if the number is negative,make it positive
            if (list.get(2).substring(0, 1).equals(MINUS)) {
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
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i)), 2)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else
                    } else if (isCube(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i)), 3)));
                            list.set(i, "");
                        }//end if
                        else if (list.get(i).equals("Infinity")) {
                            list.set(i + 1, "Infinity");
                            list.set(i, "");
                        }//end else
                    } else if (isInverse(list.get(i + 1))) {
                        if (isNumber(list.get(i))) {
                            list.set(i + 1, String.valueOf(1 / Double.valueOf(list.get(i))));
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
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i - 1)), Double.valueOf(list.get(i + 1)))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                    }//end if
                    else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i + 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) == 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) > 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) == 0) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) < 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i - 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) == 1) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) > 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) == 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) < 0) {
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
                            list.set(i, String.valueOf(Math.sqrt(Double.valueOf(list.get(i + 1)))));
                            list.set(i + 1, "");
                        }//end if
                        if (list.get(i).equals(CUBE_ROOT)) {
                            list.set(i, String.valueOf(Math.cbrt(Double.valueOf(list.get(i + 1)))));
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
                            list.set(i + 1, String.valueOf(Math.pow(Double.valueOf(list.get(i - 1)), Double.valueOf(list.get(i + 1)))));
                            list.set(i - 1, "");
                            list.set(i, "");
                        }//end if
                    }//end if
                    else if (list.get(i - 1).equals("Infinity") && !list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i + 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) == 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) > 0) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) == 0) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i + 1)) < 1 && Double.valueOf(list.get(i + 1)) < 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        }
                    } else if (!list.get(i - 1).equals("Infinity") && list.get(i + 1).equals("Infinity")) {
                        if (Double.valueOf(list.get(i - 1)) > 1) {
                            list.set(i + 1, "Infinity");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) == 1) {
                            list.set(i + 1, "1.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) > 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) == 0) {
                            list.set(i + 1, "0.0");
                            list.set(i - 1, "");
                            list.set(i, "");
                        } else if (Double.valueOf(list.get(i - 1)) < 1 && Double.valueOf(list.get(i - 1)) < 0) {
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
                            list.set(i + 1, String.valueOf(Double.valueOf(Maths.fact(list.get(i - 1))) / (Double.valueOf(Maths.fact(String.valueOf(Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))))))));
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
                            list.set(i + 1, String.valueOf(Double.valueOf(Maths.fact(list.get(i - 1))) / (Double.valueOf(Maths.fact(String.valueOf(Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))))) * Double.valueOf(Maths.fact(list.get(i + 1))))));
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
                            list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) * Double.valueOf(list.get(i + 1))));
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
                            list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) / Double.valueOf(list.get(i + 1))));
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
                            list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) % Double.valueOf(list.get(i + 1))));
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
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) == 0));
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
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) > 0));
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
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) >= 0));
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
                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) < 0));
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

                                list.set(i + 1, String.valueOf((Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))) <= 0));
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
                                list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) + Double.valueOf(list.get(i + 1))));
                                list.set(i - 1, "");
                                list.set(i, "");
                            }//end else
                            else if (list.get(i).equals(MINUS) && isNumber(list.get(i - 1)) && isNumber(list.get(i + 1))) {
                                list.set(i + 1, String.valueOf(Double.valueOf(list.get(i - 1)) - Double.valueOf(list.get(i + 1))));
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
     *
     * @param scanner is a list of scanner functions, gotten during the
     * evaluation of sets of data that contain functions that need to be
     * evaluated instead of numbers.If the data set does not contain functions
     * e.g avg(2,3,7,1,0,9,5), then method solve will easily solve it. But if it
     * does e.g avg(2,sin,3,5,cos,(,5,) ), then we invoke this method in class
     * Set's constructor before we evaluate the data set. Note this is method is
     * not called directly by MathExpression objects but by objects of class Set
     * invoked by a MathExpression object.
     *
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
                j = LISTS.prevIndexOf(scanner, i, "("); //index of enclosing bracket of ) above
                List<String> sub = scanner.subList(j, i + 1);
                solve(sub);

            }//end try
            catch (IndexOutOfBoundsException indexerr) {
                break;
            }//end catch
        }//end while

        return scanner;
    }//end method solveSubPortions()

    private static void moreJunkExamples() {
        System.out.println(0xFF000000);
        System.out.println(0xFF888888);
        System.out.println(0xFFFFFFFF);
        Function f = FunctionManager.add("f(x,y) = x - x/y");

        int iterations = 1000000;

        double start = System.nanoTime();

        double val = 0;
        for (int i = 1; i < iterations; i++) {
            f.calc(2, 3);
        }

        start = System.nanoTime() - start;

        System.out.println("Time: " + (start / iterations) + " ns");
    }

    private static void junkExamples() {

        MathExpression linear = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);C=matrix_sub(M,N);C;");
        System.out.println("soln: " + linear.solve());

        MathExpression expr = new MathExpression("tri_mat(M)");
        System.out.println(expr.solve());

        expr = new MathExpression("echelon(M)");
        System.out.println(expr.solve());

        Function matrixFunction = FunctionManager.lookUp("M");

        Matrix matrix = matrixFunction.getMatrix();
        System.out.println("underlying matrix: " + matrix);

        Matrix inv = matrix.inverse();

        System.out.println("inverted matrix: " + inv);

        matrix.multiply(inv);

        System.out.println("mul matrix: " + matrix);

        FunctionManager.add("f(x,y) = x-x/y");
        Function fxy = FunctionManager.lookUp("f");

        double iterations = 100;

        for (int i = 0; i < iterations; i++) {
            fxy.calc(i + 3);
        }

        MathExpression parserng = new MathExpression("f(x,y) = x-x/y; f(2,3);f(2,5);");
        System.out.println("SEE???????\n " + parserng.solve());

        /*
         MathExpression f = new MathExpression("x=17;3*x+1/x");//runs in about 2.3 milliSecs

         System.out.println(f.solve());
         f.setExpression("x^3");
         System.out.println(f.variableManager);

         System.out.println(f.solve());
         *
         String fun =
         "x=3;x^3";

         MathExpression f = new MathExpression(fun);
         String ans ="";
         double start = System.nanoTime();

         for(int i=0;i<1;i++){
         ans = f.solve();
         }

         double time = (System.nanoTime()-start)/1.0E6;
         System.out.println("ans = "+ans+" calculated in "+time+" ms");

         */
        /**
         * On balanced CPU usage mode, Before optimization with StringBuilder,
         * f.solve() takes about 7ms to run the above 1000 string operation and
         * about 500ms to run the MathExpression constructor's parsing
         * techniques.
         *
         * After optimization,Sim
         *
         *
         */
        //3,log(2,4),8,9
        //FunctionManager.add("f=@(x)sin(x)-x");//A=4;sum(3A,4,4+cos(8),5,6+sin(3),2!,7A,3E-8+6,4)
        //Formula...................t_root(@(x)2.2x^3+3*x+8*x^0)
        //MathExpression express = new MathExpression("sum(root(f,2,3),1,2,4,2,9)");//³√ diff(@(x)3*x²+5*³√(x),3),diff(@(x)3*x,3)
        //MathExpression express = new MathExpression("sum(root(f,2,3),1,2,4,2,9)");//³√ diff(@(x)3*x²+5*³√(x),3),diff(@(x)3*x,3)
        //³√ diff(@(x)3*x²+5*³√(x),3),diff(@(x)3*x,3)
        /*
         3 4  3 4   13 20
         1 2  1 2   5  8                 -22
         */
        FunctionManager.add("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        FunctionManager.add("M1=@(2,2)(3,4,1,2)");
        FunctionManager.add("M2=@(2,2)(3,4,1,2)");

        FunctionManager.add("N=@(x)(sin(x))");
        FunctionManager.add("r=@(x)(ln(sin(x)))");

        //WORK ON sum(3,-2sin(3)^2,4,5) error //matrix_mul(@(2,2)(3,1,4,2),@(2,2)(2,-9,-4,3))...sum(3,2sin(4),5,-3cos(2*sin(5)),4,1,3)
        //matrix_mul(invert(@(2,2)(3,1,4,2)),@(2,2)(2,-9,-4,3)).............matrix_mul(invert(@(2,2)(3,1,4,2)),@(2,2)(2,-9,-4,3))
        //matrix_mul(invert(@(2,2)(3,1,4,2)),matrix_mul(M2,2sin(3)sum(3,6,5,3)cos(9/8*6)det(@(2,2)(2,-9,-4,3))))
        System.out.println("lookup M: " + FunctionManager.lookUp("M"));
        //MathExpression expr = new MathExpression("f=3;5f");//BUGGY
        //MathExpression expr = new MathExpression("quad(@(x)3*x-2+3*x^2)");//BUGGY
        //MathExpression expr = new MathExpression("root(@(x)3*x-sin(x)-0.5,2)");//BUGGY
        MathExpression exprs = new MathExpression("r1=4;r1*5");

        //A+k.A+AxB+A^c
        System.out.println("scanner: " + exprs.scanner);
        System.out.println("solution: " + exprs.solve());

        expr.setExpression("44+22*(3)");
        System.out.println("solution--: " + exprs.solve());

        System.out.println("return type: " + exprs.returnType);
        System.out.println("FunctionManager: " + FunctionManager.FUNCTIONS);
        System.out.println("VariableManager: " + VariableManager.VARIABLES);

        MathExpression expression = new MathExpression("x=0;sin(ln(x))");

        for (int i = 0; i < 100; i++) {
            expression.setValue("x", i + "");
            System.out.println(expression.solve());
        }

        System.out.println(">>> Finished.");

        Function f = FunctionManager.lookUp("N");

        double start = System.nanoTime();

        iterations = 100;

        for (int i = 0; i < iterations; i++) {
            f.calc(i + 3);
        }

        double elapsedNanos = (System.nanoTime() - start) / iterations;

        System.out.println("DONE: " + (elapsedNanos / 1.0E6) + " ms");

        MathExpression ex = new MathExpression("det(@(5,5)(-21,12,13,64,5,6,2.7,18,9,0,4,2,3,4,23,6,7,8,9,0,1,2,32,4,5));");
        System.out.println("determinant: " + ex.solve());

        /**
         * On my Macbook Pro, 16GB RAM; 2.6 GHz Intel Core i7
         *
         * The code runs the solve() method at 3.8 microseconds.
         */
    }

    public static void main(String args[]) {
        MathExpression expr = new MathExpression("M=@(3,3)(3,4,1,2,4,7,9,1,-2)");
        System.out.println(expr.solve());
        //    junkExamples();
        moreJunkExamples();

    }//end method

    @Override
    public String serialize() {
        return Serializer.serialize(this);
    }

    public MathExpression parse(String enc) {
        return (MathExpression) Serializer.deserialize(enc);
    }

}//end class MathExpression
