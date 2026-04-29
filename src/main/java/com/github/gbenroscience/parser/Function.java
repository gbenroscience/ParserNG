/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.interfaces.Savable;
import com.github.gbenroscience.logic.DRG_MODE;

import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;

import com.github.gbenroscience.math.matrix.expressParser.Matrix;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.parser.methods.MethodRegistry;
import com.github.gbenroscience.parser.turbo.tools.*;
import com.github.gbenroscience.util.ErrorLog;
import com.github.gbenroscience.util.FunctionManager;
import static com.github.gbenroscience.util.FunctionManager.*;
import com.github.gbenroscience.util.Serializer;
import com.github.gbenroscience.util.Utils;
import com.github.gbenroscience.util.VariableManager;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class Function implements Savable, MethodRegistry.MethodAction {

    /**
     * The dependent variable
     */
    private Variable dependentVariable;
    /**
     * The independent variables.
     */
    private ArrayList<Variable> independentVariables = new ArrayList<>();

    /**
     * The type of the function
     */
    private TYPE type = TYPE.ALGEBRAIC_EXPRESSION;

    /**
     * If the object is an algebraic expression, its details are stored here.
     * The math expression on the RHS.
     */
    private MathExpression mathExpression;

    private FastCompositeExpression turboExpr;

    /**
     * If the object is a {@link Matrix} its data is stored here.
     */
    private Matrix matrix;

    ErrorLog errorLog = new ErrorLog();

    public Function() {
    }

    /**
     *
     * @param matrix A Matrix to be used to initialize the function..
     */
    public Function(Matrix matrix) {
        this.matrix = matrix;
        this.type = TYPE.MATRIX;

        String fName = this.matrix.getName();

        Function oldFunc = FUNCTIONS.get(fName);
        Variable v = VariableManager.lookUp(fName);//check if a Variable has this name in the Variables registry
        if (v != null) {
            VariableManager.delete(fName);//if so delete it.
        }//end if 

        if (oldFunc != null) {//function does not exist in registry
            FunctionManager.delete(fName);
        }
        FunctionManager.add(this);
        FunctionManager.update();
    }

    /**
     *
     * @param input The user input into the system, usually of the form:
     * F(x,y,z,w,....)=mathexpr; or F= @(x,y,z,w,...)mathexpr.....where mathexpr
     * is an algebraic expression in terms of x,y,z,w,... OR JUST PLAIN
     * @(x,y,...)expr in which case an anonymous function will be created
     *
     */
    public Function(String input) throws InputMismatchException {
        try {
            input = Utils.unwrapBracket(input);
            input = rewriteAsStandardFunction(STRING.purifier(input), errorLog);//Change function to standard form immediately
            parseInput(input);
        } catch (Exception e) {
            e.printStackTrace();
            String err = "Bad Function Syntax--" + input;
            errorLog.info(err);
            throw new InputMismatchException(err);
        }
    }//end constructor

    /**
     * Takes a string in the format: F(args)=expr and rewrites it as
     * F=@(args)expr
     *
     * @param input The input string
     * @return the properly formatted string
     */
    private static String rewriteAsStandardFunction(String input, ErrorLog log) {
        int indexOfOpenBrac = -1;
        int indexOfCloseBrac = -1;
        int indexOfAt = -1;
        int indexOfEquals = -1;

        for (int i = 0; i < input.length(); i++) {
            switch (input.charAt(i)) {
                case '(':
                    indexOfOpenBrac = indexOfOpenBrac == -1 ? i : indexOfOpenBrac;
                    break;
                case ')':
                    indexOfCloseBrac = indexOfCloseBrac == -1 ? i : indexOfCloseBrac;
                    break;
                case '@':
                    indexOfAt = indexOfAt == -1 ? i : indexOfAt;
                    break;
                case '=':
                    indexOfEquals = indexOfEquals == -1 ? i : indexOfEquals;
                    break;
                default:
                    break;
            }
            if (indexOfOpenBrac != -1 && indexOfCloseBrac != -1 && indexOfEquals != -1 && indexOfAt != -1) {
                break;
            }
        }

        if (indexOfEquals == -1) {//MUST BE AN ANONYMOUS Function....e.g @(args)expr format alone was entered
            if (indexOfAt == 0) {//ENFORCE MUST BE ANONYMOUS FUNCTION with no assignmenr statemet
                if (indexOfOpenBrac == indexOfAt + 1) {
                    if (indexOfCloseBrac > indexOfOpenBrac) {
                        return input;
                    } else {
                        String err = "Bracket Error in Function creation";
                        log.info(err);
                        throw new InputMismatchException(err);
                    }
                } else {
                    String err = "Function definition syntax error in token structure";
                    log.info(err);
                    throw new InputMismatchException(err);
                }
            } else {
                String err = "The function is supposed to be an anonymous one. SYNATX ERROR";
                log.info(err);
                throw new InputMismatchException(err);
            }
        }

        if (indexOfOpenBrac == -1 || indexOfCloseBrac == -1) {// MUST HAVES, if not INVALID FUNCTION
            String err = "Core tokens not found in Function expression.. one or more of (, ) and = not found!";
            log.info(err);
            throw new InputMismatchException(err);
        }

        int computeCloseBracIndex = Bracket.getComplementIndex(true, indexOfOpenBrac, input);//this is the index of the matching close bracket for the args list
        if (computeCloseBracIndex != indexOfCloseBrac) {//Is a major structural flaw in the input...e.g f=@(((args))expr, but this is not allowed! only f=@(args)expr is
            String err = "Multiple brackets not allowed on args list e.g: f((x)) is not allowed, only f(x) and f=@((x)) is not allowed";
            log.info(err);
            throw new InputMismatchException(err);
        }

        if (indexOfOpenBrac < indexOfEquals && indexOfCloseBrac < indexOfEquals && indexOfOpenBrac < indexOfCloseBrac) {//GOTCHA in f(args)=expr format
            return input.substring(0, indexOfOpenBrac) + "=@" + input.substring(indexOfOpenBrac, indexOfCloseBrac + 1) + input.substring(indexOfEquals + 1);//Convert to standard and exit
        }
        //check if already standard input....F=@(args)expr     
        if (indexOfAt != -1 && indexOfEquals < indexOfAt && indexOfAt < indexOfOpenBrac && indexOfOpenBrac < indexOfCloseBrac) {//likely standard input, exit early
            if (indexOfAt - indexOfEquals == 1 && indexOfOpenBrac - indexOfAt == 1) {
                return input;
            } else {
                String err = "Function definition is not valid. Invalid token structure";
                log.info(err);
                throw new InputMismatchException(err);
            }
        }
        String err = "Your Function definition is not valid.. " + input;
        log.info(err);
        throw new InputMismatchException(err);

    }

    public void setType(TYPE type) {
        this.type = type;
    }

    public TYPE getType() {
        return type;
    }

    public void updateArgs(double... x) {
        MathExpression m = getMathExpression();
        if (m != null) {
            m.updateArgs(x);
        }
    }

    /**
     * @return the value of the function with these variables set.
     */
    public double calc() {
        return mathExpression.solveGeneric().scalar;
    }

    /**
     *
     * @param x A list of variable values to set for the function. The supplied
     * value list is applied to the function's parameter list in the order they
     * were supplied in the original question.
     * @return the value of the function with these variables set.
     */
    public MathExpression.EvalResult calc(MathExpression.EvalResult[] x) {

        if (type == TYPE.ALGEBRAIC_EXPRESSION) {
            if (x.length == independentVariables.size()) {

                for (int i = 0; i < x.length; i++) {
                    Variable v = independentVariables.get(i);
                    String vName = v.getName();
                    int slot = mathExpression.getVariable(vName).getFrameIndex();
                    mathExpression.updateSlot(slot, x[i].scalar);
                }
                return mathExpression.solveGeneric();
            }
        } else if (type == TYPE.MATRIX) {
            MathExpression.EvalResult res = new MathExpression.EvalResult();
            return res.wrap(matrix);
        }
        return new MathExpression.EvalResult();
    }

    @Override
    public MathExpression.EvalResult calc(MathExpression.EvalResult nextResult, int arity, MathExpression.EvalResult[] x) {
        if (type == TYPE.ALGEBRAIC_EXPRESSION) {
            if (x.length == independentVariables.size()) {

                for (int i = 0; i < x.length; i++) {
                    Variable v = independentVariables.get(i);
                    String vName = v.getName();
                    int slot = mathExpression.getVariable(vName).getFrameIndex();
                    mathExpression.updateSlot(slot, x[i].scalar);
                }
                return nextResult.wrap(mathExpression.solveGeneric());
            }
        } else if (type == TYPE.MATRIX) {
            return nextResult.wrap(matrix);
        }
        return nextResult;
    }

    public static boolean assignObject(String input, ErrorLog log) {
        /**
         * Check if it is a function assignment operation...e.g:
         * f=matrix_mul(A,B)
         *
         */
        input = STRING.purifier(input);

        int equalsIndex = input.indexOf("=");
        int semiColonIndex = input.indexOf(";");

        int indexOfOpenBrac = input.indexOf("(");
        if (equalsIndex == -1 && semiColonIndex == -1 && indexOfOpenBrac == -1) {
            String err = "Wrong Input!";
            log.info(err);
            throw new InputMismatchException(err);
        }

        /**
         * Check if the user used the form f(x)=.... instead of f=@(x).... If so
         * convert to the latter format, and then recompute the necessary
         * indexes.
         */
        if (indexOfOpenBrac != -1 && indexOfOpenBrac < equalsIndex) {
            if (semiColonIndex == -1) {
                input = input.substring(0, indexOfOpenBrac)
                        + "=@" + input.substring(indexOfOpenBrac, Bracket.getComplementIndex(true, indexOfOpenBrac, input) + 1) + "(" + input.substring(equalsIndex + 1) + ")";
            } else {
                input = input.substring(0, semiColonIndex);
                input = input.substring(0, indexOfOpenBrac)
                        + "=@" + input.substring(indexOfOpenBrac, Bracket.getComplementIndex(true, indexOfOpenBrac, input) + 1) + "(" + input.substring(equalsIndex + 1) + ");";
            }

            //recompute the indexes.
            equalsIndex = input.indexOf("=");
            semiColonIndex = input.indexOf(";");
        }

        boolean success = false;

        if (equalsIndex != -1 && semiColonIndex != -1) {
            String newFuncName = input.substring(0, equalsIndex);
            boolean isVarNamesList = isParameterList("(" + newFuncName + ")");
            boolean hasCommas = newFuncName.contains(",");
            String rhs = input.substring(equalsIndex + 1, semiColonIndex);

            if (Number.validNumber(rhs)) {
                if (Variable.isVariableString(newFuncName)) {
                    Variable v = VariableManager.lookUp(newFuncName);
                    if (v == null) {
                        VariableManager.VARIABLES.put(newFuncName, new Variable(newFuncName, Double.parseDouble(rhs), false));
                    } else {
                        v.setValue(rhs);
                    }
                } else if (isVarNamesList) {
                    List<String> vars = new Scanner(newFuncName, false, ",").scan();
                    for (String var : vars) {
                        Variable v = VariableManager.lookUp(var);
                        if (v == null) {
                            VariableManager.VARIABLES.put(var, new Variable(var, Double.parseDouble(rhs), false));
                        } else {
                            v.setValue(rhs);
                        }
                    }
                }
                success = true;
            } else {
                MathExpression expr = null;
                if (rhs.startsWith("@")) {
                    Function anonFunc = new Function(rhs);
                    FunctionManager.update(anonFunc.dependentVariable.getName(), newFuncName);
                    if (anonFunc.isMatrix()) {
                        anonFunc.getMatrix().setName(newFuncName);
                    } else {
                        anonFunc.dependentVariable = new Variable(newFuncName);
                    }

                    FunctionManager.update(anonFunc);
                    return true;
                } else {
                    expr = new MathExpression(rhs);
                }

                List<String> scanner = expr.getScanner();

                if (!scanner.isEmpty() && scanner.get(0).equals("(")) {
                    int compIndex = Bracket.getComplementIndex(true, 0, scanner);
                    if (compIndex == scanner.size() - 1) {
                        scanner.remove(0);
                        scanner.remove(scanner.size() - 1);
                    }
                }

                int sz = scanner.size();
                if ((sz == 3 && scanner.get(1).startsWith(ANON_PREFIX))
                        || (sz == 1 && scanner.get(0).startsWith(ANON_PREFIX))) {//function assigments will always be like this: [(,anon1,)] when they get here

                    String anonFn = sz == 3 ? scanner.get(1) : scanner.get(0);
                    Function f = FunctionManager.lookUp(anonFn);
                    if (f != null) {
                        FunctionManager.delete(anonFn);
                        if (f.getType() == TYPE.ALGEBRAIC_EXPRESSION) {
                            Variable v = VariableManager.lookUp(newFuncName);
                            if (v != null) {
                                f.setDependentVariable(v);
                            } else {
                                f.setDependentVariable(new Variable(newFuncName));
                            }
                            FunctionManager.add(f);
                        } else if (f.getType() == TYPE.MATRIX) {
                            f.getMatrix().setName(newFuncName);
                            FunctionManager.add(f);
                        }
                    } else {
                        f = new Function(newFuncName + "=" + rhs + ";");
                        FunctionManager.add(f);
                    }
                    return true;
                }
                if (sz == 1 && Variable.isVariableString(rhs)) {//Handle assignments like D=C
                    Function f = FunctionManager.lookUp(rhs);
                    if (f != null) {
                        Function q = f.copy();
                        q.setDependentVariable(new Variable(newFuncName));
                        if (q.type == TYPE.MATRIX) {
                            q.getMatrix().setName(newFuncName);
                        }
                        FunctionManager.add(q);
                        return true;
                    }

                }
                if (scanner.get(0).equals(Declarations.ROTOR)) {
                    sz = scanner.size();
                    if (sz == 10) {//point or expression rotation---[rot, (, F, ,, pi, ,, anon2, ,, anon3, )]
                        String fn = scanner.get(2);
                        Function f = FunctionManager.lookUp(fn);
                        MathExpression.EvalResult ev = expr.solveGeneric();
                        if (f.type == TYPE.STRING || f.type == TYPE.ALGEBRAIC_EXPRESSION) {
                            String err = "rotating a function usually creates an implicit expression, do not try to assign the result of function rotation to a function.";
                            log.info(err);
                            throw new RuntimeException(err);
                            /**
                             * String fname = f.getFullName(); String fullName =
                             * "@" + fname.substring(fname.indexOf("(")); String
                             * fexpr = newFuncName + "=" + fullName +
                             * ev.textRes;
                             *
                             * Function rotF = new Function(fexpr); rotF =
                             * FunctionManager.lookUp(newFuncName); Variable v =
                             * VariableManager.lookUp(newFuncName); if (v !=
                             * null) { rotF.setDependentVariable(v); } else {
                             * rotF.setDependentVariable(new
                             * Variable(newFuncName)); }
                             * FunctionManager.add(rotF); return true;
                             */
                        }
                        if (f.type == TYPE.MATRIX) {//is a Point rotation
                            String fname = f.getFullName();
                            String fullName = "@" + fname.substring(fname.indexOf("("));
                            String fexpr = newFuncName + "=" + fullName + ev.toString().replace('[', '(').replace(']', ')');
                            Function rotP = new Function(fexpr);
                            FunctionManager.add(rotP);
                            return true;
                        }
                    } else if (sz == 12) {// line rotation: [rot, (, P, ,, Q ,, pi, ,, anon2, ,, anon3, )] P AND Q are 2 pts on the line

                        String pn = scanner.get(2);
                        String qn = scanner.get(4);
                        Function p = FunctionManager.lookUp(pn);
                        Function q = FunctionManager.lookUp(qn);
                        MathExpression.EvalResult ev = expr.solveGeneric();
                        if (p.type == TYPE.MATRIX && q.type == TYPE.MATRIX) {//is a Point rotation
                            ev.matrix.setName(newFuncName);
                            Function rotP = new Function(ev.matrix);
                            FunctionManager.add(rotP);
                            return true;
                        }
                    }

                }
                MathExpression.EvalResult val = expr.solveGeneric();

                String referenceName = null;

                if (Variable.isVariableString(newFuncName) || isVarNamesList) {
                    Function f;
                    switch (expr.getReturnType()) {
                        case MATRIX:
                            if (isVarNamesList && hasCommas) {
                                String err = "Initialize a function at a time!";
                                log.info(err);
                                throw new InputMismatchException(err);
                            }
                            val.matrix.setName(newFuncName);
                            Function fm = new Function(val.matrix);
                            success = true;
                            break;
                        case ALGEBRAIC_EXPRESSION:
                            if (isVarNamesList && hasCommas) {
                                String err = "Initialize a function at a time!!";
                                log.info(err);
                                throw new InputMismatchException(err);
                            }
                            f = FunctionManager.lookUp(referenceName);
                            FunctionManager.FUNCTIONS.put(newFuncName, new Function(newFuncName + "=" + f.expressionForm()));
                            success = true;
                            break;
                        case VECTOR:
                            if (isVarNamesList && hasCommas) {
                                String err = "Initialize a function at a time!!!";
                                log.info(err);
                                throw new InputMismatchException(err);
                            }
                            f = FunctionManager.lookUp(referenceName);
                            FunctionManager.FUNCTIONS.put(newFuncName, new Function(newFuncName + "=" + f.expressionForm()));
                            success = true;
                            break;
                        case STRING://for now, this only confirms that a comma separated list has been returned

                            break;
                        case NUMBER:
                            if (isVarNamesList && hasCommas) {
                                List<String> vars = new Scanner(newFuncName, false, ",").scan();
                                for (String var : vars) {
                                    Variable v = VariableManager.lookUp(var);
                                    if (v == null) {
                                        v = new Variable(var, val.scalar, false);
                                    } else {
                                        v.setValue(val.scalar);
                                    }
                                    VariableManager.VARIABLES.put(var, v);
                                }
                                success = true;
                            } else {
                                Variable v = VariableManager.lookUp(newFuncName);
                                if (v == null) {
                                    v = new Variable(newFuncName, val.scalar, false);
                                } else {
                                    v.setValue(val.scalar);
                                }
                                VariableManager.VARIABLES.put(newFuncName, v);
                                success = true;
                            }

                            break;
                        case VOID:

                            break;

                        default:

                            break;

                    }//end switch statement
                }//end if
                else {
                    String err = "Syntax Error---" + newFuncName;
                    log.info(err);
                    throw new InputMismatchException(err);
                }
            }//end else
        }
        if (success) {
            FunctionManager.update();
            VariableManager.update();
        }
        return success;
    }

    /**
     * Assigns a stored anonymous function to become a named function.
     *
     * @param fname
     * @param anonFuncName The name of the stored anonymous fucntion
     */
    public static boolean assignAnonymousToNamedFunction(String fname, String anonFuncName) {
        Function f = FunctionManager.lookUp(anonFuncName);
        FunctionManager.delete(anonFuncName);
        if (f.isMatrix()) {
            f.getMatrix().setName(fname);
            FunctionManager.add(f);
            return true;
        } else {
            Variable v = VariableManager.lookUp(fname);
            if (v == null) {
                v = new Variable(fname);
            }
            f.setDependentVariable(v);
            FunctionManager.add(f);
            return true;
        }
    }

    /**
     *
     * @param input The user input into the system, usually of the form:
     * F=@(x,y,z,w)mathexpr;..where mathexpr is an algebraic expression in terms
     * of x,y,z,...
     *
     */
    private void parseInput(String input) {
        int equalsIndex = input.indexOf("=");
        int atIndex = input.indexOf("@");
        if (atIndex == -1) {
            String err = "Function Syntax error! " + input;
            errorLog.info(err);
            throw new InputMismatchException();
        }
        String funcName = equalsIndex == -1 ? null : input.substring(0, equalsIndex);//may be null for a anonymous function input

        Function anonFn = null;

        int firstIndexOfClose = input.indexOf(")");
        int indexOfFirstOpenBrac = input.indexOf("(");
        String vars = input.substring(indexOfFirstOpenBrac + 1, firstIndexOfClose);//GETS x,y,z,w...,t print of @(x,y,z,w...,t)expr
        String expr = input.substring(Bracket.getComplementIndex(true, indexOfFirstOpenBrac, input) + 1).trim();
        List<String> varList = new Scanner(vars, false, ",").scan();
        ArrayList<Variable> indVars = new ArrayList<>(varList.size());

        int numCount = 0;
        int varCount = 0;
        for (int i = 0; i < varList.size(); i++) {
            try {
                String tkn = varList.get(i);
                if (Variable.isVariableString(tkn)) {
                    varCount++;
                    Variable searchVal = VariableManager.lookUp(varList.get(i));
                    Variable v = searchVal != null ? searchVal : new Variable(varList.get(i), 0.0, false);
                    indVars.add(v);
                    vars = vars.concat(varList.get(i) + "=" + v.getValue() + ";");//build variable command list
                }//end if
                else if (Number.isNumber(tkn)) {
                    numCount++;
                }
            }//end try
            catch (IndexOutOfBoundsException boundsException) {
                break;
            }//end catch
        }//end for
        if (numCount > 0 && varCount > 0) {//mixed args, not acceptable, either number args for matrices and vectors or variable args for math expression
            String err = "Bad args for function! Matrix definition must have args that are "
                    + "purely numbers and must be 2 in number. Variable definition must have args that are purely variable names.";
            errorLog.info(err);
            throw new RuntimeException(err);
        }
        boolean isMathExpr = varCount == varList.size();
        boolean isMatrix = numCount == varList.size() && numCount == 2;
        boolean isVector = numCount == varList.size() && numCount == 1;

        //Remove bogus enclosing brackets on an expression e.g(((x+2)))
        while (expr.startsWith("(") && expr.endsWith(")") && Bracket.getComplementIndex(true, 0, expr) == expr.length() - 1) {
            expr = expr.substring(1, expr.length() - 1).trim();
        }

        if (isMathExpr) {
            anonFn = FunctionManager.lockDownAnon(varList.toArray(new String[0]));
            String dependentVar = anonFn.getName();
            //input = dependentVar + "="+input;
            anonFn.setDependentVariable(new Variable(dependentVar));
            anonFn.setIndependentVariables(indVars);
            anonFn.type = TYPE.ALGEBRAIC_EXPRESSION;
            MathExpression me = new MathExpression(expr);

            anonFn.setMathExpression(me);
            //FunctionManager.update(anonFn);
        } else if (isMatrix) {
            int rows = Integer.parseInt(varList.get(0));
            int cols = Integer.parseInt(varList.get(1));
            List<String> entries = new Scanner(expr, false, "(", ")", ",", ";").scan();
            int sz = entries.size();

            if (rows * cols != sz) {
                String err = "Invalid matrix! rows x cols must be equal to items supplied in matrix list. Expected: " + (rows * cols) + ", Found: " + sz + " items";
                errorLog.info(err);
                throw new RuntimeException(err);
            }
            double[] flatArray = new double[sz];
            try {
                for (int i = 0; i < sz; i++) {
                    flatArray[i] = Double.parseDouble(entries.get(i));
                }
            } catch (Exception e) {
                String err = "Elements of a matrix must be numbers!";
                errorLog.info(err);
                throw new RuntimeException(err);
            }
            Matrix m = new Matrix(flatArray, rows, cols);
            anonFn = FunctionManager.lockDownAnon(varList.toArray(new String[0]));
            String dependentVar = anonFn.getName();
            anonFn.setDependentVariable(new Variable(dependentVar));
            m.setName(dependentVar);
            anonFn.matrix = m;
            anonFn.type = TYPE.MATRIX;
            //FunctionManager.update(anonFn); 
        } else if (isVector) {
            int rows = Integer.parseInt(varList.get(0));
            List<String> entries = new Scanner(expr, false, "(", ")", ",").scan();
            int sz = entries.size();

            if (rows != sz) {
                String err = "Invalid matrix! rows x cols must be equal to items supplied in matrix list. Expected: " + (rows) + ", Found: " + sz + " items";
                errorLog.info(err);
                throw new RuntimeException(err);
            }
            double[] flatArray = new double[sz];
            try {
                for (int i = 0; i < sz; i++) {
                    flatArray[i] = Double.parseDouble(entries.get(i));
                }
            } catch (Exception e) {
                     String err = "Elements of a vector must be numbers!";
                errorLog.info(err);
                throw new RuntimeException(err);
            }
            Matrix m = new Matrix(flatArray, 1, sz);
            anonFn = FunctionManager.lockDownAnon(varList.toArray(new String[0]));
            m.setName(anonFn.getName());
            String dependentVar = anonFn.getName();
            anonFn.setDependentVariable(new Variable(dependentVar));
            anonFn.matrix = m;
            anonFn.type = TYPE.VECTOR;
            // FunctionManager.update(anonFn);

        } else {
       String err = "SYNTAX ERROR IN FUNCTION";
                errorLog.info(err);
            throw new InputMismatchException(err);
        }

        //DONE PROCESSIING anon function side of F=@(args)expr
        //Now deal with normal function assignments e.g F=@(x,y,z,...)expr, Use a recursive hack!
        this.dependentVariable = anonFn.isMatrix() ? anonFn.dependentVariable : (funcName != null ? new Variable(funcName) : (anonFn.dependentVariable != null ? anonFn.dependentVariable : null));
        this.independentVariables = anonFn.independentVariables;
        this.mathExpression = anonFn.mathExpression;
        this.turboExpr = anonFn.turboExpr;
        this.matrix = anonFn.matrix;
        if (this.matrix != null && funcName != null) {
            this.matrix.setName(funcName);
        }
        this.type = anonFn.type;
        if (funcName != null) {
            FunctionManager.update(anonFn.getName(), funcName);
        }

    }//end method

    public void setTurboExpr(FastCompositeExpression turboExpr) {
        this.turboExpr = turboExpr;
    }

    public FastCompositeExpression getTurboExpr() {
        return turboExpr;
    }

    public void setDependentVariable(Variable dependentVariable) {
        this.dependentVariable = dependentVariable;
    }

    public Variable getDependentVariable() {
        return dependentVariable;
    }

    public void setMathExpression(MathExpression mathExpression) {
        try {
            this.mathExpression = mathExpression;
            this.type = TYPE.ALGEBRAIC_EXPRESSION;
            this.turboExpr = mathExpression.compileTurbo();
        } catch (Throwable ex) {
            this.turboExpr = null;
        }
    }

    public FastCompositeExpression getWideningArgsExpression() {
        try {
            this.turboExpr = TurboEvaluatorFactory.getCompiler(mathExpression, true).compile();
        } catch (Throwable ex) {
            ex.printStackTrace();
        }
        return this.turboExpr;
    }

    public FastCompositeExpression getArrayArgsExpression() {
        try {
            this.turboExpr = TurboEvaluatorFactory.getCompiler(mathExpression, false).compile();
        } catch (Throwable ex) {
            ex.printStackTrace();
        }
        return this.turboExpr;
    }

    public void setMatrix(Matrix m) {
        this.matrix = m;
        this.matrix.setName(this.getName());
        this.type = TYPE.MATRIX;
    }

    public MathExpression getMathExpression() {
        return mathExpression;
    }

    public void setIndependentVariables(ArrayList<Variable> independentVariables) {
        this.independentVariables = independentVariables;
    }

    public ArrayList<Variable> getIndependentVariables() {
        return independentVariables;
    }

    /**
     *
     * @return the number of independent variables possessed by this Function
     * object.
     */
    public int numberOfParameters() {
        if (type == TYPE.VECTOR) {
            return 1;
        }
        if (type == TYPE.MATRIX) {
            return 2;
        }

        return independentVariables.size();
    }

    public int getArity() {
        return this.independentVariables.size();
    }

    /**
     *
     * @param list A string containing info. about the arguments to be passed to
     * the Function. The format is (num1,num2,num3,....) The information should
     * be numbers only
     * @return true if its format is valid.
     */
    private static boolean isDimensionsList(String list) {
        list = STRING.purifier(list);
        Scanner cs = new Scanner(list, true, "(", ",", ")");
        List<String> scan = cs.scan();

        if (scan.get(0).equals("(") && scan.get(scan.size() - 1).equals(")")) {
            scan.remove(0);
            scan.remove(scan.size() - 1);
        } else {
            return false;
        }

        int sz = scan.size();
        for (int i = 0; i < sz; i++) {
            try {
                if (!Number.validNumber(scan.get(i)) && !Operator.isComma(scan.get(i))) {
                    return false;
                }//end if
                if ((i == 0 || i == scan.size() - 1)) {
                    if (!Number.validNumber(scan.get(i))) {
                        return false;
                    }
                }//end if
                else {
                    if (Number.validNumber(scan.get(i)) && !Operator.isComma(scan.get(i + 1))) {
                        return false;
                    }//end if
                    if (Operator.isComma(scan.get(i)) && !Number.validNumber(scan.get(i + 1))) {
                        return false;
                    }//end if
                }
            }//end try
            catch (IndexOutOfBoundsException ioobe) {

            }//end catch
        }//end for

        return true;
    }//end method

    /**
     *
     * @param list A string containing info. about the arguments to be passed to
     * the Function. The format is (var1,var2,var3,....)
     * @return true if its format is valid.
     */
    private static boolean isParameterList(String list) {
        list = STRING.purifier(list);
        Scanner cs = new Scanner(list, true, "(", ",", ")");
        List<String> scan = cs.scan();

        if (scan.get(0).equals("(") && scan.get(scan.size() - 1).equals(")")) {
            scan.remove(0);
            scan.remove(scan.size() - 1);
        } else {
            return false;
        }

        int sz = scan.size();
        if (sz == 0) {
            return false;
        } else if (sz == 1) {
            if (Variable.isVariableString(scan.get(0))) {
                return true;
            }
        }

        for (int i = 0; i < sz; i++) {
            try {
                //Only allowd tokens are variables and commas
                if (!Variable.isVariableString(scan.get(i)) && !Operator.isComma(scan.get(i))) {
                    return false;
                }//end if
                //The first and the last tokens must be variables
                if (i == 0 || i == sz - 1) {
                    if (!Variable.isVariableString(scan.get(i))) {
                        return false;
                    }
                }//end if
                else {
                    if (Variable.isVariableString(scan.get(i)) && !Operator.isComma(scan.get(i + 1))) {
                        return false;
                    }//end if
                    if (Operator.isComma(scan.get(i)) && !Variable.isVariableString(scan.get(i + 1))) {
                        return false;
                    }//end if
                }
            }//end try
            catch (IndexOutOfBoundsException ioobe) {

            }//end catch
        }//end for

        return true;
    }//end method

    public Matrix getMatrix() {
        return matrix;
    }

    /**
     *
     * @param name The name of the variable.
     * @return the Variable if it exists in the parameter list of this Function.
     * Returns null if no Variable having that name is to be found in the
     * parameter list of this Function.
     *
     */
    public Variable getIndependentVariable(String name) {
        if (type != TYPE.ALGEBRAIC_EXPRESSION) {
            return null;
        }
        int sz = independentVariables.size();
        for (int i = 0; i < sz; i++) {
            if (independentVariables.get(i).getName().equalsIgnoreCase(name)) {
                return independentVariables.get(i);
            }//end if
        }//end for loop
        return null;
    }//end method

    /**
     *
     * @param var The name of the Variable to check.
     * @return true if this object has an independent variable that goes by the
     * given name.
     */
    public boolean hasIndependentVariable(String var) {
        if (type != TYPE.ALGEBRAIC_EXPRESSION) {
            return false;
        }
        Variable v = new Variable(var);
        return independentVariables.contains(v);
    }//end method

    /**
     *
     * @param paramList A string containing info. about the arguments to be
     * passed to the Function. The format is (var1,var2,var3,....)
     * @return an array containing the parameters, if its format is valid.
     */
    public static String[] getParameters(String paramList) {
        paramList = STRING.purifier(paramList);
        Scanner cs = new Scanner(paramList, false, "(", ",", ")");
        List<String> scan = cs.scan();

        String str[] = new String[scan.size()];
        if (isParameterList(paramList)) {
            return scan.toArray(str);
        } else {
            throw new InputMismatchException("Syntax Error In Parameter List " + paramList);
        }
    }

    /**
     *
     * @param name The name of the Variable object to check, whether or not it
     * is a parameter of this Function object.
     * @return true if a Variable of the specified name exists in the parameter
     * list of this Function object.
     */
    public boolean isParam(String name) {
        return getIndependentVariable(name) != null;
    }

    /**
     *
     * @param matrix The {@link Matrix} object to be wrapped in a function
     * @return the name assigned to the anonymous function created.
     */
    public static synchronized String storeAnonymousMatrixFunction(Matrix matrix) {
        Function f = FunctionManager.lockDownAnon();
        matrix.setName(f.getName());
        f.setType(TYPE.MATRIX);
        f.setMatrix(matrix);
        FunctionManager.update(f);
        return f.getName();
    }

    /**
     *
     * @param expression The expression used to create the function...e.g
     * @(x)sin(x-1)^cos(x)
     * @return the name assigned to the anonymous function created.
     */
    public static synchronized Function storeAnonymousFunction(String expression) {
        Function f = FunctionManager.lockDownAnon();
        f.setType(TYPE.ALGEBRAIC_EXPRESSION);
        MathExpression me = new MathExpression(expression);
        f.setMathExpression(me);
        f.dependentVariable.setName(f.getName());
        String names[] = me.getVariablesNames();
        for (String n : names) {
            Variable v = VariableManager.lookUp(n);
            if (v != null) {
                f.independentVariables.add(v);
            } else {
                f.independentVariables.add(new Variable(n));
            }
        }
        FunctionManager.update(f);
        return f;
    }

    public boolean isMatrix() {
        return this.type == TYPE.MATRIX && matrix != null;
    }

    /**
     * @param args
     * @deprecated Deprecated this in favor of
     * {@link Function#evalArgs(double...)} which is far faster due to direct
     * passing of args
     * @return the value of a function when valid arguments are passed into its
     * parentheses. e.g if the fullname of the Function is f(x,y,c), this method
     * could be passed..f(3,-4,9)
     */
    public String evalArgs(String args) {
        if (type != TYPE.ALGEBRAIC_EXPRESSION) {
            return null;
        }
        String str = getFullName();
//StringBuilder;
        Scanner cs = new Scanner(args, false, "(", ")", ",");
        List<String> l = cs.scan();

        if (l.get(0).equals(dependentVariable.getName())) {
            l.remove(0);
            int sz = l.size();
            int sz1 = independentVariables.size();
            if (sz == sz1) {

                String vars = "";
                for (int i = 0; i < sz; i++) {
                    String token = l.get(i);
                    if (!Number.validNumber(token) && !Variable.isVariableString(token)) {
                        String err = "Unrecognized Value or Variable: " + l.get(i);
                        errorLog.info(err);
                        throw new NumberFormatException(err);
                    }//end if
                    else {
                        String v = independentVariables.get(i).getName();
                        vars = vars.concat(v + "=" + l.get(i) + ";");//build variable command.
                        mathExpression.setValue(v, Double.parseDouble(l.get(i)));
                    }
                }//end for 
                mathExpression.getVariableManager().parseCommand(vars);
                return mathExpression.solve();
            }//end if
            else {
                 String err = "Invalid Argument List! " + sz1 + " arguments expected!";
                        errorLog.info(err);
                throw new InputMismatchException(err);
            }//end else

        }//end if
        else {
               String err = "Pass Arguments To The Format: " + str;
                        errorLog.info(err);
            throw new InputMismatchException(err);
        }//end else

    }//end method

    public String evalArgs(double... args) {
        if (type != TYPE.ALGEBRAIC_EXPRESSION) {
            return null;
        }

        mathExpression.updateArgs(args);
        return mathExpression.solve();
    }//end method

    public double evalArgsTurbo(double... args) {
        if (type != TYPE.ALGEBRAIC_EXPRESSION) {
            return Double.NaN;
        }
        return turboExpr.applyScalar(args);
    }//end method

    /**
     *
     * @param rangeDescr Describes the range between which this Function object
     * should be plotted. e.g. x:-10:10:0.0001
     * @return an 2D array containing two 1d arrays. The first array
     * containsAlgebraicFunction the values that the Function object will have
     * for all values specified for the range of the independent variable. The
     * second array containsAlgebraicFunction the values that the independent
     * variable will assume in its given range. If the rangeDescr parameter is
     * not valid.. it returns a 2D array containing 2 null arrays.
     */
    public double[][] evalRange(String rangeDescr) {

//x:0:10:xStep
        String variableName = rangeDescr.substring(0, rangeDescr.indexOf(":"));
        rangeDescr = rangeDescr.substring(rangeDescr.indexOf(":") + 1);

        if (isParam(variableName)) {
            String xStart = rangeDescr.substring(0, rangeDescr.indexOf(":"));
            rangeDescr = rangeDescr.substring(rangeDescr.indexOf(":") + 1);
            String xEnd = rangeDescr.substring(0, rangeDescr.indexOf(":"));
            rangeDescr = rangeDescr.substring(rangeDescr.indexOf(":") + 1);
            double xStep = Double.parseDouble(rangeDescr);

            double x1 = Double.parseDouble(xStart);
            double x2 = Double.parseDouble(xEnd);
            int sz = (int) ((x2 - x1) / xStep);

            double[][] results = new double[2][sz + 1];

            if (x1 > x2) {
                double p = x1;
                x1 = x2;
                x2 = p;
            }
            int i = 0;
            int len = sz + 1;
            int xSlot = mathExpression.getVariable(variableName).getFrameIndex();
            for (double x = x1; i < len && x <= x2; x += xStep, i++) {
                mathExpression.updateSlot(xSlot, x);
                results[0][i] = mathExpression.solveGeneric().scalar;
                results[1][i] = x;
            }//end for
            return results;
        }//end if

        return new double[][]{null, null};
    }//end method

    /**
     *
     * @param xLower The lower limit from which plotting begins.
     * @param xUpper The upper limit from which plotting begins.
     * @param xStep The plot step.
     * @param variableName The name of the independent(the horizontal axis
     * variable..usually x)
     * @param DRG States whether the function should be evaluated in Degrees,
     * Radians, and Grad.
     * @return an 2D array containing two 1d arrays. The first array
     * containsAlgebraicFunction the values that the Function object will have
     * for all values specified for the range of the independent variable. The
     * second array containsAlgebraicFunction the values that the independent
     * variable will assume in its given range. If the rangeDescr parameter is
     * not valid.. it returns a2D array containing 2 null arrays.
     */
    public double[][] evalRange(double xLower, double xUpper, double xStep, String variableName, int DRG) {

        if (xLower > xUpper) {
            double p = xLower;
            xLower = xUpper;
            xUpper = p;
        }
        int sz = (int) ((xUpper - xLower) / xStep);

        double[][] results = new double[2][sz + 1];
        int len = sz + 1;
        int i = 0;
        int xSlot = mathExpression.getVariable(variableName).getFrameIndex();
        for (double x = xLower; i < len && x <= xUpper; x += xStep, i++) {
            mathExpression.updateSlot(xSlot, x);
            results[0][i] = mathExpression.solveGeneric().scalar;
            results[1][i] = x;
        }//end for

        return results;

    }//end method

    public double[][] evalRange(double xLower, double xUpper, double xStep, String variableName, int DRG, boolean turbo) {
        if (!turbo) {
            return evalRange(xLower, xUpper, xStep, variableName, DRG);
        }
        if (xLower > xUpper) {
            double p = xLower;
            xLower = xUpper;
            xUpper = p;
        }
        int sz = (int) ((xUpper - xLower) / xStep);

        double[][] results = new double[2][sz + 1];
        int len = sz + 1;
        int i = 0;
        double[] args = new double[1];
        for (double x = xLower; i < len && x <= xUpper; x += xStep, i++) {
            args[0] = x;
            results[0][i] = turboExpr.applyScalar(args);
            results[1][i] = x;
        }//end for

        return results;

    }//end method

    /**
     * Prints the content of a 2D array
     *
     * @param obj
     */
    public static void print2DArray(Object[][] obj) {
        int rows = obj.length;
        for (int j = 0; j < rows; j++) {
            Object[] values = obj[j];
            for (int i = 0; i < values.length; i++) {
                System.out.println(values[i]);
            }//end for loop.
        }//end for loop.
    }//end method

    /**
     *
     * @return the value of the function based on assigned values of its
     * variables and constants.
     */
    public String eval() {
        return mathExpression != null ? mathExpression.solve() : null;
    }

    /**
     *
     * @param str The input string to check if or not it is of the format of the
     * full name of a Function. e.g. F(x),p(x,y) e.t.c.
     * @return true if the input string has the format of the full name of a
     * Function. This method will be used to identify when the input into a
     * calculator is to be treated as a Function problem...i.e. identify when
     * the user is trying to create or modify a Function. F(x,y,x1,c,e3,u)
     */
    public static boolean isFunctionFullName(String str) {
        if (str.contains("(") && str.contains(")")) {
            str = STRING.purifier(str);
            Scanner cs = new Scanner(str, false, "(", ")", ",");
            List<String> l = cs.scan();
            if (Variable.isVariableString(l.get(0))) {
                int sz = l.size();
                for (int i = 1; i < sz; i++) {
                    if (!Variable.isVariableString(l.get(i))) {
                        return false;
                    }//end false
                }//end for loop
                return true;
            }//end if
            else {
                return false;
            }//end else
        }//end if
        else {
            return false;
        }//end else
    }//end method

    /**
     *
     * @return the standard math form of this Function object.
     */
    @Override
    public String toString() {

        String fullname = getFullName();
        String paramList = fullname.substring(fullname.indexOf("("));
        switch (type) {
            case ALGEBRAIC_EXPRESSION:
                return getName() + "=@" + paramList + mathExpression.getExpression();
            case MATRIX:
                return getName() + "=@" + paramList + "(" + matrixToCommaList(matrix) + ")";
            case VECTOR:
                return getName() + "=@" + paramList + "(" + matrixToCommaList(matrix) + ")";
            default:
                return "";
        }

    }//end method

    public boolean isAnonymous() {
        return isAnonymous(this);
    }

    public static boolean isAnonymous(Function f) {
        switch (f.type) {
            case ALGEBRAIC_EXPRESSION:
                return f.dependentVariable.getName().startsWith(FunctionManager.ANON_PREFIX);
            case MATRIX:
                return f.matrix.getName().startsWith(FunctionManager.ANON_PREFIX);
            case VECTOR:
                return f.matrix.getName().startsWith(FunctionManager.ANON_PREFIX);
            default:
                return false;
        }

    }

    public static boolean isAnonymous(String name) {
        return name.startsWith(FunctionManager.ANON_PREFIX);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Function) {
            Function f = (Function) obj;
            return toString().equals(f.toString());
        }
        return false;
    }

    /**
     * @return the dependent variable together with its independent variables
     * within its circular parentheses. e.g if the Function is y=x^2, this
     * method will return y(x)
     */
    public String getFullName() {
        switch (type) {
            case ALGEBRAIC_EXPRESSION:
                if (dependentVariable == null) {
                    return null;
                }
                String str = dependentVariable.getName() + "(";
                int sz = independentVariables.size();
                for (int i = 0; i < sz; i++) {
                    str += (independentVariables.get(i).getName() + ",");
                }//end for
                str = str.substring(0, str.length() - 1);
                return str + ")";
            case MATRIX:
                return getName() + "(" + matrix.getRows() + "," + matrix.getCols() + ")";
            case VECTOR:
                return getName() + "(" + matrix.getRows() + "," + matrix.getCols() + ")";
            default:
                return "";
        }
    }

    /**
     *
     * @return the simple name of the function. e.g if the function is
     * y=@(x)cos(x), then this method returns 'y'.
     */
    public String getName() {
        switch (type) {
            case ALGEBRAIC_EXPRESSION:
                return dependentVariable == null ? null : dependentVariable.getName();
            case MATRIX:
            case VECTOR:
                return matrix.getName();

            default:
                return "";
        }
    }

    /**
     *
     * @return the determinant of the function if it is of type
     * {@link Function#MATRIX} Otherwise it returns {@link Double#NaN}
     */
    public double calcDet() {
        if (type == TYPE.MATRIX) {
            return matrix.determinant();
        }
        return Double.NaN;
    }

    /**
     *
     * @return the inverse of the matrix-function if it is of type
     * {@link Function#MATRIX} Otherwise it returns null
     */
    public Matrix calcInverse() {
        if (type == TYPE.MATRIX) {
            return matrix.inverse();
        }
        return null;
    }

    /**
     *
     * @return the triangular Matrix of the function if it is of type
     * {@link Function#MATRIX} Otherwise it returns null
     */
    public Matrix triangularMatrix() {
        if (type == TYPE.MATRIX) {
            return matrix.reduceToTriangularMatrix();
        }
        return null;
    }

    /**
     *
     * @return the row-reduced echelon matrix of the function if it is of type
     * {@link Function#MATRIX} Otherwise it returns null
     */
    public Matrix reduceToEchelon() {
        if (type == TYPE.MATRIX) {
            return matrix.reduceToRowEchelonMatrix();
        }
        return null;
    }

    /**
     *
     * @return the sub-expression on the right hand side of the assignment '='
     * sign in the expression that created this Function object.
     */
    public String expressionForm() {
        String toString = toString();
        return toString.substring(toString.indexOf("=") + 1);
    }

    /**
     * @param data The input list The input list is such that: The first 2
     * entries specify the number of rows and columns. The remaining entries are
     * the Matrix's entries.
     * @return a number list which represents the row-reduced echelon Matrix
     * that the entries reduce to.
     */
    public static Matrix listToMatrix(List<String> data) {

        int numRows = Integer.parseInt(data.get(0));
        int numCols = Integer.parseInt(data.get(1));

        data.subList(0, 2).clear();
        int size = data.size();

        if (numRows * numCols == size) {

            double arr[][] = new double[numRows][numCols];
            int row = 0;
            int col = 0;
            for (int i = 0; i < size; i++) {
                arr[row][col] = Double.parseDouble(data.get(i));
                if (col == numCols - 1) {
                    col = 0;
                    row++;
                } else {
                    col++;
                }
            }
            return new Matrix(arr);

        }
        return null;
    }

    /**
     * The input list is such that: The first 2 entries specify the number of
     * rows and columns. The remaining entries are the Matrix's entries.
     *
     * @return a list containing the data of the Matrix. This list does not
     * contain the dimensions of the Matrix, only the data.
     */
    public static List<String> matrixToList(Matrix mat) {

        int numRows = mat.getRows();
        int numCols = mat.getCols();

        List<String> data = new ArrayList<>();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                data.add(mat.getElem(i, j) + "");
            }
        }

        return data;

    }

    /**
     * @param mat The {@link Matrix} object
     * @return a comma separated string containing the data of the Matrix. This
     * string does not contain the dimensions of the Matrix, only the data.
     */
    public static String matrixToCommaList(Matrix mat) {

        StringBuilder str = new StringBuilder();
        double[] flatArr = mat.getFlatArray();

        for (int j = 0; j < flatArr.length; j++) {
            str.append(flatArr[j]).append(",");
        }

        return str.substring(0, str.length() - 1);
    }

    /**
     * Creates a deep copy of this Function instance. Essential for thread-safe
     * parallel integration.
     *
     * Each thread gets an independent Function with: - Same expression and
     * parsed structure - Independent state variables - No shared mutable
     * references
     *
     * @return A new Function instance safe for concurrent use
     */
    public Function copy() {
        try {
            // Create new instance from expression
            Function copy = new Function();
            if (independentVariables != null) {
                copy.independentVariables = new ArrayList<>(independentVariables);
            }
            if (dependentVariable != null) {
                copy.dependentVariable = new Variable(dependentVariable.getName(), dependentVariable.getValue());
            }
            if (mathExpression != null) {
                copy.mathExpression = mathExpression.copy();
                try {
                    copy.turboExpr = copy.mathExpression.getCompiledTurbo();
                } catch (Throwable ex) {
                    Logger.getLogger(Function.class.getName()).log(Level.SEVERE, null, ex);
                }
            }

            copy.matrix = matrix == null ? null : new Matrix(matrix.getFlatArray(), matrix.getRows(), matrix.getCols());
            if (copy.matrix != null) {
                copy.matrix.setName(matrix.getName());
            }
            copy.type = this.type;

            return copy;

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to copy Function: " + e.getMessage(), e);
        }
    }

    public static Function parse(String enc) {
        return (Function) Serializer.deserialize(enc);
    }

    public static void main(String args[]) {

        Function f = new Function("y=@(x)sin(x)-cos(x)");
        double s = System.nanoTime();
        f.evalRange(0, 10000, 0.001, "x", 1);
        double d = System.nanoTime() - s;
        System.out.println("t1 = " + d + "ns");

        s = System.nanoTime();
        f.evalRange(0, 10000, 0.001, "x", 1, true);
        d = System.nanoTime() - s;
        System.out.println("t2 = " + d + "ns");

        System.out.println(Function.rewriteAsStandardFunction("f(x)=sin(x)-cos(x)",f.errorLog));
        System.out.println(Function.rewriteAsStandardFunction("f(x,y,z)=sin(x+y)-cos(z-2*x)",f.errorLog));
        System.out.println(Function.rewriteAsStandardFunction("f=@(x)sin(x)-cos(x)",f.errorLog));
        System.out.println(Function.rewriteAsStandardFunction("f=@(x)sin(x)-cos(x)",f.errorLog));

        FunctionManager.add("K=@(3,3)(2,3,4,9,8,1,9,8,1);");

        System.out.println("K=" + FunctionManager.lookUp("K").getMatrix());
        MathExpression addMat = new MathExpression("w=6*5;K=@(2,3)(2,3,4,9,8,1);M=@(3,3)(1,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);v=eigpoly(@(3,3)(2,1,5,6,9,2,4,3,8));c=v(30);3*M+N;");

        System.out.println("soln: " + addMat.solve());

        System.out.println(FunctionManager.FUNCTIONS);

        Function func = new Function("p=@(x)sin(x)+x+x^2");
        FunctionManager.add(func);

        func.updateArgs(4);
        System.out.println(func.calc());

        int count = 100000;

        double start = System.nanoTime();
        String val = null;
        for (int i = 1; i <= count; i++) {
            val = func.evalArgs(i);
        }
        double duration = System.nanoTime() - start;
        System.out.println("Std-Eval took: " + duration / count + "ns");
        System.out.println("val1 = " + val);

        start = System.nanoTime();
        for (int i = 1; i <= count; i++) {
            val = func.evalArgs("p(" + i + ")");
        }
        System.out.println("val2 = " + val);

        duration = System.nanoTime() - start;
        System.out.println("Old-Eval took: " + (duration / (count * 1.0E6)) + "ms");

        double v = 0;
        start = System.nanoTime();
        for (int i = 1; i <= count; i++) {
            v = func.evalArgsTurbo(i);
        }
        System.out.println("val3 = " + v);

        duration = System.nanoTime() - start;
        System.out.println("Turbo-Eval took: " + duration / count + "ns");

    }//end method

    @Override
    public String serialize() {
        return Serializer.serialize(this);
    }

}//end class
