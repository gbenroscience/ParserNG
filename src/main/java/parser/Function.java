/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser;

import interfaces.Savable;
import parser.methods.Method;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import math.matrix.expressParser.Matrix;
import util.FunctionManager;
import util.Serializer;
import util.VariableManager;

/**
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class Function implements Savable {

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
    private int type = ALGEBRAIC;

    public static final int ALGEBRAIC = 1;
    public static final int MATRIX = 2;
    public static final int LIST = 3;

    /**
     * If the object is an algebraic expression, its details are stored here.
     * The math expression on the RHS.
     */
    private MathExpression mathExpression;
    /**
     * If the object is a {@link Matrix} its data is stored here.
     */
    private Matrix matrix;

    /**
     *
     * @param matrix A Matrix to be used to initialize the function..
     */
    public Function(Matrix matrix) {
        this.matrix = matrix;
        this.type = MATRIX;
        FunctionManager.add(this);
    }

    /**
     *
     * @param input The user input into the system, usually of the form:
     * F(x,y,z,w,....)=mathexpr; or F= @(x,y,z,w,...)mathexpr ...where mathexpr
     * is an algebraic expression in terms of x,y,z,w,...
     *
     */
    public Function(String input) throws InputMismatchException {
        try {
            input = STRING.purifier(input);

            int openIndex = input.indexOf("(");
            int equalsIndex = input.indexOf("=");
            int atIndex = input.indexOf("@");

            if (equalsIndex == -1) {
                boolean anonymous = input.startsWith("@");
                if (anonymous) {
                    parseInput(input);
                    return;
                }
                throw new InputMismatchException("Bad function syntax!");
            }

            /**
             * F=@(x,y,z,w,...)mathexpr OR F(x,y,z,w,...)=mathexpr
             */
            String tokenAfterEquals = input.substring(equalsIndex + 1, equalsIndex + 2);
            if (atIndex != -1 && atIndex < openIndex) {
                //The enclosing if assumes that the user is creating a function using the anonymous function assignment format.
                if (atIndex != openIndex - 1) {
                    throw new InputMismatchException("Error in function format... anonymous function assignment format must have the `@` preceding the `(`");
                    //error...token between at symbol and param list
                } else if (!tokenAfterEquals.equals("@")) {
                    //Avoid this nonsense: f=kdkdk@(x,...)expr
                    throw new InputMismatchException("Error in function format... anonymous function assignment format must have the `=` preceding the `@`");
                    //cool... function created with anonymous function assignment
                }
            }

            if (openIndex == -1 || equalsIndex == -1) {
                throw new InputMismatchException("Bad function format!");
            }
            int close = Bracket.getComplementIndex(true, openIndex, input);
            String name = null;
            /**
             * If function is in this format: //f(...) = expr Force it to be in
             * this format: //f=@(...)expr
             */
            if (openIndex < equalsIndex) {
                name = input.substring(0, openIndex);
                input = name + "=@" + input.substring(openIndex, close + 1) + input.substring(equalsIndex + 1);
            }

            openIndex = input.indexOf("(");
            equalsIndex = input.indexOf("=");
            close = Bracket.getComplementIndex(true, openIndex, input);

            if (name == null) {
                name = input.substring(0, equalsIndex);
            }

            if (!Variable.isVariableString(name)) {
                throw new InputMismatchException("Bad name for Function.");
            }

            String paramsList = input.substring(openIndex + 1, close);
            List<String> params = new CustomScanner(paramsList, false, ",").scan();

            int size = params.size();
            boolean notAlgebraic = true;
            /**
             * This loop should check if all arguments in the params list are
             * numbers... This is necessary for the input to be a Matrix
             * function or a List
             */
            for (String p : params) {
                try {
                    Double.parseDouble(p);
                } catch (Exception e) {
                    notAlgebraic = false;//algebraic argument found....exit
                    break;
                }
            }//end for loop
        
            if (notAlgebraic) {
                if (size == 1) {   
                    int listSize = Integer.parseInt(params.get(0));
                    type = LIST;
                } else if (size == 2) {  
                    //A matrix definition...A(2,3)=(3,2,4,5,3,1)------A=@(3,3)(3,4,32,3,4,4,3,3,4)
                    int rows = Integer.parseInt(params.get(0));
                    int cols = Integer.parseInt(params.get(1));
                    int indexOfLastCloseBrac = input.lastIndexOf(")");
                    int compIndexOfLastCloseBrac = Bracket.getComplementIndex(false, indexOfLastCloseBrac, input);
                    String list = input.substring(compIndexOfLastCloseBrac, indexOfLastCloseBrac + 1);
                    if (!list.startsWith("(") || !list.endsWith(")")) {
                        throw new InputMismatchException("Invalid Matrix Format...Circular Parentheses missing");
                    }
                    list = list.substring(1, list.length() - 1);

                    List<String> matrixData = new CustomScanner(list, false, ",").scan();
                    if (rows * cols == matrixData.size()) {
                        matrixData.add(0, cols + "");
                        matrixData.add(0, rows + "");
  
                        //Validate the entries
                        for (int i = 0; i < matrixData.size(); i++) {
                            try {
                                Double.parseDouble(matrixData.get(i));
                            } catch (Exception e) {
                                throw new InputMismatchException("Invalid Matrix Data..." + matrixData.get(i) + " found.");
                            }
                        }
                        this.matrix = listToMatrix(matrixData);
                        type = MATRIX;
                        this.matrix.setName(name);

                    } else {
                        throw new InputMismatchException("Invalid number of entries found in Matrix Data---for input: " + input);
                    }

                }//end else if params-list  size == 2

            }//if not an algebraic function
            else {//input is algebraic like this..f(a,b..)=expr
                parseInput(input);
            }

        } catch (Exception e) {
            e.printStackTrace();
            throw new InputMismatchException("Bad Function Syntax--" + input);
        }

    }//end constructor

    public void setType(int type) {
        this.type = type;
    }

    public int getType() {
        return type;
    }

    /**
     *
     * @param x A list of variable values to set for the function. The supplied
     * value list is applied to the function's parameter list in the order they
     * were supplied in the original question.
     * @return the value of the function with these variables set.
     */
    public double calc(double... x) {
        int i = 0;
        if (x.length == independentVariables.size()) {
            for (Variable var : independentVariables) {
                mathExpression.setValue(var.getName(), String.valueOf(x[i++]));
            }

            return Double.parseDouble(mathExpression.solve());
        }
        return Double.NaN;
    }

    public static boolean assignObject(String input) {

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
            throw new InputMismatchException("Wrong Input!");
        }

        /**
         * Check if the user used the form f(x)=.... instead of f=@(x).... If so
         * convert to the latter format, and then recompute the necessary
         * indexes.
         */
        if (indexOfOpenBrac != -1 && indexOfOpenBrac < equalsIndex) {
            input = input.substring(0, indexOfOpenBrac)
                    + "=@" + input.substring(indexOfOpenBrac, Bracket.getComplementIndex(true, indexOfOpenBrac, input) + 1) + input.substring(equalsIndex + 1);
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
                    VariableManager.VARIABLES.put(newFuncName, new Variable(newFuncName, rhs, false));
                } else if (isVarNamesList) {
                    List<String> vars = new CustomScanner(newFuncName, false, ",").scan();
                    for (String var : vars) {
                        VariableManager.VARIABLES.put(var, new Variable(var, rhs, false));
                    }
                }

                success = true;
            } else {
                MathExpression expr = new MathExpression(rhs);
                String val = expr.solve();
                String referenceName = expr.getReturnObjectName();

                if (Variable.isVariableString(newFuncName) || isVarNamesList) {
                    Function f;
                    switch (expr.getReturnType()) {
                        case MATRIX:
                            if (isVarNamesList && hasCommas) {
                                throw new InputMismatchException("Initialize a function at a time!");
                            }
                            f = FunctionManager.lookUp(referenceName);
                            FunctionManager.FUNCTIONS.put(newFuncName, new Function(newFuncName + "=" + f.expressionForm()));
                            success = true;
                            break;
                        case ALGEBRAIC_EXPRESSION:
                            if (isVarNamesList && hasCommas) {
                                throw new InputMismatchException("Initialize a function at a time!");
                            }
                            f = FunctionManager.lookUp(referenceName);
                            FunctionManager.FUNCTIONS.put(newFuncName, new Function(newFuncName + "=" + f.expressionForm()));
                            success = true;
                            break;
                        case LIST:
                            if (isVarNamesList && hasCommas) {
                                throw new InputMismatchException("Initialize a function at a time!");
                            }
                            f = FunctionManager.lookUp(referenceName);
                            FunctionManager.FUNCTIONS.put(newFuncName, new Function(newFuncName + "=" + f.expressionForm()));
                            success = true;
                            break;
                        case STRING://for now, this only confirms that a comma separated list has been returned

                            break;
                        case NUMBER:
                            if (isVarNamesList && hasCommas) {
                                List<String> vars = new CustomScanner(newFuncName, false, ",").scan();
                                for (String var : vars) {
                                    VariableManager.VARIABLES.put(var, new Variable(var, val, false));
                                }
                                success = true;
                            } else {
                                VariableManager.VARIABLES.put(newFuncName, new Variable(newFuncName, val, false));
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
                    throw new InputMismatchException("Syntax Error---" + newFuncName);
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
     *
     * @param input The user input into the system, usually of the form:
     * F=@(x,y,z,w)mathexpr;..where mathexpr is an algebraic expression in terms
     * of x,y,z,...
     *
     */
    private void parseInput(String input) {
    
        input = input.trim();
        if (input.contains("@")) {
  
            boolean anonymous = input.startsWith("@");
            if (anonymous) {
                input = "anon" + (FunctionManager.countAnonymousFunctions() + 1) + "=".concat(input);
            }

            String[] cutUpInput = new String[3];

            cutUpInput[0] = input.substring(0, input.indexOf("=")).trim();//---function-name
            cutUpInput[1] = input.substring(input.indexOf("@") + 1, input.indexOf(")") + 1).trim();//---(x,y,....)..params list
            cutUpInput[2] = input.substring(input.indexOf(")") + 1);//--the expression

            CustomScanner cs = new CustomScanner(cutUpInput[1], false, ",", "(", ")");
            List<String> scan = cs.scan();
       
            if (Variable.isVariableString(cutUpInput[0]) && isParameterList(cutUpInput[1])) {
                if (cutUpInput[0].startsWith("anon") && !anonymous) {
                    throw new InputMismatchException("Function Name Cannot Start With \'anon\'.\n \'anon\' is a reserved name for anonymous functions.");
                } else if (Method.isInBuiltMethod(cutUpInput[0])) {
                    throw new InputMismatchException(cutUpInput[0] + " is a reserved name for inbuilt methods.");
                } else {
                    setDependentVariable(new Variable(cutUpInput[0]));
                }
                String vars = "";
                for (int i = 0; i < scan.size(); i++) {
                    try {
                        if (Variable.isVariableString(scan.get(i))) {
                            independentVariables.add(new Variable(scan.get(i), "0.0", false));
                            vars = vars.concat(scan.get(i) + "=" + "0.0;");//build variable command list
                        }//end if
                    }//end try
                    catch (IndexOutOfBoundsException boundsException) {
                        break;
                    }//end catch
                }//end for
 
                while (cutUpInput[2].startsWith("(") && cutUpInput[2].endsWith(")") && Bracket.getComplementIndex(true, 0, cutUpInput[2]) == cutUpInput[2].length() - 1) {
                    cutUpInput[2] = cutUpInput[2].substring(1, cutUpInput[2].length() - 1).trim();
                }
 
                setMathExpression(new MathExpression(vars.concat(cutUpInput[2].trim())));
                if (!mathExpression.isCorrectFunction()) {
                    throw new InputMismatchException("SYNTAX ERROR IN FUNCTION");
                }
            }//end if
            else {
                if (isDimensionsList(cutUpInput[1])) {
                    Function f = new Function(input);
                    this.matrix = f.matrix;
                    this.type = f.type;
                    return;
                }
                throw new InputMismatchException("Syntax Error: Format Is: F=@(x,y,z,...)mathexpr");
            }//end else
        }//end if
        else {
            throw new InputMismatchException("Syntax Error: Format Is: F=@(x,y,z,...)mathexpr");
        }//end else
 
    }//end method

    public void setDependentVariable(Variable dependentVariable) {
        this.dependentVariable = dependentVariable;
    }

    public Variable getDependentVariable() {
        return dependentVariable;
    }

    public void setMathExpression(MathExpression mathExpression) {
        this.mathExpression = mathExpression;
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
        if (type == LIST) {
            return 1;
        }
        if (type == MATRIX) {
            return 2;
        }

        return independentVariables.size();
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
        CustomScanner cs = new CustomScanner(list, true, "(", ",", ")");
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
        CustomScanner cs = new CustomScanner(list, true, "(", ",", ")");
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
        if (type != ALGEBRAIC) {
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
        if (type != ALGEBRAIC) {
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
        CustomScanner cs = new CustomScanner(paramList, false, "(", ",", ")");
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
        int num = FunctionManager.countAnonymousFunctions();
        String name = "anon" + (num + 1);

        matrix.setName(name);
        FunctionManager.add(new Function(matrix));
        return name;
    }
    
    
    /**
     *
     * @param expression The expression used to create the function...e.g @(x)sin(x-1)^cos(x)
     * @return the name assigned to the anonymous function created.
     */
    public static synchronized String storeAnonymousFunction(String expression) {
        int num = FunctionManager.countAnonymousFunctions();
        String name = "anon" + (num + 1);
        
        
        String tempName = "temp"+System.nanoTime();
        
        Function f = new Function(tempName+"="+expression);
        f.dependentVariable.setName(name);

      
        FunctionManager.add(f);
        return name;
    }
    
    

    /**
     * @param args
     * @return the value of a function when valid arguments are passed into its
     * parentheses. e.g if the fullname of the Function is f(x,y,c), this method
     * could be passed..f(3,-4,9)
     */
    public String evalArgs(String args) {
        if (type != ALGEBRAIC) {
            return null;
        }
        String str = getFullName();
//StringBuilder;
        CustomScanner cs = new CustomScanner(args, false, "(", ")", ",");
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
                        throw new NumberFormatException("Unrecognized Value or Variable: " + l.get(i));
                    }//end if
                    else {
                        vars = vars.concat(independentVariables.get(i).getName() + "=" + l.get(i) + ";");//build variable command.
                    }
                }//end for
                mathExpression.getVariableManager().parseCommand(vars);
                return mathExpression.solve();
            }//end if
            else {
                throw new InputMismatchException("Invalid Argument List! " + sz1 + " arguments expected!");
            }//end else

        }//end if
        else {
            throw new InputMismatchException("Pass Arguments To The Format: " + str);
        }//end else

    }//end method

    /**
     *
     * @param rangeDescr Describes the range between which this Function object
     * should be plotted. e.g. x:-10:10:0.0001
     * @return an 2D array containing two 1d arrays. The first array contains
     * the values that the Function object will have for all values specified
     * for the range of the independent variable. The second array contains the
     * values that the independent variable will assume in its given range. If
     * the rangeDescr parameter is not valid.. it returns a2D array containing 2
     * null arrays.
     */
    public String[][] evalRange(String rangeDescr) {

//x:0:10:xStep
        String variableName = rangeDescr.substring(0, rangeDescr.indexOf(":"));
        rangeDescr = rangeDescr.substring(rangeDescr.indexOf(":") + 1);

        if (isParam(variableName)) {
            String xStart = rangeDescr.substring(0, rangeDescr.indexOf(":"));
            rangeDescr = rangeDescr.substring(rangeDescr.indexOf(":") + 1);
            String xEnd = rangeDescr.substring(0, rangeDescr.indexOf(":"));
            rangeDescr = rangeDescr.substring(rangeDescr.indexOf(":") + 1);
            double xStep = Double.valueOf(rangeDescr);

            double x1 = Double.valueOf(xStart);
            double x2 = Double.valueOf(xEnd);
            int sz = (int) ((x2 - x1) / xStep);

            String[][] results = new String[2][sz + 1];

            if (x1 > x2) {
                double p = x1;
                x1 = x2;
                x2 = p;
            }
            int i = 0;
            int len = sz + 1;
            for (double x = x1; i < len && x <= x2; x += xStep, i++) {
                String xStr = String.valueOf(x);
                mathExpression.setValue(variableName, xStr);
                results[0][i] = mathExpression.solve();
                results[1][i] = xStr;
            }//end for
            return results;
        }//end if

        return new String[][]{null, null};
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
     * @return an 2D array containing two 1d arrays. The first array contains
     * the values that the Function object will have for all values specified
     * for the range of the independent variable. The second array contains the
     * values that the independent variable will assume in its given range. If
     * the rangeDescr parameter is not valid.. it returns a2D array containing 2
     * null arrays.
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
        for (double x = xLower; i < len && x <= xUpper; x += xStep, i++) {
            mathExpression.setValue(variableName, String.valueOf(x));
            mathExpression.setDRG(DRG);
            results[0][i] = Double.parseDouble(mathExpression.solve());
            results[1][i] = x;
        }//end for

        return results;

    }//end method

    /**
     * Prints the content of a 2D array
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
        mathExpression.setDRG(1);
        return mathExpression.solve();
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
            CustomScanner cs = new CustomScanner(str, false, "(", ")", ",");
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
            case ALGEBRAIC:
                return getName() + "=@" + paramList + mathExpression.getExpression();
            case MATRIX:
                return getName() + "=@" + paramList + "(" + matrixToCommaList(matrix) + ")";
            case LIST:
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
            case ALGEBRAIC:
                return f.dependentVariable.getName().startsWith("anon");
            case MATRIX:
                return f.matrix.getName().startsWith("anon");
            case LIST:
                return f.matrix.getName().startsWith("anon");
            default:
                return false;

        }

    }

    public static boolean isAnonymous(String name) {
        return name.startsWith("anon");
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
            case ALGEBRAIC:

                String str = dependentVariable.getName() + "(";
                int sz = independentVariables.size();
                for (int i = 0; i < sz; i++) {
                    str += (independentVariables.get(i).getName() + ",");
                }//end for
                str = str.substring(0, str.length() - 1);
                return str + ")";
            case MATRIX:
                return getName() + "(" + matrix.getRows() + "," + matrix.getCols() + ")";
            case LIST:
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
            case ALGEBRAIC:
                return dependentVariable.getName();
            case MATRIX:
                return matrix.getName();
            case LIST:
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
        if (type == MATRIX) {
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
        if (type == MATRIX) {
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
        if (type == MATRIX) {
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
        if (type == MATRIX) {
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

        int numRows = mat.getRows();
        int numCols = mat.getCols();
        StringBuilder str = new StringBuilder();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                str.append(mat.getElem(i, j)).append(",");
            }
        }

        return str.substring(0, str.length() - 1);

    }

    public static Function parse(String enc) {
        return (Function) Serializer.deserialize(enc);
    }

    public static void main(String args[]) {
        
       MathExpression addMat = new MathExpression("w=6*5;K=@(2,3)(2,3,4,9,8,1);M=@(3,3)(1,4,1,2,4,7,9,1,-2);N=@(3,3)(4,1,8,2,1,3,5,1,9);v=eigpoly(@(3,3)(2,1,5,6,9,2,4,3,8));c=v(30);3*M+N;");
  
       System.out.println("soln: "+addMat.solve());
        
               System.out.println(FunctionManager.FUNCTIONS);
        
       
        Function func = new Function("p=@(x)sin(x)+x+x^2");
        FunctionManager.add(func);
        System.out.println(func.calc(4));
        
        int count = 10000;
        
        
        double start = System.nanoTime();
        for(int i=1;i<=count;i++){
            String val = func.evalArgs("p("+i+")");
        }
        double duration = System.nanoTime() - start;
        System.out.println("Eval took: "+(duration/(count*1.0E6))+"ms");
        
        
        for(int i=1;i<=count;i++){
            String val = func.evalArgs("p("+i+")");
        }
         duration = System.nanoTime() - start;
        System.out.println("Eval took: "+(duration/(count*1.0E6))+"ms");

    }//end method

    @Override
    public String serialize() {
        return Serializer.serialize(this);
    }

}//end class
