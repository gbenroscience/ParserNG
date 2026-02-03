/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser.methods;

import logic.DRG_MODE;
import parser.CustomScanner;
import parser.Function;
import static parser.STRING.*;
import static parser.Number.*;
import static parser.methods.Declarations.*;
import parser.Set;

import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import math.Maths;
import math.matrix.expressParser.Matrix;
import parser.MathExpression;
import parser.Operator;
import util.FunctionManager;

/**
 * Models the methods that perform calculations in the parser.
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class Method {

    /**
     * The method name. This is a sequence of letters,digits,underscore, and
     * dollar sign characters,which must start either with a letter,a dollar
     * sign or an underscore.It may end with the inverse symbol,also.
     */
    private String name;
    /**
     * The method parameters.
     */
    private String[] parameters;
    /**
     * The trig mode. DEGREES,RADIANS OR GRADS.
     */
    private int DRG = 1;

    /**
     *
     * @param name The method name
     * @param parameters The parameters to the method
     */
    public Method(String name, String... parameters) {
        this.name = isMethodName(name) ? name : "";
        this.parameters = parameters;
        if (this.name.isEmpty()) {
            throw new InputMismatchException("Invalid Method Name!");
        }
    }

    /**
     *
     * @param methodName The name of the method.
     * @return true if the method name represents one that can operate on
     * Function objects or anonymous Functions.
     */
    public static boolean isFunctionOperatingMethod(String methodName) {
        return methodName.equals(DIFFERENTIATION) || methodName.equals(INTEGRATION) || methodName.equals(QUADRATIC) || methodName.equals(GENERAL_ROOT)
                || methodName.equals(TARTAGLIA_ROOTS) || methodName.equals(PLOT) || methodName.equals(MATRIX_MULTIPLY)
                || methodName.equals(MATRIX_DIVIDE) || methodName.equals(MATRIX_ADD) || methodName.equals(MATRIX_SUBTRACT)
                || methodName.equals(MATRIX_POWER) || methodName.equals(MATRIX_EDIT) || methodName.equals(MATRIX_TRANSPOSE)
                || methodName.equals(DETERMINANT) || methodName.equals(MATRIX_ADJOINT) || methodName.equals(MATRIX_COFACTORS)
                || methodName.equals(MATRIX_EIGENPOLY) || methodName.equals(MATRIX_EIGENVEC) || methodName.equals(PRINT);
    }

    /**
     *
     * @param expression Initializes the attributes of objects of this class by
     * parsing the expression parameter. The format of expression is:
     * methodname(args_1,args_2,....args_N)
     */
    public Method(String expression) {
        parseExpression(expression);
    }

    /**
     * @return An array containing the names of all functions defined by the
     * user and inbuilt into the software.
     */
    public static String[] getAllFunctions() {
        int sz = FunctionManager.FUNCTIONS.size();
        String[] userDefined = new String[sz + getInbuiltMethods().length];
        String[] keyset = FunctionManager.FUNCTIONS.keySet().toArray(new String[]{});
        System.arraycopy(keyset, 0, userDefined, 0, keyset.length);
        System.arraycopy(getInbuiltMethods(), 0, userDefined, keyset.length, getInbuiltMethods().length);
        return userDefined;
    }//end method.

    /**
     *
     * @param expression The expression to parse. The format of expression is:
     * methodname(args_1,args_2,....args_N)
     */
    public final void parseExpression(String expression) {

        CustomScanner cs = new CustomScanner(expression, false, "(", ")", ",");
        List<String> l = cs.scan();
        int sz = l.size();
        if (isMethodName(l.get(0))) {
            String params[] = new String[sz - 1];
            params = l.subList(1, sz).toArray(params);
            setParameters(params);
            setName(l.get(0));
        }//end if
        else {
            throw new InputMismatchException("Invalid Method Name!");
        }//end else

    }//end method

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setParameters(String[] parameters) {
        this.parameters = parameters;
    }

    public String[] getParameters() {
        return parameters;
    }

    public void setDRG(int DRG) {
        this.DRG = DRG;
    }

    public int getDRG() {
        return DRG;
    }

    /**
     * @param op the String to check
     * @return true if the operator is a statistical operator that returns items
     * in a list e.g sort(
     */
    public static boolean isListReturningStatsMethod(String op) {
        return (op.equals(SORT) || op.equals(MODE) || op.equals(RANDOM) || op.equals(QUADRATIC) || op.equals(TARTAGLIA_ROOTS)
                || op.equals(INVERSE_MATRIX) || op.equals(LINEAR_SYSTEM) || op.equals(TRIANGULAR_MATRIX) || op.equals(ECHELON_MATRIX))
                || op.equals(MATRIX_MULTIPLY) || op.equals(MATRIX_DIVIDE) || op.equals(MATRIX_ADD) || op.equals(MATRIX_SUBTRACT) || op.equals(MATRIX_POWER)
                || op.equals(MATRIX_TRANSPOSE) || op.equals(MATRIX_EDIT);
    }

    /**
     * @param op the String to check
     * @return true if the operator is a statistical operator that operates on a
     * data set and returns a single value: e.g sum(4,3,2,2...)
     */
    public static boolean isNumberReturningStatsMethod(String op) {
        return isStatsMethod(op) && !isListReturningStatsMethod(op);
    }

    /**
     * @param op the String to check
     * @return true if the operator is the random(rnd) operator
     */
    public static boolean isRandom(String op) {
        return (op.equals(RANDOM));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the log operator the form is
     * log(num,base)
     */
    public static boolean isLogToAnyBase(String op) {
        return (op.equals(LOG));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the log operator the form is
     * log-¹(num,base)
     */
    public static boolean isAntiLogToAnyBase(String op) {
        return (op.equals(LOG_INV) || op.equals(LOG_INV_ALT));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the natural log operator the form is
     * log-¹(num,base)
     */
    public static boolean isNaturalLog(String op) {
        return (op.equals(LN));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the inverse natural log operator the form
     * is log-¹(num,base)
     */
    public static boolean isInverseNaturalLog(String op) {
        return (op.equals(LN_INV) || op.equals(LN_INV_ALT));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the inverse natural log operator the form
     * is log-¹(num,base)
     */
    public static boolean isExpMethod(String op) {
        return (op.equals(EXP));
    }

    /**
     * @param op the String to check
     * @return true if the operator is the log operator or the log-¹ operator
     *
     */
    public static boolean isLogOrAntiLogToAnyBase(String op) {
        return isLogToAnyBase(op) || isAntiLogToAnyBase(op);
    }

    public boolean isDeterminant(String op) {
        return op.equals(DETERMINANT);
    }

    public boolean isMatrixInvert(String op) {
        return op.equals(INVERSE_MATRIX);
    }

    public boolean isLinearSys(String op) {
        return op.equals(LINEAR_SYSTEM);
    }

    public boolean isTriMat(String op) {
        return op.equals(TRIANGULAR_MATRIX);
    }

    public boolean isEchelon(String op) {
        return op.equals(ECHELON_MATRIX);
    }

    public boolean isMatrixCofactors(String op) {
        return op.equals(MATRIX_COFACTORS);
    }

    public boolean isMatrixAdjoint(String op) {
        return op.equals(MATRIX_ADJOINT);
    }

    public boolean isMatrixEigenVec(String op) {
        return op.equals(MATRIX_EIGENVEC);
    }

    public boolean isMatrixEigenPoly(String op) {
        return op.equals(MATRIX_EIGENPOLY);
    }

    public boolean isPrint(String op) {
        return op.equals(PRINT);
    }

    /**
     * @param op the String to check
     * @return true if the operator is a statistical operator..basically any
     * system function that takes more than 1 argument. So by definition even
     * the <code>log</code> and <code>alog</code> (and its log-¹ variant) are
     * included here. e.g
     * sum,prod,min,max,avg,var,rms,cov,s_d,st_err,rng,mrng,med,mode,rnd
     */
    public static boolean isStatsMethod(String op) {
        return (isUserDefinedFunction(op) || isLogOrAntiLogToAnyBase(op) || isBasicNumericalFunction(op)
                || isMatrixMethod(op) || isHardcodedStatsMethod(op)
                || op.equals(POW) || op.equals(DIFFERENTIATION)
                || op.equals(INTEGRATION) || op.equals(GENERAL_ROOT) || op.equals(QUADRATIC)
                || op.equals(TARTAGLIA_ROOTS) || op.equals(PERMUTATION) || op.equals(COMBINATION)
                || op.equals(LOG) || op.equals(LOG_INV) || op.equals(LOG_INV_ALT) || op.equals(PRINT));
    }//end method

    /**
     *
     * @param op The method name
     * @return true if the method is capable of acting on one or more matrix
     * functions
     */
    public static boolean isMatrixMethod(String op) {
        return op.equals(LINEAR_SYSTEM) || op.equals(DETERMINANT) || op.equals(INVERSE_MATRIX)
                || op.equals(TRIANGULAR_MATRIX) || op.equals(ECHELON_MATRIX) || op.equals(MATRIX_MULTIPLY)
                || op.equals(MATRIX_DIVIDE) || op.equals(MATRIX_ADD) || op.equals(MATRIX_SUBTRACT)
                || op.equals(MATRIX_POWER) || op.equals(MATRIX_EDIT) || op.equals(MATRIX_TRANSPOSE)
                || op.equals(MATRIX_COFACTORS) || op.equals(MATRIX_ADJOINT) || op.equals(MATRIX_EIGENPOLY) || op.equals(MATRIX_EIGENVEC);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the matrix multiplication method name
     */
    public static boolean isMatrixMul(String op) {
        return op.equals(MATRIX_MULTIPLY);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the matrix division method name
     */
    public static boolean isMatrixDiv(String op) {
        return op.equals(MATRIX_DIVIDE);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the matrix addition method name
     */
    public static boolean isMatrixAdd(String op) {
        return op.equals(MATRIX_ADD);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the matrix subtraction method name
     */
    public static boolean isMatrixSub(String op) {
        return op.equals(MATRIX_SUBTRACT);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the matrix power method name
     */
    public static boolean isMatrixPow(String op) {
        return op.equals(MATRIX_POWER);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the transpose method name
     */
    public static boolean isTranspose(String op) {
        return op.equals(MATRIX_TRANSPOSE);
    }

    /**
     *
     * @param op The method name
     * @return true if the method is the matrix edit method name
     */
    public static boolean isMatrixEdit(String op) {
        return op.equals(MATRIX_EDIT);
    }

    /**
     *
     * @return true if the Function name has been defined by the user in the
     * user's workspace.
     */
    public static boolean isUserDefinedFunction(String op) {
        return FunctionManager.lookUp(op) != null;
    }//end method

    /**
     * A fix for stuff like sum,(,13,+,3,)...
     *
     */
    private static void quickFixCompoundStructuresInStatsExpression(List<String> list) {

        String methodName = list.get(0);

        if (!isStatsMethod(methodName)) {
            return;
        }
        int sz = list.size();
        for (int i = 0; i < sz; i++) {
            String token = list.get(i);
            if (Operator.isBinaryOperator(token)) {

                StringBuilder b = new StringBuilder();

                for (i = 1; i < sz; i++) {
                    b.append(list.get(i));
                }

                String fun = b.toString();

                MathExpression f = new MathExpression(fun);
                String val = f.solve();
                list.clear();
                list.add(methodName);
                list.add("(");
                list.add(val);
                list.add(")");
                break;
            }
        }

    }

    /**
     *
     * @param list A list containing a portion of a scanned function that has
     * information about a method and its parameters..e.g. [sin,(,3.14,)] , or
     * [matrix_edit,(,M,3,4,-90,)] may be grabbed from a scanner output and sent
     * to this method to evaluate.
     * @param DRG The trigonometric mode in which to run the method.
     * @return a {@link List} object which is the output of the method's
     * operation.
     */
    public static List<String> run(List<String> list, DRG_MODE DRG) {

        quickFixCompoundStructuresInStatsExpression(list);
        String name = list.get(0);

        String result = "";
        list.subList(0, 2).clear();//remove the method name and its opening bracket.
        list.remove(list.size() - 1);//remove the closing bracket.
        int sz = list.size();

        if (isStatsMethod(name)) {
            for (BasicNumericalMethod basicNumericalMethod : getBasicNumericalMethods()) {
                if (name.equals(basicNumericalMethod.getName())) {
                    Set set = new Set(list);
                    //TODO, make basicNumericalMethod statefull (initialize by reflection)
                    //basicNumericalMethod.setRadDegGrad(DRG);
                    result = basicNumericalMethod.solve(new ArrayList<>(set.getData()));
                    list.clear();
                    list.add(result);
                    return list;
                }
            }
            if (name.equals(PROD)) {
                Set set = new Set(list);
                result = String.valueOf(set.prod());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(MEDIAN)) {
                Set set = new Set(list);
                result = String.valueOf(set.median());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(RANGE)) {
                Set set = new Set(list);
                result = String.valueOf(set.rng());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(MID_RANGE)) {
                Set set = new Set(list);
                result = String.valueOf(set.mrng());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(MAX)) {
                Set set = new Set(list);
                result = String.valueOf(set.max());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(MIN)) {
                Set set = new Set(list);
                result = String.valueOf(set.min());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(ROOT_MEAN_SQUARED)) {
                Set set = new Set(list);
                result = String.valueOf(set.rms());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(COEFFICIENT_OF_VARIATION)) {
                Set set = new Set(list);
                result = String.valueOf(set.cov());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(STD_DEV)) {
                Set set = new Set(list);
                result = String.valueOf(set.std_dev());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(VARIANCE)) {
                Set set = new Set(list);
                result = String.valueOf(set.var());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(STD_ERR)) {
                Set set = new Set(list);
                result = String.valueOf(set.std_err());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(DIFFERENTIATION)) {
                Set set = new Set(list);
                result = String.valueOf(set.differentiate());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(INTEGRATION)) {
                Set set = new Set(list);
                result = String.valueOf(set.integrate());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(GENERAL_ROOT)) {
                Set set = new Set(list);
                result = String.valueOf(set.rootOfEquation());
                list.clear();
                list.add(result);
                return list;
            } else if ((name.equals(DETERMINANT))) {
                Set set = new Set(list);
                result = String.valueOf(set.determinant());
                list.clear();
                list.add(result);
                return list;
            } else if (isUserDefinedFunction(name)) {
                try {
                    Set set = new Set(list);
                    result = String.valueOf(set.evaluateUserDefinedFunction(name));
                    list.clear();
                    list.add(result);
                    return list;
                }//end try
                catch (ClassNotFoundException cnfe) {
                    //System.out.println( " Function Not Defined!");
                }//end catch
            }//end else if
            else if (name.equals(LOG)) {
                if (sz == 1) {
                    result = String.valueOf(Maths.logToAnyBase(Double.valueOf(list.get(0)), 10));
                    list.clear();
                    list.add(result);
                } else if (sz == 2) {
                    result = String.valueOf(Maths.logToAnyBase(Double.valueOf(list.get(0)), Double.valueOf(list.get(1))));
                    list.clear();
                    list.add(result);
                }

                return list;
            } else if ((name.equals(LOG_INV) || name.equals(LOG_INV_ALT))) {
                if (sz == 1) {
                    result = String.valueOf(Maths.antiLogToAnyBase(Double.valueOf(list.get(0)), 10));
                    list.clear();
                    list.add(result);
                    return list;
                } else if (sz == 2) {
                    result = String.valueOf(Maths.antiLogToAnyBase(Double.valueOf(list.get(0)), Double.valueOf(list.get(1))));
                    list.clear();
                    list.add(result);
                    return list;
                }
            } else if (name.equals(QUADRATIC)) {
                Set set = new Set(list);
                result = set.quadraticRoots();
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(TARTAGLIA_ROOTS)) {
                Set set = new Set(list);
                result = set.tartaglianRoots();
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(POW) && sz == 2) {
                Set set = new Set(list);
                result = String.valueOf(set.power());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(PERMUTATION) && sz == 2) {
                Set set = new Set(list);
                result = String.valueOf(set.permutation());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(COMBINATION) && sz == 2) {
                Set set = new Set(list);
                result = String.valueOf(set.combination());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(MODE)) {
                Set set = new Set(list);
                result = String.valueOf(set.mode());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(RANDOM) && sz >= 0 && sz <= 2) {
                Set set = new Set(list);
                result = String.valueOf(set.random());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(SORT)) {
                Set set = new Set(list);
                result = String.valueOf(set.sort());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(INVERSE_MATRIX)) {
                Set set = new Set(list);
                Matrix matrix = set.invert();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(LINEAR_SYSTEM)) {
                Set set = new Set(list);
                Matrix matrix = set.solveSystem();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(TRIANGULAR_MATRIX)) {
                Set set = new Set(list);
                Matrix matrix = set.triMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(ECHELON_MATRIX)) {
                Set set = new Set(list);
                Matrix matrix = set.echelon();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_MULTIPLY)) {
                Set set = new Set(list);
                Matrix matrix = set.multiplyMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_DIVIDE)) {
                Set set = new Set(list);
                Matrix matrix = set.divideMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_ADD)) {
                Set set = new Set(list);
                Matrix matrix = set.addMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_SUBTRACT)) {
                Set set = new Set(list);
                Matrix matrix = set.subtractMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_POWER)) {
                Set set = new Set(list);
                Matrix matrix = set.powerMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_TRANSPOSE)) {
                Set set = new Set(list);
                Matrix matrix = set.transpose();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_EDIT)) {//matrix_edit(M,row,col,val)
                Set set = new Set(list);
                Matrix matrix = set.editMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_ADJOINT)) {
                Set set = new Set(list);
                Matrix matrix = set.adjoint();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_COFACTORS)) {
                Set set = new Set(list);
                Matrix matrix = set.cofactorMatrix();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_EIGENPOLY)) {
                Set set = new Set(list);
                String poly = set.eigenPoly();

                list.clear();
                String ref = "anon" + (FunctionManager.countAnonymousFunctions() + 1);

                Function.storeAnonymousFunction("@(" + Matrix.lambda + ")" + poly);
                list.add(ref);
                return list;
            } else if (name.equals(MATRIX_EIGENVEC)) {
                Set set = new Set(list);
                Matrix matrix = set.eigenVectors();
                list.clear();
                String ref = Function.storeAnonymousMatrixFunction(matrix);
                list.add(ref);
                return list;
            } else if (name.equals(PRINT)) {

                Set set = new Set(list);
                set.print();
                list.clear();
                list.add("0");
                return list;
            }

        } else {

            if (sz == 1) {
                try {
                    if (!list.get(0).equals("Infinity") && isNumber(list.get(0))) {
                        if (DRG == DRG_MODE.DEG) {
                            if (name.equals(SIN)) {
                                result = String.valueOf(Maths.sinDegToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(SINH)) {
                                result = String.valueOf(Math.sinh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COS)) {
                                result = String.valueOf(Maths.cosDegToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSH)) {
                                result = String.valueOf(Math.cosh(Double.valueOf(list.get(0))));
                            } else if (name.equals(TAN)) {
                                result = String.valueOf(Maths.tanDegToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(TANH)) {
                                result = String.valueOf(Math.tanh(Double.valueOf(list.get(0))));
                            } else if (name.equals(SEC)) {
                                result = String.valueOf(1 / Maths.cosDegToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(SECH)) {
                                result = String.valueOf(1 / Math.cosh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSEC)) {
                                result = String.valueOf(1 / Maths.sinDegToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSECH)) {
                                result = String.valueOf(1 / Math.sinh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COT)) {
                                result = String.valueOf(1 / Maths.tanDegToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(COTH)) {
                                result = String.valueOf(Math.tanh(Double.valueOf(list.get(0))));
                            } else if (name.equals(EXP) || name.equals(LN_INV_ALT) || name.equals(LN_INV)) {
                                result = String.valueOf(Math.exp(Double.valueOf(list.get(0))));
                            } else if (name.equals(LN)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0))));
                            } else if (name.equals(LG)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0))) / Math.log(10.0));
                            } else if (name.equals(ARC_SIN) || name.equals(ARC_SIN_ALT)) {
                                result = String.valueOf(Maths.asinRadToDeg(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COS) || name.equals(ARC_COS_ALT)) {
                                result = String.valueOf(Maths.acosRadToDeg(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_TAN) || name.equals(ARC_TAN_ALT)) {
                                result = String.valueOf(Maths.atanRadToDeg(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_SINH) || name.equals(ARC_SINH_ALT)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0)) + Math.sqrt(Math.pow(Double.valueOf(list.get(0)), 2) + 1)));
                            } else if (name.equals(ARC_COSH) || name.equals(ARC_COSH_ALT)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0)) + Math.sqrt(Math.pow(Double.valueOf(list.get(0)), 2) - 1)));
                            } else if (name.equals(ARC_TANH) || name.equals(ARC_TANH_ALT)) {
                                result = String.valueOf(0.5 * Math.log((1 + Double.valueOf(list.get(0))) / (1 - Double.valueOf(list.get(0)))));
                            } else if (name.equals(ARC_SEC) || name.equals("asec")) {
                                result = String.valueOf(Maths.acosRadToDeg(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COSEC) || name.equals("acsc")) {
                                result = String.valueOf(Maths.asinRadToDeg(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COT) || name.equals("acot")) {
                                result = String.valueOf(Maths.atanRadToDeg(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_SECH) || name.equals(ARC_SECH_ALT)) {
                                result = String.valueOf(Math.log((1 + Math.sqrt(1 - Math.pow(Double.valueOf(list.get(0)), 2))) / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COSECH) || name.equals(ARC_COSECH_ALT)) {
                                result = String.valueOf(Math.log((1 + Math.sqrt(1 + Math.pow(Double.valueOf(list.get(0)), 2))) / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COTH) || name.equals(ARC_COTH_ALT)) {
                                result = String.valueOf(0.5 * Math.log((Double.valueOf(list.get(0)) + 1) / (Double.valueOf(list.get(0)) - 1)));
                            } else if (name.equals(LG_INV) || name.equals(LG_INV_ALT)) {
                                result = String.valueOf(Math.pow(10, Double.valueOf(list.get(0))));
                            } else if (name.equals(SQRT)) {
                                result = String.valueOf(Math.sqrt(Double.valueOf(list.get(0))));
                            } else if (name.equals(CBRT)) {
                                result = String.valueOf(Math.cbrt(Double.valueOf(list.get(0))));
                            } else if (name.equals(SQUARE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(val * val);
                            } else if (name.equals(CUBE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(val * val * val);
                            } else if (name.equals(INVERSE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(1 / val);
                            } else if (name.equals(FACT)) {
                                result = Maths.fact(list.get(0));
                            }

                            list.clear();
                            list.add(result);
                            return list;

//add more definitions below....
                        }//end if DRG == 0
                        else if (DRG == DRG_MODE.RAD) {
                            if (name.equals(SIN)) {
                                result = String.valueOf(Math.sin(Double.valueOf(list.get(0))));
                            } else if (name.equals(SINH)) {
                                result = String.valueOf(Math.sinh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COS)) {
                                result = String.valueOf(Math.cos(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSH)) {
                                result = String.valueOf(Math.cosh(Double.valueOf(list.get(0))));
                            } else if (name.equals(TAN)) {
                                result = String.valueOf(Math.tan(Double.valueOf(list.get(0))));
                            } else if (name.equals(TANH)) {
                                result = String.valueOf(Math.tanh(Double.valueOf(list.get(0))));
                            } else if (name.equals(SEC)) {
                                result = String.valueOf(1 / Math.cos(Double.valueOf(list.get(0))));
                            } else if (name.equals(SECH)) {
                                result = String.valueOf(1 / Math.cosh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSEC)) {
                                result = String.valueOf(1 / Math.sin(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSECH)) {
                                result = String.valueOf(1 / Math.sinh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COT)) {
                                result = String.valueOf(1 / Math.tan(Double.valueOf(list.get(0))));
                            } else if (name.equals(COTH)) {
                                result = String.valueOf(Math.tanh(Double.valueOf(list.get(0))));
                            } else if (name.equals(EXP) || name.equals(LN_INV_ALT) || name.equals(LN_INV)) {
                                result = String.valueOf(Math.exp(Double.valueOf(list.get(0))));
                            } else if (name.equals(LN)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0))));
                            } else if (name.equals(LG)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0))) / Math.log(10.0));
                            } else if (name.equals(ARC_SIN) || name.equals(ARC_SIN_ALT)) {
                                result = String.valueOf(Math.asin(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COS) || name.equals(ARC_COS_ALT)) {
                                result = String.valueOf(Math.acos(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_TAN) || name.equals(ARC_TAN_ALT)) {
                                result = String.valueOf(Math.atan(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_SINH) || name.equals(ARC_SINH_ALT)) {
                                result = String.valueOf(Maths.asinh(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_COSH) || name.equals(ARC_COSH_ALT)) {
                                result = String.valueOf(Maths.acosh(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_TANH) || name.equals(ARC_TANH_ALT)) {
                                result = String.valueOf(Maths.atanh(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_SEC) || name.equals("asec")) {
                                result = String.valueOf(Math.acos(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COSEC) || name.equals("acsc")) {
                                result = String.valueOf(Math.asin(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COT) || name.equals("acot")) {
                                result = String.valueOf(Math.atan(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_SECH) || name.equals(ARC_SECH_ALT)) {
                                result = String.valueOf(Maths.asech(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_COSECH) || name.equals(ARC_COSECH_ALT)) {
                                result = String.valueOf(Maths.acsch(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_COTH) || name.equals(ARC_COTH_ALT)) {
                                result = String.valueOf(Maths.acoth(Double.parseDouble(list.get(0))));
                            } else if (name.equals(LG_INV) || name.equals(LG_INV_ALT)) {
                                result = String.valueOf(Math.pow(10, Double.valueOf(list.get(0))));
                            } else if (name.equals(SQRT)) {
                                result = String.valueOf(Math.sqrt(Double.valueOf(list.get(0))));
                            } else if (name.equals(CBRT)) {
                                result = String.valueOf(Math.cbrt(Double.valueOf(list.get(0))));
                            } else if (name.equals(SQUARE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(val * val);
                            } else if (name.equals(CUBE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(val * val * val);
                            } else if (name.equals(INVERSE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(1 / val);
                            } else if (name.equals(FACT)) {
                                result = Maths.fact(list.get(0));
                            }

                            list.clear();
                            list.add(result);
                            return list;

//add more definitions below....
                        }//end else if DRG == 1
                        else if (DRG == DRG_MODE.GRAD) {
                            if (name.equals(SIN)) {
                                result = String.valueOf(Maths.sinGradToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(SINH)) {
                                result = String.valueOf(Math.sinh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COS)) {
                                result = String.valueOf(Maths.cosGradToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSH)) {
                                result = String.valueOf(Math.cosh(Double.valueOf(list.get(0))));
                            } else if (name.equals(TAN)) {
                                result = String.valueOf(Maths.tanGradToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(TANH)) {
                                result = String.valueOf(Math.tanh(Double.valueOf(list.get(0))));
                            } else if (name.equals(SEC)) {
                                result = String.valueOf(1 / Maths.cosGradToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(SECH)) {
                                result = String.valueOf(1 / Math.cosh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSEC)) {
                                result = String.valueOf(1 / Maths.sinGradToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(COSECH)) {
                                result = String.valueOf(1 / Math.sinh(Double.valueOf(list.get(0))));
                            } else if (name.equals(COT)) {
                                result = String.valueOf(1 / Maths.tanGradToRad(Double.valueOf(list.get(0))));
                            } else if (name.equals(COTH)) {
                                result = String.valueOf(Math.tanh(Double.valueOf(list.get(0))));
                            } else if (name.equals(EXP) || name.equals(LN_INV_ALT) || name.equals(LN_INV)) {
                                result = String.valueOf(Math.exp(Double.valueOf(list.get(0))));
                            } else if (name.equals(LN)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0))));
                            } else if (name.equals(LG)) {
                                result = String.valueOf(Math.log(Double.valueOf(list.get(0))) / Math.log(10.0));
                            } else if (name.equals(ARC_SIN) || name.equals(ARC_SIN_ALT)) {
                                result = String.valueOf(Maths.asinRadToGrad(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COS) || name.equals(ARC_COS_ALT)) {
                                result = String.valueOf(Maths.acosRadToGrad(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_TAN) || name.equals(ARC_TAN_ALT)) {
                                result = String.valueOf(Maths.atanRadToGrad(Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_SINH) || name.equals(ARC_SINH_ALT)) {
                                result = String.valueOf(Maths.asinh(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_COSH) || name.equals(ARC_COSH_ALT)) {
                                result = String.valueOf(Maths.acosh(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_TANH) || name.equals(ARC_TANH_ALT)) {
                                result = String.valueOf(Maths.atanh(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_SEC) || name.equals("asec")) {
                                result = String.valueOf(Maths.acosRadToGrad(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COSEC) || name.equals("acsc")) {
                                result = String.valueOf(Maths.asinRadToGrad(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_COT) || name.equals("acot")) {
                                result = String.valueOf(Maths.atanRadToGrad(1 / Double.valueOf(list.get(0))));
                            } else if (name.equals(ARC_SECH) || name.equals(ARC_SECH_ALT)) {
                                result = String.valueOf(Maths.asech(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_COSECH) || name.equals(ARC_COSECH_ALT)) {
                                result = String.valueOf(Maths.acsch(Double.parseDouble(list.get(0))));
                            } else if (name.equals(ARC_COTH) || name.equals(ARC_COTH_ALT)) {
                                result = String.valueOf(Maths.acoth(Double.parseDouble(list.get(0))));
                            } else if (name.equals(LG_INV) || name.equals(LG_INV_ALT)) {
                                result = String.valueOf(Math.pow(10, Double.valueOf(list.get(0))));
                            } else if (name.equals(SQRT)) {
                                result = String.valueOf(Math.sqrt(Double.valueOf(list.get(0))));
                            } else if (name.equals(CBRT)) {
                                result = String.valueOf(Math.cbrt(Double.valueOf(list.get(0))));
                            } else if (name.equals(SQUARE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(val * val);
                            } else if (name.equals(CUBE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(val * val * val);
                            } else if (name.equals(INVERSE)) {
                                double val = Double.valueOf(list.get(0));
                                result = String.valueOf(1 / val);
                            } else if (name.equals(FACT)) {
                                result = Maths.fact(list.get(0));
                            }
                            list.clear();
                            list.add(result);
                            return list;
//add more definitions below....
                        }//end else if DRG == 2
                    }//end if list.get(0).equals("Infinity")
                    else if (list.get(0).equals("Infinity")) {
                        list.add("Infinity");
                        return list;
                    }
                }//end try
                catch (NumberFormatException numerror) {

                }//end catch
                catch (NullPointerException nullerror) {

                }//end catch
                catch (IndexOutOfBoundsException inderror) {

                }//end catch

                list.add("Syntax Error!");
                return list;
            }//end if parameters.length==1

        }//end else

        throw new IllegalArgumentException(" Unknown function: " + name);
    }//end method

    /**
     *
     * @param methodName The name of the method
     * @return true if the method has been defined by the user or is defined by
     * the parser.
     */
    public static boolean isUnaryPreOperatorORDefinedMethod(String methodName) {
        return isDefinedMethod(methodName) || parser.Operator.isUnaryPreOperator(methodName);
    }//end method

    /**
     *
     * @param methodName The name of the method
     * @return true if the method has been defined by the user or is defined by
     * the parser.
     */
    public static boolean isDefinedMethod(String methodName) {
        return arrayContains(getInbuiltMethods(), methodName) || FunctionManager.contains(methodName);
    }//end method

    /**
     *
     * @param methodName The name of the method
     * @return true if the method is defined by the parser as a core inbuilt
     * function.
     */
    public static boolean isInBuiltMethod(String methodName) {
        return arrayContains(getInbuiltMethods(), methodName);
    }//end method

    /**
     *
     * @param array An array of strings
     * @param str The string to check for.
     * @return true if the array contains the specified string.
     */
    public static boolean arrayContains(String array[], String str) {
        for (String s : array) {
            if (s.equals(str)) {
                return true;
            }//end if
        }//end for
        return false;
    }//end method

    /**
     *
     * @param name The string to check if or not it represents a valid method
     * name.<br>
     * <b color = 'red'>
     * The method may or may not have been defined. But once it represents a
     * valid method name, this method will return true. In contrast, the
     * 'isDefinedMethod' checks whether or not the method has been
     * afore-defined.
     * </b>
     * @return true if the string represents a valid method name.
     */
    public static boolean isMethodName(String name) {
        try {
            String end = "";
            if (name.equals("-¹")) {
                return false;
            }//end if
            else if (name.endsWith("-¹") && !name.equals("-¹")) {
                end = "-¹";
                name = name.substring(0, name.length() - 2);
            }
            if (isMethodNameBeginner(name.substring(0, 1))) {
                int len = name.length();
                for (int i = 0; i < len; i++) {
                    if (!isMethodNameBuilder(name.substring(i, i + 1))) {
                        return false;
                    }//end if
                }//end for loop

                return true;
            }//end if
            return false;
        }//end try
        catch (IndexOutOfBoundsException | NullPointerException boundsException) {
            return false;
        }
    }//end method

    /**
     *
     * @param name The string to check.
     * @return true if the string is part of the valid characters that can be
     * used to build a method name.
     */
    public static boolean isMethodNameBuilder(String name) {
        if (isMethodNameBeginner(name) || isDigit(name)) {
            return true;
        }//end if
        return false;
    }//end method

    public static boolean isMethodNameBeginner(String name) {
        if (name == null || name.isEmpty()) {
            return false;
        }

        // First character
        char first = name.charAt(0);

        // Allow underscore or dollar sign directly
        if (first == '_' || first == '$') {
            return true;
        }

        // If it's a letter and not a perm/comb operator, it's valid
        return Character.isLetter(first) && !parser.Operator.isPermOrComb(name);
    }

    @Override
    public String toString() {
        String out = name.concat("(");
        int sz = parameters.length;
        for (int i = 0; i < sz; i++) {
            out = out.concat(parameters[i].concat(","));
        }//end for loop.
        out = out.substring(0, out.length());
        out = out.concat(")");

        return out;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Method) {
            Method m = (Method) obj;
            return m.name.equals(this.name) && m.parameters.length == this.parameters.length;
        }
        return false;
    }

    /**
     * Appends the contents of the 2 arrays and returns the result as a new
     * array object.
     *
     * @param arr1 The first array.
     * @param arr2 The second array.
     */
    private static String[] join(String[] arr1, String arr2[]) {
        String larger[] = new String[arr1.length + arr2.length];

        int len = larger.length;
        int subLen1 = arr1.length;
        int subLen2 = arr2.length;
        for (int i = 0; i < len; i++) {
            if (i < subLen1) {
                larger[i] = arr1[i];
            }//end if
            else {
                larger[i] = arr2[i];
            }//end else

        }//end for

        return larger;
    }//end method.

    private static boolean isHardcodedStatsMethod(String op) {
        for (String x : getStatsMethods()) {
            if (x.equals(op)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Scans the given expression for statistical operators
     *
     * @param expr The input expression
     * @return true if it finds statistical operators in the expression.
     */
    public static boolean hasStatsMethod(String expr) {
        CustomScanner cs = new CustomScanner(expr, true, getStatsMethods());
        List<String> scan = cs.scan();
        int size = scan.size();
        for (int i = 0; i < size; i++) {
            try {
                if (isStatsMethod(scan.get(i)) && scan.get(i + 1).startsWith("(")) {
                    return true;
                }//end if
            }//end try
            catch (IndexOutOfBoundsException boundsException) {
                return false;
            }
        }//end for
        return false;
    }

    public static void main(String args[]) {
        Method m = new Method("F(x,y,z)");
        System.out.println(Method.isMethodName("F"));
        System.out.println(isMethodNameBeginner("Č"));

        StringBuilder builder = new StringBuilder();
        for (String name : getInbuiltMethods()) {
            builder.append(name).append(",");
        }

        System.out.println(builder);

    }

}//end class
