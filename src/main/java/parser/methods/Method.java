/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package parser.methods;

import parser.CustomScanner;
import parser.Function;
import static parser.STRING.*;
import static parser.Number.*;
import static parser.Operator.*;
import parser.Set;
import parser.TYPE;
import java.util.InputMismatchException;
import java.util.List;
import math.Maths;
import math.matrix.expressParser.Matrix;
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

    public static final String SIN = "sin";
    public static final String COS = "cos";
    public static final String TAN = "tan";
    public static final String SINH = "sinh";
    public static final String COSH = "cosh";
    public static final String TANH = "tanh";
    public static final String ARC_SIN = "sin-¹";
    public static final String ARC_COS = "cos-¹";
    public static final String ARC_TAN = "tan-¹";
    public static final String ARC_SINH = "sinh-¹";
    public static final String ARC_COSH = "cosh-¹";
    public static final String ARC_TANH = "tanh-¹";
    public static final String SEC = "sec";
    public static final String COSEC = "csc";
    public static final String COT = "cot";
    public static final String SECH = "sech";
    public static final String COSECH = "csch";
    public static final String COTH = "coth";
    public static final String ARC_SEC = "sec-¹";
    public static final String ARC_COSEC = "csc-¹";
    public static final String ARC_COT = "cot-¹";
    public static final String ARC_SECH = "sech-¹";
    public static final String ARC_COSECH = "csch-¹";
    public static final String ARC_COTH = "coth-¹";
    public static final String EXP = "exp";
    public static final String LN = "ln";
    public static final String LG = "lg";
    public static final String LOG = "log";
    public static final String LN_INV = "ln-¹";
    public static final String LG_INV = "lg-¹";
    public static final String LOG_INV = "log-¹";
    public static final String ARC_SIN_ALT = "asin";
    public static final String ARC_COS_ALT = "acos";
    public static final String ARC_TAN_ALT = "atan";
    public static final String ARC_SINH_ALT = "asinh";
    public static final String ARC_COSH_ALT = "acosh";
    public static final String ARC_TANH_ALT = "atanh";
    public static final String ARC_SEC_ALT = "asec";
    public static final String ARC_COSEC_ALT = "acsc";
    public static final String ARC_COT_ALT = "acot";
    public static final String ARC_SECH_ALT = "asech";
    public static final String ARC_COSECH_ALT = "acsch";
    public static final String ARC_COTH_ALT = "acoth";
    public static final String LN_INV_ALT = "aln";
    public static final String LG_INV_ALT = "alg";
    public static final String LOG_INV_ALT = "alog";
    public static final String FLOOR = "floor";
    public static final String CEIL = "ceil";
    public static final String SQRT = "sqrt";
    public static final String CBRT = "cbrt";
    public static final String INVERSE = "inverse";
    public static final String SQUARE = "square";
    public static final String CUBE = "cube";
    public static final String POW = "pow";
    public static final String FACT = "fact";
    public static final String COMBINATION = "comb";
    public static final String PERMUTATION = "perm";
    public static final String SUM = "sum";
    public static final String PROD = "prod";
    public static final String AVG = "avg";
    public static final String MEDIAN = "med";
    public static final String MODE = "mode";
    public static final String RANGE = "rng";
    public static final String MID_RANGE = "mrng";
    public static final String ROOT_MEAN_SQUARED = "rms";
    public static final String COEFFICIENT_OF_VARIATION = "cov";
    public static final String MIN = "min";
    public static final String MAX = "max";
    public static final String STD_DEV = "s_d";
    public static final String VARIANCE = "variance";
    public static final String STD_ERR = "st_err";
    public static final String RANDOM = "rnd";
    public static final String SORT = "sort";
    public static final String PLOT = "plot";
    public static final String PRINT = "print";
    public static final String DIFFERENTIATION = "diff";
    public static final String INTEGRATION = "intg";
    public static final String QUADRATIC = "quad";
    public static final String TARTAGLIA_ROOTS = "t_root";
    public static final String GENERAL_ROOT = "root";
    public static final String LINEAR_SYSTEM = "linear_sys";
    public static final String DETERMINANT = "det";
    public static final String INVERSE_MATRIX = "invert";
    public static final String TRIANGULAR_MATRIX = "tri_mat";
    public static final String ECHELON_MATRIX = "echelon";
    public static final String MATRIX_MULTIPLY = "matrix_mul";
    public static final String MATRIX_DIVIDE = "matrix_div";
    public static final String MATRIX_ADD = "matrix_add";
    public static final String MATRIX_SUBTRACT = "matrix_sub";
    public static final String MATRIX_POWER = "matrix_pow";
    public static final String MATRIX_TRANSPOSE = "transpose";
    public static final String MATRIX_EDIT = "matrix_edit";
    public static final String MATRIX_COFACTORS = "cofactor";
    public static final String MATRIX_ADJOINT = "adjoint";
    public static final String MATRIX_EIGENVEC = "eigvec";
    public static final String MATRIX_EIGENPOLY = "eigpoly";

    /**
     * A list of all inbuilt methods of the parser of this software.The user is
     * free to define his own functions.
     *
     */
    public static final String[] inbuiltMethods
            = new String[]{
                SIN,
                COS,
                TAN,
                SINH,
                COSH,
                TANH,
                ARC_SIN,
                ARC_COS,
                ARC_TAN,
                ARC_SINH,
                ARC_COSH,
                ARC_TANH,
                SEC,
                COSEC,
                COT,
                SECH,
                COSECH,
                COTH,
                ARC_SEC,
                ARC_COSEC,
                ARC_COT,
                ARC_SECH,
                ARC_COSECH,
                ARC_COTH,
                EXP,
                LN,
                LG,
                LOG,
                LN_INV,
                LG_INV,
                LOG_INV,
                ARC_SIN_ALT,
                ARC_COS_ALT,
                ARC_TAN_ALT,
                ARC_SINH_ALT,
                ARC_COSH_ALT,
                ARC_TANH_ALT,
                ARC_SEC_ALT,
                ARC_COSEC_ALT,
                ARC_COT_ALT,
                ARC_SECH_ALT,
                ARC_COSECH_ALT,
                ARC_COTH_ALT,
                LN_INV_ALT,
                LG_INV_ALT,
                LOG_INV_ALT,
                FLOOR,
                CEIL,
                SQRT,
                CBRT,
                INVERSE,
                SQUARE,
                CUBE,
                POW,
                FACT,
                PRINT,
                COMBINATION,
                PERMUTATION,
                SUM,
                PROD,
                AVG,
                MEDIAN,
                MODE,
                RANGE,
                MID_RANGE,
                ROOT_MEAN_SQUARED,
                COEFFICIENT_OF_VARIATION,
                MIN,
                MAX,
                STD_DEV,
                VARIANCE,
                STD_ERR,
                RANDOM,
                SORT,
                PLOT,
                DIFFERENTIATION,
                INTEGRATION,
                QUADRATIC,
                TARTAGLIA_ROOTS,
                GENERAL_ROOT,
                LINEAR_SYSTEM,
                DETERMINANT,
                INVERSE_MATRIX,
                TRIANGULAR_MATRIX,
                ECHELON_MATRIX,
                MATRIX_MULTIPLY,
                MATRIX_DIVIDE,
                MATRIX_ADD,
                MATRIX_SUBTRACT,
                MATRIX_POWER,
                MATRIX_TRANSPOSE,
                MATRIX_EDIT,
                MATRIX_COFACTORS,
                MATRIX_ADJOINT,
                MATRIX_EIGENVEC,
                MATRIX_EIGENPOLY

            };

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
     * @param typeName The name of the method
     * @return the return type
     */
    public static String returnTypeDef(String typeName) {
        switch (typeName) {
            case SIN:
                return TYPE.NUMBER.toString();
            case COS:
                return TYPE.NUMBER.toString();
            case TAN:
                return TYPE.NUMBER.toString();
            case SINH:
                return TYPE.NUMBER.toString();
            case COSH:
                return TYPE.NUMBER.toString();
            case TANH:
                return TYPE.NUMBER.toString();
            case ARC_SIN:
                return TYPE.NUMBER.toString();
            case ARC_COS:
                return TYPE.NUMBER.toString();
            case ARC_TAN:
                return TYPE.NUMBER.toString();
            case ARC_SINH:
                return TYPE.NUMBER.toString();
            case ARC_COSH:
                return TYPE.NUMBER.toString();
            case ARC_TANH:
                return TYPE.NUMBER.toString();
            case SEC:
                return TYPE.NUMBER.toString();
            case COSEC:
                return TYPE.NUMBER.toString();
            case COT:
                return TYPE.NUMBER.toString();
            case SECH:
                return TYPE.NUMBER.toString();
            case COSECH:
                return TYPE.NUMBER.toString();
            case COTH:
                return TYPE.NUMBER.toString();
            case ARC_SEC:
                return TYPE.NUMBER.toString();
            case ARC_COSEC:
                return TYPE.NUMBER.toString();
            case ARC_COT:
                return TYPE.NUMBER.toString();
            case ARC_SECH:
                return TYPE.NUMBER.toString();
            case ARC_COSECH:
                return TYPE.NUMBER.toString();
            case ARC_COTH:
                return TYPE.NUMBER.toString();

            case EXP:
                return TYPE.NUMBER.toString();
            case LN:
                return TYPE.NUMBER.toString();
            case LG:
                return TYPE.NUMBER.toString();
            case LOG:
                return TYPE.NUMBER.toString();
            case LN_INV:
                return TYPE.NUMBER.toString();
            case LG_INV:
                return TYPE.NUMBER.toString();
            case LOG_INV:
                return TYPE.NUMBER.toString();
            case ARC_SIN_ALT:
                return TYPE.NUMBER.toString();
            case ARC_COS_ALT:
                return TYPE.NUMBER.toString();
            case ARC_TAN_ALT:
                return TYPE.NUMBER.toString();
            case ARC_SINH_ALT:
                return TYPE.NUMBER.toString();
            case ARC_COSH_ALT:
                return TYPE.NUMBER.toString();
            case ARC_TANH_ALT:
                return TYPE.NUMBER.toString();
            case ARC_SEC_ALT:
                return TYPE.NUMBER.toString();
            case ARC_COSEC_ALT:
                return TYPE.NUMBER.toString();
            case ARC_COT_ALT:
                return TYPE.NUMBER.toString();
            case ARC_SECH_ALT:
                return TYPE.NUMBER.toString();
            case ARC_COSECH_ALT:
                return TYPE.NUMBER.toString();
            case ARC_COTH_ALT:
                return TYPE.NUMBER.toString();
            case LN_INV_ALT:
                return TYPE.NUMBER.toString();
            case LG_INV_ALT:
                return TYPE.NUMBER.toString();
            case LOG_INV_ALT:
                return TYPE.NUMBER.toString();
            case FLOOR:
                return TYPE.NUMBER.toString();
            case CEIL:
                return TYPE.NUMBER.toString();
            case SQRT:
                return TYPE.NUMBER.toString();
            case CBRT:
                return TYPE.NUMBER.toString();
            case INVERSE:
                return TYPE.NUMBER.toString();
            case SQUARE:
                return TYPE.NUMBER.toString();
            case CUBE:
                return TYPE.NUMBER.toString();
            case POW:
                return TYPE.NUMBER.toString();
            case FACT:
                return TYPE.NUMBER.toString();
            case PRINT:
                return TYPE.VOID.toString();
            case COMBINATION:
                return TYPE.NUMBER.toString();
            case PERMUTATION:
                return TYPE.NUMBER.toString();
            case SUM:
                return TYPE.NUMBER.toString();
            case PROD:
                return TYPE.NUMBER.toString();
            case AVG:
                return TYPE.NUMBER.toString();
            case MEDIAN:
                return TYPE.NUMBER.toString();
            case MODE:
                return TYPE.LIST.toString();
            case RANGE:
                return TYPE.NUMBER.toString();
            case MID_RANGE:
                return TYPE.NUMBER.toString();
            case ROOT_MEAN_SQUARED:
                return TYPE.NUMBER.toString();
            case COEFFICIENT_OF_VARIATION:
                return TYPE.NUMBER.toString();
            case MIN:
                return TYPE.NUMBER.toString();
            case MAX:
                return TYPE.NUMBER.toString();
            case STD_DEV:
                return TYPE.NUMBER.toString();
            case VARIANCE:
                return TYPE.NUMBER.toString();
            case STD_ERR:
                return TYPE.NUMBER.toString();
            case RANDOM:
                return TYPE.LIST.toString();
            case SORT:
                return TYPE.NUMBER.toString();
            case PLOT:
                return TYPE.VOID.toString();
            case DIFFERENTIATION:
                return TYPE.NUMBER.toString();
            case INTEGRATION:
                return TYPE.NUMBER.toString();
            case QUADRATIC:
                return TYPE.LIST.toString();
            case TARTAGLIA_ROOTS:
                return TYPE.LIST.toString();
            case GENERAL_ROOT:
                return TYPE.NUMBER.toString();
            case LINEAR_SYSTEM:
                return TYPE.LIST.toString();
            case DETERMINANT:
                return TYPE.NUMBER.toString();
            case INVERSE_MATRIX:
                return TYPE.MATRIX.toString();
            case TRIANGULAR_MATRIX:
                return TYPE.MATRIX.toString();
            case ECHELON_MATRIX:
                return TYPE.MATRIX.toString();
            case MATRIX_MULTIPLY:
                return TYPE.MATRIX.toString();
            case MATRIX_DIVIDE:
                return TYPE.MATRIX.toString();
            case MATRIX_ADD:
                return TYPE.MATRIX.toString();
            case MATRIX_SUBTRACT:
                return TYPE.MATRIX.toString();
            case MATRIX_POWER:
                return TYPE.MATRIX.toString();
            case MATRIX_TRANSPOSE:
                return TYPE.MATRIX.toString();
            case MATRIX_EDIT:
                return TYPE.VOID.toString();
            case MATRIX_ADJOINT:
                return TYPE.MATRIX.toString();
            case MATRIX_COFACTORS:
                return TYPE.MATRIX.toString();
            case MATRIX_EIGENPOLY:
                return TYPE.ALGEBRAIC_EXPRESSION.toString();
            case MATRIX_EIGENVEC:
                return TYPE.MATRIX.toString();

            default:
                return TYPE.NUMBER.toString();
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
        String[] userDefined = new String[sz + inbuiltMethods.length];
        String[] keyset = FunctionManager.FUNCTIONS.keySet().toArray(new String[]{});
        System.arraycopy(keyset, 0, userDefined, 0, keyset.length);
        System.arraycopy(inbuiltMethods, 0, userDefined, keyset.length, inbuiltMethods.length);
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
        return (op.equals(SUM) || op.equals(PROD) || op.equals(AVG) || op.equals(MEDIAN) || op.equals(MODE)
                || op.equals(RANGE) || op.equals(MID_RANGE) || op.equals(ROOT_MEAN_SQUARED) || op.equals(COEFFICIENT_OF_VARIATION) || op.equals(MIN)
                || op.equals(MAX) || op.equals(STD_DEV) || op.equals(VARIANCE) || op.equals(STD_ERR) || op.equals(RANDOM)
                || op.equals(SORT) || isUserDefinedFunction(op) || isLogOrAntiLogToAnyBase(op) || op.equals(POW) || op.equals(DIFFERENTIATION)
                || op.equals(INTEGRATION)
                || op.equals(GENERAL_ROOT) || op.equals(QUADRATIC) || op.equals(TARTAGLIA_ROOTS) || op.equals(PERMUTATION) || op.equals(COMBINATION)
                || op.equals(LOG) || op.equals(LOG_INV) || op.equals(LOG_INV_ALT) || isMatrixMethod(op) || op.equals(PRINT));
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
     *
     * @param list A list containing a portion of a scanned function that has
     * information about a method and its parameters..e.g. [sin,(,3.14,)] , or
     * [matrix_edit,(,M,3,4,-90,)] may be grabbed from a scanner output and sent
     * to this method to evaluate.
     * @param DRG The trigonometric mode in which to run the method.
     * @return a {@link List} object which is the output of the method's
     * operation.
     */
    public static List<String> run(List<String> list, int DRG) {
        
        String name = list.get(0);

        String result = "";
        list.subList(0, 2).clear();//remove the method name and its opening bracket.
        list.remove(list.size() - 1);//remove the closing bracket.
        int sz = list.size();
        if (isStatsMethod(name)) {

            if (name.equals(SUM)) {
                Set set = new Set(list);
                result = String.valueOf(set.sum());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(PROD)) {
                Set set = new Set(list);
                result = String.valueOf(set.prod());
                list.clear();
                list.add(result);
                return list;
            } else if (name.equals(AVG)) {
                Set set = new Set(list);
                result = String.valueOf(set.avg());
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
                        if (DRG == 0) {
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
                            } else if (name.equals(CEIL)) {
                                result = String.valueOf(Math.ceil(Double.valueOf(list.get(0))));
                            } else if (name.equals(FLOOR)) {
                                result = String.valueOf(Math.floor(Double.valueOf(list.get(0))));
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
                        else if (DRG == 1) {

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
                            } else if (name.equals(CEIL)) {
                                result = String.valueOf(Math.ceil(Double.valueOf(list.get(0))));
                            } else if (name.equals(FLOOR)) {
                                result = String.valueOf(Math.floor(Double.valueOf(list.get(0))));
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
                        else if (DRG == 2) {

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
                            } else if (name.equals(CEIL)) {
                                result = String.valueOf(Math.ceil(Double.valueOf(list.get(0))));
                            } else if (name.equals(FLOOR)) {
                                result = String.valueOf(Math.floor(Double.valueOf(list.get(0))));
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
        return isDefinedMethod(methodName) || isUnaryPreOperator(methodName);
    }//end method

    /**
     *
     * @param methodName The name of the method
     * @return true if the method has been defined by the user or is defined by
     * the parser.
     */
    public static boolean isDefinedMethod(String methodName) {
        return arrayContains(inbuiltMethods, methodName) || FunctionManager.contains(methodName);
    }//end method

    /**
     *
     * @param methodName The name of the method
     * @return true if the method is defined by the parser as a core inbuilt
     * function.
     */
    public static boolean isInBuiltMethod(String methodName) {
        return arrayContains(inbuiltMethods, methodName);
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

    /**
     *
     * @param name The string to check.
     * @return true if the string can be the first character in a method name.
     */
    public static boolean isMethodNameBeginner(String name) {

        if (!isPermOrComb(name) && Character.isLetter(name.toCharArray()[0]) || name.equals("_") || name.equals("$")) {
            return true;
        }//end if
        return false;
    }//end method

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

    /**
     *
     * @return all the statistical methods used by the parser.
     */
    public static String[] getStatsMethods() {
        return new String[]{
            SUM, PROD, AVG, MEDIAN, MODE, RANGE, MID_RANGE, ROOT_MEAN_SQUARED, COEFFICIENT_OF_VARIATION, MIN, MAX, STD_DEV, VARIANCE, STD_ERR, RANDOM, SORT
        };

    }

    /**
     * Scans the given expression for statistical operators
     *
     * @param expr The input expression
     * @return true if it finds statistical operators in the expression.
     */
    public static boolean hasStatsMethod(String expr) {

        String[] statsoperators
                = new String[]{
                    SUM, PROD, AVG, MEDIAN, MODE, RANGE, MID_RANGE, ROOT_MEAN_SQUARED, COEFFICIENT_OF_VARIATION, MIN, MAX, STD_DEV, VARIANCE, STD_ERR, RANDOM, SORT
                };

        CustomScanner cs = new CustomScanner(expr, true, statsoperators);
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
        for (String name : inbuiltMethods) {
            builder.append(name).append(",");
        }

        System.out.println(builder);

    }

}//end class
