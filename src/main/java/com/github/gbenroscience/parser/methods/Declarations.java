package com.github.gbenroscience.parser.methods;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.parser.methods.ext.Avg;
import com.github.gbenroscience.parser.methods.ext.Abs;
import com.github.gbenroscience.parser.methods.ext.AvgN;
import com.github.gbenroscience.parser.methods.ext.CeilFloor;
import com.github.gbenroscience.parser.methods.ext.Count;
import com.github.gbenroscience.parser.methods.ext.Echo;
import com.github.gbenroscience.parser.methods.ext.Geom;
import com.github.gbenroscience.parser.methods.ext.GeomN;
import com.github.gbenroscience.parser.methods.ext.Gsum;
import com.github.gbenroscience.parser.methods.ext.Sum;
import com.github.gbenroscience.parser.methods.ext.Lengths;
import com.github.gbenroscience.parser.methods.ext.Rounding;

public class Declarations {


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
    public static final String SQRT = "sqrt";
    public static final String CBRT = "cbrt";
    public static final String INVERSE = "inverse";
    public static final String SQUARE = "square";
    public static final String CUBE = "cube";
    public static final String POW = "pow";
    public static final String FACT = "fact";
    public static final String COMBINATION = "comb";
    public static final String PERMUTATION = "perm";
    public static final String PROD = "prod";
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
    public static final String HELP = "help";

    /**
     * A list user registered hardcoded methods
     */
    private static final List<BasicNumericalMethod> BASIC_NUMERICAL_METHODS = new ArrayList<>();

    static List<BasicNumericalMethod> getBasicNumericalMethods() {
        return Collections.unmodifiableList(BASIC_NUMERICAL_METHODS);
    }

    public static void registerBasicNumericalMethod(BasicNumericalMethod basicNumericalMethod) {
        unregisterBasicNumericalMethod(basicNumericalMethod.getClass());
        BASIC_NUMERICAL_METHODS.add(basicNumericalMethod);
    }

    public static void unregisterBasicNumericalMethod(Class clazz) {
        for(int i = BASIC_NUMERICAL_METHODS.size()-1; i>=0; i--) {
            if (BASIC_NUMERICAL_METHODS.get(i).getClass().equals(clazz)){
                BASIC_NUMERICAL_METHODS.remove(i);
            }
        }
    }

    static {
        registerBasicNumericalMethod(new Echo());
        registerBasicNumericalMethod(new Echo.EchoN());
        registerBasicNumericalMethod(new Echo.EchoNI());
        registerBasicNumericalMethod(new Echo.EchoI());
        registerBasicNumericalMethod(new AvgN());
        registerBasicNumericalMethod(new GeomN());
        registerBasicNumericalMethod(new Sum());
        registerBasicNumericalMethod(new Avg());
        registerBasicNumericalMethod(new Geom());
        registerBasicNumericalMethod(new Count());
        registerBasicNumericalMethod(new Gsum());
        registerBasicNumericalMethod(new Rounding.RoundX());
        registerBasicNumericalMethod(new Rounding.RoundN());
        registerBasicNumericalMethod(new Rounding.RoundDigitsN());
        registerBasicNumericalMethod(new Rounding.Round());
        registerBasicNumericalMethod(new Abs());
        registerBasicNumericalMethod(new Lengths.Length());
        registerBasicNumericalMethod(new Lengths.LengthDecimal());
        registerBasicNumericalMethod(new Lengths.LengthFractional());
        registerBasicNumericalMethod(new CeilFloor.Ceil());
        registerBasicNumericalMethod(new CeilFloor.CeilN());
        registerBasicNumericalMethod(new CeilFloor.CeilDigitsN());
        registerBasicNumericalMethod(new CeilFloor.Floor());
        registerBasicNumericalMethod(new CeilFloor.FloorN());
        registerBasicNumericalMethod(new CeilFloor.FloorDigitsN());
    }

    /**
     * A list of all inbuilt methods of the parser of this software.The user is
     * free to define his own functions.
     */
    public static String[] getInbuiltMethods() {
        return createInBuiltMethods();
    }

    public static String[] createInBuiltMethods() {
        List<String> stats = Arrays.asList(getStatsMethods());
        List<String> rest = Arrays.asList(
                new String[]{HELP, SIN, COS, TAN, SINH, COSH, TANH, ARC_SIN, ARC_COS, ARC_TAN, ARC_SINH, ARC_COSH, ARC_TANH, SEC, COSEC, COT, SECH, COSECH, COTH, ARC_SEC, ARC_COSEC, ARC_COT, ARC_SECH,
                        ARC_COSECH, ARC_COTH, EXP, LN, LG, LOG, LN_INV, LG_INV, LOG_INV, ARC_SIN_ALT, ARC_COS_ALT, ARC_TAN_ALT, ARC_SINH_ALT, ARC_COSH_ALT, ARC_TANH_ALT, ARC_SEC_ALT, ARC_COSEC_ALT,
                        ARC_COT_ALT, ARC_SECH_ALT, ARC_COSECH_ALT, ARC_COTH_ALT, LN_INV_ALT, LG_INV_ALT, LOG_INV_ALT, SQRT, CBRT, INVERSE, SQUARE, CUBE, POW, FACT, PRINT, COMBINATION,
                        PERMUTATION, PLOT, DIFFERENTIATION, INTEGRATION, QUADRATIC, TARTAGLIA_ROOTS, GENERAL_ROOT, LINEAR_SYSTEM, DETERMINANT, INVERSE_MATRIX, TRIANGULAR_MATRIX, ECHELON_MATRIX,
                        MATRIX_MULTIPLY, MATRIX_DIVIDE, MATRIX_ADD, MATRIX_SUBTRACT, MATRIX_POWER, MATRIX_TRANSPOSE, MATRIX_EDIT, MATRIX_COFACTORS, MATRIX_ADJOINT, MATRIX_EIGENVEC, MATRIX_EIGENPOLY});
        List<String> r = new ArrayList<>(stats.size() + rest.size());
        r.addAll(stats);
        r.addAll(rest);
        r.addAll(methodsToNames(BASIC_NUMERICAL_METHODS));
        Collections.sort(r);
        return r.toArray(new String[r.size()]);
    }

    private static Collection<String> methodsToNames(List<BasicNumericalMethod> basicNumericalMethods) {
        List<String> result = new ArrayList<>(basicNumericalMethods.size());
        for (BasicNumericalMethod method : basicNumericalMethods) {
            result.add(method.getName());
        }
        return result;
    }

    /**
     * @param typeName The name of the method
     * @return the return type
     */
    public static String returnTypeDef(String typeName) {
        for (BasicNumericalMethod basicNumericalMethod : BASIC_NUMERICAL_METHODS) {
            if (typeName.equals(basicNumericalMethod.getName())) {
                return basicNumericalMethod.getType();
            }
        }
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
            case PROD:
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
     * @return all the statistical methods used by the parser.
     */
    static String[] getStatsMethods() {
        return new String[]{PROD, MEDIAN, MODE, RANGE, MID_RANGE, ROOT_MEAN_SQUARED, COEFFICIENT_OF_VARIATION, MIN, MAX, STD_DEV, VARIANCE, STD_ERR, RANDOM, SORT};

    }

    public static boolean isBasicNumericalFunction(String op){
        for(BasicNumericalMethod basicNumericalMethod: Declarations.getBasicNumericalMethods()){
            if (op.equals(basicNumericalMethod.getName())){
                return true;
            }
        }
        return false;
    }

    public static DRG_MODE degGradRadFromVariable() {
        String userDrgMode = System.getenv(DRG_MODE.DEG_MODE_VARIABLE);
        if (userDrgMode == null) {
            userDrgMode = System.getProperty(DRG_MODE.DEG_MODE_VARIABLE);
        }
        if (userDrgMode == null) {
            return DRG_MODE.RAD;
        }
        if (userDrgMode.toUpperCase().trim().equals("DEG")) {
            return DRG_MODE.DEG;
        } else if (userDrgMode.toUpperCase().trim().equals("RAD")) {
            return DRG_MODE.RAD;
        } if (userDrgMode.toUpperCase().trim().equals("GRAD")) {
            return DRG_MODE.GRAD;
        } else {
            return DRG_MODE.RAD;
        }
    }
}
