package com.github.gbenroscience.parser.methods;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.github.gbenroscience.logic.DRG_MODE;
import com.github.gbenroscience.parser.MathExpression;
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
    public static final String ARC_SIN = "sin-¹";
    public static final String ARC_COS = "cos-¹";
    public static final String ARC_TAN = "tan-¹";
    public static final String ATAN2 = "atan2";
    public static final String SEC = "sec";
    public static final String COSEC = "csc";
    public static final String COT = "cot";
    public static final String ARC_SEC = "sec-¹";
    public static final String ARC_COSEC = "csc-¹";
    public static final String ARC_COT = "cot-¹";
    public static final String ARC_SIN_ALT = "asin";
    public static final String ARC_COS_ALT = "acos";
    public static final String ARC_TAN_ALT = "atan";
    public static final String ARC_SEC_ALT = "asec";
    public static final String ARC_COSEC_ALT = "acsc";
    public static final String ARC_COT_ALT = "acot";

    public static final String SINH = "sinh";
    public static final String COSH = "cosh";
    public static final String TANH = "tanh";
    public static final String ARC_SINH = "sinh-¹";
    public static final String ARC_COSH = "cosh-¹";
    public static final String ARC_TANH = "tanh-¹";
    public static final String SECH = "sech";
    public static final String COSECH = "csch";
    public static final String COTH = "coth";
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
    public static final String ARC_SINH_ALT = "asinh";
    public static final String ARC_COSH_ALT = "acosh";
    public static final String ARC_TANH_ALT = "atanh";
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

    public static final String ABS = "abs";
    public static final String FLOOR = "floor";
    public static final String CEIL = "ceil";

    public static final String LIST_SUM = "listsum";
    public static final String SUM = "sum";
    public static final String PROD = "prod";
    public static final String MEAN = "mean";
    public static final String AVG = "listavg";
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
    public static final String AUTO_DIFF = "autodiff";
    public static final String AUTO_DIFF_N = "autodiffN";
    public static final String INTEGRATION = "intg";

    public static final String DIFF_EQN = "diffeqn";
    public static final String DIFF_EQN_PATH = "diffeqnPath";
    public static final String DIFF_EQN_HO = "diffeqnHO";
    public static final String DIFF_EQN_PATH_HO = "diffeqnPathHO";

    public static final String QUADRATIC = "quadratic";
    public static final String TARTAGLIA_ROOTS = "t_root";
    public static final String GENERAL_ROOT = "root";
    public static final String ROTOR = "rot";
    public static final String NOW = "now";
    public static final String NANOS = "nanos";
    /**
     * May take its input matrix as a list or as an anonymous function or as a
     * Matrix variable. e.g linear_sys(3,2,-1,4,5,-8) [returns the raw matrix
     * data] or linear_sys(@(2,3)(3,2,-1,4,5,-8)) [returns a function that
     * points to the solution matrix] or linear_sys(A) [returns a function that
     * points to the solution matrix]
     */
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
    public static final String MATRIX_EIGENVALUES = "eigvalues";
    public static final String MATRIX_EIGENPOLY = "eigpoly";
    public static final String SUB_MATRIX = "sub_mat";
    public static final String RANDOM_MATRIX = "rnd_mat";
    public static final String MATRIX_MINOR = "matrix_minor";
    public static final String FMA = "fma";// fma(A, B, C) -> A * B + C
    public static final String VMA = "vma";// vma(A, B, C) -> A * B + C
    public static final String IF = "if";//if(x > 0.5, x * 2, 0)
    public static final String ERF = "erf";
    public static final String ERFC = "erfc";
    public static final String GELU = "gelu";
    public static final String FAST_GELU = "fast_gelu";
    public static final String SWIGLU = "swiglu";
    public static final String GEGLU = "geglu";
    public static final String HELP = "help";
    
    public static final int ARITY_ZERO = 0;
    public static final int ARITY_ONE = 1;
    public static final int ARITY_TWO = 2;
    public static final int ARITY_THREE = 3;
    public static final int ARITY_N = -10;
    
    /**
     * Returns the number of arguments an inbuilt method expects, expressed with
     * the {@code ARITY_*} constants of this class.
     *
     * <ul>
     * <li>{@link #ARITY_ZERO} - takes no arguments ({@code now()},
     * {@code nanos()}).</li>
     * <li>{@link #ARITY_ONE} - exactly one argument: the scalar functions
     * (trigonometric, hyperbolic, exponential, logarithmic, roots, rounding
     * without a precision, ...), {@code fact}, {@code erf}, {@code gelu}, and
     * the solvers that take a single function ({@code quadratic},
     * {@code t_root}).</li>
     * <li>{@link #ARITY_TWO} - exactly two: {@code pow}, {@code atan2},
     * {@code comb}, {@code perm}, and the "N, value" rounding family
     * ({@code roundN}, {@code floorN}, {@code ceilDigitsN}, ...).</li>
     * <li>{@link #ARITY_THREE} - exactly three: {@code fma}, {@code vma},
     * {@code if}.</li>
     * <li>{@link #ARITY_N} - anything else: methods that accept a variable
     * number of arguments. This covers the statistics methods, the matrix and
     * linear-algebra methods, the calculus / differential-equation / root
     * finding commands, the {@code echo*} family, and every method that is
     * <em>overloaded</em> on argument count, e.g. {@code log(x)} /
     * {@code log(x, base)}, {@code geglu(x)} / {@code geglu(a, b)},
     * {@code swiglu(x)} / {@code swiglu(a, b)}, {@code rnd()} /
     * {@code rnd(n)}. A single int cannot describe "one or two", so such
     * methods report {@code ARITY_N}; use
     * {@link #acceptsInbuiltMethodArgCount(String, int)} for a precise check.
     * Basic numerical methods registered by users carry no arity information
     * and also report {@code ARITY_N}.</li>
     * </ul>
     *
     * The DRG-mode variants produced by
     * {@link #getTrigFuncDRGVariant(String, DRG_MODE)} (e.g. {@code sin_deg})
     * have the arity of the method they were derived from.
     *
     * @param inbuiltMethodName the name of the inbuilt method
     * @return one of {@link #ARITY_ZERO}, {@link #ARITY_ONE},
     * {@link #ARITY_TWO}, {@link #ARITY_THREE} or {@link #ARITY_N}
     * @throws IllegalArgumentException if the name is null or is not the name
     * of an inbuilt method
     */
    public static final int getInbuiltMethodArity(String inbuiltMethodName) {
        if (inbuiltMethodName == null) {
            throw new IllegalArgumentException("Method name must not be null");
        }
        switch (inbuiltMethodName) {
            // ---- no arguments -------------------------------------------------
            case NOW:
            case NANOS:
                return ARITY_ZERO;

            // ---- exactly one argument -----------------------------------------
            case SIN: case COS: case TAN:
            case ARC_SIN: case ARC_COS: case ARC_TAN:
            case ARC_SIN_ALT: case ARC_COS_ALT: case ARC_TAN_ALT:
            case SEC: case COSEC: case COT:
            case ARC_SEC: case ARC_COSEC: case ARC_COT:
            case ARC_SEC_ALT: case ARC_COSEC_ALT: case ARC_COT_ALT:
            case SINH: case COSH: case TANH:
            case ARC_SINH: case ARC_COSH: case ARC_TANH:
            case ARC_SINH_ALT: case ARC_COSH_ALT: case ARC_TANH_ALT:
            case SECH: case COSECH: case COTH:
            case ARC_SECH: case ARC_COSECH: case ARC_COTH:
            case ARC_SECH_ALT: case ARC_COSECH_ALT: case ARC_COTH_ALT:
            case EXP:
            case LN: case LG:
            case LN_INV: case LG_INV:
            case LN_INV_ALT: case LG_INV_ALT:
            case SQRT: case CBRT: case INVERSE: case SQUARE: case CUBE:
            case FACT:
            case ABS: case FLOOR: case CEIL:
            case ERF: case ERFC: case GELU: case FAST_GELU:
            case QUADRATIC: case TARTAGLIA_ROOTS:
            // basic numerical methods (ext package) taking a single number
            case "round":
            case "length": case "lengthDecimal": case "lengthFractional":
                return ARITY_ONE;

            // ---- exactly two arguments ----------------------------------------
            case POW: case ATAN2:
            case COMBINATION: case PERMUTATION:
            // "N, value" rounding family (ext package)
            case "roundN": case "roundX": case "roundDigitsN":
            case "ceilN": case "ceilDigitsN":
            case "floorN": case "floorDigitsN":
                return ARITY_TWO;

            // ---- exactly three arguments --------------------------------------
            case FMA: case VMA:
            case IF:
                return ARITY_THREE;

            // ---- variable / overloaded ----------------------------------------
            // log, log-1, alog, geglu, swiglu, rnd and help are overloaded on
            // the argument count; the rest are genuinely n-ary.
            case LOG: case LOG_INV: case LOG_INV_ALT:
            case GEGLU: case SWIGLU:
            case RANDOM: case HELP:
            case SUM: case LIST_SUM: case PROD: case MEAN: case AVG:
            case MEDIAN: case MODE: case RANGE: case MID_RANGE:
            case ROOT_MEAN_SQUARED: case COEFFICIENT_OF_VARIATION:
            case MIN: case MAX: case STD_DEV: case VARIANCE: case STD_ERR:
            case SORT: case PLOT: case PRINT:
            case DIFFERENTIATION: case AUTO_DIFF: case AUTO_DIFF_N:
            case INTEGRATION: case GENERAL_ROOT: case ROTOR:
            case DIFF_EQN: case DIFF_EQN_PATH:
            case DIFF_EQN_HO: case DIFF_EQN_PATH_HO:
            case LINEAR_SYSTEM: case DETERMINANT: case INVERSE_MATRIX:
            case TRIANGULAR_MATRIX: case ECHELON_MATRIX:
            case MATRIX_MULTIPLY: case MATRIX_DIVIDE: case MATRIX_ADD:
            case MATRIX_SUBTRACT: case MATRIX_POWER: case MATRIX_TRANSPOSE:
            case MATRIX_EDIT: case MATRIX_COFACTORS: case MATRIX_ADJOINT:
            case MATRIX_EIGENVEC: case MATRIX_EIGENVALUES:
            case MATRIX_EIGENPOLY: case SUB_MATRIX: case RANDOM_MATRIX:
            case MATRIX_MINOR:
            // basic numerical methods (ext package) taking a list
            case "echo": case "echon": case "echoi": case "echoni":
            case "avg": case "avgN": case "geom": case "geomN":
            case "count": case "gsum":
                return ARITY_N;

            default:
                break;
        }

        // DRG variants such as "sin_deg", "cos_rad", "atan_grad".
        for (String suffix : DRG_SUFFIXES) {
            if (inbuiltMethodName.endsWith(suffix)) {
                String base = inbuiltMethodName.substring(0, inbuiltMethodName.length() - suffix.length());
                if (!base.isEmpty() && isInbuiltMethodName(base)) {
                    return getInbuiltMethodArity(base);
                }
            }
        }

        // Registered basic numerical methods (including user supplied ones)
        // carry no arity metadata: treat them as variable-arity.
        if (isInbuiltMethodName(inbuiltMethodName)) {
            return ARITY_N;
        }
        throw new IllegalArgumentException("Not an inbuilt method: " + inbuiltMethodName);
    }

    private static final String[] DRG_SUFFIXES = {"_deg", "_rad", "_grad"};

    /**
     * Whether an inbuilt method accepts exactly {@code argCount} arguments.
     * Unlike {@link #getInbuiltMethodArity(String)}, this understands methods
     * overloaded on argument count: {@code log} and {@code log-1} accept 1 or
     * 2, {@code geglu} and {@code swiglu} accept 1 or 2, {@code rnd} accepts 0
     * or 1, {@code help} accepts 0 or 1. Fixed-arity methods accept exactly
     * their arity; every other variable-arity method accepts any non-negative
     * count (their own implementations validate the specifics).
     *
     * @param inbuiltMethodName the name of the inbuilt method
     * @param argCount the number of arguments supplied
     * @return true if the call shape is acceptable
     * @throws IllegalArgumentException if the name is null or not inbuilt
     */
    public static boolean acceptsInbuiltMethodArgCount(String inbuiltMethodName, int argCount) {
        int arity = getInbuiltMethodArity(inbuiltMethodName);
        if (arity != ARITY_N) {
            return argCount == arity;
        }
        switch (inbuiltMethodName) {
            case LOG: case LOG_INV:
            case GEGLU: case SWIGLU:
                return argCount == 1 || argCount == 2;
            case LOG_INV_ALT:
                return argCount == 2; // the implementation reads both arguments
            case RANDOM: case HELP:
                return argCount == 0 || argCount == 1;
            default:
                return argCount >= 0;
        }
    }

    public static final String getTrigFuncDRGVariant(String methodName, DRG_MODE drg) {
        String mode = drg == DRG_MODE.DEG ? "_deg" : (drg == DRG_MODE.RAD ? "_rad" : "_grad");
        switch (methodName) {
            case SIN:
                return SIN + mode;
            case COS:
                return COS + mode;
            case TAN:
                return TAN + mode;
            case ARC_SIN:
                return ARC_SIN + mode;
            case ARC_COS:
                return ARC_COS + mode;
            case ARC_TAN:
                return ARC_TAN + mode;
            case SEC:
                return SEC + mode;
            case COSEC:
                return COSEC + mode;
            case COT:
                return COT + mode;
            case ARC_SEC:
                return ARC_SEC + mode;
            case ARC_COSEC:
                return ARC_COSEC + mode;
            case ARC_COT:
                return ARC_COT + mode;
            case ARC_SIN_ALT:
                return ARC_SIN_ALT + mode;
            case ARC_COS_ALT:
                return ARC_COS_ALT + mode;
            case ARC_TAN_ALT:
                return ARC_TAN_ALT + mode;
            case ARC_SEC_ALT:
                return ARC_SEC_ALT + mode;
            case ARC_COSEC_ALT:
                return ARC_COSEC_ALT + mode;
            case ARC_COT_ALT:
                return ARC_COT_ALT + mode;
            default:
                return methodName;
        }
    }

    /**
     * A list user registered hard-coded methods
     */
    private static final List<BasicNumericalMethod> BASIC_NUMERICAL_METHODS = new ArrayList<>();

    static List<BasicNumericalMethod> getBasicNumericalMethods() {
        return Collections.unmodifiableList(BASIC_NUMERICAL_METHODS);
    }

    // Declared early so it is initialised before any static block can reach it.
    private static final String[] STATS_METHODS = {LIST_SUM, PROD, MEDIAN, MODE, RANGE, MID_RANGE, ROOT_MEAN_SQUARED, COEFFICIENT_OF_VARIATION, MIN, MAX, STD_DEV, VARIANCE, STD_ERR,
        RANDOM, SORT, NANOS, NOW, IF};

    /**
     * Immutable snapshot of everything derived from the registered methods. It
     * used to be rebuilt (and re-sorted) on <em>every</em> call to
     * {@link #getInbuiltMethods()}, which the scanner reaches several times per
     * token via {@code Method.isDefinedMethod}. It is now built lazily once and
     * discarded only when a basic numerical method is (un)registered.
     */
    private static final class Snapshot {

        final String[] sortedNames;
        final Set<String> inbuilt;
        final Set<String> basicNames;

        Snapshot(String[] sortedNames, Set<String> basicNames) {
            this.sortedNames = sortedNames;
            this.inbuilt = new HashSet<>(Arrays.asList(sortedNames));
            this.basicNames = basicNames;
        }
    }

    // Deliberately no initializer: the static block below registers methods and
    // must not be undone by field-initialization order.
    private static volatile Snapshot snapshot;

    private static Snapshot snapshot() {
        Snapshot s = snapshot;
        if (s == null) {
            Set<String> basic = new HashSet<>();
            for (BasicNumericalMethod m : BASIC_NUMERICAL_METHODS) {
                basic.add(m.getName());
            }
            s = new Snapshot(createInBuiltMethods(), basic);
            snapshot = s;
        }
        return s;
    }


    /**
     * Zero-allocation membership test for the names of the core inbuilt
     * methods (same answer as scanning {@link #getInbuiltMethods()}).
     *
     * @param name the candidate method name
     * @return true if it is an inbuilt method name
     */
    public static boolean isInbuiltMethodName(String name) {
        return snapshot().inbuilt.contains(name);
    }

    /**
     * Opaque identity of the current inbuilt-method snapshot. It changes exactly
     * when a basic numerical method is (un)registered, so callers that cache
     * something derived from the inbuilt method names can compare it by identity
     * to know whether their cache is still valid.
     */
    public static Object inbuiltMethodsVersion() {
        return snapshot();
    }

    /**
     * Shared, read-only view of the sorted inbuilt method names. Callers must
     * not modify the array; use {@link #getInbuiltMethods()} for a private copy.
     */
    static String[] inbuiltMethodsShared() {
        return snapshot().sortedNames;
    }

    public static void registerBasicNumericalMethod(BasicNumericalMethod basicNumericalMethod) {
        unregisterBasicNumericalMethod(basicNumericalMethod.getClass());
        BASIC_NUMERICAL_METHODS.add(basicNumericalMethod);
        snapshot = null;
        /**
         * Add to the new method registry also
         */
        MethodRegistry.registerMethod(basicNumericalMethod.getName(), (MathExpression.EvalResult nextRes, int arity, MathExpression.EvalResult[] args) -> {
            List<String> tokens = new ArrayList<>();
            for (MathExpression.EvalResult arg : args) {
                if (arg.type == MathExpression.EvalResult.TYPE_SCALAR) {
                    tokens.add(String.valueOf(arg.scalar));
                } else if (arg.type == MathExpression.EvalResult.TYPE_STRING) {
                    tokens.add(arg.textRes);
                }
            }
            String res = basicNumericalMethod.solve(tokens);
            nextRes.wrap(Double.parseDouble(res));
            return nextRes;
        });
    }

    public static void unregisterBasicNumericalMethod(Class clazz) {
        for (int i = BASIC_NUMERICAL_METHODS.size() - 1; i >= 0; i--) {
            if (BASIC_NUMERICAL_METHODS.get(i).getClass().equals(clazz)) {
                BASIC_NUMERICAL_METHODS.remove(i);
            }
        }
        snapshot = null;
    }

    public static void registerBasicNumericalMethodInMethodRegistry(BasicNumericalMethod basicNumericalMethod) {
        MethodRegistry.registerMethod(basicNumericalMethod.getName(), (MathExpression.EvalResult nextRes, int arity, MathExpression.EvalResult[] args) -> {
            List<String> tokens = new ArrayList<>();
            for (MathExpression.EvalResult arg : args) {
                if (arg.type == MathExpression.EvalResult.TYPE_STRING) {
                    tokens.add(arg.textRes);
                }
            }
            //System.out.println("tokens: "+tokens); 
            nextRes.wrap(basicNumericalMethod.solve(tokens));
            return nextRes;
        });
    }

    static {
        //System.out.println("registering BasicNumeralMethods--------------------------------------------------------------------------------");
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
     *
     * @return Array of strings
     */
    public static String[] getInbuiltMethods() {
        return snapshot().sortedNames.clone();
    }

    /**
     *
     * @return an array containing all builtin methods, basic numeral methods,
     * but not user defined functions
     */
    public static String[] createInBuiltMethods() {
        List<String> stats = Arrays.asList(getStatsMethods());

        String[] functionConstants = {
            SIN, COS, TAN, SINH, COSH, TANH, ARC_SIN, ARC_COS, ARC_TAN,
            ARC_SINH, ARC_COSH, ARC_TANH, SEC, COSEC, COT, SECH, COSECH,
            COTH, ARC_SEC, ARC_COSEC, ARC_COT, ARC_SECH, ARC_COSECH,
            ARC_COTH, EXP, LN, LG, LOG, LN_INV, LG_INV, LOG_INV,
            ARC_SIN_ALT, ARC_COS_ALT, ARC_TAN_ALT, ARC_SINH_ALT,
            ARC_COSH_ALT, ARC_TANH_ALT, ARC_SEC_ALT, ARC_COSEC_ALT,
            ARC_COT_ALT, ARC_SECH_ALT, ARC_COSECH_ALT, ARC_COTH_ALT,
            LN_INV_ALT, LG_INV_ALT, LOG_INV_ALT, SQRT, CBRT, INVERSE,
            ABS, CEIL, FLOOR, IF,
            ERF, ERFC, GELU, GEGLU, SWIGLU, FAST_GELU,
            SQUARE, CUBE, POW, ATAN2, FACT, COMBINATION, PERMUTATION, SUM, LIST_SUM, PROD,
            MEDIAN, MODE, RANGE, MID_RANGE, ROOT_MEAN_SQUARED,
            COEFFICIENT_OF_VARIATION, MIN, MAX, STD_DEV, VARIANCE,
            STD_ERR, RANDOM, SORT, PLOT, PRINT, DIFFERENTIATION, AUTO_DIFF, AUTO_DIFF_N,
            INTEGRATION, 
            DIFF_EQN, DIFF_EQN_PATH, DIFF_EQN_HO, DIFF_EQN_PATH_HO, 
            QUADRATIC, TARTAGLIA_ROOTS, GENERAL_ROOT,
            LINEAR_SYSTEM, DETERMINANT, INVERSE_MATRIX, TRIANGULAR_MATRIX,
            ECHELON_MATRIX, MATRIX_MULTIPLY, MATRIX_DIVIDE, MATRIX_ADD,
            MATRIX_SUBTRACT, MATRIX_POWER, MATRIX_TRANSPOSE, MATRIX_EDIT,
            MATRIX_COFACTORS, MATRIX_ADJOINT, MATRIX_EIGENVEC, MATRIX_EIGENVALUES, SUB_MATRIX, RANDOM_MATRIX, MATRIX_MINOR,
            MATRIX_EIGENPOLY, HELP, NOW, NANOS, ROTOR
        };

        List<String> rest = Arrays.asList(functionConstants);

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
            case ATAN2:
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
            case IF:
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
            case ABS:
                return TYPE.NUMBER.toString();
            case CEIL:
                return TYPE.NUMBER.toString();
            case FLOOR:
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
            case ERF:
                return TYPE.NUMBER.toString();
            case ERFC:
                return TYPE.NUMBER.toString();
            case GELU:
                return TYPE.NUMBER.toString();
            case GEGLU:
                return TYPE.NUMBER.toString();
            case SWIGLU:
                return TYPE.NUMBER.toString();
            case FAST_GELU:
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
                return TYPE.VECTOR.toString();
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
                return TYPE.NUMBER.toString();//rnd() and rnd(n) return one number, not a list
            case SORT:
                return TYPE.VECTOR.toString();
            case SUM:
                return TYPE.NUMBER.toString();
            case LIST_SUM:
                return TYPE.NUMBER.toString();
            case PLOT:
                return TYPE.VOID.toString();
            case DIFFERENTIATION:
                return TYPE.NUMBER.toString();
            case AUTO_DIFF:
                return TYPE.NUMBER.toString();
            case AUTO_DIFF_N:
                return TYPE.VECTOR.toString();

            case INTEGRATION:
                return TYPE.NUMBER.toString();

            case DIFF_EQN:
                return TYPE.NUMBER.toString();
            case DIFF_EQN_PATH:
                return TYPE.MATRIX.toString();
            case DIFF_EQN_HO:
                return TYPE.NUMBER.toString();
            case DIFF_EQN_PATH_HO:
                return TYPE.MATRIX.toString();

            case QUADRATIC:
                return TYPE.VECTOR.toString();
            case TARTAGLIA_ROOTS:
                return TYPE.VECTOR.toString();
            case GENERAL_ROOT:
                return TYPE.NUMBER.toString();
            case LINEAR_SYSTEM:
                return TYPE.VECTOR.toString();
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
            case MATRIX_EIGENVALUES:
                return TYPE.MATRIX.toString();
            case SUB_MATRIX:
                return TYPE.MATRIX.toString();
            case RANDOM_MATRIX:
                return TYPE.MATRIX.toString();
            case MATRIX_MINOR:
                return TYPE.MATRIX.toString();
            case NOW:
                return TYPE.NUMBER.toString();
            case NANOS:
                return TYPE.NUMBER.toString();
            default:
                return TYPE.NUMBER.toString();
        }

    }

    /**
     * @return all the statistical methods used by the parser.
     */
    /**
     * Shared array (package-private, read-only by contract) instead of a fresh
     * copy on every call.
     */
    static String[] getStatsMethods() {
        return STATS_METHODS;
    }

    public static boolean isBasicNumericalFunction(String op) {
        return snapshot().basicNames.contains(op);
    }

    private static final class DrgEnv {

        static final String VALUE = System.getenv(DRG_MODE.DEG_MODE_VARIABLE);
    }

    public static DRG_MODE degGradRadFromVariable() {
        // The process environment cannot change after start-up, so read it once;
        // the system property can, so it is still consulted on every call.
        String userDrgMode = DrgEnv.VALUE;
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
        }
        if (userDrgMode.toUpperCase().trim().equals("GRAD")) {
            return DRG_MODE.GRAD;
        } else {
            return DRG_MODE.RAD;
        }
    }

    public static void main(String[] args) {
        
        System.out.println(new MathExpression("floor(8.72)").solve());
        System.out.println(new MathExpression("ceil(8.72)").solve());
        System.out.println(new MathExpression("round(8.72)").solve());
        createInBuiltMethods();
    }
}