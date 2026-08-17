package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common;

import com.github.gbenroscience.math.differentialcalculus.equations.standard.DifferentialEquations;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.parser.TYPE;
import com.github.gbenroscience.util.FunctionManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reads the calling-convention arguments (equation(s), t0, y0, tEnd, h,
 * method, points, presentationStrategy) off a
 * diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call — t0/tEnd/h/method/
 * points/presentationStrategy from {@link Token#getRawArgs()} text (the
 * pragmatic path settled on over re-deriving the same information
 * structurally, per the explicit "if too difficult, get it from
 * token.getRawArgs()" fallback), but y0 from the real compiled {@code
 * Token}, since a bracketed vector literal like {@code (1, 0, 0, 0, 0)}
 * compiles to a single {@code MATRIX}-kind token named {@code anonN} rather
 * than staying literal text — re-splitting rawArgs text on commas breaks
 * the moment y0 contains any expression with its own nested comma (a
 * function call, another vector), and doesn't reflect how ParserNG
 * actually represents it. The real values are read via
 * {@code FunctionManager.lookUp(name).getMatrix().getFlatArray()}. Bracket
 * text-splitting is kept only as a last-resort fallback for a shape that
 * somehow isn't a MATRIX or NUMBER token.
 *
 * <h2>Argument 0: a single equation, or an explicit system</h2>
 * Argument 0 is either a single equation (the classic form, unquoted,
 * unwrapped — returned verbatim as text, this class has no opinion on how
 * it gets isolated or compiled) or an explicit system of equations, given
 * as {@code @(n)("eq1", ..., "eqN")} — an ARRAY-kind literal whose elements
 * are quoted equation strings. Equations can't be given as bare (unquoted)
 * sub-expressions inside the array the way y0's numeric vector elements
 * can, because y[k]-style symbols in equation text aren't evaluable at
 * parse time the way constant numbers are — quoting them as strings is
 * what makes the array a literal, eagerly-resolvable value ParserNG can
 * compile up front, exactly like y0's vector literal already is. Each
 * equation in the array is written LHS-RHS with {@code = 0} omitted, and
 * always divides out the symbol {@code y[n]} where n == the system's
 * component count (== y0.length) — constant across every equation in the
 * system, the same convention a single diffeqn/diffeqnHO equation already
 * uses for its own order. See {@link EquationCoefficientResolver}'s javadoc
 * for how each equation is actually compiled.
 *
 * <h2>What this does NOT parse</h2>
 * The equation text(s) themselves are returned verbatim, as text. This
 * class has no opinion on how they get isolated or compiled into an
 * ODEFunction; that is entirely {@link EquationRuntime}'s job, via {@link
 * PostfixArgumentIsolator} and {@link EquationCoefficientResolver} — see the
 * latter's javadoc for why it's the one open dependency in this pipeline.
 *
 * <h2>Positional argument layout</h2>
 * <pre>
 * diffeqn:        [equation | @(n)("eq1",...,"eqN"), t0, y0, tEnd, h?, method?]
 * diffeqnPath:    [equation | @(n)("eq1",...,"eqN"), t0, y0, tEnd, h?, method?, points?, presentationStrategy?]
 * diffeqnHO:      [equation, t0, y0, tEnd, h?, method?]        (y0 always a vector here; array form not accepted)
 * diffeqnPathHO:  [equation, t0, y0, tEnd, h?, method?, points?, presentationStrategy?]  (array form not accepted)
 * </pre>
 * h and method are optional for every kind; points and presentationStrategy
 * are additionally available on the *_PATH kinds only, and each is
 * INDEPENDENTLY optional — a call may supply neither, just points, just
 * presentationStrategy, or both. Because of that, the trailing argument at
 * index 6 is disambiguated by content rather than assumed to always be
 * points: if it parses as a number it's points (and index 7, if present, is
 * then presentationStrategy); if it doesn't parse as a number it's read as
 * presentationStrategy directly, and points is left at its default. This is
 * what lets {@code diffeqnPathHO(eqn, t0, y0, tEnd, h, method, "state")}
 * work — points omitted, presentationStrategy supplied — without the
 * parser trying (and failing) to read "state" as a number.
 *
 * All optional arguments fall back to a documented default ({@link
 * #DEFAULT_H}, {@link #DEFAULT_METHOD}, {@link
 * #DEFAULT_PRESENTATION_STRATEGY}) when omitted, rather than silently
 * guessing something else — a caller who cares about the exact solver
 * behavior should always pass them all explicitly. presentationStrategy is
 * currently only consumed downstream by diffeqnPathHO; diffeqnPath accepts
 * and parses it for forward compatibility, but nothing reads it yet for
 * that kind.
 */
public final class DiffEqnArgParser {

    /**
     * Applied when the call omits the optional h/initialStep argument.
     */
    public static final double DEFAULT_H = 0.01;

    /**
     * Applied when the call omits the optional method argument.
     */
    public static final ODESolverMethod DEFAULT_METHOD
            = ODESolverMethod.RK4;

    /**
     * Applied when the call omits the optional presentationStrategy argument.
     */
    public static final PresentationStrategy DEFAULT_PRESENTATION_STRATEGY
            = PresentationStrategy.TRAJECTORY;

    private DiffEqnArgParser() {
    }

    public static DiffEqnCall.Kind classify(Token callToken) {
        if (callToken == null || callToken.name == null) {
            throw new IllegalArgumentException("callToken and callToken.name must not be null");
        }
        switch (callToken.name) {
            case "diffeqn":
                return DiffEqnCall.Kind.DIFFEQN;
            case "diffeqnPath":
                return DiffEqnCall.Kind.DIFFEQN_PATH;
            case "diffeqnHO":
                return DiffEqnCall.Kind.DIFFEQN_HO;
            case "diffeqnPathHO":
                return DiffEqnCall.Kind.DIFFEQN_PATH_HO;
            default:
                throw new IllegalArgumentException(
                        "Not a diffeqn-family call: '" + callToken.name + "' — expected one of "
                        + "diffeqn, diffeqnPath, diffeqnHO, diffeqnPathHO.");
        }
    }

    /**
     * @param fullCallPostfix the WHOLE parsed postfix, with the diffeqn-family
     * call as its last token (the postfix root) — needed, not just the call
     * token alone, so y0 and argument 0 (equation or equation array) can each
     * be isolated as their own real {@code Token}(s) (see class javadoc)
     * rather than re-parsed from text.
     * @return
     */
    public static DiffEqnCall parse(Token[] fullCallPostfix) {
        Token callToken = fullCallPostfix[fullCallPostfix.length - 1];
        DiffEqnCall.Kind kind = classify(callToken);
        String[] raw = callToken.getRawArgs();
        int minArgs = 4; // equation(s), t0, y0, tEnd
        if (raw == null || raw.length < minArgs) {
            throw new IllegalArgumentException(
                    kind + " call needs at least " + minArgs + " arguments (equation(s), t0, y0, tEnd), got "
                    + (raw == null ? 0 : raw.length));
        }

        EquationResolution eqRes = resolveEquations(fullCallPostfix, raw[0]);

        double t0 = parseDouble(raw[1], "t0");
        double[] y0 = resolveY0(fullCallPostfix, raw[2]);
        double tEnd = parseDouble(raw[3], "tEnd");

        boolean higherOrderKind = kind == DiffEqnCall.Kind.DIFFEQN_HO || kind == DiffEqnCall.Kind.DIFFEQN_PATH_HO;
        if (higherOrderKind && eqRes.fromArray) {
            throw new IllegalArgumentException(
                    kind + " takes a single higher-order equation, not an equation array — use diffeqn/"
                    + "diffeqnPath with @(n)(\"eq1\", ..., \"eqN\") for an explicit system instead.");
        }
        if (eqRes.fromArray && eqRes.texts.length != y0.length) {
            throw new IllegalArgumentException(
                    "Equation array has " + eqRes.texts.length + " equation(s) but y0 has " + y0.length
                    + " component(s) — for an explicit system these must match one-to-one.");
        }

        double h = raw.length > 4 && !raw[4].isEmpty() ? parseDouble(raw[4], "h") : DEFAULT_H;
        ODESolverMethod method = raw.length > 5 && !raw[5].isEmpty()
                ? parseMethod(raw[5]) : DEFAULT_METHOD;

        boolean pathVariant = kind == DiffEqnCall.Kind.DIFFEQN_PATH || kind == DiffEqnCall.Kind.DIFFEQN_PATH_HO;

        int points = -1;
        PresentationStrategy presentationStrategy = DEFAULT_PRESENTATION_STRATEGY;

        if (pathVariant) {
            String arg6 = raw.length > 6 ? raw[6] : "";
            String arg7 = raw.length > 7 ? raw[7] : "";

            if (!arg6.isEmpty()) {
                Double maybePoints = tryParseDouble(arg6);
                if (maybePoints != null) {
                    // arg6 is numeric -> points. presentationStrategy, if present, is arg7.
                    points = maybePoints.intValue();
                    if (!arg7.isEmpty()) {
                        presentationStrategy = parsePresentationStrategy(arg7);
                    }
                } else {
                    // arg6 isn't numeric -> it's presentationStrategy, and points was omitted.
                    presentationStrategy = parsePresentationStrategy(arg6);
                    if (!arg7.isEmpty()) {
                        throw new IllegalArgumentException(
                                "Unexpected 8th argument '" + arg7 + "' — the 7th argument ('" + arg6
                                + "') was already read as the presentation strategy, so no further "
                                + "trailing arguments are expected. If you meant to supply points, "
                                + "put it before the presentation strategy.");
                    }
                }
            } else if (!arg7.isEmpty()) {
                // points slot explicitly left empty, but presentationStrategy was still supplied.
                presentationStrategy = parsePresentationStrategy(arg7);
            }
        }

        return new DiffEqnCall(kind, eqRes.texts, eqRes.fromArray, t0, y0, tEnd, h, method, points, presentationStrategy);
    }

    private static double parseDouble(String raw, String argName) {
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Could not parse " + argName + " as a number: '" + raw + "'", e);
        }
    }

    /**
     * Like {@link #parseDouble}, but never throws — returns null on failure
     * instead. Used to disambiguate whether a trailing optional argument is
     * numeric (points) or not (presentationStrategy) without relying on a
     * fixed position, since the two are independently optional.
     */
    private static Double tryParseDouble(String raw) {
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** Holds argument 0's resolved equation text(s), plus whether the array syntax was used. */
    private static final class EquationResolution {
        final String[] texts;
        final boolean fromArray;

        EquationResolution(String[] texts, boolean fromArray) {
            this.texts = texts;
            this.fromArray = fromArray;
        }
    }

    /**
     * Resolves argument 0 into one or more equation strings. If it's a single
     * ARRAY-kind token (the @(n)("eq1", ..., "eqN") system syntax), its
     * elements are read back via FunctionManager — the same technique
     * already used for y0's numeric vector literal — since each equation is
     * a quoted string (expression trees can't be eagerly evaluated at parse
     * time the way constant numbers can, so quoting is what makes the array
     * a literal ParserNG can resolve up front). Otherwise falls back to the
     * classic single unquoted-equation form, unchanged.
     */
    private static EquationResolution resolveEquations(Token[] fullCallPostfix, String rawText) {
        Token[] eq0Tokens = PostfixArgumentIsolator.isolateArgument(fullCallPostfix, 0);
        if (eq0Tokens.length == 1 && (eq0Tokens[0].kind == Token.FUNCTION || eq0Tokens[0].kind == Token.FUNCTION_HANDLE) && eq0Tokens[0].functionTokenType == TYPE.ARRAY) {
            Function handle = FunctionManager.lookUp(eq0Tokens[0].name);
            String[] texts = handle.getArray();
            if (texts == null || texts.length == 0) {
                throw new IllegalArgumentException("Equation array must contain at least one equation.");
            }
            return new EquationResolution(texts, true);
        }
        return new EquationResolution(new String[]{rawText}, false);
    }

    /**
     * Resolves y0 (argument index 2) from its real, isolated {@code Token}
     * rather than raw text where possible — see class javadoc for why. A
     * bracketed vector literal compiles to a single {@code MATRIX}-kind token
     * named {@code anonN}: its real values are read via {@code
     * FunctionManager.lookUp(name).getMatrix().getFlatArray()}. A bare scalar
     * compiles to a single {@code NUMBER}-kind token: its value is read
     * directly off the token. Anything else falls back to splitting rawText on
     * commas — a defensive last resort, not the primary path.
     */
    private static double[] resolveY0(Token[] fullCallPostfix, String rawText) {
        Token[] y0Tokens = PostfixArgumentIsolator.isolateArgument(fullCallPostfix, 2);
        if (y0Tokens.length == 1) {
            Token t = y0Tokens[0];
            if (t.kind == Token.MATRIX) {
                Function handle = FunctionManager.lookUp(t.name);
                return handle.getMatrix().getFlatArray();
            }
            if (t.kind == Token.NUMBER) {
                return new double[]{t.value};
            }
        }
        return parseY0FromText(rawText);
    }

    /**
     * Last-resort fallback for the rare case y0's isolated token isn't a plain
     * MATRIX or NUMBER — parses a bare scalar ("1", "1.0") or a
     * parenthesized/bracketed comma-separated LITERAL vector ("(1, 0)", "[1, 0,
     * 0]"). Does not handle a vector containing expressions with their own
     * nested commas; {@link #resolveY0} should always have already handled that
     * case via the real MATRIX token before this runs.
     */
    private static double[] parseY0FromText(String raw) {
        String trimmed = raw.trim();
        boolean bracketed = (trimmed.startsWith("(") && trimmed.endsWith(")"))
                || (trimmed.startsWith("[") && trimmed.endsWith("]"));
        if (!bracketed) {
            return new double[]{parseDouble(trimmed, "y0")};
        }
        String inner = trimmed.substring(1, trimmed.length() - 1);
        String[] parts = inner.split(",");
        List<Double> values = new ArrayList<>(parts.length);
        for (String part : parts) {
            if (!part.isEmpty()) {
                values.add(parseDouble(part, "y0 component"));
            }
        }
        double[] y0 = new double[values.size()];
        for (int i = 0; i < y0.length; i++) {
            y0[i] = values.get(i);
        }
        return y0;
    }

    public static ODESolverMethod parseMethod(String raw) {
        String cleaned = raw.trim();
        if (cleaned.length() >= 2 && cleaned.startsWith("\"") && cleaned.endsWith("\"")) {
            cleaned = cleaned.substring(1, cleaned.length() - 1);
        }
        switch (cleaned.toLowerCase()) {
            case "euler":
                return ODESolverMethod.EULER;
            case "rk4":
                return ODESolverMethod.RK4;
            case "rk45":
                return ODESolverMethod.RK45_DORMAND_PRINCE;
            case "implicit_euler":
                return ODESolverMethod.IMPLICIT_EULER;
            case "bdf2":
                return ODESolverMethod.BDF2;
            default:
                throw new IllegalArgumentException(
                        "Unrecognized method '" + raw + "' — expected one of "
                        + "\"euler\", \"rk4\", \"rk45\", \"implicit_euler\", \"bdf2\".");
        }
    }

    /**
     * Parses the optional presentationStrategy argument. Accepts an
     * unquoted or quoted "state"/"trajectory" (case-insensitive), matching
     * {@link #parseMethod}'s convention for string-valued arguments.
     */
    public static PresentationStrategy parsePresentationStrategy(String raw) {
        String cleaned = raw.trim();
        if (cleaned.length() >= 2 && cleaned.startsWith("\"") && cleaned.endsWith("\"")) {
            cleaned = cleaned.substring(1, cleaned.length() - 1);
        }
        switch (cleaned.toLowerCase()) {
            case "trajectory":
                return PresentationStrategy.TRAJECTORY;
            case "state":
                return PresentationStrategy.STATE;
            default:
                throw new IllegalArgumentException(
                        "Unrecognized presentation strategy '" + raw + "' — expected one of "
                        + "\"trajectory\" or \"state\".");
        }
    }
}