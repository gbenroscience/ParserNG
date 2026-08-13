package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common;

import com.github.gbenroscience.math.differentialcalculus.equations.standard.DifferentialEquations;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.parser.MathExpression.Token;
import com.github.gbenroscience.util.FunctionManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reads the calling-convention arguments (t0, y0, tEnd, h, method, points) off
 * a diffeqn/diffeqnPath/diffeqnHO/diffeqnPathHO call — t0/tEnd/h/method/ points
 * from {@link Token#getRawArgs()} text (the pragmatic path settled on over
 * re-deriving the same information structurally, per the explicit "if too
 * difficult, get it from token.getRawArgs()" fallback), but y0 from the real
 * compiled {@code Token}, since a bracketed vector literal like
 * {@code (1, 0, 0, 0, 0)} compiles to a single {@code MATRIX}-kind token named
 * {@code anonN} rather than staying literal text — re-splitting rawArgs text on
 * commas breaks the moment y0 contains any expression with its own nested comma
 * (a function call, another vector), and doesn't reflect how ParserNG actually
 * represents it. The real values are read via
 * {@code FunctionManager.lookUp(name).getMatrix().getFlatArray()}. Bracket
 * text-splitting is kept only as a last-resort fallback for a shape that
 * somehow isn't a MATRIX or NUMBER token.
 *
 * <h2>What this does NOT parse</h2>
 * rawArgs[0] — the raw equation itself, already rearranged to {@code
 * LHS-RHS} with the {@code =} sign omitted — is returned verbatim, as text.
 * This class has no opinion on how it gets isolated or compiled into an
 * ODEFunction; that is entirely {@link EquationRuntime}'s job, via {@link
 * PostfixArgumentIsolator} and {@link EquationCoefficientResolver} — see the
 * latter's javadoc for why it's the one open dependency in this pipeline.
 *
 * <h2>Positional argument layout</h2>
 * <pre>
 * diffeqn:        [equation, t0, y0, tEnd, h?, method?]
 * diffeqnPath:     [equation, t0, y0, tEnd, h?, method?, points?]
 * diffeqnHO:       [equation, t0, y0, tEnd, h?, method?]        (y0 always a vector here)
 * diffeqnPathHO:   [equation, t0, y0, tEnd, h?, method?, points?]
 * </pre> h and method are optional per the calling convention; when omitted
 * this class applies a documented default ({@link #DEFAULT_H}, {@link
 * #DEFAULT_METHOD}) rather than silently guessing something else — a caller who
 * cares about the exact solver behavior should always pass both.
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
     * token alone, so y0 can be isolated as its own real {@code Token} (see
     * class javadoc) rather than re-parsed from text.
     * @return
     */
    public static DiffEqnCall parse(Token[] fullCallPostfix) {
        Token callToken = fullCallPostfix[fullCallPostfix.length - 1];
        DiffEqnCall.Kind kind = classify(callToken);
        String[] raw = callToken.getRawArgs();
        int minArgs = 4; // equation, t0, y0, tEnd
        if (raw == null || raw.length < minArgs) {
            throw new IllegalArgumentException(
                    kind + " call needs at least " + minArgs + " arguments (equation, t0, y0, tEnd), got "
                    + (raw == null ? 0 : raw.length));
        }

        String rhsText = raw[0];
        double t0 = parseDouble(raw[1], "t0");
        double[] y0 = resolveY0(fullCallPostfix, raw[2]);
        double tEnd = parseDouble(raw[3], "tEnd");
        System.out.println("args: " + Arrays.toString(raw));

        double h = raw.length > 4 && !raw[4].isEmpty() ? parseDouble(raw[4], "h") : DEFAULT_H;
        ODESolverMethod method = raw.length > 5 && !raw[5].isEmpty()
                ? parseMethod(raw[5]) : DEFAULT_METHOD;

        int points = -1;
        boolean pathVariant = kind == DiffEqnCall.Kind.DIFFEQN_PATH || kind == DiffEqnCall.Kind.DIFFEQN_PATH_HO;
        if (pathVariant && raw.length > 6 && !raw[6].isEmpty()) {
            points = (int) parseDouble(raw[6], "points");
        }

        return new DiffEqnCall(kind, rhsText, t0, y0, tEnd, h, method, points);
    }

    private static double parseDouble(String raw, String argName) {
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Could not parse " + argName + " as a number: '" + raw + "'", e);
        }
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
        System.out.println("raw: " + raw);
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
            default:
                throw new IllegalArgumentException(
                        "Unrecognized method '" + raw + "' — expected one of "
                        + "\"euler\", \"rk4\", \"rk45\", \"implicit_euler\".");
        }
    }
}
