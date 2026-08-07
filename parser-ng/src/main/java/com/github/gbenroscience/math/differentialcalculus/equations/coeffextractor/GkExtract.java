/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor;

/**
 * @author GBEMIRO
 */
import java.math.BigDecimal;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Production-ready coefficient extractor for differential-equation expressions
 * of the form
 *
 * <pre>
 *   A1(t) * y[n] + A2(t) * f1(y[n-1]) + A3(t) * f2(y[n-2]) + ... - B(t)
 * </pre>
 *
 * where {@code y} may be any identifier (y, u, v, x, theta, ...), the
 * derivative order is written with bracket notation ({@code y[k]}), and every
 * additive term below the leading (highest-order) term may be wrapped in an
 * arbitrary function ({@code f(y[n-1])}) — that wrapping function, if present,
 * is retained verbatim as part of the "derivative term" label, not stripped.
 *
 * <p>Only the <em>top-level additive structure</em> of the expression is
 * meaningful for extraction: each additive (+/-) term is inspected, split
 * into "the single multiplicative factor that contains a bracketed
 * derivative atom" (the derivative term) and "everything else multiplied
 * together" (the coefficient). Terms with no derivative atom at all are
 * collected together as the right-hand side ({@code B(t)}, key {@code "RHS"}).
 *
 * <p>Accepts either a raw string or a pre-tokenized ("scanned") list of
 * token strings. Both entry points transparently unwrap a surrounding
 * {@code diffeqn(...)}, {@code diffeqnPath(...)}, {@code diffeqnHO(...)},
 * {@code diffeqnHOPath(...)} (or any other {@code diffeqn*(...)}) call,
 * discarding any trailing arguments (order, path, anonymous-name markers,
 * etc.) after the first top-level comma — regardless of what those trailing
 * arguments look like syntactically.
 *
 * <h2>Thread-safety</h2>
 * This class is stateless (all methods are static, all mutable state is
 * local to a single call) and is safe for concurrent use.
 */
public final class GkExtract {

    private GkExtract() {}

    // -------------------------------------------------------------------------
    // Public result type
    // -------------------------------------------------------------------------

    public static final class ExtractionResult {
        /** Derivative-term label -&gt; coefficient source text. A bare derivative term with no explicit coefficient factor is reported as {@code "1"}. */
        public final Map<String, String> coefficientMap;
        /** Coefficients, in the same order as {@link #derivativeTerms}. */
        public final String[] coefficients;
        /** Derivative-term labels (e.g. {@code "y[3]"}, {@code "sin(y[2])"}, or {@code "RHS"}). */
        public final String[] derivativeTerms;

        ExtractionResult(Map<String, String> map, String[] coeffs, String[] terms) {
            this.coefficientMap  = Collections.unmodifiableMap(new LinkedHashMap<>(map));
            this.coefficients    = coeffs.clone();
            this.derivativeTerms = terms.clone();
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(256);
            sb.append("ExtractionResult{\n");
            for (int i = 0; i < coefficients.length; i++) {
                sb.append("  [").append(i).append("]  coeff = \"")
                  .append(coefficients[i]).append("\"   term = \"")
                  .append(derivativeTerms[i]).append("\"\n");
            }
            sb.append('}');
            return sb.toString();
        }
    }

    // -------------------------------------------------------------------------
    // Public entry points
    // -------------------------------------------------------------------------

    public static ExtractionResult extractFromString(String expression) {
        return extractFromString(expression, null);
    }

    public static ExtractionResult extractFromString(String expression, String dependentVar) {
        Objects.requireNonNull(expression, "expression must not be null");
        String cleaned = expression.trim();
        if (cleaned.isEmpty()) {
            throw new IllegalArgumentException("expression must not be blank");
        }
        List<Token> tokens = stripDiffEqnWrapper(tokenize(cleaned));
        return extractFromTokenList(tokens, dependentVar);
    }

    public static ExtractionResult extractFromTokens(List<String> rawTokens) {
        return extractFromTokens(rawTokens, null);
    }

    public static ExtractionResult extractFromTokens(List<String> rawTokens, String dependentVar) {
        Objects.requireNonNull(rawTokens, "tokens must not be null");
        if (rawTokens.isEmpty()) {
            throw new IllegalArgumentException("token list must not be empty");
        }
        // Re-tokenize the joined scanned form so that spacing/segmentation of the
        // original scan can never desynchronize from our own lexical rules.
        List<Token> tokens = stripDiffEqnWrapper(tokenize(String.join("", rawTokens)));
        return extractFromTokenList(tokens, dependentVar);
    }

    // -------------------------------------------------------------------------
    // Token type
    // -------------------------------------------------------------------------

    private enum TokenType {
        NUMBER, IDENT,
        PLUS, MINUS, MUL, DIV, POW,
        LPAREN, RPAREN, LBRACKET, RBRACKET,
        COMMA, SYMBOL, EOF
    }

    private static final class Token {
        final TokenType type;
        final String text;
        Token(TokenType type, String text) {
            this.type = type;
            this.text = text;
        }
        @Override public String toString() { return type + "(" + text + ")"; }
    }

    // -------------------------------------------------------------------------
    // Tokenizer
    // -------------------------------------------------------------------------
    //
    // NOTE: the tokenizer is deliberately lenient. Any character it does not
    // recognize (e.g. '@', ':', '#') is emitted as a SYMBOL token rather than
    // throwing. This matters because diffeqn(...)/diffeqnHO(...) wrappers can
    // carry trailing arguments (order markers, anonymous-function tags such
    // as "@(1,3)(2,3,4)") whose syntax we never need to understand — we only
    // need the token *stream* to correctly track paren depth while we locate
    // and slice out the first top-level argument. If a SYMBOL token ends up
    // inside the actual expression being parsed (not discarded as part of a
    // wrapper), the parser will reject it with a precise, localized error.
    // -------------------------------------------------------------------------

    private static List<Token> tokenize(String src) {
        List<Token> tokens = new ArrayList<>();
        int i = 0, n = src.length();

        while (i < n) {
            char c = src.charAt(i);
            if (Character.isWhitespace(c)) { i++; continue; }

            // number (decimal + scientific)
            if (Character.isDigit(c) || (c == '.' && i + 1 < n && Character.isDigit(src.charAt(i + 1)))) {
                int start = i++;
                while (i < n && (Character.isDigit(src.charAt(i)) || src.charAt(i) == '.')) i++;
                if (i < n && (src.charAt(i) == 'e' || src.charAt(i) == 'E')) {
                    int save = i;
                    int j = i + 1;
                    if (j < n && (src.charAt(j) == '+' || src.charAt(j) == '-')) j++;
                    if (j < n && Character.isDigit(src.charAt(j))) {
                        i = j;
                        while (i < n && Character.isDigit(src.charAt(i))) i++;
                    } else {
                        i = save; // 'e'/'E' wasn't a real exponent marker; leave it for IDENT lexing
                    }
                }
                tokens.add(new Token(TokenType.NUMBER, src.substring(start, i)));
                continue;
            }

            // identifier
            if (Character.isLetter(c) || c == '_') {
                int start = i++;
                while (i < n && (Character.isLetterOrDigit(src.charAt(i)) || src.charAt(i) == '_')) i++;
                tokens.add(new Token(TokenType.IDENT, src.substring(start, i)));
                continue;
            }

            switch (c) {
                case '+': tokens.add(new Token(TokenType.PLUS, "+")); break;
                case '-': tokens.add(new Token(TokenType.MINUS, "-")); break;
                case '*': case '\u00b7': tokens.add(new Token(TokenType.MUL, "*")); break;
                case '/': tokens.add(new Token(TokenType.DIV, "/")); break;
                case '^': tokens.add(new Token(TokenType.POW, "^")); break;
                case '(': tokens.add(new Token(TokenType.LPAREN, "(")); break;
                case ')': tokens.add(new Token(TokenType.RPAREN, ")")); break;
                case '[': tokens.add(new Token(TokenType.LBRACKET, "[")); break;
                case ']': tokens.add(new Token(TokenType.RBRACKET, "]")); break;
                case ',': tokens.add(new Token(TokenType.COMMA, ",")); break;
                default:  tokens.add(new Token(TokenType.SYMBOL, String.valueOf(c))); break;
            }
            i++;
        }
        tokens.add(new Token(TokenType.EOF, ""));
        return tokens;
    }

    // -------------------------------------------------------------------------
    // diffeqn(...) / diffeqnPath(...) / diffeqnHO(...) / diffeqnHOPath(...) unwrap
    // -------------------------------------------------------------------------
    //
    // Works directly on the token stream (not on re-joined/re-tokenized text),
    // so it behaves identically no matter which public entry point produced
    // the tokens. Locates the first identifier that *starts with* "diffeqn"
    // immediately followed by '(', then extracts everything up to the first
    // top-level comma inside that call (or the matching ')' if there is no
    // comma) as the expression to parse. Trailing arguments — order, path,
    // anonymous-name tags, whatever syntax they use — are discarded whole
    // and never need to be lexically valid on their own.
    // -------------------------------------------------------------------------

    private static List<Token> stripDiffEqnWrapper(List<Token> tokens) {
        for (int i = 0; i < tokens.size() - 1; i++) {
            Token t = tokens.get(i);
            if (t.type == TokenType.IDENT && t.text.startsWith("diffeqn")
                    && tokens.get(i + 1).type == TokenType.LPAREN) {
                int depth = 1;
                int j = i + 2;
                int firstTopComma = -1;
                for (; j < tokens.size() && depth > 0; j++) {
                    TokenType tt = tokens.get(j).type;
                    if (tt == TokenType.LPAREN) depth++;
                    else if (tt == TokenType.RPAREN) depth--;
                    else if (tt == TokenType.COMMA && depth == 1 && firstTopComma < 0) firstTopComma = j;
                }
                if (depth != 0) {
                    throw new IllegalArgumentException("Unbalanced parentheses in diffeqn(...) wrapper");
                }
                int end = (firstTopComma >= 0) ? firstTopComma : (j - 1); // j-1 = matching ')'
                if (end <= i + 2) {
                    throw new IllegalArgumentException("Empty expression inside diffeqn(...) wrapper");
                }
                List<Token> inner = new ArrayList<>(tokens.subList(i + 2, end));
                inner.add(new Token(TokenType.EOF, ""));
                return inner;
            }
        }
        return tokens;
    }

    // -------------------------------------------------------------------------
    // Parser
    // -------------------------------------------------------------------------

    private static final class Parser {
        private final List<Token> tokens;
        private final String restrictedDep;
        private int pos = 0;

        Parser(List<Token> tokens, String restrictedDep) {
            this.tokens = tokens;
            this.restrictedDep = restrictedDep;
        }

        private Token peek()    { return tokens.get(pos); }
        private Token advance() { return tokens.get(pos++); }
        private Token previous() { return pos > 0 ? tokens.get(pos - 1) : new Token(TokenType.EOF, ""); }

        private boolean match(TokenType... types) {
            for (TokenType t : types) {
                if (peek().type == t) { advance(); return true; }
            }
            return false;
        }

        private void expect(TokenType type, String msg) {
            if (peek().type != type)
                throw new IllegalArgumentException(msg + " (found " + peek() + " at position " + pos + ")");
            advance();
        }

        Expr parse() {
            Expr e = parseExpression();
            if (peek().type != TokenType.EOF)
                throw new IllegalArgumentException("Unexpected trailing input: " + peek());
            return e;
        }

        private Expr parseExpression() {
            Expr left = parseTerm();
            while (peek().type == TokenType.PLUS || peek().type == TokenType.MINUS) {
                Token op = advance();
                left = new Binary(op.text, left, parseTerm());
            }
            return left;
        }

        private Expr parseTerm() {
            Expr left = parsePower();
            while (true) {
                if (peek().type == TokenType.MUL || peek().type == TokenType.DIV) {
                    Token op = advance();
                    left = new Binary(op.text, left, parsePower());
                } else if (startsImplicitFactor(peek()) && endsImplicitFactor(previous())) {
                    left = new Binary("*", left, parsePower()); // implicit multiplication
                } else break;
            }
            return left;
        }

        /**
         * Tokens that may OPEN an implicitly-multiplied factor. Deliberately
         * excludes PLUS/MINUS: a '+' or '-' immediately after something that
         * looks like the end of a factor must always be treated as a binary
         * addition/subtraction operator (handled by {@link #parseExpression()}),
         * never as the start of a new implicit-multiplication factor via a
         * unary sign. Blurring this distinction is what silently merges
         * separate additive terms into one bogus product.
         */
        private boolean startsImplicitFactor(Token t) {
            return t.type == TokenType.NUMBER || t.type == TokenType.IDENT || t.type == TokenType.LPAREN;
        }

        /** Tokens that may CLOSE a factor eligible for implicit multiplication. */
        private boolean endsImplicitFactor(Token t) {
            return t.type == TokenType.NUMBER || t.type == TokenType.IDENT
                || t.type == TokenType.RPAREN || t.type == TokenType.RBRACKET;
        }

        private Expr parsePower() {
            Expr left = parseUnary();
            if (match(TokenType.POW)) left = new Binary("^", left, parseUnary());
            return left;
        }

        private Expr parseUnary() {
            if (match(TokenType.PLUS))  return parseUnary();
            if (match(TokenType.MINUS)) return new Unary("-", parseUnary());
            return parsePrimary();
        }

        private Expr parsePrimary() {
            Token t = peek();

            if (t.type == TokenType.NUMBER) {
                advance();
                return new Literal(t.text);
            }

            if (t.type == TokenType.IDENT) {
                advance();
                String name = t.text;

                // derivative atom: ident [ number ]
                if (match(TokenType.LBRACKET)) {
                    if (peek().type != TokenType.NUMBER)
                        throw new IllegalArgumentException("Expected a non-negative integer order inside "
                                + name + "[...] (found " + peek() + ")");
                    String index = advance().text;
                    expect(TokenType.RBRACKET, "Expected ']' to close " + name + "[" + index);
                    if (restrictedDep != null && !restrictedDep.isEmpty() && !name.equals(restrictedDep))
                        return new Ident(name + "[" + index + "]");
                    return new DerivAtom(name + "[" + index + "]");
                }

                // function call
                if (match(TokenType.LPAREN)) {
                    List<Expr> args = new ArrayList<>();
                    if (peek().type != TokenType.RPAREN) {
                        do { args.add(parseExpression()); }
                        while (match(TokenType.COMMA));
                    }
                    expect(TokenType.RPAREN, "Expected ')' to close call to " + name + "(...)");
                    return new FuncCall(name, args);
                }

                return new Ident(name);
            }

            if (match(TokenType.LPAREN)) {
                Expr e = parseExpression();
                expect(TokenType.RPAREN, "Expected ')'");
                return new Paren(e);
            }

            throw new IllegalArgumentException("Unexpected token in primary: " + t + " at position " + pos);
        }
    }

    // -------------------------------------------------------------------------
    // AST nodes
    // -------------------------------------------------------------------------

    private interface Expr {
        String toSource();
        boolean containsDeriv();
    }

    private static final class Literal implements Expr {
        final String value;
        Literal(String v) { value = v; }
        public String toSource() { return value; }
        public boolean containsDeriv() { return false; }
    }

    private static final class Ident implements Expr {
        final String name;
        Ident(String n) { name = n; }
        public String toSource() { return name; }
        public boolean containsDeriv() { return false; }
    }

    private static final class DerivAtom implements Expr {
        final String text;
        DerivAtom(String t) { text = t; }
        public String toSource() { return text; }
        public boolean containsDeriv() { return true; }
    }

    private static final class Unary implements Expr {
        final String op;
        final Expr expr;
        Unary(String op, Expr e) { this.op = op; this.expr = e; }
        public String toSource() { return op + parenIfBinary(expr); }
        public boolean containsDeriv() { return expr.containsDeriv(); }
    }

    private static final class Binary implements Expr {
        final String op;
        final Expr left, right;
        Binary(String op, Expr l, Expr r) { this.op = op; left = l; right = r; }
        public String toSource() { return left.toSource() + op + right.toSource(); }
        public boolean containsDeriv() { return left.containsDeriv() || right.containsDeriv(); }
    }

    private static final class FuncCall implements Expr {
        final String name;
        final List<Expr> args;
        FuncCall(String n, List<Expr> a) { name = n; args = a; }
        public String toSource() {
            StringBuilder sb = new StringBuilder(name).append('(');
            for (int i = 0; i < args.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(args.get(i).toSource());
            }
            return sb.append(')').toString();
        }
        public boolean containsDeriv() {
            for (Expr a : args) if (a.containsDeriv()) return true;
            return false;
        }
    }

    private static final class Paren implements Expr {
        final Expr inner;
        Paren(Expr e) { inner = e; }
        public String toSource() { return "(" + inner.toSource() + ")"; }
        public boolean containsDeriv() { return inner.containsDeriv(); }
    }

    private static String parenIfBinary(Expr e) {
        return (e instanceof Binary) ? "(" + e.toSource() + ")" : e.toSource();
    }

    // -------------------------------------------------------------------------
    // Extraction core
    // -------------------------------------------------------------------------

    private static ExtractionResult extractFromTokenList(List<Token> tokens, String dependentVar) {
        Parser parser = new Parser(tokens, dependentVar);
        Expr root = unwrap(parser.parse());

        List<Expr> additiveTerms = new ArrayList<>();
        flattenSum(root, additiveTerms, false);

        Map<String, String> map = new LinkedHashMap<>();

        for (Expr rawTerm : additiveTerms) {
            Expr term = unwrap(rawTerm);
            TermSplit split = splitTerm(term);

            if (split == null) {
                // No derivative atom anywhere in this additive term -> it belongs to B(t).
                mergeSigned(map, "RHS", term.toSource());
                continue;
            }

            String key   = split.derivative.toSource();
            // An absent coefficient factor means the term is just the bare derivative
            // (e.g. "y[2]" alone) — its implicit coefficient is 1, and we report that
            // explicitly rather than as an empty string, since "" reads as "missing"
            // rather than "one".
            String coeff = (split.coefficient == null) ? "1" : split.coefficient.toSource();

            if (map.containsKey(key)) {
                map.put(key, joinSigned(map.get(key), coeff));
            } else {
                map.put(key, coeff);
            }
        }

        if (map.isEmpty() || (map.size() == 1 && map.containsKey("RHS"))) {
            throw new IllegalArgumentException("No derivative terms (ident[n]) found in expression");
        }

        String[] coeffs = new String[map.size()];
        String[] terms  = new String[map.size()];
        int idx = 0;
        for (Map.Entry<String, String> e : map.entrySet()) {
            terms[idx]  = e.getKey();
            coeffs[idx] = e.getValue();
            idx++;
        }
        return new ExtractionResult(map, coeffs, terms);
    }

    /** Recursively removes outer {@link Paren} wrappers. */
    private static Expr unwrap(Expr e) {
        while (e instanceof Paren) e = ((Paren) e).inner;
        return e;
    }

    private static void flattenSum(Expr e, List<Expr> out, boolean negate) {
        e = unwrap(e);
        if (e instanceof Binary) {
            Binary b = (Binary) e;
            if (b.op.equals("+")) {
                flattenSum(b.left,  out, negate);
                flattenSum(b.right, out, negate);
                return;
            }
            if (b.op.equals("-")) {
                flattenSum(b.left,  out, negate);
                flattenSum(b.right, out, !negate);
                return;
            }
        }
        out.add(negate ? new Unary("-", e) : e);
    }

    private static final class TermSplit {
        final Expr coefficient; // null -> implicit 1
        final Expr derivative;
        TermSplit(Expr c, Expr d) { coefficient = c; derivative = d; }
    }

    private static TermSplit splitTerm(Expr term) {
        List<Expr> factors = new ArrayList<>();
        flattenProduct(term, factors);

        Expr derivFactor = null;
        List<Expr> coeffFactors = new ArrayList<>();

        for (Expr f : factors) {
            if (f.containsDeriv()) {
                if (derivFactor != null) {
                    // Multiple independent derivative-bearing factors multiplied together
                    // in one additive term (e.g. y[2]*y[1]) can't be cleanly split into a
                    // single coefficient * single derivative term — keep the whole additive
                    // term intact as its own key rather than silently dropping information.
                    return new TermSplit(null, term);
                }
                derivFactor = f;
            } else {
                coeffFactors.add(f);
            }
        }

        if (derivFactor == null) return null; // pure RHS contribution

        Expr coeff = coeffFactors.isEmpty() ? null : combineCoeffFactors(coeffFactors);
        return new TermSplit(coeff, derivFactor);
    }

    /**
     * Flattens a product, descending into {@link Unary} minus and {@link Paren}
     * nodes. A leading unary minus becomes the factor literal {@code "-1"}.
     * Division is handled specially (see {@link #flattenDivision}) so that a
     * plain numeric-coefficient division like {@code 5/x} is preserved as a
     * single clean factor instead of being rewritten as {@code 5*1/x}.
     */
    private static void flattenProduct(Expr e, List<Expr> out) {
        e = unwrap(e);

        if (e instanceof Unary) {
            Unary u = (Unary) e;
            if (u.op.equals("-")) {
                out.add(new Literal("-1"));
                flattenProduct(u.expr, out);
                return;
            }
        }

        if (e instanceof Binary) {
            Binary b = (Binary) e;
            if (b.op.equals("*")) {
                flattenProduct(b.left,  out);
                flattenProduct(b.right, out);
                return;
            }
            if (b.op.equals("/")) {
                flattenDivision(b, out);
                return;
            }
        }

        out.add(e);
    }

    private static void flattenDivision(Binary div, List<Expr> out) {
        boolean rightHasDeriv = div.right.containsDeriv();
        boolean leftHasDeriv  = div.left.containsDeriv();

        if (rightHasDeriv) {
            // Derivative in the denominator can't be cleanly turned into a
            // multiplicative coefficient factor — keep the division intact and
            // opaque so it is treated as a single (derivative-bearing) factor.
            out.add(div);
        } else if (leftHasDeriv) {
            // derivative / coefficient  ==  derivative * (1/coefficient)
            flattenProduct(div.left, out);
            out.add(new Binary("/", new Literal("1"), div.right));
        } else {
            // Pure coefficient division (e.g. 5/x): keep as one clean opaque
            // factor rather than exploding it into 5*1/x.
            out.add(div);
        }
    }

    /**
     * Multiplies a list of coefficient factors together, folding any leading
     * run of plain numeric literals into a single simplified literal (e.g.
     * {@code -1} and {@code 3} fold into {@code -3}) instead of leaving
     * {@code -1*3*x} unsimplified.
     */
    private static Expr combineCoeffFactors(List<Expr> factors) {
        List<Expr> merged = new ArrayList<>();
        BigDecimal literalAcc = null;

        for (Expr f : factors) {
            if (f instanceof Literal) {
                BigDecimal v = tryParse(((Literal) f).value);
                if (v != null) {
                    literalAcc = (literalAcc == null) ? v : literalAcc.multiply(v);
                    continue;
                }
            }
            if (literalAcc != null) {
                merged.add(new Literal(formatNumber(literalAcc)));
                literalAcc = null;
            }
            merged.add(f);
        }
        if (literalAcc != null) {
            merged.add(new Literal(formatNumber(literalAcc)));
        }

        Expr result = merged.get(0);
        for (int i = 1; i < merged.size(); i++) {
            result = new Binary("*", result, merged.get(i));
        }
        return result;
    }

    private static BigDecimal tryParse(String numeric) {
        try {
            return new BigDecimal(numeric);
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    private static String formatNumber(BigDecimal v) {
        if (v.compareTo(BigDecimal.ZERO) == 0) return "0";
        return v.stripTrailingZeros().toPlainString();
    }

    /** Joins two already-signed source fragments without producing "+-" or "--". */
    private static String joinSigned(String prev, String next) {
        if (prev == null || prev.isEmpty()) return next;
        if (next == null || next.isEmpty()) return prev;
        if (next.startsWith("-") || next.startsWith("+")) return prev + next;
        return prev + "+" + next;
    }

    private static void mergeSigned(Map<String, String> map, String key, String fragment) {
        map.merge(key, fragment, GkExtract::joinSigned);
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    /** Highest derivative order (the {@code n} in {@code y[n]}) found in the result. */
    public static int highestOrder(ExtractionResult result) {
        int max = -1;
        Pattern p = Pattern.compile("\\[(\\d+)\\]");
        for (String term : result.derivativeTerms) {
            Matcher m = p.matcher(term);
            while (m.find()) max = Math.max(max, Integer.parseInt(m.group(1)));
        }
        return max;
    }

    // -------------------------------------------------------------------------
    // Demo / self-check main
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.println("=== 1. Classic example with y ===");
        String expr1 = "(3x^2)*y[4]+(5*sin(x))*sin(y[3])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0]";
        System.out.println(extractFromString(expr1));
        System.out.println("Highest order = " + highestOrder(extractFromString(expr1)));
        System.out.println();

        System.out.println("=== 2. Different dependent variable (u) ===");
        String expr2 = "u[2] + sin(x)*cos(u[1]) - 4*u[0] + 7";
        System.out.println(extractFromString(expr2));
        System.out.println();

        System.out.println("=== 3. Restricted to a specific name (only v) ===");
        String expr3 = "v[3] + w[1] + 2*v[0] - sin(w[2])";
        System.out.println(extractFromString(expr3, "v"));
        System.out.println();

        System.out.println("=== 4. From a scanned-style token list ===");
        List<String> tokens = Arrays.asList(
                "(", "diffeqn", "(", "(", "(", "3", "*", "x", "^", "2", ")", "*", "y", "[", "4", "]",
                "+", "(", "5", "*", "sin", "(", "x", ")", ")", "*", "sin", "(", "y", "[", "3", "]", ")",
                "+", "(", "5", "/", "x", ")", "*", "ln", "(", "y", "[", "2", "]", ")",
                "-", "3", "*", "y", "[", "1", "]",
                "+", "3", "*", "x", "*", "y", "[", "0", "]",
                ")", ",", "1", ",", "0", ",", "anon1", ")", ")"
        );
        System.out.println(extractFromTokens(tokens));
        System.out.println();

        System.out.println("=== 5. Edge cases (5 + sin(y[0]) must stay separate terms) ===");
        String expr5 = "y[2] - (3*x)*y[1] + 5 + sin(y[0])";
        System.out.println(extractFromString(expr5));
        System.out.println();

        System.out.println("=== 6. Raw string WITH the diffeqn(...) wrapper and a junk trailing arg ===");
        // Mirrors the caller's own example: diffeqn(EXPR, 1, 0, @(1,3)(2,3,4))
        // The tokenizer no longer chokes on '@' and the wrapper is stripped
        // on the string entry point too (previously only the token entry point did this).
        String wrapped = "diffeqn((3x^2)*y[4]+(5*sin(x))*sin(y[3])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0], 1, 0, @(1,3)(2,3,4))";
        System.out.println(extractFromString(wrapped));
        System.out.println();

        System.out.println("=== 7. diffeqnHO / diffeqnPath naming variants also unwrap correctly ===");
        String ho = "diffeqnHOPath(y[2] - 4*y[0] + 9, somePath, 2, anon7)";
        System.out.println(extractFromString(ho));
        
        
        System.out.println("=== 8. diffeqnHO / diffeqnPath naming variants also unwrap correctly ===");
        String ho1 = "diffeqnHOPath((3x^2)*y[4]+(5*sin(x))*sin(y[3])*cos(y[2])+(5/x)*ln(y[2])-3*y[1]+3*x*y[0], somePath, 2, anon7)";
        System.out.println(extractFromString(ho1));
    }
}