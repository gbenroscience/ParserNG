package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.AggregateSpec;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BetweenExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.ComparisonExpr;
import com.github.gbenroscience.sqlv1.ast.CompOp;
import com.github.gbenroscience.sqlv1.ast.InExpr;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.NotExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Hand-written recursive-descent parser for parser-ng-sql's {@code SELECT}
 * subset.
 *
 * <h2>Grammar</h2>
 * <pre>
 * query
 *     ::= SELECT select_list
 *         FROM table_reference
 *         [WHERE boolean_expression]
 *         [GROUP BY expression_list]
 *         [HAVING boolean_expression]
 *         [ORDER BY order_item (',' order_item)*]
 *         [LIMIT integer_literal]
 * select_list
 *     ::= '*'
 *       | select_item (',' select_item)*
 * select_item
 *     ::= (expression | aggregate_call) [AS identifier]
 * aggregate_call
 *     ::= ('SUM'|'COUNT'|'AVG'|'MIN'|'MAX') '(' (expression | '*') ')'
 *                                              -- '*' only valid for COUNT
 * order_item
 *     ::= expression [ASC | DESC]
 * boolean_expression
 *     ::= or_expression
 * or_expression
 *     ::= and_expression ('OR' and_expression)*
 * and_expression
 *     ::= not_expression ('AND' not_expression)*
 * not_expression
 *     ::= 'NOT' not_expression
 *       | predicate
 * predicate
 *     ::= comparison
 *       | expression 'IS' ['NOT'] 'NULL'
 *       | expression ['NOT'] 'BETWEEN' expression 'AND' expression
 *       | expression ['NOT'] 'IN' '(' expression_list ')'
 *       | '(' boolean_expression ')'
 * comparison
 *     ::= expression comparison_operator expression
 * comparison_operator
 *     ::= '=' | '!=' | '&lt;&gt;' | '&lt;' | '&lt;=' | '&gt;' | '&gt;='
 * expression
 *     ::= additive_expression
 * additive_expression
 *     ::= multiplicative_expression (('+' | '-') multiplicative_expression)*
 * multiplicative_expression
 *     ::= power_expression (('*' | '/' | '%') power_expression)*
 * power_expression
 *     ::= unary_expression ('^' unary_expression)*
 * unary_expression
 *     ::= ('+' | '-') unary_expression
 *       | primary
 * primary
 *     ::= literal | identifier | function_call | case_expression
 *       | cast_expression | '(' expression ')'
 *       | '(' boolean_expression ')'          -- see "Embedded boolean conditions"
 * case_expression
 *     ::= 'CASE' ('WHEN' boolean_expression 'THEN' expression)+
 *         'ELSE' expression 'END'             -- searched CASE; see "CASE / CAST"
 *       | 'CASE' expression ('WHEN' expression 'THEN' expression)+
 *         'ELSE' expression 'END'             -- simple CASE; see "CASE / CAST"
 * cast_expression
 *     ::= 'CAST' '(' expression 'AS' identifier ')'
 *                                              -- see "CASE / CAST"
 * function_call
 *     ::= identifier '(' [expression_list] ')'
 * expression_list
 *     ::= expression (',' expression)*        -- each element may itself be
 *                                                 a boolean_expression; see below
 * literal
 *     ::= integer_literal | decimal_literal | string_literal
 *       | boolean_literal | 'NULL'
 * </pre>
 * This is exactly the grammar requested for parser-ng-sql v1: enough SQL to
 * make {@code SELECT} an expressive, vectorized-Arrow computation language,
 * deliberately stopping short of {@code JOIN}, {@code INSERT},
 * {@code UPDATE}, {@code DELETE}, transactions, or catalogs — plus two
 * amendments to the original {@code expression}/{@code query} productions:
 * "Embedded boolean conditions" (below), which turned out to be too narrow
 * for ParserNG's own idioms, and "{@code GROUP BY}/{@code CASE}/{@code CAST}"
 * (also below), a deliberate, scoped expansion of the original v1 shape.
 *
 * <h2>{@code CASE} / {@code CAST}</h2>
 * Both are handled entirely at the SQL layer, by desugaring into ParserNG
 * text ParserNG already understands — no change to how expressions are
 * captured/compiled downstream (see "Why expressions are captured, not
 * rebuilt" below).
 * <p>
 * Both {@code CASE} forms are supported. <i>Searched</i> {@code CASE}
 * ({@code CASE WHEN cond THEN expr ... ELSE expr END}) parses each
 * {@code WHEN} condition with exactly the same {@link #parseBooleanExpression()}
 * machinery a {@code WHERE} clause uses (normalized to negation-normal
 * form, rendered via {@link BoolExprs#renderFused}) and may not contain
 * {@code IS [NOT] NULL} for the same reason {@link #tryParseNestedCondition()}
 * excludes it (see "Embedded boolean conditions"). <i>Simple</i> {@code CASE}
 * ({@code CASE operand WHEN value THEN expr ... ELSE expr END}) parses an
 * arithmetic {@code operand} expression immediately after {@code CASE}
 * (bounded by {@code WHEN}, exactly like a {@code GROUP BY}/{@code ORDER BY}
 * key — see {@link #buildExpressionText()}'s default case), then builds
 * each branch's condition as {@code operand == value} using ParserNG's own
 * equality operator (see {@code CompOp.EQ}), so {@code CASE cat WHEN 1
 * THEN ... END} is exactly equivalent to writing
 * {@code CASE WHEN cat == 1 THEN ... END} by hand. Both forms require
 * {@code ELSE} (not optional the way standard SQL allows, implicitly
 * defaulting to {@code NULL}): ParserNG has no numeric {@code NULL} literal
 * usable in an arbitrary arithmetic position — a bare {@code NULL} token
 * parses as an ordinary (and always unbound) <em>variable</em> named
 * {@code NULL}, not a literal, confirmed via
 * {@code ArrowExpressionEvaluator#requiredVariableNames()} against a real
 * {@code parser-ng-3.0.7} build — so there is nothing sound to default to.
 * {@link #buildCaseExpressionText()} throws a precise
 * {@link SqlSyntaxException} at parse time if {@code ELSE} is missing,
 * rather than silently compiling something that can only fail later, at
 * evaluate time, with a confusing "missing variable NULL" binding error.
 * The whole construct desugars right-to-left into nested ParserNG
 * {@code if(cond, then, else)} calls — verified against a real
 * {@code parser-ng-arrow} 3.0.7 build to be a genuine three-argument
 * ternary function, not an implicit-multiplication false positive (see
 * this module's {@code ArrowSqlDemo}/README for the verification
 * methodology this project uses for every ParserNG built-in before relying
 * on it).
 * <p>
 * {@code CAST(expr AS type)} supports numeric target types only, since
 * every ParserNG value is already a floating-point number under the hood:
 * {@code INT}/{@code INTEGER}/{@code SMALLINT}/{@code BIGINT}/{@code LONG}
 * truncate toward zero, and {@code DOUBLE}/{@code FLOAT}/{@code REAL}/
 * {@code NUMERIC}/{@code DECIMAL} are a no-op identity (already the
 * representation in use). Any other target type is a compile-time
 * {@link SqlSyntaxException}. The truncation itself deliberately does
 * <em>not</em> use a ParserNG {@code floor}/{@code ceil}/{@code round}/
 * {@code trunc}-named call: none of those exist in a real
 * {@code parser-ng-3.0.7} build — each one either fails to compile at all,
 * or (worse, and the reason this is called out explicitly) silently
 * "succeeds" by parsing as <em>implicit multiplication</em> of an unbound
 * variable of that name against the parenthesized argument (verified via
 * {@code ArrowExpressionEvaluator#requiredVariableNames()}: e.g.
 * {@code "floor(x)"} fails outright with {@code Unknown function: floor},
 * while {@code "trunc(x)"} compiles but reports required variables
 * {@code [trunc, x]} — i.e. it is silently {@code trunc * x}, not a
 * function call at all). Truncation toward zero is instead built from two
 * confirmed-real primitives: {@code expr - (expr % 1)}, verified by direct
 * evaluation to match Java's {@code (long)} cast semantics exactly,
 * including sign handling for negative operands (ParserNG's {@code %} is
 * ordinary remainder — sign follows the dividend, not always non-negative
 * mathematical modulo).
 *
 * <h2>{@code GROUP BY} / aggregates</h2>
 * {@code SUM}/{@code COUNT}/{@code AVG}/{@code MIN}/{@code MAX} are
 * recognized contextually in a {@code select_item} (an identifier matching
 * one of those names, case-insensitively, immediately followed by
 * {@code '('} — see {@link #tryParseAggregate()}) rather than being lexer
 * keywords, so a real column or ParserNG function coincidentally sharing
 * one of those names is never shadowed. Aggregation itself is evaluated
 * entirely in Java by {@code ArrowQuery}, not by ParserNG (see
 * {@link AggFunc}'s javadoc) — this parser's only job is to recognize the
 * call shape and capture the argument text. See
 * {@code SelectStatement}'s javadoc, "{@code GROUP BY} / aggregates — a
 * deliberately strict subset", for the (standard-SQL) restriction this
 * module places on which non-aggregate {@code SELECT} items a grouped query
 * may contain.
 *
 * <h2>Embedded boolean conditions</h2>
 * ParserNG's own expression language natively evaluates comparisons and
 * {@code &&}/{@code ||} as ordinary truthy (0/1) values — that is exactly
 * how idioms like {@code if(sin(x) > 0, tan(x), 0.2)} work: the condition
 * argument is just an expression like any other. A strictly-arithmetic
 * {@code expression} grammar (as originally specified) cannot express that
 * function call at all, since {@code sin(x) > 0} is not itself an
 * {@code expression} — it is a {@code comparison}, one level up in the
 * grammar. That was an oversight, not an intended restriction, and is fixed
 * here: {@code primary}'s parenthesized-grouping form and each element of a
 * {@code function_call}'s {@code expression_list} may now <i>also</i> be a
 * full {@code boolean_expression} (comparisons, {@code AND}/{@code OR}/
 * {@code NOT}, {@code BETWEEN}, {@code IN} — every {@link BoolExpr} leaf
 * type except {@link IsNullExpr}, see below), which is parsed with the
 * exact same machinery as a {@code WHERE} clause
 * ({@link #parseBooleanExpression()}), normalized to negation-normal form,
 * and rendered to ParserNG's native {@code &&}/{@code ||}/{@code ==} text
 * ({@link BoolExprs#renderFused(BoolExpr)}) before being spliced into the
 * surrounding expression text. The same applies to a {@code select_item}'s
 * own top-level {@code expression} — {@code SELECT x > 0 AS flag FROM t} is
 * valid for the same reason.
 *
 * <p>
 * <b>Why this is safe, and where it stops.</b> Trying "is this a boolean
 * condition?" first and falling back to plain arithmetic on failure (see
 * {@link #tryParseNestedCondition()}) is only sound where the attempt is
 * <i>bounded</i> by a delimiter that can never itself be mistaken for a
 * continuation of a {@code boolean_expression} — a function argument or
 * grouping paren's own closing {@code ')'}, an argument list's {@code ','},
 * or a {@code select_item}'s {@code ','}/{@code FROM}. It is deliberately
 * <b>not</b> attempted for a {@code comparison}'s own operands or a
 * {@code BETWEEN}'s bounds ({@link #buildExpressionText()} is used there
 * directly, unchanged): those are bounded by {@code AND}/{@code OR}-adjacent
 * tokens (the {@code AND} inside {@code BETWEEN...AND...}, or whatever
 * {@code AND}/{@code OR} follows a complete predicate in the enclosing
 * {@code WHERE} clause) which — unlike {@code ')'}, {@code ','}, or
 * {@code FROM} — genuinely can continue a {@code boolean_expression}, so a
 * greedy trial parse there would swallow tokens that belong to the
 * <i>outer</i> clause instead of stopping at the intended boundary. Nesting
 * further inside a paren/argument that this position itself opens is still
 * fully supported (e.g. {@code x = (y > 0)} works: the right-hand operand
 * is parsed with {@link #buildExpressionText()}, which recurses into
 * {@link #buildParenGroupText()} for the {@code (y > 0)} term, and that
 * inner position <i>is</i> bounded by its own {@code ')'}).
 *
 * <p>
 * {@link IsNullExpr} is the one {@link BoolExpr} leaf type that is never
 * accepted in an embedded position: it has no ParserNG rendering at all
 * (see its javadoc), so {@link #tryParseNestedCondition()} throws a precise
 * {@link SqlSyntaxException} if a successfully-parsed nested condition
 * turns out to contain one — {@code IS [NOT] NULL} may only be used
 * directly in a {@code WHERE} clause.
 *
 * <h2>Why (the rest of) expressions are captured, not rebuilt</h2>
 * Outside of the embedded-boolean-condition case above, the arithmetic
 * portion of {@code expression} (additive/multiplicative/power/unary/
 * primary/function_call) is — by design — <i>exactly</i> ParserNG's own
 * {@code MathExpression} arithmetic grammar: the same operators
 * ({@code + - * / % ^}), the same parenthesized grouping, the same
 * {@code identifier(args...)} function-call syntax, the same literal forms.
 * {@link #buildExpressionText()} reconstructs this near-verbatim (token
 * text is reassembled with normalized single-space separation rather than
 * re-derived), recursing only where it must — into
 * {@link #buildParenGroupText()} for a parenthesized group and
 * {@link #buildFunctionCallText(String)} for a function call's argument
 * list — specifically so those two constructs can each independently host
 * an embedded boolean condition as described above. Every inbuilt and
 * user-registered ParserNG function is supported automatically and exactly
 * as ParserNG itself defines it; this layer never needs to know a
 * function's name or arity, because it never re-derives arithmetic
 * semantics — it only ever reassembles tokens or splices in already-
 * rendered boolean text.
 *
 * <h2>The leading-{@code '('} ambiguity in {@code predicate}</h2>
 * {@code predicate}'s parenthesized-group alternative
 * ({@code '(' boolean_expression ')'}) and {@code expression}'s own
 * parenthesized-grouping alternative ({@code primary ::= ... | '('
 * expression ')'}) both start with {@code '('}, and cannot be told apart by
 * a fixed amount of lookahead — {@code (x + 1) > 5} must parse as
 * {@code expression '>' expression}, while {@code (x > 5)} must parse as a
 * parenthesized {@code boolean_expression}. {@link #parsePredicate()}
 * resolves this the standard way for a small hand-written parser: it
 * speculatively tries the {@code '(' boolean_expression ')'} reading first,
 * and falls back to {@code expression ...} on failure, restoring the token
 * cursor via a saved mark. This backtracking is bounded to one token of
 * ambiguity (the leading paren) and never recurses unboundedly, since a
 * successful {@code boolean_expression} parse always terminates at a
 * matching {@code ')'}. {@link #buildParenGroupText()} resolves the same
 * ambiguity the same way for a parenthesized group appearing inside an
 * arithmetic expression (e.g. deciding between {@code (y > 0)} and
 * {@code (y + 1)} as a term).
 *
 * @author GBEMIRO
 */
public final class SqlParser {

    private final List<Token> tokens;
    private int pos;

    private SqlParser(List<Token> tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    /**
     * Parses {@code sql} as a single {@code query} per the grammar
     * documented on this class.
     *
     * @throws SqlSyntaxException if {@code sql} does not conform to the
     * grammar (this includes lexical errors from {@link SqlLexer}, and
     * trailing input after a complete, otherwise-valid query)
     */
    public static SelectStatement parse(String sql) {
        List<Token> tokens = SqlLexer.tokenize(sql);
        return new SqlParser(tokens).parseQuery();
    }

    // =====================================================================
    // query / select_list / select_item
    // =====================================================================

    private SelectStatement parseQuery() {
        expect(TokenType.SELECT, "Expected SELECT at the start of a query");

        boolean selectAll = false;
        List<SelectItem> items = new ArrayList<>();

        if (peekType() == TokenType.STAR) {
            advance();
            selectAll = true;
        } else {
            items.add(parseSelectItem());
            while (peekType() == TokenType.COMMA) {
                advance();
                items.add(parseSelectItem());
            }
        }

        expect(TokenType.FROM, "Expected FROM after the select list");
        String table = expectIdentifier("Expected a table/source identifier after FROM");

        BoolExpr where = null;
        if (peekType() == TokenType.WHERE) {
            advance();
            where = BoolExprs.toNnf(parseBooleanExpression());
        }

        List<String> groupBy = new ArrayList<>();
        if (peekType() == TokenType.GROUP) {
            advance();
            expect(TokenType.BY, "Expected BY after GROUP");
            groupBy.add(buildExpressionText());
            while (peekType() == TokenType.COMMA) {
                advance();
                groupBy.add(buildExpressionText());
            }
        }

        BoolExpr having = null;
        if (peekType() == TokenType.HAVING) {
            advance();
            having = BoolExprs.toNnf(parseBooleanExpression());
        }

        List<OrderItem> orderBy = new ArrayList<>();
        if (peekType() == TokenType.ORDER) {
            advance();
            expect(TokenType.BY, "Expected BY after ORDER");
            orderBy.add(parseOrderItem());
            while (peekType() == TokenType.COMMA) {
                advance();
                orderBy.add(parseOrderItem());
            }
        }

        Integer limit = null;
        if (peekType() == TokenType.LIMIT) {
            advance();
            Token n = expect(TokenType.NUMBER, "Expected a non-negative integer after LIMIT");
            if (n.text().indexOf('.') >= 0) {
                throw new SqlSyntaxException("LIMIT must be a plain integer, not '" + n.text() + "'", n.start());
            }
            try {
                limit = Integer.parseInt(n.text());
            } catch (NumberFormatException overflow) {
                throw new SqlSyntaxException("LIMIT value '" + n.text() + "' is out of range", n.start());
            }
        }

        expect(TokenType.EOF, "Unexpected trailing input after a complete query");

        try {
            return new SelectStatement(selectAll, items, table, where, groupBy, having, orderBy, limit);
        } catch (IllegalArgumentException invalidShape) {
            throw new SqlSyntaxException(invalidShape.getMessage(), 0);
        }
    }

    /**
     * Parses one {@code order_item ::= expression [ASC | DESC]}. The
     * expression is bounded by {@code ','}, {@code LIMIT}, or end of input —
     * none of which can continue a {@code boolean_expression} — but an
     * {@code ORDER BY} key is, per the grammar, an ordinary {@code expression},
     * not a full {@code boolean_expression}, so {@link #buildExpressionText()}
     * is used directly here rather than {@link #parseValueExpression()}.
     */
    private OrderItem parseOrderItem() {
        String expr = buildExpressionText();
        boolean descending = false;
        if (peekType() == TokenType.ASC) {
            advance();
        } else if (peekType() == TokenType.DESC) {
            advance();
            descending = true;
        }
        return new OrderItem(expr, descending);
    }

    private SelectItem parseSelectItem() {
        AggregateSpec agg = tryParseAggregate();
        if (agg != null) {
            String alias = null;
            if (peekType() == TokenType.AS) {
                advance();
                alias = expectIdentifier("Expected an alias identifier after AS");
            }
            return SelectItem.aggregate(agg, alias);
        }
        // A select_item's own expression is bounded by ',' or FROM, neither
        // of which can continue a boolean_expression -- safe to try a
        // boolean condition first. See class javadoc, "Embedded boolean
        // conditions".
        String expr = parseValueExpression();
        String alias = null;
        if (peekType() == TokenType.AS) {
            advance();
            alias = expectIdentifier("Expected an alias identifier after AS");
        }
        return SelectItem.plain(expr, alias);
    }

    /**
     * Speculatively parses an {@code aggregate_call} at the current
     * position, or returns {@code null} (restoring the token cursor) if the
     * upcoming tokens do not form one — either the identifier is not one of
     * {@code SUM}/{@code COUNT}/{@code AVG}/{@code MIN}/{@code MAX}
     * (case-insensitively), or it is but is not immediately followed by
     * {@code '('} (a real column or variable that merely happens to share
     * one of those names, used bare — e.g. a column literally named
     * {@code sum}). Contextual disambiguation only, exactly like
     * {@link #tryParseNestedCondition()}; see class javadoc,
     * "{@code GROUP BY} / aggregates".
     */
    private AggregateSpec tryParseAggregate() {
        if (peekType() != TokenType.IDENTIFIER) {
            return null;
        }
        AggFunc func = AggFunc.fromName(peek().text());
        if (func == null) {
            return null;
        }
        int mark = pos;
        advance(); // consume the function-name identifier
        if (peekType() != TokenType.LPAREN) {
            pos = mark;
            return null;
        }
        advance(); // consume '('
        if (func == AggFunc.COUNT && peekType() == TokenType.STAR) {
            advance();
            expect(TokenType.RPAREN, "Expected ')' to close COUNT(*)");
            return new AggregateSpec(AggFunc.COUNT, null, true);
        }
        if (peekType() == TokenType.STAR) {
            error(func + "(*) is not valid; only COUNT(*) is");
        }
        String argExpr = parseValueExpression();
        expect(TokenType.RPAREN, "Expected ')' to close " + func + "(...)");
        return new AggregateSpec(func, argExpr, false);
    }

    // =====================================================================
    // boolean_expression / or_expression / and_expression / not_expression
    // =====================================================================

    private BoolExpr parseBooleanExpression() {
        return parseOrExpression();
    }

    private BoolExpr parseOrExpression() {
        BoolExpr left = parseAndExpression();
        while (peekType() == TokenType.OR) {
            advance();
            BoolExpr right = parseAndExpression();
            left = new OrExpr(left, right);
        }
        return left;
    }

    private BoolExpr parseAndExpression() {
        BoolExpr left = parseNotExpression();
        while (peekType() == TokenType.AND) {
            advance();
            BoolExpr right = parseNotExpression();
            left = new AndExpr(left, right);
        }
        return left;
    }

    private BoolExpr parseNotExpression() {
        if (peekType() == TokenType.NOT) {
            advance();
            return new NotExpr(parseNotExpression());
        }
        return parsePredicate();
    }

    // =====================================================================
    // predicate / comparison
    // =====================================================================

    private BoolExpr parsePredicate() {
        if (peekType() == TokenType.LPAREN) {
            int mark = pos;
            try {
                advance(); // consume '('
                BoolExpr inner = parseBooleanExpression();
                expect(TokenType.RPAREN, "Expected ')' to close a parenthesized boolean expression");
                return inner;
            } catch (SqlSyntaxException backtrack) {
                // Not a "( boolean_expression )" after all -- rewind and
                // fall through to try it as an expression-led predicate
                // instead (e.g. "(x + 1) > 5"). See this class's javadoc,
                // "The leading-'(' ambiguity in predicate".
                pos = mark;
            }
        }

        // Deliberately buildExpressionText(), not parseValueExpression():
        // a predicate's own left operand is bounded by AND/OR-adjacent
        // tokens, not by ')' or ',' -- see class javadoc, "Embedded boolean
        // conditions", on why the boolean-first trial is unsound here.
        String left = buildExpressionText();
        return parsePredicateTail(left);
    }

    private BoolExpr parsePredicateTail(String left) {
        switch (peekType()) {
            case EQ: {
                advance();
                return new ComparisonExpr(left, CompOp.EQ, buildExpressionText());
            }
            case NEQ: {
                advance();
                return new ComparisonExpr(left, CompOp.NEQ, buildExpressionText());
            }
            case LT: {
                advance();
                return new ComparisonExpr(left, CompOp.LT, buildExpressionText());
            }
            case LE: {
                advance();
                return new ComparisonExpr(left, CompOp.LE, buildExpressionText());
            }
            case GT: {
                advance();
                return new ComparisonExpr(left, CompOp.GT, buildExpressionText());
            }
            case GE: {
                advance();
                return new ComparisonExpr(left, CompOp.GE, buildExpressionText());
            }
            case IS: {
                advance();
                boolean negated = false;
                if (peekType() == TokenType.NOT) {
                    advance();
                    negated = true;
                }
                expect(TokenType.NULL, "Expected NULL after IS" + (negated ? " NOT" : ""));
                return new IsNullExpr(left, negated);
            }
            case NOT: {
                advance();
                if (peekType() == TokenType.BETWEEN) {
                    return parseBetweenTail(left, true);
                }
                if (peekType() == TokenType.IN) {
                    return parseInTail(left, true);
                }
                error("Expected BETWEEN or IN after NOT");
                break;
            }
            case BETWEEN: {
                return parseBetweenTail(left, false);
            }
            case IN: {
                return parseInTail(left, false);
            }
            default:
                error("Expected a comparison operator, IS, BETWEEN, or IN");
        }
        throw new AssertionError("unreachable: error() always throws");
    }

    private BoolExpr parseBetweenTail(String left, boolean negated) {
        expect(TokenType.BETWEEN, "Expected BETWEEN");
        // buildExpressionText(), not parseValueExpression(): the low bound
        // is bounded by the literal 'AND' keyword, which can also continue
        // a boolean_expression -- see class javadoc.
        String low = buildExpressionText();
        expect(TokenType.AND, "Expected AND in BETWEEN ... AND ...");
        String high = buildExpressionText();
        return new BetweenExpr(left, low, high, negated);
    }

    private BoolExpr parseInTail(String left, boolean negated) {
        expect(TokenType.IN, "Expected IN");
        expect(TokenType.LPAREN, "Expected '(' after IN");
        // parseValueExpression(): each value is bounded by ',' or the list's
        // own ')', neither of which can continue a boolean_expression --
        // safe to try a boolean condition first, same as a function
        // argument. See class javadoc, "Embedded boolean conditions".
        List<String> values = new ArrayList<>();
        values.add(parseValueExpression());
        while (peekType() == TokenType.COMMA) {
            advance();
            values.add(parseValueExpression());
        }
        expect(TokenType.RPAREN, "Expected ')' to close IN (...)");
        return new InExpr(left, values, negated);
    }

    // =====================================================================
    // expression text construction -- see class javadoc, "Embedded boolean
    // conditions" and "Why (the rest of) expressions are captured, not
    // rebuilt"
    // =====================================================================

    /**
     * Parses one {@code expression}-grammar slot that is safely
     * <i>bounded</i> by a delimiter which can never continue a
     * {@code boolean_expression} (a function argument, an {@code IN}-list
     * value, or a {@code select_item}): tries a full boolean condition
     * first, falling back to plain arithmetic. See class javadoc.
     */
    private String parseValueExpression() {
        String nested = tryParseNestedCondition();
        if (nested != null) {
            return nested;
        }
        return buildExpressionText();
    }

    /**
     * Speculatively parses the upcoming tokens as a complete
     * {@code boolean_expression}, normalizes it to negation-normal form,
     * and renders it to ParserNG text -- or returns {@code null} (restoring
     * the token cursor) if the upcoming tokens do not form a valid boolean
     * condition at all, so the caller can fall back to
     * {@link #buildExpressionText()}.
     *
     * @throws SqlSyntaxException if a boolean condition parses successfully
     * but contains an {@link IsNullExpr} -- that leaf type is never valid
     * in an embedded position (see class javadoc)
     */
    private String tryParseNestedCondition() {
        int mark = pos;
        BoolExpr cond;
        try {
            cond = BoolExprs.toNnf(parseBooleanExpression());
        } catch (SqlSyntaxException notABooleanCondition) {
            pos = mark;
            return null;
        }
        if (BoolExprs.containsIsNull(cond)) {
            throw new SqlSyntaxException(
                    "IS [NOT] NULL cannot be used inside a nested expression, function "
                            + "argument, or grouping -- it has no ParserNG rendering. Use it "
                            + "only directly in a WHERE clause.",
                    tokens.get(mark).start());
        }
        return BoolExprs.renderFused(cond);
    }

    /**
     * Parses one arithmetic {@code expression} (additive/multiplicative/
     * power/unary/primary/function_call), reassembling its token text. Stops
     * without consuming at the first token that can never continue an
     * arithmetic expression at this level: a comparison operator, {@code AS},
     * {@code AND}/{@code OR}/{@code NOT}/{@code BETWEEN}/{@code IN}/
     * {@code IS}, a comma or closing paren belonging to an enclosing
     * construct, {@code FROM}, {@code WHERE}, or end of input.
     */
    private String buildExpressionText() {
        StringBuilder sb = new StringBuilder();
        while (true) {
            switch (peekType()) {
                case IDENTIFIER: {
                    Token idTok = advance();
                    if (peekType() == TokenType.LPAREN) {
                        appendToken(sb, buildFunctionCallText(idTok.text()));
                    } else {
                        appendToken(sb, idTok.text());
                    }
                    break;
                }
                case NUMBER:
                case TRUE:
                case FALSE:
                case NULL:
                    appendToken(sb, advance().text());
                    break;
                case STRING:
                    appendToken(sb, quoteStringLiteral(advance().text()));
                    break;
                case LPAREN:
                    appendToken(sb, buildParenGroupText());
                    break;
                case CASE:
                    appendToken(sb, buildCaseExpressionText());
                    break;
                case CAST:
                    appendToken(sb, buildCastExpressionText());
                    break;
                case PLUS:
                case MINUS:
                case SLASH:
                case STAR:
                case PERCENT:
                case CARET:
                    appendToken(sb, advance().text());
                    break;
                default:
                    // RPAREN, COMMA (both belonging to an enclosing
                    // construct at this level), AS, FROM, WHERE, AND, OR,
                    // NOT, BETWEEN, IN, IS, EQ, NEQ, LT, LE, GT, GE, WHEN,
                    // THEN, ELSE, END, GROUP, BY, HAVING, ORDER, ASC, DESC,
                    // LIMIT, EOF -- none of these can ever legally continue
                    // an arithmetic `expression`.
                    if (sb.length() == 0) {
                        error("Expected an expression");
                    }
                    return sb.toString();
            }
        }
    }

    /**
     * Parses a {@code case_expression} (the {@code CASE} keyword has not
     * yet been consumed) and desugars it into nested ParserNG
     * {@code if(cond, then, else)} calls, right-to-left, so that
     * <pre>{@code
     * CASE WHEN c1 THEN r1 WHEN c2 THEN r2 ELSE r3 END
     * }</pre>
     * becomes {@code (if(c1, r1, if(c2, r2, r3)))}. See class javadoc,
     * "{@code CASE} / {@code CAST}", for why only the searched form is
     * supported, why {@code ELSE} is mandatory, and the verification this
     * desugaring target ({@code if}) was given against a real ParserNG
     * build.
     */
    private String buildCaseExpressionText() {
        Token caseTok = expect(TokenType.CASE, "Expected CASE");
        List<String> conditions = new ArrayList<>();
        List<String> results = new ArrayList<>();

        // A "simple" CASE has an operand expression before the first WHEN;
        // a "searched" CASE goes straight to WHEN. buildExpressionText()
        // safely stops at WHEN (it is not part of the arithmetic-expression
        // continuation set -- see its own default case), so this lookahead
        // is unambiguous with no backtracking needed.
        String operand = null;
        if (peekType() != TokenType.WHEN) {
            operand = buildExpressionText();
        }

        if (peekType() != TokenType.WHEN) {
            error("Expected WHEN in CASE (either 'CASE WHEN cond THEN expr ... ELSE expr END', or "
                    + "'CASE operand WHEN value THEN expr ... ELSE expr END')");
        }
        while (peekType() == TokenType.WHEN) {
            advance();
            String conditionText;
            if (operand != null) {
                // Simple CASE: each WHEN's own expression is compared for
                // equality against the CASE operand -- 'CASE cat WHEN 1
                // THEN ... END' means 'CASE WHEN cat == 1 THEN ... END'.
                // ParserNG's equality operator is "==" (see CompOp.EQ),
                // matching the searched form's own rendering exactly.
                String comparand = buildExpressionText();
                conditionText = "((" + operand + ") == (" + comparand + "))";
            } else {
                BoolExpr cond = BoolExprs.toNnf(parseBooleanExpression());
                if (BoolExprs.containsIsNull(cond)) {
                    throw new SqlSyntaxException(
                            "IS [NOT] NULL cannot be used inside a CASE WHEN condition -- it has no ParserNG "
                                    + "rendering. See SqlParser's class javadoc, \"Embedded boolean conditions\".",
                            caseTok.start());
                }
                conditionText = BoolExprs.renderFused(cond);
            }
            conditions.add(conditionText);
            expect(TokenType.THEN, "Expected THEN after a WHEN condition");
            results.add(parseValueExpression());
        }

        if (peekType() != TokenType.ELSE) {
            throw new SqlSyntaxException(
                    "CASE requires an explicit ELSE branch in this grammar -- there is no numeric NULL "
                            + "literal for a missing ELSE to fall back to. See SqlParser's class javadoc, "
                            + "\"CASE / CAST\".",
                    peek().start());
        }
        advance();
        String elseExpr = parseValueExpression();
        expect(TokenType.END, "Expected END to close CASE");

        String result = elseExpr;
        for (int i = conditions.size() - 1; i >= 0; i--) {
            result = "if(" + conditions.get(i) + ", " + results.get(i) + ", " + result + ")";
        }
        return "(" + result + ")";
    }

    /**
     * Parses a {@code cast_expression} (the {@code CAST} keyword has not
     * yet been consumed) and desugars it into ParserNG text. See class
     * javadoc, "{@code CASE} / {@code CAST}", for exactly which target
     * types are supported and why the truncation formula used for integer
     * targets is built from {@code %} rather than any
     * {@code floor}/{@code ceil}/{@code round}/{@code trunc}-named ParserNG
     * call (none of which exist in a real build).
     */
    private String buildCastExpressionText() {
        expect(TokenType.CAST, "Expected CAST");
        expect(TokenType.LPAREN, "Expected '(' after CAST");
        String inner = parseValueExpression();
        expect(TokenType.AS, "Expected AS inside CAST(... AS type)");
        Token typeTok = expect(TokenType.IDENTIFIER, "Expected a target type name after AS in CAST(... AS type)");
        expect(TokenType.RPAREN, "Expected ')' to close CAST(...)");

        String type = typeTok.text().toUpperCase(Locale.ROOT);
        switch (type) {
            case "INT":
            case "INTEGER":
            case "SMALLINT":
            case "BIGINT":
            case "LONG":
                // Truncation toward zero via (expr) - ((expr) % 1), verified
                // against a real parser-ng-arrow 3.0.7 build to match
                // Java's (long) cast semantics exactly, including sign
                // handling for negative operands.
                return "((" + inner + ") - ((" + inner + ") % 1))";
            case "DOUBLE":
            case "FLOAT":
            case "REAL":
            case "NUMERIC":
            case "DECIMAL":
                return "(" + inner + ")";
            default:
                throw new SqlSyntaxException(
                        "Unsupported CAST target type '" + typeTok.text() + "' -- parser-ng-sql only supports "
                                + "numeric CAST targets (INT/INTEGER/BIGINT/SMALLINT/LONG or "
                                + "DOUBLE/FLOAT/REAL/NUMERIC/DECIMAL), since every ParserNG value is already "
                                + "a floating-point number under the hood.",
                        typeTok.start());
        }
    }

    /**
     * Parses a parenthesized {@code primary}: either a nested
     * {@code boolean_expression} (see class javadoc, "Embedded boolean
     * conditions") or a plain grouped {@code expression} -- tries the
     * former first, falling back to the latter, exactly mirroring
     * {@link #parsePredicate()}'s own leading-{@code '('} disambiguation.
     */
    private String buildParenGroupText() {
        expect(TokenType.LPAREN, "Expected '('");
        String nested = tryParseNestedCondition();
        if (nested != null) {
            expect(TokenType.RPAREN, "Expected ')' to close a parenthesized condition");
            // BoolExprs.renderFused always returns already self-parenthesized
            // text (every leaf/And/Or render wraps itself), so returning it
            // as-is here avoids a redundant "((...))" double-wrap.
            return nested;
        }
        String inner = buildExpressionText();
        expect(TokenType.RPAREN, "Expected ')' to close a parenthesized expression");
        return "(" + inner + ")";
    }

    /**
     * Parses a {@code function_call}'s {@code '(' [expression_list] ')'}
     * tail (the identifier itself, {@code name}, has already been
     * consumed). Each argument is parsed with {@link #parseValueExpression()}
     * so it may independently be a plain value or a full boolean condition
     * -- see class javadoc, "Embedded boolean conditions".
     */
    private String buildFunctionCallText(String name) {
        expect(TokenType.LPAREN, "Expected '(' after function name '" + name + "'");
        StringBuilder sb = new StringBuilder(name).append('(');
        if (peekType() != TokenType.RPAREN) {
            sb.append(parseValueExpression());
            while (peekType() == TokenType.COMMA) {
                advance();
                sb.append(", ").append(parseValueExpression());
            }
        }
        expect(TokenType.RPAREN, "Expected ')' to close call to '" + name + "'");
        sb.append(')');
        return sb.toString();
    }

    /**
     * Re-quotes a {@link Token#text()} of type {@link TokenType#STRING}
     * (which {@link SqlLexer} stores already unescaped/unquoted) back into
     * ParserNG-safe single-quoted text, doubling any embedded {@code '}
     * exactly as {@link SqlLexer} itself accepts on the way in.
     */
    private static String quoteStringLiteral(String decoded) {
        return "'" + decoded.replace("'", "''") + "'";
    }

    private static void appendToken(StringBuilder sb, String text) {
        if (sb.length() > 0 && sb.charAt(sb.length() - 1) != '(') {
            sb.append(' ');
        }
        sb.append(text);
    }

    // =====================================================================
    // token-stream plumbing
    // =====================================================================

    private Token peek() {
        return tokens.get(pos);
    }

    private TokenType peekType() {
        return tokens.get(pos).type();
    }

    private Token advance() {
        return tokens.get(pos++);
    }

    private Token expect(TokenType type, String message) {
        if (peekType() != type) {
            error(message);
        }
        return advance();
    }

    private String expectIdentifier(String message) {
        return expect(TokenType.IDENTIFIER, message).text();
    }

    private void error(String message) {
        Token t = peek();
        throw new SqlSyntaxException(
                message + "; found " + t.type() + " '" + t.text() + "'", t.start());
    }
}
