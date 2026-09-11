package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.AggregateKind;
import com.github.gbenroscience.sqlv1.ast.AndExpr;
import com.github.gbenroscience.sqlv1.ast.BetweenExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.ComparisonExpr;
import com.github.gbenroscience.sqlv1.ast.CompOp;
import com.github.gbenroscience.sqlv1.ast.InExpr;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.NotExpr;
import com.github.gbenroscience.sqlv1.ast.OrExpr;
import com.github.gbenroscience.sqlv1.ast.OrderItem;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

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
 *     ::= expression [AS identifier]
 *       | aggregate_call [AS identifier]
 * aggregate_call
 *     ::= ('COUNT' | 'SUM' | 'AVG' | 'MIN' | 'MAX') '(' (expression | '*') ')'
 *                                                -- '*' only valid with COUNT;
 *                                                   see "Aggregates and GROUP BY"
 * order_item
 *     ::= expression ['ASC' | 'DESC']
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
 *     ::= literal | identifier | function_call | case_expression | cast_expression
 *       | '(' expression ')'
 *       | '(' boolean_expression ')'          -- see "Embedded boolean conditions"
 * function_call
 *     ::= identifier '(' [expression_list] ')'
 * case_expression
 *     ::= 'CASE' expression ('WHEN' expression 'THEN' expression)+ ['ELSE' expression] 'END'
 *                                                -- "simple" CASE; see "CASE/WHEN/THEN/ELSE/END"
 *       | 'CASE' ('WHEN' boolean_expression 'THEN' expression)+ ['ELSE' expression] 'END'
 *                                                -- "searched" CASE (no operand before the
 *                                                   first WHEN)
 * cast_expression
 *     ::= 'CAST' '(' expression 'AS' identifier ')'
 * expression_list
 *     ::= expression (',' expression)*        -- each element may itself be
 *                                                 a boolean_expression; see below
 * literal
 *     ::= integer_literal | decimal_literal | string_literal
 *       | boolean_literal | 'NULL'
 * </pre>
 * This is the grammar for parser-ng-sql: enough SQL to make {@code SELECT}
 * an expressive, vectorized-Arrow computation language, deliberately
 * stopping short of {@code JOIN}, {@code INSERT}, {@code UPDATE},
 * {@code DELETE}, transactions, or catalogs — plus two amendments to the
 * original {@code expression} production (below).
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
 * own top-level {@code expression}, and to a {@code CASE} branch's
 * {@code THEN}/{@code ELSE} expression — {@code SELECT x > 0 AS flag FROM t}
 * and {@code CASE WHEN a THEN b > 0 ELSE c < 0 END} are both valid for the
 * same reason.
 *
 * <p>
 * <b>Why this is safe, and where it stops.</b> Trying "is this a boolean
 * condition?" first and falling back to plain arithmetic on failure (see
 * {@link #tryParseNestedCondition()}) is only sound where the attempt is
 * <i>bounded</i> by a delimiter that can never itself be mistaken for a
 * continuation of a {@code boolean_expression} — a function argument or
 * grouping paren's own closing {@code ')'}, an argument list's {@code ','},
 * a {@code select_item}'s {@code ','}/{@code FROM}, or a {@code CASE}
 * branch's {@code WHEN}/{@code ELSE}/{@code END}. It is deliberately
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
 * directly in a {@code WHERE}/{@code HAVING} clause, and the same
 * restriction applies to a searched {@code CASE}'s {@code WHEN} conditions
 * (see {@link #buildCaseExpressionText()}).
 *
 * <h2>{@code CASE}/{@code WHEN}/{@code THEN}/{@code ELSE}/{@code END}</h2>
 * ParserNG has no {@code CASE} operator of its own, but does have the
 * {@code if(cond, then, else)} idiom described above. A {@code CASE}
 * expression compiles to a right-nested chain of {@code if(...)} calls
 * ({@link #buildCaseExpressionText()}): the {@code n}-th {@code WHEN}
 * becomes the {@code n}-th nesting level's condition/then-branch, and the
 * innermost {@code else}-branch is either the {@code ELSE} expression or,
 * if none was written, the literal {@code NULL} (already an accepted
 * arithmetic-expression literal — see {@link #buildExpressionText()}'s
 * {@code NULL} case). Both grammar forms are supported: <i>searched</i>
 * {@code CASE} (no operand — each {@code WHEN} carries its own full
 * {@code boolean_expression} condition, parsed and rendered exactly like a
 * {@code WHERE} clause, including the {@link IsNullExpr} restriction above)
 * and <i>simple</i> {@code CASE} (an operand expression immediately after
 * {@code CASE}, compared with {@code =} against each {@code WHEN}'s own
 * expression to build that branch's condition).
 *
 * <h2>{@code CAST}</h2>
 * {@code CAST(expr AS type)} has no ParserNG operator either, and — since
 * every value passing through this module is already a ParserNG/Arrow
 * {@code float64} or {@code float32} (see {@code ArrowQuery#isFloat64}) —
 * there is no storage-representation change to perform. {@link #renderCastText}
 * recognizes two families of target type name (matched case-insensitively,
 * exactly like a keyword, even though lexically it is just an identifier —
 * see {@link SqlLexer}):
 * <ul>
 * <li>{@code INT}/{@code INTEGER}/{@code SMALLINT}/{@code BIGINT}/{@code LONG}
 * — truncates toward zero, rendered as
 * {@code if(expr >= 0, floor(expr), -1 * floor(-1 * (expr)))} (assumes
 * ParserNG provides a {@code floor} function, as a general-purpose math
 * expression language would).</li>
 * <li>{@code FLOAT}/{@code REAL}/{@code DOUBLE}/{@code DECIMAL}/{@code NUMERIC}
 * — a pure no-op, rendered as {@code (expr)}, since the value is already
 * floating-point.</li>
 * </ul>
 * Any other target type name is rejected with a {@link SqlSyntaxException}
 * at parse time.
 *
 * <h2>Aggregates and {@code GROUP BY}</h2>
 * {@code COUNT}/{@code SUM}/{@code AVG}/{@code MIN}/{@code MAX} are
 * recognized only at the top level of a {@code select_item} — an identifier
 * spelling one of those names (case-insensitively) immediately followed by
 * {@code '('} ({@link #parseSelectItem()}) — never nested inside a larger
 * expression, matching standard SQL's prohibition on nested aggregates. The
 * function's single argument is parsed with {@link #parseValueExpression()}
 * (so it may itself embed a boolean condition, {@code CASE}, or {@code CAST}),
 * except for {@code COUNT(*)}, the one aggregate call that takes a bare
 * {@code '*'} instead. {@code GROUP BY}'s own expression list uses
 * {@link #buildExpressionText()} directly (a {@code GROUP BY} key is always
 * a plain arithmetic expression, e.g. a column) — see
 * {@code SelectStatement}'s javadoc for the rule that every non-aggregate
 * {@code select_item} must match one of these expressions verbatim, and
 * {@code ArrowQuery}'s "Aggregation strategy" for how grouping and the
 * aggregate functions are actually computed.
 *
 * <h2>{@code HAVING} and {@code ORDER BY}</h2>
 * Both are parsed with exactly the machinery already used for {@code WHERE}
 * ({@link #parseBooleanExpression()} for {@code HAVING}; {@link #buildExpressionText()}
 * per key for {@code ORDER BY}) — see {@code ArrowQuery} for how each is
 * evaluated once the query is compiled against a concrete schema, including
 * the recommendation to alias an aggregate {@code select_item} referenced
 * from either clause.
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
 * {@link #buildParenGroupText()} for a parenthesized group,
 * {@link #buildFunctionCallText(String)} for a function call's argument
 * list, {@link #buildCaseExpressionText()} for a {@code CASE} expression,
 * and {@link #buildCastExpressionText()} for a {@code CAST} expression —
 * specifically so those constructs can each independently host an embedded
 * boolean condition as described above. Every inbuilt and user-registered
 * ParserNG function is supported automatically and exactly as ParserNG
 * itself defines it; this layer never needs to know a function's name or
 * arity (except for the aggregate names and {@code CASE}/{@code CAST}
 * themselves, which are structural keywords/recognized names, not ordinary
 * ParserNG functions) — it never re-derives arithmetic semantics, it only
 * ever reassembles tokens or splices in already-rendered text.
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

    private static final Set<String> AGGREGATE_FUNCTION_NAMES = Set.of("COUNT", "SUM", "AVG", "MIN", "MAX");

    private static final Set<String> CAST_TRUNCATING_TYPES =
            Set.of("INT", "INTEGER", "SMALLINT", "BIGINT", "LONG");

    private static final Set<String> CAST_NOOP_TYPES =
            Set.of("FLOAT", "REAL", "DOUBLE", "DECIMAL", "NUMERIC");

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
     * @throws IllegalArgumentException if {@code sql} is structurally valid
     * but violates an aggregate/{@code GROUP BY} rule enforced by
     * {@link SelectStatement}'s constructor (e.g. a plain select-list
     * column absent from {@code GROUP BY})
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

        List<String> groupBy = List.of();
        if (peekType() == TokenType.GROUP) {
            advance();
            expect(TokenType.BY, "Expected BY after GROUP");
            List<String> gb = new ArrayList<>();
            gb.add(buildExpressionText());
            while (peekType() == TokenType.COMMA) {
                advance();
                gb.add(buildExpressionText());
            }
            groupBy = gb;
        }

        BoolExpr having = null;
        if (peekType() == TokenType.HAVING) {
            advance();
            having = BoolExprs.toNnf(parseBooleanExpression());
        }

        List<OrderItem> orderBy = List.of();
        if (peekType() == TokenType.ORDER) {
            advance();
            expect(TokenType.BY, "Expected BY after ORDER");
            List<OrderItem> ob = new ArrayList<>();
            ob.add(parseOrderItem());
            while (peekType() == TokenType.COMMA) {
                advance();
                ob.add(parseOrderItem());
            }
            orderBy = ob;
        }

        Integer limit = null;
        if (peekType() == TokenType.LIMIT) {
            advance();
            limit = parseLimitValue();
        }

        expect(TokenType.EOF, "Unexpected trailing input after a complete query");

        return new SelectStatement(selectAll, items, table, where, groupBy, having, orderBy, limit);
    }

    private SelectItem parseSelectItem() {
        if (peekType() == TokenType.IDENTIFIER) {
            String upper = peek().text().toUpperCase(Locale.ROOT);
            if (AGGREGATE_FUNCTION_NAMES.contains(upper) && peekTypeAt(1) == TokenType.LPAREN) {
                return parseAggregateSelectItem(upper);
            }
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
        return new SelectItem(expr, alias);
    }

    /**
     * Parses an {@code aggregate_call}'s {@code '(' (expression | '*') ')'}
     * tail (the function-name identifier, already confirmed to be one of
     * {@link #AGGREGATE_FUNCTION_NAMES} followed by {@code '('}, is
     * consumed here) and its optional {@code AS} alias. See class javadoc,
     * "Aggregates and GROUP BY".
     */
    private SelectItem parseAggregateSelectItem(String funcNameUpper) {
        Token nameTok = advance(); // the function-name identifier
        advance(); // '('
        AggregateKind kind = AggregateKind.valueOf(funcNameUpper);

        boolean star = false;
        String argText = null;
        if (kind == AggregateKind.COUNT && peekType() == TokenType.STAR) {
            advance();
            star = true;
        } else if (peekType() == TokenType.STAR) {
            error("'*' is only valid as the argument to COUNT");
        } else {
            argText = parseValueExpression();
        }
        expect(TokenType.RPAREN, "Expected ')' to close call to '" + nameTok.text() + "'");

        String canonicalText = nameTok.text() + "(" + (star ? "*" : argText) + ")";

        String alias = null;
        if (peekType() == TokenType.AS) {
            advance();
            alias = expectIdentifier("Expected an alias identifier after AS");
        }
        return SelectItem.aggregate(kind, star, argText, canonicalText, alias);
    }

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

    private Integer parseLimitValue() {
        Token n = expect(TokenType.NUMBER, "Expected a non-negative integer after LIMIT");
        if (n.text().indexOf('.') >= 0) {
            throw new SqlSyntaxException("LIMIT requires a plain integer, not '" + n.text() + "'", n.start());
        }
        try {
            return Integer.valueOf(n.text());
        } catch (NumberFormatException tooLarge) {
            throw new SqlSyntaxException("LIMIT value is too large: '" + n.text() + "'", n.start());
        }
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
     * value, a {@code select_item}, or a {@code CASE} branch's
     * {@code THEN}/{@code ELSE} expression): tries a full boolean condition
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
                            + "only directly in a WHERE/HAVING clause.",
                    tokens.get(mark).start());
        }
        return BoolExprs.renderFused(cond);
    }

    /**
     * Parses one arithmetic {@code expression} (additive/multiplicative/
     * power/unary/primary/function_call/case_expression/cast_expression),
     * reassembling its token text. Stops without consuming at the first
     * token that can never continue an arithmetic expression at this
     * level: a comparison operator, {@code AS}, {@code AND}/{@code OR}/
     * {@code NOT}/{@code BETWEEN}/{@code IN}/{@code IS}, a comma or closing
     * paren belonging to an enclosing construct, {@code FROM}, {@code WHERE},
     * {@code GROUP}, {@code HAVING}, {@code ORDER}, {@code LIMIT},
     * {@code WHEN}/{@code THEN}/{@code ELSE}/{@code END} belonging to an
     * enclosing {@code CASE}, or end of input.
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
                    // construct at this level), AS, FROM, WHERE, GROUP, BY,
                    // HAVING, ORDER, ASC, DESC, LIMIT, AND, OR, NOT,
                    // BETWEEN, IN, IS, WHEN, THEN, ELSE, END, EQ, NEQ, LT,
                    // LE, GT, GE, EOF -- none of these can ever legally
                    // continue an arithmetic `expression`.
                    if (sb.length() == 0) {
                        error("Expected an expression");
                    }
                    return sb.toString();
            }
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
     * Parses a {@code case_expression} (the leading {@code CASE} has not
     * yet been consumed) and renders it to a right-nested chain of
     * ParserNG {@code if(cond, then, else)} calls. See class javadoc,
     * "CASE/WHEN/THEN/ELSE/END".
     */
    private String buildCaseExpressionText() {
        Token caseTok = expect(TokenType.CASE, "Expected CASE");

        // A "simple" CASE has an operand expression before the first WHEN;
        // a "searched" CASE goes straight to WHEN. buildExpressionText()
        // safely stops at WHEN (it is not part of the arithmetic-expression
        // continuation set), so this lookahead is unambiguous.
        String operand = null;
        if (peekType() != TokenType.WHEN) {
            operand = buildExpressionText();
        }

        List<String> conditions = new ArrayList<>();
        List<String> thenBranches = new ArrayList<>();
        while (peekType() == TokenType.WHEN) {
            advance();
            String conditionText;
            if (operand != null) {
                String comparand = buildExpressionText();
                conditionText = "(" + operand + " == " + comparand + ")";
            } else {
                BoolExpr cond = BoolExprs.toNnf(parseBooleanExpression());
                if (BoolExprs.containsIsNull(cond)) {
                    throw new SqlSyntaxException(
                            "IS [NOT] NULL cannot be used in a CASE WHEN condition -- it has "
                                    + "no ParserNG rendering. Use it only directly in a "
                                    + "WHERE/HAVING clause.",
                            caseTok.start());
                }
                conditionText = BoolExprs.renderFused(cond);
            }
            expect(TokenType.THEN, "Expected THEN after CASE WHEN condition");
            String thenText = parseValueExpression();
            conditions.add(conditionText);
            thenBranches.add(thenText);
        }
        if (conditions.isEmpty()) {
            error("CASE requires at least one WHEN ... THEN ... clause");
        }

        String elseText = "NULL";
        if (peekType() == TokenType.ELSE) {
            advance();
            elseText = parseValueExpression();
        }
        expect(TokenType.END, "Expected END to close CASE");

        String result = elseText;
        for (int i = conditions.size() - 1; i >= 0; i--) {
            result = "if(" + conditions.get(i) + ", " + thenBranches.get(i) + ", " + result + ")";
        }
        return result;
    }

    /**
     * Parses a {@code cast_expression} (the leading {@code CAST} has not
     * yet been consumed) and renders it per {@link #renderCastText}. See
     * class javadoc, "CAST".
     */
    private String buildCastExpressionText() {
        expect(TokenType.CAST, "Expected CAST");
        expect(TokenType.LPAREN, "Expected '(' after CAST");
        // parseValueExpression(): this position is bounded by AS, which
        // can never continue a boolean_expression -- safe to try a boolean
        // condition first, same as a function argument.
        String inner = parseValueExpression();
        expect(TokenType.AS, "Expected AS in CAST(... AS type)");
        Token typeTok = expect(TokenType.IDENTIFIER, "Expected a target type after AS");
        expect(TokenType.RPAREN, "Expected ')' to close CAST(...)");
        return renderCastText(inner, typeTok);
    }

    /**
     * @return ParserNG text implementing {@code CAST(innerText AS typeTok)}
     * -- see class javadoc, "CAST", for the two supported type families
     * @throws SqlSyntaxException if {@code typeTok} does not spell a
     * recognized target type
     */
    private static String renderCastText(String innerText, Token typeTok) {
        String type = typeTok.text().toUpperCase(Locale.ROOT);
        if (CAST_TRUNCATING_TYPES.contains(type)) {
            return "if(" + innerText + " >= 0, floor(" + innerText + "), -1 * floor(-1 * (" + innerText + ")))";
        }
        if (CAST_NOOP_TYPES.contains(type)) {
            return "(" + innerText + ")";
        }
        throw new SqlSyntaxException(
                "Unsupported CAST target type '" + typeTok.text() + "'; supported types are "
                        + "INT/INTEGER/SMALLINT/BIGINT/LONG (truncating) and "
                        + "FLOAT/REAL/DOUBLE/DECIMAL/NUMERIC (no-op, already floating-point)",
                typeTok.start());
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

    /**
     * @return the type of the token {@code offset} positions ahead of the
     * current position, or {@link TokenType#EOF} if that would run past
     * the end of the token stream (which cannot happen for a well-formed
     * stream, since {@link SqlLexer} always appends a trailing
     * {@link TokenType#EOF} token, but this stays safe regardless of
     * {@code offset})
     */
    private TokenType peekTypeAt(int offset) {
        int idx = pos + offset;
        return idx < tokens.size() ? tokens.get(idx).type() : TokenType.EOF;
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