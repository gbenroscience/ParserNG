package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import java.util.List;

/**
 * A dependency-free smoke test for the SQL-text -&gt; AST -&gt; ParserNG-text
 * pipeline (everything in this module that does not touch Arrow).
 *
 * <p>
 * This exists alongside the proper JUnit 5 tests
 * ({@code SqlParserTest}, {@code BoolExprsTest}) so the parser core can be
 * exercised with nothing but a JDK on the classpath — useful in environments
 * without network access to a Maven repository (JUnit included). Run it with:
 * <pre>
 * javac -d out (this file, TokenType, Token, SqlLexer, SqlSyntaxException,
 * SqlParser, and everything under the ast package -- i.e. every source file
 * in this module except ArrowQuery and ArrowSql, which need Arrow on the
 * classpath).
 * java -cp out com.github.gbenroscience.sqlv1.ParserSmokeTest
 * </pre>
 * It exits with a non-zero status and prints every failure if anything is
 * wrong; otherwise it prints a summary of passed checks.
 *
 * @author GBEMIRO
 */
public final class ParserSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        // ---- basic SELECT * ----
        check("select star, no where", () -> {
            SelectStatement s = SqlParser.parse("SELECT * FROM points");
            expect(s.selectAll(), "selectAll");
            expect(s.items().isEmpty(), "items empty");
            expectEq("points", s.table());
            expect(s.where() == null, "where null");
        });

        // ---- select list with alias, from the task's own example ----
        check("select list with alias and where", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT x, y, sqrt(x * x + y * y) AS magnitude FROM points WHERE magnitude > 10");
            expect(!s.selectAll(), "not selectAll");
            expectEq(3, s.items().size());
            expectEq("x", s.items().get(0).exprText());
            expect(s.items().get(0).alias() == null, "x has no alias");
            expectEq("y", s.items().get(1).exprText());
            SelectItem third = s.items().get(2);
            expectEq("sqrt(x * x + y * y)", third.exprText());
            expectEq("magnitude", third.alias());
            expectEq("magnitude", third.outputName());
            expectEq("points", s.table());
            String rendered = BoolExprs.renderFused(s.where());
            expectEq("(magnitude > 10)", rendered);
        });

        // ---- second example from the task text ----
        check("single expression select, no alias", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT sqrt(x * x + y * y) AS distance FROM data WHERE x > 10");
            expectEq(1, s.items().size());
            expectEq("sqrt(x * x + y * y)", s.items().get(0).exprText());
            expectEq("distance", s.items().get(0).alias());
            expectEq("(x > 10)", BoolExprs.renderFused(s.where()));
        });

        // ---- AND / OR / parens precedence ----
        check("and/or grouping renders correctly", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT x FROM t WHERE (x >= 200 AND x <= 300) OR (x = 4 OR y = 15)");
            String rendered = BoolExprs.renderFused(s.where());
            expectEq("(((x >= 200) && (x <= 300)) || ((x == 4) || (y == 15)))", rendered);
        });

        // ---- NOT pushed through comparison (De Morgan / NNF) ----
        check("NOT comparison flips operator", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT x > 5");
            expectEq("(x <= 5)", BoolExprs.renderFused(s.where()));
        });

        // ---- NOT (AND) -> OR of negations ----
        check("NOT (a AND b) -> (NOT a) OR (NOT b)", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x > 5 AND y < 3)");
            expectEq("((x <= 5) || (y >= 3))", BoolExprs.renderFused(s.where()));
        });

        // ---- NOT (OR) -> AND of negations ----
        check("NOT (a OR b) -> (NOT a) AND (NOT b)", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x > 5 OR y < 3)");
            expectEq("((x <= 5) && (y >= 3))", BoolExprs.renderFused(s.where()));
        });

        // ---- double negation cancels ----
        check("NOT NOT x = 1 -> x == 1", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT NOT x = 1");
            expectEq("(x == 1)", BoolExprs.renderFused(s.where()));
        });

        // ---- BETWEEN / NOT BETWEEN ----
        check("BETWEEN renders as range AND", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x BETWEEN 1 AND 10");
            expectEq("(x >= 1 && x <= 10)", BoolExprs.renderFused(s.where()));
        });
        check("NOT BETWEEN renders as excluded range OR", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x NOT BETWEEN 1 AND 10");
            expectEq("(x < 1 || x > 10)", BoolExprs.renderFused(s.where()));
        });
        check("NOT (x BETWEEN ...) also flips via NNF", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x BETWEEN 1 AND 10)");
            expectEq("(x < 1 || x > 10)", BoolExprs.renderFused(s.where()));
        });

        // ---- IN / NOT IN ----
        check("IN renders as disjunction of equalities", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x IN (1, 2, 3)");
            expectEq("(x == 1 || x == 2 || x == 3)", BoolExprs.renderFused(s.where()));
        });
        check("NOT IN renders as conjunction of inequalities", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x NOT IN (1, 2, 3)");
            expectEq("(x != 1 && x != 2 && x != 3)", BoolExprs.renderFused(s.where()));
        });

        // ---- IS NULL / IS NOT NULL ----
        check("IS NULL / IS NOT NULL parse and containsIsNull is detected", () -> {
            SelectStatement s1 = SqlParser.parse("SELECT x FROM t WHERE x IS NULL");
            expect(BoolExprs.containsIsNull(s1.where()), "contains is null");
            SelectStatement s2 = SqlParser.parse("SELECT x FROM t WHERE x IS NOT NULL");
            expect(BoolExprs.containsIsNull(s2.where()), "contains is not null");
            SelectStatement s3 = SqlParser.parse("SELECT x FROM t WHERE NOT (x IS NULL)");
            expect(BoolExprs.containsIsNull(s3.where()), "contains is null after NOT");
            // NOT (x IS NULL) should normalize to (x IS NOT NULL)
            BoolExpr where = s3.where();
            expect(where instanceof com.github.gbenroscience.sqlv1.ast.IsNullExpr n && n.negated(),
                    "NOT IS NULL normalizes to IS NOT NULL");
        });

        // ---- ambiguous leading paren: (x + 1) > 5 vs (x > 5) ----
        check("(expr) > n parses as comparison, not grouped boolean", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE (x + 1) > 5");
            expectEq("((x + 1) > 5)", BoolExprs.renderFused(s.where()));
        });
        check("(bool) parses as grouped boolean", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE (x > 5)");
            expectEq("(x > 5)", BoolExprs.renderFused(s.where()));
        });

        // ---- function calls, nested parens, commas inside calls ----
        // v1 originally restricted `expression` to pure arithmetic (no
        // comparisons), which meant a comparison used as a function
        // *argument* -- ParserNG's own `if(sin(x) > 0, tan(x), 0.2)` idiom
        // -- couldn't be expressed. That was an oversight in the original
        // grammar, now fixed: see SqlParser's "Embedded boolean conditions".
        check("if(cond, a, b) with a comparison condition now parses", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT if(sin(x) > 0, tan(x), 0.2) AS y FROM t");
            expectEq("if((sin(x) > 0), tan(x), 0.2)", s.items().get(0).exprText());
            expectEq("y", s.items().get(0).alias());
        });
        check("nested function calls with multiple purely-arithmetic args still work", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT if(x, tan(x), 0.2) AS y, pow(sin(x), 2) + pow(cos(x), 2) FROM t");
            expectEq("if(x, tan(x), 0.2)", s.items().get(0).exprText());
            expectEq("y", s.items().get(0).alias());
            expectEq("pow(sin(x), 2) + pow(cos(x), 2)", s.items().get(1).exprText());
        });
        check("compound && condition as a function argument", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT if(x > 0 AND y > 0, 1, 0) AS both_positive FROM t");
            expectEq("if(((x > 0) && (y > 0)), 1, 0)", s.items().get(0).exprText());
        });
        check("a select item may itself be a bare boolean condition", () -> {
            SelectStatement s = SqlParser.parse("SELECT x > 0 AS flag FROM t");
            expectEq("(x > 0)", s.items().get(0).exprText());
            expectEq("flag", s.items().get(0).alias());
        });
        check("a boolean condition combines with arithmetic via a grouping paren", () -> {
            SelectStatement s = SqlParser.parse("SELECT 1 + (x > 0) AS n FROM t");
            expectEq("1 + (x > 0)", s.items().get(0).exprText());
        });
        check("x = (y > 0) still uses the unbounded comparison-operand path correctly", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x = (y > 0)");
            expectEq("(x == (y > 0))", BoolExprs.renderFused(s.where()));
        });
        check("BETWEEN ... AND ... is not confused by the new embedded-boolean trial", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x BETWEEN 1 AND 10 AND y > 0");
            expectEq("((x>=1&&x<=10)&&(y>0))", BoolExprs.renderFused(s.where()).replace(" ", ""));
        });
        check("IS NULL is rejected inside a nested/embedded position", () -> {
            expectThrows(() -> SqlParser.parse("SELECT if(x IS NULL, 0, x) FROM t"));
        });

        // ---- multiple select items, some passthrough, arithmetic exprs ----
        check("filterProject-shaped multi-column select", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT x, y, x * y AS product, sqrt(x * x + y * y) "
                    + "FROM points WHERE x > 10 AND y < 20");
            expectEq(4, s.items().size());
            expectEq("x", s.items().get(0).outputName());
            expectEq("y", s.items().get(1).outputName());
            expectEq("product", s.items().get(2).outputName());
            expectEq("sqrt(x * x + y * y)", s.items().get(3).outputName());
            expectEq("((x>10)&&(y<20))",
                    BoolExprs.renderFused(s.where()).replace(" ", ""));
        });

        // ---- comparison operator variety, including <> ----
        check("all comparison operators lex/parse", () -> {
            for (String[] pair : new String[][]{
                {"=", "=="}, {"!=", "!="}, {"<>", "!="},
                {"<", "<"}, {"<=", "<="}, {">", ">"}, {">=", ">="}
            }) {
                SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x " + pair[0] + " 1");
                expectEq("(x " + pair[1] + " 1)", BoolExprs.renderFused(s.where()));
            }
        });

        // ---- case-insensitive keywords, case-sensitive identifiers ----
        check("keywords case-insensitive, identifiers case-sensitive", () -> {
            SelectStatement s = SqlParser.parse("select X, Y from Points where X > 1 and Y < 2");
            expectEq("X", s.items().get(0).exprText());
            expectEq("Y", s.items().get(1).exprText());
            expectEq("Points", s.table());
        });

        // ---- string literal lexing (grammar allows it even if the Arrow
        //      backend only evaluates numeric columns) ----
        check("string literal lexes", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE tag = 'abc'");
            expectEq("(tag == abc)", BoolExprs.renderFused(s.where()).replace("'", ""));
        });

        // ---- syntax error cases ----
        check("missing FROM is rejected", () -> {
            expectThrows(() -> SqlParser.parse("SELECT x WHERE x > 1"));
        });
        check("trailing garbage is rejected", () -> {
            expectThrows(() -> SqlParser.parse("SELECT x FROM t; DROP TABLE t"));
        });
        check("dangling operator is rejected", () -> {
            expectThrows(() -> SqlParser.parse("SELECT x FROM t WHERE x >"));
        });
        check("empty select list is rejected", () -> {
            expectThrows(() -> SqlParser.parse("SELECT FROM t"));
        });
        check("unterminated string is rejected", () -> {
            expectThrows(() -> SqlParser.parse("SELECT x FROM t WHERE tag = 'abc"));
        });

        // ---- decimal number literals ----
        check("decimal number literals parse", () -> {
            SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x > 3.14 AND x < 100.5");
            expectEq("((x>3.14)&&(x<100.5))", BoolExprs.renderFused(s.where()).replace(" ", ""));
        });

        // ---- unary minus/plus inside an expression span ----
        check("unary minus/plus inside expressions", () -> {
            SelectStatement s = SqlParser.parse("SELECT -x + +y AS z FROM t WHERE x > -5");
            expectEq("- x + + y", s.items().get(0).exprText());
            expectEq("(x > - 5)", BoolExprs.renderFused(s.where()));
        });

        // ---- whitespace/newlines are tolerated everywhere ----
        check("whitespace and newlines are tolerated", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT\n  x,\n  y\nFROM\n  points\nWHERE\n  x > 1\n  AND y < 2\n");
            expectEq(2, s.items().size());
            expectEq("((x>1)&&(y<2))", BoolExprs.renderFused(s.where()).replace(" ", ""));
        });

        // ---- BETWEEN and IN combined with AND/OR ----
        check("mixed BETWEEN/IN/AND/OR renders correctly", () -> {
            SelectStatement s = SqlParser.parse(
                    "SELECT x FROM t WHERE x BETWEEN 1 AND 10 AND y IN (2, 4, 6)");
            expectEq("((x>=1&&x<=10)&&(y==2||y==4||y==6))",
                    BoolExprs.renderFused(s.where()).replace(" ", ""));
        });

        // ---- CompOp.negate() is its own inverse (an involution) ----
        check("CompOp.negate is an involution covering all six operators", () -> {
            for (com.github.gbenroscience.sqlv1.ast.CompOp op : com.github.gbenroscience.sqlv1.ast.CompOp.values()) {
                expectEq(op, op.negate().negate());
                expect(op != op.negate(), "an operator must never be its own negation: " + op);
            }
        });

        System.out.println();
        System.out.println(passed + " passed, " + failed + " failed");
        if (failed > 0) {
            System.exit(1);
        }
    }

    // =====================================================================
    // tiny test harness
    // =====================================================================

    private interface Check {
        void run();
    }

    private static void check(String name, Check c) {
        try {
            c.run();
            passed++;
            System.out.println("PASS  " + name);
        } catch (Throwable t) {
            failed++;
            System.out.println("FAIL  " + name + "  -- " + t);
        }
    }

    private static void expect(boolean condition, String what) {
        if (!condition) {
            throw new AssertionError("expected " + what);
        }
    }

    private static void expectEq(Object expected, Object actual) {
        if (!java.util.Objects.equals(expected, actual)) {
            throw new AssertionError("expected <" + expected + "> but was <" + actual + ">");
        }
    }

    private static void expectThrows(Runnable r) {
        try {
            r.run();
        } catch (SqlSyntaxException expected) {
            return;
        }
        throw new AssertionError("expected SqlSyntaxException but none was thrown");
    }
}
