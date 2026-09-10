package com.github.gbenroscience.sqlv1.demo;

import com.github.gbenroscience.sqlv1.SqlParser;
import com.github.gbenroscience.sqlv1.SqlSyntaxException;
import com.github.gbenroscience.sqlv1.ast.BoolExpr;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import java.util.List;

/**
 * A tour of parser-ng-sql's {@code SELECT} grammar: 35 example queries,
 * grouped by feature, each parsed with {@link SqlParser#parse(String)} and
 * printed alongside a summary of the resulting {@link SelectStatement} --
 * its table, its select list (with any {@code AS} aliases), and its
 * {@code WHERE} clause rendered to the ParserNG expression text that
 * {@code ArrowQuery} would ultimately compile.
 *
 * <p>
 * This class deliberately depends only on {@code SqlParser}/{@code ast} --
 * not on {@code ArrowQuery}/{@code ArrowSql} or Arrow itself -- so it can be
 * compiled and run with nothing but this module's own sources on the
 * classpath, as a quick first look at the grammar before wiring up a real
 * {@code VectorSchemaRoot}. For a full end-to-end example (compiling
 * against a real Arrow schema and executing), see {@code ArrowQuery}'s and
 * {@code ArrowSql}'s own class javadoc.
 *
 * <p>
 * Run it with:
 * <pre>
 * java -cp &lt;this module's compiled classes&gt; com.github.gbenroscience.sqlv1.demo.SqlDemo
 * </pre>
 *
 * @author GBEMIRO
 */
public final class SqlDemo {

    private record Example(String description, String sql) {
    }

    private static final List<Example> EXAMPLES = List.of(

            // ---- SELECT list forms -----------------------------------------------
            new Example("SELECT * -- every column, no filter",
                    "SELECT * FROM points"),
            new Example("Plain column list, no aliases",
                    "SELECT x, y FROM points"),
            new Example("Computed column with AS alias",
                    "SELECT sqrt(x*x + y*y) AS distance FROM data WHERE x > 10"),
            new Example("Mix of passthrough and aliased computed columns",
                    "SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM points WHERE magnitude > 10"),
            new Example("Unaliased computed column keeps its own expression text as its name",
                    "SELECT x * y FROM points"),
            new Example("Several computed columns in one query",
                    "SELECT x, y, x * y AS product, sqrt(x*x + y*y) FROM points WHERE x > 10 AND y < 20"),
            new Example("Unary +/- inside an expression",
                    "SELECT -x + +y AS z FROM t"),
            new Example("Trigonometric identity via nested function calls",
                    "SELECT pow(sin(x), 2) + pow(cos(x), 2) AS identity FROM t"),
            new Example("Keywords are case-insensitive; identifiers are case-sensitive",
                    "select X, Y from Points where X > 1 and Y < 2"),

            // ---- comparison operators ----------------------------------------------
            new Example("Equality ( '=' -> ParserNG '==' )",
                    "SELECT x FROM t WHERE x = 5"),
            new Example("Inequality, '!=' spelling",
                    "SELECT x FROM t WHERE x != 5"),
            new Example("Inequality, '<>' spelling (also renders to '!=')",
                    "SELECT x FROM t WHERE x <> 5"),
            new Example("Less than",
                    "SELECT x FROM t WHERE x < 5"),
            new Example("Less than or equal",
                    "SELECT x FROM t WHERE x <= 5"),
            new Example("Greater than",
                    "SELECT x FROM t WHERE x > 5"),
            new Example("Greater than or equal",
                    "SELECT x FROM t WHERE x >= 5"),
            new Example("Decimal literals on both sides",
                    "SELECT x FROM t WHERE x > 3.14 AND x < 100.5"),
            new Example("String literal comparison (grammar-complete; Arrow backend is numeric-only)",
                    "SELECT x FROM t WHERE tag = 'abc'"),

            // ---- AND / OR / NOT / grouping -----------------------------------------
            new Example("AND of two comparisons",
                    "SELECT x FROM t WHERE x > 5 AND y < 10"),
            new Example("OR of two comparisons",
                    "SELECT x FROM t WHERE x > 5 OR y < 10"),
            new Example("Parenthesized groups combined with OR",
                    "SELECT x FROM t WHERE (x >= 200 AND x <= 300) OR (x = 4 OR y = 15)"),
            new Example("NOT on a single comparison (flips the operator via De Morgan/NNF)",
                    "SELECT x FROM t WHERE NOT x > 5"),
            new Example("NOT of a parenthesized AND (becomes an OR of negations)",
                    "SELECT x FROM t WHERE NOT (x > 5 AND y < 3)"),
            new Example("'(x + 1) > 5' -- arithmetic grouping, not a boolean group",
                    "SELECT x FROM t WHERE (x + 1) > 5"),

            // ---- BETWEEN / IN ---------------------------------------------------
            new Example("BETWEEN (inclusive range)",
                    "SELECT x FROM t WHERE x BETWEEN 1 AND 10"),
            new Example("NOT BETWEEN (excluded range)",
                    "SELECT x FROM t WHERE x NOT BETWEEN 1 AND 10"),
            new Example("IN (disjunction of equalities)",
                    "SELECT x FROM t WHERE x IN (1, 2, 3)"),
            new Example("NOT IN (conjunction of inequalities)",
                    "SELECT x FROM t WHERE x NOT IN (1, 2, 3)"),
            new Example("BETWEEN and IN combined with AND",
                    "SELECT x FROM t WHERE x BETWEEN 1 AND 10 AND y IN (2, 4, 6)"),

            // ---- IS [NOT] NULL ---------------------------------------------------
            new Example("IS NULL (evaluated via Arrow validity bitmaps, not a ParserNG expression)",
                    "SELECT x FROM t WHERE x IS NULL"),
            new Example("IS NOT NULL",
                    "SELECT x FROM t WHERE x IS NOT NULL"),

            // ---- embedded boolean conditions (function args / grouping / select items)
            new Example("A comparison used as a function argument -- if(cond, a, b)",
                    "SELECT if(sin(x) > 0, tan(x), 0.2) AS y FROM t"),
            new Example("A compound AND condition as a function argument",
                    "SELECT if(x > 0 AND y > 0, 1, 0) AS both_positive FROM t"),
            new Example("A select item that is itself a bare boolean condition",
                    "SELECT x > 0 AS flag FROM t"),
            new Example("A boolean condition combined with arithmetic via a grouping paren",
                    "SELECT 1 + (x > 0) AS n FROM t"),
            new Example("A comparison operand containing a nested parenthesized condition",
                    "SELECT x FROM t WHERE x = (y > 0)")
    );

    public static void main(String[] args) {
        System.out.println("parser-ng-sql grammar tour -- " + EXAMPLES.size() + " examples\n");
        int i = 0;
        for (Example example : EXAMPLES) {
            i++;
            System.out.println("=== [" + i + "] " + example.description() + " ===");
            System.out.println("SQL:      " + example.sql());
            try {
                SelectStatement stmt = SqlParser.parse(example.sql());
                System.out.println("Table:    " + stmt.table());
                System.out.println("Columns:  " + describeColumns(stmt));
                System.out.println("Where:    " + describeWhere(stmt.where()));
            } catch (SqlSyntaxException e) {
                System.out.println("REJECTED: " + e.getMessage());
            }
            System.out.println();
        }
    }

    private static String describeColumns(SelectStatement stmt) {
        if (stmt.selectAll()) {
            return "* (all columns)";
        }
        StringBuilder sb = new StringBuilder();
        for (SelectItem item : stmt.items()) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(item.exprText());
            if (item.alias() != null) {
                sb.append(" AS ").append(item.alias());
            }
        }
        return sb.toString();
    }

    private static String describeWhere(BoolExpr where) {
        if (where == null) {
            return "(none)";
        }
        if (BoolExprs.containsIsNull(where)) {
            return "(contains IS [NOT] NULL -- ArrowQuery evaluates this leaf-by-leaf "
                    + "against Arrow validity bitmaps at execute() time rather than as a "
                    + "single fused ParserNG expression; see ArrowQuery's \"Predicate "
                    + "evaluation strategy\")";
        }
        return BoolExprs.renderFused(where) + "   [ParserNG text ArrowQuery would compile]";
    }

    private SqlDemo() {
    }
}