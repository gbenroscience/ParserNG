package com.github.gbenroscience.sqlv1;

import com.github.gbenroscience.sqlv1.ast.AggFunc;
import com.github.gbenroscience.sqlv1.ast.BoolExprs;
import com.github.gbenroscience.sqlv1.ast.CompOp;
import com.github.gbenroscience.sqlv1.ast.IsNullExpr;
import com.github.gbenroscience.sqlv1.ast.SelectItem;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The parser-ng-sql v1 conformance suite: exercises
 * {@link SqlLexer}/{@link SqlParser} (SQL text -&gt; AST) and
 * {@link BoolExprs} (NNF normalization + ParserNG-text rendering) against
 * the grammar documented on {@link SqlParser}, independently of Arrow.
 *
 * <p>
 * This is the same set of checks as
 * {@code com.github.gbenroscience.sqlv1.ParserSmokeTest} (a dependency-free
 * harness that can run with nothing but a JDK, useful in environments
 * without Maven Central access), expressed in ordinary JUnit 5 for the
 * module's real build (`mvn test`).
 *
 * @author GBEMIRO
 */
class SqlParserTest {

    // =====================================================================
    // SELECT list / FROM / aliasing
    // =====================================================================

    @Test
    void selectStarWithNoWhereClause() {
        SelectStatement s = SqlParser.parse("SELECT * FROM points");
        assertTrue(s.selectAll());
        assertTrue(s.items().isEmpty());
        assertEquals("points", s.table());
        assertNull(s.where());
    }

    @Test
    void selectListWithAliasAndWhere() {
        SelectStatement s = SqlParser.parse(
                "SELECT x, y, sqrt(x * x + y * y) AS magnitude FROM points WHERE magnitude > 10");
        assertFalse(s.selectAll());
        assertEquals(3, s.items().size());
        assertEquals("x", s.items().get(0).exprText());
        assertNull(s.items().get(0).alias());
        assertEquals("y", s.items().get(1).exprText());
        SelectItem third = s.items().get(2);
        assertEquals("sqrt(x * x + y * y)", third.exprText());
        assertEquals("magnitude", third.alias());
        assertEquals("magnitude", third.outputName());
        assertEquals("points", s.table());
        assertEquals("(magnitude > 10)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void singleExpressionSelectWithAlias() {
        SelectStatement s = SqlParser.parse(
                "SELECT sqrt(x * x + y * y) AS distance FROM data WHERE x > 10");
        assertEquals(1, s.items().size());
        assertEquals("sqrt(x * x + y * y)", s.items().get(0).exprText());
        assertEquals("distance", s.items().get(0).alias());
        assertEquals("(x > 10)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void unaliasedComputedColumnIsNamedAfterItsOwnExpressionText() {
        SelectStatement s = SqlParser.parse("SELECT x * y FROM t");
        assertEquals("x * y", s.items().get(0).outputName());
    }

    @Test
    void multipleSelectItemsMixingPassthroughAndComputedColumns() {
        SelectStatement s = SqlParser.parse(
                "SELECT x, y, x * y AS product, sqrt(x * x + y * y) "
                        + "FROM points WHERE x > 10 AND y < 20");
        assertEquals(4, s.items().size());
        assertEquals("x", s.items().get(0).outputName());
        assertEquals("y", s.items().get(1).outputName());
        assertEquals("product", s.items().get(2).outputName());
        assertEquals("sqrt(x * x + y * y)", s.items().get(3).outputName());
        assertEquals("((x>10)&&(y<20))", BoolExprs.renderFused(s.where()).replace(" ", ""));
    }

    @Test
    void keywordsAreCaseInsensitiveButIdentifiersAreCaseSensitive() {
        SelectStatement s = SqlParser.parse("select X, Y from Points where X > 1 and Y < 2");
        assertEquals("X", s.items().get(0).exprText());
        assertEquals("Y", s.items().get(1).exprText());
        assertEquals("Points", s.table());
    }

    @Test
    void whitespaceAndNewlinesAreTolerated() {
        SelectStatement s = SqlParser.parse(
                "SELECT\n  x,\n  y\nFROM\n  points\nWHERE\n  x > 1\n  AND y < 2\n");
        assertEquals(2, s.items().size());
        assertEquals("((x>1)&&(y<2))", BoolExprs.renderFused(s.where()).replace(" ", ""));
    }

    // =====================================================================
    // AND / OR / NOT / grouping
    // =====================================================================

    @Test
    void andOrGroupingRendersWithCorrectStructure() {
        SelectStatement s = SqlParser.parse(
                "SELECT x FROM t WHERE (x >= 200 AND x <= 300) OR (x = 4 OR y = 15)");
        assertEquals(
                "(((x >= 200) && (x <= 300)) || ((x == 4) || (y == 15)))",
                BoolExprs.renderFused(s.where()));
    }

    @Test
    void notOnAComparisonFlipsTheOperator() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT x > 5");
        assertEquals("(x <= 5)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void notOfAnAndBecomesOrOfNegations() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x > 5 AND y < 3)");
        assertEquals("((x <= 5) || (y >= 3))", BoolExprs.renderFused(s.where()));
    }

    @Test
    void notOfAnOrBecomesAndOfNegations() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x > 5 OR y < 3)");
        assertEquals("((x <= 5) && (y >= 3))", BoolExprs.renderFused(s.where()));
    }

    @Test
    void doubleNegationCancelsOut() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT NOT x = 1");
        assertEquals("(x == 1)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void parenthesizedArithmeticThenComparisonIsNotMisreadAsAGroupedBoolean() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE (x + 1) > 5");
        assertEquals("((x + 1) > 5)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void parenthesizedComparisonParsesAsAGroupedBoolean() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE (x > 5)");
        assertEquals("(x > 5)", BoolExprs.renderFused(s.where()));
    }

    // =====================================================================
    // BETWEEN / IN
    // =====================================================================

    @Test
    void betweenRendersAsAnInclusiveRangeConjunction() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x BETWEEN 1 AND 10");
        assertEquals("(x >= 1 && x <= 10)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void notBetweenRendersAsAnExcludedRangeDisjunction() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x NOT BETWEEN 1 AND 10");
        assertEquals("(x < 1 || x > 10)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void notOfAParenthesizedBetweenAlsoFlipsViaNnf() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x BETWEEN 1 AND 10)");
        assertEquals("(x < 1 || x > 10)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void inRendersAsADisjunctionOfEqualities() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x IN (1, 2, 3)");
        assertEquals("(x == 1 || x == 2 || x == 3)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void notInRendersAsAConjunctionOfInequalities() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x NOT IN (1, 2, 3)");
        assertEquals("(x != 1 && x != 2 && x != 3)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void betweenAndInCombineCorrectlyWithAnd() {
        SelectStatement s = SqlParser.parse(
                "SELECT x FROM t WHERE x BETWEEN 1 AND 10 AND y IN (2, 4, 6)");
        assertEquals(
                "((x>=1&&x<=10)&&(y==2||y==4||y==6))",
                BoolExprs.renderFused(s.where()).replace(" ", ""));
    }

    // =====================================================================
    // IS [NOT] NULL
    // =====================================================================

    @Test
    void isNullIsDetectedByContainsIsNull() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x IS NULL");
        assertTrue(BoolExprs.containsIsNull(s.where()));
        assertTrue(s.where() instanceof IsNullExpr n && !n.negated());
    }

    @Test
    void isNotNullIsDetectedByContainsIsNull() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x IS NOT NULL");
        assertTrue(BoolExprs.containsIsNull(s.where()));
        assertTrue(s.where() instanceof IsNullExpr n && n.negated());
    }

    @Test
    void notOfIsNullNormalizesToIsNotNull() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE NOT (x IS NULL)");
        assertTrue(s.where() instanceof IsNullExpr n && n.negated());
    }

    @Test
    void renderLeafRejectsIsNullDirectly() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x IS NULL");
        assertThrows(IllegalArgumentException.class, () -> BoolExprs.renderLeaf(s.where()));
        assertThrows(IllegalArgumentException.class, () -> BoolExprs.renderFused(s.where()));
    }

    // =====================================================================
    // comparison operators / literals / functions
    // =====================================================================

    @Test
    void everyComparisonOperatorLexesAndRendersToItsParserNgSpelling() {
        assertOperatorRenders("=", "==");
        assertOperatorRenders("!=", "!=");
        assertOperatorRenders("<>", "!=");
        assertOperatorRenders("<", "<");
        assertOperatorRenders("<=", "<=");
        assertOperatorRenders(">", ">");
        assertOperatorRenders(">=", ">=");
    }

    private void assertOperatorRenders(String sqlOp, String parserNgOp) {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x " + sqlOp + " 1");
        assertEquals("(x " + parserNgOp + " 1)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void compOpNegateIsAnInvolutionForAllSixOperators() {
        for (CompOp op : CompOp.values()) {
            assertEquals(op, op.negate().negate());
            assertTrue(op != op.negate(), "an operator must never be its own negation: " + op);
        }
    }

    @Test
    void decimalNumberLiteralsParse() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x > 3.14 AND x < 100.5");
        assertEquals("((x>3.14)&&(x<100.5))", BoolExprs.renderFused(s.where()).replace(" ", ""));
    }

    @Test
    void unaryMinusAndPlusInsideExpressions() {
        SelectStatement s = SqlParser.parse("SELECT -x + +y AS z FROM t WHERE x > -5");
        assertEquals("- x + + y", s.items().get(0).exprText());
        assertEquals("(x > - 5)", BoolExprs.renderFused(s.where()));
    }

    @Test
    void nestedFunctionCallsWithMultiplePurelyArithmeticArguments() {
        SelectStatement s = SqlParser.parse(
                "SELECT if(x, tan(x), 0.2) AS y, pow(sin(x), 2) + pow(cos(x), 2) FROM t");
        assertEquals("if(x, tan(x), 0.2)", s.items().get(0).exprText());
        assertEquals("y", s.items().get(0).alias());
        assertEquals("pow(sin(x), 2) + pow(cos(x), 2)", s.items().get(1).exprText());
    }

    // =====================================================================
    // embedded boolean conditions (function arguments, grouping parens,
    // select items) -- see SqlParser's "Embedded boolean conditions"
    // =====================================================================

    @Test
    void ifWithAComparisonConditionAsAFunctionArgumentNowParses() {
        SelectStatement s = SqlParser.parse(
                "SELECT if(sin(x) > 0, tan(x), 0.2) AS y FROM t");
        assertEquals("if((sin(x) > 0), tan(x), 0.2)", s.items().get(0).exprText());
        assertEquals("y", s.items().get(0).alias());
    }

    @Test
    void compoundAndConditionAsAFunctionArgument() {
        SelectStatement s = SqlParser.parse(
                "SELECT if(x > 0 AND y > 0, 1, 0) AS both_positive FROM t");
        assertEquals("if(((x > 0) && (y > 0)), 1, 0)", s.items().get(0).exprText());
    }

    @Test
    void selectItemMayItselfBeABareBooleanCondition() {
        SelectStatement s = SqlParser.parse("SELECT x > 0 AS flag FROM t");
        assertEquals("(x > 0)", s.items().get(0).exprText());
        assertEquals("flag", s.items().get(0).alias());
    }

    @Test
    void booleanConditionCombinesWithArithmeticViaAGroupingParen() {
        SelectStatement s = SqlParser.parse("SELECT 1 + (x > 0) AS n FROM t");
        assertEquals("1 + (x > 0)", s.items().get(0).exprText());
    }

    @Test
    void comparisonOperandMayContainAParenthesizedBooleanTerm() {
        // The comparison's own right-hand operand still uses the unbounded
        // buildExpressionText() path (not the boolean-first trial), but a
        // nested "(y > 0)" *inside* it is still fully supported, since that
        // inner paren is bounded by its own ')'.
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x = (y > 0)");
        assertEquals("(x == (y > 0))", BoolExprs.renderFused(s.where()));
    }

    @Test
    void betweenAndIsNotConfusedByTheEmbeddedBooleanTrial() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE x BETWEEN 1 AND 10 AND y > 0");
        assertEquals("((x>=1&&x<=10)&&(y>0))", BoolExprs.renderFused(s.where()).replace(" ", ""));
    }

    @Test
    void isNullIsRejectedInsideANestedOrEmbeddedPosition() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT if(x IS NULL, 0, x) FROM t"));
    }

    @Test
    void stringLiteralLexesForGrammarCompleteness() {
        // The grammar allows string_literal even though the Arrow backend
        // in this module only ever evaluates numeric (float32/float64)
        // columns; this only checks that it lexes/parses without error.
        SelectStatement s = SqlParser.parse("SELECT x FROM t WHERE tag = 'abc'");
        assertEquals("(tag == abc)", BoolExprs.renderFused(s.where()).replace("'", ""));
    }

    // =====================================================================
    // syntax errors
    // =====================================================================

    @Test
    void missingFromIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x WHERE x > 1"));
    }

    @Test
    void trailingGarbageAfterAValidQueryIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t; DROP TABLE t"));
    }

    @Test
    void danglingComparisonOperatorIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t WHERE x >"));
    }

    @Test
    void emptySelectListIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT FROM t"));
    }

    @Test
    void unterminatedStringLiteralIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t WHERE tag = 'abc"));
    }

    // =====================================================================
    // GROUP BY / HAVING / aggregates -- see SqlParser's "GROUP BY /
    // aggregates" and SelectStatement's "GROUP BY / aggregates -- a
    // deliberately strict subset"
    // =====================================================================

    @Test
    void sumCountAvgMinMaxAllRecognizedAsAggregateSelectItems() {
        SelectStatement s = SqlParser.parse(
                "SELECT cat, SUM(x) AS s, COUNT(*) AS n, AVG(x) AS a, MIN(x) AS mn, MAX(x) AS mx "
                        + "FROM t GROUP BY cat");
        assertEquals(6, s.items().size());
        assertFalse(s.items().get(0).isAggregate());
        assertEquals("cat", s.items().get(0).outputName());

        SelectItem sum = s.items().get(1);
        assertTrue(sum.isAggregate());
        assertEquals(AggFunc.SUM, sum.aggregate().func());
        assertEquals("x", sum.aggregate().argExprText());
        assertFalse(sum.aggregate().star());
        assertEquals("s", sum.outputName());

        SelectItem count = s.items().get(2);
        assertEquals(AggFunc.COUNT, count.aggregate().func());
        assertTrue(count.aggregate().star());
        assertNull(count.aggregate().argExprText());

        assertEquals(List.of("cat"), s.groupBy());
        assertTrue(s.isGrouped());
    }

    @Test
    void sumOfStarIsRejectedOnlyCountStarIsValid() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT SUM(*) FROM t"));
    }

    @Test
    void bareColumnMerelySharingAnAggregateFunctionsNameIsNotTreatedAsOne() {
        SelectStatement s = SqlParser.parse("SELECT sum FROM t");
        assertFalse(s.items().get(0).isAggregate());
        assertEquals("sum", s.items().get(0).exprText());
        assertFalse(s.isGrouped());
    }

    @Test
    void anAggregateItemWithNoExplicitGroupByIsStillAGroupedWholeTableQuery() {
        SelectStatement s = SqlParser.parse("SELECT COUNT(*) AS n FROM t");
        assertTrue(s.groupBy().isEmpty());
        assertTrue(s.isGrouped());
    }

    @Test
    void havingParsesAndRendersLikeWhere() {
        SelectStatement s = SqlParser.parse("SELECT cat, SUM(x) AS s FROM t GROUP BY cat HAVING SUM(x) > 100");
        assertEquals("(SUM(x) > 100)", BoolExprs.renderFused(s.having()));
    }

    @Test
    void havingWithoutGroupByOrAnAggregateItemIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t HAVING x > 1"));
    }

    @Test
    void selectStarCannotBeCombinedWithGroupBy() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT * FROM t GROUP BY cat"));
    }

    // =====================================================================
    // ORDER BY / LIMIT
    // =====================================================================

    @Test
    void orderByDefaultsToAscending() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t ORDER BY x");
        assertEquals(1, s.orderBy().size());
        assertEquals("x", s.orderBy().get(0).exprText());
        assertFalse(s.orderBy().get(0).descending());
    }

    @Test
    void orderByMultipleKeysWithExplicitAscDesc() {
        SelectStatement s = SqlParser.parse("SELECT x, y FROM t ORDER BY x ASC, y DESC");
        assertEquals(2, s.orderBy().size());
        assertFalse(s.orderBy().get(0).descending());
        assertTrue(s.orderBy().get(1).descending());
    }

    @Test
    void limitParsesAsAPlainNonNegativeInteger() {
        SelectStatement s = SqlParser.parse("SELECT x FROM t LIMIT 10");
        assertEquals(10, s.limit().intValue());
    }

    @Test
    void limitRejectsADecimalLiteral() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t LIMIT 10.5"));
    }

    @Test
    void limitRejectsANegativeLiteral() {
        // The grammar has no unary minus in this position -- LIMIT expects
        // a plain NUMBER token immediately, so this fails at the lexical/
        // parse level rather than ever reaching SelectStatement's own
        // defensive "LIMIT must be non-negative" check.
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT x FROM t LIMIT -5"));
    }

    @Test
    void whereGroupByHavingOrderByLimitComposeInOneQuery() {
        SelectStatement s = SqlParser.parse(
                "SELECT cat, SUM(x) AS total FROM t WHERE x > 0 GROUP BY cat "
                        + "HAVING SUM(x) > 50 ORDER BY total DESC LIMIT 5");
        assertEquals("(x > 0)", BoolExprs.renderFused(s.where()));
        assertEquals(List.of("cat"), s.groupBy());
        assertEquals("(SUM(x) > 50)", BoolExprs.renderFused(s.having()));
        assertEquals(1, s.orderBy().size());
        assertTrue(s.orderBy().get(0).descending());
        assertEquals(5, s.limit().intValue());
    }

    // =====================================================================
    // CASE / CAST -- see SqlParser's "CASE / CAST"
    // =====================================================================

    @Test
    void searchedCaseDesugarsRightToLeftIntoNestedIf() {
        SelectStatement s = SqlParser.parse(
                "SELECT CASE WHEN x > 0 THEN 1 WHEN x < 0 THEN -1 ELSE 0 END AS sign FROM t");
        assertEquals("(if((x > 0), 1, if((x < 0), - 1, 0)))", s.items().get(0).exprText());
        assertEquals("sign", s.items().get(0).alias());
    }

    @Test
    void simpleCaseComparesItsOperandForEqualityWithEachWhenValue() {
        SelectStatement s = SqlParser.parse(
                "SELECT CASE cat WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END AS label FROM t");
        assertEquals("(if(((cat) == (1)), 'a', if(((cat) == (2)), 'b', 'c')))", s.items().get(0).exprText());
    }

    @Test
    void caseWithoutElseIsRejected() {
        assertThrows(SqlSyntaxException.class,
                () -> SqlParser.parse("SELECT CASE WHEN x > 0 THEN 1 END AS s FROM t"));
    }

    @Test
    void isNullInsideACaseWhenConditionIsRejected() {
        assertThrows(SqlSyntaxException.class,
                () -> SqlParser.parse("SELECT CASE WHEN x IS NULL THEN 1 ELSE 0 END AS s FROM t"));
    }

    @Test
    void castToAnIntegerTypeTruncatesTowardZero() {
        SelectStatement s = SqlParser.parse("SELECT CAST(x AS INT) AS xi FROM t");
        assertEquals("((x) - ((x) % 1))", s.items().get(0).exprText());
    }

    @Test
    void castToAFloatingTypeIsANoOpIdentity() {
        SelectStatement s = SqlParser.parse("SELECT CAST(x AS DOUBLE) AS xd FROM t");
        assertEquals("(x)", s.items().get(0).exprText());
    }

    @Test
    void castToANonNumericTargetTypeIsRejected() {
        assertThrows(SqlSyntaxException.class, () -> SqlParser.parse("SELECT CAST(x AS VARCHAR) AS xs FROM t"));
    }
}