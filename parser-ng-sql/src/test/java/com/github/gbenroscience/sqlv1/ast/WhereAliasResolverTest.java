package com.github.gbenroscience.sqlv1.ast;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Exercises {@link WhereAliasResolver} entirely at the AST level, with no
 * Arrow dependency — mirrors {@code SqlParserTest}'s dependency-free style.
 *
 * @author GBEMIRO
 */
class WhereAliasResolverTest {

    @Test
    void comparisonAgainstAliasIsExpandedOnBothOperandsIndependently() {
        ComparisonExpr expr = new ComparisonExpr("magnitude", CompOp.GT, "60");
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(new ComparisonExpr("(sqrt(x * x + y * y))", CompOp.GT, "60"), resolved);
    }

    @Test
    void nonAliasIdentifiersAreLeftAlone() {
        ComparisonExpr expr = new ComparisonExpr("x", CompOp.GT, "20");
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(expr, resolved);
    }

    @Test
    void emptyAliasMapIsANoOpAndReturnsTheSameInstance() {
        ComparisonExpr expr = new ComparisonExpr("magnitude", CompOp.GT, "60");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, Map.of());

        assertEquals(expr, resolved);
    }

    @Test
    void nullWhereIsReturnedAsIs() {
        assertEquals(null, WhereAliasResolver.resolve(null, Map.of("a", "1")));
    }

    @Test
    void aliasLikeSubstringInsideALongerIdentifierIsNotTouched() {
        // "magnitude2" must not be corrupted by an alias named "magnitude".
        ComparisonExpr expr = new ComparisonExpr("magnitude2", CompOp.GT, "60");
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(expr, resolved);
    }

    @Test
    void identifierImmediatelyFollowedByParenIsTreatedAsAFunctionCallNotAnAlias() {
        // A function literally named the same as an alias must not be rewritten.
        ComparisonExpr expr = new ComparisonExpr("magnitude(x)", CompOp.GT, "60");
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(expr, resolved);
    }

    @Test
    void aliasNameInsideAStringLiteralIsNotTouched() {
        ComparisonExpr expr = new ComparisonExpr("status", CompOp.EQ, "'magnitude'");
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(expr, resolved);
    }

    @Test
    void chainedAliasesResolveTransitively() {
        ComparisonExpr expr = new ComparisonExpr("b", CompOp.GT, "5");
        Map<String, String> aliases = Map.of(
                "a", "sqrt(x)",
                "b", "a * 2");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(new ComparisonExpr("((sqrt(x)) * 2)", CompOp.GT, "5"), resolved);
    }

    @Test
    void cyclicAliasesThrowInsteadOfLoopingForever() {
        ComparisonExpr expr = new ComparisonExpr("a", CompOp.GT, "5");
        Map<String, String> aliases = Map.of(
                "a", "b + 1",
                "b", "a + 1");

        assertThrows(IllegalArgumentException.class, () -> WhereAliasResolver.resolve(expr, aliases));
    }

    @Test
    void betweenBoundsAndTargetAreAllSubstituted() {
        BetweenExpr expr = new BetweenExpr("magnitude", "lo", "hi", false);
        Map<String, String> aliases = Map.of(
                "magnitude", "sqrt(x * x + y * y)",
                "lo", "10",
                "hi", "90");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(new BetweenExpr("(sqrt(x * x + y * y))", "(10)", "(90)", false), resolved);
    }

    @Test
    void everyInValueIsSubstitutedIndependently() {
        InExpr expr = new InExpr("magnitude", List.of("a", "50", "a"), false);
        Map<String, String> aliases = Map.of(
                "magnitude", "sqrt(x * x + y * y)",
                "a", "10");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(new InExpr("(sqrt(x * x + y * y))", List.of("(10)", "50", "(10)"), false), resolved);
    }

    @Test
    void isNullTargetIsSubstituted() {
        IsNullExpr expr = new IsNullExpr("magnitude", true);
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(new IsNullExpr("(sqrt(x * x + y * y))", true), resolved);
    }

    @Test
    void andOrStructureIsPreservedAroundSubstitutedLeaves() {
        BoolExpr expr = new AndExpr(
                new ComparisonExpr("magnitude", CompOp.GT, "60"),
                new ComparisonExpr("x", CompOp.LT, "90"));
        Map<String, String> aliases = Map.of("magnitude", "sqrt(x * x + y * y)");

        BoolExpr resolved = WhereAliasResolver.resolve(expr, aliases);

        assertEquals(new AndExpr(
                new ComparisonExpr("(sqrt(x * x + y * y))", CompOp.GT, "60"),
                new ComparisonExpr("x", CompOp.LT, "90")), resolved);
    }
}