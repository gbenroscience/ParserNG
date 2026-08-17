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
package com.github.gbenroscience.parser;

/**
 *
 * @author GBEMIRO
 */ 
 
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;
import static org.junit.jupiter.api.Assertions.*;
 

/**
 * Test suite for {@link Scanner}, covering:
 *  - Backward compatibility of all six legacy constructors
 *  - Static/dynamic token matching, ordering, and identifier-boundary safety
 *  - identifierPartExtra() subscript grouping
 *  - The new delimited-region feature (quoted strings, block comments, custom
 *    raw blocks) that lets ParserNG treat quoted text as an opaque literal
 *  - Escaping, marker stripping, and all three UnterminatedDelimiterPolicy values
 *  - A battery of math expressions and the full diffeqn(...) scenario that
 *    motivated this feature
 *
 * Compatible with JUnit 4 and Java 8 (no List.of, no var, no records).
 *
 * NOTE: org.junit.Assert.assertThrows requires JUnit 4.13+. If your project
 * pins an older JUnit 4 version, replace the try/assertThrows pattern below
 * with a manual try/catch + fail().
 */
public class ScannerTest {

    // =====================================================================
    // 1. LEGACY CONSTRUCTOR BACKWARD COMPATIBILITY
    //    These exercise the pre-existing public constructors directly (not
    //    the Builder) to prove none of them changed behavior.
    // =====================================================================

    @Test
    public void legacyConstructor_varargTokensOnly() {
        // Scanner(String, boolean, String...)
        Scanner sc = new Scanner("a+b*c", true, "+", "*");
        Assertions.assertEquals(Arrays.asList("a", "+", "b", "*", "c"), sc.scan());
    }

    @Test
    public void legacyConstructor_moreTokensPlusVarargs() {
        // Scanner(String, boolean, String[], String...)
        Scanner sc = new Scanner("a+b*c-d", true, new String[]{"+", "*"}, "-");
        assertEquals(Arrays.asList("a", "+", "b", "*", "c", "-", "d"), sc.scan());
    }

    @Test
    public void legacyConstructor_threeTokenArrays() {
        // Scanner(String, boolean, String[], String[], String...)
        Scanner sc = new Scanner("a+b*c-d/e", true, new String[]{"+"}, new String[]{"*"}, "-", "/");
        assertEquals(Arrays.asList("a", "+", "b", "*", "c", "-", "d", "/", "e"), sc.scan());
    }

    @Test
    public void legacyConstructor_predicatePlusVarargs() {
        // Scanner(String, boolean, Predicate<String>, String...)
        Predicate<String> anyLetters = w -> !w.isEmpty() && w.chars().allMatch(Character::isLetter);
        Scanner sc = new Scanner("index in range", true, anyLetters, "in");
        // spaces kept as literals since ignoreWhitespace defaults to false on legacy ctors
        assertEquals(Arrays.asList("index", " ", "in", " ", "range"), sc.scan());
    }

    @Test
    public void legacyConstructor_predicatePlusMoreTokensPlusVarargs() {
        // Scanner(String, boolean, Predicate<String>, String[], String...)
        Predicate<String> neverMatch = w -> false;
        Scanner sc = new Scanner("a+b*c", true, neverMatch, new String[]{"+"}, "*");
        assertEquals(Arrays.asList("a", "+", "b", "*", "c"), sc.scan());
    }

    @Test
    public void legacyConstructor_predicatePlusThreeTokenArrays() {
        // Scanner(String, boolean, Predicate<String>, String[], String[], String...)
        Predicate<String> neverMatch = w -> false;
        Scanner sc = new Scanner("a+b*c-d/e", true, neverMatch, new String[]{"+"}, new String[]{"*"}, "-", "/");
        assertEquals(Arrays.asList("a", "+", "b", "*", "c", "-", "d", "/", "e"), sc.scan());
    }

    @Test
    public void legacyConstructor_includeTokensFalseOmitsTokensFromOutput() {
        Predicate<String> dynamicRules = w ->
                (w.startsWith("anon") && w.substring(4).matches("\\d+")) || w.startsWith("_");
        Scanner sc = new Scanner("print(anon9,_anon2)", false, dynamicRules, "print", "(", ")", ",");
        assertEquals(Arrays.asList("anon9", "_anon2"), sc.scan());
    }

    // =====================================================================
    // 2. DYNAMIC MATCHING, STATIC-FIRST VS DYNAMIC-FIRST, BOUNDARY SAFETY
    // =====================================================================

    @Test
    public void dynamicMatch_legacyConstructorKeepsSpacesAsLiterals() {
        Predicate<String> dynamicRules = w ->
                (w.startsWith("anon") && w.substring(4).matches("\\d+")) || w.startsWith("_");
        Scanner sc = new Scanner("print( anon9 , _anon2 , $C )", true, dynamicRules,
                "print", "(", ")", ",");
        assertEquals(Arrays.asList("print", "(", " ", "anon9", " ", ",", " ", "_anon2", " ", ",", " $C ", ")"),
                sc.scan());
    }

    @Test
    public void dynamicMatch_builderIgnoreWhitespaceStripsSpaces() {
        Predicate<String> dynamicRules = w ->
                (w.startsWith("anon") && w.substring(4).matches("\\d+")) || w.startsWith("_");
        Scanner sc = new Scanner.Builder("print( anon9 , _anon2 , $C )")
                .includeTokens(true)
                .ignoreWhitespace(true)
                .withDynamicMatcher(dynamicRules)
                .addTokens(new String[]{"print", "(", ")", ","})
                .build();
        assertEquals(Arrays.asList("print", "(", "anon9", ",", "_anon2", ",", "$C", ")"), sc.scan());
    }

    @Test
    public void dynamicMatch_staticFirstDoesNotSplitLongerIdentifier() {
        // Static token "in" must not fire inside "index" - boundary guard.
        Predicate<String> anyLetters = w -> w.chars().allMatch(Character::isLetter);
        Scanner sc = new Scanner.Builder("index in range")
                .ignoreWhitespace(true)
                .withDynamicMatcher(anyLetters)
                .addTokens(new String[]{"in"})
                .build();
        assertEquals(Arrays.asList("index", "in", "range"), sc.scan());
    }

    @Test
    public void dynamicMatch_dynamicFirstShadowsStaticTokenButSameResultHere() {
        Predicate<String> anyLetters = w -> w.chars().allMatch(Character::isLetter);
        Scanner sc = new Scanner.Builder("index in range")
                .ignoreWhitespace(true)
                .withDynamicMatcher(anyLetters)
                .addTokens(new String[]{"in"})
                .matchDynamicFirst(true)
                .build();
        // Same output as static-first here, but mechanistically different: the
        // dynamic predicate claims "in" whole before the static token ever gets
        // a chance, per the documented shadowing behavior.
        assertEquals(Arrays.asList("index", "in", "range"), sc.scan());
    }

    // =====================================================================
    // 3. identifierPartExtra()
    // =====================================================================

    @Test
    public void identifierPartExtra_withoutExtraCharsSubscriptSplits() {
        Predicate<String> isSubscriptedVar = w -> w.matches("[a-zA-Z]\\w*(\\[\\d+])?");
        Scanner sc = new Scanner.Builder("y[4]+3*x")
                .ignoreWhitespace(true)
                .withDynamicMatcher(isSubscriptedVar)
                .addTokens(new String[]{"+", "*"})
                .matchDynamicFirst(true)
                .build();
        assertEquals(Arrays.asList("y", "[4]", "+", "3", "*", "x"), sc.scan());
    }

    @Test
    public void identifierPartExtra_withExtraCharsSubscriptStaysWhole() {
        Predicate<String> isSubscriptedVar = w -> w.matches("[a-zA-Z]\\w*(\\[\\d+])?");
        Scanner sc = new Scanner.Builder("y[4]+3*x")
                .ignoreWhitespace(true)
                .withDynamicMatcher(isSubscriptedVar)
                .addTokens(new String[]{"+", "*"})
                .matchDynamicFirst(true)
                .identifierPartExtra('[', ']')
                .build();
        assertEquals(Arrays.asList("y[4]", "+", "3", "*", "x"), sc.scan());
    }

    // =====================================================================
    // 4. CACHING AND IMMUTABILITY
    // =====================================================================

    @Test
    public void caching_secondScanCallReturnsSameListInstance() {
        Scanner sc = new Scanner.Builder("a+b").ignoreWhitespace(true).addTokens(new String[]{"+"}).build();
        List<String> first = sc.scan();
        List<String> second = sc.scan();
        assertSame(first, second);
    }

    @Test
    public void caching_independentScannersProduceIndependentResults() {
        Scanner sc1 = new Scanner.Builder("a+b").ignoreWhitespace(true).addTokens(new String[]{"+"}).build();
        Scanner sc2 = new Scanner.Builder("c-d").ignoreWhitespace(true).addTokens(new String[]{"-"}).build();
        assertEquals(Arrays.asList("a", "+", "b"), sc1.scan());
        assertEquals(Arrays.asList("c", "-", "d"), sc2.scan());
    }

    @Test
    public void caching_sameBuilderCanBuildMultipleIndependentScanners() {
        Scanner.Builder builder = new Scanner.Builder("x+y").ignoreWhitespace(true).addTokens(new String[]{"+"});
        Scanner sc1 = builder.build();
        Scanner sc2 = builder.build();
        assertEquals(sc1.scan(), sc2.scan());
    }

    // =====================================================================
    // 5. NULL-SAFETY AND BUILDER VALIDATION
    // =====================================================================

    @Test
    public void nullSafety_nullDynamicMatcherDefaultsToAlwaysFalse() {
        Scanner sc = new Scanner.Builder("a+b")
                .ignoreWhitespace(true)
                .withDynamicMatcher(null)
                .addTokens(new String[]{"+"})
                .build();
        assertEquals(Arrays.asList("a", "+", "b"), sc.scan());
    }

    @Test
    public void nullSafety_addTokensIgnoresNullArrayArgument() {
        Scanner sc = new Scanner.Builder("a+b")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"+"}, null)
                .build();
        assertEquals(Arrays.asList("a", "+", "b"), sc.scan());
    }

    @Test
    public void nullSafety_builderRejectsNullInput() {
        assertThrows(NullPointerException.class, () -> new Scanner.Builder(null));
    }

    @Test
    public void nullSafety_onUnterminatedDelimiterNullDefaultsToConsumeToEnd() {
        Scanner sc = new Scanner.Builder("foo(\"hello")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "("})
                .addDelimitedRegion("\"", "\"")
                .onUnterminatedDelimiter(null)
                .build();
        assertEquals(Arrays.asList("foo", "(", "\"hello"), sc.scan());
    }

    @Test
    public void nullSafety_unterminatedDelimiterExceptionExposesPositionAndMarkers() {
        Scanner sc = new Scanner.Builder("foo(\"hello + x * y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "(", ")", "+", "*"})
                .addDelimitedRegion("\"", "\"")
                .onUnterminatedDelimiter(Scanner.UnterminatedDelimiterPolicy.THROW)
                .build();
        Scanner.UnterminatedDelimiterException ex = assertThrows(
                Scanner.UnterminatedDelimiterException.class, sc::scan);
        assertEquals("\"", ex.getDelimiterStart());
        assertEquals("\"", ex.getDelimiterEnd());
        assertEquals(4, ex.getPosition());
    }

    // =====================================================================
    // 6. MATH EXPRESSIONS, QUOTED STRINGS, AND THE diffeqn(...) SCENARIO
    //    (this is the batch that motivated the delimited-region feature)
    // =====================================================================

    @Test
    void plusMinus() {
        Scanner.Builder b = new Scanner.Builder("2+2")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("2", "+", "2"), scanner.scan());
    }

    @Test
    void division() {
        Scanner.Builder b = new Scanner.Builder("10/2")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("10", "/", "2"), scanner.scan());
    }

    @Test
    void power() {
        Scanner.Builder b = new Scanner.Builder("2^10")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("2", "^", "10"), scanner.scan());
    }

    @Test
    void functionCallsSinCos() {
        Scanner.Builder b = new Scanner.Builder("sin(x)+cos(y)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("sin", "(", "x", ")", "+", "cos", "(", "y", ")"), scanner.scan());
    }

    @Test
    void sqrtCall() {
        Scanner.Builder b = new Scanner.Builder("sqrt(4)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("sqrt", "(", "4", ")"), scanner.scan());
    }

    @Test
    void mixedOpsWithParens() {
        Scanner.Builder b = new Scanner.Builder("a*(b+c)-d/e")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("a", "*", "(", "b", "+", "c", ")", "-", "d", "/", "e"), scanner.scan());
    }

    @Test
    void logicalNot() {
        Scanner.Builder b = new Scanner.Builder("!true")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("!", "true"), scanner.scan());
    }

    @Test
    void lessThan() {
        Scanner.Builder b = new Scanner.Builder("x<y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", "<", "y"), scanner.scan());
    }

    @Test
    void greaterThan() {
        Scanner.Builder b = new Scanner.Builder("x>y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", ">", "y"), scanner.scan());
    }

    @Test
    void lessEqualSplitsIntoTwoTokens() {
        Scanner.Builder b = new Scanner.Builder("x<=y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", "<", "=", "y"), scanner.scan());
    }

    @Test
    void greaterEqualSplitsIntoTwoTokens() {
        Scanner.Builder b = new Scanner.Builder("x>=y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", ">", "=", "y"), scanner.scan());
    }

    @Test
    void notEqualSplitsIntoTwoTokens() {
        Scanner.Builder b = new Scanner.Builder("x!=y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", "!", "=", "y"), scanner.scan());
    }

    @Test
    void tripleNestedParens() {
        Scanner.Builder b = new Scanner.Builder("(a+b)*(c-d)/(e+f)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("(", "a", "+", "b", ")", "*", "(", "c", "-", "d", ")", "/", "(", "e", "+", "f", ")"), scanner.scan());
    }

    @Test
    void deeplyNestedParens() {
        Scanner.Builder b = new Scanner.Builder("2*(3+4*(5-1))")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("2", "*", "(", "3", "+", "4", "*", "(", "5", "-", "1", ")", ")"), scanner.scan());
    }

    @Test
    void piTimesRSquared() {
        Scanner.Builder b = new Scanner.Builder("pi*r^2")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("pi", "*", "r", "^", "2"), scanner.scan());
    }

    @Test
    void eulerFormula() {
        Scanner.Builder b = new Scanner.Builder("e^(i*pi)+1")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("e", "^", "(", "i", "*", "pi", ")", "+", "1"), scanner.scan());
    }

    @Test
    void logAndLn() {
        Scanner.Builder b = new Scanner.Builder("log(x)+ln(y)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("log", "(", "x", ")", "+", "ln", "(", "y", ")"), scanner.scan());
    }

    @Test
    void maxOfThree() {
        Scanner.Builder b = new Scanner.Builder("max(a,b,c)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("max", "(", "a", ",", "b", ",", "c", ")"), scanner.scan());
    }

    @Test
    void bracketSubscriptStaysLiteral() {
        Scanner.Builder b = new Scanner.Builder("a[0]+b[1]")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("a[0]", "+", "b[1]"), scanner.scan());
    }

    @Test
    void multiDimBracketSubscript() {
        Scanner.Builder b = new Scanner.Builder("matrix[i][j]")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("matrix[i][j]"), scanner.scan());
    }

    @Test
    void xSquaredPlus3xMinus1() {
        Scanner.Builder b = new Scanner.Builder("x^2+3*x-1")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", "^", "2", "+", "3", "*", "x", "-", "1"), scanner.scan());
    }

    @Test
    void productOfSums() {
        Scanner.Builder b = new Scanner.Builder("(x+1)*(x-1)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("(", "x", "+", "1", ")", "*", "(", "x", "-", "1", ")"), scanner.scan());
    }

    @Test
    void simpleAssignment() {
        Scanner.Builder b = new Scanner.Builder("a=b+c")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("a", "=", "b", "+", "c"), scanner.scan());
    }

    @Test
    void functionDefinition() {
        Scanner.Builder b = new Scanner.Builder("f(x)=x^2")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("f", "(", "x", ")", "=", "x", "^", "2"), scanner.scan());
    }

    @Test
    void multipleVariables() {
        Scanner.Builder b = new Scanner.Builder("x1+x2*x3")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x1", "+", "x2", "*", "x3"), scanner.scan());
    }

    @Test
    void unaryMinus() {
        Scanner.Builder b = new Scanner.Builder("-x+3")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("-", "x", "+", "3"), scanner.scan());
    }

    @Test
    void doubleMinus() {
        Scanner.Builder b = new Scanner.Builder("3--2")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("3", "-", "-", "2"), scanner.scan());
    }

    @Test
    void doublyNestedParens() {
        Scanner.Builder b = new Scanner.Builder("((a+b))")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("(", "(", "a", "+", "b", ")", ")"), scanner.scan());
    }

    @Test
    void functionCallWithArgs() {
        Scanner.Builder b = new Scanner.Builder("rk4(1,2,3)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("rk4", "(", "1", ",", "2", ",", "3", ")"), scanner.scan());
    }

    @Test
    void atSignChainedCalls() {
        Scanner.Builder b = new Scanner.Builder("@(1)(2)(3)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("@", "(", "1", ")", "(", "2", ")", "(", "3", ")"), scanner.scan());
    }

    @Test
    void pipeChain() {
        Scanner.Builder b = new Scanner.Builder("x|y|z")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", "|", "y", "|", "z"), scanner.scan());
    }

    @Test
    void doublePipe() {
        Scanner.Builder b = new Scanner.Builder("x||y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("x", "|", "|", "y"), scanner.scan());
    }

    @Test
    void underscoreIdentifiers() {
        Scanner.Builder b = new Scanner.Builder("a_b+c_d")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("a_b", "+", "c_d"), scanner.scan());
    }

    @Test
    void dollarSignIdentifiers() {
        Scanner.Builder b = new Scanner.Builder("$x+$y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("$x", "+", "$y"), scanner.scan());
    }

    @Test
    void scientificNotationSplitsOnMinus() {
        Scanner.Builder b = new Scanner.Builder("1e10+2e-5")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("1e10", "+", "2e", "-", "5"), scanner.scan());
    }

    @Test
    void quotedAdditionExpr() {
        Scanner.Builder b = new Scanner.Builder("\"a+b\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"a+b\""), scanner.scan());
    }

    @Test
    void quotedAdditionExprSums() {
        Scanner.Builder b = new Scanner.Builder("\"33+22\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"33+22\""), scanner.scan());
    }

    @Test
    void quotedMultiplyMinusExpr() {
        Scanner.Builder b = new Scanner.Builder("\"3*x-1\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"3*x-1\""), scanner.scan());
    }

    @Test
    void quotedExprNotSplitByPlus() {
        Scanner.Builder b = new Scanner.Builder("\"a+b\"+\"c+d\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"a+b\"", "+", "\"c+d\""), scanner.scan());
    }

    @Test
    void singleQuotedMathExpr() {
        Scanner.Builder b = new Scanner.Builder("'single quoted math: a+b*c'")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("'single quoted math: a+b*c'"), scanner.scan());
    }

    @Test
    void quotedExprWithNestedParens() {
        Scanner.Builder b = new Scanner.Builder("\"nested (paren) inside\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"nested (paren) inside\""), scanner.scan());
    }

    @Test
    void diffeqnFullExpression() {
        Scanner.Builder b = new Scanner.Builder("diffeqn(@(10)(\"a+b\",\"33+22\", \"3*x-1\"),4, 11, 0.01, rk4, 100,  state|trajectory)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", "@", "(", "10", ")", "(", "\"a+b\"", ",", "\"33+22\"", ",", "\"3*x-1\"", ")", ",", "4", ",", "11", ",", "0.01", ",", "rk4", ",", "100", ",", "state", "|", "trajectory", ")"), scanner.scan());
    }

    @Test
    void diffeqnSingleSpaceVariant() {
        Scanner.Builder b = new Scanner.Builder("diffeqn(@(10)(\"a+b\",\"33+22\", \"3*x-1\"),4, 11, 0.01, rk4, 100, state|trajectory)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", "@", "(", "10", ")", "(", "\"a+b\"", ",", "\"33+22\"", ",", "\"3*x-1\"", ")", ",", "4", ",", "11", ",", "0.01", ",", "rk4", ",", "100", ",", "state", "|", "trajectory", ")"), scanner.scan());
    }

    @Test
    void diffeqnWithTrailingArithmetic() {
        Scanner.Builder b = new Scanner.Builder("diffeqn(@(10)(\"a+b\",\"33+22\", \"3*x-1\"),4, 11, 0.01, rk4, 100,  state|trajectory)+1")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", "@", "(", "10", ")", "(", "\"a+b\"", ",", "\"33+22\"", ",", "\"3*x-1\"", ")", ",", "4", ",", "11", ",", "0.01", ",", "rk4", ",", "100", ",", "state", "|", "trajectory", ")", "+", "1"), scanner.scan());
    }

    @Test
    void diffeqnWithApostropheInsideDoubleQuotes() {
        Scanner.Builder b = new Scanner.Builder("diffeqn(\"y'=x+y\", 0, 5, 0.1, rk4, 50, state)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", "\"y'=x+y\"", ",", "0", ",", "5", ",", "0.1", ",", "rk4", ",", "50", ",", "state", ")"), scanner.scan());
    }

    @Test
    void diffeqnEmptyArgs() {
        Scanner.Builder b = new Scanner.Builder("diffeqn()")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", ")"), scanner.scan());
    }

    @Test
    void diffeqnEmptyCommaArg() {
        Scanner.Builder b = new Scanner.Builder("diffeqn(,)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", ",", ")"), scanner.scan());
    }

    @Test
    void diffeqnFullExpressionStripped() {
        Scanner.Builder b = new Scanner.Builder("diffeqn(@(10)(\"a+b\",\"33+22\", \"3*x-1\"),4, 11, 0.01, rk4, 100,  state|trajectory)")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0')
                .stripDelimiterMarkers(true);
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("diffeqn", "(", "@", "(", "10", ")", "(", "a+b", ",", "33+22", ",", "3*x-1", ")", ",", "4", ",", "11", ",", "0.01", ",", "rk4", ",", "100", ",", "state", "|", "trajectory", ")"), scanner.scan());
    }

    @Test
    void quotesKeepMarkersByDefault() {
        Scanner.Builder b = new Scanner.Builder("foo(\"hello + world\", 'a,b,c') + x")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "(", ")", ",", "+"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("foo", "(", "\"hello + world\"", ",", "'a,b,c'", ")", "+", "x"), scanner.scan());
    }

    @Test
    void quotesStrippedWhenConfigured() {
        Scanner.Builder b = new Scanner.Builder("foo(\"hello + world\", 'a,b,c') + x")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "(", ")", ",", "+"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0')
                .stripDelimiterMarkers(true);
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("foo", "(", "hello + world", ",", "a,b,c", ")", "+", "x"), scanner.scan());
    }

    @Test
    void emptyDoubleQuotedString() {
        Scanner.Builder b = new Scanner.Builder("\"\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"\""), scanner.scan());
    }

    @Test
    void emptySingleQuotedString() {
        Scanner.Builder b = new Scanner.Builder("''")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("''"), scanner.scan());
    }

    @Test
    void adjacentDifferentQuoteTypes() {
        Scanner.Builder b = new Scanner.Builder("\"double\"'single'")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"double\"", "'single'"), scanner.scan());
    }

    @Test
    void fourConsecutiveDoubleQuotesPairUp() {
        Scanner.Builder b = new Scanner.Builder("\"\"\"\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"\"", "\"\""), scanner.scan());
    }

    @Test
    void quotedCommaNotSplit() {
        Scanner.Builder b = new Scanner.Builder("\"a,b,c\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{",", "+"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"a,b,c\""), scanner.scan());
    }

    @Test
    void quotedPlusNotSplit() {
        Scanner.Builder b = new Scanner.Builder("\"a+b\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"+"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"a+b\""), scanner.scan());
    }

    @Test
    void escapedQuoteDoesNotTerminateRegion() {
        Scanner.Builder b = new Scanner.Builder("say(\"a \\\"b\\\" c\")")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"say", "(", ")"})
                .addDelimitedRegion("\"", "\"", '\\');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("say", "(", "\"a \\\"b\\\" c\"", ")"), scanner.scan());
    }

    @Test
    void withoutEscapeCharInnerQuoteTerminatesEarly() {
        Scanner.Builder b = new Scanner.Builder("say(\"a \\\"b\\\" c\")")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"say", "(", ")"})
                .addDelimitedRegion("\"", "\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("say", "(", "\"a \\\"", "b\\", "\" c\"", ")"), scanner.scan());
    }

    @Test
    void blockCommentTreatedAsOpaqueRegion() {
        Scanner.Builder b = new Scanner.Builder("a+b /* this + is - a * comment */ + c")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0')
                .addDelimitedRegion("/*", "*/", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("a", "+", "b", "/* this + is - a * comment */", "+", "c"), scanner.scan());
    }

    @Test
    void customDoubleBraceDelimiter() {
        Scanner.Builder b = new Scanner.Builder("template {{ raw + text * here }} end")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!", "{", "}"})
                .addDelimitedRegion("{{", "}}", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("template", "{{ raw + text * here }}", "end"), scanner.scan());
    }

    @Test
    void longerStartMarkerTakesPrecedence() {
        Scanner.Builder b = new Scanner.Builder("\"\"\"triple \"inner\" quoted\"\"\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"\"\"", "\"\"\"", '\0')
                .addDelimitedRegion("\"", "\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"\"\"triple \"inner\" quoted\"\"\""), scanner.scan());
    }

    @Test
    void withoutTripleRegisteredSingleQuoteSplitsIt() {
        Scanner.Builder b = new Scanner.Builder("\"\"\"triple \"inner\" quoted\"\"\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"\"", "\"triple \"", "inner", "\" quoted\"", "\"\""), scanner.scan());
    }

    @Test
    void unterminatedConsumeToEndDefault() {
        Scanner.Builder b = new Scanner.Builder("foo(\"hello + x * y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "(", ")", "+", "*"})
                .addDelimitedRegion("\"", "\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("foo", "(", "\"hello + x * y"), scanner.scan());
    }

    @Test
    void unterminatedTreatAsNoMatchFallsBackToNormalTokenizing() {
        Scanner.Builder b = new Scanner.Builder("foo(\"hello + x * y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "(", ")", "+", "*"})
                .addDelimitedRegion("\"", "\"", '\0')
                .onUnterminatedDelimiter(Scanner.UnterminatedDelimiterPolicy.TREAT_AS_NO_MATCH);
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("foo", "(", "\"hello", "+", "x", "*", "y"), scanner.scan());
    }

    @Test
    void unterminatedThrowRaisesException() {
        Scanner.Builder b = new Scanner.Builder("foo(\"hello + x * y")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"foo", "(", ")", "+", "*"})
                .addDelimitedRegion("\"", "\"", '\0')
                .onUnterminatedDelimiter(Scanner.UnterminatedDelimiterPolicy.THROW);
        Scanner scanner = b.build();
        Scanner.UnterminatedDelimiterException ex = assertThrows(
                Scanner.UnterminatedDelimiterException.class, scanner::scan);
        assertTrue(ex.getMessage().contains("Unterminated"));
    }

    @Test
    void emptyInputProducesEmptyList() {
        Scanner.Builder b = new Scanner.Builder("")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Collections.emptyList(), scanner.scan());
    }

    @Test
    void wholeInputIsJustAQuoteChar() {
        Scanner.Builder b = new Scanner.Builder("\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\""), scanner.scan());
    }

    @Test
    void trailingLoneQuoteAfterIdentifier() {
        Scanner.Builder b = new Scanner.Builder("abc\"")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"", "\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("abc", "\""), scanner.scan());
    }

    @Test
    void startMarkerLongerThanRemainingInput() {
        Scanner.Builder b = new Scanner.Builder("\"a")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"(", ")", ",", "@", "+", "-", "*", "/", "^", "|", "=", "<", ">", "!"})
                .addDelimitedRegion("\"\"\"", "\"\"\"", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("\"a"), scanner.scan());
    }

    @Test
    void wordBoundaryBetweenStaticTokenAndIdentifier() {
        Scanner.Builder b = new Scanner.Builder("index in range")
                .ignoreWhitespace(true)
                .addTokens(new String[]{"in"})
                .addDelimitedRegion("\"", "\"", '\0')
                .addDelimitedRegion("'", "'", '\0');
        Scanner scanner = b.build();
        assertEquals(Arrays.asList("index", "in", "range"), scanner.scan());
    }

}