/*
 * Optimized DataSetFormatter1 - 2025 Edition (Java 8 Compatible)
 * 
 * Preprocesses comma-separated function arguments by intelligently wrapping
 * complex expressions in parentheses. This makes variable-argument functions
 * (sum, sort, mode, etc.) much easier for MathExpression to parse.
 * 
 * Improvements over 2011 version:
 *   • Safer list processing (new-list builder, no index hell)
 *   • Modern Java 8 features and style
 *   • Cleaner, more maintainable code
 *   • Defensive copies + null safety
 *   • Exact same bracketing rules and output
 * 
 * @since 2011 (original) / 2025 (optimized)
 * @author JIBOYE Oluwagbemiro Olaoluwa (original logic)
 * @author Optimized 2025
 */
package com.github.gbenroscience.parser;
/*
 * DataSetFormatter1 v3 - Ultra Fast 2025 Edition (Java 8 Compatible)
 * 
 * Blazing fast argument bracketing with minimal allocations.
 * Produces 100% identical output to previous versions.
 */


import com.github.gbenroscience.parser.methods.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import static com.github.gbenroscience.parser.Variable.*;
import static com.github.gbenroscience.parser.Number.*;
import static com.github.gbenroscience.parser.Bracket.*;

public class DataSetFormatter {

    /** The trimmed input; kept so the token list can be produced lazily. */
    private final String text;
    /** Tokenised and bracketed form. Null until needed when the fast path applied. */
    private List<String> dataset;
    /** Cached result of {@link #getFormattedDataSet()}. */
    private String formatted;

    public static final String COMMA_MASK = "?";
    public static final String OPEN_CIRC_BRAC_MASK = "<<<<";
    public static final String CLOSE_CIRC_BRAC_MASK = ">>>>";
    

    public DataSetFormatter(String text) {
        Objects.requireNonNull(text, "Expression text cannot be null");
        this.text = text.trim();
        if (needsProcessing(this.text)) {
            this.dataset = processExpression(this.text);
        } else {
            // No closing bracket and no mask look-alike: the token stream
            // concatenates back to exactly the input and nothing is rewritten, so
            // skip tokenising altogether.
            this.formatted = this.text;
        }
    }

    /**
     * The formatter only ever rewrites text around a ')' token, and
     * {@link #restoreMasks} only touches tokens that equal one of the three mask
     * strings. If the text contains none of those characters, the output is the
     * input verbatim.
     */
    private static boolean needsProcessing(String t) {
        return t.indexOf(')') >= 0
                || t.indexOf('?') >= 0
                || t.contains(OPEN_CIRC_BRAC_MASK)
                || t.contains(CLOSE_CIRC_BRAC_MASK);
    }

    public List<String> getDataset() {
        if (dataset == null) {
            dataset = processExpression(text);
        }
        return new ArrayList<>(dataset); // defensive copy
    }

    public String getFormattedDataSet() {
        String f = formatted;
        if (f == null) {
            StringBuilder sb = new StringBuilder(text.length() + 16);
            for (String token : dataset) {
                sb.append(token);
            }
            f = sb.toString();
            formatted = f;
        }
        return f;
    }

    private List<String> processExpression(String expression) {
        // Same vocabulary as before (all function names, comma, round and square
        // brackets) but the table is cached instead of rebuilt for every input.
        Scanner csc = TokenTables.dataSetScanner(expression);

        List<String> tokens = csc.scan();

        for (int i = 0; i < tokens.size(); i++) {
            if (isCloseBracket(tokens.get(i))) {
                int closeIdx = i;
                int openIdx = Bracket.getComplementIndex(false, Bracket.BracketMode.CIRCULAR_CLOSE, closeIdx, tokens);

                tokens.set(openIdx, OPEN_CIRC_BRAC_MASK);
                tokens.set(closeIdx, CLOSE_CIRC_BRAC_MASK);

                wrapArgumentsInPlace(tokens, openIdx + 1, closeIdx);
            }
        }

        restoreMasks(tokens);
        return tokens;
    }

    /**
     * v3 core: index-based groups + manual copy + single buffer per bracket
     */
    private void wrapArgumentsInPlace(List<String> tokens, int start, int end) {
        // Single argument (no comma): the rewrite is either a no-op or just a pair
        // of masked brackets around it, so skip the buffer and the sublist clear.
        boolean hasComma = false;
        for (int i = start; i < end; i++) {
            if (",".equals(tokens.get(i))) {
                hasComma = true;
                break;
            }
        }
        if (!hasComma) {
            if (start < end && shouldWrap(tokens, start, end)) {
                tokens.add(end, CLOSE_CIRC_BRAC_MASK);
                tokens.add(start, OPEN_CIRC_BRAC_MASK);
            }
            return;
        }

        List<String> buffer = new ArrayList<>((end - start) * 3 / 2 + 8);
        int argStart = start;

        for (int i = start; i <= end; i++) {
            boolean atEnd = (i == end);
            boolean isComma = !atEnd && ",".equals(tokens.get(i));

            if (atEnd || isComma) {
                if (argStart < i) {
                    if (shouldWrap(tokens, argStart, i)) {
                        buffer.add(OPEN_CIRC_BRAC_MASK); 
                        copyTokens(tokens, buffer, argStart, i);
                        buffer.add(CLOSE_CIRC_BRAC_MASK); 
                    } else {
                        copyTokens(tokens, buffer, argStart, i);
                    }
                }
                if (!atEnd) {
                    buffer.add(COMMA_MASK);
                }
                argStart = i + 1;
            }
        }

        // One-shot replace (very fast)
        tokens.subList(start, end).clear();
        tokens.addAll(start, buffer);
    }

    private static void copyTokens(List<String> source, List<String> dest, int from, int to) {
        for (int j = from; j < to; j++) {
            dest.add(source.get(j));
        }
    }

    private boolean shouldWrap(List<String> tokens, int from, int to) {
        if (from >= to) return false;
        String first = tokens.get(from);
        int length = to - from;

        return !isAtOperator(first)
                && !Method.isListReturningStatsMethod(first)
                && !Method.isFunctionOperatingMethod(first)
                && (length > 2 || (!validNumber(first) 
                && !isVariableString(first)));
    }

    private void restoreMasks(List<String> tokens) {
        for (int i = 0; i < tokens.size(); i++) {
            String t = tokens.get(i);
            switch (t) {
                case COMMA_MASK: tokens.set(i, ","); break;
                case OPEN_CIRC_BRAC_MASK: tokens.set(i, "("); break;
                case CLOSE_CIRC_BRAC_MASK: tokens.set(i, ")"); break; 
            }
        }
    }

 
    static String testOld(int m, int n) {
        StringBuilder huge = new StringBuilder("sum(");
        for (int i = 0; i < m; i++) {
            huge.append("1+2*3-4,");
        }
        huge.append("5)");
        String bigExpr = huge.toString();
        String txt = null;
        long start = System.nanoTime();
        for (int i = 0; i < n; i++) {
            txt = new DataSetFormatter1(bigExpr).getFormattedDataSet();
        }
        long time = (System.nanoTime() - start) / n;

        System.out.println("Time: " + (time / 1_000_000.0) + " ms");
        System.out.println("old-out.length: " + txt.length());
        return txt;
    }

    static String testNew(int m, int n) {
        StringBuilder huge = new StringBuilder("sum(");
        for (int i = 0; i < m; i++) {
            huge.append("1+2*3-4,");
        }
        huge.append("5)");
        String bigExpr = huge.toString();
        String txt = null;
        long start = System.nanoTime();
        for (int i = 0; i < n; i++) {
            txt = new DataSetFormatter(bigExpr).getFormattedDataSet();
        }
        long time = (System.nanoTime() - start) / n;

        System.out.println("Time: " + (time / 1_000_000.0) + " ms");
        System.out.println("new-out.length: " + txt.length());
        return txt;
    }

    public static void main(String[] args) {
        String simple = "sum(1+1,3*2)";
        String complex = "sum(sin(3),cos(3),ln(345),sort(3,-4,5,-6,13,2,4,5,sum(3,4,5,6,9,12,23),sum(3,4,8,9,2000)),12000,mode(3,2,2,1),mode(1,5,7,7,1,1,7))";
        DataSetFormatter f1 = new DataSetFormatter(simple);
        System.out.println("Original : " + simple);
        System.out.println("Formatted: " + f1.getFormattedDataSet());
        
        
        DataSetFormatter f2 = new DataSetFormatter(complex);
        System.out.println("Original : " + complex);
        System.out.println("Formatted: " + f2.getFormattedDataSet());

       String t1 = testOld(10, 100000);
       String t2 = testNew(10, 100000);
       System.out.println("output is same: "+(t1.equals(t2)));
    }

}