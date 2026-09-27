package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.ast.*;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Finds which identifiers a raw ParserNG expression text may read as variables. Deliberately
 * OVER-approximate: a spurious extra column only costs some extra decoding, while a missing one would
 * make the query fail (or worse). Scanning mirrors {@code WhereAliasResolver}: quoted strings are
 * skipped, and an identifier directly followed by {@code (} is a function call, not a column.
 */
final class ColumnRefs {

    private ColumnRefs() { }

    static void identifiers(String text, Set<String> out) {
        if (text == null) return;
        int n = text.length(), i = 0;
        while (i < n) {
            char c = text.charAt(i);
            if (c == '\'' || c == '"') {
                i = skipString(text, i, c);
            } else if (Character.isLetter(c) || c == '_') {
                int s = i++;
                while (i < n && (Character.isLetterOrDigit(text.charAt(i)) || text.charAt(i) == '_')) i++;
                int j = i;
                while (j < n && Character.isWhitespace(text.charAt(j))) j++;
                if (!(j < n && text.charAt(j) == '(')) out.add(text.substring(s, i));
            } else {
                i++;
            }
        }
    }

    private static int skipString(String t, int start, char q) {
        int i = start + 1, n = t.length();
        while (i < n) {
            if (t.charAt(i) == q) {
                if (i + 1 < n && t.charAt(i + 1) == q) { i += 2; continue; }
                return i + 1;
            }
            i++;
        }
        return n;
    }

    /** Every identifier any clause of {@code st} could read as a column. */
    static Set<String> referencedBy(SelectStatement st) {
        Set<String> ids = new LinkedHashSet<>();
        for (SelectItem it : st.items()) {
            identifiers(it.isAggregate() ? it.aggregate().argExprText() : it.exprText(), ids);
        }
        collect(st.where(), ids);
        for (String g : st.groupBy()) identifiers(g, ids);
        collect(st.having(), ids);
        for (OrderItem o : st.orderBy()) identifiers(o.exprText(), ids);
        return ids;
    }

    private static void collect(BoolExpr e, Set<String> ids) {
        if (e == null) return;
        switch (e) {
            case AndExpr(var l, var r) -> { collect(l, ids); collect(r, ids); }
            case OrExpr(var l, var r) -> { collect(l, ids); collect(r, ids); }
            case NotExpr(var inner) -> collect(inner, ids);
            case ComparisonExpr(var l, var op, var r) -> { identifiers(l, ids); identifiers(r, ids); }
            case BetweenExpr(var t, var lo, var hi, var neg) -> { identifiers(t, ids); identifiers(lo, ids); identifiers(hi, ids); }
            case InExpr(var t, var vs, var neg) -> { identifiers(t, ids); for (String v : vs) identifiers(v, ids); }
            case IsNullExpr(var t, var neg) -> identifiers(t, ids);
        }
    }

    static Set<String> referencedBy(Collection<String> texts) {
        Set<String> ids = new LinkedHashSet<>();
        for (String t : texts) identifiers(t, ids);
        return ids;
    }
}
