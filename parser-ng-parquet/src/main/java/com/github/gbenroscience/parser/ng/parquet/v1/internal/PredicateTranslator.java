package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import org.apache.parquet.filter2.predicate.FilterApi;
import org.apache.parquet.filter2.predicate.FilterPredicate;
import org.apache.parquet.filter2.predicate.Operators;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.parquet.schema.Type;

import java.time.LocalDate;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Translates a {@link Predicate} into a parquet-java {@link FilterPredicate} used ONLY for pruning.
 * Returns {@code null} when nothing can be safely pushed. See {@link Predicate} for the soundness
 * rules, and {@link #negate} for specifically how {@link Predicate.Not} is handled.
 */
public final class PredicateTranslator {

    private PredicateTranslator() {}

    /** @return a pruning predicate, or null if no part of {@code p} is safely pushable. */
    public static FilterPredicate translate(Predicate p, MessageType schema) {
        if (p == null) return null;
        if (p instanceof Predicate.And a) {
            FilterPredicate l = translate(a.left(), schema), r = translate(a.right(), schema);
            if (l == null) return r;
            if (r == null) return l;
            return FilterApi.and(l, r);
        }
        if (p instanceof Predicate.Or o) {
            FilterPredicate l = translate(o.left(), schema), r = translate(o.right(), schema);
            return (l == null || r == null) ? null : FilterApi.or(l, r);
        }
        if (p instanceof Predicate.Cmp c) return cmp(c, schema);
        if (p instanceof Predicate.In in) return inList(in, schema);
        if (p instanceof Predicate.IsNull n) return isNull(n, schema);
        if (p instanceof Predicate.Not not) {
            Predicate inner = not.inner();
            // Fast path: these four leaf types never push regardless of negation (see their own
            // Javadoc), so there is no point building the De Morgan expansion just to discover that.
            // Not required for correctness -- translate()'s fallthrough already returns null for any
            // of these -- purely an optimization for the common "Not wrapping one directly" case.
            if (inner instanceof Predicate.ColCmp || inner instanceof Predicate.Like
                    || inner instanceof Predicate.Regex || inner instanceof Predicate.Custom) {
                return null;
            }
            return translate(negate(inner), schema);
        }
        // Predicate.ColCmp / Predicate.Like / Predicate.Regex / Predicate.Custom: none of parquet-java's
        // FilterApi has any concept of these (column-to-column, patterns, or an opaque caller callback),
        // so they always fall through here -- never pushed, always sound (see Predicate's Javadoc for
        // exactly why each one is a hard limit of pruning, not a translation gap).
        return null;
    }

    /**
     * Rewrites {@code NOT(p)} to negation-normal form -- De Morgan's laws pushing the negation down
     * to each leaf -- rather than ever handing parquet-java's {@code FilterApi} a literal
     * {@code not(...)}. This is deliberate, not merely equivalent: negating a range comparison is
     * exactly the case parquet-java's own {@code LogicalInverseRewriter} exists to handle carefully
     * for statistics-based pruning (raw {@code not()} composed with stats-based row-group filtering
     * is a well-documented parquet-java footgun). Rewriting here, before translation, removes any
     * dependency on that being applied correctly downstream: every leaf this method ever hands
     * {@link #translate} is already a plain, non-negated comparison.
     *
     * <p>{@link Predicate.In} has no direct negated counterpart in parquet-java's algebra, so
     * {@code NOT(c IN (v1, v2, ...))} expands to an {@link Predicate.And} chain of {@code c != v1},
     * {@code c != v2}, ... -- the same expansion {@code parser-ng-sql}'s own WHERE-clause splitter
     * already performs upstream, mirrored here for callers who build a raw {@link Predicate} directly
     * without going through that layer.
     */
    private static Predicate negate(Predicate p) {
        if (p instanceof Predicate.And a) return Predicate.or(negate(a.left()), negate(a.right()));
        if (p instanceof Predicate.Or o) return Predicate.and(negate(o.left()), negate(o.right()));
        if (p instanceof Predicate.Not n) return n.inner(); // double negation cancels
        if (p instanceof Predicate.Cmp c) return new Predicate.Cmp(c.column(), flip(c.op()), c.value());
        if (p instanceof Predicate.In in) {
            Predicate acc = null;
            for (Object v : in.values()) {
                Predicate ne = Predicate.ne(in.column(), v);
                acc = (acc == null) ? ne : Predicate.and(acc, ne);
            }
            return acc; // In's compact constructor rejects an empty values list, so acc is never null here
        }
        if (p instanceof Predicate.IsNull n) return new Predicate.IsNull(n.column(), !n.negated());
        // ColCmp / Like / Regex / Custom: never pushed regardless of sign. Wrapping in Not rather than
        // returning the bare node is not load-bearing for correctness here (translate()'s own
        // fallthrough already returns null for any of these, wrapped or not) -- it just keeps this
        // method's output uniformly "a Predicate", not "a Predicate, except sometimes still negated."
        return new Predicate.Not(p);
    }

    private static Predicate.Op flip(Predicate.Op op) {
        return switch (op) {
            case EQ -> Predicate.Op.NE;
            case NE -> Predicate.Op.EQ;
            case LT -> Predicate.Op.GE;
            case LE -> Predicate.Op.GT;
            case GT -> Predicate.Op.LE;
            case GE -> Predicate.Op.LT;
        };
    }

    private static PrimitiveType prim(MessageType schema, String col) {
        if (col.indexOf('.') >= 0 || !schema.containsField(col)) return null; // dotted paths are ambiguous
        Type t = schema.getType(col);
        return t.isPrimitive() ? t.asPrimitiveType() : null;
    }

    private static boolean signedInt(LogicalTypeAnnotation lt, int width) {
        return lt instanceof LogicalTypeAnnotation.IntLogicalTypeAnnotation i && i.isSigned() && i.getBitWidth() == width;
    }

    // ---- literal coercion: exact or exactly-widening only; null => not pushable ----

    private static Integer asInt32(PrimitiveType pt, Object v) {
        LogicalTypeAnnotation lt = pt.getLogicalTypeAnnotation();
        if (lt == null || signedInt(lt, 32)) return v instanceof Integer i ? i : null;
        if (lt instanceof LogicalTypeAnnotation.DateLogicalTypeAnnotation) {
            if (v instanceof LocalDate d) {
                long day = d.toEpochDay();
                return (day >= Integer.MIN_VALUE && day <= Integer.MAX_VALUE) ? (Integer) (int) day : null;
            }
        }
        return null;
    }

    private static Long asInt64(PrimitiveType pt, Object v) {
        LogicalTypeAnnotation lt = pt.getLogicalTypeAnnotation();
        if (lt != null && !signedInt(lt, 64)) return null;
        if (v instanceof Long l) return l;
        if (v instanceof Integer i) return (long) i;
        return null;
    }

    private static Float asFloat(PrimitiveType pt, Object v) {
        if (pt.getLogicalTypeAnnotation() != null || !(v instanceof Float f) || f.isNaN()) return null;
        return f;
    }

    private static Double asDouble(PrimitiveType pt, Object v) {
        if (pt.getLogicalTypeAnnotation() != null) return null;
        Double d = null;
        if (v instanceof Double x) d = x;
        else if (v instanceof Float x) d = (double) x;
        else if (v instanceof Integer x) d = (double) x;
        return (d == null || d.isNaN()) ? null : d;
    }

    private static Binary asString(PrimitiveType pt, Object v) {
        return (pt.getLogicalTypeAnnotation() instanceof LogicalTypeAnnotation.StringLogicalTypeAnnotation
                && v instanceof String s) ? Binary.fromString(s) : null;
    }

    // ---- leaves ----

    private static FilterPredicate cmp(Predicate.Cmp c, MessageType schema) {
        PrimitiveType pt = prim(schema, c.column());
        if (pt == null) return null;
        String n = c.column();
        Object v = c.value();
        Predicate.Op op = c.op();
        switch (pt.getPrimitiveTypeName()) {
            case INT32: {
                Integer x = asInt32(pt, v);
                return x == null ? null : ord(FilterApi.intColumn(n), op, x);
            }
            case INT64: {
                Long x = asInt64(pt, v);
                return x == null ? null : ord(FilterApi.longColumn(n), op, x);
            }
            case FLOAT: {
                Float x = asFloat(pt, v);
                return x == null ? null : ord(FilterApi.floatColumn(n), op, x);
            }
            case DOUBLE: {
                Double x = asDouble(pt, v);
                return x == null ? null : ord(FilterApi.doubleColumn(n), op, x);
            }
            case BOOLEAN: {
                if (!(v instanceof Boolean b) || pt.getLogicalTypeAnnotation() != null) return null;
                if (op == Predicate.Op.EQ) return FilterApi.eq(FilterApi.booleanColumn(n), b);
                if (op == Predicate.Op.NE) return FilterApi.notEq(FilterApi.booleanColumn(n), b);
                return null;
            }
            case BINARY: {
                Binary x = asString(pt, v);
                if (x == null) return null;
                if (op == Predicate.Op.EQ) return FilterApi.eq(FilterApi.binaryColumn(n), x);
                if (op == Predicate.Op.NE) return FilterApi.notEq(FilterApi.binaryColumn(n), x);
                return null; // no string range pushdown, see Predicate javadoc
            }
            default:
                return null;
        }
    }

    private static <T extends Comparable<T>, C extends Operators.Column<T> & Operators.SupportsLtGt>
            FilterPredicate ord(C col, Predicate.Op op, T v) {
        switch (op) {
            case EQ: return FilterApi.eq(col, v);
            case NE: return FilterApi.notEq(col, v);
            case LT: return FilterApi.lt(col, v);
            case LE: return FilterApi.ltEq(col, v);
            case GT: return FilterApi.gt(col, v);
            case GE: return FilterApi.gtEq(col, v);
            default: return null;
        }
    }

    private static FilterPredicate inList(Predicate.In in, MessageType schema) {
        PrimitiveType pt = prim(schema, in.column());
        if (pt == null) return null;
        String n = in.column();
        List<?> vals = in.values();
        switch (pt.getPrimitiveTypeName()) {
            case INT32: {
                Set<Integer> s = new LinkedHashSet<>();
                for (Object o : vals) { Integer x = asInt32(pt, o); if (x == null) return null; s.add(x); }
                return FilterApi.in(FilterApi.intColumn(n), s);
            }
            case INT64: {
                Set<Long> s = new LinkedHashSet<>();
                for (Object o : vals) { Long x = asInt64(pt, o); if (x == null) return null; s.add(x); }
                return FilterApi.in(FilterApi.longColumn(n), s);
            }
            case DOUBLE: {
                Set<Double> s = new LinkedHashSet<>();
                for (Object o : vals) { Double x = asDouble(pt, o); if (x == null) return null; s.add(x); }
                return FilterApi.in(FilterApi.doubleColumn(n), s);
            }
            case FLOAT: {
                Set<Float> s = new LinkedHashSet<>();
                for (Object o : vals) { Float x = asFloat(pt, o); if (x == null) return null; s.add(x); }
                return FilterApi.in(FilterApi.floatColumn(n), s);
            }
            case BINARY: {
                Set<Binary> s = new LinkedHashSet<>();
                for (Object o : vals) { Binary x = asString(pt, o); if (x == null) return null; s.add(x); }
                return FilterApi.in(FilterApi.binaryColumn(n), s);
            }
            default:
                return null;
        }
    }

    /** IS NULL == eq(col, null); IS NOT NULL == notEq(col, null) in parquet's filter algebra. */
    private static FilterPredicate isNull(Predicate.IsNull p, MessageType schema) {
        PrimitiveType pt = prim(schema, p.column());
        if (pt == null) return null;
        String n = p.column();
        boolean neg = p.negated();
        switch (pt.getPrimitiveTypeName()) {
            case INT32: return neg ? FilterApi.notEq(FilterApi.intColumn(n), (Integer) null) : FilterApi.eq(FilterApi.intColumn(n), (Integer) null);
            case INT64: return neg ? FilterApi.notEq(FilterApi.longColumn(n), (Long) null) : FilterApi.eq(FilterApi.longColumn(n), (Long) null);
            case FLOAT: return neg ? FilterApi.notEq(FilterApi.floatColumn(n), (Float) null) : FilterApi.eq(FilterApi.floatColumn(n), (Float) null);
            case DOUBLE: return neg ? FilterApi.notEq(FilterApi.doubleColumn(n), (Double) null) : FilterApi.eq(FilterApi.doubleColumn(n), (Double) null);
            case BOOLEAN: return neg ? FilterApi.notEq(FilterApi.booleanColumn(n), (Boolean) null) : FilterApi.eq(FilterApi.booleanColumn(n), (Boolean) null);
            case BINARY: return neg ? FilterApi.notEq(FilterApi.binaryColumn(n), (Binary) null) : FilterApi.eq(FilterApi.binaryColumn(n), (Binary) null);
            default: return null;
        }
    }
}
