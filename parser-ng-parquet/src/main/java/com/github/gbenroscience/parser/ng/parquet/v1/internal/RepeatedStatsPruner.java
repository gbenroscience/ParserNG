package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;
import org.apache.parquet.column.statistics.Statistics;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ColumnChunkMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Row-group pruning for predicates over leaves inside LISTs and MAPs, from footer column statistics.
 *
 * <p>parquet-java cannot do this itself: its {@code FilterApi} validator rejects repeated columns outright,
 * so {@link PredicateTranslator} never hands it such a leaf. This class fills the gap for the common
 * "events with a list of line items" shape: it removes row groups from the footer before any reader is
 * opened over it, and the surviving footer then goes through parquet-java's own pruning as usual (the two
 * stages are independent necessary conditions, so a group is read only if both allow it).
 *
 * <h2>Meaning of a predicate over a repeated leaf</h2>
 * "At least one element satisfies it" (a row with an empty or null list has no element, so it satisfies
 * nothing). Column statistics summarise every element of the chunk, which is exactly what that question
 * needs: if no element in the whole chunk can satisfy the comparison, no row can. A null or missing
 * statistic, a literal that does not coerce, an unsupported type or operator: all mean "cannot rule the
 * group out", never "prune".
 *
 * <h2>Why negation is not pushed</h2>
 * {@code NOT (some element > 5)} means "no element > 5", which is not "some element <= 5" (an empty list
 * satisfies the first and not the second). So under a {@code Not}, a repeated leaf is treated as unknown.
 * De Morgan is still applied above the leaves (via a negation flag), so the flat and struct-nested siblings
 * of such a leaf keep whatever pruning they had.
 *
 * <p>Supported here: INT32/INT64 (including DATE and INT32/INT64-backed DECIMAL), FLOAT/DOUBLE (never
 * {@code NE}, and never when a statistic is NaN), BOOLEAN ({@code EQ}/{@code NE}) and STRING
 * ({@code EQ}/{@code IN}, compared as unsigned bytes like the statistics themselves), plus
 * {@code IS NOT NULL} ({@code IS NULL} is never pruned on).
 */
public final class RepeatedStatsPruner {

    private RepeatedStatsPruner() {}

    /**
     * @return {@code footer} itself if nothing can be or was pruned, otherwise a footer holding only the
     *         row groups that may contain a match (same file metadata, so reader schemas are unaffected)
     */
    public static ParquetMetadata prune(Predicate predicate, ParquetMetadata footer) {
        if (predicate == null) return footer;
        Ctx ctx = new Ctx(footer.getFileMetaData().getSchema());
        if (!touchesRepeated(predicate, ctx)) return footer; // nothing for this stage to do: keep the common path untouched
        List<BlockMetaData> kept = new ArrayList<>();
        for (BlockMetaData b : footer.getBlocks()) if (mayMatch(predicate, false, ctx, b)) kept.add(b);
        if (kept.size() == footer.getBlocks().size()) return footer;
        return new ParquetMetadata(footer.getFileMetaData(), kept);
    }

    private static final class Ctx {
        final MessageType schema;
        final Map<String, ColumnPaths.Resolved> cache = new HashMap<>();

        Ctx(MessageType schema) { this.schema = schema; }

        /** The leaf if {@code column} is a repeated leaf, else null (flat/struct leaves belong to parquet-java). */
        ColumnPaths.Resolved repeatedLeaf(String column) {
            ColumnPaths.Resolved r = cache.computeIfAbsent(column, c -> ColumnPaths.resolve(schema, c));
            return (r != null && r.repeated()) ? r : null;
        }
    }

    private static boolean touchesRepeated(Predicate p, Ctx ctx) {
        if (p instanceof Predicate.And a) return touchesRepeated(a.left(), ctx) || touchesRepeated(a.right(), ctx);
        if (p instanceof Predicate.Or o) return touchesRepeated(o.left(), ctx) || touchesRepeated(o.right(), ctx);
        if (p instanceof Predicate.Not n) return touchesRepeated(n.inner(), ctx);
        if (p instanceof Predicate.Cmp c) return ctx.repeatedLeaf(c.column()) != null;
        if (p instanceof Predicate.In in) return ctx.repeatedLeaf(in.column()) != null;
        if (p instanceof Predicate.IsNull n) return ctx.repeatedLeaf(n.column()) != null;
        return false;
    }

    /** Three-valued in effect: false only when the row group provably holds no match; true means "cannot tell". */
    private static boolean mayMatch(Predicate p, boolean negated, Ctx ctx, BlockMetaData block) {
        if (p instanceof Predicate.And a) {
            return negated
                    ? mayMatch(a.left(), true, ctx, block) || mayMatch(a.right(), true, ctx, block)
                    : mayMatch(a.left(), false, ctx, block) && mayMatch(a.right(), false, ctx, block);
        }
        if (p instanceof Predicate.Or o) {
            return negated
                    ? mayMatch(o.left(), true, ctx, block) && mayMatch(o.right(), true, ctx, block)
                    : mayMatch(o.left(), false, ctx, block) || mayMatch(o.right(), false, ctx, block);
        }
        if (p instanceof Predicate.Not n) return mayMatch(n.inner(), !negated, ctx, block);
        if (negated) return true; // a repeated leaf's "some element" does not survive negation (see class doc)

        String column;
        if (p instanceof Predicate.Cmp c) column = c.column();
        else if (p instanceof Predicate.In in) column = in.column();
        else if (p instanceof Predicate.IsNull n) column = n.column();
        else return true;

        ColumnPaths.Resolved r = ctx.repeatedLeaf(column);
        if (r == null) return true;
        ColumnChunkMetaData chunk = chunk(block, r.physicalPath());
        if (chunk == null) return true;
        Statistics<?> st = chunk.getStatistics();
        if (st == null) return true;

        boolean allNull = st.isNumNullsSet() && st.getNumNulls() == chunk.getValueCount(); // no non-null element anywhere

        if (p instanceof Predicate.IsNull n) {
            // IS NOT NULL: some element is non-null, ruled out only when every entry is null. IS NULL is never
            // pruned: an element can be null because a struct around it is null, and whether the chunk's null
            // count includes that is up to the writer (see PredicateTranslator's IS NULL note).
            return !n.negated() || !allNull;
        }
        if (allNull) return false; // every comparison needs a non-null element
        if (!st.hasNonNullValue()) return true; // statistics unusable: cannot tell
        try {
            if (p instanceof Predicate.Cmp c) return cmpMayMatch(r.leaf(), st, c.op(), c.value());
            for (Object v : ((Predicate.In) p).values()) {
                if (cmpMayMatch(r.leaf(), st, Predicate.Op.EQ, v)) return true; // an uncoercible literal also answers true
            }
            return false;
        } catch (ClassCastException e) {
            return true; // a statistic of an unexpected class: never prune on it
        }
    }

    private static ColumnChunkMetaData chunk(BlockMetaData block, String[] path) {
        for (ColumnChunkMetaData cc : block.getColumns()) {
            if (Arrays.equals(cc.getPath().toArray(), path)) return cc;
        }
        return null;
    }

    private static boolean cmpMayMatch(PrimitiveType pt, Statistics<?> st, Predicate.Op op, Object v) {
        Object min = st.genericGetMin(), max = st.genericGetMax();
        if (min == null || max == null) return true;
        LogicalTypeAnnotation lt = pt.getLogicalTypeAnnotation();
        switch (pt.getPrimitiveTypeName()) {
            case INT32: {
                Integer lit = PredicateTranslator.asInt32(pt, v);
                if (lit == null && lt instanceof LogicalTypeAnnotation.DecimalLogicalTypeAnnotation dec) {
                    lit = PredicateTranslator.asDecimalInt32(dec, v);
                }
                return lit == null || range((Integer) min, (Integer) max, op, lit, true);
            }
            case INT64: {
                Long lit = PredicateTranslator.asInt64(pt, v);
                if (lit == null && lt instanceof LogicalTypeAnnotation.DecimalLogicalTypeAnnotation dec) {
                    lit = PredicateTranslator.asDecimalInt64(dec, v);
                }
                return lit == null || range((Long) min, (Long) max, op, lit, true);
            }
            case FLOAT: {
                Float lit = PredicateTranslator.asFloat(pt, v);
                Float lo = (Float) min, hi = (Float) max;
                if (lit == null || lo.isNaN() || hi.isNaN()) return true;
                return range(lo, hi, op, lit, false);
            }
            case DOUBLE: {
                Double lit = PredicateTranslator.asDouble(pt, v);
                Double lo = (Double) min, hi = (Double) max;
                if (lit == null || lo.isNaN() || hi.isNaN()) return true;
                return range(lo, hi, op, lit, false);
            }
            case BOOLEAN: {
                if (!(v instanceof Boolean b) || lt != null) return true;
                if (op != Predicate.Op.EQ && op != Predicate.Op.NE) return true;
                return range((Boolean) min, (Boolean) max, op, b, true);
            }
            case BINARY: {
                Binary lit = PredicateTranslator.asString(pt, v);
                if (lit == null || op != Predicate.Op.EQ) return true; // no string range or NE pruning, as in PredicateTranslator
                Binary lo = (Binary) min, hi = (Binary) max;
                return lo.compareTo(lit) <= 0 && lit.compareTo(hi) <= 0;
            }
            default:
                return true;
        }
    }

    /**
     * @param neOk whether {@code NE} may prune (only for exact types: never floating point, where NaN
     *             elements are invisible to min/max yet satisfy {@code !=})
     */
    private static <T extends Comparable<T>> boolean range(T min, T max, Predicate.Op op, T lit, boolean neOk) {
        return switch (op) {
            case EQ -> min.compareTo(lit) <= 0 && lit.compareTo(max) <= 0;
            case NE -> !neOk || !(min.compareTo(lit) == 0 && max.compareTo(lit) == 0);
            case LT -> min.compareTo(lit) < 0;
            case LE -> min.compareTo(lit) <= 0;
            case GT -> max.compareTo(lit) > 0;
            case GE -> max.compareTo(lit) >= 0;
        };
    }
}
