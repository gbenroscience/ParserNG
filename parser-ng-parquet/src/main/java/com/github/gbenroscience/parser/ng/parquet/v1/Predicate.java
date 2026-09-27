package com.github.gbenroscience.parser.ng.parquet.v1;

import java.util.Arrays;
import java.util.List;

/**
 * A storage-level <b>pruning</b> predicate. It is SQL-agnostic: a planner builds it from whatever
 * conjuncts it can prove safe, and everything else stays an Arrow-side expression.
 *
 * <h2>Contract: pruning only, never exact filtering</h2>
 * A scan given a predicate returns a <i>superset</i> of the matching rows: row groups that cannot
 * possibly match are skipped, the rest are emitted whole. The caller MUST still evaluate the same
 * condition on the emitted batches. This is what makes pushdown semantics-preserving: dropping a
 * row group is only ever done when metadata proves no row in it can match.
 *
 * <h2>What is actually pushed down</h2>
 * A leaf is translated only when the literal's Java type is an exact/widening match for the
 * column's physical + logical type; otherwise that leaf is silently NOT used for pruning (never
 * coerced). AND drops untranslatable sides (sound: still a superset); OR is dropped entirely if
 * either side is untranslatable. Range comparisons on strings are never pushed (UTF-16 vs
 * unsigned-byte ordering differ for supplementary characters); NaN literals are never pushed.
 */
public sealed interface Predicate {

    enum Op { EQ, NE, LT, LE, GT, GE }

    record Cmp(String column, Op op, Object value) implements Predicate {
        public Cmp {
            require(column, "column");
            if (op == null) throw new IllegalArgumentException("op must not be null");
            if (value == null) throw new IllegalArgumentException("value must not be null; use isNull()/isNotNull()");
        }
    }

    record In(String column, List<?> values) implements Predicate {
        public In {
            require(column, "column");
            if (values == null || values.isEmpty()) throw new IllegalArgumentException("IN list must not be empty");
            values = List.copyOf(values); // also rejects null elements
        }
    }

    record IsNull(String column, boolean negated) implements Predicate {
        public IsNull {
            require(column, "column");
        }
    }

    record And(Predicate left, Predicate right) implements Predicate {
        public And {
            if (left == null || right == null) throw new IllegalArgumentException("operands must not be null");
        }
    }

    record Or(Predicate left, Predicate right) implements Predicate {
        public Or {
            if (left == null || right == null) throw new IllegalArgumentException("operands must not be null");
        }
    }

    static Predicate eq(String c, Object v) { return new Cmp(c, Op.EQ, v); }
    static Predicate ne(String c, Object v) { return new Cmp(c, Op.NE, v); }
    static Predicate lt(String c, Object v) { return new Cmp(c, Op.LT, v); }
    static Predicate le(String c, Object v) { return new Cmp(c, Op.LE, v); }
    static Predicate gt(String c, Object v) { return new Cmp(c, Op.GT, v); }
    static Predicate ge(String c, Object v) { return new Cmp(c, Op.GE, v); }
    static Predicate in(String c, Object... v) { return new In(c, Arrays.asList(v)); }
    static Predicate isNull(String c) { return new IsNull(c, false); }
    static Predicate isNotNull(String c) { return new IsNull(c, true); }
    static Predicate and(Predicate l, Predicate r) { return new And(l, r); }
    static Predicate or(Predicate l, Predicate r) { return new Or(l, r); }

    private static void require(String s, String what) {
        if (s == null || s.isEmpty()) throw new IllegalArgumentException(what + " must not be null/empty");
    }
}
