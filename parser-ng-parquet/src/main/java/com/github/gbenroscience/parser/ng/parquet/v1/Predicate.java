package com.github.gbenroscience.parser.ng.parquet.v1;

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * A storage-level predicate: SQL-agnostic (a planner builds it from whatever conjuncts it can prove
 * safe) and dependency-free (nothing outside {@code java.util.*}/{@code java.util.regex.*} — no
 * Arrow, no Parquet types, so it costs nothing to construct, pass around, or unit-test in isolation
 * from either).
 *
 * <h2>Two ways this gets used, and which node types each one actually uses</h2>
 * <ul>
 *   <li>{@code ParquetScan.pushdown(...)} (default): {@code internal.PredicateTranslator} turns
 *       whatever it safely can into a parquet-java {@code FilterPredicate} for row-group/page
 *       PRUNING. A scan given only this returns a SUPERSET — the caller re-checks. Only {@link Cmp},
 *       {@link In}, {@link IsNull}, {@link And}, {@link Or}, and {@link Not} (rewritten via De
 *       Morgan's laws, never pushed as a literal Parquet {@code not()} — see
 *       {@code PredicateTranslator}'s Javadoc on why) can ever be pushed; {@link ColCmp},
 *       {@link Like}, {@link Regex}, and {@link Custom} are never pushed and simply prune nothing
 *       for that leaf, which is always sound (see each type's own Javadoc for why).
 *   <li>{@code ParquetScan.pushdown(...).exactFilter()}: {@code internal.RowFilterEvaluator} turns
 *       the ENTIRE predicate — every node type here — into an exact, row-by-row test. This is the
 *       only mode where {@link ColCmp}/{@link Like}/{@link Regex}/{@link Custom} actually filter
 *       anything; without {@code exactFilter()} they exist only as documentation of intent that
 *       currently contributes no pruning.
 * </ul>
 *
 * <h2>What is actually pushed down for pruning (unchanged from before this revision)</h2>
 * A {@link Cmp}/{@link In}/{@link IsNull} leaf is translated only when the literal's Java type is an
 * exact/widening match for the column's physical + logical type; otherwise that leaf is silently NOT
 * used for pruning (never coerced). {@link And} drops untranslatable sides (sound: still a superset);
 * {@link Or} is dropped entirely if either side is untranslatable. Range comparisons on strings are
 * never pushed (UTF-16 vs unsigned-byte ordering differ for supplementary characters); NaN literals
 * are never pushed.
 *
 * <h2>What {@code exactFilter()} can express that pushdown never could, even in principle</h2>
 * {@link ColCmp} (Parquet row-group/page statistics summarize one column at a time; there is no
 * cross-column joint range to prune on, ever — this is a hard limit of what statistics-based pruning
 * can mean, not a gap in this translator) and, for the same string-ordering reason plain ranges
 * aren't pushed, string range comparisons ARE sound for {@link Cmp}/{@link Like}/{@link Regex} once
 * they're evaluated exactly on already-decoded {@code String}s instead of Parquet's on-disk bytes.
 */
public sealed interface Predicate {

    enum Op { EQ, NE, LT, LE, GT, GE }

    /** Column vs. literal. Unchanged from before this revision. */
    record Cmp(String column, Op op, Object value) implements Predicate {
        public Cmp {
            require(column, "column");
            if (op == null) throw new IllegalArgumentException("op must not be null");
            if (value == null) throw new IllegalArgumentException("value must not be null; use isNull()/isNotNull()");
        }
    }

    /** Column IN (literal, literal, ...). Unchanged from before this revision. */
    record In(String column, List<?> values) implements Predicate {
        public In {
            require(column, "column");
            if (values == null || values.isEmpty()) throw new IllegalArgumentException("IN list must not be empty");
            values = List.copyOf(values); // also rejects null elements
        }
    }

    /** IS [NOT] NULL. Unchanged from before this revision. */
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

    /**
     * General negation of an arbitrary subtree — the gap the rest of this model used to have no
     * answer for beyond {@code IsNull}'s built-in {@code negated} flag.
     *
     * <p>Pushdown: {@code PredicateTranslator} never hands a literal {@code Not} to parquet-java's
     * {@code FilterApi.not(...)}. Negating a range comparison is exactly the case parquet-java's own
     * {@code LogicalInverseRewriter} exists to handle carefully (statistics-based pruning under a raw
     * NOT is a well-known footgun in parquet-java itself) — rather than depend on that being applied
     * correctly downstream, {@code PredicateTranslator} rewrites {@code Not} to negation-normal form
     * itself, at this level, before translation ever begins: De Morgan's laws push the negation down
     * to each leaf, {@code NOT(Cmp(c,EQ,v))} becomes {@code Cmp(c,NE,v)} and so on for every
     * {@link Op}, {@code NOT(In(c,vs))} becomes an {@link And} chain of {@link Op#NE}, and
     * {@code NOT(IsNull)} simply flips its own flag. A {@code Not} wrapping something that already
     * doesn't push ({@link ColCmp}/{@link Like}/{@link Regex}/{@link Custom}) still doesn't push —
     * negating "nothing to prune on" is still "nothing to prune on," which stays sound.
     *
     * <p>Exact filtering: trivial — {@code !inner}. No rewriting needed; boolean negation composes
     * correctly at this layer without any of the statistics-pruning subtlety pushdown has to worry about.
     */
    record Not(Predicate inner) implements Predicate {
        public Not {
            if (inner == null) throw new IllegalArgumentException("inner must not be null");
        }
    }

    /**
     * Column vs. column (e.g. {@code discount_price < list_price}) — never pushed for pruning (row-group
     * and page statistics describe one column's range at a time; there is no notion of "these two
     * columns' ranges relate this way" for Parquet's stats-based pruning to exploit, so this always
     * yields no pruning for this leaf, soundly), but a first-class, exactly-evaluated leaf under
     * {@code exactFilter()}: both columns must resolve to the same Arrow type (no cross-type
     * numeric widening — consistent with this model's "never coerce" rule everywhere else), and both
     * must be flat, top-level, and part of the projection, exactly like a {@link Cmp}'s single column.
     */
    record ColCmp(String left, Op op, String right) implements Predicate {
        public ColCmp {
            require(left, "left");
            require(right, "right");
            if (op == null) throw new IllegalArgumentException("op must not be null");
        }
    }

    /**
     * SQL {@code LIKE}: {@code %} matches any sequence (including empty), {@code _} matches exactly
     * one character, {@code escape} (if non-{@code null}) escapes a literal {@code %}, {@code _}, or
     * itself when placed immediately before one. Compiled to a {@link Pattern} eagerly, in this
     * record's compact constructor, specifically so a malformed pattern (an escape character with
     * nothing valid after it) fails at the moment the predicate is built, not later when a scan opens.
     *
     * <p>Never pushed for pruning, even the common "starts with a literal prefix" case: that would
     * need exactly the byte-level-vs-{@code String}-ordering safety this model already declines for
     * plain string ranges (see class Javadoc) — a % or _ anywhere in the pattern only makes that
     * reasoning harder, not easier, so it is not attempted at all rather than attempted unsoundly for
     * some patterns and not others.
     */
    record Like(String column, String pattern, boolean caseInsensitive, Character escape, Pattern compiled) implements Predicate {
        public Like {
            require(column, "column");
            if (pattern == null) throw new IllegalArgumentException("pattern must not be null");
            if (compiled == null) compiled = compileLike(pattern, caseInsensitive, escape);
        }

        // compiled is derived from the other four fields and java.util.regex.Pattern does not itself
        // override equals()/hashCode() (reference identity) -- without this override, two Like
        // predicates built separately from identical (column, pattern, caseInsensitive, escape)
        // would compare unequal purely because each compiled its own distinct Pattern object.
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Like other)) return false;
            return column.equals(other.column) && pattern.equals(other.pattern)
                    && caseInsensitive == other.caseInsensitive
                    && java.util.Objects.equals(escape, other.escape);
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(column, pattern, caseInsensitive, escape);
        }
    }

    /**
     * Raw {@link Pattern} match against a column's decoded {@code String} value (Java regex
     * semantics — not POSIX, not Parquet anything). Never pushed for pruning (parquet-java's
     * {@code FilterApi} has no regex concept at all); exact-only, like {@link Like}.
     */
    record Regex(String column, Pattern pattern) implements Predicate {
        public Regex {
            require(column, "column");
            if (pattern == null) throw new IllegalArgumentException("pattern must not be null");
        }

        // Same reasoning as Like's override: java.util.regex.Pattern is reference-identity-equals,
        // so two Regex predicates compiled separately from the same source+flags would otherwise
        // compare unequal.
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Regex other)) return false;
            return column.equals(other.column) && pattern.pattern().equals(other.pattern.pattern())
                    && pattern.flags() == other.pattern.flags();
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(column, pattern.pattern(), pattern.flags());
        }
    }

    /**
     * Escape hatch for anything this model has no primitive for — computed/functional expressions
     * ({@code UPPER(x) = 'A'}, {@code x + 1 > 10}), or any other row-level test a caller can compute
     * but this module has no business knowing the meaning of. Carries no executable code itself (see
     * class Javadoc — {@code Predicate} stays dependency-free); the actual test is supplied
     * separately, by id, via {@code ParquetScan.withCustomPredicate(id, impl)} against the
     * {@code CustomPredicate} interface. Never pushed for pruning — parquet-java cannot know what an
     * opaque id means, so this leaf always contributes no pruning, soundly. Meaningless without
     * {@code exactFilter()}: under plain {@code pushdown()}, a {@code Custom} node is simply inert.
     *
     * @param id a caller-chosen key matched against {@code ParquetScan.withCustomPredicate(id, ...)};
     *           {@code exactFilter()} throws, naming this id, if nothing was registered for it
     * @param touchedColumns columns the implementation reads, used only to validate they are part of
     *           the projection before any row is scanned — unlike every other leaf type, these need
     *           NOT be flat/top-level, since a custom implementation may know how to navigate a
     *           nested column itself
     */
    record Custom(String id, Set<String> touchedColumns) implements Predicate {
        public Custom {
            require(id, "id");
            touchedColumns = touchedColumns == null ? Set.of() : Set.copyOf(touchedColumns);
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
    static Predicate not(Predicate p) { return new Not(p); }

    static Predicate colEq(String l, String r) { return new ColCmp(l, Op.EQ, r); }
    static Predicate colNe(String l, String r) { return new ColCmp(l, Op.NE, r); }
    static Predicate colLt(String l, String r) { return new ColCmp(l, Op.LT, r); }
    static Predicate colLe(String l, String r) { return new ColCmp(l, Op.LE, r); }
    static Predicate colGt(String l, String r) { return new ColCmp(l, Op.GT, r); }
    static Predicate colGe(String l, String r) { return new ColCmp(l, Op.GE, r); }

    static Predicate like(String column, String pattern) { return new Like(column, pattern, false, '\\', null); }
    static Predicate ilike(String column, String pattern) { return new Like(column, pattern, true, '\\', null); }
    static Predicate like(String column, String pattern, boolean caseInsensitive, Character escape) {
        return new Like(column, pattern, caseInsensitive, escape, null);
    }

    static Predicate regex(String column, String javaRegex) {
        try {
            return new Regex(column, Pattern.compile(javaRegex));
        } catch (PatternSyntaxException e) {
            throw new IllegalArgumentException("Invalid regex for column '" + column + "': " + e.getMessage(), e);
        }
    }
    static Predicate regex(String column, Pattern compiled) { return new Regex(column, compiled); }

    static Predicate custom(String id, String... touchedColumns) {
        return new Custom(id, touchedColumns == null ? Set.of() : Set.of(touchedColumns));
    }

    private static void require(String s, String what) {
        if (s == null || s.isEmpty()) throw new IllegalArgumentException(what + " must not be null/empty");
    }

    /** SQL LIKE -> Java regex: {@code %}->{@code .*}, {@code _}->{@code .}, escape char escapes the next literal char. */
    private static Pattern compileLike(String likePattern, boolean caseInsensitive, Character escape) {
        StringBuilder re = new StringBuilder(likePattern.length() * 2);
        int n = likePattern.length();
        for (int i = 0; i < n; i++) {
            char c = likePattern.charAt(i);
            if (escape != null && c == escape) {
                if (i + 1 >= n) {
                    throw new IllegalArgumentException("LIKE pattern '" + likePattern + "' ends with a dangling escape character");
                }
                char next = likePattern.charAt(++i);
                if (next != '%' && next != '_' && next != escape) {
                    throw new IllegalArgumentException("LIKE pattern '" + likePattern
                            + "': escape character must be followed by '%', '_', or itself, was '" + next + "'");
                }
                re.append(Pattern.quote(String.valueOf(next)));
            } else if (c == '%') {
                re.append(".*");
            } else if (c == '_') {
                re.append('.');
            } else {
                re.append(Pattern.quote(String.valueOf(c)));
            }
        }
        int flags = caseInsensitive ? (Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE) : 0;
        return Pattern.compile(re.toString(), flags);
    }
}
