package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.parquet.schema.Type;

import java.util.ArrayList;
import java.util.List;

/**
 * Resolves a predicate's dotted column name to one Parquet leaf, and to the route through the decoded
 * Arrow vectors that reaches it. The Arrow shape mirrors {@link NodePlan}'s rules exactly (struct to
 * struct vector, LIST/MAP or bare {@code repeated} to list vector, one data vector per list).
 *
 * <h2>Accepted spellings</h2>
 * A name is tried first as an exact top-level field (so a flat column keeps its meaning, even if its name
 * contains a dot). Otherwise it is split on {@code '.'} and walked from a top-level field. Both forms of a
 * LIST/MAP are accepted, and may be mixed:
 * <ul>
 *   <li>the physical Parquet path: {@code items.list.element.price}, {@code attrs.key_value.value};</li>
 *   <li>the logical path, which skips the synthetic wrapper group: {@code items.element.price},
 *       {@code attrs.value}, and, when the element is a struct, simply {@code items.price}.</li>
 * </ul>
 * Field names that themselves contain a dot cannot be addressed below the top level.
 *
 * <h2>Semantics a caller must apply</h2>
 * {@link Resolved#repeated()} means at least one list lies between the top-level column and the leaf, so a
 * row holds zero or more values for it. Predicates over such a leaf mean "at least one element satisfies"
 * (see {@code Predicate}); that quantifier does not survive negation, which pruning must respect.
 */
public final class ColumnPaths {

    private ColumnPaths() {}

    public enum StepKind {
        /** Child of a struct vector, by name. */
        FIELD,
        /** Elements of a list (or map) vector: every position in the row's element range. */
        LIST
    }

    /** One hop from a vector to the vector holding the next level; {@code name} is null for {@link StepKind#LIST}. */
    public record Step(StepKind kind, String name) {}

    /**
     * @param topLevel     the top-level field the path starts in
     * @param physicalPath the leaf's exact Parquet column path (what statistics and filters are keyed by)
     * @param leaf         the primitive leaf type
     * @param maxRep       maximum repetition level of the leaf (0 when no list lies above it)
     * @param steps        route from the top-level vector to the leaf vector
     */
    public record Resolved(String topLevel, String[] physicalPath, PrimitiveType leaf, int maxRep, List<Step> steps) {
        public String physicalDotted() { return String.join(".", physicalPath); }
        public boolean repeated() { return maxRep > 0; }
        /** True when the leaf is a plain top-level column (no struct or list hop). */
        public boolean flat() { return steps.isEmpty(); }
    }

    /** @return the resolved leaf, or null if {@code column} does not name exactly one primitive leaf. Never throws. */
    public static Resolved resolve(MessageType schema, String column) {
        try {
            return resolveInternal(schema, column);
        } catch (RuntimeException e) {
            return null;
        }
    }

    private static Resolved resolveInternal(MessageType schema, String column) {
        if (schema == null || column == null || column.isEmpty()) return null;
        if (schema.containsField(column)) {
            Type t = schema.getType(column);
            if (!t.isPrimitive()) return null;
            List<Step> steps = new ArrayList<>(1);
            if (t.isRepetition(Type.Repetition.REPEATED)) steps.add(new Step(StepKind.LIST, null));
            return finish(schema, column, new String[] {column}, t.asPrimitiveType(), steps);
        }
        String[] tok = column.split("\\.", -1);
        if (tok.length < 2 || !schema.containsField(tok[0])) return null;
        for (String s : tok) if (s.isEmpty()) return null;

        Walk w = new Walk(tok);
        w.physical.add(tok[0]);
        Type top = schema.getType(tok[0]);
        return w.enter(top, 1) ? w.result(schema, tok[0]) : null;
    }

    /** Mutable walk state; one instance per resolve call. */
    private static final class Walk {
        final String[] tok;
        final List<String> physical = new ArrayList<>();
        final List<Step> steps = new ArrayList<>();
        PrimitiveType leaf;

        Walk(String[] tok) { this.tok = tok; }

        Resolved result(MessageType schema, String top) {
            return finish(schema, top, physical.toArray(new String[0]), leaf, steps);
        }

        /**
         * {@code t} has been matched and we are standing on its vector; {@code i} is the next unconsumed token.
         * Returns true when a primitive leaf consumed the last token.
         */
        boolean enter(Type t, int i) {
            if (t.isRepetition(Type.Repetition.REPEATED)) {
                // Bare repeated field: a list of its own type; the element is t itself.
                steps.add(new Step(StepKind.LIST, null));
                return below(t, i);
            }
            if (t.isPrimitive()) {
                if (i != tok.length) return false;
                leaf = t.asPrimitiveType();
                return true;
            }
            GroupType g = t.asGroupType();
            LogicalTypeAnnotation lt = g.getLogicalTypeAnnotation();
            boolean isMap = lt instanceof LogicalTypeAnnotation.MapLogicalTypeAnnotation
                    || lt instanceof LogicalTypeAnnotation.MapKeyValueTypeAnnotation;
            if (lt instanceof LogicalTypeAnnotation.ListLogicalTypeAnnotation || isMap) {
                if (g.getFieldCount() != 1 || !g.getType(0).isRepetition(Type.Repetition.REPEATED)) return false;
                steps.add(new Step(StepKind.LIST, null));
                return listBody(g, g.getType(0), isMap, i);
            }
            return below(t, i);
        }

        /** Struct children (or, for a bare repeated group, its own children) matched against {@code tok[i]}. */
        boolean below(Type t, int i) {
            if (t.isPrimitive()) {
                if (i != tok.length) return false;
                leaf = t.asPrimitiveType();
                return true;
            }
            if (i >= tok.length) return false;
            GroupType g = t.asGroupType();
            if (!g.containsField(tok[i])) return false;
            Type c = g.getType(tok[i]);
            physical.add(c.getName());
            // A struct child is reached by name; a list child's own vector is the field vector, so FIELD then its steps.
            steps.add(new Step(StepKind.FIELD, c.getName()));
            return enter(c, i + 1);
        }

        boolean listBody(GroupType listGroup, Type r, boolean isMap, int i) {
            int at = i;
            String wrapper = r.getName();
            boolean wrapperTyped = at < tok.length && tok[at].equals(wrapper);
            if (wrapperTyped) at++;
            physical.add(wrapper);
            if (r.isPrimitive()) {
                // Legacy 2-level list of primitives: the repeated primitive is the leaf.
                if (!wrapperTyped || at != tok.length) return false;
                leaf = r.asPrimitiveType();
                return true;
            }
            GroupType rg = r.asGroupType();
            if (isMap) {
                // Elements are key_value structs; children are addressed by field name.
                return childOf(rg, at);
            }
            boolean tuple = rg.getName().equals("array") || rg.getName().equals(listGroup.getName() + "_tuple");
            if (rg.getFieldCount() == 1 && !tuple) {
                // 3-level list: the data vector IS the single child ("element"); no FIELD hop into it.
                Type c = rg.getType(0);
                if (at < tok.length && tok[at].equals(c.getName())) {
                    physical.add(c.getName());
                    return enter(c, at + 1);
                }
                // Implicit element: "items.price" reaches into a struct element.
                if (at < tok.length && !c.isPrimitive() && c.asGroupType().containsField(tok[at])) {
                    physical.add(c.getName());
                    return childOf(c.asGroupType(), at);
                }
                return false;
            }
            return childOf(rg, at);
        }

        /** Data vector is a struct of {@code g}'s fields; hop into one by name. */
        boolean childOf(GroupType g, int i) {
            if (i >= tok.length || !g.containsField(tok[i])) return false;
            Type c = g.getType(tok[i]);
            physical.add(c.getName());
            steps.add(new Step(StepKind.FIELD, c.getName()));
            return enter(c, i + 1);
        }
    }

    private static Resolved finish(MessageType schema, String top, String[] path, PrimitiveType leaf, List<Step> steps) {
        if (leaf == null || !schema.containsPath(path)) return null;
        int maxRep = schema.getColumnDescription(path).getMaxRepetitionLevel();
        long lists = steps.stream().filter(s -> s.kind() == StepKind.LIST).count();
        if (lists != maxRep) return null; // route and schema disagree: refuse rather than navigate wrongly
        return new Resolved(top, path, leaf, maxRep, List.copyOf(steps));
    }
}
