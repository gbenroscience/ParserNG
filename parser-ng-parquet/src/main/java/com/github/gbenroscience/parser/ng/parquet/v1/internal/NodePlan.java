package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScanException;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetSource;
import java.util.ArrayList;
import java.util.List;

/**
 * Immutable description of how each projected top-level Parquet field maps onto an Arrow vector tree,
 * plus the level numbers ({@code rep}, {@code reach}, {@code def}) the {@link LevelWalker} rule needs.
 * Built once per scan and shared by every decoder thread; it holds no per-batch state.
 *
 * <p>Supported shapes: primitives, structs, LIST (3-level and the legacy 2-level / tuple forms),
 * MAP, bare {@code repeated} fields (as lists), and any nesting of these. Unsupported leaf types fail
 * with the column named.
 */
public final class NodePlan {

    /** Level bookkeeping common to every node. */
    public abstract static class Node {
        public final String name;
        public final int id;
        /** Repeated ancestors enclosing this node's slots (r). */
        public final int rep;
        /** Definition level an entry needs before this node has a slot at all. */
        public final int reach;
        /** Definition level at which this node itself is non-null (d). */
        public final int def;
        public final boolean nullable;
        Field field;
        int firstLeaf = -1;

        Node(String name, int id, int rep, int reach, int def, boolean nullable) {
            this.name = name; this.id = id; this.rep = rep; this.reach = reach; this.def = def; this.nullable = nullable;
        }

        public Field field() { return field; }
        public int firstLeaf() { return firstLeaf; }
    }

    public static final class Leaf extends Node {
        public final ColumnPlan plan;
        public final int leafIndex;
        boolean retain;

        Leaf(String name, int id, int rep, int reach, int def, boolean nullable, ColumnPlan plan, int leafIndex) {
            super(name, id, rep, reach, def, nullable);
            this.plan = plan; this.leafIndex = leafIndex;
            this.field = plan.field();
            this.firstLeaf = leafIndex;
        }

        /** True if a container above needs this leaf's level stream kept. */
        public boolean retainLevels() { return retain; }
    }

    public static final class Struct extends Node {
        public final Node[] children;

        Struct(String name, int id, int rep, int reach, int def, boolean nullable, Node[] children) {
            super(name, id, rep, reach, def, nullable);
            this.children = children;
            List<Field> fs = new ArrayList<>(children.length);
            for (Node c : children) fs.add(c.field);
            this.field = new Field(name, new FieldType(nullable, ArrowType.Struct.INSTANCE, null), fs);
            this.firstLeaf = children[0].firstLeaf;
        }
    }

    /** A LIST or MAP (a map is a list of key/value structs). */
    public static final class Listy extends Node {
        public final Node element;
        public final boolean isMap;
        /** Level numbers of the inner repeated group R. */
        public final int repR, defR;

        Listy(String name, int id, int rep, int reach, int def, boolean nullable,
              int repR, int defR, Node element, boolean isMap) {
            super(name, id, rep, reach, def, nullable);
            this.repR = repR; this.defR = defR; this.element = element; this.isMap = isMap;
            this.field = new Field(name,
                    new FieldType(nullable, isMap ? new ArrowType.Map(false) : ArrowType.List.INSTANCE, null),
                    List.of(element.field));
            this.firstLeaf = element.firstLeaf;
        }
    }

    private final Node[] tops;
    private final Leaf[] leaves;
    private final int nodeCount;

    private NodePlan(Node[] tops, Leaf[] leaves, int nodeCount) {
        this.tops = tops; this.leaves = leaves; this.nodeCount = nodeCount;
    }

    public Node[] tops() { return tops; }
    public int nodeCount() { return nodeCount; }
    public int leafCount() { return leaves.length; }
    public Leaf leaf(int i) { return leaves[i]; }

    public List<Field> fields() {
        List<Field> fs = new ArrayList<>(tops.length);
        for (Node n : tops) fs.add(n.field);
        return fs;
    }

    /** True when every top-level node is a plain leaf: the fast flat decode path applies to all of them. */
    public boolean isFlat() {
        for (Node n : tops) if (!(n instanceof Leaf)) return false;
        return true;
    }

    // ------------------------------------------------------------------ builder

    public static NodePlan build(MessageType projected, ColumnDescriptor[] descriptors, ParquetSource file) {
        Builder b = new Builder(file);
        Node[] tops = new Node[projected.getFieldCount()];
        for (int i = 0; i < tops.length; i++) tops[i] = b.build(projected.getType(i), new State(0, 0, 0));
        if (b.leaves.size() != descriptors.length) {
            throw new ParquetScanException("Internal error: planned " + b.leaves.size() + " leaves but the schema has "
                    + descriptors.length, file);
        }
        for (Leaf l : b.leaves) {
            ColumnDescriptor d = descriptors[l.leafIndex];
            if (l.def != d.getMaxDefinitionLevel() || l.rep != d.getMaxRepetitionLevel()) {
                throw new ParquetScanException("Internal error: level mismatch (planned rep/def " + l.rep + "/" + l.def
                        + " vs file " + d.getMaxRepetitionLevel() + "/" + d.getMaxDefinitionLevel() + ")",
                        file, -1, String.join(".", d.getPath()), null);
            }
        }
        for (Node n : b.containers) b.leaves.get(n.firstLeaf).retain = true;
        return new NodePlan(tops, b.leaves.toArray(new Leaf[0]), b.nextId);
    }

    /** Context above a node: repeated count, definition count, and reach level. */
    private record State(int rep, int def, int reach) { }

    private static final class Builder {
        final ParquetSource file;
        final List<Leaf> leaves = new ArrayList<>();
        final List<Node> containers = new ArrayList<>();
        int nextId = 0;

        Builder(ParquetSource file) { this.file = file; }

        Node build(Type t, State s) {
            if (t.isRepetition(Type.Repetition.REPEATED)) return bareRepeated(t, s);
            int d = s.def() + (t.isRepetition(Type.Repetition.OPTIONAL) ? 1 : 0);
            boolean nullable = t.isRepetition(Type.Repetition.OPTIONAL);
            if (t.isPrimitive()) return leaf(t.asPrimitiveType(), t.getName(), nullable, s.rep(), s.reach(), d);
            GroupType g = t.asGroupType();
            LogicalTypeAnnotation lt = g.getLogicalTypeAnnotation();
            if (lt instanceof LogicalTypeAnnotation.ListLogicalTypeAnnotation) return list(g, s, d, nullable, false);
            if (lt instanceof LogicalTypeAnnotation.MapLogicalTypeAnnotation
                    || lt instanceof LogicalTypeAnnotation.MapKeyValueTypeAnnotation) return list(g, s, d, nullable, true);
            return struct(g.getName(), g, s.rep(), s.reach(), d, nullable, new State(s.rep(), d, s.reach()));
        }

        Leaf leaf(PrimitiveType pt, String name, boolean nullable, int rep, int reach, int def) {
            ColumnPlan plan = ColumnPlan.forPrimitive(pt, name, nullable, def, file);
            Leaf l = new Leaf(name, nextId++, rep, reach, def, nullable, plan, leaves.size());
            leaves.add(l);
            return l;
        }

        /** {@code children} are built in the context {@code childState}; the node's own numbers are given. */
        Struct struct(String name, GroupType g, int rep, int reach, int def, boolean nullable, State childState) {
            int id = nextId++;
            Node[] kids = new Node[g.getFieldCount()];
            for (int i = 0; i < kids.length; i++) kids[i] = build(g.getType(i), childState);
            Struct st = new Struct(name, id, rep, reach, def, nullable, kids);
            containers.add(st);
            return st;
        }

        Node list(GroupType g, State s, int d, boolean nullable, boolean isMap) {
            if (g.getFieldCount() != 1 || !g.getType(0).isRepetition(Type.Repetition.REPEATED)) {
                throw new ParquetScanException("Malformed " + (isMap ? "MAP" : "LIST")
                        + ": expected exactly one repeated child", file, -1, g.getName(), null);
            }
            Type r = g.getType(0);
            int repR = s.rep() + 1, defR = d + 1;
            State es = new State(repR, defR, defR);
            int id = nextId++;
            Node elem;
            if (isMap) {
                if (r.isPrimitive() || r.asGroupType().getFieldCount() != 2) {
                    throw new ParquetScanException("Malformed MAP: key_value must be a group of key and value",
                            file, -1, g.getName(), null);
                }
                elem = struct("entries", r.asGroupType(), repR, defR, defR, false, es);
            } else if (r.isPrimitive()) {
                elem = leaf(r.asPrimitiveType(), r.getName(), false, repR, defR, defR);
            } else {
                GroupType rg = r.asGroupType();
                boolean tuple = rg.getName().equals("array") || rg.getName().equals(g.getName() + "_tuple");
                elem = (rg.getFieldCount() == 1 && !tuple)
                        ? build(rg.getType(0), es)
                        : struct(rg.getName(), rg, repR, defR, defR, false, es);
            }
            Listy l = new Listy(g.getName(), id, s.rep(), s.reach(), d, nullable, repR, defR, elem, isMap);
            containers.add(l);
            return l;
        }

        /** A {@code repeated} field outside LIST/MAP (legacy): a non-null list of its own type. */
        Node bareRepeated(Type t, State s) {
            int repR = s.rep() + 1, defR = s.def() + 1;
            State es = new State(repR, defR, defR);
            int id = nextId++;
            Node elem = t.isPrimitive()
                    ? leaf(t.asPrimitiveType(), t.getName(), false, repR, defR, defR)
                    : struct(t.getName(), t.asGroupType(), repR, defR, defR, false, es);
            Listy l = new Listy(t.getName(), id, s.rep(), s.reach(), s.def(), false, repR, defR, elem, false);
            containers.add(l);
            return l;
        }
    }
}
