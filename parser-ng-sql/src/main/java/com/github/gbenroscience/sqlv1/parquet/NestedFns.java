package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.sqlv1.QuotedIdentifiers;
import com.github.gbenroscience.sqlv1.ast.SelectStatement;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Nested field access: {@code s.field}, {@code s.items[2].price}, {@code items[0]}, {@code m['key']}, and
 * {@code CARDINALITY(list_or_map)}.
 *
 * <p>The parser turns a path into a {@link QuotedIdentifiers} reference to the whole path text. This class
 * resolves each such reference against the file's Arrow schema (struct field, list element by position,
 * map value by key), then replaces it in the statement by a <em>derived column</em> of the leaf's own type,
 * filled per batch by walking the struct, list and map vectors. From then on the leaf is an ordinary typed
 * column: numeric leaves enter expressions, aggregates, {@code GROUP BY} and {@code ORDER BY}; string,
 * boolean, date and timestamp leaves meet the native kernels, string functions and the select list. Results
 * are copied with the vector's own {@code copyFromSafe}, so a value is never re-encoded.
 *
 * <p><b>Rules.</b> List indexes are zero-based, as written. An index past the end, a missing map key, and a
 * null anywhere along the path give null, never an error. A path through a list needs an index at that
 * step. A map key is a string literal (or an integer for an integer-keyed map); on a duplicate key the
 * first entry wins. A path must end at a primitive leaf; {@code CARDINALITY} instead takes a list or map
 * and returns its element count ({@code Float8}, null for a null list). Nested predicates are never pushed
 * to row-group pruning, and {@code explain()} says so; select the whole top-level column to get the
 * struct, list or map itself. A field whose own name contains {@code .}, {@code [} or {@code ]} cannot be
 * addressed.
 */
final class NestedFns {

    private NestedFns() { }

    record Result(SelectStatement stmt, List<NativeLeaves.Mask> masks, List<DerivedColumns.Column> columns, int used) { }

    private static final Pattern REF = Pattern.compile("\\bpngq(?:[a-p]{2})+z\\b");

    /** True if {@code real} (a decoded reference) names a path into a nested file column. */
    static boolean isPath(String real, TableSchema file) {
        if (real == null) return false;
        int cut = firstCut(real);
        return cut > 0 && cut < real.length() && file.hasFileColumn(real.substring(0, cut))
                && file.kind(real.substring(0, cut)) == ColumnKind.OTHER;
    }

    /** The top-level column a reference names: the path's first segment, or the name itself. */
    static String topLevel(String real) {
        int cut = firstCut(real);
        return cut > 0 ? real.substring(0, cut) : real;
    }

    private static int firstCut(String s) {
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == '.' || s.charAt(i) == '[') return i;
        return -1;
    }

    /** The Arrow field the path {@code real} ends at, or {@code null} if it does not resolve to a value (never throws). */
    static Field leafField(String real, TableSchema file, Schema arrow) {
        try {
            return new Ctx(file, arrow, 0).resolve(real, false).end;
        } catch (ParquetSqlException notAValue) {
            return null;
        }
    }

    static Result rewrite(SelectStatement st, TableSchema file, Schema arrow, int firstIndex) {
        Ctx ctx = new Ctx(file, arrow, firstIndex);
        SelectStatement out = StatementRewriter.map(st, ctx::text);
        return new Result(out, List.copyOf(ctx.masks), List.copyOf(ctx.columns), ctx.next - firstIndex);
    }

    // ------------------------------------------------------------------ steps

    private enum StepKind { FIELD, INDEX, KEY }

    private record Step(StepKind kind, String name, long index, int ordinal) { }

    /** A resolved path: the top-level column, the steps, and the Arrow field the path ends at. */
    private static final class Path {
        final String text;
        final String top;
        final List<Step> steps = new ArrayList<>();
        Field end;

        // per-batch binding
        private FieldVector[] chain;
        FieldVector vec; // the vector the last locate() ended at
        int idx;

        Path(String text, String top) {
            this.text = text;
            this.top = top;
        }

        void bind(VectorSchemaRoot src) {
            FieldVector v = src.getVector(top);
            if (v == null) throw new ParquetSqlException("Column '" + top + "' is missing from the scan (needed by " + text + ")");
            chain = new FieldVector[steps.size() + 1];
            chain[0] = v;
            for (int k = 0; k < steps.size(); k++) {
                FieldVector cur = chain[k];
                Step s = steps.get(k);
                chain[k + 1] = switch (s.kind) {
                    case FIELD -> (FieldVector) ((org.apache.arrow.vector.complex.StructVector) cur).getChildByOrdinal(s.ordinal);
                    case INDEX, KEY -> {
                        FieldVector data = ((ListVector) cur).getDataVector();
                        if (s.kind == StepKind.KEY || cur instanceof org.apache.arrow.vector.complex.MapVector) {
                            yield (FieldVector) ((org.apache.arrow.vector.complex.StructVector) data).getChildByOrdinal(1); // value
                        }
                        yield data;
                    }
                };
            }
            vec = chain[steps.size()];
        }

        /** Walks the path for {@code row}; false if a null or a missing element is met. Sets {@link #idx}. */
        boolean locate(int row) {
            return locateContainer(row);
        }

        /** Like {@link #locate} for a path that ends at a list or map: sets {@link #idx} to the container's row. */
        boolean locateContainer(int row) {
            int at = row;
            for (int k = 0; k < steps.size(); k++) {
                FieldVector cur = chain[k];
                if (cur.isNull(at)) return false;
                at = step(k, cur, at);
                if (at < 0) return false;
            }
            if (chain[steps.size()].isNull(at)) return false;
            idx = at;
            return true;
        }

        private int step(int k, FieldVector cur, int at) {
            Step s = steps.get(k);
            switch (s.kind) {
                case FIELD:
                    return at;
                case INDEX: {
                    ListVector lv = (ListVector) cur;
                    if (lv instanceof org.apache.arrow.vector.complex.MapVector) return findKey(lv, at, null, s.index);
                    long pos = (long) lv.getElementStartIndex(at) + s.index;
                    return pos >= lv.getElementEndIndex(at) ? -1 : (int) pos;
                }
                default:
                    return findKey((ListVector) cur, at, s.name, 0);
            }
        }

        /** Index of the entry of the map at {@code row} whose key equals {@code key} (string) or {@code num}, else -1. */
        private int findKey(ListVector mv, int row, String key, long num) {
            org.apache.arrow.vector.complex.StructVector entries = (org.apache.arrow.vector.complex.StructVector) mv.getDataVector();
            FieldVector keys = (FieldVector) entries.getChildByOrdinal(0);
            int from = mv.getElementStartIndex(row), to = mv.getElementEndIndex(row);
            if (key != null) {
                if (!(keys instanceof VarCharVector kv)) return -1;
                byte[] want = key.getBytes(StandardCharsets.UTF_8);
                for (int i = from; i < to; i++) {
                    if (kv.isNull(i)) continue;
                    int s = kv.getStartOffset(i), len = kv.getEndOffset(i) - s;
                    if (len != want.length) continue;
                    boolean eq = true;
                    for (int b = 0; b < len; b++) {
                        if (kv.getDataBuffer().getByte(s + b) != want[b]) { eq = false; break; }
                    }
                    if (eq) return i;
                }
                return -1;
            }
            for (int i = from; i < to; i++) {
                if (keys.isNull(i)) continue;
                Object o = keys.getObject(i);
                if (o instanceof Number n && n.longValue() == num) return i;
            }
            return -1;
        }
    }

    // ------------------------------------------------------------------ derived columns

    /** A leaf of a path, as a column of the leaf's own type. */
    private static final class PathColumn extends DerivedColumns.Column {
        private final Path path;
        private final Field template;

        PathColumn(String name, Path path, Field leaf) {
            super(name, path.text, TableSchema.kindOf(leaf.getType()), leaf.getType());
            this.path = path;
            this.template = leaf;
        }

        @Override FieldVector newVector(BufferAllocator alloc) {
            return new Field(name, FieldType.nullable(template.getType()), null).createVector(alloc);
        }

        @Override void fill(VectorSchemaRoot src, int n, FieldVector out) {
            path.bind(src);
            FieldVector leaf = path.vec;
            for (int i = 0; i < n; i++) {
                if (path.locate(i)) out.copyFromSafe(path.idx, i, leaf);
                else nullAt(out, i);
            }
        }
    }

    /** Sets row {@code i} of {@code v} to null for any primitive vector type. */
    static void nullAt(FieldVector v, int i) {
        if (v instanceof org.apache.arrow.vector.BaseFixedWidthVector f) {
            f.setNull(i);
        } else if (v instanceof org.apache.arrow.vector.BaseVariableWidthVector b) {
            b.setNull(i);
        } else {
            throw new ParquetSqlException("A nested leaf of vector type " + v.getClass().getSimpleName() + " is not supported");
        }
    }

    /** {@code CARDINALITY(path)}: the element count of a list or map, as a number. */
    private static final class CardMask extends NativeLeaves.Mask {
        private final Path path;

        CardMask(String name, String text, Path path) {
            super(name, text + " (derived Float8 column)");
            this.path = path;
        }

        @Override void eval(VectorSchemaRoot src, int n, Float8Vector out) {
            path.bind(src);
            ListVector lv = (ListVector) path.vec;
            for (int i = 0; i < n; i++) {
                if (!path.locateContainer(i)) out.setNull(i);
                else out.set(i, lv.getElementEndIndex(path.idx) - lv.getElementStartIndex(path.idx));
            }
        }
    }

    // ------------------------------------------------------------------ rewriting context

    private static final class Ctx {
        final TableSchema file;
        final Schema arrow;
        int next;
        final List<NativeLeaves.Mask> masks = new ArrayList<>();
        final List<DerivedColumns.Column> columns = new ArrayList<>();
        final Map<String, String> byKey = new LinkedHashMap<>();

        Ctx(TableSchema file, Schema arrow, int firstIndex) {
            this.file = file;
            this.arrow = arrow;
            this.next = firstIndex;
        }

        String text(String text) {
            String t = cardinalities(text);
            if (!QuotedIdentifiers.containsReference(t)) return t;
            Matcher m = REF.matcher(t);
            StringBuilder sb = new StringBuilder();
            while (m.find()) {
                String ref = m.group();
                String real = QuotedIdentifiers.decode(ref);
                String repl = ref;
                if (real != null && !file.hasFileColumn(ref) && isPath(real, file)) repl = leaf(real);
                m.appendReplacement(sb, Matcher.quoteReplacement(repl));
            }
            return m.appendTail(sb).toString();
        }

        private String cardinalities(String text) {
            StringFns.Call c = StringFns.findCall(text, 0, Set.of("CARDINALITY"));
            if (c == null) return text;
            StringBuilder out = new StringBuilder();
            int copied = 0;
            while (c != null) {
                out.append(text, copied, c.start());
                String arg = text.substring(c.argStart(), c.argEnd()).trim();
                String real = QuotedIdentifiers.real(arg);
                String key = "C:" + real;
                String name = byKey.get(key);
                if (name == null) {
                    Path p = resolve(real, true);
                    name = NativeLeaves.maskName(next++);
                    masks.add(new CardMask(name, text.substring(c.start(), c.end()), p));
                    byKey.put(key, name);
                }
                out.append(name);
                copied = c.end();
                c = StringFns.findCall(text, copied, Set.of("CARDINALITY"));
            }
            return out.append(text, copied, text.length()).toString();
        }

        private String leaf(String real) {
            String key = "P:" + real;
            String name = byKey.get(key);
            if (name != null) return name;
            Path p = resolve(real, false);
            name = NativeLeaves.maskName(next++);
            columns.add(new PathColumn(name, p, p.end));
            byKey.put(key, name);
            return name;
        }

        /** Resolves {@code real} (a path) against the file's Arrow schema. */
        private Path resolve(String real, boolean container) {
            String top = topLevel(real);
            Field f = arrow.findField(top);
            if (f == null || file.kind(top) == null) {
                throw new ParquetSqlException("Unknown column \"" + top + "\" in " + real);
            }
            Path p = new Path(real, top);
            String rest = real.substring(top.length());
            int i = 0;
            while (i < rest.length()) {
                char c = rest.charAt(i);
                if (c == '.') {
                    int j = i + 1;
                    while (j < rest.length() && rest.charAt(j) != '.' && rest.charAt(j) != '[') j++;
                    String name = rest.substring(i + 1, j);
                    if (!(f.getType() instanceof ArrowType.Struct)) {
                        throw new ParquetSqlException(real + ": '" + describe(f) + "' is " + f.getType() + ", not a struct, so it has no field '" + name + "'");
                    }
                    int ord = -1;
                    for (int k = 0; k < f.getChildren().size(); k++) if (f.getChildren().get(k).getName().equals(name)) { ord = k; break; }
                    if (ord < 0) {
                        List<String> have = new ArrayList<>();
                        for (Field ch : f.getChildren()) have.add(ch.getName());
                        throw new ParquetSqlException(real + ": struct has no field '" + name + "' (fields: " + have + ")");
                    }
                    p.steps.add(new Step(StepKind.FIELD, name, 0, ord));
                    f = f.getChildren().get(ord);
                    i = j;
                } else if (c == '[') {
                    int close;
                    String key = null;
                    long num = 0;
                    if (rest.startsWith("['", i)) {
                        StringBuilder sb = new StringBuilder();
                        int j = i + 2;
                        while (true) {
                            if (j >= rest.length()) throw new ParquetSqlException("Malformed path " + real);
                            char d = rest.charAt(j);
                            if (d == '\'') {
                                if (j + 1 < rest.length() && rest.charAt(j + 1) == '\'') { sb.append('\''); j += 2; continue; }
                                break;
                            }
                            sb.append(d);
                            j++;
                        }
                        key = sb.toString();
                        close = j + 1; // the ']'
                    } else {
                        close = rest.indexOf(']', i);
                        if (close < 0) throw new ParquetSqlException("Malformed path " + real);
                        num = Long.parseLong(rest.substring(i + 1, close));
                    }
                    if (f.getType() instanceof ArrowType.Map) {
                        Field entries = f.getChildren().get(0);
                        Field keyF = entries.getChildren().get(0);
                        if (key != null && !(keyF.getType() instanceof ArrowType.Utf8)) {
                            throw new ParquetSqlException(real + ": the map's keys are " + keyF.getType() + ", not strings; use an integer key");
                        }
                        if (key == null && TableSchema.kindOf(keyF.getType()) == ColumnKind.STRING) {
                            throw new ParquetSqlException(real + ": the map's keys are strings; use ['key']");
                        }
                        p.steps.add(key != null ? new Step(StepKind.KEY, key, 0, 1) : new Step(StepKind.INDEX, null, num, 1));
                        f = entries.getChildren().get(1);
                    } else if (f.getType() instanceof ArrowType.List) {
                        if (key != null) throw new ParquetSqlException(real + ": a list is indexed by position, not by key");
                        p.steps.add(new Step(StepKind.INDEX, null, num, 0));
                        f = f.getChildren().get(0);
                    } else {
                        throw new ParquetSqlException(real + ": " + describe(f) + " is " + f.getType() + ", not a list or map, so it cannot be indexed");
                    }
                    i = close + 1;
                } else {
                    throw new ParquetSqlException("Malformed path " + real);
                }
            }
            if (container) {
                if (!(f.getType() instanceof ArrowType.List || f.getType() instanceof ArrowType.Map)) {
                    throw new ParquetSqlException("CARDINALITY needs a list or map, but " + real + " is " + f.getType());
                }
            } else {
                if (f.getType() instanceof ArrowType.Struct || f.getType() instanceof ArrowType.List || f.getType() instanceof ArrowType.Map) {
                    throw new ParquetSqlException(real + " ends at a " + f.getType() + ", not a value. Name a field inside it, index into it, or use CARDINALITY(" + real + ")");
                }
                if (TableSchema.kindOf(f.getType()) == ColumnKind.OTHER) {
                    throw new ParquetSqlException(real + " has the unsupported type " + f.getType());
                }
            }
            p.end = f;
            return p;
        }

        private static String describe(Field f) {
            return f.getName();
        }
    }
}
