package com.github.gbenroscience.parser.ng.parquet.v1.internal;

import com.github.gbenroscience.parser.ng.parquet.v1.CustomPredicate;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScanException;
import com.github.gbenroscience.parser.ng.parquet.v1.Predicate;

import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.FixedSizeBinaryVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetSource;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;

/**
 * Compiles a {@link Predicate} into a tree that can be evaluated, row by row, directly against the
 * already-decoded {@link FieldVector}s of a {@code VectorSchemaRoot} — the "go-slow, exact" half of
 * {@code ParquetScan.exactFilter()} (see that method's Javadoc for when to reach for it instead of
 * the default pruning-only {@code pushdown()}), and the ONLY evaluator that gives every
 * {@link Predicate} node type — including {@link Predicate.Not}, {@link Predicate.ColCmp},
 * {@link Predicate.Like}, {@link Predicate.Regex}, and {@link Predicate.Custom}, none of which
 * {@code PredicateTranslator} can ever push down (see each type's own Javadoc for exactly why each
 * one is a hard limit of statistics-based pruning, not a translation gap) — actual filtering power.
 *
 * <h2>Validated once, at construction — not per row, not silently</h2>
 * Every column a {@link Predicate.Cmp}/{@link Predicate.In}/{@link Predicate.IsNull}/
 * {@link Predicate.ColCmp}/{@link Predicate.Like}/{@link Predicate.Regex} leaf touches must be a
 * flat, top-level leaf that is also part of the projection; a {@link Predicate.Custom} leaf's
 * {@code touchedColumns} need only be part of the projection (not necessarily flat — see that
 * record's Javadoc for why). Every literal must be an exact/widening type match for that column's
 * Arrow type, using the identical coercion rules {@code PredicateTranslator} already applies for
 * pushdown. {@link Predicate.ColCmp} additionally requires both columns to resolve to the exact same
 * Arrow kind — no cross-type numeric widening between two columns, consistent with this model's
 * "never coerce" rule everywhere else. Any violation throws {@link ParquetScanException} naming the
 * offending column, at construction, before any row is ever read.
 *
 * <h2>Null semantics</h2>
 * Standard SQL {@code WHERE} semantics: a null operand makes {@link Predicate.Cmp}/
 * {@link Predicate.In}/{@link Predicate.ColCmp}/{@link Predicate.Like}/{@link Predicate.Regex}
 * evaluate to {@code false} regardless of operator (a comparison against {@code NULL} is unknown,
 * and an unknown {@code WHERE} result excludes the row) — including {@code NE}.
 * {@link Predicate.IsNull}/{@code IsNotNull} are the only built-in nodes that test nullity directly;
 * a {@link Predicate.Custom} implementation owns its own null handling entirely (see
 * {@link CustomPredicate}'s Javadoc).
 */
public final class RowFilterEvaluator {

    private final Node root;

    public RowFilterEvaluator(Predicate predicate, List<Field> fields, Set<String> flatLeafNames,
                               Map<String, CustomPredicate> customPredicates, ParquetSource file) {
        Map<String, Integer> indexOf = new HashMap<>();
        for (int i = 0; i < fields.size(); i++) indexOf.put(fields.get(i).getName(), i);
        Map<String, CustomPredicate> customs = customPredicates == null ? Map.of() : customPredicates;
        this.root = compile(predicate, fields, indexOf, flatLeafNames, customs, file);
    }

    /** @return true if the row at {@code row} in {@code batch} satisfies the original predicate exactly. */
    public boolean matches(VectorSchemaRoot batch, int row) {
        return root.eval(batch, row);
    }

    // =====================================================================
    // compiled node tree
    // =====================================================================

    private interface Node {
        boolean eval(VectorSchemaRoot batch, int row);
    }

    private record AndNode(Node l, Node r) implements Node {
        public boolean eval(VectorSchemaRoot b, int row) { return l.eval(b, row) && r.eval(b, row); }
    }

    private record OrNode(Node l, Node r) implements Node {
        public boolean eval(VectorSchemaRoot b, int row) { return l.eval(b, row) || r.eval(b, row); }
    }

    /** Boolean negation composes trivially at this layer -- no De Morgan rewriting needed (contrast PredicateTranslator.negate). */
    private record NotNode(Node inner) implements Node {
        public boolean eval(VectorSchemaRoot b, int row) { return !inner.eval(b, row); }
    }

    /** Kind of the Arrow column being compared -- resolved once at compile time, not per row. */
    private enum Kind { BOOL, INT8, INT16, INT32, DATE_DAY, INT64, TIMESTAMP, FLOAT, DOUBLE, UTF8,
                         UINT8, UINT16, UINT32, UINT64, DECIMAL, UUID }

    private final class CmpNode implements Node {
        final int colIndex;
        final Kind kind;
        final Predicate.Op op;
        final long numLit;      // BOOL(0/1)/INT8/16/32/DATE_DAY/INT64/TIMESTAMP/UINT8/16/32(widened non-negative)/UINT64(raw bit pattern)
        final float floatLit;   // FLOAT
        final double doubleLit; // DOUBLE
        final String strLit;    // UTF8
        final BigDecimal decLit; // DECIMAL
        final byte[] uuidLit;    // UUID

        CmpNode(int colIndex, Kind kind, Predicate.Op op, long numLit, float floatLit, double doubleLit,
                String strLit, BigDecimal decLit, byte[] uuidLit) {
            this.colIndex = colIndex; this.kind = kind; this.op = op;
            this.numLit = numLit; this.floatLit = floatLit; this.doubleLit = doubleLit; this.strLit = strLit;
            this.decLit = decLit; this.uuidLit = uuidLit;
        }

        public boolean eval(VectorSchemaRoot batch, int row) {
            FieldVector v = batch.getVector(colIndex);
            if (v.isNull(row)) return false; // NULL compared to anything: unknown -> excluded, even for NE
            return switch (kind) {
                case BOOL -> cmpLong(((BitVector) v).get(row), op, numLit);
                case INT8 -> cmpLong(((TinyIntVector) v).get(row), op, numLit);
                case INT16 -> cmpLong(((SmallIntVector) v).get(row), op, numLit);
                case INT32 -> cmpLong(((IntVector) v).get(row), op, numLit);
                case DATE_DAY -> cmpLong(((DateDayVector) v).get(row), op, numLit); // DateDayVector is NOT an IntVector
                case INT64 -> cmpLong(((BigIntVector) v).get(row), op, numLit);
                case TIMESTAMP -> cmpLong(((TimeStampVector) v).get(row), op, numLit); // every unit/tz variant extends TimeStampVector, none extends BigIntVector
                case FLOAT -> cmpDouble(((Float4Vector) v).get(row), op, floatLit);
                case DOUBLE -> cmpDouble(((Float8Vector) v).get(row), op, doubleLit);
                case UTF8 -> cmpString(readUtf8((VarCharVector) v, row), op, strLit);
                // UINT8/16/32 always fit as a non-negative long once masked -- ordinary signed cmpLong
                // is then exact, no unsigned-comparison special-casing needed (see the class's literal
                // coercion notes). Only UINT64 can exceed Long.MAX_VALUE and genuinely needs one.
                case UINT8 -> cmpLong(((UInt1Vector) v).get(row) & 0xFFL, op, numLit);
                case UINT16 -> cmpLong(((UInt2Vector) v).get(row) & 0xFFFFL, op, numLit);
                case UINT32 -> cmpLong(((UInt4Vector) v).get(row) & 0xFFFFFFFFL, op, numLit);
                case UINT64 -> cmpUnsignedLong(((UInt8Vector) v).get(row), op, numLit);
                case DECIMAL -> cmpBigDecimal(((DecimalVector) v).getObject(row), op, decLit);
                case UUID -> cmpUuid(uuidBytes((FixedSizeBinaryVector) v, row), op, uuidLit); // op is EQ/NE only -- enforced at compile time
            };
        }
    }

    private final class InNode implements Node {
        final int colIndex;
        final Kind kind;
        final Set<Long> numSet;     // BOOL/INT8/16/32/DATE_DAY/INT64/TIMESTAMP/UINT8/16/32(widened)/UINT64(raw bit pattern)
        final Set<Float> floatSet;  // FLOAT
        final Set<Double> doubleSet; // DOUBLE
        final Set<String> strSet;   // UTF8
        final List<BigDecimal> decList; // DECIMAL -- BigDecimal.equals() is scale-sensitive (1.50 != 1.5), so a
                                         // HashSet would silently miss matches; linear-scan with compareTo()==0 instead
        final List<byte[]> uuidList;    // UUID -- byte[] has no useful equals/hashCode either, same reasoning

        InNode(int colIndex, Kind kind, Set<Long> numSet, Set<Float> floatSet, Set<Double> doubleSet,
               Set<String> strSet, List<BigDecimal> decList, List<byte[]> uuidList) {
            this.colIndex = colIndex; this.kind = kind;
            this.numSet = numSet; this.floatSet = floatSet; this.doubleSet = doubleSet; this.strSet = strSet;
            this.decList = decList; this.uuidList = uuidList;
        }

        public boolean eval(VectorSchemaRoot batch, int row) {
            FieldVector v = batch.getVector(colIndex);
            if (v.isNull(row)) return false;
            return switch (kind) {
                case BOOL -> numSet.contains((long) ((BitVector) v).get(row));
                case INT8 -> numSet.contains((long) ((TinyIntVector) v).get(row));
                case INT16 -> numSet.contains((long) ((SmallIntVector) v).get(row));
                case INT32 -> numSet.contains((long) ((IntVector) v).get(row));
                case DATE_DAY -> numSet.contains((long) ((DateDayVector) v).get(row));
                case INT64 -> numSet.contains(((BigIntVector) v).get(row));
                case TIMESTAMP -> numSet.contains(((TimeStampVector) v).get(row));
                case FLOAT -> floatSet.contains(((Float4Vector) v).get(row));
                case DOUBLE -> doubleSet.contains(((Float8Vector) v).get(row));
                case UTF8 -> strSet.contains(readUtf8((VarCharVector) v, row));
                // Equality is bit-pattern equality regardless of signed/unsigned interpretation, so IN
                // needs none of CmpNode's unsigned-comparison care -- the raw widened/bit-pattern value
                // (see numLit's Javadoc above) is exactly what went into numSet, so contains() just works.
                case UINT8 -> numSet.contains(((UInt1Vector) v).get(row) & 0xFFL);
                case UINT16 -> numSet.contains(((UInt2Vector) v).get(row) & 0xFFFFL);
                case UINT32 -> numSet.contains(((UInt4Vector) v).get(row) & 0xFFFFFFFFL);
                case UINT64 -> numSet.contains(((UInt8Vector) v).get(row));
                case DECIMAL -> {
                    BigDecimal got = ((DecimalVector) v).getObject(row);
                    boolean found = false;
                    for (BigDecimal cand : decList) if (got.compareTo(cand) == 0) { found = true; break; }
                    yield found;
                }
                case UUID -> {
                    byte[] got = uuidBytes((FixedSizeBinaryVector) v, row);
                    boolean found = false;
                    for (byte[] cand : uuidList) if (Arrays.equals(got, cand)) { found = true; break; }
                    yield found;
                }
            };
        }
    }

    private record IsNullNode(int colIndex, boolean negated) implements Node {
        public boolean eval(VectorSchemaRoot batch, int row) {
            boolean isNull = batch.getVector(colIndex).isNull(row);
            return negated != isNull;
        }
    }

    /** Both columns already validated (at compile time) to resolve to the same {@link Kind}. */
    private final class ColCmpNode implements Node {
        final int leftIndex, rightIndex;
        final Kind kind;
        final Predicate.Op op;

        ColCmpNode(int leftIndex, int rightIndex, Kind kind, Predicate.Op op) {
            this.leftIndex = leftIndex; this.rightIndex = rightIndex; this.kind = kind; this.op = op;
        }

        public boolean eval(VectorSchemaRoot batch, int row) {
            FieldVector l = batch.getVector(leftIndex), r = batch.getVector(rightIndex);
            if (l.isNull(row) || r.isNull(row)) return false;
            return switch (kind) {
                case BOOL -> cmpLong(((BitVector) l).get(row), op, ((BitVector) r).get(row));
                case INT8 -> cmpLong(((TinyIntVector) l).get(row), op, ((TinyIntVector) r).get(row));
                case INT16 -> cmpLong(((SmallIntVector) l).get(row), op, ((SmallIntVector) r).get(row));
                case INT32 -> cmpLong(((IntVector) l).get(row), op, ((IntVector) r).get(row));
                case DATE_DAY -> cmpLong(((DateDayVector) l).get(row), op, ((DateDayVector) r).get(row));
                case INT64 -> cmpLong(((BigIntVector) l).get(row), op, ((BigIntVector) r).get(row));
                case TIMESTAMP -> cmpLong(((TimeStampVector) l).get(row), op, ((TimeStampVector) r).get(row));
                case FLOAT -> cmpDouble(((Float4Vector) l).get(row), op, ((Float4Vector) r).get(row));
                case DOUBLE -> cmpDouble(((Float8Vector) l).get(row), op, ((Float8Vector) r).get(row));
                case UTF8 -> cmpString(readUtf8((VarCharVector) l, row), op, readUtf8((VarCharVector) r, row));
                case UINT8 -> cmpLong(((UInt1Vector) l).get(row) & 0xFFL, op, ((UInt1Vector) r).get(row) & 0xFFL);
                case UINT16 -> cmpLong(((UInt2Vector) l).get(row) & 0xFFFFL, op, ((UInt2Vector) r).get(row) & 0xFFFFL);
                case UINT32 -> cmpLong(((UInt4Vector) l).get(row) & 0xFFFFFFFFL, op, ((UInt4Vector) r).get(row) & 0xFFFFFFFFL);
                case UINT64 -> cmpUnsignedLong(((UInt8Vector) l).get(row), op, ((UInt8Vector) r).get(row));
                case DECIMAL -> cmpBigDecimal(((DecimalVector) l).getObject(row), op, ((DecimalVector) r).getObject(row));
                case UUID -> cmpUuid(uuidBytes((FixedSizeBinaryVector) l, row), op, uuidBytes((FixedSizeBinaryVector) r, row));
            };
        }
    }

    private record LikeNode(int colIndex, Pattern compiled) implements Node {
        public boolean eval(VectorSchemaRoot batch, int row) {
            FieldVector v = batch.getVector(colIndex);
            if (v.isNull(row)) return false;
            return compiled.matcher(readUtf8((VarCharVector) v, row)).matches();
        }
    }

    private record RegexNode(int colIndex, Pattern pattern) implements Node {
        public boolean eval(VectorSchemaRoot batch, int row) {
            FieldVector v = batch.getVector(colIndex);
            if (v.isNull(row)) return false;
            return pattern.matcher(readUtf8((VarCharVector) v, row)).find();
        }
    }

    private record CustomNode(CustomPredicate impl) implements Node {
        public boolean eval(VectorSchemaRoot batch, int row) { return impl.test(batch, row); }
    }

    // =====================================================================
    // compilation (validation happens here, once, not per row)
    // =====================================================================

    private Node compile(Predicate p, List<Field> fields, Map<String, Integer> indexOf,
                          Set<String> flatLeafNames, Map<String, CustomPredicate> customs, ParquetSource file) {
        if (p instanceof Predicate.And a) return new AndNode(compile(a.left(), fields, indexOf, flatLeafNames, customs, file), compile(a.right(), fields, indexOf, flatLeafNames, customs, file));
        if (p instanceof Predicate.Or o) return new OrNode(compile(o.left(), fields, indexOf, flatLeafNames, customs, file), compile(o.right(), fields, indexOf, flatLeafNames, customs, file));
        if (p instanceof Predicate.Not n) return new NotNode(compile(n.inner(), fields, indexOf, flatLeafNames, customs, file));
        if (p instanceof Predicate.Cmp c) return compileCmp(c, fields, indexOf, flatLeafNames, file);
        if (p instanceof Predicate.In in) return compileIn(in, fields, indexOf, flatLeafNames, file);
        if (p instanceof Predicate.IsNull n) {
            int idx = resolveFlat(n.column(), indexOf, flatLeafNames, file);
            return new IsNullNode(idx, n.negated());
        }
        if (p instanceof Predicate.ColCmp cc) return compileColCmp(cc, fields, indexOf, flatLeafNames, file);
        if (p instanceof Predicate.Like lk) return compileLike(lk, fields, indexOf, flatLeafNames, file);
        if (p instanceof Predicate.Regex rx) return compileRegex(rx, fields, indexOf, flatLeafNames, file);
        if (p instanceof Predicate.Custom cu) return compileCustom(cu, indexOf, customs, file);
        throw new ParquetScanException("Unrecognized Predicate node " + p.getClass(), file, -1, null, null);
    }

    private Node compileCmp(Predicate.Cmp c, List<Field> fields, Map<String, Integer> indexOf, Set<String> flatLeafNames, ParquetSource file) {
        int idx = resolveFlat(c.column(), indexOf, flatLeafNames, file);
        Kind kind = kindOf(fields.get(idx), file);
        if (kind == Kind.BOOL && c.op() != Predicate.Op.EQ && c.op() != Predicate.Op.NE) {
            throw new ParquetScanException("Only EQ/NE are meaningful on a BOOLEAN column", file, -1, c.column(), null);
        }
        if (kind == Kind.UUID && c.op() != Predicate.Op.EQ && c.op() != Predicate.Op.NE) {
            throw new ParquetScanException("Only EQ/NE are supported on a UUID column (no defined total order "
                    + "is applied here -- see kindOf's Javadoc)", file, -1, c.column(), null);
        }
        return switch (kind) {
            case BOOL -> new CmpNode(idx, kind, c.op(), boolLit(c.value(), c.column()) ? 1 : 0, 0, 0, null, null, null);
            case INT8, INT16, INT32, DATE_DAY -> new CmpNode(idx, kind, c.op(), numericLitAsLong(c.value(), kind, c.column()), 0, 0, null, null, null);
            case INT64, TIMESTAMP -> new CmpNode(idx, kind, c.op(), numericLitAsLong(c.value(), kind, c.column()), 0, 0, null, null, null);
            case UINT8, UINT16, UINT32, UINT64 -> new CmpNode(idx, kind, c.op(), unsignedLitAsLong(c.value(), kind, c.column()), 0, 0, null, null, null);
            case FLOAT -> new CmpNode(idx, kind, c.op(), 0, floatLitValue(c.value(), c.column()), 0, null, null, null);
            case DOUBLE -> new CmpNode(idx, kind, c.op(), 0, 0, doubleLitValue(c.value(), c.column()), null, null, null);
            case UTF8 -> new CmpNode(idx, kind, c.op(), 0, 0, 0, stringLit(c.value(), c.column()), null, null);
            case DECIMAL -> new CmpNode(idx, kind, c.op(), 0, 0, 0, null, bigDecimalLit(c.value(), c.column()), null);
            case UUID -> new CmpNode(idx, kind, c.op(), 0, 0, 0, null, null, uuidLitBytes(c.value(), c.column()));
        };
    }

    private Node compileIn(Predicate.In in, List<Field> fields, Map<String, Integer> indexOf, Set<String> flatLeafNames, ParquetSource file) {
        int idx = resolveFlat(in.column(), indexOf, flatLeafNames, file);
        Kind kind = kindOf(fields.get(idx), file);
        Set<Long> nums = null; Set<Float> floats = null; Set<Double> doubles = null; Set<String> strs = null;
        List<BigDecimal> decs = null; List<byte[]> uuids = null;
        switch (kind) {
            case BOOL, INT8, INT16, INT32, DATE_DAY, INT64, TIMESTAMP -> {
                nums = new HashSet<>();
                for (Object v : in.values()) nums.add(numericLitAsLong(v, kind, in.column()));
            }
            case UINT8, UINT16, UINT32, UINT64 -> {
                nums = new HashSet<>();
                for (Object v : in.values()) nums.add(unsignedLitAsLong(v, kind, in.column()));
            }
            case FLOAT -> {
                floats = new HashSet<>();
                for (Object v : in.values()) floats.add(floatLitValue(v, in.column()));
            }
            case DOUBLE -> {
                doubles = new HashSet<>();
                for (Object v : in.values()) doubles.add(doubleLitValue(v, in.column()));
            }
            case UTF8 -> {
                strs = new HashSet<>();
                for (Object v : in.values()) strs.add(stringLit(v, in.column()));
            }
            case DECIMAL -> {
                decs = new java.util.ArrayList<>();
                for (Object v : in.values()) decs.add(bigDecimalLit(v, in.column()));
            }
            case UUID -> {
                uuids = new java.util.ArrayList<>();
                for (Object v : in.values()) uuids.add(uuidLitBytes(v, in.column()));
            }
        }
        return new InNode(idx, kind, nums, floats, doubles, strs, decs, uuids);
    }

    private Node compileColCmp(Predicate.ColCmp cc, List<Field> fields, Map<String, Integer> indexOf, Set<String> flatLeafNames, ParquetSource file) {
        int li = resolveFlat(cc.left(), indexOf, flatLeafNames, file);
        int ri = resolveFlat(cc.right(), indexOf, flatLeafNames, file);
        Kind lk = kindOf(fields.get(li), file), rk = kindOf(fields.get(ri), file);
        if (lk != rk) {
            throw new ParquetScanException("Column-to-column comparison requires matching types; '" + cc.left()
                    + "' is " + lk + " but '" + cc.right() + "' is " + rk + " -- no cross-type widening is applied",
                    file, -1, cc.left() + " vs " + cc.right(), null);
        }
        if (lk == Kind.BOOL && cc.op() != Predicate.Op.EQ && cc.op() != Predicate.Op.NE) {
            throw new ParquetScanException("Only EQ/NE are meaningful between two BOOLEAN columns", file, -1, cc.left(), null);
        }
        if (lk == Kind.UUID && cc.op() != Predicate.Op.EQ && cc.op() != Predicate.Op.NE) {
            throw new ParquetScanException("Only EQ/NE are supported between two UUID columns", file, -1, cc.left(), null);
        }
        return new ColCmpNode(li, ri, lk, cc.op());
    }

    private Node compileLike(Predicate.Like lk, List<Field> fields, Map<String, Integer> indexOf, Set<String> flatLeafNames, ParquetSource file) {
        int idx = resolveFlat(lk.column(), indexOf, flatLeafNames, file);
        requireUtf8(fields.get(idx), file);
        return new LikeNode(idx, lk.compiled());
    }

    private Node compileRegex(Predicate.Regex rx, List<Field> fields, Map<String, Integer> indexOf, Set<String> flatLeafNames, ParquetSource file) {
        int idx = resolveFlat(rx.column(), indexOf, flatLeafNames, file);
        requireUtf8(fields.get(idx), file);
        return new RegexNode(idx, rx.pattern());
    }

    private Node compileCustom(Predicate.Custom cu, Map<String, Integer> indexOf, Map<String, CustomPredicate> customs, ParquetSource file) {
        CustomPredicate impl = customs.get(cu.id());
        if (impl == null) {
            throw new ParquetScanException("exactFilter() references custom predicate id '" + cu.id()
                    + "' but no implementation was registered -- call ParquetScan.withCustomPredicate(\""
                    + cu.id() + "\", ...) before open()", file, -1, cu.id(), null);
        }
        // Custom columns need not be flat -- see Predicate.Custom's Javadoc -- but must still exist in
        // the projection, so a typo/forgot-to-select is caught here rather than as a confusing NPE
        // deep inside the caller's own implementation later.
        for (String col : cu.touchedColumns()) {
            if (!indexOf.containsKey(col)) {
                throw new ParquetScanException("Custom predicate '" + cu.id() + "' references column '" + col
                        + "' which is not part of the projection", file, -1, col, null);
            }
        }
        return new CustomNode(impl);
    }

    private int resolveFlat(String column, Map<String, Integer> indexOf, Set<String> flatLeafNames, ParquetSource file) {
        if (!flatLeafNames.contains(column)) {
            throw new ParquetScanException(
                    "exactFilter() requires every predicate column to be a flat, top-level column; "
                    + "'" + column + "' is nested or does not exist", file, -1, column, null);
        }
        Integer idx = indexOf.get(column);
        if (idx == null) {
            throw new ParquetScanException(
                    "exactFilter() requires every predicate column to also be part of the projection "
                    + "(select(...)); '" + column + "' is not selected", file, -1, column, null);
        }
        return idx;
    }

    private void requireUtf8(Field f, ParquetSource file) {
        if (kindOf(f, file) != Kind.UTF8) {
            throw new ParquetScanException("LIKE/regex predicates require a STRING column; '" + f.getName()
                    + "' is " + f.getType(), file, -1, f.getName(), null);
        }
    }

    private Kind kindOf(Field f, ParquetSource file) {
        ArrowType at = f.getType();
        if (at instanceof ArrowType.Bool) return Kind.BOOL;
        if (at instanceof ArrowType.Date) return Kind.DATE_DAY;
        if (at instanceof ArrowType.Timestamp) return Kind.TIMESTAMP;
        if (at instanceof ArrowType.Int i) {
            // getIsSigned() matters here: an unsigned column's vector is UInt1/2/4/8Vector, none of
            // which IS-A TinyIntVector/SmallIntVector/IntVector/BigIntVector -- treating it as the
            // signed Kind by bit width alone would compile fine and then throw ClassCastException on
            // the first row evaluated (or, worse, if the two vector families ever did share a common
            // cast target, silently apply signed comparison semantics to an unsigned column).
            if (i.getIsSigned()) {
                return switch (i.getBitWidth()) {
                    case 8 -> Kind.INT8;
                    case 16 -> Kind.INT16;
                    case 32 -> Kind.INT32;
                    case 64 -> Kind.INT64;
                    default -> throw unsupported(f, file);
                };
            }
            return switch (i.getBitWidth()) {
                case 8 -> Kind.UINT8;
                case 16 -> Kind.UINT16;
                case 32 -> Kind.UINT32;
                case 64 -> Kind.UINT64;
                default -> throw unsupported(f, file);
            };
        }
        if (at instanceof ArrowType.FloatingPoint fp) {
            return switch (fp.getPrecision()) {
                case SINGLE -> Kind.FLOAT;
                case DOUBLE -> Kind.DOUBLE;
                default -> throw unsupported(f, file);
            };
        }
        if (at instanceof ArrowType.Utf8) return Kind.UTF8;
        if (at instanceof ArrowType.Decimal) return Kind.DECIMAL;
        if (at instanceof ArrowType.FixedSizeBinary fsb) {
            // 16 bytes is, in this reader, only ever produced for the UUID logical type (see
            // ColumnPlan.forPrimitive) -- no other FIXED_LEN_BYTE_ARRAY use produces a bare
            // FixedSizeBinary Arrow type (DECIMAL on FIXED_LEN_BYTE_ARRAY maps to ArrowType.Decimal
            // instead), so this check is exact, not a heuristic.
            if (fsb.getByteWidth() == 16) return Kind.UUID;
            throw unsupported(f, file);
        }
        throw unsupported(f, file);
    }

    private ParquetScanException unsupported(Field f, ParquetSource file) {
        return new ParquetScanException(
                "exactFilter() does not support filtering column '" + f.getName() + "' of type " + f.getType()
                + " (raw BINARY without STRING/ENUM/JSON annotation is not filterable exactly either -- "
                + "same scope as pushdown's PredicateTranslator)", file, -1, f.getName(), null);
    }

    // ---- literal coercion: mirrors PredicateTranslator's rules exactly; never widens beyond them ----

    private static boolean boolLit(Object v, String column) {
        if (!(v instanceof Boolean b)) throw badLiteral(column, v, "Boolean");
        return b;
    }

    private static long numericLitAsLong(Object v, Kind kind, String column) {
        if (kind == Kind.DATE_DAY) {
            if (v instanceof LocalDate d) return d.toEpochDay();
            throw badLiteral(column, v, "LocalDate (DATE column)");
        }
        if (kind == Kind.BOOL) {
            if (v instanceof Boolean b) return b ? 1 : 0;
            throw badLiteral(column, v, "Boolean");
        }
        if (v instanceof Integer i) return i;
        if ((kind == Kind.INT64 || kind == Kind.TIMESTAMP) && v instanceof Long l) return l;
        throw badLiteral(column, v, (kind == Kind.INT64 || kind == Kind.TIMESTAMP) ? "Integer or Long" : "Integer");
    }

    private static float floatLitValue(Object v, String column) {
        if (!(v instanceof Float f) || f.isNaN()) throw badLiteral(column, v, "Float (non-NaN)");
        return f;
    }

    private static double doubleLitValue(Object v, String column) {
        Double d = null;
        if (v instanceof Double x) d = x;
        else if (v instanceof Float x) d = (double) x;
        else if (v instanceof Integer x) d = (double) x;
        if (d == null || d.isNaN()) throw badLiteral(column, v, "Double, Float or Integer (non-NaN)");
        return d;
    }

    private static String stringLit(Object v, String column) {
        if (!(v instanceof String s)) throw badLiteral(column, v, "String");
        return s;
    }

    /**
     * UINT8/16/32 literals are widened to their non-negative long representation (matching what
     * {@code CmpNode}/{@code InNode} mask the vector's own value down to -- see those classes'
     * {@code numLit} Javadoc), rejecting negative Java literals as almost certainly a mistake (a
     * negative int/long was never a valid unsigned value to begin with).
     *
     * <p>UINT64 is a real, stated limitation: values above {@link Long#MAX_VALUE} exist in a valid
     * UINT64 column but cannot be written as an ordinary non-negative Java {@code long} literal at
     * all -- expressing one would require accepting the value's raw two's-complement bit pattern as a
     * negative {@code long}, which is exactly the kind of "looks like a bug, might be intentional"
     * ambiguity this method refuses to guess through. Only the bottom half of UINT64's range
     * (0..Long.MAX_VALUE) is reachable as a literal here.
     */
    private static long unsignedLitAsLong(Object v, Kind kind, String column) {
        long raw;
        if (v instanceof Integer i) raw = i;
        else if (v instanceof Long l) raw = l;
        else throw badLiteral(column, v, "Integer or Long (non-negative)");
        if (raw < 0) {
            throw badLiteral(column, v, kind == Kind.UINT64
                    ? "a non-negative Long (values above Long.MAX_VALUE are not expressible as a literal here)"
                    : "a non-negative Integer or Long");
        }
        long max = switch (kind) {
            case UINT8 -> 0xFFL;
            case UINT16 -> 0xFFFFL;
            case UINT32 -> 0xFFFFFFFFL;
            default -> Long.MAX_VALUE; // UINT64: already checked non-negative above, nothing tighter to enforce
        };
        if (raw > max) throw badLiteral(column, v, "a value representable in " + kind);
        return raw;
    }

    /** No rescaling: {@code cmpBigDecimal}/{@code cmpUuid} below compare by value ({@code compareTo}/{@code compareTo}==0), not by {@code equals()}, so a literal's scale need not match the column's. */
    private static BigDecimal bigDecimalLit(Object v, String column) {
        if (!(v instanceof BigDecimal d)) throw badLiteral(column, v, "BigDecimal");
        return d;
    }

    /** Accepts a {@link UUID} literal directly, encoded exactly as Parquet's UUID logical type does: 16 bytes, big-endian, most-significant-bits first (RFC 4122 layout -- the same one {@code UUID.toString()} reflects). */
    private static byte[] uuidLitBytes(Object v, String column) {
        if (!(v instanceof UUID u)) throw badLiteral(column, v, "java.util.UUID");
        byte[] b = new byte[16];
        long msb = u.getMostSignificantBits(), lsb = u.getLeastSignificantBits();
        for (int i = 0; i < 8; i++) b[i] = (byte) (msb >>> (8 * (7 - i)));
        for (int i = 0; i < 8; i++) b[8 + i] = (byte) (lsb >>> (8 * (7 - i)));
        return b;
    }

    private static ParquetScanException badLiteral(String column, Object value, String expected) {
        return new ParquetScanException("exactFilter() literal for '" + column + "' must be " + expected
                + ", was " + (value == null ? "null" : value.getClass().getSimpleName()), (ParquetSource) null, -1, column, null);
    }

    private static String readUtf8(VarCharVector v, int row) {
        return new String(v.get(row), StandardCharsets.UTF_8);
    }

    // ---- comparisons ----

    private static boolean cmpLong(long a, Predicate.Op op, long b) {
        return switch (op) {
            case EQ -> a == b; case NE -> a != b;
            case LT -> a < b; case LE -> a <= b;
            case GT -> a > b; case GE -> a >= b;
        };
    }

    private static boolean cmpDouble(double a, Predicate.Op op, double b) {
        return switch (op) {
            case EQ -> a == b; case NE -> a != b;
            case LT -> a < b; case LE -> a <= b;
            case GT -> a > b; case GE -> a >= b;
        };
    }

    private static boolean cmpString(String a, Predicate.Op op, String b) {
        int c = a.compareTo(b);
        return switch (op) {
            case EQ -> c == 0; case NE -> c != 0;
            case LT -> c < 0; case LE -> c <= 0;
            case GT -> c > 0; case GE -> c >= 0;
        };
    }

    /** The one comparison in this class that genuinely cannot use {@code cmpLong}: UINT64 values above {@link Long#MAX_VALUE} have their sign bit set as a raw {@code long}, so ordinary {@code <}/{@code >} would misorder them against smaller values. */
    private static boolean cmpUnsignedLong(long a, Predicate.Op op, long b) {
        int c = Long.compareUnsigned(a, b);
        return switch (op) {
            case EQ -> c == 0; case NE -> c != 0;
            case LT -> c < 0; case LE -> c <= 0;
            case GT -> c > 0; case GE -> c >= 0;
        };
    }

    private static boolean cmpBigDecimal(BigDecimal a, Predicate.Op op, BigDecimal b) {
        int c = a.compareTo(b); // value comparison, not equals() -- 1.50 and 1.5 compare equal here, as they should
        return switch (op) {
            case EQ -> c == 0; case NE -> c != 0;
            case LT -> c < 0; case LE -> c <= 0;
            case GT -> c > 0; case GE -> c >= 0;
        };
    }

    /** {@code op} is guaranteed EQ or NE by the compile-time checks in {@code compileCmp}/{@code compileColCmp} -- no ordering case to handle. */
    private static boolean cmpUuid(byte[] a, Predicate.Op op, byte[] b) {
        boolean eq = Arrays.equals(a, b);
        return op == Predicate.Op.EQ ? eq : !eq;
    }

    private static byte[] uuidBytes(FixedSizeBinaryVector v, int row) {
        return v.get(row); // FixedSizeBinaryVector.get returns a fresh copy, safe to hand to Arrays.equals/store
    }
}