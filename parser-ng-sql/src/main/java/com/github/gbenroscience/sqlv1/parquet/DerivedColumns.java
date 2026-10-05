package com.github.gbenroscience.sqlv1.parquet;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.ArrayList;
import java.util.List;

/**
 * Columns the bridge computes per scan batch and adds to it as <em>virtual</em> columns: string
 * function results ({@link StringFns}), nested field access ({@link NestedFns}) and temporal
 * expressions ({@link TemporalFns}). A virtual column has a name, a {@link ColumnKind} and an Arrow
 * type, so everything downstream (the string, date, timestamp and boolean kernels, numeric widening,
 * the select list) reads it exactly like a file column.
 *
 * <p>Columns are filled in list order and a column may read the ones before it, so a function of a
 * nested field ({@code UPPER(user.name)}) needs no special case. Vectors are reused from batch to
 * batch and released after each execution.
 */
final class DerivedColumns {

    private DerivedColumns() { }

    /** One derived column and how to fill it. */
    abstract static class Column {
        final String name;
        final String text;
        final ColumnKind kind;
        final ArrowType type;

        Column(String name, String text, ColumnKind kind, ArrowType type) {
            this.name = name;
            this.text = text;
            this.kind = kind;
            this.type = type;
        }

        /** A new, named, empty vector of this column's type. */
        abstract FieldVector newVector(BufferAllocator alloc);

        /**
         * Writes rows {@code [0, n)} of {@code out} (already reset): a value or a null for each. {@code src}
         * is the scan batch plus every earlier derived column.
         */
        abstract void fill(VectorSchemaRoot src, int n, FieldVector out);
    }

    /** The reusable vectors of one plan's derived columns. */
    static final class Holder implements AutoCloseable {
        private final List<Column> defs;
        private FieldVector[] vecs;
        private BufferAllocator owner;

        Holder(List<Column> defs) {
            this.defs = List.copyOf(defs);
        }

        boolean isEmpty() {
            return defs.isEmpty();
        }

        /**
         * {@code src} plus one vector per derived column, evaluated for rows {@code [0, n)}. The result is a
         * light holder over the batch's vectors and this holder's: do not close it.
         */
        VectorSchemaRoot extend(VectorSchemaRoot src, BufferAllocator alloc) {
            if (defs.isEmpty()) return src;
            if (vecs == null || owner != alloc) {
                release();
                vecs = new FieldVector[defs.size()];
                owner = alloc;
                try {
                    for (int d = 0; d < vecs.length; d++) {
                        vecs[d] = defs.get(d).newVector(alloc);
                        vecs[d].allocateNew();
                    }
                } catch (RuntimeException e) {
                    release();
                    throw e;
                }
            }
            int n = src.getRowCount();
            List<FieldVector> all = new ArrayList<>(src.getFieldVectors());
            List<Field> fields = new ArrayList<>(all.size() + vecs.length);
            for (FieldVector f : all) fields.add(f.getField());
            VectorSchemaRoot cur = src;
            for (int d = 0; d < vecs.length; d++) {
                FieldVector out = vecs[d];
                out.reset();
                defs.get(d).fill(cur, n, out);
                out.setValueCount(n);
                all.add(out);
                fields.add(out.getField());
                cur = new VectorSchemaRoot(new Schema(fields), new ArrayList<>(all), n);
            }
            return cur;
        }

        void release() {
            if (vecs != null) {
                for (FieldVector v : vecs) if (v != null) v.close();
                vecs = null;
                owner = null;
            }
        }

        @Override
        public void close() {
            release();
        }
    }
}
