package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.TransferPair;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Single-threaded scan that reuses one root's buffers for every batch. */
final class SequentialSource implements BatchSource {

    private final Path file;
    private final RowGroupDecoder decoder;
    private final BufferAllocator allocator;
    private final int batchSize;
    private final FieldVector[] vectors;
    private final VectorSchemaRoot root;
    private boolean closed;

    SequentialSource(Path file, RowGroupDecoder decoder, List<Field> fields, BufferAllocator allocator, int batchSize) {
        this.file = file;
        this.decoder = decoder;
        this.allocator = allocator;
        this.batchSize = batchSize;
        this.vectors = new FieldVector[fields.size()];
        List<FieldVector> list = new ArrayList<>(vectors.length);
        try {
            for (int i = 0; i < vectors.length; i++) {
                vectors[i] = fields.get(i).createVector(allocator);
                list.add(vectors[i]);
            }
        } catch (RuntimeException e) {
            for (FieldVector v : list) v.close();
            throw e;
        }
        this.root = new VectorSchemaRoot(new Schema(fields), list, 0);
    }

    @Override public Schema schema() { return root.getSchema(); }

    @Override
    public VectorSchemaRoot root() {
        ensureOpen();
        return root;
    }

    @Override
    public boolean next() {
        ensureOpen();
        try {
            while (decoder.remaining() == 0) {
                if (!decoder.nextFilteredRowGroup()) {
                    root.setRowCount(0);
                    return false;
                }
            }
            int n = (int) Math.min(batchSize, decoder.remaining());
            RowGroupDecoder.prepare(vectors, batchSize);
            decoder.fill(vectors, n);
            root.setRowCount(n);
            return true;
        } catch (ParquetScanException e) {
            throw e;
        } catch (IOException | RuntimeException e) {
            throw new ParquetScanException("Failed reading Parquet data", file, -1, null, e);
        }
    }

    @Override
    public VectorSchemaRoot detach() {
        ensureOpen();
        List<FieldVector> out = new ArrayList<>(vectors.length);
        for (FieldVector v : vectors) {
            TransferPair tp = v.getTransferPair(allocator);
            tp.transfer();
            out.add((FieldVector) tp.getTo());
        }
        int rows = root.getRowCount();
        root.setRowCount(0);
        return new VectorSchemaRoot(root.getSchema(), out, rows);
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("ParquetBatchReader is closed");
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        decoder.close();
        try { root.close(); } catch (RuntimeException ignored) { }
    }
}
