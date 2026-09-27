package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.NodePlan;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.filter2.compat.FilterCompat;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.schema.MessageType;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Decodes row groups on a small worker pool while delivering batches to the caller in the SAME order a
 * sequential scan would: row groups in file order, batches within a row group in order. Only decoding is
 * parallel; nothing about scan semantics changes.
 *
 * <h2>Bounded concurrency, two knobs</h2>
 * <ul>
 *   <li>{@code parallelism} readers/decoders exist, so at most {@code parallelism} row groups are ever
 *       being decoded at once — never one task per row group.</li>
 *   <li>The submission window is bounded at {@code parallelism + 1} in-flight row groups, so decoded-but-
 *       not-yet-consumed batches cannot grow without bound ahead of a slow consumer.</li>
 * </ul>
 * Each worker owns one {@link ParquetFileReader} for its own file handle and codec state (parquet-java's
 * decompressors are not safe to share across threads), obtained by re-opening the file once per worker.
 * Row-group-level survivorship was already decided once, up front, by the caller ({@code survivorOrdinals}
 * is exactly the row groups this scan needs to visit), so each worker reader jumps straight to its
 * assigned ordinal via {@link RowGroupDecoder#readRowGroup} rather than re-deriving that decision. It is,
 * however, opened with the <em>same</em> record filter and {@code useColumnIndexFilter(true)} as the
 * sequential path, so {@code readFilteredRowGroup} can still prune pages within an already-decided
 * row group — page-level pruning is not row-group-level pruning and both modes need it independently.
 * See {@link RowGroupDecoder}'s class Javadoc for how the decode loop stays correct either way.
 *
 * <p>Unlike {@link SequentialSource}, each row group's batches are freshly allocated (not reused), since
 * they are produced on a worker thread and handed to the consumer thread. This trades some allocation for
 * concurrency; {@link SequentialSource} remains the zero-steady-state-allocation path.
 *
 * <p><b>Known limitation:</b> if {@link #close()} runs while a worker is mid-decode, that worker's
 * {@link RowGroupDecoder} may not be returned to the pool before shutdown and is closed directly instead;
 * this is safe (no leak) but means an in-flight task is abandoned rather than allowed to finish.
 */
final class ParallelSource implements BatchSource {

    private final Path file;
    private final ExecutorService executor;
    private final BlockingQueue<RowGroupDecoder> pool;
    private final List<RowGroupDecoder> allDecoders;
    private final ArrayDeque<Future<List<VectorSchemaRoot>>> window = new ArrayDeque<>();
    private final int windowSize;
    private final List<Field> fields;
    private final BufferAllocator allocator;
    private final int batchSize;
    private final Schema schema;
    private final int[] ordinals;
    private int nextOrdinalPos;
    private int survivorCounter;

    private List<VectorSchemaRoot> currentBatchList;
    private int currentBatchPos;
    private VectorSchemaRoot current;
    private boolean currentDetached;
    private boolean finished;
    private boolean closed;

    ParallelSource(Path file, MessageType projected, ColumnDescriptor[] descs, NodePlan nodePlan,
                   List<Field> fields, List<Integer> survivorOrdinals, FilterCompat.Filter filter,
                   int parallelism, int batchSize, long maxRowGroupBytes,
                   BufferAllocator allocator, ScanMetrics metrics) {
        this.file = file;
        this.fields = fields;
        this.allocator = allocator;
        this.batchSize = batchSize;
        this.schema = new Schema(fields);
        this.windowSize = parallelism + 1;
        this.ordinals = survivorOrdinals.stream().mapToInt(Integer::intValue).toArray();
        if (metrics != null) metrics.parallelism = parallelism;

        this.pool = new ArrayBlockingQueue<>(parallelism);
        this.allDecoders = new ArrayList<>(parallelism);
        try {
            for (int i = 0; i < parallelism; i++) {
                // Built fresh per worker, deliberately NOT hoisted out of this loop and shared: this
                // options object owns a Configuration and (through ParquetFileReader) ends up owning
                // codec-resolution/decompressor state, none of which parquet-java or Hadoop's
                // Configuration guarantee is safe under concurrent first-use from multiple threads.
                // A single shared instance here is compatible with a single-threaded caller and with
                // SequentialSource, which is exactly why this bug did not show up there — it only
                // needs two *concurrent* first-touches of the same options/codec state, which is
                // precisely what parallelism >= 2 workers opening readers that then decode their first
                // (compressed) row group at the same time produces. See this class's own class Javadoc:
                // "parquet-java's decompressors are not safe to share across threads" -- that invariant
                // was upheld for the ParquetFileReader/RowGroupDecoder objects themselves (one each per
                // worker, pooled, never checked out by two threads at once) but was being silently
                // violated one layer up, by handing every worker's reader the same options instance.
                ParquetReadOptions workerOpts = ParquetReadOptions.builder()
                        .useColumnIndexFilter(true).withRecordFilter(filter).build();
                ParquetFileReader r = ParquetFileReader.open(new LocalInputFile(file), workerOpts);
                r.setRequestedSchema(projected);
                RowGroupDecoder d = new RowGroupDecoder(file, r, projected, nodePlan, descs, maxRowGroupBytes, metrics);
                allDecoders.add(d);
                pool.add(d);
            }
        } catch (IOException e) {
            for (RowGroupDecoder d : allDecoders) d.close();
            throw new ParquetScanException("Cannot open parallel Parquet workers", file, -1, null, e);
        }

        this.executor = Executors.newFixedThreadPool(parallelism, r -> {
            Thread t = new Thread(r, "parquet-scan-" + file.getFileName());
            t.setDaemon(true);
            return t;
        });

        for (int i = 0; i < windowSize && nextOrdinalPos < ordinals.length; i++) submitNext();
    }

    private void submitNext() {
        final int ordinal = ordinals[nextOrdinalPos++];
        final int survivorIndex = survivorCounter++;
        window.addLast(executor.submit(() -> decodeOneRowGroup(ordinal, survivorIndex)));
    }

    private List<VectorSchemaRoot> decodeOneRowGroup(int ordinal, int survivorIndex) throws Exception {
        RowGroupDecoder d = pool.take();
        List<VectorSchemaRoot> out = new ArrayList<>();
        try {
            d.readRowGroup(ordinal, survivorIndex);
            while (d.remaining() > 0) {
                int n = (int) Math.min(batchSize, d.remaining());
                FieldVector[] arr = new FieldVector[fields.size()];
                List<FieldVector> vecs = new ArrayList<>(arr.length);
                for (int i = 0; i < arr.length; i++) {
                    FieldVector v = fields.get(i).createVector(allocator);
                    v.setInitialCapacity(n);
                    v.allocateNew();
                    arr[i] = v;
                    vecs.add(v);
                }
                try {
                    d.fill(arr, n);
                } catch (RuntimeException e) {
                    for (FieldVector v : vecs) v.close();
                    throw e;
                }
                out.add(new VectorSchemaRoot(new Schema(fields), vecs, n));
            }
            return out;
        } finally {
            pool.put(d);
        }
    }

    @Override public Schema schema() { return schema; }

    @Override
    public boolean next() {
        ensureOpen();
        advanceCurrent();
        return current != null;
    }

    private void advanceCurrent() {
        if (currentDetached) { current = null; currentDetached = false; }
        else if (current != null) { current.close(); current = null; }

        while (true) {
            if (currentBatchList != null && currentBatchPos < currentBatchList.size()) {
                current = currentBatchList.get(currentBatchPos++);
                return;
            }
            if (finished) { current = null; return; }
            if (window.isEmpty()) { finished = true; current = null; return; }
            try {
                currentBatchList = window.pollFirst().get();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new ParquetScanException("Interrupted while waiting for a decoded row group", file, -1, null, ie);
            } catch (ExecutionException ee) {
                Throwable c = ee.getCause();
                throw (c instanceof ParquetScanException pse) ? pse
                        : new ParquetScanException("Parallel decode failed", file, -1, null, c);
            }
            currentBatchPos = 0;
            if (nextOrdinalPos < ordinals.length) submitNext();
        }
    }

    @Override
    public VectorSchemaRoot root() {
        ensureOpen();
        if (current == null) throw new IllegalStateException("next() has not returned true");
        return current;
    }

    @Override
    public VectorSchemaRoot detach() {
        VectorSchemaRoot r = root();
        currentDetached = true;
        return r;
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("ParquetBatchReader is closed");
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (current != null && !currentDetached) current.close();
        if (currentBatchList != null) {
            for (int i = currentBatchPos; i < currentBatchList.size(); i++) currentBatchList.get(i).close();
        }
        for (Future<List<VectorSchemaRoot>> f : window) {
            try {
                for (VectorSchemaRoot r : f.get()) r.close();
            } catch (Exception ignored) { }
        }
        executor.shutdownNow();
        try { executor.awaitTermination(30, TimeUnit.SECONDS); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
        for (RowGroupDecoder d : allDecoders) d.close();
    }
}
