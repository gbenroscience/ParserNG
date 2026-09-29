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
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.schema.MessageType;

import java.io.IOException;
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
 *
 * <h2>Row-group addressing</h2>
 * Every worker reader is opened with the <em>same</em> record filter and {@code useColumnIndexFilter(true)}
 * as the sequential path, so each one applies the same row-group-level pruning at open time and ends up with
 * the same surviving row-group list ({@link ParquetFileReader#getRowGroups()}). Work is addressed by
 * <b>position in that list</b> -- exactly what {@code ParquetFileReader.readFilteredRowGroup(int)} indexes --
 * not by footer ordinal; see {@link RowGroupDecoder#readFilteredRowGroup}. Because the address is only
 * meaningful if every worker's list matches the one the caller computed, each worker verifies its list against
 * {@code expectedStartingPos} (the surviving blocks' file offsets, in order) at construction and fails fast on
 * any mismatch instead of silently decoding the wrong data. The same reader then prunes pages within the
 * row group via {@code readFilteredRowGroup}. See {@link RowGroupDecoder}'s class Javadoc for how the decode
 * loop stays correct either way.
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

    private final ParquetSource file;
    private final ExecutorService executor;
    private final BlockingQueue<RowGroupDecoder> pool;
    private final List<RowGroupDecoder> allDecoders;
    private final ArrayDeque<Future<List<VectorSchemaRoot>>> window = new ArrayDeque<>();
    private final int windowSize;
    private final List<Field> fields;
    private final BufferAllocator allocator;
    private final int batchSize;
    private final Schema schema;
    private final int survivorCount;
    private int nextIndex;

    private List<VectorSchemaRoot> currentBatchList;
    private int currentBatchPos;
    private VectorSchemaRoot current;
    private boolean currentDetached;
    private boolean finished;
    private boolean closed;

    ParallelSource(ParquetSource file, MessageType projected, ColumnDescriptor[] descs, NodePlan nodePlan,
                   List<Field> fields, long[] expectedStartingPos, FilterCompat.Filter filter,
                   int parallelism, int batchSize, long maxRowGroupBytes,
                   BufferAllocator allocator, ScanMetrics metrics) {
        this.file = file;
        this.fields = fields;
        this.allocator = allocator;
        this.batchSize = batchSize;
        this.schema = new Schema(fields);
        this.windowSize = parallelism + 1;
        this.survivorCount = expectedStartingPos.length;
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
                ParquetReadOptions workerOpts = ParquetBatchReader.newReadOptions(filter);
                ParquetFileReader r = ParquetFileReader.open(file.input(), workerOpts);
                RowGroupDecoder d;
                try {
                    verifySurvivors(r, expectedStartingPos);
                    r.setRequestedSchema(projected);
                    d = new RowGroupDecoder(file, r, projected, nodePlan, descs, maxRowGroupBytes, metrics);
                } catch (RuntimeException e) {
                    try { r.close(); } catch (IOException | RuntimeException ignored) { }
                    throw e;
                }
                allDecoders.add(d);
                pool.add(d);
            }
        } catch (IOException e) {
            for (RowGroupDecoder d : allDecoders) d.close();
            throw new ParquetScanException("Cannot open parallel Parquet workers", file, -1, null, e);
        } catch (RuntimeException e) {
            for (RowGroupDecoder d : allDecoders) d.close();
            throw e;
        }

        this.executor = Executors.newFixedThreadPool(parallelism, r -> {
            Thread t = new Thread(r, "parquet-scan-" + file.shortName());
            t.setDaemon(true);
            return t;
        });

        for (int i = 0; i < windowSize && nextIndex < survivorCount; i++) submitNext();
    }

    /**
     * Fails fast if this worker's surviving row-group list is not the list the caller computed: index-based
     * addressing is only sound if they are identical, in order.
     */
    private void verifySurvivors(ParquetFileReader r, long[] expectedStartingPos) {
        List<BlockMetaData> mine = r.getRowGroups();
        boolean ok = mine.size() == expectedStartingPos.length;
        for (int i = 0; ok && i < expectedStartingPos.length; i++) {
            ok = mine.get(i).getStartingPos() == expectedStartingPos[i];
        }
        if (!ok) {
            throw new ParquetScanException("Parallel worker's row-group pruning disagrees with the scan's ("
                    + mine.size() + " vs " + expectedStartingPos.length + " surviving row groups, or different blocks); "
                    + "the file may have changed while being scanned", file, -1, null, null);
        }
    }

    private void submitNext() {
        final int index = nextIndex++;
        window.addLast(executor.submit(() -> decodeOneRowGroup(index)));
    }

    private List<VectorSchemaRoot> decodeOneRowGroup(int index) throws Exception {
        RowGroupDecoder d = pool.take();
        List<VectorSchemaRoot> out = new ArrayList<>();
        try {
            d.readFilteredRowGroup(index);
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
            if (nextIndex < survivorCount) submitNext();
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
