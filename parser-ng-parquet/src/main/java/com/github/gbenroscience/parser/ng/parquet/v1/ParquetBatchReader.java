package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.NodePlan;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.PredicateTranslator;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.RowFilterEvaluator;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.filter2.compat.FilterCompat;
import org.apache.parquet.filter2.predicate.FilterPredicate;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Streams a Parquet file as Arrow batches: primitives and nested (struct/list/map) columns alike.
 *
 * <h2>Pruning</h2>
 * Row-group-level pruning (statistics/dictionary/Bloom filter) happens once, up front, against the
 * footer, in both modes. Page-level pruning (column indexes) happens too, in both modes: sequential
 * mode's one reader and every one of parallel mode's worker readers are opened with the same
 * translated predicate and {@code useColumnIndexFilter(true)}, and both read row groups through
 * parquet-java's <em>filtered</em> entry points (see {@link RowGroupDecoder}'s Javadoc). Either way,
 * pushdown here only ever prunes -- the caller must still re-evaluate the original predicate against
 * every emitted batch; see {@link Predicate}'s Javadoc for the exact soundness contract.
 *
 * <h2>Go-fast vs. go-slow: pushdown alone, or exactFilter()</h2>
 * By default ({@code ParquetScan.pushdown(...)} without {@code .exactFilter()}), this class only
 * prunes: an emitted batch may still contain non-matching rows and the caller must re-check. Calling
 * {@code .exactFilter()} additionally routes every surviving row through a {@link RowFilterEvaluator}
 * (built once, up front, by {@link com.github.gbenroscience.parser.ng.parquet.v1.internal.RowFilterEvaluator}) and emits
 * only true matches, via {@link FilteredSource}. See {@code ParquetScan.exactFilter()}'s Javadoc for
 * when each mode is the right choice, and {@link FilteredSource}'s Javadoc for exactly what exact
 * mode costs relative to pruning-only.
 *
 * <h2>Ownership</h2>
 * <ul>
 *   <li>Sequential mode ({@code parallelism == 1}, no exactFilter): the reader owns ONE reusable
 *       {@link VectorSchemaRoot}, reallocated only when a vector's capacity is too small. Steady state
 *       allocates no Arrow memory.</li>
 *   <li>Parallel mode ({@code parallelism > 1}): each row group's batches are freshly allocated on a
 *       worker thread, since they cross a thread boundary; see {@link ParallelSource}.</li>
 *   <li>{@code exactFilter()} mode: every emitted batch is freshly allocated via selective copy,
 *       regardless of parallelism -- see {@link FilteredSource}.</li>
 *   <li>{@link #root()} is valid only until the next {@link #next()} or {@link #close()}.
 *       {@link #detach()} moves/hands off the current batch (no copy) to a root the CALLER owns and must
 *       close; the reader continues from its next batch.</li>
 *   <li>The caller-supplied allocator is never closed by the reader.</li>
 * </ul>
 *
 * <h2>Nested types</h2>
 * A projected column may be a struct, a 3-level {@code LIST}/{@code MAP} (or the legacy 2-level/tuple
 * forms), or any nesting of these over supported primitive leaves. See {@code internal.NodePlan} /
 * {@code internal.LevelWalker} for how Parquet's repetition/definition levels become Arrow list offsets
 * and struct validity. Unsupported leaf types fail fast, naming the column. {@code exactFilter()}
 * predicates, unlike pushdown predicates, are restricted to flat top-level columns only -- see
 * {@link RowFilterEvaluator}'s Javadoc.
 *
 * <h2>Zero-copy vs. copy</h2>
 * Values are decompressed and decoded by the native decode engine, then written once into Arrow
 * buffers: decode plus one copy, not zero-copy. Only {@link #detach()} and downstream transfer are
 * zero-copy (exactFilter mode's selective-copy batches can still be detached zero-copy from this
 * point on; the copy already happened building them, not on detach).
 */
public final class ParquetBatchReader implements AutoCloseable {

    private final ParquetSource file;
    private final BatchSource source;
    private final ScanMetrics metrics; // null when disabled
    private boolean closed;

    ParquetBatchReader(ParquetSource file, List<String> columns, Predicate predicate, int batchSize,
                       boolean metricsOn, long maxRowGroupBytes, int parallelism, boolean exactFilter,
                       Map<String, CustomPredicate> customPredicates, BufferAllocator allocator) {
        if (allocator == null) throw new NullPointerException("allocator");
        this.file = file;
        ParquetFileReader probeOrFiltered = null;
        try {
            // The footer is read and parsed exactly once per scan; every reader below is opened over it.
            ParquetMetadata footer = file.readFooter();
            MessageType fileSchema = footer.getFileMetaData().getSchema();

            FilterCompat.Filter filter = FilterCompat.NOOP;
            FilterPredicate fp = PredicateTranslator.translate(predicate, fileSchema);
            if (fp != null) filter = FilterCompat.get(fp);
            // useColumnIndexFilter(true): page-level pruning within a surviving row group, not just
            // row-group-level pruning. This reader is reused directly as the decode reader in
            // sequential mode (below), and the same filter + flag are threaded into ParallelSource
            // for each of its worker readers, so both modes get identical pruning behavior -- see
            // RowGroupDecoder's class Javadoc for exactly how the decode loop stays correct either way.
            ParquetReadOptions opts = newReadOptions(filter);

            ParquetFileReader filtered = file.openReader(footer, opts);
            probeOrFiltered = filtered;
            MessageType projected = project(fileSchema, columns);
            List<ColumnDescriptor> colList = projected.getColumns();
            ColumnDescriptor[] descriptors = colList.toArray(new ColumnDescriptor[0]);
            NodePlan nodePlan = NodePlan.build(projected, descriptors, file);
            List<Field> fields = nodePlan.fields();

            // Validated -- and, if invalid, thrown -- before opening any row-group reader/worker: an
            // exactFilter() request that cannot actually be honored (predicate touches a nested column,
            // or a column outside the projection) should fail before this scan does any I/O, not after.
            RowFilterEvaluator exactEvaluator = null;
            if (exactFilter) {
                Set<String> flatLeafNames = new HashSet<>();
                for (NodePlan.Node top : nodePlan.tops()) if (top instanceof NodePlan.Leaf) flatLeafNames.add(top.name);
                exactEvaluator = new RowFilterEvaluator(predicate, fields, flatLeafNames, customPredicates, file);
            }

            ScanMetrics m = null;
            if (metricsOn) {
                m = new ScanMetrics();
                m.rowGroupsInFile = filtered.getFooter().getBlocks().size();
                m.rowGroupsAfterPruning = filtered.getRowGroups().size();
            }
            this.metrics = m;

            BatchSource built;
            if (parallelism <= 1) {
                filtered.setRequestedSchema(projected);
                RowGroupDecoder decoder = new RowGroupDecoder(
                        file, filtered, projected, nodePlan, descriptors, maxRowGroupBytes, m);
                probeOrFiltered = null; // ownership moves to the decoder
                built = new SequentialSource(file, decoder, fields, allocator, batchSize);
            } else {
                // Workers address row groups by position in the reader's surviving-block list (what
                // readFilteredRowGroup(int) indexes), NOT by footer ordinal. Record each survivor's file
                // offset so every worker can verify its own pruning produced this exact list.
                List<BlockMetaData> survivorBlocks = filtered.getRowGroups();
                long[] survivorStartingPos = new long[survivorBlocks.size()];
                for (int i = 0; i < survivorStartingPos.length; i++) survivorStartingPos[i] = survivorBlocks.get(i).getStartingPos();
                filtered.close(); // parallel workers open their own readers; this one is no longer needed
                probeOrFiltered = null;
                built = new ParallelSource(file, footer, projected, descriptors, nodePlan, fields,
                        survivorStartingPos, filter, parallelism, batchSize, maxRowGroupBytes, allocator, m);
            }
            this.source = (exactEvaluator != null) ? new FilteredSource(built, exactEvaluator, allocator) : built;
        } catch (IOException e) {
            closeQuietly(probeOrFiltered);
            throw new ParquetScanException("Cannot open Parquet file", file, -1, null, e);
        } catch (RuntimeException e) {
            closeQuietly(probeOrFiltered);
            if (e instanceof ParquetScanException) throw e;
            throw new ParquetScanException("Invalid or unsupported Parquet file", file, -1, null, e);
        }
    }

    /**
     * The one place scan read options are built, used by this class's own reader and by every
     * {@link ParallelSource} worker (a fresh instance per call; see the note there on why they are not shared).
     * Parallel addressing requires every reader to compute the SAME surviving row-group list, so all
     * pruning switches are pinned explicitly here rather than left to library defaults that could differ
     * between call sites or parquet-java versions.
     */
    static ParquetReadOptions newReadOptions(FilterCompat.Filter filter) {
        return ParquetReadOptions.builder()
                .useStatsFilter(true).useDictionaryFilter(true).useBloomFilter(true)
                .useColumnIndexFilter(true).withRecordFilter(filter).build();
    }

    private static MessageType project(MessageType schema, List<String> columns) {
        if (columns == null) return schema;
        Set<String> seen = new HashSet<>();
        List<Type> fs = new ArrayList<>(columns.size());
        for (String c : columns) {
            if (!schema.containsField(c)) {
                throw new ParquetScanException("Unknown column; available: " + schema.getFields().stream()
                        .map(Type::getName).toList(), (ParquetSource) null, -1, c, null);
            }
            if (seen.add(c)) fs.add(schema.getType(c));
        }
        return new MessageType(schema.getName(), fs);
    }

    /** Schema of the batches (projected columns only). */
    public Schema schema() { return source.schema(); }

    /** Present only if the scan was built with {@code withMetrics(true)}. */
    public ScanMetrics metrics() { return metrics; }

    /** The current batch. See class docs for lifetime. */
    public VectorSchemaRoot root() {
        ensureOpen();
        return source.root();
    }

    /**
     * Advances to the next batch.
     *
     * @return false at end of data; true if {@link #root()} now holds 1..batchSize rows
     */
    public boolean next() {
        ensureOpen();
        return source.next();
    }

    /** Hands the current batch to the caller (zero-copy); the reader continues with its next batch. */
    public VectorSchemaRoot detach() {
        ensureOpen();
        return source.detach();
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("ParquetBatchReader is closed");
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        source.close();
    }

    private static void closeQuietly(ParquetFileReader r) {
        try { if (r != null) r.close(); } catch (IOException | RuntimeException ignored) { }
    }
}
