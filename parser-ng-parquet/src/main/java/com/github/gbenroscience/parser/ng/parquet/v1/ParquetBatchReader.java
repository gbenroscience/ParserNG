package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.NodePlan;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.PredicateTranslator;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.RepeatedStatsPruner;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.RowFilterEvaluator;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.PanamaSnappyCodecFactory;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.PureJavaZstdCodecFactory;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.filter2.compat.FilterCompat;
import org.apache.parquet.filter2.predicate.FilterPredicate;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopCodecs;
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
 * Row-group-level pruning happens once, up front, against the footer, in both modes: parquet-java's
 * statistics/dictionary/Bloom filters for flat and struct-nested leaves, plus footer-statistics pruning
 * for leaves inside lists and maps ({@code internal.RepeatedStatsPruner}). Page-level pruning (column
 * indexes) is deliberately off, see {@link #newReadOptions}. Either way, pushdown here only ever prunes --
 * the caller must still re-evaluate the original predicate against every emitted batch; see
 * {@link Predicate}'s Javadoc for the exact soundness contract.
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
 * and struct validity. Unsupported leaf types fail fast, naming the column. Predicates may name a leaf
 * inside a struct, list or map by a dotted path (for example {@code "user.address.zip"} or
 * {@code "items.price"}), both for pruning and for {@code exactFilter()}; a leaf under a list means
 * "some element satisfies it". See {@code internal.ColumnPaths}, {@code internal.RepeatedStatsPruner} and
 * {@link RowFilterEvaluator}.
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
                       Map<String, CustomPredicate> customPredicates, boolean pinWorkerThreads,
                       Boolean useJavaSnappyDecoder, Boolean useJavaZstdDecoder, BufferAllocator allocator) {
        if (allocator == null) throw new NullPointerException("allocator");
        this.file = file;
        ParquetFileReader probeOrFiltered = null;
        try {
            // The footer is read and parsed exactly once per scan; every reader below is opened over it.
            ParquetMetadata footer = file.readFooter();
            MessageType fileSchema = footer.getFileMetaData().getSchema();

            // Row-group pruning happens in two independent stages, both only ever removing groups that provably
            // hold no match. Stage 1 (here): predicates over leaves inside lists/maps, which parquet-java's own
            // filter cannot express, are checked against footer statistics and the losers are dropped from the
            // footer every reader is then opened over. Stage 2 (below): parquet-java's stats/dictionary/Bloom
            // filter for flat and struct-nested leaves. A group is read only if both stages allow it.
            ParquetMetadata scanFooter = RepeatedStatsPruner.prune(predicate, footer);

            FilterCompat.Filter filter = FilterCompat.NOOP;
            FilterPredicate fp = PredicateTranslator.translate(predicate, fileSchema);
            if (fp != null) filter = FilterCompat.get(fp);
            // The same filter is threaded into ParallelSource for each of its worker readers, so both modes
            // prune identically (see newReadOptions for why page-level pruning is deliberately off).
            ParquetReadOptions opts = newReadOptions(filter, useJavaSnappyDecoder, useJavaZstdDecoder);

            ParquetFileReader filtered = file.openReader(scanFooter, opts);
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
                exactEvaluator = new RowFilterEvaluator(predicate, fields, flatLeafNames, customPredicates, file, projected);
            }

            ScanMetrics m = null;
            if (metricsOn) {
                m = new ScanMetrics();
                m.rowGroupsInFile = footer.getBlocks().size(); // the file's own count, before either pruning stage
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
                built = new ParallelSource(file, scanFooter, projected, descriptors, nodePlan, fields,
                        survivorStartingPos, filter, parallelism, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder,
                        batchSize, maxRowGroupBytes, allocator, m);
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
     *
     * <p><b>Column-index (page-level) pruning is deliberately OFF.</b> With it on, parquet-java hands back a
     * page store holding only the pages that overlap the surviving row ranges. Pages are kept whole, and each
     * column's pages start on its own row boundaries, so different columns of the same row group can begin at
     * different rows, and a column can hold rows outside the ranges. {@code RowGroupDecoder} reads each column
     * straight from the first retained page and never consults the store's row-index selection, so with
     * trimmed pages the columns of one batch could belong to different rows. Row-group pruning (statistics,
     * dictionary, Bloom filter, and {@link RepeatedStatsPruner}) has no such hazard, because a group is read
     * whole or not at all. Re-enabling this needs the decoder to honour the row ranges per column, including
     * for list columns where a page may only be skipped at a record boundary.
     */
    static ParquetReadOptions newReadOptions(FilterCompat.Filter filter) {
        return newReadOptions(filter, null, null);
    }

    /** SNAPPY override only; ZSTD defers to the system property. Kept for callers that predate the ZSTD option. */
    static ParquetReadOptions newReadOptions(FilterCompat.Filter filter, Boolean useJavaSnappyDecoder) {
        return newReadOptions(filter, useJavaSnappyDecoder, null);
    }

    /**
     * {@code useJavaSnappyDecoder}: this scan's {@code ParquetScan#useJavaSnappyDecoder} override, or
     * {@code null} to defer to the {@code -Dparserng.snappy=panama} system property (the pre-existing,
     * JVM-wide default both still fall back to). Explicit {@code TRUE}/{@code FALSE} here always wins.
     *
     * <p>{@code useJavaZstdDecoder}: the same for ZSTD -- this scan's {@code ParquetScan#useJavaZstdDecoder}
     * override, or {@code null} to defer to the {@code -Dparserng.zstd=java} system property.
     *
     * <p>The two are independent and compose: the factories are decorators that each intercept only their own
     * codec and forward everything else, so enabling both stacks them over parquet-java's default factory.
     */
    static ParquetReadOptions newReadOptions(FilterCompat.Filter filter, Boolean useJavaSnappyDecoder, Boolean useJavaZstdDecoder) {
        ParquetReadOptions.Builder b = ParquetReadOptions.builder()
                .useStatsFilter(true).useDictionaryFilter(true).useBloomFilter(true)
                .useColumnIndexFilter(false).withRecordFilter(filter);
        boolean javaSnappy = (useJavaSnappyDecoder != null)
                ? useJavaSnappyDecoder
                : "panama".equals(System.getProperty("parserng.snappy"));
        boolean javaZstd = (useJavaZstdDecoder != null)
                ? useJavaZstdDecoder
                : "java".equals(System.getProperty("parserng.zstd"));
        // A fresh factory per call, like the options themselves: decompressors are never shared across threads.
        if (javaSnappy || javaZstd) {
            CompressionCodecFactory factory = HadoopCodecs.newFactory(0);
            if (javaSnappy) factory = new PanamaSnappyCodecFactory(factory);
            if (javaZstd) factory = new PureJavaZstdCodecFactory(factory);
            b.withCodecFactory(factory);
        }
        return b.build();
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
