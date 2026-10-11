package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

import org.apache.parquet.io.InputFile;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Immutable scan specification for one Parquet file: projection + pruning predicate + batch size +
 * parallelism. SQL-agnostic: a planner (e.g. parser-ng-sql's Parquet bridge) builds one of these from
 * its own plan.
 *
 * <pre>{@code
 * try (BufferAllocator alloc = new RootAllocator();
 *      ParquetBatchReader r = ParquetScan.scan(path)
 *              .select("x", "y")
 *              .pushdown(Predicate.gt("x", 100))   // pruning hint: result is a SUPERSET
 *              .parallelism(4)                     // decode up to 4 row groups concurrently
 *              .open(alloc)) {
 *     while (r.next()) {
 *         VectorSchemaRoot batch = r.root();       // valid until the next next()/close()
 *         // ... hand to parser-ng-arrow, which applies the exact predicate
 *     }
 * }
 * }</pre>
 *
 * <p>The spec deliberately calls the predicate method {@code pushdown}, not {@code filter}: by
 * default it prunes, it does not filter. See {@link Predicate}. Call {@link #exactFilter()} to opt
 * into exact row-level filtering instead — the fast/exact choice is explicit and per-scan; see that
 * method's Javadoc for the tradeoff.
 *
 * <p>Reads only — this module no longer includes a writer.
 */
public final class ParquetScan {

    public static final int DEFAULT_BATCH_SIZE = 32_768;
    public static final long DEFAULT_MAX_ROW_GROUP_BYTES = 2L << 30; // 2 GiB uncompressed, untrusted-input guard
    /** Sequential by default: predictable, zero-steady-state-allocation. Opt into parallel with {@link #parallelism}. */
    public static final int DEFAULT_PARALLELISM = 1;

    private final ParquetSource file;
    private final List<String> columns; // null = all
    private final Predicate predicate;  // null = none
    private final int batchSize;
    private final boolean metrics;
    private final long maxRowGroupBytes;
    private final int parallelism;
    private final boolean exactFilter;
    private final Map<String, CustomPredicate> customPredicates; // never null; empty by default
    private final boolean pinWorkerThreads;
    /** null = no per-scan override, defer to the {@code -Dparserng.snappy} system property. See {@link #useJavaSnappyDecoder}. */
    private final Boolean useJavaSnappyDecoder;
    /** null = no per-scan override, defer to the {@code -Dparserng.zstd} system property. See {@link #useJavaZstdDecoder}. */
    private final Boolean useJavaZstdDecoder;

    private ParquetScan(ParquetSource file, List<String> columns, Predicate predicate, int batchSize,
                        boolean metrics, long maxRowGroupBytes, int parallelism, boolean exactFilter,
                        Map<String, CustomPredicate> customPredicates, boolean pinWorkerThreads,
                        Boolean useJavaSnappyDecoder, Boolean useJavaZstdDecoder) {
        this.file = file;
        this.columns = columns;
        this.predicate = predicate;
        this.batchSize = batchSize;
        this.metrics = metrics;
        this.maxRowGroupBytes = maxRowGroupBytes;
        this.parallelism = parallelism;
        this.exactFilter = exactFilter;
        this.customPredicates = customPredicates;
        this.pinWorkerThreads = pinWorkerThreads;
        this.useJavaSnappyDecoder = useJavaSnappyDecoder;
        this.useJavaZstdDecoder = useJavaZstdDecoder;
    }

    /** Scans a local file. */
    public static ParquetScan scan(Path file) {
        return scan(ParquetSource.of(file));
    }

    /** Scans any parquet-java {@link InputFile}; see {@link ParquetSource} for what that lets you read. */
    public static ParquetScan scan(InputFile file) {
        return scan(ParquetSource.of(file));
    }

    /** Scans a {@link ParquetSource}, e.g. {@code ParquetSource.hadoop("s3a://bucket/key.parquet", conf)}. */
    public static ParquetScan scan(ParquetSource file) {
        if (file == null) throw new NullPointerException("file");
        return new ParquetScan(file, null, null, DEFAULT_BATCH_SIZE, false, DEFAULT_MAX_ROW_GROUP_BYTES,
                DEFAULT_PARALLELISM, false, Map.of(), false, null, null);
    }

    /** Top-level columns to read. Unlisted columns are never decoded. Must be non-empty. */
    public ParquetScan select(String... cols) {
        if (cols == null || cols.length == 0) throw new IllegalArgumentException("select() needs at least one column");
        return new ParquetScan(file, List.copyOf(Arrays.asList(cols)), predicate, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /**
     * Pruning predicate. Columns may be flat, or dotted paths into structs, lists and maps (see
     * {@link Predicate}); leaves under a list are pruned from footer statistics with "some element" meaning.
     * By default this is a pruning hint ONLY — the result is a SUPERSET, and the
     * caller must still apply the same condition itself (see class Javadoc's example, and
     * {@link Predicate}). Replaces any earlier predicate; combine with {@link Predicate#and}.
     * Chain {@link #exactFilter()} onto this call to make the scan itself apply the condition
     * exactly instead.
     */
    public ParquetScan pushdown(Predicate p) {
        return new ParquetScan(file, columns, p, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /**
     * Registers the implementation behind one {@link Predicate.Custom} id. Only consulted under
     * {@link #exactFilter()} — see {@link CustomPredicate}'s Javadoc for the full contract (in
     * particular: null handling and exception propagation are the implementation's own
     * responsibility, not handled for it). Calling this again with the same id replaces the previous
     * registration; {@code exactFilter()} throws, naming the id, if a {@link Predicate.Custom} node
     * references one that was never registered.
     */
    public ParquetScan withCustomPredicate(String id, CustomPredicate impl) {
        if (id == null || id.isEmpty()) throw new IllegalArgumentException("id must not be null/empty");
        if (impl == null) throw new NullPointerException("impl");
        Map<String, CustomPredicate> updated = new LinkedHashMap<>(customPredicates);
        updated.put(id, impl);
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, Map.copyOf(updated), pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /**
     * Go-slow, exact mode: escalates {@link #pushdown} from a pruning hint to a real row filter.
     * Row-group pruning still happens exactly as before (this is additive, not a
     * replacement), but every row that survives pruning is then tested against the predicate again,
     * and only true matches are emitted — see {@code FilteredSource} for the mechanism and its cost.
     *
     * <h2>When to reach for this</h2>
     * Use it when this scan's output is the end of the line for filtering (ad hoc queries, tests,
     * anything that isn't handing batches to a downstream engine that already re-applies the
     * predicate). Skip it — stay on the default pruning-only {@link #pushdown} — when a downstream
     * consumer (a query engine, {@code parser-ng-arrow}, etc.) is going to evaluate the same
     * condition anyway; exact mode would then just pay row-level filtering cost twice for the same
     * result.
     *
     * <h2>Requirements, checked at {@link #open}</h2>
     * Every column a {@link Predicate.Cmp}, {@link Predicate.In}, {@link Predicate.IsNull},
     * {@link Predicate.Like} or {@link Predicate.Regex} leaf touches must name a primitive leaf: a flat
     * top-level column, or a dotted path into a struct, list or map ({@code "user.address.zip"},
     * {@code "items.price"}; see {@link Predicate} for the spellings and the "some element" meaning of a
     * path through a list). Its top-level column must be part of the projection ({@link #select}, or the
     * default "select everything"). {@link Predicate.ColCmp} still requires two flat top-level columns, and
     * a {@link Predicate.Custom} leaf's {@code touchedColumns} need only be projected (see that record's
     * Javadoc). A {@link Predicate.Not},
     * {@link Predicate.ColCmp}, {@link Predicate.Like}, and {@link Predicate.Regex} are all fully
     * supported here even though none of them can ever be pushed for pruning — see {@link Predicate}'s
     * Javadoc on exactly which node types pushdown can and cannot use. A predicate that fails either
     * requirement throws {@link ParquetScanException} naming the offending column, rather than
     * silently falling back to pruning-only behavior; a caller who opted into "exact" is entitled to
     * know if this scan cannot actually deliver that.
     *
     * @throws IllegalStateException if called before {@link #pushdown}
     */
    public ParquetScan exactFilter() {
        if (predicate == null) throw new IllegalStateException("exactFilter() needs a predicate — call pushdown(...) first");
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, true, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    public ParquetScan batchSize(int rows) {
        if (rows <= 0) throw new IllegalArgumentException("batchSize must be positive");
        return new ParquetScan(file, columns, predicate, rows, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    public ParquetScan withMetrics(boolean enabled) {
        return new ParquetScan(file, columns, predicate, batchSize, enabled, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /** Rejects any surviving row group whose declared uncompressed size exceeds this. */
    public ParquetScan maxRowGroupBytes(long bytes) {
        if (bytes <= 0) throw new IllegalArgumentException("maxRowGroupBytes must be positive");
        return new ParquetScan(file, columns, predicate, batchSize, metrics, bytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /**
     * Number of row groups decoded concurrently. {@code 1} (the default) is fully sequential and reuses
     * one set of Arrow buffers for every batch. A value above 1 uses that many decoder threads/file
     * handles and delivers batches in the SAME order a sequential scan would, at the cost of allocating a
     * fresh batch per row group instead of reusing buffers. Choosing a value larger than the file's row
     * group count wastes threads; the caller/planner is expected to pick a sensible number (e.g.
     * {@code min(desiredParallelism, availableProcessors())}) — this class applies no adaptive logic.
     */
    public ParquetScan parallelism(int rowGroupsAtOnce) {
        if (rowGroupsAtOnce <= 0) throw new IllegalArgumentException("parallelism must be positive");
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, rowGroupsAtOnce, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /**
     * Opt-in, best-effort pinning of the parallel decoder threads to CPUs, one logical CPU per worker, spread
     * over distinct physical cores first (SMT siblings only after every core has a worker). Off by default,
     * because whether it helps is workload- and machine-dependent and it can hurt.
     *
     * <ul>
     *   <li>Only the decoder threads created for {@link #parallelism} {@code > 1} are pinned. With
     *       {@code parallelism == 1} decoding runs on the caller's own thread, which is never pinned by this
     *       library, so the flag has no effect there.</li>
     *   <li>Linux and Windows only, and only if the JVM was started with
     *       {@code --enable-native-access=ALL-UNNAMED}. Anything else (macOS, blocked native access, a
     *       cpuset that refuses the CPU, more than 64 Windows CPUs in group 0) leaves the affected workers
     *       unpinned and the scan otherwise unchanged; it never throws or changes results.</li>
     *   <li>A pinned thread cannot be moved off a busy core. If the consumer thread, other scans, GC or other
     *       processes compete for the same cores it can be slower than leaving the scheduler free.</li>
     *   <li>Check {@link ScanMetrics#workersPinned()} (needs {@link #withMetrics}) to see how many workers were
     *       actually pinned, and compare wall time with and without the flag on your own hardware.</li>
     * </ul>
     */
    public ParquetScan pinWorkerThreads(boolean pin) {
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pin, useJavaSnappyDecoder, useJavaZstdDecoder);
    }

    /** Whether {@link #pinWorkerThreads} was requested (default false). */
    public boolean pinsWorkerThreads() { return pinWorkerThreads; }

    /**
     * Per-scan override of which SNAPPY decompressor this scan uses, taking precedence over the
     * {@code -Dparserng.snappy} system property for this scan only. {@code true} forces the pure-Java
     * decoder (no native/JNI library involved -- see {@code internal.codec.SnappyDecompressor}),
     * {@code false} forces the stock {@code snappy-java} (JNI-wrapped native) decoder regardless of the
     * system property. Unset (the default) defers to the system property, which itself defaults to the
     * stock decoder.
     *
     * <p>Independently correct either way (see {@code SnappyDecompressor}'s fuzz suite), but, as measured
     * on the benchmarks in this module's README, not currently faster than the native decoder it can
     * replace -- reach for this to avoid the native dependency, not for a performance win, until that
     * changes.
     */
    public ParquetScan useJavaSnappyDecoder(boolean useJava) {
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJava, useJavaZstdDecoder);
    }

    /** This scan's SNAPPY-decoder override: {@code TRUE}/{@code FALSE} if {@link #useJavaSnappyDecoder} was called, else {@code null} (defers to the system property). */
    public Boolean javaSnappyDecoderOverride() { return useJavaSnappyDecoder; }

    /**
     * Per-scan override of which ZSTD decompressor this scan uses, taking precedence over the
     * {@code -Dparserng.zstd=java} system property for this scan only. {@code true} forces the pure-Java decoder
     * (no native/JNI library involved -- see {@code internal.codec.ZstdDecompressor}; {@code zstd-jni} is not even
     * loaded), {@code false} forces parquet-java's stock ZSTD codec ({@code zstd-jni}, JNI-wrapped native) regardless
     * of the system property. Unset (the default) defers to the system property, which itself defaults to the
     * stock codec.
     *
     * <p>Composes with {@link #useJavaSnappyDecoder}: enabling both serves SNAPPY and ZSTD in pure Java; all other
     * codecs always use parquet-java's own.
     *
     * <p>Independently correct either way (the decoder is checked against thousands of frames produced by the
     * reference libzstd, plus corrupted input), but not faster than the native decoder it replaces -- on the
     * standalone decoder benchmarks it runs at roughly half of libzstd's speed. Reach for this to avoid the native
     * dependency (no {@code zstd-jni}, no JNI), not for a performance win.
     */
    public ParquetScan useJavaZstdDecoder(boolean useJava) {
        return new ParquetScan(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJava);
    }

    /** This scan's ZSTD-decoder override: {@code TRUE}/{@code FALSE} if {@link #useJavaZstdDecoder} was called, else {@code null} (defers to the system property). */
    public Boolean javaZstdDecoderOverride() { return useJavaZstdDecoder; }

    /**
     * Opens the scan. The caller owns {@code allocator} (it must outlive the reader and any detached
     * roots); the returned reader owns its file handle(s) and must be closed.
     */
    public ParquetBatchReader open(BufferAllocator allocator) {
        return new ParquetBatchReader(file, columns, predicate, batchSize, metrics, maxRowGroupBytes, parallelism, exactFilter, customPredicates, pinWorkerThreads, useJavaSnappyDecoder, useJavaZstdDecoder, allocator);
    }

    /** Convenience loop; each batch is valid only for the duration of the callback. */
    public void forEachBatch(BufferAllocator allocator, Consumer<VectorSchemaRoot> consumer) {
        try (ParquetBatchReader r = open(allocator)) {
            while (r.next()) consumer.accept(r.root());
        }
    }
}
