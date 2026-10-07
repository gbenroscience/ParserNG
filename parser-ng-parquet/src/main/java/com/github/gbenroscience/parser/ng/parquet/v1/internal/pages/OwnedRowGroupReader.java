package com.github.gbenroscience.parser.ng.parquet.v1.internal.pages;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.SnappyDecompressor;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.Encoding;
import org.apache.parquet.column.page.DataPage;
import org.apache.parquet.column.page.DataPageV1;
import org.apache.parquet.column.page.DataPageV2;
import org.apache.parquet.column.page.DictionaryPage;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.column.page.PageReader;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.format.DataPageHeader;
import org.apache.parquet.format.DataPageHeaderV2;
import org.apache.parquet.format.DictionaryPageHeader;
import org.apache.parquet.format.PageHeader;
import org.apache.parquet.format.PageType;
import org.apache.parquet.format.Util;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ColumnChunkMetaData;
import org.apache.parquet.hadoop.metadata.ColumnPath;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.InputFile;
import org.apache.parquet.io.ParquetDecodingException;
import org.apache.parquet.io.SeekableInputStream;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

/**
 * Reads the pages of one row group's projected column chunks <b>itself</b>, instead of going through
 * {@code ParquetFileReader.readFilteredRowGroup}, and hands them to the decode engine as an ordinary
 * {@link PageReadStore} (already decompressed, exactly the contract {@code FastColumnCursor} asserts).
 *
 * <h2>What it removes</h2>
 * parquet-java reads a chunk into freshly allocated buffers, wraps them in a chain of {@code BytesInput}s,
 * and decompresses every page into a newly allocated array. Here, per row group:
 * <ul>
 *   <li><b>One read per contiguous run of chunks</b>, into one reusable {@code byte[]} (adjacent projected chunks
 *       are coalesced into a single {@code readFully}, like parquet-java's {@code ConsecutivePartList}).</li>
 *   <li><b>UNCOMPRESSED pages are zero-copy slices</b> of that buffer.</li>
 *   <li><b>Compressed data pages decompress into one reusable array per column</b> (grown geometrically, reused
 *       page after page and row group after row group), so a steady-state scan allocates nothing per page.</li>
 *   <li>SNAPPY goes straight from the chunk buffer to the page buffer through {@link SnappyDecompressor}'s array
 *       path when {@code javaSnappy} is set; otherwise (and for every other codec) through parquet-java's own
 *       {@code BytesInputDecompressor}.</li>
 * </ul>
 *
 * <h2>Lifetime contract (the part that must hold for buffer reuse to be correct)</h2>
 * Everything returned by {@link #load} is valid only until the next {@code load} or {@link #close()}.
 * {@code RowGroupDecoder} replaces all of its cursors before it asks for the next row group, and a cursor drops
 * its page references the moment it loads the next page, so nothing is read after its buffer is reused. This is
 * <em>enforced</em>, not assumed: every handed-out {@link PageReader} carries a generation number and throws
 * {@link IllegalStateException} if used after the owner moved on, so a lifetime bug fails loudly in tests
 * instead of decoding recycled bytes. Setting {@code -Dparserng.ownPages.reuseBuffers=false} keeps the
 * single-read / zero-copy chunk path but allocates a fresh array per decompressed page, which isolates a
 * suspected reuse problem.
 *
 * <h2>Memory</h2>
 * Per reader (so per parallel worker) the retained heap is, at most: the row group's projected chunk bytes (only
 * while they total {@value #DEFAULT_MAX_BUFFERED_MIB} MiB or less, see below), plus one decompressed-page buffer per
 * projected column. A row group whose projected chunks total more than {@code -Dparserng.ownPages.maxBuffered}
 * bytes is instead read <b>windowed</b>: each column keeps only a small read-ahead window of its own chunk
 * ({@code -Dparserng.ownPages.window}, default 1 MiB, shrunk when many columns are projected so that the windows
 * together stay inside the budget; always at least as large as the largest page). A window is refilled from the
 * file when the cursor reaches its end, keeping the unread tail, so the file is still read sequentially within
 * every column and each byte is read once. {@code maxBuffered=0} forces windowed reads, a negative value
 * disables them (the pre-windowing behaviour: one buffer for the whole row group, 1 GiB cap). See
 * {@link #retainedBytes()}.
 *
 * <h2>When it steps aside</h2>
 * {@link #load} returns {@code null} -- and the caller reads that row group through parquet-java, which then
 * produces its own diagnostics -- for anything unusual: an encrypted chunk, a projected column missing from the
 * row group, overlapping or out-of-file byte ranges, or (only when windowing is disabled) a row group needing more
 * than 1 GiB of chunk buffer.
 * {@link #fallbackCount()} reports how often that happened. Malformed pages <em>inside</em> a chunk surface lazily
 * as {@link ParquetDecodingException} (the cursor / {@code RowGroupDecoder} attach file, row group and column).
 *
 * <h2>Not done, by design</h2>
 * Page CRC checksums are not verified (parquet-java's default is also off), and page-index / column-index
 * pruning is not applied (this module reads whole row groups; see {@code ParquetBatchReader#newReadOptions}).
 * Index pages and unknown page types are skipped, as parquet-java does.
 *
 * <p>Not thread-safe: one instance per {@code RowGroupDecoder}, which is single-threaded by contract.
 */
public final class OwnedRowGroupReader implements AutoCloseable {

    /** {@code -Dparserng.ownPages=true} turns page-reading ownership on JVM-wide. Default off. */
    public static final String PROP_ENABLE = "parserng.ownPages";
    /** {@code -Dparserng.ownPages.reuseBuffers=false} disables per-column page-buffer reuse (default true). */
    public static final String PROP_REUSE = "parserng.ownPages.reuseBuffers";

    /** Above this, a row group's chunk bytes are not buffered here; parquet-java's multi-buffer path takes over. */
    static final long MAX_CHUNK_BUFFER = 1L << 30;
    /** Per-buffer retention cap: a one-off giant page is allocated, used, and dropped rather than pinned forever. */
    static final int MAX_RETAINED_BUFFER = 16 << 20;
    /** {@code -Dparserng.ownPages.maxBuffered=BYTES}: row groups whose projected chunks exceed this are read windowed. */
    public static final String PROP_MAX_BUFFERED = "parserng.ownPages.maxBuffered";
    /** {@code -Dparserng.ownPages.window=BYTES}: the largest per-column read-ahead window in windowed mode. */
    public static final String PROP_WINDOW = "parserng.ownPages.window";
    static final long DEFAULT_MAX_BUFFERED = 16L << 20;
    static final int DEFAULT_MAX_BUFFERED_MIB = 16;
    static final int DEFAULT_WINDOW = 1 << 20;
    /** Windows are never shrunk below this by the many-columns rule (tests may go lower via the explicit constructor). */
    static final int MIN_WINDOW = 128 << 10;
    /** A page header is parsed from a probe this large first, then from larger and larger prefixes, up to the cap. */
    static final int HEADER_PROBE = 4096;
    static final int MAX_HEADER_BYTES = 4 << 20;
    /** Largest possible Snappy expansion is 64 output bytes per 3 input bytes (~21.4x); 22x plus slack is safe. */
    private static final long SNAPPY_MAX_RATIO = 22;
    private static final byte[] EMPTY = new byte[0];

    public static boolean enabled() {
        return Boolean.parseBoolean(System.getProperty(PROP_ENABLE, "false"));
    }

    private final InputFile input;
    private final String fileName;
    private final CompressionCodecFactory codecs;
    private final boolean javaSnappy;
    private final boolean reuse;
    private final long maxPageBytes;
    private final long maxBuffered;
    private final int windowBytes;

    private SeekableInputStream in;
    private long fileLength = -1;
    private byte[] chunkBuf = EMPTY;
    private byte[][] pageBufs = new byte[0][];
    private byte[][] winBufs = new byte[0][];
    private final PositionedInput headerIn = new PositionedInput();
    private final Map<CompressionCodecName, CompressionCodecFactory.BytesInputDecompressor> decompressors =
            new EnumMap<>(CompressionCodecName.class);

    private int generation;
    private boolean closed;
    private long fallbacks;
    private long rowGroupsLoaded;
    private long windowedLoaded;

    /**
     * @param codecs        supplies decompressors for every codec this class does not decompress itself; released on {@link #close()}
     * @param javaSnappy    decompress SNAPPY with {@link SnappyDecompressor} instead of {@code codecs}
     * @param maxPageBytes  hard cap on any single page's declared uncompressed size (decompression-bomb guard)
     */
    public OwnedRowGroupReader(InputFile input, String fileName, CompressionCodecFactory codecs,
                               boolean javaSnappy, long maxPageBytes) {
        this(input, fileName, codecs, javaSnappy, maxPageBytes,
                Long.getLong(PROP_MAX_BUFFERED, DEFAULT_MAX_BUFFERED),
                (int) Math.max(1L, Math.min(Integer.MAX_VALUE >> 1, Long.getLong(PROP_WINDOW, DEFAULT_WINDOW))));
    }

    /**
     * As above, with the memory knobs explicit (the system properties are only read by the shorter constructor).
     *
     * @param maxBuffered row groups whose projected chunks total more than this many bytes are read windowed;
     *                    {@code 0} always windows, a negative value never does
     * @param windowBytes the largest per-column read-ahead window in windowed mode
     */
    public OwnedRowGroupReader(InputFile input, String fileName, CompressionCodecFactory codecs,
                               boolean javaSnappy, long maxPageBytes, long maxBuffered, int windowBytes) {
        if (input == null) throw new NullPointerException("input");
        if (codecs == null) throw new NullPointerException("codecs");
        this.input = input;
        this.fileName = fileName == null ? String.valueOf(input) : fileName;
        this.codecs = codecs;
        this.javaSnappy = javaSnappy;
        this.maxPageBytes = Math.min(maxPageBytes <= 0 ? Integer.MAX_VALUE : maxPageBytes, Integer.MAX_VALUE - 16L);
        this.reuse = Boolean.parseBoolean(System.getProperty(PROP_REUSE, "true"));
        this.maxBuffered = maxBuffered;
        this.windowBytes = Math.max(1, windowBytes);
    }

    /** How many row groups were declined (and so must be read by parquet-java). */
    public long fallbackCount() { return fallbacks; }

    /** How many row groups this reader served itself. */
    public long loadedRowGroups() { return rowGroupsLoaded; }

    /** How many of those were read windowed (see the class comment, "Memory"). */
    public long windowedRowGroups() { return windowedLoaded; }

    /**
     * Bytes of heap this reader is currently holding on to for reuse: the chunk buffer, every pooled page buffer
     * and every pooled read window. Dictionaries and pages above the retention cap are allocated per use and are
     * not counted.
     */
    public long retainedBytes() {
        long sum = chunkBuf.length;
        for (byte[] b : pageBufs) if (b != null) sum += b.length;
        for (byte[] b : winBufs) if (b != null) sum += b.length;
        return sum;
    }

    /**
     * Loads {@code block}'s chunks for {@code descs}.
     *
     * @return a page store for exactly those columns, or {@code null} if this row group must be read by parquet-java
     */
    public PageReadStore load(BlockMetaData block, ColumnDescriptor[] descs, int rowGroupIndex) throws IOException {
        if (closed) throw new IllegalStateException("closed");
        generation++; // invalidates every PageReader handed out for the previous row group, before its buffers are touched
        final int n = descs.length;

        // ---- 1. locate each projected column's chunk -------------------------------------------------------
        Map<ColumnPath, ColumnChunkMetaData> byPath = new HashMap<>();
        for (ColumnChunkMetaData c : block.getColumns()) byPath.put(c.getPath(), c);
        ColumnChunkMetaData[] chunks = new ColumnChunkMetaData[n];
        for (int i = 0; i < n; i++) {
            ColumnChunkMetaData c = byPath.get(ColumnPath.get(descs[i].getPath()));
            if (c == null || c.isEncrypted()) return declined();
            chunks[i] = c;
        }

        // ---- 2. validate byte ranges ----------------------------------------------------------------------
        if (fileLength < 0) fileLength = input.getLength();
        final long[] start = new long[n];
        final int[] size = new int[n];
        for (int i = 0; i < n; i++) {
            ColumnChunkMetaData c = chunks[i];
            long s = c.getStartingPos();
            long len = c.getTotalSize();
            if (s < 0 || len < 0 || len > MAX_CHUNK_BUFFER || s > fileLength - len) return declined();
            if (len == 0 && c.getValueCount() > 0) return declined();
            if (c.getValueCount() < 0) return declined();
            start[i] = s;
            size[i] = (int) len;
        }

        // ---- 3. order by file offset, reject overlaps, total the bytes ----------------------------------------
        final int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = 1; i < n; i++) { // insertion sort: n is the projected column count
            int v = order[i];
            int j = i - 1;
            while (j >= 0 && start[order[j]] > start[v]) { order[j + 1] = order[j]; j--; }
            order[j + 1] = v;
        }
        long total = 0;
        int nonEmpty = 0;
        long prevEnd = Long.MIN_VALUE;
        for (int k = 0; k < n; k++) {
            int i = order[k];
            if (size[i] == 0) continue;                 // nothing to read; the pager yields no pages
            if (start[i] < prevEnd) return declined();  // overlapping ranges: leave it to parquet-java
            total += size[i];
            nonEmpty++;
            prevEnd = start[i] + size[i];
        }
        final boolean windowed = maxBuffered >= 0 && total > maxBuffered;
        if (!windowed && total > MAX_CHUNK_BUFFER) return declined();
        if (pageBufs.length < n) pageBufs = Arrays.copyOf(pageBufs, n);

        final ChunkPages[] readers = new ChunkPages[n];
        if (windowed) {
            // ---- 4w. no up-front read: every column gets a small window onto its own chunk -------------------
            if (nonEmpty > 0 && in == null) in = input.newStream();
            if (winBufs.length < n) winBufs = Arrays.copyOf(winBufs, n);
            long share = (maxBuffered > 0 ? maxBuffered : DEFAULT_MAX_BUFFERED) / Math.max(1, nonEmpty);
            int w = (int) Math.max(Math.min(MIN_WINDOW, windowBytes), Math.min((long) windowBytes, share));
            for (int i = 0; i < n; i++) {
                ColumnChunkMetaData c = chunks[i];
                readers[i] = new ChunkPages(i, String.join(".", descs[i].getPath()), rowGroupIndex, c.getCodec(), true,
                        start[i], start[i] + size[i], 0, (int) Math.min((long) w, Math.max(1, size[i])),
                        c.getValueCount(), generation);
            }
            windowedLoaded++;
        } else {
            // ---- 4. coalesce exactly-adjacent chunks into runs, read each run with one readFully ----------------
            final int[] bufOff = new int[n];
            final long[] gStart = new long[n];
            final int[] gLen = new int[n];
            final int[] gBuf = new int[n];
            int groups = 0;
            long running = 0;
            prevEnd = Long.MIN_VALUE;
            for (int k = 0; k < n; k++) {
                int i = order[k];
                if (size[i] == 0) { bufOff[i] = 0; continue; }
                if (groups > 0 && start[i] == prevEnd) {        // adjacent: extend the current run
                    bufOff[i] = gBuf[groups - 1] + gLen[groups - 1];
                    gLen[groups - 1] += size[i];
                } else {
                    gStart[groups] = start[i];
                    gLen[groups] = size[i];
                    gBuf[groups] = (int) running;
                    bufOff[i] = (int) running;
                    groups++;
                }
                running += size[i];
                prevEnd = start[i] + size[i];
            }
            ensureChunkBuf((int) total);
            if (groups > 0 && in == null) in = input.newStream();
            for (int g = 0; g < groups; g++) {
                in.seek(gStart[g]);
                in.readFully(chunkBuf, gBuf[g], gLen[g]);
            }
            for (int i = 0; i < n; i++) {
                ColumnChunkMetaData c = chunks[i];
                readers[i] = new ChunkPages(i, String.join(".", descs[i].getPath()), rowGroupIndex, c.getCodec(), false,
                        start[i], start[i] + size[i], bufOff[i], 0, c.getValueCount(), generation);
            }
        }

        // ---- 5. hand out lazily-parsed page readers --------------------------------------------------------
        final Map<ColumnPath, ChunkPages> byColumn = new HashMap<>();
        for (int i = 0; i < n; i++) byColumn.put(chunks[i].getPath(), readers[i]);
        rowGroupsLoaded++;
        return new Store(byColumn, block.getRowCount(), generation);
    }

    private PageReadStore declined() {
        fallbacks++;
        return null;
    }

    private void ensureChunkBuf(int needed) {
        if (chunkBuf.length >= needed) return;
        // Row groups of one file are near-uniform, so ~6% slack is enough to absorb most size jitter without
        // pinning a quarter more than the largest row group needs.
        chunkBuf = new byte[(int) Math.min(MAX_CHUNK_BUFFER, (long) needed + (needed >> 4))];
    }

    /** A page buffer of at least {@code len} bytes for column {@code slot}; fresh when reuse is off or the size is extreme. */
    private byte[] acquirePageBuf(int slot, int len) {
        if (!reuse || len > MAX_RETAINED_BUFFER) return new byte[len];
        byte[] b = pageBufs[slot];
        if (b == null || b.length < len) {
            int grown = (int) Math.min(MAX_RETAINED_BUFFER, Math.max((long) len, b == null ? 0L : (long) b.length + (b.length >> 1)));
            b = new byte[grown];
            pageBufs[slot] = b;
        }
        return b;
    }

    /** A read window of at least {@code need} bytes for column {@code slot} (pooled unless extreme). */
    private byte[] acquireWindow(int slot, int need) {
        if (!reuse || need > MAX_RETAINED_BUFFER) return new byte[need];
        byte[] b = winBufs[slot];
        if (b == null || b.length < need) {
            int grown = (int) Math.min(MAX_RETAINED_BUFFER, Math.max((long) need, b == null ? 0L : (long) b.length + (b.length >> 2)));
            b = new byte[grown];
            winBufs[slot] = b;
        }
        return b;
    }

    private CompressionCodecFactory.BytesInputDecompressor decompressorFor(CompressionCodecName codec) {
        return decompressors.computeIfAbsent(codec, codecs::getDecompressor);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        generation++;
        try { if (in != null) in.close(); } catch (IOException | RuntimeException ignored) { }
        in = null;
        try { codecs.release(); } catch (RuntimeException ignored) { }
        decompressors.clear();
        chunkBuf = EMPTY;
        pageBufs = new byte[0][];
        winBufs = new byte[0][];
    }

    // =====================================================================================================
    // Page store / per-chunk page reader
    // =====================================================================================================

    private final class Store implements PageReadStore {
        private final Map<ColumnPath, ChunkPages> byColumn;
        private final long rowCount;
        private final int gen;

        Store(Map<ColumnPath, ChunkPages> byColumn, long rowCount, int gen) {
            this.byColumn = byColumn;
            this.rowCount = rowCount;
            this.gen = gen;
        }

        @Override
        public PageReader getPageReader(ColumnDescriptor descriptor) {
            if (gen != generation) throw new IllegalStateException("page store used after its row group was released (" + fileName + ")");
            ChunkPages p = byColumn.get(ColumnPath.get(descriptor.getPath()));
            if (p == null) {
                throw new IllegalArgumentException("Could not find page reader for column " + String.join(".", descriptor.getPath()));
            }
            return p;
        }

        @Override
        public long getRowCount() { return rowCount; }

        public void close() { }
    }

    private final class ChunkPages implements PageReader {
        private final int slot;
        private final String column;
        private final int rowGroup;
        private final CompressionCodecName codec;
        private final boolean windowed;   // true: bytes come through a per-column window; false: from the shared chunkBuf
        private final long fileStart;     // file offset of this chunk's first byte
        private final long fileEnd;       // file offset one past its last byte
        private final int begin;          // whole-buffer mode: index of the chunk's first byte in chunkBuf
        private final int window;         // windowed mode: preferred read-ahead, in bytes
        private final long totalValues;
        private final int gen;

        // windowed mode: win[0 .. winValid) holds file bytes [winStart, winStart + winValid)
        private byte[] win;
        private long winStart;
        private int winValid;

        private long pos;                 // file offset of the next unparsed page header
        private long valuesRead;
        private boolean dictChecked;
        private DictionaryPage dictionary;
        private PageHeader pending;       // header parsed by peek() but not yet consumed
        private long payloadPos;          // file offset of the pending header's payload

        ChunkPages(int slot, String column, int rowGroup, CompressionCodecName codec, boolean windowed,
                   long fileStart, long fileEnd, int begin, int window, long totalValues, int gen) {
            this.slot = slot;
            this.column = column;
            this.rowGroup = rowGroup;
            this.codec = codec;
            this.windowed = windowed;
            this.fileStart = fileStart;
            this.fileEnd = fileEnd;
            this.begin = begin;
            this.window = window;
            this.totalValues = totalValues;
            this.gen = gen;
            this.pos = fileStart;
            this.win = windowed && reuse && winBufs[slot] != null ? winBufs[slot] : EMPTY;
        }

        @Override
        public long getTotalValueCount() { return totalValues; }

        @Override
        public DictionaryPage readDictionaryPage() {
            checkAlive();
            if (!dictChecked) {
                dictChecked = true;
                if (pos < fileEnd) {
                    PageHeader h = peek();
                    if (h.getType() == PageType.DICTIONARY_PAGE) {
                        dictionary = buildDictionary(h);
                        consume(h);
                    }
                }
            }
            return dictionary;
        }

        @Override
        public DataPage readPage() {
            checkAlive();
            if (!dictChecked) readDictionaryPage(); // a dictionary page, if any, precedes every data page
            while (true) {
                if (valuesRead >= totalValues) return null;
                if (pos >= fileEnd) {
                    throw corrupt("column chunk ended after " + valuesRead + " of " + totalValues + " values", null);
                }
                PageHeader h = peek();
                switch (h.getType()) {
                    case DATA_PAGE -> {
                        DataPage page = buildV1(h);
                        valuesRead += h.getData_page_header().getNum_values();
                        consume(h);
                        return page;
                    }
                    case DATA_PAGE_V2 -> {
                        DataPage page = buildV2(h);
                        valuesRead += h.getData_page_header_v2().getNum_values();
                        consume(h);
                        return page;
                    }
                    case DICTIONARY_PAGE ->
                        throw corrupt("more than one dictionary page, or a dictionary page after data pages", null);
                    default -> consume(h); // INDEX_PAGE and unknown types: skipped, like parquet-java
                }
            }
        }

        // ---- byte access ---------------------------------------------------------------------------------

        /** The array that {@link #at} indexes into; fetch it AFTER calling {@code at}, which may replace the window. */
        private byte[] arr() { return windowed ? win : chunkBuf; }

        /**
         * Index in {@link #arr()} of file offset {@code off}, with bytes {@code [off, off + len)} guaranteed present.
         * Windowed mode refills the window first when they are not; anything handed out earlier from this column's
         * window is invalid afterwards, which is exactly the "previous page is dead once the next one is requested"
         * contract this class already imposes on the pooled page buffers.
         */
        private int at(long off, int len) {
            if (!windowed) return begin + (int) (off - fileStart);
            if (off < winStart || off + len > winStart + winValid) refill(off, len);
            return (int) (off - winStart);
        }

        private void refill(long off, int len) {
            // Whatever the window already holds from `off` on is the front of what is needed next: keep it.
            int keep = 0;
            if (off >= winStart && off < winStart + winValid) keep = (int) (winStart + winValid - off);
            final long readFrom = off + keep;
            final long remaining = fileEnd - readFrom;
            if ((long) len > keep + remaining) throw corrupt("read of " + len + " bytes at chunk offset " + (off - fileStart) + " runs past the end of the column chunk", null);
            // Resident bytes wanted from `off`: the request, or a full read-ahead window if that is larger; never past the chunk.
            final int target = (int) Math.min(Math.max((long) len, (long) window), keep + remaining);
            byte[] w = win;
            if (w.length < target) {
                byte[] grown = acquireWindow(slot, target);
                if (keep > 0) System.arraycopy(w, (int) (off - winStart), grown, 0, keep);
                w = grown;
                win = w;
            } else if (keep > 0 && off != winStart) {
                System.arraycopy(w, (int) (off - winStart), w, 0, keep); // overlap-safe compaction to the front
            }
            final int toRead = target - keep;
            if (toRead > 0) {
                try {
                    in.seek(readFrom);
                    in.readFully(w, keep, toRead);
                } catch (IOException | RuntimeException e) {
                    winStart = 0;
                    winValid = 0;
                    throw corrupt("cannot read " + toRead + " bytes of the column chunk at file offset " + readFrom, e);
                }
            }
            winStart = off;
            winValid = target;
        }

        // ---- header handling -------------------------------------------------------------------------

        private PageHeader peek() {
            if (pending != null) return pending;
            final long avail = fileEnd - pos;
            int want = (int) Math.min(avail, windowed ? Math.min(HEADER_PROBE, Math.max(window, 8)) : avail);
            PageHeader h;
            int headerLen;
            while (true) {
                int i = at(pos, want);
                byte[] a = arr();
                int limit = windowed ? winValid - i : (int) avail;
                headerIn.reset(a, i, limit);
                try {
                    h = Util.readPageHeader(headerIn);
                    headerLen = headerIn.position() - i;
                    break;
                } catch (IOException | RuntimeException e) {
                    // Windowed: the header may simply extend past what is resident. Retry with a longer prefix until the
                    // whole rest of the chunk (or the header cap) has been offered.
                    if (windowed && winStart + winValid < fileEnd && want < MAX_HEADER_BYTES) {
                        want = (int) Math.min(Math.min(avail, (long) MAX_HEADER_BYTES), Math.max((long) limit * 2, (long) want * 2));
                        continue;
                    }
                    throw corrupt("cannot parse page header at chunk offset " + (pos - fileStart), e);
                }
            }
            int compressed = h.getCompressed_page_size();
            int uncompressed = h.getUncompressed_page_size();
            if (compressed < 0 || uncompressed < 0) {
                throw corrupt("negative page size (compressed=" + compressed + ", uncompressed=" + uncompressed + ")", null);
            }
            if (uncompressed > maxPageBytes) {
                throw corrupt("page declares " + uncompressed + " uncompressed bytes, above the limit " + maxPageBytes, null);
            }
            payloadPos = pos + headerLen;
            if (payloadPos + compressed > fileEnd) {
                throw corrupt("page payload of " + compressed + " bytes runs past the end of its column chunk", null);
            }
            pending = h;
            return pending;
        }

        private void consume(PageHeader h) {
            pos = payloadPos + h.getCompressed_page_size();
            pending = null;
        }

        // ---- page construction -----------------------------------------------------------------------

        private DataPage buildV1(PageHeader h) {
            DataPageHeader dh = h.getData_page_header();
            if (dh == null) throw corrupt("DATA_PAGE without a data page header", null);
            if (dh.getNum_values() < 0) throw corrupt("negative value count " + dh.getNum_values(), null);
            int compressed = h.getCompressed_page_size();
            int uncompressed = h.getUncompressed_page_size();
            int off = at(payloadPos, compressed);
            BytesInput bytes = inflate(arr(), off, compressed, uncompressed, codec != CompressionCodecName.UNCOMPRESSED, true);
            return new DataPageV1(bytes, dh.getNum_values(), uncompressed, null,
                    encoding(dh.getRepetition_level_encoding()), encoding(dh.getDefinition_level_encoding()),
                    encoding(dh.getEncoding()));
        }

        private DataPage buildV2(PageHeader h) {
            DataPageHeaderV2 dh = h.getData_page_header_v2();
            if (dh == null) throw corrupt("DATA_PAGE_V2 without a data page header", null);
            if (dh.getNum_values() < 0 || dh.getNum_rows() < 0 || dh.getNum_nulls() < 0) {
                throw corrupt("negative count in V2 page header", null);
            }
            int compressed = h.getCompressed_page_size();
            int uncompressed = h.getUncompressed_page_size();
            int rl = dh.getRepetition_levels_byte_length();
            int dl = dh.getDefinition_levels_byte_length();
            if (rl < 0 || dl < 0 || (long) rl + dl > compressed || (long) rl + dl > uncompressed) {
                throw corrupt("V2 level sections (" + rl + " + " + dl + " bytes) do not fit the page (compressed="
                        + compressed + ", uncompressed=" + uncompressed + ")", null);
            }
            int base = at(payloadPos, compressed);
            byte[] a = arr();
            // Levels are never compressed in V2; only the values section is, and only when is_compressed (default true).
            BytesInput rlBytes = BytesInput.from(a, base, rl);
            BytesInput dlBytes = BytesInput.from(a, base + rl, dl);
            boolean valuesCompressed = dh.isIs_compressed() && codec != CompressionCodecName.UNCOMPRESSED;
            BytesInput data = inflate(a, base + rl + dl, compressed - rl - dl, uncompressed - rl - dl, valuesCompressed, true);
            return DataPageV2.uncompressed(dh.getNum_rows(), dh.getNum_nulls(), dh.getNum_values(),
                    rlBytes, dlBytes, encoding(dh.getEncoding()), data, null);
        }

        private DictionaryPage buildDictionary(PageHeader h) {
            DictionaryPageHeader dh = h.getDictionary_page_header();
            if (dh == null) throw corrupt("DICTIONARY_PAGE without a dictionary page header", null);
            if (dh.getNum_values() < 0) throw corrupt("negative dictionary size " + dh.getNum_values(), null);
            int compressed = h.getCompressed_page_size();
            int uncompressed = h.getUncompressed_page_size();
            int off = at(payloadPos, compressed);
            byte[] a = arr();
            // Not pooled: a dictionary lives for the whole row group.
            BytesInput bytes = inflate(a, off, compressed, uncompressed, codec != CompressionCodecName.UNCOMPRESSED, false);
            if (windowed && codec == CompressionCodecName.UNCOMPRESSED) {
                // The zero-copy slice would point into a window that is recycled page by page; the dictionary must outlive that.
                bytes = BytesInput.from(Arrays.copyOfRange(a, off, off + compressed));
            }
            return new DictionaryPage(bytes, uncompressed, dh.getNum_values(), encoding(dh.getEncoding()));
        }

        /**
         * The decompressed bytes of {@code src[off .. off + compressedLen)}, as exactly {@code uncompressedLen} bytes.
         * UNCOMPRESSED data is a zero-copy slice of {@code src}; everything else is written to a (pooled, when
         * {@code pooled}) array.
         */
        private BytesInput inflate(byte[] src, int off, int compressedLen, int uncompressedLen, boolean compressed, boolean pooled) {
            if (!compressed) {
                if (compressedLen != uncompressedLen) {
                    throw corrupt("uncompressed page has compressed size " + compressedLen + " != uncompressed size " + uncompressedLen, null);
                }
                return BytesInput.from(src, off, compressedLen);
            }
            if (uncompressedLen == 0) return BytesInput.empty(); // nothing to produce, whatever the (possibly empty) payload
            try {
                if (codec == CompressionCodecName.SNAPPY && javaSnappy) {
                    if (uncompressedLen > SNAPPY_MAX_RATIO * compressedLen + 64) {
                        throw corrupt("page declares " + uncompressedLen + " bytes from only " + compressedLen
                                + " Snappy bytes (beyond the format's maximum expansion)", null);
                    }
                    long declared = SnappyDecompressor.uncompressedLength(src, off, off + compressedLen);
                    if (declared != uncompressedLen) {
                        throw corrupt("Snappy preamble says " + declared + " bytes but the page header says " + uncompressedLen, null);
                    }
                    byte[] dst = pooled ? acquirePageBuf(slot, uncompressedLen) : new byte[uncompressedLen];
                    SnappyDecompressor.decompress(src, off, off + compressedLen, dst, 0, uncompressedLen);
                    return BytesInput.from(dst, 0, uncompressedLen);
                }
                BytesInput out = decompressorFor(codec).decompress(BytesInput.from(src, off, compressedLen), uncompressedLen);
                if (out.size() != uncompressedLen) {
                    throw corrupt(codec + " produced " + out.size() + " bytes but the page header says " + uncompressedLen, null);
                }
                return out;
            } catch (ParquetDecodingException e) {
                throw e; // already attributed above
            } catch (IOException | RuntimeException e) {
                // Includes unchecked failures from third-party decompressors (snappy-java, zstd-jni, ...) on corrupt
                // input: whatever they throw, the caller sees one attributed exception type.
                throw corrupt("cannot decompress " + codec + " page", e);
            }
        }

        private void checkAlive() {
            if (gen != generation) {
                throw new IllegalStateException("page reader for column '" + column + "' used after its row group was released ("
                        + fileName + ", row group " + rowGroup + ")");
            }
        }

        private ParquetDecodingException corrupt(String why, Throwable cause) {
            String msg = "Corrupt Parquet page in " + fileName + " (row group " + rowGroup + ", column '" + column + "'): " + why;
            return cause == null ? new ParquetDecodingException(msg) : new ParquetDecodingException(msg, cause);
        }
    }

    private static Encoding encoding(org.apache.parquet.format.Encoding e) {
        if (e == null) return null;
        return Encoding.valueOf(e.name()); // identical names; this is exactly what ParquetMetadataConverter does
    }

    /** ByteArrayInputStream over a window of a shared array, reusable, with the read position exposed. */
    private static final class PositionedInput extends ByteArrayInputStream {
        PositionedInput() { super(EMPTY); }

        void reset(byte[] b, int off, int len) {
            this.buf = b;
            this.pos = off;
            this.count = off + len;
            this.mark = off;
        }

        int position() { return pos; }
    }
}
