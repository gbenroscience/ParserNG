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
 * <h2>When it steps aside</h2>
 * {@link #load} returns {@code null} -- and the caller reads that row group through parquet-java, which then
 * produces its own diagnostics -- for anything unusual: an encrypted chunk, a projected column missing from the
 * row group, overlapping or out-of-file byte ranges, or a row group needing more than 1 GiB of chunk buffer.
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
    static final int MAX_RETAINED_BUFFER = 64 << 20;
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

    private SeekableInputStream in;
    private long fileLength = -1;
    private byte[] chunkBuf = EMPTY;
    private byte[][] pageBufs = new byte[0][];
    private final PositionedInput headerIn = new PositionedInput();
    private final Map<CompressionCodecName, CompressionCodecFactory.BytesInputDecompressor> decompressors =
            new EnumMap<>(CompressionCodecName.class);

    private int generation;
    private boolean closed;
    private long fallbacks;
    private long rowGroupsLoaded;

    /**
     * @param codecs        supplies decompressors for every codec this class does not decompress itself; released on {@link #close()}
     * @param javaSnappy    decompress SNAPPY with {@link SnappyDecompressor} instead of {@code codecs}
     * @param maxPageBytes  hard cap on any single page's declared uncompressed size (decompression-bomb guard)
     */
    public OwnedRowGroupReader(InputFile input, String fileName, CompressionCodecFactory codecs,
                               boolean javaSnappy, long maxPageBytes) {
        if (input == null) throw new NullPointerException("input");
        if (codecs == null) throw new NullPointerException("codecs");
        this.input = input;
        this.fileName = fileName == null ? String.valueOf(input) : fileName;
        this.codecs = codecs;
        this.javaSnappy = javaSnappy;
        this.maxPageBytes = Math.min(maxPageBytes <= 0 ? Integer.MAX_VALUE : maxPageBytes, Integer.MAX_VALUE - 16L);
        this.reuse = Boolean.parseBoolean(System.getProperty(PROP_REUSE, "true"));
    }

    /** How many row groups were declined (and so must be read by parquet-java). */
    public long fallbackCount() { return fallbacks; }

    /** How many row groups this reader served itself. */
    public long loadedRowGroups() { return rowGroupsLoaded; }

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

        // ---- 3. plan coalesced reads: sort by file offset, merge exactly-adjacent chunks --------------------
        final int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = 1; i < n; i++) { // insertion sort: n is the projected column count
            int v = order[i];
            int j = i - 1;
            while (j >= 0 && start[order[j]] > start[v]) { order[j + 1] = order[j]; j--; }
            order[j + 1] = v;
        }
        final int[] bufOff = new int[n];
        long total = 0;
        final long[] gStart = new long[n];
        final int[] gLen = new int[n];
        final int[] gBuf = new int[n];
        int groups = 0;
        long prevEnd = Long.MIN_VALUE;
        for (int k = 0; k < n; k++) {
            int i = order[k];
            if (size[i] == 0) { bufOff[i] = 0; continue; } // nothing to read; the pager yields no pages
            if (start[i] < prevEnd) return declined();      // overlapping ranges: leave it to parquet-java
            if (groups > 0 && start[i] == prevEnd) {        // adjacent: extend the current group
                bufOff[i] = gBuf[groups - 1] + gLen[groups - 1];
                gLen[groups - 1] += size[i];
            } else {
                gStart[groups] = start[i];
                gLen[groups] = size[i];
                gBuf[groups] = (int) total;
                bufOff[i] = (int) total;
                groups++;
            }
            total += size[i];
            if (total > MAX_CHUNK_BUFFER) return declined();
            prevEnd = start[i] + size[i];
        }

        // ---- 4. read ---------------------------------------------------------------------------------------
        ensureChunkBuf((int) total);
        if (groups > 0 && in == null) in = input.newStream();
        for (int g = 0; g < groups; g++) {
            in.seek(gStart[g]);
            in.readFully(chunkBuf, gBuf[g], gLen[g]);
        }
        if (pageBufs.length < n) pageBufs = Arrays.copyOf(pageBufs, n);

        // ---- 5. hand out lazily-parsed page readers --------------------------------------------------------
        final ChunkPages[] readers = new ChunkPages[n];
        final Map<ColumnPath, ChunkPages> byColumn = new HashMap<>();
        for (int i = 0; i < n; i++) {
            ColumnChunkMetaData c = chunks[i];
            readers[i] = new ChunkPages(i, String.join(".", descs[i].getPath()), rowGroupIndex, c.getCodec(),
                    bufOff[i], bufOff[i] + size[i], c.getValueCount(), generation);
            byColumn.put(c.getPath(), readers[i]);
        }
        rowGroupsLoaded++;
        return new Store(byColumn, block.getRowCount(), generation);
    }

    private PageReadStore declined() {
        fallbacks++;
        return null;
    }

    private void ensureChunkBuf(int needed) {
        if (chunkBuf.length >= needed) return;
        // Geometric growth, but never past what the next row group could plausibly need by more than 25%.
        int grown = (int) Math.min(MAX_CHUNK_BUFFER, Math.max((long) needed, (long) chunkBuf.length + (chunkBuf.length >> 2)));
        chunkBuf = new byte[grown];
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
        private final int begin;          // absolute start of this chunk in chunkBuf
        private final int end;            // absolute end of this chunk in chunkBuf
        private final long totalValues;
        private final int gen;

        private int pos;                  // absolute position in chunkBuf of the next unparsed page header
        private long valuesRead;
        private boolean dictChecked;
        private DictionaryPage dictionary;
        private PageHeader pending;       // header parsed by peek() but not yet consumed
        private int payloadPos;           // absolute position of the pending header's payload

        ChunkPages(int slot, String column, int rowGroup, CompressionCodecName codec,
                   int begin, int end, long totalValues, int gen) {
            this.slot = slot;
            this.column = column;
            this.rowGroup = rowGroup;
            this.codec = codec;
            this.begin = begin;
            this.pos = begin;
            this.end = end;
            this.totalValues = totalValues;
            this.gen = gen;
        }

        @Override
        public long getTotalValueCount() { return totalValues; }

        @Override
        public DictionaryPage readDictionaryPage() {
            checkAlive();
            if (!dictChecked) {
                dictChecked = true;
                if (pos < end) {
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
                if (pos >= end) {
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

        // ---- header handling -------------------------------------------------------------------------

        private PageHeader peek() {
            if (pending == null) {
                headerIn.reset(chunkBuf, pos, end - pos);
                PageHeader h;
                try {
                    h = Util.readPageHeader(headerIn);
                } catch (IOException | RuntimeException e) {
                    throw corrupt("cannot parse page header at chunk offset " + (pos - begin), e);
                }
                int compressed = h.getCompressed_page_size();
                int uncompressed = h.getUncompressed_page_size();
                if (compressed < 0 || uncompressed < 0) {
                    throw corrupt("negative page size (compressed=" + compressed + ", uncompressed=" + uncompressed + ")", null);
                }
                if (uncompressed > maxPageBytes) {
                    throw corrupt("page declares " + uncompressed + " uncompressed bytes, above the limit " + maxPageBytes, null);
                }
                payloadPos = headerIn.position();
                if ((long) payloadPos + compressed > end) {
                    throw corrupt("page payload of " + compressed + " bytes runs past the end of its column chunk", null);
                }
                pending = h;
            }
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
            BytesInput bytes = inflate(payloadPos, compressed, uncompressed, codec != CompressionCodecName.UNCOMPRESSED, true);
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
            // Levels are never compressed in V2; only the values section is, and only when is_compressed (default true).
            BytesInput rlBytes = BytesInput.from(chunkBuf, payloadPos, rl);
            BytesInput dlBytes = BytesInput.from(chunkBuf, payloadPos + rl, dl);
            boolean valuesCompressed = dh.isIs_compressed() && codec != CompressionCodecName.UNCOMPRESSED;
            BytesInput data = inflate(payloadPos + rl + dl, compressed - rl - dl, uncompressed - rl - dl, valuesCompressed, true);
            return DataPageV2.uncompressed(dh.getNum_rows(), dh.getNum_nulls(), dh.getNum_values(),
                    rlBytes, dlBytes, encoding(dh.getEncoding()), data, null);
        }

        private DictionaryPage buildDictionary(PageHeader h) {
            DictionaryPageHeader dh = h.getDictionary_page_header();
            if (dh == null) throw corrupt("DICTIONARY_PAGE without a dictionary page header", null);
            if (dh.getNum_values() < 0) throw corrupt("negative dictionary size " + dh.getNum_values(), null);
            int uncompressed = h.getUncompressed_page_size();
            // Not pooled: a dictionary lives for the whole row group.
            BytesInput bytes = inflate(payloadPos, h.getCompressed_page_size(), uncompressed,
                    codec != CompressionCodecName.UNCOMPRESSED, false);
            return new DictionaryPage(bytes, uncompressed, dh.getNum_values(), encoding(dh.getEncoding()));
        }

        /**
         * The decompressed bytes of {@code [off, off + compressedLen)} of the chunk buffer, as exactly
         * {@code uncompressedLen} bytes. UNCOMPRESSED data is a zero-copy slice; everything else is written to a
         * (pooled, when {@code pooled}) array.
         */
        private BytesInput inflate(int off, int compressedLen, int uncompressedLen, boolean compressed, boolean pooled) {
            if (!compressed) {
                if (compressedLen != uncompressedLen) {
                    throw corrupt("uncompressed page has compressed size " + compressedLen + " != uncompressed size " + uncompressedLen, null);
                }
                return BytesInput.from(chunkBuf, off, compressedLen);
            }
            if (uncompressedLen == 0) return BytesInput.empty(); // nothing to produce, whatever the (possibly empty) payload
            try {
                if (codec == CompressionCodecName.SNAPPY && javaSnappy) {
                    if (uncompressedLen > SNAPPY_MAX_RATIO * compressedLen + 64) {
                        throw corrupt("page declares " + uncompressedLen + " bytes from only " + compressedLen
                                + " Snappy bytes (beyond the format's maximum expansion)", null);
                    }
                    long declared = SnappyDecompressor.uncompressedLength(chunkBuf, off, off + compressedLen);
                    if (declared != uncompressedLen) {
                        throw corrupt("Snappy preamble says " + declared + " bytes but the page header says " + uncompressedLen, null);
                    }
                    byte[] dst = pooled ? acquirePageBuf(slot, uncompressedLen) : new byte[uncompressedLen];
                    SnappyDecompressor.decompress(chunkBuf, off, off + compressedLen, dst, 0, uncompressedLen);
                    return BytesInput.from(dst, 0, uncompressedLen);
                }
                BytesInput out = decompressorFor(codec).decompress(BytesInput.from(chunkBuf, off, compressedLen), uncompressedLen);
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
