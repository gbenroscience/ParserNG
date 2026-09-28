package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScanException;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.column.Encoding;
import org.apache.parquet.column.page.*;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.PrimitiveType;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Native column decode engine used in place of {@code org.apache.parquet.column.impl.ColumnReaderImpl}:
 * same pull contract, decoded via direct page-byte access instead of {@code ColumnReaderImpl}'s
 * converter-bound dispatch chain.
 *
 * <h2>This class genuinely implements {@link ColumnReader}</h2>
 * An earlier revision of this class only matched {@code ColumnReader}'s method signatures without
 * declaring {@code implements ColumnReader} — which meant it could not actually be assigned anywhere
 * {@code RowGroupDecoder}/{@code ColumnPlan} declare a {@code ColumnReader} (an interface, not a
 * shape Java accepts structurally), making the "drop-in replacement" claim inaccurate for every
 * column, flat or nested. This revision declares the interface for real. {@link #writeCurrentValueToConverter()}
 * is a deliberate no-op: this cursor exists specifically so callers read values through the typed
 * getters below instead of through parquet-mr's record-assembly {@code Converter} API, so that method
 * is never expected to be invoked by anything in this module; if it ever is, that is a sign this
 * cursor was handed to code written for the record-assembly path instead. {@link #getBinary()}
 * allocates a wrapper on demand (only if a caller actually uses this interface-mandated accessor) —
 * everything in this module uses the zero-copy {@link #binData()}/{@link #binOffset()}/
 * {@link #binLength()} triple instead and never pays for it.
 *
 * <p><b>Unverified against the real interface, highest remaining risk in this class:</b> the exact
 * complete method set of {@code org.apache.parquet.column.ColumnReader} was not available to check
 * against a compiler in the environment this was written in. The methods implemented below
 * ({@code getTotalValueCount}, {@code consume}, {@code getCurrentRepetitionLevel},
 * {@code getCurrentDefinitionLevel}, {@code getDescriptor}, {@code writeCurrentValueToConverter},
 * {@code getCurrentValueDictionaryID}, {@code getBinary}, {@code getBoolean}, {@code getDouble},
 * {@code getFloat}, {@code getInteger}, {@code getLong}, {@code skip}) are a best-confidence
 * reconstruction of that interface. {@code getDescriptor()} was the one member of that set actually
 * missing as of the previous revision of this class — a genuine omission, not a guess that turned out
 * wrong: the constructor already received the exact {@link ColumnDescriptor} to return, it simply
 * hadn't been retained in a field. Fixed by storing it and adding the accessor; every other method
 * below was cross-checked against this fix and needed no further change. Two specific, mechanical
 * first-build fixes still to expect for the remaining, lower-confidence methods: (1) if
 * {@code mvn compile} reports a missing abstract method, add it following the pattern of the methods
 * already here (return the already-decoded scalar for the current entry); (2) if it reports
 * {@code method does not override or implement a method from a supertype} for
 * {@code getCurrentValueDictionaryID} or {@code skip} specifically (the two methods here with the
 * least confidence behind them), delete the {@code @Override} annotation from that one method —
 * nothing in this module calls either method today, so a wrong guess there is a compile-time
 * annotation fix, not a behavioral one.
 *
 * <h2>No decompressor: pages are asserted to already be decompressed, never guessed at</h2>
 * An earlier revision accepted a {@code CompressionCodecFactory.BytesInputDecompressor} and decided
 * whether to decompress a {@code DataPageV1} by comparing {@code getBytes().size()} against
 * {@code getUncompressedSize()} — a heuristic that could, in principle, guess wrong and silently
 * decode garbage rather than fail. This revision removes that guess entirely, on this reasoning:
 * {@code ColumnReaderImpl} — the exact class this engine replaces, used unmodified by every prior
 * revision of this module — never decompresses anything itself either, and reads pages directly off
 * whatever {@link PageReader} it was given. The only reason that has always worked is that a
 * {@link PageReadStore} obtained from {@code ParquetFileReader.readNextRowGroup()}/
 * {@code readNextFilteredRowGroup()} — the exact, only call path this module uses to obtain one —
 * already hands back decompressed page bytes; decompression happens once, earlier, inside
 * {@code ParquetFileReader}'s own row-group-reading pipeline. Given that, a decompressor parameter
 * here was solving a problem that (for this module's actual call path) does not exist, while adding
 * a real one: guessing wrong is a silent-corruption risk, and sourcing a
 * {@code CompressionCodecFactory} correctly was never actually wired up by any prior revision. This
 * revision instead <b>asserts</b> the expected already-decompressed state and throws a specific,
 * named {@link ParquetScanException} if that assertion is violated — DataPageV1 via
 * {@code raw.length == page.getUncompressedSize()} (V1 has no explicit compressed/uncompressed flag
 * to check instead), DataPageV2 via its own explicit {@code isCompressed()} flag. Either this
 * assumption holds (matching every prior revision's implicit behavior) or a scan fails loudly with a
 * clear diagnostic naming the file and column — never a silent wrong guess. <b>This assumption has
 * not been tested against a real file compressed with SNAPPY/GZIP/ZSTD</b> — doing so, specifically,
 * is the single most valuable test to run before trusting this class on compressed production data.
 *
 * <p>Unsupported and rejected by name rather than silently mis-decoded: {@code DELTA_BINARY_PACKED},
 * {@code DELTA_LENGTH_BYTE_ARRAY}, {@code DELTA_BYTE_ARRAY}, {@code BYTE_STREAM_SPLIT},
 * {@code INT96}/{@code FIXED_LEN_BYTE_ARRAY} physical types.
 *
 * <p>Not thread-safe; one instance per column per row group per {@code RowGroupDecoder}
 * (constructed fresh in {@code begin()} for every row group, matching {@code ColumnReaderImpl}'s own
 * per-row-group lifecycle — this class is not reused across row groups the way the top-level Arrow
 * vectors are).
 *
 * <h2>Nested (struct/list/map) columns</h2>
 * This class has no special-casing for {@code maxRepetitionLevel > 0} or {@code maxDefinitionLevel > 1}
 * — {@link #getCurrentRepetitionLevel()}/{@link #getCurrentDefinitionLevel()} report whatever the
 * page's own decoded rep/def streams say, generically, via {@link RleBitPackingDecoder}, for any
 * column shape. {@code RowGroupDecoder}'s nested-column path ({@code readLeaf}, driving
 * {@code internal.NodePlan}/{@code internal.LevelWalker}) only ever calls the same
 * {@code ColumnReader} method set a flat column's decode uses, so — on the reasoning that this class
 * genuinely satisfies that interface as of this revision — it should require no changes to serve a
 * nested leaf. That inference has <b>not been exercised end to end</b>: no test in this module
 * constructs a {@code ColumnDescriptor} with {@code maxRepetitionLevel > 0} and drives it through
 * this specific class (as opposed to through {@code ColumnReaderImpl}, which every nested-type test
 * so far has actually exercised).
 *
 * <h2>Dense value indexing (nullable and nested columns)</h2>
 * A Parquet data page stores levels for every <em>entry</em> but values (PLAIN values or dictionary
 * indices) only for the <em>present</em> entries, i.e. those with {@code definition level == maxDef}. Every
 * value array here ({@code idxBuf}, the typed scratch arrays) is therefore <b>dense</b>: it holds exactly
 * {@code presentCount} elements, and {@link #denseBefore} tracks how many present entries precede the
 * current one. Indexing a value array by entry position (as an earlier revision did for dictionary
 * indices) is only right when the page has no nulls.
 *
 * <h2>Bulk paths</h2>
 * For fixed-width numeric columns ({@code INT32/INT64/FLOAT/DOUBLE}) every PLAIN page is decoded up
 * front with the SIMD kernels in {@link SimdBulkDecode} -- for required, nullable and nested leaves alike,
 * since only the present values are stored and the kernels are told exactly how many there are.
 * Flat (non-repeated) numeric columns can additionally be drained a page-segment at a time through
 * {@link #beginBulk}/{@code bulkXxx()}/{@link #endBulk()}, which hands the caller the decoded page arrays
 * (or, for dictionary pages, one tight typed gather) plus the definition levels, so the caller writes its
 * output in one loop with no per-value {@code consume()}/getter/{@code switch} dispatch.
 */
public final class FastColumnCursor implements ColumnReader {

    private final ColumnDescriptor descriptor;
    private final PageReader pageReader;
    private final int maxDef, maxRep;
    private final PrimitiveType.PrimitiveTypeName physical;
    private final Path file;
    private final String columnName;
    private final long totalValueCount;
    private final boolean bulkEligible; // physical is a fixed-width numeric type (PLAIN pages are SIMD-decoded up front)

    private DictionaryCache dictionary;
    private long consumedCount = 0;
    /** Decompressed bytes of every dictionary/data page this cursor has loaded so far (see {@link #pageBytesLoaded()}). */
    private long pageBytesLoaded = 0;

    private int[] pageDef = new int[0];
    private int[] pageRep = new int[0];
    private int pageCount = 0;
    private int pagePos = 0;
    /** Present entries (def == maxDef) strictly before {@link #pagePos} in the current page = index of the current entry in the dense value arrays. */
    private int denseBefore = 0;

    private boolean pageIsDictionary;
    private boolean bulkPlain; // true for the current page: values already sit in one of the scratch arrays below
    private int[] idxBuf = new int[0];
    private int[] intScratch = new int[0];
    private long[] longScratch = new long[0];
    private float[] floatScratch = new float[0];
    private double[] doubleScratch = new double[0];
    private final ByteReader valueBytes = new ByteReader();

    // bulk-segment state (see beginBulk/endBulk)
    private int segEntries, segPresent, segBase;
    private int[] gatherInts = new int[0];
    private long[] gatherLongs = new long[0];
    private float[] gatherFloats = new float[0];
    private double[] gatherDoubles = new double[0];
    private int boolBitPos;

    // decoded state for the CURRENT entry, valid only when its definition level == maxDef
    private int curInt;
    private long curLong;
    private float curFloat;
    private double curDouble;
    private boolean curBool;
    private byte[] curBinData;
    private int curBinOff, curBinLen;

    public FastColumnCursor(ColumnDescriptor descriptor, PageReadStore pages, Path file, String columnName) {
        this.descriptor = descriptor;
        this.maxDef = descriptor.getMaxDefinitionLevel();
        this.maxRep = descriptor.getMaxRepetitionLevel();
        this.physical = descriptor.getPrimitiveType().getPrimitiveTypeName();
        this.bulkEligible = switch (physical) {
            case INT32, INT64, FLOAT, DOUBLE -> true;
            default -> false;
        };
        this.file = file;
        this.columnName = columnName;
        this.pageReader = pages.getPageReader(descriptor);
        this.totalValueCount = pageReader.getTotalValueCount();

        DictionaryPage dictPage = pageReader.readDictionaryPage();
        if (dictPage != null) {
            dictionary = new DictionaryCache(dictPage, physical, file, columnName);
            pageBytesLoaded += dictPage.getUncompressedSize();
        }
        if (totalValueCount > 0) {
            loadNextNonEmptyPage();
            decodeCurrentValue();
        }
    }

    @Override public ColumnDescriptor getDescriptor() { return descriptor; }

    /**
     * Total decompressed payload bytes of the dictionary page and every data page loaded so far (data pages
     * include their repetition/definition level streams). Monotonic; pages are loaded lazily, so this is only
     * complete once the column chunk has been fully consumed. Plain field, not atomic: same single-thread
     * ownership as the rest of this class. Feeds {@code ScanMetrics#uncompressedBytesDecoded()}.
     */
    public long pageBytesLoaded() { return pageBytesLoaded; }

    @Override public long getTotalValueCount() { return totalValueCount; }
    @Override public int getCurrentDefinitionLevel() { return pageDef[pagePos]; }
    @Override public int getCurrentRepetitionLevel() { return pageRep[pagePos]; }

    @Override
    public void consume() {
        consumedCount++;
        if (pageDef[pagePos] == maxDef) denseBefore++; // pageDef is filled with maxDef (== 0) for required columns
        pagePos++;
        if (consumedCount >= totalValueCount) return; // exhausted: caller must not read further, matches ColumnReader
        if (pagePos >= pageCount) {
            loadNextNonEmptyPage();
            pagePos = 0;
        }
        decodeCurrentValue();
    }

    @Override public int getInteger() { return curInt; }
    @Override public long getLong() { return curLong; }
    @Override public float getFloat() { return curFloat; }
    @Override public double getDouble() { return curDouble; }
    @Override public boolean getBoolean() { return curBool; }

    @Override
    public Binary getBinary() {
        // On-demand allocation: this is the one interface-mandated accessor that must produce an
        // object. Everything in this module reads binData()/binOffset()/binLength() directly and
        // never calls this, so the allocation this implies is paid only if something outside this
        // module's own decode path actually calls it.
        return Binary.fromConstantByteArray(curBinData, curBinOff, curBinLen);
    }

    /** Least-confidence override in this class — see the class Javadoc's "first-build fixes" note. */
    @Override
    public int getCurrentValueDictionaryID() {
        return (pageIsDictionary && pageDef[pagePos] == maxDef) ? idxBuf[denseBefore] : -1;
    }

    /** Least-confidence override in this class — see the class Javadoc's "first-build fixes" note. */
    @Override
    public void skip() {
        // This cursor always decodes the current value eagerly as part of loading/advancing (see
        // decodeCurrentValue()), so there is no cheaper "advance without decoding" path to take;
        // skip() and consume() are equivalent here.
        consume();
    }

    @Override
    public void writeCurrentValueToConverter() {
        // Deliberate no-op -- see class Javadoc.
    }

    /** Valid only immediately after a present ({@code def == maxDef}) BINARY/UTF8 entry. Zero-copy: do not retain past the next {@link #consume()}. */
    public byte[] binData() { return curBinData; }
    public int binOffset() { return curBinOff; }
    public int binLength() { return curBinLen; }

    // ---------------- page loading ----------------

    private void loadNextNonEmptyPage() {
        DataPage page;
        do {
            page = pageReader.readPage();
            if (page == null) {
                throw new ParquetScanException("Column chunk ended before all " + totalValueCount + " values were read (read "
                        + consumedCount + ")", file, -1, columnName, null);
            }
            if (page instanceof DataPageV1 v1) decodeV1(v1);
            else if (page instanceof DataPageV2 v2) decodeV2(v2);
            else throw new ParquetScanException("Unknown DataPage implementation " + page.getClass(), file, -1, columnName, null);
        } while (pageCount == 0);
    }

    /** Arrays only ever need to hold exactly {@code count} — {@link RleBitPackingDecoder#decode} never writes past that regardless of run sizes in the stream. */
    private void ensureLevelCapacity(int count) {
        if (pageDef.length < count) pageDef = new int[count];
        if (pageRep.length < count) pageRep = new int[count];
    }

    private static int bitWidthFor(int maxLevel) {
        if (maxLevel == 0) return 0;
        return 32 - Integer.numberOfLeadingZeros(maxLevel);
    }

    private void decodeV1(DataPageV1 page) {
        byte[] raw;
        try {
            raw = page.getBytes().toByteArray();
        } catch (IOException e) {
            throw new ParquetScanException("Failed to read page bytes", file, -1, columnName, e);
        }
        if (raw.length != page.getUncompressedSize()) {
            throw new ParquetScanException("Page declares " + raw.length + " bytes but an uncompressed size of "
                    + page.getUncompressedSize() + " -- this decode engine expects PageReadStore obtained from "
                    + "ParquetFileReader.readNextRowGroup()/readNextFilteredRowGroup(), which decompresses eagerly; "
                    + "if this page is reaching this class still compressed, it is being used against a different "
                    + "PageReadStore source than it was designed for", file, -1, columnName, null);
        }
        pageBytesLoaded += raw.length;
        int count = page.getValueCount();
        pageCount = count;
        ensureLevelCapacity(count);
        ByteReader br = new ByteReader();
        br.wrap(raw, 0, raw.length);

        if (maxRep > 0) {
            int len = br.readIntLE();
            ByteReader sub = new ByteReader();
            sub.wrap(raw, br.pos, len);
            RleBitPackingDecoder.decode(sub, bitWidthFor(maxRep), pageRep, count);
            br.pos += len;
        } else {
            java.util.Arrays.fill(pageRep, 0, count, 0);
        }
        if (maxDef > 0) {
            int len = br.readIntLE();
            ByteReader sub = new ByteReader();
            sub.wrap(raw, br.pos, len);
            RleBitPackingDecoder.decode(sub, bitWidthFor(maxDef), pageDef, count);
            br.pos += len;
        } else {
            java.util.Arrays.fill(pageDef, 0, count, maxDef);
        }
        setUpValueDecode(br, raw.length - br.pos, page.getValueEncoding(), count);
    }

    private void decodeV2(DataPageV2 page) {
        int count = page.getValueCount();
        pageCount = count;
        ensureLevelCapacity(count);
        try {
            if (maxRep > 0) {
                byte[] repBytes = page.getRepetitionLevels().toByteArray();
                ByteReader sub = new ByteReader();
                sub.wrap(repBytes, 0, repBytes.length);
                RleBitPackingDecoder.decode(sub, bitWidthFor(maxRep), pageRep, count);
            } else {
                java.util.Arrays.fill(pageRep, 0, count, 0);
            }
            if (maxDef > 0) {
                byte[] defBytes = page.getDefinitionLevels().toByteArray();
                ByteReader sub = new ByteReader();
                sub.wrap(defBytes, 0, defBytes.length);
                RleBitPackingDecoder.decode(sub, bitWidthFor(maxDef), pageDef, count);
            } else {
                java.util.Arrays.fill(pageDef, 0, count, maxDef);
            }
            if (page.isCompressed()) {
                throw new ParquetScanException("Data Page V2 reports isCompressed()=true but this decode engine has "
                        + "no decompressor by design (see class Javadoc); pages from ParquetFileReader.readNextRowGroup()/"
                        + "readNextFilteredRowGroup() are expected to already be decompressed", file, -1, columnName, null);
            }
            byte[] raw = page.getData().toByteArray();
            pageBytesLoaded += page.getRepetitionLevels().size() + page.getDefinitionLevels().size() + raw.length;
            ByteReader br = new ByteReader();
            br.wrap(raw, 0, raw.length);
            setUpValueDecode(br, raw.length, page.getDataEncoding(), count);
        } catch (IOException e) {
            throw new ParquetScanException("Failed to decode Data Page V2", file, -1, columnName, e);
        }
    }

    /** Number of entries in {@code pageDef[0, count)} that carry a value (definition level == maxDef). */
    private int countPresent(int count) {
        if (maxDef == 0) return count;
        final int[] d = pageDef;
        final int m = maxDef;
        int present = 0;
        for (int i = 0; i < count; i++) if (d[i] == m) present++;
        return present;
    }

    private ParquetScanException corruptPage(String why, Throwable cause) {
        return new ParquetScanException(why, file, -1, columnName, cause);
    }

    /**
     * Prepares value decoding for a freshly loaded page. Value bytes and dictionary indices exist only for
     * the <em>present</em> entries, so everything below is sized and bounded by {@code present}, never by
     * {@code count} (the entry count): decoding {@code count} indices from a stream that holds
     * {@code present} of them would run off the end of any page that contains a null.
     */
    private void setUpValueDecode(ByteReader br, int dataLen, Encoding enc, int count) {
        boolean dict = enc == Encoding.PLAIN_DICTIONARY || enc == Encoding.RLE_DICTIONARY;
        if (!dict && enc != Encoding.PLAIN) {
            throw new ParquetScanException("Unsupported value encoding " + enc
                    + " (only PLAIN and dictionary are decoded natively; DELTA*/BYTE_STREAM_SPLIT are not)",
                    file, -1, columnName, null);
        }
        pageIsDictionary = dict;
        boolBitPos = 0;
        bulkPlain = false;
        denseBefore = 0;
        final int present = countPresent(count);
        if (dict) {
            if (dictionary == null) {
                throw corruptPage("Dictionary-encoded data page but the column chunk has no dictionary page", null);
            }
            if (present == 0) return; // an all-null page may omit even the bit-width byte
            int bitWidth = br.readByte() & 0xFF; // 1-byte bit width prefix precedes the RLE index stream
            if (idxBuf.length < present) idxBuf = new int[present];
            ByteReader idxReader = new ByteReader();
            idxReader.wrap(br.buf, br.pos, dataLen - 1);
            RleBitPackingDecoder.decode(idxReader, bitWidth, idxBuf, present);
            // Dictionary gather is intentionally NOT vectorized -- see SimdBulkDecode's class doc: measured
            // 4.3x-4.5x SLOWER with IntVector.fromArray's indexed gather on the development host. The scalar
            // gather (per value in decodeCurrentValue, or one tight typed loop in beginBulk) is the measured choice.
        } else if (bulkEligible) {
            bulkPlain = true;
            final int width = (physical == PrimitiveType.PrimitiveTypeName.INT32 || physical == PrimitiveType.PrimitiveTypeName.FLOAT) ? 4 : 8;
            if ((long) present * width > dataLen) {
                throw corruptPage("PLAIN page holds " + dataLen + " value bytes but its " + present
                        + " present entries need " + ((long) present * width), null);
            }
            switch (physical) {
                case INT32 -> {
                    if (intScratch.length < present) intScratch = new int[present];
                    SimdBulkDecode.readIntsLE(br.buf, br.pos, intScratch, 0, present);
                }
                case INT64 -> {
                    if (longScratch.length < present) longScratch = new long[present];
                    SimdBulkDecode.readLongsLE(br.buf, br.pos, longScratch, 0, present);
                }
                case FLOAT -> {
                    if (floatScratch.length < present) floatScratch = new float[present];
                    SimdBulkDecode.readFloatsLE(br.buf, br.pos, floatScratch, 0, present);
                }
                case DOUBLE -> {
                    if (doubleScratch.length < present) doubleScratch = new double[present];
                    SimdBulkDecode.readDoublesLE(br.buf, br.pos, doubleScratch, 0, present);
                }
                default -> throw new IllegalStateException("bulkEligible implies a fixed-width numeric physical type: " + physical);
            }
        } else {
            valueBytes.wrap(br.buf, br.pos, dataLen);
        }
    }

    // ---------------- bulk segments (flat numeric columns) ----------------

    /** True if this column can be drained with {@link #beginBulk}: non-repeated and a fixed-width numeric physical type. */
    public boolean supportsBulk() { return maxRep == 0 && bulkEligible; }

    /**
     * Starts a bulk segment of up to {@code max} entries, never crossing a page boundary; returns the
     * segment's entry count {@code m >= 1}. Until {@link #endBulk()}, the caller reads:
     * <ul>
     *   <li>the values via the accessor for this column's physical type ({@link #bulkInts()}, {@link #bulkLongs()},
     *       {@link #bulkFloats()}, {@link #bulkDoubles()}): the segment's <em>present</em> values are
     *       {@code arr[bulkBase() .. bulkBase() + bulkPresent())}, in entry order;</li>
     *   <li>for nullable columns, the entries' definition levels {@code bulkDefs()[bulkDefBase() + e]},
     *       {@code e in [0, m)}: entry {@code e} has a value iff its level equals {@code maxDef}, and takes the
     *       next unread present value. For required columns every entry is present and no levels need reading.</li>
     * </ul>
     * Returned arrays are the cursor's own scratch: read-only, valid only until {@link #endBulk()}.
     */
    public int beginBulk(int max) {
        if (!supportsBulk()) throw new IllegalStateException("column '" + columnName + "' does not support bulk reads");
        if (max <= 0) throw new IllegalArgumentException("max must be positive: " + max);
        if (consumedCount >= totalValueCount) {
            throw corruptPage("Bulk read past the end of the column chunk (" + totalValueCount + " values)", null);
        }
        final int m = Math.min(max, pageCount - pagePos);
        int present = m;
        if (maxDef > 0) {
            present = 0;
            final int[] d = pageDef;
            final int end = pagePos + m;
            for (int i = pagePos; i < end; i++) if (d[i] == maxDef) present++;
        }
        segEntries = m;
        segPresent = present;
        if (pageIsDictionary) {
            gatherSegment(denseBefore, present);
            segBase = 0;
        } else {
            segBase = denseBefore;
        }
        return m;
    }

    private void gatherSegment(int from, int n) {
        if (n == 0) return;
        try {
            switch (physical) {
                case INT32 -> { if (gatherInts.length < n) gatherInts = new int[n]; dictionary.gatherInts(idxBuf, from, gatherInts, n); }
                case INT64 -> { if (gatherLongs.length < n) gatherLongs = new long[n]; dictionary.gatherLongs(idxBuf, from, gatherLongs, n); }
                case FLOAT -> { if (gatherFloats.length < n) gatherFloats = new float[n]; dictionary.gatherFloats(idxBuf, from, gatherFloats, n); }
                case DOUBLE -> { if (gatherDoubles.length < n) gatherDoubles = new double[n]; dictionary.gatherDoubles(idxBuf, from, gatherDoubles, n); }
                default -> throw new IllegalStateException("bulk gather implies a fixed-width numeric physical type: " + physical);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw corruptPage("Dictionary index out of range (dictionary has " + dictionary.size() + " entries)", e);
        }
    }

    public int[] bulkInts() { return pageIsDictionary ? gatherInts : intScratch; }
    public long[] bulkLongs() { return pageIsDictionary ? gatherLongs : longScratch; }
    public float[] bulkFloats() { return pageIsDictionary ? gatherFloats : floatScratch; }
    public double[] bulkDoubles() { return pageIsDictionary ? gatherDoubles : doubleScratch; }
    /** Index in the {@code bulkXxx()} array of the segment's first present value. */
    public int bulkBase() { return segBase; }
    /** Number of present (non-null) values in the segment. */
    public int bulkPresent() { return segPresent; }
    public int[] bulkDefs() { return pageDef; }
    /** Index in {@link #bulkDefs()} of the segment's first entry. */
    public int bulkDefBase() { return pagePos; }

    /** Consumes the segment opened by {@link #beginBulk}, leaving the cursor on the first entry after it (loading the next page if needed). */
    public void endBulk() {
        consumedCount += segEntries;
        pagePos += segEntries;
        denseBefore += segPresent;
        segEntries = 0;
        segPresent = 0;
        if (consumedCount >= totalValueCount) return; // exhausted, same as consume()
        if (pagePos >= pageCount) {
            loadNextNonEmptyPage();
            pagePos = 0;
        }
        decodeCurrentValue();
    }

    // ---------------- current-value decode ----------------

    private void decodeCurrentValue() {
        if (pageDef[pagePos] != maxDef) return; // null entry: nothing was written to the value stream (pageDef == maxDef everywhere when maxDef == 0)
        if (bulkPlain) {
            final int d = denseBefore;
            switch (physical) {
                case INT32 -> curInt = intScratch[d];
                case INT64 -> curLong = longScratch[d];
                case FLOAT -> curFloat = floatScratch[d];
                case DOUBLE -> curDouble = doubleScratch[d];
                default -> throw new IllegalStateException("bulkPlain implies a fixed-width numeric physical type: " + physical);
            }
        } else if (pageIsDictionary) {
            gatherDictionary(idxBuf[denseBefore]);
        } else {
            readPlain();
        }
    }

    private void gatherDictionary(int idx) {
        if (idx < 0 || idx >= dictionary.size()) {
            throw corruptPage("Dictionary index " + idx + " out of range (dictionary has " + dictionary.size() + " entries)", null);
        }
        switch (physical) {
            case INT32 -> curInt = dictionary.getInt(idx);
            case INT64 -> curLong = dictionary.getLong(idx);
            case FLOAT -> curFloat = dictionary.getFloat(idx);
            case DOUBLE -> curDouble = dictionary.getDouble(idx);
            case BINARY -> { curBinData = dictionary.binData(); curBinOff = dictionary.binOffset(idx); curBinLen = dictionary.binLength(idx); }
            default -> throw new ParquetScanException("Dictionary encoding not supported for " + physical, file, -1, columnName, null);
        }
    }

    private void readPlain() {
        switch (physical) {
            case INT32 -> curInt = valueBytes.readIntLE();
            case INT64 -> curLong = valueBytes.readLongLE();
            case FLOAT -> curFloat = valueBytes.readFloatLE();
            case DOUBLE -> curDouble = valueBytes.readDoubleLE();
            case BINARY -> {
                int len = valueBytes.readIntLE();
                curBinData = valueBytes.buf;
                curBinOff = valueBytes.pos;
                curBinLen = len;
                valueBytes.pos += len;
            }
            case BOOLEAN -> {
                int byteIdx = valueBytes.pos + (boolBitPos >>> 3);
                int bit = (valueBytes.buf[byteIdx] >>> (boolBitPos & 7)) & 1;
                curBool = bit != 0;
                boolBitPos++;
            }
            default -> throw new ParquetScanException(
                    "Physical type " + physical + " is not supported by the native decode engine yet (INT96/FIXED_LEN_BYTE_ARRAY)",
                    file, -1, columnName, null);
        }
    }
}