package com.github.gbenroscience.parser.ng.parquet.v1.internal.codec;

import java.util.Arrays;


/**
 * Pure-Java, allocation-free (after construction), one-shot Zstandard (RFC 8878) decompressor: the
 * ZSTD counterpart of {@link SnappyDecompressor}, installed by {@link PureJavaZstdCodecFactory}.
 *
 * <p>Derived from the standalone {@code zstdj} decoder (safe engine). Full format support: concatenated and
 * skippable frames, raw/RLE/compressed blocks, raw/RLE/Huffman (1 and 4 stream, treeless) literals, FSE-compressed
 * Huffman weights, predefined/RLE/FSE/repeat sequence tables, repeat-offset history, XXH64 content checksums, and
 * dictionaries. It is exercised against thousands of frames produced by the reference libzstd (all levels and
 * strategies, streamed multi-block frames, checksums, dictionaries) and against corrupted input.
 *
 * <p><b>Deliberately scalar.</b> The standalone decoder can use the Vector API for literal/match copies. That
 * is left out here: this module already runs {@code SimdBulkDecode} on {@code ByteVector.SPECIES_PREFERRED}, and
 * mixing Vector API species in one JVM pollutes the shared type profiles so that C2 boxes vectors (allocation
 * and a large slowdown in BOTH users). Not touching the Vector API at all makes that impossible, at a measured cost
 * of roughly 5-12% of decode speed. The standalone decoder's 2-symbol Huffman, {@code sun.misc.Unsafe} and
 * emulated-prefetch options are also omitted: none measured faster.
 *
 * <p>Single-shot: the whole frame's output must fit in the destination array (it doubles as the match window).
 * Not thread-safe; use one instance per thread (the codec factory is already one-per-reader). Up to 64 bytes past the
 * returned length of {@code dst} may be overwritten, but only inside {@code [dstOff, dstOff + dstCap)}.
 * Offset codes above 30 (windows of 2 GiB or more) are rejected.
 */
public final class ZstdDecompressor {

    /** Corrupt, truncated or unsupported input, or an undersized output buffer. Stackless (cheap to throw). */
    public static final class ZstdException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        public ZstdException(String msg) { super(msg, null, false, false); }
    }

    private static ZstdException err(String m) { return new ZstdException(m); }

    private static final java.lang.invoke.VarHandle LONG = java.lang.invoke.MethodHandles
            .byteArrayViewVarHandle(long[].class, java.nio.ByteOrder.LITTLE_ENDIAN);
    private static final java.lang.invoke.VarHandle INT = java.lang.invoke.MethodHandles
            .byteArrayViewVarHandle(int[].class, java.nio.ByteOrder.LITTLE_ENDIAN);
    private static final java.lang.invoke.VarHandle SHORT = java.lang.invoke.MethodHandles
            .byteArrayViewVarHandle(short[].class, java.nio.ByteOrder.LITTLE_ENDIAN);

    private static long rd64(byte[] a, int i) { return (long) LONG.get(a, i); }
    private static int rd32(byte[] a, int i) { return (int) INT.get(a, i); }
    private static void wr64(byte[] a, int i, long v) { LONG.set(a, i, v); }
    private static void wr16(byte[] a, int i, int v) { SHORT.set(a, i, (short) v); }
    private static void wr8(byte[] a, int i, int v) { a[i] = (byte) v; }
    private static long ldL(long[] a, int i) { return a[i]; }
    private static int ldI(int[] a, int i) { return a[i]; }


    private static final int MAGIC = 0xFD2FB528;
    private static final int DICT_MAGIC = 0xEC30A437;
    private static final int SKIP_MAGIC = 0x184D2A50;
    private static final int MAX_BLOCK = 128 * 1024;
    /** Slack required behind any wild copy. Must be >= largest vector chunk (64 bytes on AVX-512). */
    private static final int WILD = 64;

    // table kinds: index into cur[]/scratch[]
    private static final int LL = 0, OF = 1, ML = 2, GEN = 3;
    private static final int[] MAX_SYM = {35, 30, 52};
    private static final int[] MAX_LOG = {9, 8, 9};

    private static final int[] LL_BASE = new int[36], LL_BITS = new int[36];
    private static final int[] ML_BASE = new int[53], ML_BITS = new int[53];
    private static final long[][] PRE = new long[3][];
    private static final int[] PRE_LOG = {6, 5, 6};

    static {
        int[] llb = {16, 18, 20, 22, 24, 28, 32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        int[] llx = {1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        for (int i = 0; i < 16; i++) { LL_BASE[i] = i; LL_BITS[i] = 0; }
        for (int i = 0; i < 20; i++) { LL_BASE[16 + i] = llb[i]; LL_BITS[16 + i] = llx[i]; }
        int[] mlb = {35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};
        int[] mlx = {1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        for (int i = 0; i < 32; i++) { ML_BASE[i] = i + 3; ML_BITS[i] = 0; }
        for (int i = 0; i < 21; i++) { ML_BASE[32 + i] = mlb[i]; ML_BITS[32 + i] = mlx[i]; }

        short[] llN = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
        short[] mlN = {1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
        short[] ofN = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
        byte[] sym = new byte[512];
        int[] next = new int[256];
        PRE[LL] = new long[64]; PRE[OF] = new long[32]; PRE[ML] = new long[64];
        buildFse(llN, llN.length, 6, PRE[LL], LL, sym, next);
        buildFse(ofN, ofN.length, 5, PRE[OF], OF, sym, next);
        buildFse(mlN, mlN.length, 6, PRE[ML], ML, sym, next);
    }

    /**
     * Copies n bytes, rounding up to a multiple of 16. Needs WILD bytes of slack after both ranges. {@code dist}
     * is the distance between source and destination when they are the same array (must be &ge; 16), or
     * Integer.MAX_VALUE.
     */
    private static void wildCopy(byte[] s, int si, byte[] d, int di, int n, int dist) {
        for (int k = 0; k < n; k += 16) {
            long a = rd64(s, si + k), b = rd64(s, si + k + 8);
            wr64(d, di + k, a);
            wr64(d, di + k + 8, b);
        }
    }

    // ------------------------------------------------------------------ instance state (all preallocated)
    private final byte[] litBuf = new byte[MAX_BLOCK + WILD];
    private final long[][] scratch = {new long[512], new long[512], new long[512]};
    private final long[][] dictTab = {new long[512], new long[512], new long[512]};
    private final int[] dictTabLog = new int[3];
    private final long[][] cur = new long[3][];
    private final int[] curLog = new int[3];
    private final int[] hufScratch = new int[1 << 12], hufDict = new int[1 << 12];
    private int[] hufCur;
    private int hufCurLog, hufDictLog, hufLogTmp;
    private boolean hufDictValid;
    private final short[] norm = new short[256];
    private final byte[] fseSym = new byte[512];
    private final int[] fseNext = new int[256];
    private final long[] wTab = new long[64];
    private final byte[] hw = new byte[264];
    private final int[] rank = new int[16];
    private int nSyms, fseLog;

    private byte[] dict;
    private int dictOff, dictLen, dictId;
    private boolean dictFormatted;
    private int dRep1, dRep2, dRep3;

    private byte[] litArr;
    private int litPos, litEnd;
    private int rep1, rep2, rep3;
    private int framePos;

    private final boolean verifyChecksum;

    /** Verifies the optional XXH64 content checksum (like libzstd does). */
    public ZstdDecompressor() { this(true); }

    public ZstdDecompressor(boolean verifyChecksum) { this.verifyChecksum = verifyChecksum; }

    /**
     * Content size declared by the first frame, or -1 if unknown (streaming writers, e.g. zstd-jni's
     * output stream, do not record it) or not a zstd frame.
     */
    public static long frameContentSize(byte[] src, int off, int len) {
        int end = off + len, p = off;
        if (len < 5) return -1;
        int magic = (src[p] & 0xFF) | (src[p + 1] & 0xFF) << 8 | (src[p + 2] & 0xFF) << 16 | (src[p + 3] & 0xFF) << 24;
        if (magic != MAGIC) return -1;
        p += 4;
        int fhd = src[p++] & 0xFF, flag = fhd >>> 6;
        boolean single = (fhd & 0x20) != 0;
        if (!single) p++;
        int did = fhd & 3;
        p += did == 3 ? 4 : did;
        int n = flag == 0 ? (single ? 1 : 0) : flag == 1 ? 2 : flag == 2 ? 4 : 8;
        if (n == 0 || p + n > end) return -1;
        long v = 0;
        for (int i = 0; i < n; i++) v |= (long) (src[p + i] & 0xFF) << (8 * i);
        return flag == 1 ? v + 256 : v;
    }

    /** Sets (or with null clears) the dictionary used for subsequent frames. May allocate nothing but keeps a reference. */
    public void loadDictionary(byte[] d, int off, int len) {
        dict = null;
        dictFormatted = false;
        hufDictValid = false;
        dictId = 0;
        if (d == null) return;
        if (off < 0 || len < 0 || off + len > d.length) throw new IndexOutOfBoundsException();
        int cOff = off, cLen = len;
        int id = 0;
        boolean fmt = false;
        if (len >= 8 && rd32(d, off) == DICT_MAGIC) {
            id = rd32(d, off + 4);
            int p = off + 8, end = off + len;
            p = readHuffTable(d, p, end, hufDict);
            hufDictLog = hufLogTmp;
            int[] order = {OF, ML, LL};
            for (int k : order) {
                p = readNorm(d, p, end, MAX_SYM[k], MAX_LOG[k]);
                buildFse(norm, nSyms, fseLog, dictTab[k], k, fseSym, fseNext);
                dictTabLog[k] = fseLog;
            }
            if (p + 12 > end) throw err("truncated dictionary");
            dRep1 = rd32(d, p); dRep2 = rd32(d, p + 4); dRep3 = rd32(d, p + 8);
            if (dRep1 <= 0 || dRep2 <= 0 || dRep3 <= 0) throw err("bad dictionary repeat offsets");
            cOff = p + 12;
            cLen = end - cOff;
            fmt = true;
            hufDictValid = true;
        }
        dict = d; dictOff = cOff; dictLen = cLen; dictId = id; dictFormatted = fmt;
    }

    /**
     * Decompresses all concatenated frames in src[srcOff, srcOff+srcLen) into dst[dstOff, dstOff+dstCap).
     * @return number of bytes written
     */
    public int decompress(byte[] src, int srcOff, int srcLen, byte[] dst, int dstOff, int dstCap) {
        if (srcOff < 0 || srcLen < 0 || srcOff > src.length - srcLen
                || dstOff < 0 || dstCap < 0 || dstOff > dst.length - dstCap) throw new IndexOutOfBoundsException();
        int pos = srcOff, end = srcOff + srcLen, op = dstOff, dstEnd = dstOff + dstCap;
        while (pos < end) {
            if (end - pos < 4) throw err("truncated frame magic");
            int magic = rd32(src, pos);
            if ((magic & 0xFFFFFFF0) == SKIP_MAGIC) {
                if (end - pos < 8) throw err("truncated skippable frame");
                long sz = rd32(src, pos + 4) & 0xFFFFFFFFL;
                if (sz > end - pos - 8) throw err("truncated skippable frame");
                pos += 8 + (int) sz;
                continue;
            }
            if (magic != MAGIC) throw err("bad magic number");
            op = decodeFrame(src, pos + 4, end, dst, op, dstEnd);
            pos = framePos;
        }
        return op - dstOff;
    }

    // ------------------------------------------------------------------ frame
    private int decodeFrame(byte[] src, int p, int end, byte[] dst, int op, int dstEnd) {
        if (p >= end) throw err("truncated frame header");
        int fhd = src[p++] & 0xFF;
        int fcsFlag = fhd >>> 6;
        boolean single = (fhd & 0x20) != 0;
        if ((fhd & 0x08) != 0) throw err("reserved bit set");
        boolean checksum = (fhd & 0x04) != 0;
        int didFlag = fhd & 3;
        if (!single) {
            if (p >= end) throw err("truncated frame header");
            p++; // window descriptor: not needed, the whole output is the window
        }
        int didLen = didFlag == 3 ? 4 : didFlag;
        if (p + didLen > end) throw err("truncated frame header");
        int fid = 0;
        for (int i = 0; i < didLen; i++) fid |= (src[p + i] & 0xFF) << (8 * i);
        p += didLen;
        int fcsLen = fcsFlag == 0 ? (single ? 1 : 0) : fcsFlag == 1 ? 2 : fcsFlag == 2 ? 4 : 8;
        long fcs = -1;
        if (fcsLen > 0) {
            if (p + fcsLen > end) throw err("truncated frame header");
            long v = 0;
            for (int i = 0; i < fcsLen; i++) v |= (long) (src[p + i] & 0xFF) << (8 * i);
            fcs = fcsFlag == 1 ? v + 256 : v;
            p += fcsLen;
            if (fcs < 0 || fcs > dstEnd - op) throw err("destination buffer too small");
        }
        if (fid != 0 && (dict == null || dictId != fid)) throw err("dictionary id mismatch");

        // frame-local entropy state
        final int frameStart = op;
        hufCur = null;
        cur[0] = cur[1] = cur[2] = null;
        rep1 = 1; rep2 = 4; rep3 = 8;
        if (dict != null && dictFormatted) {
            hufCur = hufDict; hufCurLog = hufDictLog;
            for (int k = 0; k < 3; k++) { cur[k] = dictTab[k]; curLog[k] = dictTabLog[k]; }
            rep1 = dRep1; rep2 = dRep2; rep3 = dRep3;
        }

        boolean last;
        do {
            if (end - p < 3) throw err("truncated block header");
            int h = (src[p] & 0xFF) | ((src[p + 1] & 0xFF) << 8) | ((src[p + 2] & 0xFF) << 16);
            p += 3;
            last = (h & 1) != 0;
            int type = (h >>> 1) & 3;
            int size = h >>> 3;
            switch (type) {
                case 0:
                    if (size > MAX_BLOCK || size > end - p) throw err("bad raw block");
                    if (size > dstEnd - op) throw err("destination buffer too small");
                    System.arraycopy(src, p, dst, op, size);
                    p += size; op += size;
                    break;
                case 1:
                    if (size > MAX_BLOCK || p >= end) throw err("bad RLE block");
                    if (size > dstEnd - op) throw err("destination buffer too small");
                    Arrays.fill(dst, op, op + size, src[p]);
                    p++; op += size;
                    break;
                case 2:
                    if (size > MAX_BLOCK || size > end - p) throw err("bad compressed block");
                    op = decodeBlock(src, p, p + size, dst, op, dstEnd, frameStart);
                    p += size;
                    break;
                default:
                    throw err("reserved block type");
            }
        } while (!last);

        if (fcs >= 0 && op - frameStart != fcs) throw err("content size mismatch");
        if (checksum) {
            if (end - p < 4) throw err("truncated checksum");
            if (verifyChecksum && (int) xxh64(dst, frameStart, op - frameStart) != rd32(src, p))
                throw err("content checksum mismatch");
            p += 4;
        }
        framePos = p;
        return op;
    }

    // ------------------------------------------------------------------ compressed block
    private int decodeBlock(byte[] src, int p, int end, byte[] dst, int op, int dstEnd, int frameStart) {
        p = decodeLiterals(src, p, end);
        if (p >= end) throw err("truncated sequences header");
        int b = src[p++] & 0xFF;
        int nbSeq;
        if (b == 0) nbSeq = 0;
        else if (b < 128) nbSeq = b;
        else if (b < 255) {
            if (p >= end) throw err("truncated sequences header");
            nbSeq = ((b - 128) << 8) + (src[p++] & 0xFF);
        } else {
            if (p + 2 > end) throw err("truncated sequences header");
            nbSeq = (src[p] & 0xFF) + ((src[p + 1] & 0xFF) << 8) + 0x7F00;
            p += 2;
        }
        if (nbSeq == 0) {
            if (p != end) throw err("trailing data after empty sequence section");
            int n = litEnd - litPos;
            if (n > dstEnd - op) throw err("destination buffer too small");
            System.arraycopy(litArr, litPos, dst, op, n);
            return op + n;
        }
        if (p >= end) throw err("truncated sequences header");
        int modes = src[p++] & 0xFF;
        if ((modes & 3) != 0) throw err("reserved sequence mode bits");
        p = readSeqTable(LL, modes >>> 6, src, p, end);
        p = readSeqTable(OF, (modes >>> 4) & 3, src, p, end);
        p = readSeqTable(ML, (modes >>> 2) & 3, src, p, end);
        if (p >= end) throw err("missing sequence bitstream");
        return execSequences(src, p, end, nbSeq, dst, op, dstEnd, frameStart);
    }

    private int readSeqTable(int k, int mode, byte[] src, int p, int end) {
        switch (mode) {
            case 0:
                cur[k] = PRE[k]; curLog[k] = PRE_LOG[k];
                return p;
            case 1: {
                if (p >= end) throw err("truncated RLE table");
                int s = src[p] & 0xFF;
                if (s > MAX_SYM[k]) throw err("bad RLE symbol");
                scratch[k][0] = entry(k, s, 0, 0);
                cur[k] = scratch[k]; curLog[k] = 0;
                return p + 1;
            }
            case 2: {
                int q = readNorm(src, p, end, MAX_SYM[k], MAX_LOG[k]);
                buildFse(norm, nSyms, fseLog, scratch[k], k, fseSym, fseNext);
                cur[k] = scratch[k]; curLog[k] = fseLog;
                return q;
            }
            default:
                if (cur[k] == null) throw err("repeat table without previous table");
                return p;
        }
    }

    // ------------------------------------------------------------------ FSE
    /** Table entry: bits 0-15 newState base, 16-23 nbBits, 24-31 extra bits, 32-63 baseline (or symbol for GEN). */
    private static long entry(int k, int s, int ns, int nb) {
        long base;
        int extra;
        if (k == LL) { base = LL_BASE[s]; extra = LL_BITS[s]; }
        else if (k == OF) { base = 1L << s; extra = s; }
        else if (k == ML) { base = ML_BASE[s]; extra = ML_BITS[s]; }
        else { base = s; extra = 0; }
        return (long) ns | ((long) nb << 16) | ((long) extra << 24) | (base << 32);
    }

    private static void buildFse(short[] norm, int nSyms, int al, long[] out, int k, byte[] sym, int[] next) {
        int size = 1 << al, mask = size - 1, high = size - 1;
        for (int s = 0; s < nSyms; s++) {
            if (norm[s] == -1) { sym[high--] = (byte) s; next[s] = 1; }
            else next[s] = norm[s];
        }
        int step = (size >> 1) + (size >> 3) + 3, pos = 0;
        for (int s = 0; s < nSyms; s++) {
            int n = norm[s];
            for (int i = 0; i < n; i++) {
                sym[pos] = (byte) s;
                do { pos = (pos + step) & mask; } while (pos > high);
            }
        }
        if (pos != 0) throw err("corrupt FSE table");
        for (int u = 0; u < size; u++) {
            int s = sym[u] & 0xFF;
            int ns = next[s]++;
            int nb = al - (31 - Integer.numberOfLeadingZeros(ns));
            out[u] = entry(k, s, (ns << nb) - size, nb);
        }
    }

    private static int peekBits(byte[] s, int base, int end, int bp, int n) {
        int b = base + (bp >>> 3), v = 0;
        for (int k = 0; k < 4; k++) {
            int i = b + k;
            if (i < end) v |= (s[i] & 0xFF) << (8 * k);
        }
        return (v >>> (bp & 7)) & ((1 << n) - 1);
    }

    /** Reads an FSE normalized counter header; fills norm[], nSyms, fseLog; returns the position after it. */
    private int readNorm(byte[] src, int p, int end, int maxSym, int maxLog) {
        if (p >= end) throw err("truncated FSE header");
        int al = peekBits(src, p, end, 0, 4) + 5;
        if (al > maxLog) throw err("FSE accuracy log too large");
        int bp = 4;
        int remaining = (1 << al) + 1, threshold = 1 << al, nbBits = al + 1;
        int sym = 0;
        boolean prev0 = false;
        while (remaining > 1 && sym <= maxSym) {
            if (prev0) {
                int n0 = sym, r;
                do {
                    r = peekBits(src, p, end, bp, 2);
                    bp += 2;
                    n0 += r;
                } while (r == 3 && bp <= (end - p) * 8);
                if (n0 > maxSym) throw err("FSE symbol overflow");
                while (sym < n0) norm[sym++] = 0;
            }
            int max = (2 * threshold - 1) - remaining;
            int v = peekBits(src, p, end, bp, nbBits), count;
            if ((v & (threshold - 1)) < max) {
                count = v & (threshold - 1);
                bp += nbBits - 1;
            } else {
                count = v & (2 * threshold - 1);
                if (count >= threshold) count -= max;
                bp += nbBits;
            }
            count--;
            remaining -= count < 0 ? -count : count;
            norm[sym++] = (short) count;
            prev0 = count == 0;
            while (remaining < threshold) { nbBits--; threshold >>= 1; }
            if (bp > (end - p) * 8 + 0) throw err("truncated FSE header");
        }
        if (remaining != 1) throw err("corrupt FSE counts");
        if (bp > (end - p) * 8) throw err("truncated FSE header");
        nSyms = sym;
        fseLog = al;
        return p + ((bp + 7) >>> 3);
    }

    // ------------------------------------------------------------------ backward bit-stream init
    private static long initCt(byte[] s, int start, int len) {
        if (len >= 8) return rd64(s, start + len - 8);
        long c = 0;
        for (int i = 0; i < len; i++) c |= (long) (s[start + i] & 0xFF) << (8 * i);
        return c;
    }

    private static int initBc(byte[] s, int start, int len) {
        int last = s[start + len - 1] & 0xFF;
        if (last == 0) throw err("bitstream end mark missing");
        int bc = 8 - (31 - Integer.numberOfLeadingZeros(last));
        return len >= 8 ? bc : bc + (8 - len) * 8;
    }

    // ------------------------------------------------------------------ Huffman
    private int readHuffTable(byte[] src, int p, int end, int[] tab) {
        if (p >= end) throw err("truncated Huffman header");
        int hdr = src[p++] & 0xFF;
        int n;
        if (hdr >= 128) {
            n = hdr - 127;
            int nb = (n + 1) >> 1;
            if (p + nb > end) throw err("truncated Huffman weights");
            for (int i = 0; i < n; i += 2) {
                int b = src[p + (i >> 1)] & 0xFF;
                hw[i] = (byte) (b >> 4);
                if (i + 1 < n) hw[i + 1] = (byte) (b & 15);
            }
            p += nb;
        } else {
            if (hdr == 0 || hdr > end - p) throw err("bad Huffman header");
            n = fseWeights(src, p, hdr);
            p += hdr;
        }
        long sum = 0;
        for (int i = 0; i < n; i++) {
            int w = hw[i];
            if (w > 12) throw err("Huffman weight too large");
            if (w > 0) sum += 1L << (w - 1);
        }
        if (sum == 0) throw err("empty Huffman tree");
        int log = 64 - Long.numberOfLeadingZeros(sum);
        if (log > 12) throw err("Huffman tree too deep");
        long rest = (1L << log) - sum;
        if (rest <= 0 || (rest & (rest - 1)) != 0) throw err("bad Huffman weights");
        hw[n++] = (byte) (64 - Long.numberOfLeadingZeros(rest));
        Arrays.fill(rank, 0);
        for (int i = 0; i < n; i++) rank[hw[i]]++;
        int next = 0;
        for (int w = 1; w <= log; w++) {
            int c = rank[w];
            rank[w] = next;
            next += c << (w - 1);
        }
        for (int s = 0; s < n; s++) {
            int w = hw[s];
            if (w != 0) {
                int len = 1 << (w - 1), st = rank[w];
                Arrays.fill(tab, st, st + len, s | ((log + 1 - w) << 8));
                rank[w] = st + len;
            }
        }
        hufLogTmp = log;
        return p;
    }

    /** Decodes FSE-compressed Huffman weights (two interleaved states) into hw[]; returns count. */
    private int fseWeights(byte[] src, int p, int size) {
        int end = p + size;
        int q = readNorm(src, p, end, 12, 6);
        buildFse(norm, nSyms, fseLog, wTab, GEN, fseSym, fseNext);
        final int al = fseLog, len = end - q;
        if (len <= 0) throw err("missing Huffman weight stream");
        int ptr = len >= 8 ? q + len - 8 : q;
        long ct = initCt(src, q, len);
        int bc = initBc(src, q, len);
        int s1 = (int) (((ct << bc) >>> 1) >>> (63 - al)); bc += al;
        int s2 = (int) (((ct << bc) >>> 1) >>> (63 - al)); bc += al;
        int n = 0;
        while (true) {
            int nb = bc >>> 3;
            if (nb != 0) {
                if (ptr >= q + nb) { ptr -= nb; bc &= 7; ct = rd64(src, ptr); }
                else if (ptr > q) { nb = ptr - q; ptr = q; bc -= nb << 3; ct = rd64(src, ptr); }
            }
            if (n >= 255) throw err("too many Huffman weights");
            long e = wTab[s1];
            hw[n++] = (byte) (e >>> 32);
            nb = (int) (e >>> 16) & 0xFF;
            s1 = (int) (e & 0xFFFF) + (int) (((ct << bc) >>> 1) >>> (63 - nb)); bc += nb;
            if (bc > 64) { hw[n++] = (byte) (wTab[s2] >>> 32); break; }

            nb = bc >>> 3;
            if (nb != 0) {
                if (ptr >= q + nb) { ptr -= nb; bc &= 7; ct = rd64(src, ptr); }
                else if (ptr > q) { nb = ptr - q; ptr = q; bc -= nb << 3; ct = rd64(src, ptr); }
            }
            e = wTab[s2];
            hw[n++] = (byte) (e >>> 32);
            nb = (int) (e >>> 16) & 0xFF;
            s2 = (int) (e & 0xFFFF) + (int) (((ct << bc) >>> 1) >>> (63 - nb)); bc += nb;
            if (bc > 64) { hw[n++] = (byte) (wTab[s1] >>> 32); break; }
        }
        return n;
    }

    private int decodeLiterals(byte[] src, int p, int end) {
        if (p >= end) throw err("truncated literals header");
        int b0 = src[p] & 0xFF;
        int type = b0 & 3, fmt = (b0 >> 2) & 3;
        int regen, csize = 0, hdr, streams = 1;
        if (type < 2) {
            switch (fmt) {
                case 1:
                    if (p + 2 > end) throw err("truncated literals header");
                    regen = (b0 >> 4) | ((src[p + 1] & 0xFF) << 4); hdr = 2;
                    break;
                case 3:
                    if (p + 3 > end) throw err("truncated literals header");
                    regen = (b0 >> 4) | ((src[p + 1] & 0xFF) << 4) | ((src[p + 2] & 0xFF) << 12); hdr = 3;
                    break;
                default:
                    regen = b0 >> 3; hdr = 1;
            }
            p += hdr;
            if (regen > MAX_BLOCK) throw err("literals too large");
            if (type == 0) {
                if (regen > end - p) throw err("truncated raw literals");
                litArr = src; litPos = p; litEnd = p + regen;
                return p + regen;
            }
            if (p >= end) throw err("truncated RLE literals");
            Arrays.fill(litBuf, 0, regen, src[p]);
            litArr = litBuf; litPos = 0; litEnd = regen;
            return p + 1;
        }
        switch (fmt) {
            case 0: case 1: {
                if (p + 3 > end) throw err("truncated literals header");
                int v = (b0 | ((src[p + 1] & 0xFF) << 8) | ((src[p + 2] & 0xFF) << 16));
                regen = (v >>> 4) & 0x3FF; csize = (v >>> 14) & 0x3FF; hdr = 3;
                streams = fmt == 0 ? 1 : 4;
                break;
            }
            case 2: {
                if (p + 4 > end) throw err("truncated literals header");
                int v = rd32(src, p);
                regen = (v >>> 4) & 0x3FFF; csize = (v >>> 18) & 0x3FFF; hdr = 4; streams = 4;
                break;
            }
            default: {
                if (p + 5 > end) throw err("truncated literals header");
                long v = (rd32(src, p) & 0xFFFFFFFFL) | ((long) (src[p + 4] & 0xFF) << 32);
                regen = (int) ((v >>> 4) & 0x3FFFF); csize = (int) ((v >>> 22) & 0x3FFFF); hdr = 5; streams = 4;
            }
        }
        p += hdr;
        if (regen > MAX_BLOCK || csize > end - p) throw err("bad compressed literals sizes");
        int cend = p + csize;
        if (type == 2) {
            p = readHuffTable(src, p, cend, hufScratch);
            hufCur = hufScratch; hufCurLog = hufLogTmp;
        } else if (hufCur == null) {
            throw err("treeless literals without previous Huffman table");
        }
        if (streams == 1) {
            int len = cend - p;
            if (len < 1) throw err("empty Huffman stream");
            int ptr = len >= 8 ? p + len - 8 : p;
            huffTail(src, p, ptr, initCt(src, p, len), initBc(src, p, len), litBuf, 0, regen, hufCur, hufCurLog);
        } else {
            decodeHuf4(src, p, cend - p, regen, hufCur, hufCurLog);
        }
        litArr = litBuf; litPos = 0; litEnd = regen;
        return cend;
    }

    /** One-symbol-at-a-time Huffman decoding of the remainder of a backward stream; verifies exact consumption. */
    private static void huffTail(byte[] src, int start, int ptr, long ct, int bc,
                                 byte[] out, int o, int count, int[] tab, int log) {
        final int sh = 64 - log;
        for (int i = 0; i < count; i++) {
            int nb = bc >>> 3;
            if (nb != 0) {
                if (ptr >= start + nb) { ptr -= nb; bc &= 7; ct = rd64(src, ptr); }
                else if (ptr > start) { nb = ptr - start; ptr = start; bc -= nb << 3; ct = rd64(src, ptr); }
            }
            int e = ldI(tab, (int) ((ct << bc) >>> sh));
            wr8(out, o + i, e);
            bc += e >>> 8;
        }
        int nb = bc >>> 3;
        if (nb != 0) {
            if (ptr >= start + nb) { ptr -= nb; bc &= 7; }
            else if (ptr > start) { nb = ptr - start; ptr = start; bc -= nb << 3; }
        }
        if (ptr != start || bc != 64) throw err("corrupt Huffman stream");
    }

    private void decodeHuf4(byte[] src, int p, int csize, int regen, int[] tab, int log) {
        if (csize < 10 || regen < 6) throw err("bad 4-stream literals");
        final int c1 = (src[p] & 0xFF) | ((src[p + 1] & 0xFF) << 8);
        final int c2 = (src[p + 2] & 0xFF) | ((src[p + 3] & 0xFF) << 8);
        final int c3 = (src[p + 4] & 0xFF) | ((src[p + 5] & 0xFF) << 8);
        final int st1 = p + 6, st2 = st1 + c1, st3 = st2 + c2, st4 = st3 + c3, end = p + csize;
        final int c4 = end - st4;
        if (c1 == 0 || c2 == 0 || c3 == 0 || c4 <= 0) throw err("bad 4-stream sizes");
        final int seg = (regen + 3) >> 2, n4 = regen - 3 * seg;
        final byte[] out = litBuf;
        final int sh = 64 - log;
        int o1 = 0, o2 = seg, o3 = 2 * seg, o4 = 3 * seg;
        final int e1 = seg, e2 = 2 * seg, e3 = 3 * seg, e4 = regen;
        int pt1 = c1 >= 8 ? st1 + c1 - 8 : st1, pt2 = c2 >= 8 ? st2 + c2 - 8 : st2;
        int pt3 = c3 >= 8 ? st3 + c3 - 8 : st3, pt4 = c4 >= 8 ? st4 + c4 - 8 : st4;
        long ct1 = initCt(src, st1, c1), ct2 = initCt(src, st2, c2), ct3 = initCt(src, st3, c3), ct4 = initCt(src, st4, c4);
        int bc1 = initBc(src, st1, c1), bc2 = initBc(src, st2, c2), bc3 = initBc(src, st3, c3), bc4 = initBc(src, st4, c4);
        int nb, e;
        for (int it = n4 >> 2; it > 0; it--) {
            nb = bc1 >>> 3;
            if (nb != 0) {
                if (pt1 >= st1 + nb) { pt1 -= nb; bc1 &= 7; ct1 = rd64(src, pt1); }
                else if (pt1 > st1) { nb = pt1 - st1; pt1 = st1; bc1 -= nb << 3; ct1 = rd64(src, pt1); }
            }
            nb = bc2 >>> 3;
            if (nb != 0) {
                if (pt2 >= st2 + nb) { pt2 -= nb; bc2 &= 7; ct2 = rd64(src, pt2); }
                else if (pt2 > st2) { nb = pt2 - st2; pt2 = st2; bc2 -= nb << 3; ct2 = rd64(src, pt2); }
            }
            nb = bc3 >>> 3;
            if (nb != 0) {
                if (pt3 >= st3 + nb) { pt3 -= nb; bc3 &= 7; ct3 = rd64(src, pt3); }
                else if (pt3 > st3) { nb = pt3 - st3; pt3 = st3; bc3 -= nb << 3; ct3 = rd64(src, pt3); }
            }
            nb = bc4 >>> 3;
            if (nb != 0) {
                if (pt4 >= st4 + nb) { pt4 -= nb; bc4 &= 7; ct4 = rd64(src, pt4); }
                else if (pt4 > st4) { nb = pt4 - st4; pt4 = st4; bc4 -= nb << 3; ct4 = rd64(src, pt4); }
            }
            for (int r = 0; r < 4; r++) {
                e = ldI(tab, (int) ((ct1 << bc1) >>> sh)); wr8(out, o1 + r, e); bc1 += e >>> 8;
                e = ldI(tab, (int) ((ct2 << bc2) >>> sh)); wr8(out, o2 + r, e); bc2 += e >>> 8;
                e = ldI(tab, (int) ((ct3 << bc3) >>> sh)); wr8(out, o3 + r, e); bc3 += e >>> 8;
                e = ldI(tab, (int) ((ct4 << bc4) >>> sh)); wr8(out, o4 + r, e); bc4 += e >>> 8;
            }
            o1 += 4; o2 += 4; o3 += 4; o4 += 4;
        }
    
        huffTail(src, st1, pt1, ct1, bc1, out, o1, e1 - o1, tab, log);
        huffTail(src, st2, pt2, ct2, bc2, out, o2, e2 - o2, tab, log);
        huffTail(src, st3, pt3, ct3, bc3, out, o3, e3 - o3, tab, log);
        huffTail(src, st4, pt4, ct4, bc4, out, o4, e4 - o4, tab, log);
    }

    /**
     * Match copy for offset &ge; 16 that always moves the first 32 bytes without testing the length (most matches
     * are shorter), so the only data-dependent branch is the rare "longer than 32" one.
     */
    private static void wildCopyMatch(byte[] d, int m, int o, int n, int dist) {
        wr64(d, o, rd64(d, m)); wr64(d, o + 8, rd64(d, m + 8));
        wr64(d, o + 16, rd64(d, m + 16)); wr64(d, o + 24, rd64(d, m + 24));
        for (int k = 32; k < n; k += 16) {
            long a = rd64(d, m + k), b = rd64(d, m + k + 8);
            wr64(d, o + k, a);
            wr64(d, o + k + 8, b);
        }
    }

    // ------------------------------------------------------------------ sequences (decode + execute fused)
    private int execSequences(byte[] src, int bs, int be, int nbSeq, byte[] dst, int op, int dstEnd, int frameStart) {
        final long[] llT = cur[LL], ofT = cur[OF], mlT = cur[ML];
        final int len = be - bs;
        int ptr = len >= 8 ? bs + len - 8 : bs;
        long ct = initCt(src, bs, len);
        int bc = initBc(src, bs, len);
        int llS = (int) (((ct << bc) >>> 1) >>> (63 - curLog[LL])); bc += curLog[LL];
        int ofS = (int) (((ct << bc) >>> 1) >>> (63 - curLog[OF])); bc += curLog[OF];
        int mlS = (int) (((ct << bc) >>> 1) >>> (63 - curLog[ML])); bc += curLog[ML];

        int r1 = rep1, r2 = rep2, r3 = rep3;
        final byte[] lit = litArr;
        int lp = litPos;
        final int le = litEnd, litCap = lit.length;
        final int dstSafe = dstEnd - WILD, litSafe = Math.min(le, litCap - WILD);

        for (int i = nbSeq; i > 0; i--) {
            final long eo = ldL(ofT, ofS), em = ldL(mlT, mlS), el = ldL(llT, llS);
            final int ofB = (int) (eo >>> 24) & 0xFF, mlB = (int) (em >>> 24) & 0xFF, llB = (int) (el >>> 24) & 0xFF;
            final int nbL = (int) (el >>> 16) & 0xFF, nbM = (int) (em >>> 16) & 0xFF, nbO = (int) (eo >>> 16) & 0xFF;
            final int total = ofB + mlB + llB + nbL + nbM + nbO;
            int nb;
            // Refill only when this sequence's bits don't fit in the 64-bit reservoir (about every 2nd sequence).
            if (bc + total > 64) {
                nb = bc >>> 3;
                if (ptr >= bs + nb) { ptr -= nb; bc &= 7; ct = rd64(src, ptr); }
                else if (ptr > bs) { nb = ptr - bs; ptr = bs; bc -= nb << 3; ct = rd64(src, ptr); }
            }
            final int ov = (int) (eo >>> 32) + (int) (((ct << bc) >>> 1) >>> (63 - ofB)); bc += ofB;
            final int ml = (int) (em >>> 32) + (int) (((ct << bc) >>> 1) >>> (63 - mlB)); bc += mlB;
            if (total > 57) { // rare (huge offsets + long lengths): a second refill is needed mid-sequence
                nb = bc >>> 3;
                if (ptr >= bs + nb) { ptr -= nb; bc &= 7; ct = rd64(src, ptr); }
                else if (ptr > bs) { nb = ptr - bs; ptr = bs; bc -= nb << 3; ct = rd64(src, ptr); }
            }
            final int ll = (int) (el >>> 32) + (int) (((ct << bc) >>> 1) >>> (63 - llB)); bc += llB;

            if (i != 1) {
                llS = (int) (el & 0xFFFF) + (int) (((ct << bc) >>> 1) >>> (63 - nbL)); bc += nbL;
                mlS = (int) (em & 0xFFFF) + (int) (((ct << bc) >>> 1) >>> (63 - nbM)); bc += nbM;
                ofS = (int) (eo & 0xFFFF) + (int) (((ct << bc) >>> 1) >>> (63 - nbO)); bc += nbO;
            }

            // --- resolve offset / update repeat history
            int offset;
            if (ov > 3) {
                offset = ov - 3;
                r3 = r2; r2 = r1; r1 = offset;
            } else {
                int idx = ov - 1 + (ll == 0 ? 1 : 0);
                if (idx == 0) {
                    offset = r1;
                } else {
                    offset = idx == 1 ? r2 : idx == 2 ? r3 : r1 - 1;
                    if (offset <= 0) throw err("zero offset");
                    if (idx != 1) r3 = r2;
                    r2 = r1; r1 = offset;
                }
            }

            // --- execute
            final int opLit = op + ll, opEnd = opLit + ml;
            if (lp + ll <= litSafe && opEnd <= dstSafe && offset <= opLit - frameStart) {
                wildCopy(lit, lp, dst, op, ll, Integer.MAX_VALUE);
                final int m = opLit - offset;
                if (offset >= 16) {
                    wildCopyMatch(dst, m, opLit, ml, offset);
                } else {
                    shortOffsetCopy(dst, m, opLit, opEnd, offset);
                }
            } else {
                if (ll > le - lp || opEnd > dstEnd) throw err("sequence overruns buffer");
                System.arraycopy(lit, lp, dst, op, ll);
                matchSlow(dst, opLit, opEnd, offset, frameStart);
            }
            op = opEnd;
            lp += ll;
        }

        // end of bitstream must be exactly consumed
        int nb = bc >>> 3;
        if (nb != 0) {
            if (ptr >= bs + nb) { ptr -= nb; bc &= 7; }
            else if (ptr > bs) { nb = ptr - bs; ptr = bs; bc -= nb << 3; }
        }
        if (ptr != bs || bc != 64) throw err("corrupt sequence bitstream");

        rep1 = r1; rep2 = r2; rep3 = r3;
        int rem = le - lp;
        if (rem > dstEnd - op) throw err("destination buffer too small");
        System.arraycopy(lit, lp, dst, op, rem);
        return op + rem;
    }

    /** Overlapping match with offset in [1,15]; wild (may write up to 15 bytes past opEnd). */
    private static void shortOffsetCopy(byte[] d, int m, int o, int opEnd, int offset) {
        if (offset >= 8) {
            do { wr64(d, o, rd64(d, m)); m += 8; o += 8; } while (o < opEnd);
        } else if (offset == 1) {
            if (opEnd - o > 256) {
                Arrays.fill(d, o, opEnd, d[m]);
            } else {
                long pat = (d[m] & 0xFFL) * 0x0101010101010101L;
                do { wr64(d, o, pat); o += 8; } while (o < opEnd);
            }
        } else if (opEnd - o <= 16) {
            do { d[o] = d[m]; o++; m++; } while (o < opEnd);
        } else {
            int dd = offset;
            while (o < opEnd) {
                int n = Math.min(dd, opEnd - o);
                System.arraycopy(d, o - dd, d, o, n);
                o += n;
                dd <<= 1;
            }
        }
    }

    /** Exact (non-wild) match copy; also handles matches that start inside the dictionary. */
    private void matchSlow(byte[] dst, int op, int opEnd, int offset, int frameStart) {
        int hist = op - frameStart;
        if (offset > hist) {
            int dd = offset - hist;
            if (dict == null || dd > dictLen) throw err("offset beyond available history");
            int n = Math.min(dd, opEnd - op);
            System.arraycopy(dict, dictOff + dictLen - dd, dst, op, n);
            op += n;
            if (op == opEnd) return;
        }
        int m = op - offset, ml = opEnd - op;
        if (offset >= ml) {
            System.arraycopy(dst, m, dst, op, ml);
        } else {
            for (int i = 0; i < ml; i++) dst[op + i] = dst[m + i];
        }
    }

    // ------------------------------------------------------------------ XXH64
    private static final long P1 = 0x9E3779B185EBCA87L, P2 = 0xC2B2AE3D27D4EB4FL, P3 = 0x165667B19E3779F9L,
            P4 = 0x85EBCA77C2B2AE63L, P5 = 0x27D4EB2F165667C5L;

    private static long xround(long acc, long in) { return Long.rotateLeft(acc + in * P2, 31) * P1; }

    private static long xmerge(long h, long v) { return (h ^ xround(0, v)) * P1 + P4; }

    static long xxh64(byte[] a, int off, int len) {
        final int end = off + len;
        long h;
        if (len >= 32) {
            long v1 = P1 + P2, v2 = P2, v3 = 0, v4 = -P1;
            final int limit = end - 32;
            do {
                v1 = xround(v1, rd64(a, off));
                v2 = xround(v2, rd64(a, off + 8));
                v3 = xround(v3, rd64(a, off + 16));
                v4 = xround(v4, rd64(a, off + 24));
                off += 32;
            } while (off <= limit);
            h = Long.rotateLeft(v1, 1) + Long.rotateLeft(v2, 7) + Long.rotateLeft(v3, 12) + Long.rotateLeft(v4, 18);
            h = xmerge(h, v1); h = xmerge(h, v2); h = xmerge(h, v3); h = xmerge(h, v4);
        } else {
            h = P5;
        }
        h += len;
        while (off + 8 <= end) {
            h ^= xround(0, rd64(a, off));
            h = Long.rotateLeft(h, 27) * P1 + P4;
            off += 8;
        }
        if (off + 4 <= end) {
            h ^= (rd32(a, off) & 0xFFFFFFFFL) * P1;
            h = Long.rotateLeft(h, 23) * P2 + P3;
            off += 4;
        }
        while (off < end) {
            h ^= (a[off] & 0xFFL) * P5;
            h = Long.rotateLeft(h, 11) * P1;
            off++;
        }
        h ^= h >>> 33; h *= P2; h ^= h >>> 29; h *= P3; h ^= h >>> 32;
        return h;
    }
}
