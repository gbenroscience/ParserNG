package com.github.gbenroscience.parser.ng.parquet.v1.internal.codec;

import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.compression.CompressionCodecFactory.BytesInputCompressor;
import org.apache.parquet.compression.CompressionCodecFactory.BytesInputDecompressor;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.ZstdDecompressorTest.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link PureJavaZstdCodecFactory}: the parquet-java-facing wrapper. The decoder's own correctness is covered by
 * {@link ZstdDecompressorTest}; what is tested here is the contract with parquet-java -- both {@code decompress}
 * overloads (including buffer position/limit conventions), every kind of input buffer, the error mapping, and
 * above all that the delegate is NEVER asked for a ZSTD decompressor (that would load {@code zstd-jni}, the
 * native dependency this option exists to avoid). Nothing here needs a native zstd library.
 */
class PureJavaZstdCodecFactoryTest {

    /** Records what the factory asks of its delegate. */
    private static final class RecordingDelegate implements CompressionCodecFactory {
        final List<String> calls = new ArrayList<>();
        final BytesInputDecompressor marker = new BytesInputDecompressor() {
            @Override public BytesInput decompress(BytesInput b, int n) { throw new AssertionError("delegate decoded"); }
            @Override public void decompress(ByteBuffer i, int c, ByteBuffer o, int n) { throw new AssertionError("delegate decoded"); }
            @Override public void release() { }
        };
        @Override public BytesInputCompressor getCompressor(CompressionCodecName c) { calls.add("compressor:" + c); return null; }
        @Override public BytesInputDecompressor getDecompressor(CompressionCodecName c) { calls.add("decompressor:" + c); return marker; }
        @Override public void release() { calls.add("release"); }
    }

    private static final byte[] TEXT = textLike(3000, 7);

    private static BytesInputDecompressor zstd(RecordingDelegate d) {
        return new PureJavaZstdCodecFactory(d).getDecompressor(CompressionCodecName.ZSTD);
    }

    private static ByteBuffer direct(byte[] b) {
        ByteBuffer d = ByteBuffer.allocateDirect(b.length);
        d.put(b).flip();
        return d;
    }

    // ------------------------------------------------------------------ delegation

    @Test
    void zstdNeverTouchesTheDelegate() {
        RecordingDelegate d = new RecordingDelegate();
        PureJavaZstdCodecFactory f = new PureJavaZstdCodecFactory(d);
        BytesInputDecompressor a = f.getDecompressor(CompressionCodecName.ZSTD);
        BytesInputDecompressor b = f.getDecompressor(CompressionCodecName.ZSTD);
        assertSame(a, b, "one decoder per factory, created lazily");
        assertEquals(List.of(), d.calls, "asking the delegate for ZSTD would load zstd-jni");
    }

    @Test
    void everyOtherCodecAndAllCompressionAreForwarded() {
        RecordingDelegate d = new RecordingDelegate();
        PureJavaZstdCodecFactory f = new PureJavaZstdCodecFactory(d);
        assertSame(d.marker, f.getDecompressor(CompressionCodecName.SNAPPY));
        assertSame(d.marker, f.getDecompressor(CompressionCodecName.GZIP));
        assertSame(d.marker, f.getDecompressor(CompressionCodecName.UNCOMPRESSED));
        f.getCompressor(CompressionCodecName.ZSTD);   // reading is replaced, writing is not
        f.getCompressor(CompressionCodecName.SNAPPY);
        assertEquals(List.of("decompressor:SNAPPY", "decompressor:GZIP", "decompressor:UNCOMPRESSED",
                "compressor:ZSTD", "compressor:SNAPPY"), d.calls);
    }

    @Test
    void releaseIsForwardedAndTheFactoryStaysUsable() throws IOException {
        RecordingDelegate d = new RecordingDelegate();
        PureJavaZstdCodecFactory f = new PureJavaZstdCodecFactory(d);
        BytesInputDecompressor first = f.getDecompressor(CompressionCodecName.ZSTD);
        f.release();
        assertEquals(List.of("release"), d.calls);
        BytesInputDecompressor second = f.getDecompressor(CompressionCodecName.ZSTD);
        assertArrayEquals(TEXT, second.decompress(BytesInput.from(hex(TEXT_FRAME)), 3000).toByteArray());
        assertNotSame(first, second, "release() drops the decoder; a later request builds a fresh one");
    }

    @Test
    void stacksOverAnotherDecoratorWithoutEitherLosingItsCodec() {
        RecordingDelegate base = new RecordingDelegate();
        // another decorator in the middle that serves SNAPPY itself, as PanamaSnappyCodecFactory does
        CompressionCodecFactory snappyLike = new CompressionCodecFactory() {
            @Override public BytesInputCompressor getCompressor(CompressionCodecName c) { return base.getCompressor(c); }
            @Override public BytesInputDecompressor getDecompressor(CompressionCodecName c) {
                return c == CompressionCodecName.SNAPPY ? base.marker : base.getDecompressor(c);
            }
            @Override public void release() { base.release(); }
        };
        PureJavaZstdCodecFactory f = new PureJavaZstdCodecFactory(snappyLike);
        assertSame(base.marker, f.getDecompressor(CompressionCodecName.SNAPPY));
        assertNotSame(base.marker, f.getDecompressor(CompressionCodecName.ZSTD));
        f.release();
        assertEquals(List.of("release"), base.calls, "release must reach the bottom of the stack");
    }

    // ------------------------------------------------------------------ decompress(BytesInput, int)

    @Test
    void decodesAPageGivenAsBytesInput() throws IOException {
        BytesInput out = zstd(new RecordingDelegate()).decompress(BytesInput.from(hex(TEXT_FRAME)), 3000);
        assertArrayEquals(TEXT, out.toByteArray());
        assertEquals(3000, out.size());
    }

    @Test
    void acceptsEveryKindOfInputBuffer() throws IOException {
        byte[] f = hex(TEXT_FRAME);
        BytesInputDecompressor z = zstd(new RecordingDelegate());

        ByteBuffer slice = ByteBuffer.wrap(new byte[f.length + 40], 17, f.length);
        slice.put(f).flip();   // position 0..len of the wrapped region, arrayOffset 0 -> still a plain heap buffer
        ByteBuffer offsetSlice;
        {
            byte[] big = new byte[f.length + 100];
            System.arraycopy(f, 0, big, 33, f.length);
            offsetSlice = ByteBuffer.wrap(big, 33, f.length).slice(); // arrayOffset() == 33, position 0
            assertEquals(33, offsetSlice.arrayOffset());
        }
        ByteBuffer[] inputs = {ByteBuffer.wrap(f), offsetSlice, direct(f), ByteBuffer.wrap(f).asReadOnlyBuffer()};
        String[] names = {"heap", "heap slice with arrayOffset", "direct", "read-only heap"};
        for (int i = 0; i < inputs.length; i++) {
            int pos = inputs[i].position(), lim = inputs[i].limit();
            byte[] got = z.decompress(BytesInput.from(inputs[i]), 3000).toByteArray();
            assertArrayEquals(TEXT, got, names[i]);
            assertEquals(pos, inputs[i].position(), names[i] + ": input position must be left alone");
            assertEquals(lim, inputs[i].limit(), names[i] + ": input limit must be left alone");
        }
    }

    @Test
    void aPageMadeOfSeveralConcatenatedFramesIsAccepted() throws IOException {
        // Legal zstd, and libzstd's ZSTD_decompress takes it: the FIRST frame declares only 500 of the page's 1200
        // bytes, so a "declared size must equal page size" shortcut would wrongly reject this.
        byte[] out = zstd(new RecordingDelegate()).decompress(BytesInput.from(hex(CONCAT_FRAMES)), 1200).toByteArray();
        byte[] expected = new byte[1200];
        System.arraycopy(textLike(500, 1), 0, expected, 0, 500);
        System.arraycopy(textLike(700, 2), 0, expected, 500, 700);
        assertArrayEquals(expected, out);
    }

    @Test
    void oneDecoderServesManyPagesOfVaryingSizeAndInputKind() throws IOException {
        BytesInputDecompressor z = zstd(new RecordingDelegate());
        byte[] a = hex(TEXT_FRAME), b = hex(PATTERN_FRAME);
        for (int i = 0; i < 40; i++) {
            // alternating sizes and a direct (staging-buffer) input whose size grows then shrinks
            assertArrayEquals(TEXT, z.decompress(BytesInput.from(direct(a)), 3000).toByteArray());
            assertArrayEquals(patterned(8000), z.decompress(BytesInput.from(direct(b)), 8000).toByteArray());
            assertArrayEquals(TEXT, z.decompress(BytesInput.from(a), 3000).toByteArray());
        }
    }

    // ------------------------------------------------------------------ decompress(ByteBuffer, int, ByteBuffer, int)

    @Test
    void byteBufferOverloadHonoursParquetJavasPositionAndLimitContract() throws IOException {
        byte[] f = hex(TEXT_FRAME);
        BytesInputDecompressor z = zstd(new RecordingDelegate());
        // 7 junk bytes, the compressed page, then 9 bytes that must not be read or disturbed
        byte[] wire = new byte[7 + f.length + 9];
        Arrays.fill(wire, (byte) 0x7e);
        System.arraycopy(f, 0, wire, 7, f.length);
        ByteBuffer in = ByteBuffer.wrap(wire);
        in.position(7);
        byte[] outArr = new byte[5 + 3000 + 6];
        Arrays.fill(outArr, (byte) 0x33);
        ByteBuffer out = ByteBuffer.wrap(outArr);
        out.position(5);

        z.decompress(in, f.length, out, 3000);

        assertEquals(7 + f.length, in.position(), "input position advances by exactly compressedSize");
        assertEquals(wire.length, in.limit(), "input limit is restored");
        assertEquals(5 + 3000, out.position(), "output position advances by uncompressedSize");
        assertArrayEquals(TEXT, Arrays.copyOfRange(outArr, 5, 3005));
        for (int i = 0; i < 5; i++) assertEquals((byte) 0x33, outArr[i]);
        for (int i = 3005; i < outArr.length; i++) assertEquals((byte) 0x33, outArr[i], "nothing past the page may be written");
    }

    @Test
    void byteBufferOverloadWorksForDirectBuffersOnBothSides() throws IOException {
        byte[] f = hex(TEXT_FRAME);
        ByteBuffer in = direct(f);
        ByteBuffer out = ByteBuffer.allocateDirect(3000 + 10);
        zstd(new RecordingDelegate()).decompress(in, f.length, out, 3000);
        assertEquals(3000, out.position());
        assertEquals(f.length, in.position());
        out.flip();
        byte[] got = new byte[3000];
        out.get(got);
        assertArrayEquals(TEXT, got);
    }

    @Test
    void byteBufferOverloadRejectsShortInputAndShortOutput() {
        byte[] f = hex(TEXT_FRAME);
        BytesInputDecompressor z = zstd(new RecordingDelegate());
        assertThrows(IllegalArgumentException.class,
                () -> z.decompress(ByteBuffer.wrap(f, 0, 10), f.length, ByteBuffer.allocate(3000), 3000),
                "fewer input bytes than compressedSize");
        ByteBuffer small = ByteBuffer.allocate(2999);
        assertThrows(IOException.class, () -> z.decompress(ByteBuffer.wrap(f), f.length, small, 3000),
                "output buffer smaller than the page");
        assertEquals(0, small.position(), "a rejected call must not have written anything");
    }

    @Test
    void aFailedByteBufferDecodeStillRestoresTheInputLimit() {
        byte[] junk = new byte[64];
        Arrays.fill(junk, (byte) 0x11);
        ByteBuffer in = ByteBuffer.wrap(junk, 0, 64);
        in.limit(64);
        assertThrows(IOException.class, () -> zstd(new RecordingDelegate()).decompress(in, 32, ByteBuffer.allocate(100), 100));
        assertEquals(64, in.limit(), "limit must be restored even when decoding throws");
    }

    // ------------------------------------------------------------------ errors surface as IOException

    @Test
    void sizeMismatchesAreIOExceptionsWithAUsefulMessage() {
        BytesInputDecompressor z = zstd(new RecordingDelegate());
        byte[] f = hex(TEXT_FRAME);
        IOException tooSmall = assertThrows(IOException.class, () -> z.decompress(BytesInput.from(f), 2999));
        assertTrue(tooSmall.getMessage().contains("3000") && tooSmall.getMessage().contains("2999"), tooSmall.getMessage());
        IOException tooBig = assertThrows(IOException.class, () -> z.decompress(BytesInput.from(f), 3001));
        assertTrue(tooBig.getMessage().contains("3001"), tooBig.getMessage());
        assertThrows(IOException.class, () -> z.decompress(BytesInput.from(f), -1));
    }

    @Test
    void corruptTruncatedAndEmptyInputAreIOExceptionsNotRuntimeExceptions() {
        BytesInputDecompressor z = zstd(new RecordingDelegate());
        byte[] f = hex(TEXT_FRAME);
        byte[] junk = new byte[300];
        new java.util.Random(5).nextBytes(junk);
        assertThrows(IOException.class, () -> z.decompress(BytesInput.from(junk), 3000));
        assertThrows(IOException.class, () -> z.decompress(BytesInput.from(Arrays.copyOf(f, f.length / 2)), 3000));
        assertThrows(IOException.class, () -> z.decompress(BytesInput.empty(), 3000), "no frame at all, 3000 bytes expected");
        // and the decoder is still good afterwards
        assertDoesNotThrow(() -> z.decompress(BytesInput.from(f), 3000));
    }

    // ------------------------------------------------------------------ against parquet-java's own ZSTD writer

    @Test
    void decodesPagesProducedByParquetJavasOwnZstdCompressor() throws Exception {
        CompressionCodecFactory stock;
        BytesInputCompressor compressor;
        try {
            stock = org.apache.parquet.hadoop.util.HadoopCodecs.newFactory(0);
            compressor = stock.getCompressor(CompressionCodecName.ZSTD);
        } catch (Throwable t) { // zstd-jni absent: this is the one test that needs a native compressor to MAKE frames
            org.junit.jupiter.api.Assumptions.assumeTrue(false, "no zstd compressor available: " + t);
            return;
        }
        BytesInputDecompressor z = zstd(new RecordingDelegate());
        java.util.Random rnd = new java.util.Random(11);
        int[] sizes = {0, 1, 2, 15, 16, 255, 256, 4096, 65_535, 65_536, 131_072, 300_000, 1_500_000};
        for (int size : sizes) {
            for (int kind = 0; kind < 3; kind++) {
                byte[] raw = new byte[size];
                switch (kind) {
                    case 0 -> rnd.nextBytes(raw);                                          // incompressible
                    case 1 -> { for (int i = 0; i < size; i++) raw[i] = (byte) (i / 7); }  // runs and ramps
                    default -> { byte[] t = textLike(Math.max(size, 1), size + 3); System.arraycopy(t, 0, raw, 0, size); }
                }
                BytesInput packed = compressor.compress(BytesInput.from(raw));
                byte[] out = z.decompress(packed, size).toByteArray();
                assertArrayEquals(raw, out, "size=" + size + " kind=" + kind);
            }
        }
        compressor.release();
        stock.release();
    }

}
