package com.github.gbenroscience.simd.turbo.tools.llm.loader;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tensor readers of GGUFLoader against a plain per-element reference, on tensors larger than one read chunk
 * (4 MiB) and one dequantisation chunk (65536 Q8_0 blocks), so every chunk boundary is crossed.
 */
class GGUFLoaderTest {

    private static final int[] Q8_BIG = {2048, 2700};   // 5,529,600 elements = 172,800 blocks = 5.87 MB: three dequant chunks, two read chunks
    private static final int[] F32_BIG = {1024, 1500};  // 6.1 MB: two read chunks
    private static final int[] F32_SMALL = {7, 3};
    private static final int[] Q8_TINY = {32, 1};

    private static File file;
    private static byte[] q8Big, q8Tiny, f32Big, f32Small;

    private static byte[] randomQ8(Random r, int elements) {
        int blocks = elements / 32;
        ByteBuffer b = ByteBuffer.allocate(blocks * 34).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            b.putShort(Float.floatToFloat16(0.001f + r.nextFloat())); // finite, non-zero scale
            for (int j = 0; j < 32; j++) {
                b.put((byte) (r.nextInt(256) - 128));
            }
        }
        return b.array();
    }

    private static byte[] randomF32(Random r, int elements) {
        ByteBuffer b = ByteBuffer.allocate(elements * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < elements; i++) {
            b.putFloat((float) r.nextGaussian());
        }
        return b.array();
    }

    private static void str(ByteArrayOutputStream o, String s) throws IOException {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        o.write(le(b.length, 8));
        o.write(b);
    }

    private static byte[] le(long v, int bytes) {
        return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array().length == 8
                ? java.util.Arrays.copyOf(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array(), bytes) : null;
    }

    private static long elements(int[] dims) {
        long n = 1;
        for (int d : dims) {
            n *= d;
        }
        return n;
    }

    private static void tensorInfo(ByteArrayOutputStream o, String name, int[] dims, int type, long offset) throws IOException {
        str(o, name);
        o.write(le(dims.length, 4));
        for (int d : dims) {
            o.write(le(d, 8));
        }
        o.write(le(type, 4));
        o.write(le(offset, 8));
    }

    private static long align(long v) {
        return (v + 31) / 32 * 32;
    }

    @BeforeAll
    static void writeFile() throws Exception {
        Random r = new Random(7);
        q8Big = randomQ8(r, (int) elements(Q8_BIG));
        q8Tiny = randomQ8(r, (int) elements(Q8_TINY));
        f32Big = randomF32(r, (int) elements(F32_BIG));
        f32Small = randomF32(r, (int) elements(F32_SMALL));
        byte[][] data = {f32Small, q8Big, q8Tiny, f32Big};
        String[] names = {"f32.small", "q8.big", "q8.tiny", "f32.big"};
        int[][] dims = {F32_SMALL, Q8_BIG, Q8_TINY, F32_BIG};
        int[] types = {0, 8, 8, 0};

        ByteArrayOutputStream head = new ByteArrayOutputStream();
        head.write(le(0x46554747, 4));
        head.write(le(3, 4));
        head.write(le(data.length, 8));
        head.write(le(2, 8)); // two key/value pairs
        str(head, "general.architecture");
        head.write(le(8, 4)); // string
        str(head, "llama");
        str(head, "general.alignment");
        head.write(le(4, 4)); // uint32
        head.write(le(32, 4));
        long off = 0;
        for (int i = 0; i < data.length; i++) {
            tensorInfo(head, names[i], dims[i], types[i], off);
            off = align(off + data[i].length);
        }
        ByteArrayOutputStream all = new ByteArrayOutputStream();
        all.write(head.toByteArray());
        while (all.size() % 32 != 0) {
            all.write(0);
        }
        long cur = 0;
        for (byte[] d : data) {
            all.write(d);
            cur += d.length;
            while (cur % 32 != 0) { // tensor data is aligned relative to the data start
                all.write(0);
                cur++;
            }
        }
        file = File.createTempFile("gguf-loader-test", ".gguf");
        file.deleteOnExit();
        Files.write(file.toPath(), all.toByteArray());
    }

    private static float[] refDequantFloat(byte[] raw) {
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[raw.length / 34 * 32];
        int idx = 0;
        while (b.hasRemaining()) {
            float scale = Float.float16ToFloat(b.getShort());
            for (int j = 0; j < 32; j++) {
                out[idx++] = b.get() * scale;
            }
        }
        return out;
    }

    private static double[] refDequantDouble(byte[] raw) {
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        double[] out = new double[raw.length / 34 * 32];
        int idx = 0;
        while (b.hasRemaining()) {
            double scale = Float.float16ToFloat(b.getShort());
            for (int j = 0; j < 32; j++) {
                out[idx++] = b.get() * scale;
            }
        }
        return out;
    }

    private static float[] refFloat(byte[] raw) {
        float[] out = new float[raw.length / 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }

    @Test
    void rawQ8BytesAreReturnedUnchanged() throws Exception {
        GGUFLoader.GGUFFile g = GGUFLoader.load(file);
        assertArrayEquals(q8Big, g.tensors.get("q8.big").loadQ8_0());
        assertArrayEquals(q8Tiny, g.tensors.get("q8.tiny").loadQ8_0());
    }

    @Test
    void dequantisedFloatAndDoubleMatchThePerElementReferenceExactly() throws Exception {
        GGUFLoader.GGUFFile g = GGUFLoader.load(file);
        assertArrayEquals(refDequantFloat(q8Big), g.tensors.get("q8.big").loadQ8_0AsFloat(), 0f);
        assertArrayEquals(refDequantFloat(q8Tiny), g.tensors.get("q8.tiny").loadQ8_0AsFloat(), 0f);
        assertArrayEquals(refDequantDouble(q8Big), g.tensors.get("q8.big").loadQ8_0AsDouble(), 0d);
        assertArrayEquals(refDequantDouble(q8Tiny), g.tensors.get("q8.tiny").loadQ8_0AsDouble(), 0d);
    }

    @Test
    void floatReadersMatchTheReferenceAcrossChunkBoundaries() throws Exception {
        GGUFLoader.GGUFFile g = GGUFLoader.load(file);
        assertArrayEquals(refFloat(f32Big), g.tensors.get("f32.big").loadFloat(), 0f);
        assertArrayEquals(refFloat(f32Small), g.tensors.get("f32.small").loadFloat(), 0f);
        float[] f = refFloat(f32Big);
        double[] d = g.tensors.get("f32.big").loadFloatAsDouble();
        assertEquals(f.length, d.length);
        for (int i = 0; i < f.length; i++) {
            assertEquals((double) f[i], d[i], 0d, "element " + i);
        }
    }

    @Test
    void readersFallBackToTheMappingWhenTheFileCanNoLongerBeOpened() throws Exception {
        File copy = File.createTempFile("gguf-loader-gone", ".gguf");
        Files.copy(file.toPath(), copy.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        GGUFLoader.GGUFFile g = GGUFLoader.load(copy);
        boolean deleted = copy.delete(); // refused on Windows while mapped; then there is nothing to test
        assumeTrue(deleted, "the OS keeps mapped files locked");
        assertArrayEquals(q8Big, g.tensors.get("q8.big").loadQ8_0());
        assertArrayEquals(refDequantFloat(q8Big), g.tensors.get("q8.big").loadQ8_0AsFloat(), 0f);
        assertArrayEquals(refFloat(f32Big), g.tensors.get("f32.big").loadFloat(), 0f);
    }

    @Test
    void mappedViewIsStillAvailableAndAgrees() throws Exception {
        GGUFLoader.Tensor t = GGUFLoader.load(file).tensors.get("q8.tiny");
        byte[] viaBuffer = new byte[(int) t.nbytes];
        t.buffer().get(viaBuffer);
        assertArrayEquals(q8Tiny, viaBuffer);
    }

    @Test
    void wrongTypeIsRejectedWithTheSameExceptionAsBefore() throws Exception {
        GGUFLoader.GGUFFile g = GGUFLoader.load(file);
        assertThrows(IllegalArgumentException.class, () -> g.tensors.get("f32.small").loadQ8_0());
        assertThrows(IllegalArgumentException.class, () -> g.tensors.get("f32.small").loadQ8_0AsFloat());
        assertThrows(IllegalArgumentException.class, () -> g.tensors.get("f32.small").loadQ8_0AsDouble());
        assertThrows(IllegalArgumentException.class, () -> g.tensors.get("q8.tiny").loadFloat());
        assertThrows(IllegalArgumentException.class, () -> g.tensors.get("q8.tiny").loadFloatAsDouble());
    }

    @Test
    void aTruncatedFileIsStillRejectedAtLoad() throws Exception {
        File cut = File.createTempFile("gguf-loader-cut", ".gguf");
        cut.deleteOnExit();
        byte[] all = Files.readAllBytes(file.toPath());
        Files.write(cut.toPath(), java.util.Arrays.copyOf(all, all.length - 1000));
        assertThrows(IOException.class, () -> GGUFLoader.load(cut));
    }
}
