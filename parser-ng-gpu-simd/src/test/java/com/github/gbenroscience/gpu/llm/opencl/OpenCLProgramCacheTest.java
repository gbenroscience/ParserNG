package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.core.CpuReference;
import com.github.gbenroscience.gpu.llm.core.TinyModel;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The compiled-program cache (round trip, corruption, opt-out, keying) and the decode flush interval. */
class OpenCLProgramCacheTest {

    private static String probeSkip() {
        try (GpuContext c = new GpuContext()) {
            return null;
        } catch (Throwable t) {
            return "no usable OpenCL device: " + t;
        }
    }

    private static long binCount(Path dir) throws Exception {
        if (!Files.isDirectory(dir)) {
            return 0;
        }
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> p.toString().endsWith(".clbin")).count();
        }
    }

    private static void withProps(String[][] kv, ThrowingRunnable body) throws Throwable {
        String[] old = new String[kv.length];
        for (int i = 0; i < kv.length; i++) {
            old[i] = System.getProperty(kv[i][0]);
            System.setProperty(kv[i][0], kv[i][1]);
        }
        try {
            body.run();
        } finally {
            for (int i = 0; i < kv.length; i++) {
                if (old[i] == null) {
                    System.clearProperty(kv[i][0]);
                } else {
                    System.setProperty(kv[i][0], old[i]);
                }
            }
        }
    }

    private interface ThrowingRunnable {
        void run() throws Throwable;
    }

    /** One SoA GEMV on this context against the CPU reference: proves the (possibly cached) program computes correctly. */
    private static void assertProgramWorks(GpuContext c) throws Throwable {
        if (!c.soaAvailable) {
            return;
        }
        c.q8Soa = true;
        final int N = 7, K = 96;
        Random r = new Random(5);
        byte[] aos = GemvSweep.fakeQ8((long) N * (K / 32), r);
        float[] x = new float[K];
        for (int i = 0; i < K; i++) {
            x[i] = (float) r.nextGaussian();
        }
        float[] ref = CpuReference.q8Gemv(CpuReference.quantizeQ8_0(x), aos, N, K);
        MemorySegment dw = LlamaLayer.uploadBytes(c, Q8Layout.repackWeightsSoa(aos, N, K));
        MemorySegment dxf = LlamaLayer.uploadFloats(c, x);
        MemorySegment dx = LlamaLayer.allocBytes(c, Q8Layout.activationBytes(true, K));
        MemorySegment dout = LlamaLayer.allocFloats(c, N);
        try {
            LlamaLayer.quantizeActivationQ8_0(c, dxf, dx, K);
            LlamaLayer.q8_0GemvPlain(c, dx, dw, dout, N, K);
            LlamaLayer.finish(c);
            float[] got = LlamaLayer.downloadFloats(c, dout, N);
            for (int i = 0; i < N; i++) {
                assertEquals(ref[i], got[i], 1e-4f * (1f + Math.abs(ref[i])), "element " + i);
            }
        } finally {
            for (MemorySegment m : new MemorySegment[]{dw, dxf, dx, dout}) {
                LlamaLayer.freeQuietly(c, m);
            }
        }
    }

    @Test
    void secondContextLoadsTheCachedBinaryAndItStillComputesCorrectly() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        Path dir = Files.createTempDirectory("clcache");
        withProps(new String[][]{{"llm.programCache.dir", dir.toString()}, {"llm.programCache", "true"}}, () -> {
            try (GpuContext first = new GpuContext()) {
                assertFalse(first.programFromCache, "nothing cached yet");
                assertTrue(binCount(dir) >= 1, "a binary must have been written");
                assertProgramWorks(first);
            }
            try (GpuContext second = new GpuContext()) {
                assertTrue(second.programFromCache, "the second start must reuse the binary (driver refused it, or the cache was not written)");
                assertProgramWorks(second);
            }
        });
    }

    @Test
    void aCorruptedFileIsRejectedDeletedAndRebuilt() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        Path dir = Files.createTempDirectory("clcache");
        withProps(new String[][]{{"llm.programCache.dir", dir.toString()}, {"llm.programCache", "true"}}, () -> {
            try (GpuContext first = new GpuContext()) {
                assertTrue(binCount(dir) >= 1);
            }
            Path f;
            try (Stream<Path> s = Files.list(dir)) {
                f = s.filter(p -> p.toString().endsWith(".clbin")).findFirst().orElseThrow();
            }
            byte[] b = Files.readAllBytes(f);
            b[b.length / 2] ^= 0x5A; // flip a payload byte
            Files.write(f, b);
            try (GpuContext second = new GpuContext()) {
                assertFalse(second.programFromCache, "a file failing its checksum must not be used");
                assertProgramWorks(second);
            }
            try (GpuContext third = new GpuContext()) {
                assertTrue(third.programFromCache, "the rebuild must have rewritten a valid file");
            }
        });
    }

    @Test
    void disablingTheCacheWritesNothing() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        Path dir = Files.createTempDirectory("clcache");
        withProps(new String[][]{{"llm.programCache.dir", dir.toString()}, {"llm.programCache", "false"}}, () -> {
            try (GpuContext c = new GpuContext()) {
                assertFalse(c.programFromCache);
                assertProgramWorks(c);
            }
            assertEquals(0, binCount(dir));
        });
    }

    @Test
    void storeRejectsShortAndAlteredFilesAndKeysDependOnEveryInput() throws Throwable {
        Path dir = Files.createTempDirectory("clcache");
        withProps(new String[][]{{"llm.programCache.dir", dir.toString()}}, () -> {
            String k = ProgramCache.key("dev", "vendor", "1.0", "OpenCL 1.2", "source");
            assertNull(ProgramCache.read(k));
            assertTrue(ProgramCache.write(k, new byte[]{1, 2, 3, 4}));
            assertArrayEquals(new byte[]{1, 2, 3, 4}, ProgramCache.read(k));
            Files.write(dir.resolve(k + ".clbin"), new byte[10]); // shorter than the checksum header
            assertNull(ProgramCache.read(k));
            assertEquals(0, binCount(dir), "a rejected file is deleted");
            String[] others = {
                ProgramCache.key("dev2", "vendor", "1.0", "OpenCL 1.2", "source"),
                ProgramCache.key("dev", "vendor2", "1.0", "OpenCL 1.2", "source"),
                ProgramCache.key("dev", "vendor", "1.1", "OpenCL 1.2", "source"),
                ProgramCache.key("dev", "vendor", "1.0", "OpenCL 2.0", "source"),
                ProgramCache.key("dev", "vendor", "1.0", "OpenCL 1.2", "source2"),
                ProgramCache.key("de", "vvendor", "1.0", "OpenCL 1.2", "source")};
            for (String o : others) {
                assertNotEquals(k, o);
            }
        });
    }

    @Test
    void flushIntervalDoesNotChangeTheGeneratedText() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        TinyModel m = new TinyModel();
        m.layers = 4;
        m.build(21);
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        m.write(f);
        String[] text = new String[4];
        int i = 0;
        for (String every : new String[]{"0", "1", "2", "3"}) {
            int idx = i++;
            withProps(new String[][]{{"llm.flushEvery", every}}, () -> {
                try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f, new GpuContext())) {
                    LlamaGenerationConfig g = new LlamaGenerationConfig();
                    g.temperature = 0f;
                    g.repetitionPenalty = 1f;
                    g.maxNewTokens = 8;
                    g.eosTokenId = -1;
                    text[idx] = eng.generate("hello there, this is a test", g);
                }
            });
        }
        for (int j = 1; j < text.length; j++) {
            assertEquals(text[0], text[j], "flushEvery changes timing only");
        }
        withProps(new String[][]{{"llm.flushEvery", "x"}}, () -> assertThrows(IllegalArgumentException.class, Variants::flushEvery));
    }
}
