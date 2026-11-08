package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.LlamaGenerationConfig;
import com.github.gbenroscience.gpu.llm.core.TinyModel;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Files;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The concurrent load path ({@link OpenCLLlamaEngine#OpenCLLlamaEngine(File)}) and the file read-ahead. */
class OpenCLParallelLoadTest {

    private static String probeSkip() {
        try (GpuContext c = new GpuContext()) {
            return null;
        } catch (Throwable t) {
            return "no usable OpenCL device: " + t;
        }
    }

    private static String generate(OpenCLLlamaEngine eng) throws Throwable {
        LlamaGenerationConfig g = new LlamaGenerationConfig();
        g.temperature = 0f;
        g.repetitionPenalty = 1f;
        g.maxNewTokens = 8;
        g.eosTokenId = -1;
        return eng.generate("hello there, this is a test", g);
    }

    private static File tinyModel(long seed) throws Exception {
        TinyModel m = new TinyModel();
        m.q8Wo = true;
        m.build(seed);
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        m.write(f);
        return f;
    }

    private static void setProp(String k, String v) {
        if (v == null) {
            System.clearProperty(k);
        } else {
            System.setProperty(k, v);
        }
    }

    @Test
    void publicConstructorGivesTheSameTextInParallelAndSequentialMode() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        File f = tinyModel(31);
        String reference;
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f, new GpuContext())) {
            reference = generate(eng);
        }
        String old = System.getProperty("llm.parallelLoad");
        try {
            for (String mode : new String[]{"true", "false"}) {
                setProp("llm.parallelLoad", mode);
                try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f)) {
                    assertEquals(reference, generate(eng), "parallelLoad=" + mode);
                }
            }
        } finally {
            setProp("llm.parallelLoad", old);
        }
    }

    @Test
    void anInvalidModelFailsWithItsOwnErrorAndLeavesNoDeviceContextOpen() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        File junk = Files.createTempFile("junk", ".gguf").toFile();
        junk.deleteOnExit();
        byte[] garbage = new byte[4096];
        new Random(1).nextBytes(garbage);
        Files.write(junk.toPath(), garbage);
        int before = GpuContext.OPEN_CONTEXTS.get();
        String oldP = System.getProperty("llm.parallelLoad");
        System.setProperty("llm.parallelLoad", "true"); // the concurrent path is opt-in
        Throwable t;
        try {
            t = assertThrows(Throwable.class, () -> new OpenCLLlamaEngine(junk));
        } finally {
            setProp("llm.parallelLoad", oldP);
        }
        assertTrue(String.valueOf(t.getMessage()).contains("Invalid GGUF magic"), "the model's own error must surface, got: " + t);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
        while (GpuContext.OPEN_CONTEXTS.get() != before && System.nanoTime() < deadline) {
            Thread.sleep(20);
        }
        assertEquals(before, GpuContext.OPEN_CONTEXTS.get(), "the context created concurrently for a bad model must be closed");
    }

    @Test
    void aSuccessfulLoadAndCloseLeavesNoContextOpen() throws Throwable {
        String skip = probeSkip();
        assumeTrue(skip == null, skip);
        File f = tinyModel(32);
        int before = GpuContext.OPEN_CONTEXTS.get();
        String oldP = System.getProperty("llm.parallelLoad");
        System.setProperty("llm.parallelLoad", "true");
        try (OpenCLLlamaEngine eng = new OpenCLLlamaEngine(f)) {
            assertEquals(before + 1, GpuContext.OPEN_CONTEXTS.get());
        } finally {
            setProp("llm.parallelLoad", oldP);
        }
        assertEquals(before, GpuContext.OPEN_CONTEXTS.get());
    }

    @Test
    void prefetcherReadsTheWholeFileCancelsCleanlyAndHonoursItsSwitch() throws Throwable {
        File f = File.createTempFile("prefetch", ".bin");
        f.deleteOnExit();
        byte[] data = new byte[20 << 20];
        new Random(2).nextBytes(data);
        Files.write(f.toPath(), data);

        assertNull(FilePrefetcher.start(f), "off by default");
        String oldPf = System.getProperty("llm.prefetch");
        System.setProperty("llm.prefetch", "true");
        try {
            runPrefetcherChecks(f, data.length);
        } finally {
            setProp("llm.prefetch", oldPf);
        }
        String old = System.getProperty("llm.prefetch");
        try {
            System.setProperty("llm.prefetch", "false");
            assertNull(FilePrefetcher.start(f));
        } finally {
            setProp("llm.prefetch", old);
        }
    }

    private static void runPrefetcherChecks(File f, int length) throws Throwable {
        byte[] data = new byte[length];
        FilePrefetcher p = FilePrefetcher.start(f);
        assertNotNull(p);
        assertTrue(p.await(30, TimeUnit.SECONDS));
        assertEquals(data.length, p.bytesRead());

        FilePrefetcher c = FilePrefetcher.start(f);
        assertNotNull(c);
        c.cancel();
        assertTrue(c.await(30, TimeUnit.SECONDS), "a cancelled prefetch must end");
        assertTrue(c.bytesRead() <= data.length);

        assertNull(FilePrefetcher.start(new File(f.getPath() + ".missing")), "a missing file is simply not prefetched");
    }
}
