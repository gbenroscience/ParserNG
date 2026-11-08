package com.github.gbenroscience.gpu.llm.opencl;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Reads a model file sequentially on a background thread so that the operating system has it in its file cache by the
 * time the loader asks for it. Nothing is kept on the Java heap: the bytes are read into one small reused buffer and
 * dropped; the only effect is that later reads of the same file are served from memory instead of from the disk.
 *
 * <p>Why: loading a 1.3 GB GGUF took 6-7 s of pure disk waiting (about 200 MB/s) on the test laptop, and that waiting used to
 * happen strictly before or after the 3-13 s the driver needs to compile the kernels and the second or two of device
 * selection. Disk and compiler use different resources, so doing both at once hides most of the smaller one.
 *
 * <p>Safe by construction: a daemon thread, every failure swallowed (the loader then simply reads from disk as before),
 * skipped when the file is large relative to physical memory (reading ahead would then evict the pages it just loaded),
 * and enabled only with {@code -Dllm.prefetch=true}.
 */
final class FilePrefetcher {

    private static final int CHUNK = 8 << 20;

    private final File file;
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicLong bytesRead = new AtomicLong();
    private final CountDownLatch done = new CountDownLatch(1);

    private FilePrefetcher(File file) {
        this.file = file;
    }

    /** Opt-in: -Dllm.prefetch=true. Not measured on real hardware yet; on a slow disk concurrent readers can be slower than one. */
    static boolean enabled() {
        String v = System.getProperty("llm.prefetch");
        return v != null && (v.equalsIgnoreCase("true") || v.equals("1") || v.equalsIgnoreCase("on"));
    }

    /** Starts reading {@code file} ahead, or returns null if prefetching is disabled or would not be a good idea. */
    static FilePrefetcher start(File file) {
        if (!enabled() || file == null || !file.isFile()) {
            return null;
        }
        long physical = physicalMemoryBytes();
        if (physical > 0 && file.length() > physical / 2) {
            return null; // read-ahead would push out what it has just read
        }
        FilePrefetcher p = new FilePrefetcher(file);
        Thread t = new Thread(p::run, "gguf-prefetch");
        t.setDaemon(true);
        t.start();
        return p;
    }

    private static long physicalMemoryBytes() {
        try {
            java.lang.management.OperatingSystemMXBean os = java.lang.management.ManagementFactory.getOperatingSystemMXBean();
            if (os instanceof com.sun.management.OperatingSystemMXBean sun) {
                return sun.getTotalMemorySize();
            }
        } catch (Throwable ignored) {
            // unknown: prefetch anyway
        }
        return -1;
    }

    private void run() {
        try (FileChannel ch = FileChannel.open(file.toPath(), StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocateDirect(CHUNK);
            long pos = 0;
            final long size = ch.size();
            while (pos < size && !cancelled.get()) {
                buf.clear();
                int n = ch.read(buf, pos);
                if (n <= 0) {
                    break;
                }
                pos += n;
                bytesRead.set(pos);
            }
        } catch (IOException | RuntimeException ignored) {
            // the loader will read from disk exactly as it did before
        } finally {
            done.countDown();
        }
    }

    /** Stops reading ahead (the thread ends after its current read). */
    void cancel() {
        cancelled.set(true);
    }

    /** Waits until the thread has finished (end of file, cancel or failure). */
    boolean await(long timeout, TimeUnit unit) throws InterruptedException {
        return done.await(timeout, unit);
    }

    long bytesRead() {
        return bytesRead.get();
    }
}
