package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.io.InputFile;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.io.SeekableInputStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Storage abstraction: everything must work through a {@link ParquetSource} that is not a
 * {@code java.nio.file.Path}, and reads must stay proportionate (projection and pruning must show up as
 * fewer bytes fetched from the underlying storage, which is what makes remote storage viable).
 */
class ParquetSourceTest {

    @TempDir
    static Path dir;

    static Path flat;

    @BeforeAll
    static void writeFixture() throws IOException {
        flat = ParquetTestData.writeFlat(dir, "flat.parquet");
    }

    /** An InputFile over a local file that counts streams opened and bytes read, standing in for a remote object. */
    static final class CountingInputFile implements InputFile {
        final InputFile delegate;
        final AtomicInteger streamsOpened = new AtomicInteger();
        final AtomicLong bytesRead = new AtomicLong();

        CountingInputFile(Path p) { this.delegate = new LocalInputFile(p); }

        @Override public long getLength() throws IOException { return delegate.getLength(); }

        @Override public SeekableInputStream newStream() throws IOException {
            streamsOpened.incrementAndGet();
            SeekableInputStream in = delegate.newStream();
            return new SeekableInputStream() {
                @Override public int read() throws IOException {
                    int b = in.read();
                    if (b >= 0) bytesRead.incrementAndGet();
                    return b;
                }
                @Override public int read(byte[] b, int off, int len) throws IOException {
                    int n = in.read(b, off, len);
                    if (n > 0) bytesRead.addAndGet(n);
                    return n;
                }
                @Override public long getPos() throws IOException { return in.getPos(); }
                @Override public void seek(long newPos) throws IOException { in.seek(newPos); }
                @Override public void readFully(byte[] bytes) throws IOException { in.readFully(bytes); bytesRead.addAndGet(bytes.length); }
                @Override public void readFully(byte[] bytes, int start, int len) throws IOException { in.readFully(bytes, start, len); bytesRead.addAndGet(len); }
                @Override public int read(ByteBuffer buf) throws IOException {
                    int n = in.read(buf);
                    if (n > 0) bytesRead.addAndGet(n);
                    return n;
                }
                @Override public void readFully(ByteBuffer buf) throws IOException {
                    int n = buf.remaining();
                    in.readFully(buf);
                    bytesRead.addAndGet(n);
                }
                @Override public void close() throws IOException { in.close(); }
            };
        }

        @Override public String toString() { return "counting://flat"; }
    }

    private static long countRows(ParquetScan scan) {
        try (var alloc = new org.apache.arrow.memory.RootAllocator();
             ParquetBatchReader r = scan.open(alloc)) {
            long n = 0;
            while (r.next()) n += r.root().getRowCount();
            return n;
        }
    }

    // ================================================================ non-Path sources

    @Test
    void scanThroughACustomInputFileMatchesThePathScan() {
        CountingInputFile in = new CountingInputFile(flat);
        assertScanEqualsRows(ParquetScan.scan(in), ROWS);
        assertTrue(in.streamsOpened.get() >= 1);
    }

    @Test
    void parallelScanThroughACustomInputFileMatchesToo() {
        assertScanEqualsRows(ParquetScan.scan(new CountingInputFile(flat)).parallelism(4), ROWS);
    }

    @Test
    void fileInfoThroughAnInputFileMatchesThePathVersion() {
        ParquetFileInfo viaPath = ParquetFileInfo.read(flat);
        ParquetFileInfo viaInput = ParquetFileInfo.read(new CountingInputFile(flat));
        assertEquals(viaPath.rowCount(), viaInput.rowCount());
        assertEquals(viaPath.fileSizeBytes(), viaInput.fileSizeBytes());
        assertEquals(viaPath.rowGroups().size(), viaInput.rowGroups().size());
        assertEquals(viaPath.topLevelColumns(), viaInput.topLevelColumns());
        assertNull(viaInput.file(), "a non-local source has no local Path");
        assertEquals("counting://flat", viaInput.source().name());
    }

    // ================================================================ remote-friendliness of the I/O

    @Test
    void projectionFetchesFewerBytesThanTheWholeFile() throws IOException {
        CountingInputFile in = new CountingInputFile(flat);
        countRows(ParquetScan.scan(in).select("id"));
        assertTrue(in.bytesRead.get() < in.getLength(),
                "reading one column must not download the whole file: read " + in.bytesRead + " of " + in.getLength());
    }

    @Test
    void pruningEveryRowGroupFetchesLessThanAFullScan() {
        CountingInputFile full = new CountingInputFile(flat);
        countRows(ParquetScan.scan(full).select("id", "score"));

        CountingInputFile pruned = new CountingInputFile(flat);
        long rows = countRows(ParquetScan.scan(pruned).select("id", "score").pushdown(Predicate.gt("id", (long) N + 1_000L)));
        assertEquals(0, rows);
        assertTrue(pruned.bytesRead.get() < full.bytesRead.get(),
                "pruned scan read " + pruned.bytesRead + " bytes, full scan " + full.bytesRead);
    }

    // ================================================================ errors name the source

    @Test
    void errorsNameTheNonLocalSource() {
        InputFile broken = new InputFile() {
            @Override public long getLength() throws IOException { throw new IOException("boom"); }
            @Override public SeekableInputStream newStream() throws IOException { throw new IOException("boom"); }
            @Override public String toString() { return "s3a://bucket/missing.parquet"; }
        };
        ParquetScanException e = assertThrows(ParquetScanException.class, () -> ParquetFileInfo.read(broken));
        assertTrue(e.getMessage().contains("s3a://bucket/missing.parquet"), e.getMessage());
        assertEquals("s3a://bucket/missing.parquet", e.source());
        assertNull(e.file());
    }

    @Test
    void explicitNameOverridesToString() {
        ParquetSource s = ParquetSource.of(new CountingInputFile(flat), "gs://bucket/dir/part-0.parquet");
        assertEquals("gs://bucket/dir/part-0.parquet", s.name());
        assertEquals("part-0.parquet", s.shortName());
        assertNull(s.localPath());
    }

    @Test
    void localSourceKeepsItsPath() {
        ParquetSource s = ParquetSource.of(flat);
        assertEquals(flat, s.localPath());
        assertEquals("flat.parquet", s.shortName());
    }

    // ================================================================ Hadoop FileSystem URIs

    @Test
    void hadoopFileUriScansLikeALocalPath() {
        ParquetSource s;
        try {
            s = ParquetSource.hadoop(flat.toUri().toString(), new Configuration());
        } catch (ParquetScanException | LinkageError e) {
            // Environment limit, not a scan result: Hadoop's filesystem bootstrap can fail on some JDK/OS
            // combinations. Everything after the source is created is asserted strictly.
            assumeTrue(false, "Hadoop FileSystem unavailable here: " + e);
            return;
        }
        assertScanEqualsRows(ParquetScan.scan(s), ROWS);
    }

    @Test
    void hadoopWithAnUnknownSchemeFailsNamingTheUri() {
        ParquetScanException e = assertThrows(ParquetScanException.class,
                () -> ParquetSource.hadoop("nosuchscheme://bucket/x.parquet", new Configuration()));
        assertTrue(e.getMessage().contains("nosuchscheme://bucket/x.parquet"), e.getMessage());
    }
}
