package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.pages.OwnedRowGroupReader;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.column.ParquetProperties.WriterVersion;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The page-reading-ownership path ({@code -Dparserng.ownPages=true}, see {@link OwnedRowGroupReader}) must be
 * observationally identical to letting parquet-java read the pages: same rows, same order, for every codec,
 * both data page versions, with and without dictionaries, in sequential and parallel mode, with either SNAPPY
 * decoder, over files with many row groups and many pages per column chunk (so that the per-column page
 * buffers and the chunk buffer are reused hundreds of times).
 *
 * <p>The comparison is against this module's own scan with the switch OFF, not against hand-written
 * expectations: what is under test is "owning the page reads changes nothing", and the baseline path is the
 * one the rest of this suite already pins down.
 */
class OwnedPagesScanTest {

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message owned {"
            + " required int64 id;"
            + " optional double value;"
            + " optional binary name (STRING);"
            + " optional boolean flag;"
            + " required int32 bucket;"
            + " optional binary payload (STRING);"
            + "}");

    private static final int ROWS = 12_000;

    @TempDir
    Path dir;

    static Stream<Object[]> matrix() {
        List<Object[]> out = new ArrayList<>();
        for (CompressionCodecName codec : new CompressionCodecName[]{
                CompressionCodecName.UNCOMPRESSED, CompressionCodecName.SNAPPY,
                CompressionCodecName.GZIP, CompressionCodecName.ZSTD}) {
            for (WriterVersion v : WriterVersion.values()) {
                for (boolean dict : new boolean[]{true, false}) {
                    out.add(new Object[]{codec, v, dict});
                }
            }
        }
        return out.stream();
    }

    private Path write(CompressionCodecName codec, WriterVersion version, boolean dictionary, int rows) throws IOException {
        Path file = dir.resolve("owned-" + codec + "-" + version + "-" + dictionary + "-" + rows + ".parquet");
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(SCHEMA)
                .withWriterVersion(version)
                .withCompressionCodec(codec)
                .withDictionaryEncoding(dictionary)
                .withPageSize(2048)            // many small pages per chunk
                .withRowGroupSize(48L * 1024)  // many row groups
                .build()) {
            for (int i = 0; i < rows; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                if (i % 7 != 0) g.append("value", i * 0.25);
                if (i % 5 != 0) g.append("name", "name-" + (i % 97));
                if (i % 3 != 0) g.append("flag", (i / 11) % 2 == 0);
                g.append("bucket", i % 31);
                // repetitive and incompressible-ish strings in alternation, so Snappy sees both literals and copies
                if (i % 13 != 0) g.append("payload", i % 2 == 0 ? "abcabcabcabcabcabcabcabc-" + (i % 10) : Long.toHexString(i * 0x9E3779B97F4A7C15L));
                w.write(g);
            }
        }
        return file;
    }

    private static List<String> scan(ParquetScan scan, boolean ownPages) {
        String prev = System.getProperty(OwnedRowGroupReader.PROP_ENABLE);
        System.setProperty(OwnedRowGroupReader.PROP_ENABLE, Boolean.toString(ownPages));
        try (BufferAllocator alloc = new RootAllocator(); ParquetBatchReader r = scan.open(alloc)) {
            List<String> rows = new ArrayList<>();
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                for (int i = 0; i < b.getRowCount(); i++) {
                    StringBuilder sb = new StringBuilder();
                    for (FieldVector v : b.getFieldVectors()) sb.append(v.getObject(i)).append('|');
                    rows.add(sb.toString());
                }
            }
            return rows;
        } finally {
            if (prev == null) System.clearProperty(OwnedRowGroupReader.PROP_ENABLE);
            else System.setProperty(OwnedRowGroupReader.PROP_ENABLE, prev);
        }
    }

    @ParameterizedTest(name = "{0} {1} dictionary={2}")
    @MethodSource("matrix")
    void ownedPagesMatchParquetJava(CompressionCodecName codec, WriterVersion version, boolean dictionary) throws IOException {
        Path file = write(codec, version, dictionary, ROWS);
        List<String> baseline = scan(ParquetScan.scan(file), false);
        assertEquals(ROWS, baseline.size(), "baseline sanity");

        assertEquals(baseline, scan(ParquetScan.scan(file), true), "owned, stock snappy-java decoder, sequential");
        assertEquals(baseline, scan(ParquetScan.scan(file).useJavaSnappyDecoder(true), true), "owned, pure-Java Snappy decoder, sequential");
        assertEquals(baseline, scan(ParquetScan.scan(file).useJavaSnappyDecoder(true).parallelism(3), true), "owned, pure-Java Snappy decoder, parallel");
        assertEquals(baseline, scan(ParquetScan.scan(file).batchSize(97), true), "owned, odd batch size straddling page boundaries");
    }

    @Test
    void projectionReadsOnlyTheRequestedColumnsAndStillMatches() throws IOException {
        Path file = write(CompressionCodecName.SNAPPY, WriterVersion.PARQUET_1_0, true, ROWS);
        // Non-adjacent columns: exercises one chunk read per run of adjacent chunks, not one for the whole row group.
        ParquetScan sel = ParquetScan.scan(file).select("id", "payload");
        assertEquals(scan(sel, false), scan(sel, true));
    }

    @Test
    void emptyAndTinyFilesWork() throws IOException {
        for (int rows : new int[]{0, 1, 2, 17}) {
            Path file = write(CompressionCodecName.SNAPPY, WriterVersion.PARQUET_2_0, true, rows);
            List<String> baseline = scan(ParquetScan.scan(file), false);
            assertEquals(rows, baseline.size());
            assertEquals(baseline, scan(ParquetScan.scan(file).useJavaSnappyDecoder(true), true), "rows=" + rows);
        }
    }

    @Test
    void theSwitchIsOffUnlessAskedFor() {
        String prev = System.getProperty(OwnedRowGroupReader.PROP_ENABLE);
        try {
            System.clearProperty(OwnedRowGroupReader.PROP_ENABLE);
            assertFalse(OwnedRowGroupReader.enabled(), "default is the stock parquet-java page path");
            System.setProperty(OwnedRowGroupReader.PROP_ENABLE, "true");
            assertTrue(OwnedRowGroupReader.enabled());
        } finally {
            if (prev == null) System.clearProperty(OwnedRowGroupReader.PROP_ENABLE);
            else System.setProperty(OwnedRowGroupReader.PROP_ENABLE, prev);
        }
    }
}
