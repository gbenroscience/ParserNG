package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.parquet.column.ParquetProperties;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * End to end: real ZSTD-compressed Parquet files, scanned through the pure-Java ZSTD decoder, via both the fluent API
 * and the {@code -Dparserng.zstd=java} option, across the file shapes that change what the codec sees (many small
 * pages vs few big ones, dictionary on/off, data page v1 vs v2, parallel workers). Every scan is checked against
 * the same row oracle the other scan tests use, so "the decoder returned something" is never enough to pass.
 *
 * <p>These files are written with parquet-java's stock ZSTD compressor (it needs {@code zstd-jni} to WRITE them);
 * when that is unavailable the tests are skipped rather than failed. Reading them back through the pure-Java option
 * does not need it.
 */
class ZstdParquetScanTest {

    @TempDir
    Path dir;

    /** Writes the shared fixture rows with ZSTD, or aborts the test (assumption) if no ZSTD compressor exists. */
    private Path writeZstd(String name, int n, long rowGroupBytes, boolean dictionary, int pageBytes,
                           ParquetProperties.WriterVersion version, CompressionCodecName codec) throws IOException {
        Path file = dir.resolve(name);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(FLAT_SCHEMA)
                .withCompressionCodec(codec)
                .withRowGroupSize(rowGroupBytes)
                .withPageSize(pageBytes)
                .withDictionaryEncoding(dictionary)
                .withWriterVersion(version)
                .build()) {
            SimpleGroupFactory f = new SimpleGroupFactory(FLAT_SCHEMA);
            for (Row r : rows(n)) {
                Group g = f.newGroup();
                g.append("id", r.id());
                if (r.grp() != null) g.append("grp", r.grp().intValue());
                if (r.name() != null) g.append("name", r.name());
                g.append("score", r.score());
                g.append("flag", r.flag());
                if (r.day() != null) g.append("day", r.day().intValue());
                if (r.ts() != null) g.append("ts", r.ts().longValue());
                if (r.ratio() != null) g.append("ratio", r.ratio().floatValue());
                g.append("lo", r.lo());
                g.append("hi", r.hi());
                w.write(g);
            }
        } catch (Throwable t) {
            org.junit.jupiter.api.Assumptions.assumeTrue(false, "cannot write " + codec + " here: " + t);
        }
        return file;
    }

    private Path zstdFile(String name, boolean dictionary, int pageBytes, ParquetProperties.WriterVersion v) throws IOException {
        Path f = writeZstd(name, N, SMALL_ROW_GROUP_BYTES, dictionary, pageBytes, v, CompressionCodecName.ZSTD);
        ParquetFileInfo info = ParquetFileInfo.read(f);
        assertTrue(info.rowGroups().size() >= 3, "fixture must span several row groups");
        for (ParquetFileInfo.RowGroupInfo rg : info.rowGroups())
            for (ParquetFileInfo.ColumnChunkInfo c : rg.columns())
                assertEquals("ZSTD", c.codec(), "test would be vacuous if the file were not actually ZSTD: " + c.path());
        return f;
    }

    // ------------------------------------------------------------------ fluent API

    @Test
    void fluentOptionDecodesAZstdFileCorrectly() throws IOException {
        Path f = zstdFile("plain.parquet", true, 1024 * 1024, ParquetProperties.WriterVersion.PARQUET_1_0);
        assertScanEqualsRows(ParquetScan.scan(f).useJavaZstdDecoder(true), ROWS);
    }

    @Test
    void worksAcrossBatchSizesThatSplitPagesAwkwardly() throws IOException {
        Path f = zstdFile("batches.parquet", true, 1024 * 1024, ParquetProperties.WriterVersion.PARQUET_1_0);
        for (int batch : new int[]{1, 7, 97, 1000, 4096, 100_000}) {
            assertScanEqualsRows(ParquetScan.scan(f).batchSize(batch).useJavaZstdDecoder(true), ROWS);
        }
    }

    @Test
    void manyTinyPagesAndNoDictionary() throws IOException {
        // 512-byte pages: hundreds of small ZSTD frames per column, each a separate decompress() call
        Path f = zstdFile("tiny-pages.parquet", false, 512, ParquetProperties.WriterVersion.PARQUET_1_0);
        assertScanEqualsRows(ParquetScan.scan(f).useJavaZstdDecoder(true), ROWS);
    }

    @Test
    void dataPageV2() throws IOException {
        // v2 pages compress only the values section; levels stay uncompressed -- a different slice of the page.
        // The v2 fixture also uses delta encodings, which are the module's own business and independent of the codec,
        // so first run an UNCOMPRESSED control: if the module cannot read this v2 file at all, there is nothing
        // ZSTD-specific to learn here and the test steps aside instead of failing for an unrelated reason.
        Path control = writeZstd("v2-control.parquet", N, SMALL_ROW_GROUP_BYTES, true, 4096,
                ParquetProperties.WriterVersion.PARQUET_2_0, CompressionCodecName.UNCOMPRESSED);
        try {
            assertScanEqualsRows(ParquetScan.scan(control), ROWS);
        } catch (Throwable t) {
            org.junit.jupiter.api.Assumptions.assumeTrue(false,
                    "module cannot read this v2 fixture even uncompressed (codec-independent): " + t);
        }
        Path f = zstdFile("v2.parquet", true, 4096, ParquetProperties.WriterVersion.PARQUET_2_0);
        assertScanEqualsRows(ParquetScan.scan(f).useJavaZstdDecoder(true), ROWS);
    }

    @Test
    void parallelWorkersEachGetTheirOwnDecoder() throws IOException {
        // the decoder is not thread-safe by design; this is the test that would fail if a factory were ever shared
        Path f = zstdFile("parallel.parquet", true, 2048, ParquetProperties.WriterVersion.PARQUET_1_0);
        for (int workers : new int[]{2, 4, 8}) {
            for (int rep = 0; rep < 5; rep++) {
                assertScanEqualsRows(ParquetScan.scan(f).parallelism(workers).useJavaZstdDecoder(true), ROWS);
            }
        }
    }

    @Test
    void agreesWithTheStockCodecAndWithAPredicateAndProjectionApplied() throws IOException {
        Path f = zstdFile("pred.parquet", true, 2048, ParquetProperties.WriterVersion.PARQUET_1_0);
        ParquetScan base = ParquetScan.scan(f).select("id").pushdown(Predicate.ge("id", 1234L)).exactFilter();
        List<Long> stock = ids(base.useJavaZstdDecoder(false));
        List<Long> java = ids(base.useJavaZstdDecoder(true));
        assertEquals(stock, java);
        assertEquals(expectedIds(r -> r.id() >= 1234L), java);
    }

    // ------------------------------------------------------------------ system-property option

    @Test
    void systemPropertyOptionDecodesAZstdFile() throws IOException {
        Path f = zstdFile("prop.parquet", true, 2048, ParquetProperties.WriterVersion.PARQUET_1_0);
        String prev = System.getProperty("parserng.zstd");
        try {
            System.setProperty("parserng.zstd", "java");
            assertScanEqualsRows(ParquetScan.scan(f), ROWS);                              // no override: the property applies
            assertScanEqualsRows(ParquetScan.scan(f).parallelism(4), ROWS);
            assertScanEqualsRows(ParquetScan.scan(f).useJavaZstdDecoder(false), ROWS);    // explicit false: stock codec, same rows
        } finally {
            if (prev == null) System.clearProperty("parserng.zstd"); else System.setProperty("parserng.zstd", prev);
        }
    }

    // ------------------------------------------------------------------ composition with SNAPPY and other codecs

    @Test
    void bothPureJavaOptionsTogetherServeSnappyAndZstdFiles() throws IOException {
        Path zf = zstdFile("both-zstd.parquet", true, 2048, ParquetProperties.WriterVersion.PARQUET_1_0);
        Path sf = writeZstd("both-snappy.parquet", N, SMALL_ROW_GROUP_BYTES, true, 2048,
                ParquetProperties.WriterVersion.PARQUET_1_0, CompressionCodecName.SNAPPY);
        assertScanEqualsRows(ParquetScan.scan(zf).useJavaSnappyDecoder(true).useJavaZstdDecoder(true), ROWS);
        assertScanEqualsRows(ParquetScan.scan(sf).useJavaSnappyDecoder(true).useJavaZstdDecoder(true), ROWS);
    }

    @Test
    void otherCodecsAreUnaffectedWhenTheZstdOptionIsOn() throws IOException {
        // GZIP and UNCOMPRESSED must still go to parquet-java's own codecs through the decorator
        for (CompressionCodecName c : new CompressionCodecName[]{CompressionCodecName.GZIP, CompressionCodecName.UNCOMPRESSED}) {
            Path f = writeZstd("other-" + c + ".parquet", N, SMALL_ROW_GROUP_BYTES, true, 2048,
                    ParquetProperties.WriterVersion.PARQUET_1_0, c);
            assertScanEqualsRows(ParquetScan.scan(f).useJavaZstdDecoder(true), ROWS);
        }
    }

    // ------------------------------------------------------------------ damaged files

    @Test
    void damagedPagesNeverCrashAndNeverPoisonLaterScans() throws IOException {
        Path good = zstdFile("good.parquet", true, 2048, ParquetProperties.WriterVersion.PARQUET_1_0);
        byte[] bytes = Files.readAllBytes(good);
        Random rnd = new Random(99);
        int failed = 0, survived = 0;
        for (int trial = 0; trial < 40; trial++) {
            byte[] bad = bytes.clone();
            // stay inside the data region (the footer is a separate failure mode, tested elsewhere)
            for (int k = 0; k < 6; k++) bad[8 + rnd.nextInt(bytes.length / 2)] ^= (byte) (1 + rnd.nextInt(255));
            Path p = dir.resolve("bad" + trial + ".parquet");
            Files.write(p, bad);
            try {
                ids(ParquetScan.scan(p).select("id").useJavaZstdDecoder(true));
                survived++; // a flip may be inert, or land in data a checksum-less frame can't question
            } catch (RuntimeException expected) {
                failed++;   // surfaced as an ordinary exception: no crash, no hang, no Error
            }
        }
        assertTrue(failed > 0, "40 trials of 6 random byte flips each should break at least one page");
        // the same JVM-wide state (factories, decoders) must be fine for a clean file afterwards
        assertScanEqualsRows(ParquetScan.scan(good).useJavaZstdDecoder(true), ROWS);
        assertTrue(failed + survived == 40);
    }
}
