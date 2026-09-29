package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
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
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for BOOLEAN columns in Data Page V2, where the Parquet spec uses the RLE value encoding
 * (length-prefixed, bit width 1) instead of PLAIN bit-packing. Previously such files failed with
 * {@code Unsupported value encoding RLE}.
 */
class ParquetV2BooleanTest {

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message v2bool {"
            + " required int64 id;"
            + " required boolean req;"
            + " optional boolean opt;"
            + "}");

    @TempDir
    Path dir;

    // Deterministic patterns that exercise both RLE run kinds: long runs (repeated) and short alternations (bit-packed).
    private static boolean req(int i) { return (i / 300) % 2 == 0 ? (i % 7 != 0) : (i % 3 == 0); }
    private static Boolean opt(int i) { return i % 5 == 0 ? null : Boolean.valueOf((i / 50) % 2 == 0 || i % 11 == 0); }

    private Path write(WriterVersion version, int rows, int pageBytes) throws IOException {
        Path file = dir.resolve("bool-" + version + "-" + rows + "-" + pageBytes + ".parquet");
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(SCHEMA)
                .withWriterVersion(version)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .withDictionaryEncoding(false)
                .withPageSize(pageBytes)
                .build()) {
            for (int i = 0; i < rows; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                g.append("req", req(i));
                Boolean o = opt(i);
                if (o != null) g.append("opt", o);
                w.write(g);
            }
        }
        return file;
    }

    private static void assertBooleansRoundTrip(Path file, int rows) {
        try (BufferAllocator alloc = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(file).open(alloc)) {
            int idx = 0;
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                BigIntVector id = (BigIntVector) b.getVector("id");
                BitVector req = (BitVector) b.getVector("req");
                BitVector opt = (BitVector) b.getVector("opt");
                for (int i = 0; i < b.getRowCount(); i++, idx++) {
                    assertEquals(idx, id.get(i), "row order at " + idx);
                    assertEquals(req(idx) ? 1 : 0, req.get(i), "req at row " + idx);
                    Boolean expected = opt(idx);
                    if (expected == null) {
                        assertTrue(opt.isNull(i), "opt should be null at row " + idx);
                    } else {
                        assertFalse(opt.isNull(i), "opt should be present at row " + idx);
                        assertEquals(expected ? 1 : 0, opt.get(i), "opt at row " + idx);
                    }
                }
            }
            assertEquals(rows, idx);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 7, 1_000, 20_000})
    void v2PagesDecodeRequiredAndNullableBooleans(int rows) throws IOException {
        assertBooleansRoundTrip(write(WriterVersion.PARQUET_2_0, rows, 1024 * 1024), rows);
    }

    @ParameterizedTest
    @ValueSource(ints = {256, 2048})
    void v2BooleansSpanningManySmallPagesDecode(int pageBytes) throws IOException {
        int rows = 20_000;
        assertBooleansRoundTrip(write(WriterVersion.PARQUET_2_0, rows, pageBytes), rows);
    }

    @org.junit.jupiter.api.Test
    void v1BooleansStillDecode() throws IOException {
        int rows = 5_000;
        assertBooleansRoundTrip(write(WriterVersion.PARQUET_1_0, rows, 1024 * 1024), rows);
    }
}
