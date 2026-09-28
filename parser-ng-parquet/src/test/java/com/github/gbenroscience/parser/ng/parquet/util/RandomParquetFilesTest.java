package com.github.gbenroscience.parser.ng.parquet.util;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RandomParquetFilesTest {

    @TempDir
    Path dir;

    private static List<Long> longs(Path f, String col) {
        List<Long> out = new ArrayList<>();
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(f).select(col).open(a)) {
            while (r.next()) {
                BigIntVector v = (BigIntVector) r.root().getVector(col);
                for (int i = 0; i < r.root().getRowCount(); i++) out.add(v.get(i));
            }
        }
        return out;
    }

    @Test
    void writesTheRequestedNumberOfRowsAndReturnsTheFile() throws IOException {
        Path f = dir.resolve("a.parquet");
        assertEquals(f, RandomParquetFiles.write(f, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 1_234));
        assertEquals(1_234, ParquetFileInfo.read(f).rowCount());
    }

    @Test
    void sameSeedGivesTheSameDataAndDifferentSeedsDiffer() throws IOException {
        var cfg = RandomParquetFiles.Config.defaults();
        Path a = RandomParquetFiles.write(dir.resolve("a.parquet"), RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 500, cfg.seed(7));
        Path b = RandomParquetFiles.write(dir.resolve("b.parquet"), RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 500, cfg.seed(7));
        Path c = RandomParquetFiles.write(dir.resolve("c.parquet"), RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 500, cfg.seed(8));
        assertEquals(longs(a, "id"), longs(b, "id"));
        assertNotEquals(longs(a, "id"), longs(c, "id"));
    }

    @Test
    void overwritesAnExistingFile() throws IOException {
        Path f = dir.resolve("o.parquet");
        RandomParquetFiles.write(f, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 100);
        RandomParquetFiles.write(f, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 40);
        assertEquals(40, ParquetFileInfo.read(f).rowCount());
    }

    @Test
    void zeroRowsIsAllowedNegativeIsNot() throws IOException {
        Path f = dir.resolve("z.parquet");
        RandomParquetFiles.write(f, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 0);
        assertEquals(0, ParquetFileInfo.read(f).rowCount());
        assertThrows(IllegalArgumentException.class, () -> RandomParquetFiles.write(f, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, -1));
    }

    @Test
    void configuredRangesAreHonoured() throws IOException {
        var cfg = RandomParquetFiles.Config.defaults().seed(3).longRange(100, 200).doubleRange(1.0, 2.0).nullProbability(0.0);
        Path f = RandomParquetFiles.write(dir.resolve("r.parquet"), RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 400, cfg);
        for (long v : longs(f, "id")) assertTrue(v >= 100 && v <= 200, "id=" + v);
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(f).select("value").open(a)) {
            while (r.next()) {
                Float8Vector v = (Float8Vector) r.root().getVector("value");
                for (int i = 0; i < r.root().getRowCount(); i++) {
                    assertFalse(v.isNull(i), "nullProbability(0) must produce no nulls");
                    assertTrue(v.get(i) >= 1.0 && v.get(i) <= 2.0, "value=" + v.get(i));
                }
            }
        }
    }

    @Test
    void nullProbabilityOneNullsEveryOptionalFieldButNeverARequiredOne() throws IOException {
        var cfg = RandomParquetFiles.Config.defaults().seed(1).nullProbability(1.0);
        Path f = RandomParquetFiles.write(dir.resolve("n.parquet"), RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 200, cfg);
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(f).select("id", "value").open(a)) {
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                for (int i = 0; i < b.getRowCount(); i++) {
                    assertFalse(b.getVector("id").isNull(i), "REQUIRED field is always present");
                    assertTrue(b.getVector("value").isNull(i), "OPTIONAL field with p=1 is always null");
                }
            }
        }
    }

    @Test
    void smallRowGroupSizeProducesManyRowGroups() throws IOException {
        var cfg = RandomParquetFiles.Config.defaults().seed(2).rowGroupSize(4 * 1024).compression(CompressionCodecName.UNCOMPRESSED);
        Path f = RandomParquetFiles.write(dir.resolve("g.parquet"), RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 5_000, cfg);
        assertTrue(ParquetFileInfo.read(f).rowGroups().size() > 1);
    }

    @Test
    void nestedSchemaRespectsCollectionSize() throws IOException {
        var cfg = RandomParquetFiles.Config.defaults().seed(5).nullProbability(0.0).collectionSize(3, 3);
        Path f = RandomParquetFiles.write(dir.resolve("l.parquet"), RandomParquetFiles.SAMPLE_NESTED_SCHEMA, 100, cfg);
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(f).select("tags").open(a)) {
            while (r.next()) {
                var tags = (org.apache.arrow.vector.complex.ListVector) r.root().getVector("tags");
                for (int i = 0; i < r.root().getRowCount(); i++) assertEquals(3, tags.getObject(i).size());
            }
        }
    }

    @Test
    void configValidation() {
        var d = RandomParquetFiles.Config.defaults();
        assertThrows(IllegalArgumentException.class, () -> d.nullProbability(-0.1));
        assertThrows(IllegalArgumentException.class, () -> d.nullProbability(1.1));
        assertThrows(IllegalArgumentException.class, () -> d.collectionSize(-1, 2));
        assertThrows(IllegalArgumentException.class, () -> d.collectionSize(5, 2));
        assertThrows(IllegalArgumentException.class, () -> d.stringLength(4, 2));
        assertThrows(IllegalArgumentException.class, () -> d.intRange(5, 1));
        assertThrows(IllegalArgumentException.class, () -> d.longRange(5, 1));
        assertThrows(IllegalArgumentException.class, () -> d.doubleRange(5, 1));
        assertThrows(IllegalArgumentException.class, () -> d.rowGroupSize(0));
        assertThrows(NullPointerException.class, () -> d.compression(null));
    }

    @Test
    void unsupportedPrimitiveTypeFailsFastNamingTheColumn() {
        MessageType bad = MessageTypeParser.parseMessageType("message m { required int96 legacy_ts; }");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> RandomParquetFiles.write(dir.resolve("bad.parquet"), bad, 1));
        assertTrue(e.getMessage().contains("legacy_ts"), e.getMessage());
    }

    @Test
    void int32ColumnsUseTheIntRange() throws IOException {
        MessageType s = MessageTypeParser.parseMessageType("message m { required int32 n; }");
        Path f = RandomParquetFiles.write(dir.resolve("i.parquet"), s, 300, RandomParquetFiles.Config.defaults().intRange(10, 20));
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(f).open(a)) {
            while (r.next()) {
                IntVector v = (IntVector) r.root().getVector("n");
                for (int i = 0; i < r.root().getRowCount(); i++) assertTrue(v.get(i) >= 10 && v.get(i) <= 20, "n=" + v.get(i));
            }
        }
    }
}