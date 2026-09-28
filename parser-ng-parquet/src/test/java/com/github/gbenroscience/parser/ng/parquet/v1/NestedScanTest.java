package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Struct and LIST columns: repetition/definition levels must come back as Arrow list offsets and struct validity,
 * including the three distinct list states (null list, empty list, list containing a null element).
 */
class NestedScanTest {

    private static final int N = 300;

    private static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message nested {"
            + " required int64 id;"
            + " optional group tags (LIST) { repeated group list { optional binary element (STRING); } }"
            + " optional group point { required double x; required double y; }"
            + "}");

    @TempDir
    static Path dir;
    static Path file;

    /** Row i: tags null / empty / [a<i>, null?, b<i>] by i%4; point null when i%5==0. */
    @BeforeAll
    static void write() throws IOException {
        file = dir.resolve("nested.parquet");
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(SCHEMA).withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .withRowGroupSize(4 * 1024).build()) {
            for (int i = 0; i < N; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                switch (i % 4) {
                    case 0 -> { /* null list */ }
                    case 1 -> g.addGroup("tags");                                    // present but empty
                    case 2 -> { Group t = g.addGroup("tags"); t.addGroup("list").append("element", "a" + i); t.addGroup("list").append("element", "b" + i); }
                    default -> { Group t = g.addGroup("tags"); t.addGroup("list").append("element", "a" + i); t.addGroup("list"); /* null element */ }
                }
                if (i % 5 != 0) {
                    Group p = g.addGroup("point");
                    p.append("x", i * 1.0).append("y", i * 2.0);
                }
                w.write(g);
            }
        }
    }

    private static List<List<String>> expectedTags(int i) {
        return switch (i % 4) {
            case 0 -> null;
            case 1 -> List.of();
            case 2 -> List.of(List.of("a" + i, "b" + i)).get(0) == null ? null : List.<List<String>>of(List.of("a" + i, "b" + i));
            default -> null;
        };
    }

    @Test
    void schemaMapsToListAndStruct() {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(file).open(a)) {
            assertEquals(ArrowType.List.INSTANCE, r.schema().findField("tags").getType());
            assertInstanceOf(ArrowType.Struct.class, r.schema().findField("point").getType());
            assertEquals(2, r.schema().findField("point").getChildren().size());
        }
    }

    @Test
    void listStatesAndStructValidityRoundTrip() {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(file).batchSize(64).open(a)) {
            int seen = 0;
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                BigIntVector id = (BigIntVector) b.getVector("id");
                ListVector tags = (ListVector) b.getVector("tags");
                StructVector point = (StructVector) b.getVector("point");
                Float8Vector x = (Float8Vector) point.getChild("x");
                Float8Vector y = (Float8Vector) point.getChild("y");
                for (int k = 0; k < b.getRowCount(); k++, seen++) {
                    int i = (int) id.get(k);
                    assertEquals(seen, i, "row order");
                    switch (i % 4) {
                        case 0 -> assertTrue(tags.isNull(k), "row " + i + " tags should be null");
                        case 1 -> {
                            assertFalse(tags.isNull(k), "row " + i + " tags should be present");
                            assertEquals(0, tags.getObject(k).size(), "row " + i + " tags should be empty");
                        }
                        case 2 -> {
                            List<?> l = tags.getObject(k);
                            assertEquals(2, l.size(), "row " + i);
                            assertEquals("a" + i, String.valueOf(l.get(0)));
                            assertEquals("b" + i, String.valueOf(l.get(1)));
                        }
                        default -> {
                            List<?> l = tags.getObject(k);
                            assertEquals(2, l.size(), "row " + i);
                            assertEquals("a" + i, String.valueOf(l.get(0)));
                            assertNull(l.get(1), "row " + i + " second element is null");
                        }
                    }
                    if (i % 5 == 0) {
                        assertTrue(point.isNull(k), "row " + i + " point should be null");
                    } else {
                        assertFalse(point.isNull(k), "row " + i + " point should be set");
                        assertEquals(i * 1.0, x.get(k), 0.0);
                        assertEquals(i * 2.0, y.get(k), 0.0);
                    }
                }
            }
            assertEquals(N, seen);
        }
    }

    @Test
    void parallelScanOfNestedColumnsMatchesSequential() {
        List<Long> seq = allIds(ParquetScan.scan(file));
        List<Long> par = allIds(ParquetScan.scan(file).parallelism(3));
        assertEquals(seq, par);
        assertEquals(N, seq.size());
    }

    @Test
    void projectingOnlyAFlatColumnSkipsTheNestedOnes() {
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = ParquetScan.scan(file).select("id").open(a)) {
            assertEquals(1, r.schema().getFields().size());
        }
    }

    @Test
    void exactFilterRejectsNestedPredicateColumns() {
        try (BufferAllocator a = new RootAllocator()) {
            for (Predicate p : List.of(Predicate.isNull("tags"), Predicate.eq("point", 1.0d))) {
                ParquetScanException e = assertThrows(ParquetScanException.class,
                        () -> ParquetScan.scan(file).pushdown(p).exactFilter().open(a));
                assertNotNull(e.column());
            }
        }
    }

    @Test
    void exactFilterOnAFlatColumnStillWorksAlongsideNestedOnes() {
        assertEquals(List.of(7L, 8L, 9L), allIds(ParquetScan.scan(file)
                .pushdown(Predicate.and(Predicate.ge("id", 7L), Predicate.le("id", 9L))).exactFilter()));
    }

    @Test
    void customPredicateMayTouchANestedColumn() {
        ParquetScan s = ParquetScan.scan(file)
                .pushdown(Predicate.custom("hasTags", "tags"))
                .withCustomPredicate("hasTags", (batch, row) -> {
                    ListVector v = (ListVector) batch.getVector("tags");
                    return !v.isNull(row) && v.getObject(row).size() > 0;
                }).exactFilter();
        List<Long> got = allIds(s);
        List<Long> expected = new ArrayList<>();
        for (int i = 0; i < N; i++) if (i % 4 >= 2) expected.add((long) i);
        assertEquals(expected, got);
    }

    private static List<Long> allIds(ParquetScan scan) {
        List<Long> out = new ArrayList<>();
        try (BufferAllocator a = new RootAllocator(); ParquetBatchReader r = scan.open(a)) {
            while (r.next()) {
                BigIntVector v = (BigIntVector) r.root().getVector("id");
                for (int i = 0; i < r.root().getRowCount(); i++) out.add(v.get(i));
            }
        }
        return out;
    }
}