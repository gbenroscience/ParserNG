package com.github.gbenroscience.sqlv1.demo;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;

import java.io.IOException;
import java.nio.file.Path;
import java.util.SplittableRandom;

/**
 * The benchmark dataset: deterministic (fixed seed), written once before any timing. The columns are chosen so the
 * workloads can tell the code paths apart:
 * <ul>
 * <li>{@code id}: int64, strictly increasing, so a range predicate prunes row groups by statistics.</li>
 * <li>{@code value}: double in [0,1), 10% null. {@code qty}: int32 in [0,100), 5% null.</li>
 * <li>{@code name}: string, 1000 distinct values {@code name-0000..name-0999}, 5% null (dictionary friendly,
 * low-cardinality group key).</li>
 * <li>{@code day}: date, 1500 distinct days from 2022-12-08 (epoch day 19000). {@code flag}: boolean, 20% null.</li>
 * </ul>
 */
final class BenchData {

    private BenchData() { }

    static final String SCHEMA_TEXT = "message bench {"
            + "  required int64 id;"
            + "  optional double value;"
            + "  optional int32 qty;"
            + "  optional binary name (STRING);"
            + "  optional int32 day (DATE);"
            + "  optional boolean flag;"
            + "}";

    static final MessageType SCHEMA = MessageTypeParser.parseMessageType(SCHEMA_TEXT);

    /** The text of the name of row-group-independent distinct value {@code k}. */
    static String nameOf(int k) {
        return String.format("name-%04d", k);
    }

    static void write(Path file, int rows, long rowGroupBytes, CompressionCodecName codec) throws IOException {
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        SplittableRandom r = new SplittableRandom(42);
        String[] names = new String[1000];
        for (int k = 0; k < names.length; k++) names[k] = nameOf(k);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file)).withType(SCHEMA)
                .withCompressionCodec(codec).withRowGroupSize(rowGroupBytes).build()) {
            for (int i = 0; i < rows; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                if (r.nextInt(10) != 0) g.append("value", r.nextDouble());
                if (r.nextInt(20) != 0) g.append("qty", r.nextInt(100));
                if (r.nextInt(20) != 0) g.append("name", names[r.nextInt(names.length)]);
                g.append("day", 19000 + r.nextInt(1500));
                if (r.nextInt(5) != 0) g.append("flag", r.nextBoolean());
                w.write(g);
            }
        }
    }
}
