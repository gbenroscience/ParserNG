package com.github.gbenroscience.sqlv1.parquet;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Types;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Path;

/**
 * Deterministic Parquet files for {@link ParquetSqlTest}, written with parquet-java's example
 * writer (the same idiom as parser-ng-parquet's own test fixtures). Small row groups make one
 * file span many groups, and {@code id} is ascending so pruning on it is measurable.
 *
 * <p>Row {@code i} (0-based) has: {@code id=i}; {@code i32=i%100} (null when i%10==0);
 * {@code f32=i*0.5}; {@code f64=i*1.5}; {@code flag=i%3==0} (null when i%7==0);
 * {@code name="n"+i} (null when i%11==0); {@code day=19000+i%365};
 * {@code ts=1_700_000_000_000_000+i*1_000_000} micros (UTC); {@code dec32=i*100+5} unscaled, scale 2
 * (value {@code i + 0.05}); {@code dec64=i*10000} unscaled, scale 4 (value {@code i}); {@code decf=i} unscaled,
 * scale 3, precision 20 (value {@code i/1000}, backed by FIXED_LEN_BYTE_ARRAY).
 */
final class ParquetFixtures {

    private ParquetFixtures() { }

    static final MessageType SCHEMA = MessageTypeParser.parseMessageType(
            "message t {\n"
            + "  required int64 id;\n"
            + "  optional int32 i32;\n"
            + "  optional float f32;\n"
            + "  optional double f64;\n"
            + "  optional boolean flag;\n"
            + "  optional binary name (STRING);\n"
            + "  optional int32 day (DATE);\n"
            + "  optional int64 ts (TIMESTAMP(MICROS,true));\n"
            + "  optional int32 dec32 (DECIMAL(9,2));\n"
            + "  optional int64 dec64 (DECIMAL(18,4));\n"
            + "  optional fixed_len_byte_array(9) decf (DECIMAL(20,3));\n"
            + "}");

    static final long TS_BASE = 1_700_000_000_000_000L;

    static void writeMain(Path file, int rows) throws IOException {
        SimpleGroupFactory f = new SimpleGroupFactory(SCHEMA);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(SCHEMA)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED)
                .withRowGroupSize(16 * 1024)
                .withPageSize(4 * 1024)
                .withDictionaryEncoding(false)
                .build()) {
            for (int i = 0; i < rows; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                if (i % 10 != 0) g.append("i32", i % 100);
                g.append("f32", i * 0.5f);
                g.append("f64", i * 1.5);
                if (i % 7 != 0) g.append("flag", i % 3 == 0);
                if (i % 11 != 0) g.append("name", "n" + i);
                g.append("day", 19000 + i % 365);
                g.append("ts", TS_BASE + i * 1_000_000L);
                g.append("dec32", i * 100 + 5);
                g.append("dec64", i * 10_000L);
                g.append("decf", Binary.fromConstantByteArray(signed(BigInteger.valueOf(i), 9)));
                w.write(g);
            }
        }
    }

    /**
     * Columns whose names are not plain identifiers, for the quoted-identifier tests, over several row groups:
     * {@code id} int64 = i, {@code "my col"} int32 = 2i, {@code "1st"} string = "v" + i, {@code "select"} int64 = 10i,
     * {@code "a-b"} int32 = i % 7.
     */
    static void writeAwkward(Path file, int rows) throws IOException {
        MessageType t = Types.buildMessage()
                .required(PrimitiveTypeName.INT64).named("id")
                .required(PrimitiveTypeName.INT32).named("my col")
                .required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named("1st")
                .required(PrimitiveTypeName.INT64).named("select")
                .required(PrimitiveTypeName.INT32).named("a-b")
                .named("awkward");
        SimpleGroupFactory f = new SimpleGroupFactory(t);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file)).withType(t)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED).withRowGroupSize(16 * 1024).build()) {
            for (int i = 0; i < rows; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                g.append("my col", 2 * i);
                g.append("1st", "v" + i);
                g.append("select", 10L * i);
                g.append("a-b", i % 7);
                w.write(g);
            }
        }
    }

    private static final MessageType NESTED = MessageTypeParser.parseMessageType(
            "message nested {\n"
            + "  required int64 id;\n"
            + "  optional group s {\n"
            + "    optional int32 n;\n"
            + "    optional binary label (UTF8);\n"
            + "    optional group inner { optional int32 deep; }\n"
            + "  }\n"
            + "  optional group nums (LIST) { repeated group list { optional int32 element; } }\n"
            + "  optional group m (MAP) { repeated group key_value { required binary key (UTF8); optional int32 value; } }\n"
            + "  optional group items (LIST) { repeated group list { optional group element {\n"
            + "    optional int32 price; optional binary name (UTF8); } } }\n"
            + "}\n");

    /**
     * Nested columns, for the nested-access tests. Row i: {@code s} is null when i%5==0, else
     * {@code s.n = i} (null when i%9==0), {@code s.label = "L"+i} (null when i%8==0), {@code s.inner.deep = 2i}
     * (inner null when i%3==0); {@code nums} is null when i%7==0, empty when i%4==0, else (i%3)+1 elements
     * {@code i*10+k}; {@code m} is null when i%6==0, else {@code {"a": i, "b": 2i}} plus {@code "c": 7} when i is
     * even; {@code items} is null when i%10==0, else two structs {@code (price=i, name="x"+i)} and
     * {@code (price=i+1, name="y"+i)}.
     */
    static void writeNested(Path file, int rows) throws IOException {
        SimpleGroupFactory f = new SimpleGroupFactory(NESTED);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file)).withType(NESTED)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED).withDictionaryEncoding(false).build()) {
            for (int i = 0; i < rows; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                if (i % 5 != 0) {
                    Group s = g.addGroup("s");
                    if (i % 9 != 0) s.append("n", i);
                    if (i % 8 != 0) s.append("label", "L" + i);
                    if (i % 3 != 0) s.addGroup("inner").append("deep", 2 * i);
                }
                if (i % 7 != 0) {
                    Group nums = g.addGroup("nums");
                    if (i % 4 != 0) for (int k = 0; k <= i % 3; k++) nums.addGroup("list").append("element", i * 10 + k);
                }
                if (i % 6 != 0) {
                    Group m = g.addGroup("m");
                    m.addGroup("key_value").append("key", "a").append("value", i);
                    m.addGroup("key_value").append("key", "b").append("value", 2 * i);
                    if (i % 2 == 0) m.addGroup("key_value").append("key", "c").append("value", 7);
                }
                if (i % 10 != 0) {
                    Group items = g.addGroup("items");
                    Group e1 = items.addGroup("list").addGroup("element");
                    e1.append("price", i).append("name", "x" + i);
                    Group e2 = items.addGroup("list").addGroup("element");
                    e2.append("price", i + 1).append("name", "y" + i);
                }
                w.write(g);
            }
        }
    }

    private static final MessageType STRINGS = MessageTypeParser.parseMessageType(
            "message strings { required int64 id; optional binary v (STRING); }");

    /** {@code id} = position, {@code v} = the given text (a null leaves it null), for the string-to-number cast tests. */
    static void writeStrings(Path file, String... values) throws IOException {
        SimpleGroupFactory f = new SimpleGroupFactory(STRINGS);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file)).withType(STRINGS)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED).withDictionaryEncoding(false).build()) {
            for (int i = 0; i < values.length; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                if (values[i] != null) g.append("v", values[i]);
                w.write(g);
            }
        }
    }

    /** One int64 column holding a value above 2^53, for the precision-policy tests. */
    static void writeBigInt64(Path file) throws IOException {
        MessageType t = MessageTypeParser.parseMessageType("message b { required int64 id; required int64 big; }");
        SimpleGroupFactory f = new SimpleGroupFactory(t);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file)).withType(t)
                .withCompressionCodec(CompressionCodecName.UNCOMPRESSED).build()) {
            for (int i = 0; i < 3; i++) {
                Group g = f.newGroup();
                g.append("id", (long) i);
                g.append("big", i == 2 ? 9_007_199_254_740_993L : 100L + i);
                w.write(g);
            }
        }
    }

    /** Big-endian two's complement of {@code v}, sign-extended to exactly {@code len} bytes. */
    static byte[] signed(BigInteger v, int len) {
        byte[] raw = v.toByteArray();
        byte[] out = new byte[len];
        byte pad = (byte) (v.signum() < 0 ? 0xFF : 0x00);
        java.util.Arrays.fill(out, pad);
        System.arraycopy(raw, 0, out, len - raw.length, raw.length);
        return out;
    }
}
