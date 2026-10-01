package com.github.gbenroscience.sqlv1.parquet;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;

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
