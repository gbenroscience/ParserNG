/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser.ng.parquet.util;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.LocalOutputFile;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.MessageTypeParser;
import org.apache.parquet.schema.Type;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Writes a Parquet file of random data for a caller-supplied schema, to disk,
 * for exercising {@code ParquetScan}/{@code ParquetBatchReader} without
 * hand-building fixtures. Reproducible via a seed; every knob is an immutable
 * {@link Config}.
 *
 * <pre>{@code
 * Path file = RandomParquetFiles.write(
 *         Path.of("/tmp/t.parquet"), RandomParquetFiles.SAMPLE_NESTED_SCHEMA, 200_000,
 *         RandomParquetFiles.Config.defaults().seed(42).nullProbability(0.1).rowGroupSize(1 << 20));
 * }</pre>
 *
 * <h2>Scope</h2>
 * Supports exactly the primitive/logical types {@code ColumnPlan} reads
 * (BOOLEAN, INT32 [+ DATE], INT64 [+ TIMESTAMP], FLOAT, DOUBLE, BINARY [+
 * STRING/ENUM/JSON]) — using anything else in the schema fails fast, naming the
 * column, rather than writing a file the reader couldn't handle anyway. Nested
 * types are supported for the standard layouts only: a struct (an
 * OPTIONAL/REQUIRED group with no LIST/ MAP annotation), a 3-level LIST (a
 * group with the LIST annotation wrapping exactly one REPEATED group which
 * itself wraps exactly one element field), a MAP (a group with the MAP
 * annotation wrapping exactly one REPEATED {@code key_value} group of a
 * primitive key and a value), and any nesting of these. The legacy 2-level LIST
 * layout and non-primitive MAP keys are out of scope; a mismatched shape fails
 * fast with a description of what was expected, so a malformed test schema is
 * easy to spot.
 *
 * <p>
 * An OPTIONAL field (at any depth, including a LIST's element) is independently
 * randomized to be present or null, each row/entry, at
 * {@link Config#nullProbability}; a REQUIRED field is always present. This
 * exercises exactly the null-handling paths
 * {@code ColumnPlan}/{@code LevelWalker} implement, including a
 * present-but-empty list vs. a null list vs. a list with a null element.
 *
 * <p>
 * Values are uniformly random within {@link Config}'s configured ranges — there
 * is no correlation between a column's value and its row index. That is fine
 * for correctness/shape testing but makes row-group statistics uninteresting
 * for pruning tests; for those, write a small dedicated fixture whose values
 * are a function of the row index instead (as the module's own tests do).
 *
 * @author GBEMIRO
 */
public final class RandomParquetFiles {

    private RandomParquetFiles() {
    }

    /**
     * A flat schema covering every supported primitive/logical type.
     */
    public static final MessageType SAMPLE_FLAT_SCHEMA = MessageTypeParser.parseMessageType(
            "message sample {"
            + "  required int64 id;"
            + "  optional double value;"
            + "  optional float ratio;"
            + "  required boolean flag;"
            + "  optional binary name (STRING);"
            + "  optional int32 day (DATE);"
            + "  optional int64 seen_at (TIMESTAMP(MILLIS,true));"
            + "}");

    /**
     * A LIST of strings and a nested struct, both nullable — covers the common
     * nested shapes.
     */
    public static final MessageType SAMPLE_NESTED_SCHEMA = MessageTypeParser.parseMessageType(
            "message sample {"
            + "  required int64 id;"
            + "  optional group tags (LIST) {"
            + "    repeated group list {"
            + "      optional binary element (STRING);"
            + "    }"
            + "  }"
            + "  optional group point {"
            + "    required double x;"
            + "    required double y;"
            + "  }"
            + "}");

    /**
     * Immutable generation settings. Start from {@link #defaults()} and adjust
     * what you need.
     */
    public static final class Config {

        final long seed;
        final double nullProbability;
        final int minCollectionSize, maxCollectionSize;
        final int minStringLength, maxStringLength;
        final long intMin, intMax;
        final long longMin, longMax;
        final double doubleMin, doubleMax;
        final CompressionCodecName codec;
        final long rowGroupSize;
        final boolean dictionaryEncoding;

        private Config(long seed, double nullProbability, int minCollectionSize, int maxCollectionSize,
                int minStringLength, int maxStringLength, long intMin, long intMax,
                long longMin, long longMax, double doubleMin, double doubleMax,
                CompressionCodecName codec, long rowGroupSize, boolean dictionaryEncoding) {
            this.seed = seed;
            this.nullProbability = nullProbability;
            this.minCollectionSize = minCollectionSize;
            this.maxCollectionSize = maxCollectionSize;
            this.minStringLength = minStringLength;
            this.maxStringLength = maxStringLength;
            this.intMin = intMin;
            this.intMax = intMax;
            this.longMin = longMin;
            this.longMax = longMax;
            this.doubleMin = doubleMin;
            this.doubleMax = doubleMax;
            this.codec = codec;
            this.rowGroupSize = rowGroupSize;
            this.dictionaryEncoding = dictionaryEncoding;
        }

        public static Config defaults() {
            return new Config(0L, 0.15, 0, 5, 3, 12, 0, 1_000_000, 0, 1_000_000_000L, 0.0, 1_000.0,
                    CompressionCodecName.SNAPPY, 64L * 1024, true);
        }

        public Config seed(long s) {
            return copy(s, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec, rowGroupSize, dictionaryEncoding);
        }

        public Config nullProbability(double p) {
            if (p < 0 || p > 1) {
                throw new IllegalArgumentException("nullProbability must be in [0,1], was " + p);
            }
            return copy(seed, p, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec, rowGroupSize, dictionaryEncoding);
        }

        /**
         * Size range for LIST/MAP entries and for a bare {@code repeated}
         * field, inclusive on both ends.
         */
        public Config collectionSize(int min, int max) {
            if (min < 0 || max < min) {
                throw new IllegalArgumentException("need 0 <= min <= max, got " + min + ".." + max);
            }
            return copy(seed, nullProbability, min, max, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec, rowGroupSize, dictionaryEncoding);
        }

        public Config stringLength(int min, int max) {
            if (min < 0 || max < min) {
                throw new IllegalArgumentException("need 0 <= min <= max, got " + min + ".." + max);
            }
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, min, max, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec, rowGroupSize, dictionaryEncoding);
        }

        /**
         * Range for plain (non-DATE) INT32 columns.
         */
        public Config intRange(long min, long max) {
            if (max < min) {
                throw new IllegalArgumentException("max < min");
            }
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, min, max, longMin, longMax, doubleMin, doubleMax, codec, rowGroupSize, dictionaryEncoding);
        }

        /**
         * Range for plain (non-TIMESTAMP) INT64 columns.
         */
        public Config longRange(long min, long max) {
            if (max < min) {
                throw new IllegalArgumentException("max < min");
            }
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, min, max, doubleMin, doubleMax, codec, rowGroupSize, dictionaryEncoding);
        }

        public Config doubleRange(double min, double max) {
            if (max < min) {
                throw new IllegalArgumentException("max < min");
            }
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, min, max, codec, rowGroupSize, dictionaryEncoding);
        }

        public Config compression(CompressionCodecName c) {
            if (c == null) {
                throw new NullPointerException("codec");
            }
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, c, rowGroupSize, dictionaryEncoding);
        }

        /**
         * Byte threshold parquet-java uses to decide when to start a new row
         * group.
         */
        public Config rowGroupSize(long bytes) {
            if (bytes <= 0) {
                throw new IllegalArgumentException("rowGroupSize must be positive");
            }
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec, bytes, dictionaryEncoding);
        }

        public Config dictionaryEncoding(boolean enabled) {
            return copy(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength, maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec, rowGroupSize, enabled);
        }

        private Config copy(long seed, double nullProbability, int minCollectionSize, int maxCollectionSize,
                int minStringLength, int maxStringLength, long intMin, long intMax,
                long longMin, long longMax, double doubleMin, double doubleMax,
                CompressionCodecName codec, long rowGroupSize, boolean dictionaryEncoding) {
            return new Config(seed, nullProbability, minCollectionSize, maxCollectionSize, minStringLength,
                    maxStringLength, intMin, intMax, longMin, longMax, doubleMin, doubleMax, codec,
                    rowGroupSize, dictionaryEncoding);
        }
    }

    /**
     * {@link #write(Path, MessageType, int, Config)} with
     * {@link Config#defaults()}.
     */
    public static Path write(Path file, MessageType schema, int rowCount) throws IOException {
        return write(file, schema, rowCount, Config.defaults());
    }

    /**
     * Writes {@code rowCount} randomly generated records conforming to
     * {@code schema} to {@code file}, overwriting it if it already exists.
     *
     * @return {@code file}, for chaining
     * @throws IllegalArgumentException if {@code schema} uses a shape or
     * primitive type this utility does not generate data for (see class
     * javadoc's "Scope" section)
     */
    public static Path write(Path file, MessageType schema, int rowCount, Config cfg) throws IOException {
        if (rowCount < 0) {
            throw new IllegalArgumentException("rowCount must be >= 0, was " + rowCount);
        }
        Files.deleteIfExists(file);
        Random rnd = new Random(cfg.seed);
        SimpleGroupFactory factory = new SimpleGroupFactory(schema);
        try (ParquetWriter<Group> w = ExampleParquetWriter.builder(new LocalOutputFile(file))
                .withType(schema)
                .withCompressionCodec(cfg.codec)
                .withRowGroupSize(cfg.rowGroupSize)
                .withDictionaryEncoding(cfg.dictionaryEncoding)
                .build()) {
            for (int i = 0; i < rowCount; i++) {
                Group g = factory.newGroup();
                fillGroup(g, schema, rnd, cfg);
                w.write(g);
            }
        }
        return file;
    }

    // ------------------------------------------------------------------ recursive fill
    private static void fillGroup(Group g, GroupType type, Random rnd, Config cfg) {
        for (int i = 0; i < type.getFieldCount(); i++) {
            fillField(g, type.getType(i), rnd, cfg);
        }
    }

    /**
     * Populates one field of {@code g}: a bare repeated field, an
     * optional/required leaf, or a group.
     */
    private static void fillField(Group g, Type field, Random rnd, Config cfg) {
        if (field.isRepetition(Type.Repetition.REPEATED)) {
            int n = randomSize(rnd, cfg);
            for (int k = 0; k < n; k++) {
                if (field.isPrimitive()) {
                    appendValue(g, field, rnd, cfg);
                } else {
                    fillGroup(g.addGroup(field.getName()), field.asGroupType(), rnd, cfg);
                }
            }
            return;
        }
        boolean optional = field.isRepetition(Type.Repetition.OPTIONAL);
        if (optional && rnd.nextDouble() < cfg.nullProbability) {
            return; // leave the field unset: null
        }
        if (field.isPrimitive()) {
            appendValue(g, field, rnd, cfg);
            return;
        }
        GroupType gt = field.asGroupType();
        LogicalTypeAnnotation lt = gt.getLogicalTypeAnnotation();
        if (lt instanceof LogicalTypeAnnotation.ListLogicalTypeAnnotation) {
            fillList(g, gt, rnd, cfg);
        } else if (lt instanceof LogicalTypeAnnotation.MapLogicalTypeAnnotation
                || lt instanceof LogicalTypeAnnotation.MapKeyValueTypeAnnotation) {
            fillMap(g, gt, rnd, cfg);
        } else {
            fillGroup(g.addGroup(gt.getName()), gt, rnd, cfg); // plain struct
        }
    }

    /**
     * Standard 3-level LIST only: {@code LIST { repeated group list { element }
     * } } — see class javadoc.
     */
    private static void fillList(Group g, GroupType listType, Random rnd, Config cfg) {
        if (listType.getFieldCount() != 1) {
            throw new IllegalArgumentException("RandomParquetFiles supports only the standard 3-level LIST "
                    + "layout; '" + listType.getName() + "' has " + listType.getFieldCount() + " fields");
        }
        Type wrapperField = listType.getType(0);
        if (!wrapperField.isRepetition(Type.Repetition.REPEATED) || wrapperField.isPrimitive()
                || wrapperField.asGroupType().getFieldCount() != 1) {
            throw new IllegalArgumentException("RandomParquetFiles supports only the standard 3-level LIST "
                    + "layout (a REPEATED group wrapping exactly one element field); '" + listType.getName()
                    + "' does not match");
        }
        GroupType wrapper = wrapperField.asGroupType();
        Type elementField = wrapper.getType(0);
        Group listGroup = g.addGroup(listType.getName());
        int n = randomSize(rnd, cfg);
        for (int k = 0; k < n; k++) {
            fillField(listGroup.addGroup(wrapper.getName()), elementField, rnd, cfg);
        }
    }

    /**
     * Standard MAP only: {@code MAP { repeated group key_value { key; value } }
     * }, primitive keys only.
     */
    private static void fillMap(Group g, GroupType mapType, Random rnd, Config cfg) {
        if (mapType.getFieldCount() != 1) {
            throw new IllegalArgumentException("RandomParquetFiles supports only the standard MAP layout; '"
                    + mapType.getName() + "' has " + mapType.getFieldCount() + " fields");
        }
        Type kvField = mapType.getType(0);
        if (!kvField.isRepetition(Type.Repetition.REPEATED) || kvField.isPrimitive()
                || kvField.asGroupType().getFieldCount() != 2) {
            throw new IllegalArgumentException("RandomParquetFiles supports only the standard MAP layout "
                    + "(a REPEATED key_value group of a key and a value); '" + mapType.getName() + "' does not match");
        }
        GroupType keyValue = kvField.asGroupType();
        Type keyField = keyValue.getType(0);
        Type valueField = keyValue.getType(1);
        if (!keyField.isPrimitive()) {
            throw new IllegalArgumentException("RandomParquetFiles requires a primitive MAP key; '"
                    + mapType.getName() + "' has a non-primitive key");
        }
        Group mapGroup = g.addGroup(mapType.getName());
        int n = randomSize(rnd, cfg);
        for (int k = 0; k < n; k++) {
            Group entry = mapGroup.addGroup(keyValue.getName());
            appendValue(entry, keyField, rnd, cfg); // keys are always required: never randomized to null
            fillField(entry, valueField, rnd, cfg);
        }
    }

    // ------------------------------------------------------------------ leaf values
    private static void appendValue(Group g, Type field, Random rnd, Config cfg) {
        String name = field.getName();
        org.apache.parquet.schema.PrimitiveType pt = field.asPrimitiveType();
        LogicalTypeAnnotation lt = pt.getLogicalTypeAnnotation();
        switch (pt.getPrimitiveTypeName()) {
            case BOOLEAN:
                g.append(name, rnd.nextBoolean());
                break;
            case INT32:
                if (lt instanceof LogicalTypeAnnotation.DateLogicalTypeAnnotation) {
                    g.append(name, (int) randomLong(rnd, 0, 40_000)); // arbitrary ~109-year day-count range
                } else {
                    g.append(name, (int) randomLong(rnd, cfg.intMin, cfg.intMax));
                }
                break;
            case INT64:
                if (lt instanceof LogicalTypeAnnotation.TimestampLogicalTypeAnnotation) {
                    g.append(name, randomLong(rnd, 1_600_000_000_000L, 1_800_000_000_000L)); // arbitrary ms range
                } else {
                    g.append(name, randomLong(rnd, cfg.longMin, cfg.longMax));
                }
                break;
            case FLOAT:
                g.append(name, (float) (cfg.doubleMin + rnd.nextDouble() * (cfg.doubleMax - cfg.doubleMin)));
                break;
            case DOUBLE:
                g.append(name, cfg.doubleMin + rnd.nextDouble() * (cfg.doubleMax - cfg.doubleMin));
                break;
            case BINARY:
                // Group.append(String, String) works for BINARY regardless of STRING/ENUM/JSON annotation.
                g.append(name, randomString(rnd, cfg.minStringLength, cfg.maxStringLength));
                break;
            default:
                throw new IllegalArgumentException("RandomParquetFiles does not generate data for primitive type "
                        + pt.getPrimitiveTypeName() + " (column '" + name + "'); supported: BOOLEAN, INT32, "
                        + "INT64, FLOAT, DOUBLE, BINARY — matching ColumnPlan's supported set");
        }
    }

    private static int randomSize(Random rnd, Config cfg) {
        return cfg.minCollectionSize + rnd.nextInt(cfg.maxCollectionSize - cfg.minCollectionSize + 1);
    }

    private static long randomLong(Random rnd, long min, long max) {
        return min + (long) (rnd.nextDouble() * (max - min));
    }

    private static String randomString(Random rnd, int minLen, int maxLen) {
        int len = minLen + rnd.nextInt(maxLen - minLen + 1);
        String alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        StringBuilder sb = new StringBuilder(len);
        for (int i = 0; i < len; i++) {
            sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------ CLI convenience
    /**
     * {@code java ... RandomParquetFiles <outputFile> <rowCount> [seed]} —
     * writes {@link #SAMPLE_NESTED_SCHEMA}.
     *
     * @param args
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: RandomParquetFiles <outputFile> <rowCount> [seed]");
            System.exit(1);
            return;
        }
        Path file = Path.of(args[0]);
        int rowCount = Integer.parseInt(args[1]);
        Config cfg = args.length >= 3 ? Config.defaults().seed(Long.parseLong(args[2])) : Config.defaults();
        write(file, SAMPLE_NESTED_SCHEMA, rowCount, cfg);
        System.out.println("Wrote " + rowCount + " random rows to " + file.toAbsolutePath());
    }
}
