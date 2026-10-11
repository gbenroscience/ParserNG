package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.PanamaSnappyCodecFactory;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.PureJavaZstdCodecFactory;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.filter2.compat.FilterCompat;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ParquetScan#useJavaZstdDecoder} is a per-scan override of the {@code -Dparserng.zstd=java} system
 * property, read by {@link ParquetBatchReader#newReadOptions} -- the same shape as the SNAPPY option (see
 * {@link UseJavaSnappyDecoderTest}), and tested the same way: straight at that method rather than through a full
 * scan, because which {@link CompressionCodecFactory} class it installs is the entire effect of the option.
 * (That the installed factory then decodes real pages correctly is covered end to end by
 * {@code ZstdParquetScanTest}.)
 *
 * <p>What is added over the Snappy test: the two options must compose independently (both are decorators over the
 * same base factory), and every {@link ParquetScan} builder method must carry the override forward -- the private
 * constructor is called from many places, and a single call site that forgot the new argument would silently turn
 * the option off for scans built through that method.
 */
class UseJavaZstdDecoderTest {

    private static final String ZSTD_PROP = "parserng.zstd";
    private static final String SNAPPY_PROP = "parserng.snappy";

    // ------------------------------------------------------------------ override vs system property

    @Test
    void nullOverrideDefersToSystemProperty() {
        String prev = System.getProperty(ZSTD_PROP);
        try {
            System.clearProperty(ZSTD_PROP);
            assertFalse(usesJavaZstd(null, null), "property unset, no override -> stock codec");

            System.setProperty(ZSTD_PROP, "java");
            assertTrue(usesJavaZstd(null, null), "property set, no override -> Java decoder");
        } finally {
            restore(ZSTD_PROP, prev);
        }
    }

    @Test
    void explicitTrueWinsRegardlessOfProperty() {
        String prev = System.getProperty(ZSTD_PROP);
        try {
            System.clearProperty(ZSTD_PROP);
            assertTrue(usesJavaZstd(null, Boolean.TRUE), "override=true, property unset -> Java decoder");

            System.setProperty(ZSTD_PROP, "java");
            assertTrue(usesJavaZstd(null, Boolean.TRUE), "override=true, property set -> still Java decoder");
        } finally {
            restore(ZSTD_PROP, prev);
        }
    }

    @Test
    void explicitFalseWinsRegardlessOfProperty() {
        String prev = System.getProperty(ZSTD_PROP);
        try {
            System.clearProperty(ZSTD_PROP);
            assertFalse(usesJavaZstd(null, Boolean.FALSE), "override=false, property unset -> stock codec");

            System.setProperty(ZSTD_PROP, "java");
            assertFalse(usesJavaZstd(null, Boolean.FALSE), "override=false, property set -> still stock codec");
        } finally {
            restore(ZSTD_PROP, prev);
        }
    }

    @Test
    void onlyTheExactValueJavaEnablesTheProperty() {
        // Same strictness as -Dparserng.snappy=panama: a typo or a boolean-looking value must NOT silently switch
        // the decoder, because the failure mode of a wrong guess is a different code path in production.
        String prev = System.getProperty(ZSTD_PROP);
        try {
            for (String v : new String[]{"", "true", "1", "JAVA", "Java", "native", "panama", " java"}) {
                System.setProperty(ZSTD_PROP, v);
                assertFalse(usesJavaZstd(null, null), "-D" + ZSTD_PROP + "=\"" + v + "\" must not enable the Java decoder");
            }
            System.setProperty(ZSTD_PROP, "java");
            assertTrue(usesJavaZstd(null, null));
        } finally {
            restore(ZSTD_PROP, prev);
        }
    }

    // ------------------------------------------------------------------ composition with SNAPPY

    @Test
    void theTwoOptionsAreIndependentAndStack() {
        String prevZ = System.getProperty(ZSTD_PROP), prevS = System.getProperty(SNAPPY_PROP);
        try {
            System.clearProperty(ZSTD_PROP);
            System.clearProperty(SNAPPY_PROP);

            CompressionCodecFactory neither = factory(Boolean.FALSE, Boolean.FALSE);
            assertFalse(neither instanceof PanamaSnappyCodecFactory || neither instanceof PureJavaZstdCodecFactory,
                    "neither option -> the stock factory is left in place");

            CompressionCodecFactory snappyOnly = factory(Boolean.TRUE, Boolean.FALSE);
            assertTrue(snappyOnly instanceof PanamaSnappyCodecFactory, "snappy only");

            CompressionCodecFactory zstdOnly = factory(Boolean.FALSE, Boolean.TRUE);
            assertTrue(zstdOnly instanceof PureJavaZstdCodecFactory, "zstd only");

            // Both: the ZSTD decorator is outermost and forwards SNAPPY to the SNAPPY decorator beneath it. The
            // observable proof that neither displaced the other is that SNAPPY still resolves to something that is
            // not the stock decoder: ask for it and for UNCOMPRESSED, whose stock decompressor is null/no-op.
            CompressionCodecFactory both = factory(Boolean.TRUE, Boolean.TRUE);
            assertTrue(both instanceof PureJavaZstdCodecFactory, "both -> zstd decorator outermost");
            Object snappyDec = both.getDecompressor(CompressionCodecName.SNAPPY);
            Object stockSnappyDec = factory(Boolean.FALSE, Boolean.FALSE).getDecompressor(CompressionCodecName.SNAPPY);
            assertFalse(snappyDec.getClass() == stockSnappyDec.getClass(),
                    "with both on, SNAPPY must still be served by the pure-Java decorator, not the stock decompressor");
        } finally {
            restore(ZSTD_PROP, prevZ);
            restore(SNAPPY_PROP, prevS);
        }
    }

    @Test
    void theZstdOptionDoesNotDisturbSnappyPropertyHandling() {
        String prevZ = System.getProperty(ZSTD_PROP), prevS = System.getProperty(SNAPPY_PROP);
        try {
            System.clearProperty(ZSTD_PROP);
            System.setProperty(SNAPPY_PROP, "panama");
            CompressionCodecFactory f = ParquetBatchReader.newReadOptions(FilterCompat.NOOP).getCodecFactory();
            assertTrue(f instanceof PanamaSnappyCodecFactory, "snappy property alone still installs the snappy decorator only");
        } finally {
            restore(ZSTD_PROP, prevZ);
            restore(SNAPPY_PROP, prevS);
        }
    }

    // ------------------------------------------------------------------ the fluent API

    @Test
    void scanBuilderThreadsTheOverrideThrough() {
        ParquetScan base = ParquetScan.scan(java.nio.file.Path.of("unused.parquet"));
        assertNull(base.javaZstdDecoderOverride(), "a scan that never calls useJavaZstdDecoder has no override");

        ParquetScan on = base.useJavaZstdDecoder(true);
        assertTrue(on.javaZstdDecoderOverride());
        assertNull(base.javaZstdDecoderOverride(), "ParquetScan is immutable: the original instance is unchanged");

        ParquetScan off = base.useJavaZstdDecoder(false);
        assertFalse(off.javaZstdDecoderOverride());
    }

    @Test
    void theTwoOverridesDoNotShareState() {
        ParquetScan base = ParquetScan.scan(java.nio.file.Path.of("unused.parquet"));

        ParquetScan z = base.useJavaZstdDecoder(true);
        assertNull(z.javaSnappyDecoderOverride(), "setting ZSTD must not set SNAPPY");

        ParquetScan s = base.useJavaSnappyDecoder(true);
        assertNull(s.javaZstdDecoderOverride(), "setting SNAPPY must not set ZSTD");

        ParquetScan both = z.useJavaSnappyDecoder(false);
        assertTrue(both.javaZstdDecoderOverride(), "a later SNAPPY call must keep the ZSTD override");
        assertFalse(both.javaSnappyDecoderOverride());

        ParquetScan both2 = s.useJavaZstdDecoder(false);
        assertTrue(both2.javaSnappyDecoderOverride(), "a later ZSTD call must keep the SNAPPY override");
        assertFalse(both2.javaZstdDecoderOverride());
    }

    @Test
    void everyOtherBuilderMethodCarriesBothOverridesForward() {
        CustomPredicate keepAll = (batch, row) -> true;
        ParquetScan seeded = ParquetScan.scan(java.nio.file.Path.of("unused.parquet"))
                .useJavaZstdDecoder(true).useJavaSnappyDecoder(false);

        ParquetScan[] derived = {
                seeded.select("id"),
                seeded.pushdown(Predicate.ge("id", 1L)),
                seeded.pushdown(Predicate.ge("id", 1L)).exactFilter(),
                seeded.withCustomPredicate("keep", keepAll),
                seeded.batchSize(10),
                seeded.withMetrics(true),
                seeded.maxRowGroupBytes(1_000),
                seeded.parallelism(3),
                seeded.pinWorkerThreads(true),
        };
        String[] via = {"select", "pushdown", "pushdown+exactFilter", "withCustomPredicate", "batchSize", "withMetrics",
                "maxRowGroupBytes", "parallelism", "pinWorkerThreads"};
        for (int i = 0; i < derived.length; i++) {
            assertSame(Boolean.TRUE, derived[i].javaZstdDecoderOverride(), via[i] + "() dropped the ZSTD override");
            assertSame(Boolean.FALSE, derived[i].javaSnappyDecoderOverride(), via[i] + "() dropped the SNAPPY override");
        }
        // ...and chained, in the order a caller would actually write it
        ParquetScan chained = ParquetScan.scan(java.nio.file.Path.of("unused.parquet"))
                .select("id").batchSize(64).parallelism(2).useJavaZstdDecoder(true).withMetrics(true).maxRowGroupBytes(5_000);
        assertSame(Boolean.TRUE, chained.javaZstdDecoderOverride());
    }

    // ------------------------------------------------------------------ helpers

    /** The installed codec factory for these overrides; the system properties are whatever the caller arranged. */
    private static CompressionCodecFactory factory(Boolean snappyOverride, Boolean zstdOverride) {
        ParquetReadOptions opts = ParquetBatchReader.newReadOptions(FilterCompat.NOOP, snappyOverride, zstdOverride);
        return opts.getCodecFactory();
    }

    private static boolean usesJavaZstd(Boolean snappyOverride, Boolean zstdOverride) {
        return factory(snappyOverride, zstdOverride) instanceof PureJavaZstdCodecFactory;
    }

    private static void restore(String prop, String prev) {
        if (prev == null) System.clearProperty(prop);
        else System.setProperty(prop, prev);
    }
}
