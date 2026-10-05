package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.PanamaSnappyCodecFactory;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.filter2.compat.FilterCompat;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ParquetScan#useJavaSnappyDecoder} is a per-scan override of the {@code -Dparserng.snappy}
 * system property, read by {@link ParquetBatchReader#newReadOptions}. These go straight at that method
 * (package-private, same package) rather than through a full scan: which {@link
 * org.apache.parquet.compression.CompressionCodecFactory} class {@code newReadOptions} installs is the
 * entire effect of this feature, and it is otherwise invisible from outside a scan (SNAPPY decodes to the
 * same bytes either way -- that is the whole point, see {@code SnappyDecompressor}'s fuzz suite). The
 * system-property side of this (default-off, {@code -Dparserng.snappy=panama} turns it on JVM-wide) is
 * exercised by {@code ParquetScanTest}'s existing coverage; what is new here is only: an explicit
 * {@code TRUE}/{@code FALSE} always wins over the property, and {@code null} (the default -- no scan
 * ever calls {@link ParquetScan#useJavaSnappyDecoder}) defers to it.
 */
class UseJavaSnappyDecoderTest {

    @Test
    void nullOverrideDefersToSystemProperty() {
        String prev = System.getProperty("parserng.snappy");
        try {
            System.clearProperty("parserng.snappy");
            assertFalse(usesJavaDecoder(null), "property unset, no override -> stock decoder");

            System.setProperty("parserng.snappy", "panama");
            assertTrue(usesJavaDecoder(null), "property set, no override -> Java decoder");
        } finally {
            restore(prev);
        }
    }

    @Test
    void explicitTrueWinsRegardlessOfProperty() {
        String prev = System.getProperty("parserng.snappy");
        try {
            System.clearProperty("parserng.snappy");
            assertTrue(usesJavaDecoder(Boolean.TRUE), "override=true, property unset -> Java decoder");

            System.setProperty("parserng.snappy", "panama");
            assertTrue(usesJavaDecoder(Boolean.TRUE), "override=true, property set -> still Java decoder");
        } finally {
            restore(prev);
        }
    }

    @Test
    void explicitFalseWinsRegardlessOfProperty() {
        String prev = System.getProperty("parserng.snappy");
        try {
            System.clearProperty("parserng.snappy");
            assertFalse(usesJavaDecoder(Boolean.FALSE), "override=false, property unset -> stock decoder");

            System.setProperty("parserng.snappy", "panama");
            assertFalse(usesJavaDecoder(Boolean.FALSE), "override=false, property set -> still stock decoder");
        } finally {
            restore(prev);
        }
    }

    @Test
    void scanBuilderThreadsTheOverrideThrough() {
        // ParquetScan itself, not just the internal method: the fluent call must actually reach
        // newReadOptions. javaSnappyDecoderOverride() is this scan's own record of what was asked for;
        // a fresh scan has no override, and the setter is visible on the very next call in the chain.
        ParquetScan base = ParquetScan.scan(java.nio.file.Path.of("unused.parquet"));
        assertNull(base.javaSnappyDecoderOverride(), "a scan that never calls useJavaSnappyDecoder has no override");

        ParquetScan on = base.useJavaSnappyDecoder(true);
        assertTrue(on.javaSnappyDecoderOverride());
        assertNull(base.javaSnappyDecoderOverride(), "ParquetScan is immutable: the original instance is unchanged");

        ParquetScan off = base.useJavaSnappyDecoder(false);
        assertFalse(off.javaSnappyDecoderOverride());
    }

    private static boolean usesJavaDecoder(Boolean override) {
        ParquetReadOptions opts = ParquetBatchReader.newReadOptions(FilterCompat.NOOP, override);
        return opts.getCodecFactory() instanceof PanamaSnappyCodecFactory;
    }

    private static void restore(String prev) {
        if (prev == null) System.clearProperty("parserng.snappy");
        else System.setProperty("parserng.snappy", prev);
    }
}
