package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Glob normalization (Windows backslashes) and the engine-safe synthetic column names. */
class ParquetTablesTest {

    @Test
    void windowsBackslashesBecomeSlashesInGlobs() {
        assertEquals("C:/Users/x/data/*.parquet", ParquetTables.normalizeGlob("C:\\Users\\x\\data\\*.parquet", '\\'));
        assertEquals("C:/Users/x/multi/*.parquet", ParquetTables.normalizeGlob("C:\\Users\\x\\multi/*.parquet", '\\'));
    }

    @Test
    void otherPlatformsLeaveBackslashesAsGlobEscapes() {
        assertEquals("/data/a\\*b/*.parquet", ParquetTables.normalizeGlob("/data/a\\*b/*.parquet", '/'));
    }

    @Test
    void globDetection() {
        assertTrue(ParquetTables.isGlob("a/*.parquet"));
        assertTrue(ParquetTables.isGlob("a/file?.parquet"));
        assertFalse(ParquetTables.isGlob("a/file.parquet"));
    }

    @Test
    void syntheticColumnNamesAreLettersOnlyAndUnique() {
        // ParserNG rejects identifiers that start with '_'; the name is spliced into expression text.
        java.util.Set<String> seen = new java.util.HashSet<>();
        for (int i = 0; i < 1000; i++) {
            String n = NativeLeaves.maskName(i);
            assertTrue(n.matches("[a-z]+"), n);
            assertTrue(seen.add(n), "duplicate " + n);
        }
        assertEquals("pngmaska", NativeLeaves.maskName(0));
        assertEquals("pngmaskz", NativeLeaves.maskName(25));
        assertEquals("pngmaskaa", NativeLeaves.maskName(26));
        assertTrue(NumericViews.ROW_ID.matches("[a-z]+"));
    }
}
