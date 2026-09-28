package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;

class ParquetFileInfoTest {

    @TempDir
    static Path dir;
    static Path flat;

    @BeforeAll
    static void writeFixture() throws IOException {
        flat = ParquetTestData.writeFlat(dir, "flat.parquet");
    }

    @Test
    void reportsFileLevelFacts() throws IOException {
        ParquetFileInfo info = ParquetFileInfo.read(flat);
        assertEquals(flat, info.file());
        assertEquals(Files.size(flat), info.fileSizeBytes());
        assertEquals(N, info.rowCount());
        assertTrue(info.schema().contains("message flat"), info.schema());
        assertNotNull(info.createdBy());
        assertNotNull(info.keyValueMetadata());
    }

    @Test
    void rowGroupsAddUpToTheFile() {
        ParquetFileInfo info = ParquetFileInfo.read(flat);
        assertTrue(info.rowGroups().size() >= 3);
        long sum = 0;
        for (int i = 0; i < info.rowGroups().size(); i++) {
            ParquetFileInfo.RowGroupInfo rg = info.rowGroups().get(i);
            assertEquals(i, rg.index());
            assertTrue(rg.rowCount() > 0);
            assertTrue(rg.totalByteSize() > 0);
            assertEquals(FLAT_COLUMNS.size(), rg.columns().size());
            sum += rg.rowCount();
        }
        assertEquals(N, sum);
        assertEquals(info.rowCount(), sum);
    }

    @Test
    void columnListsFollowFileOrder() {
        ParquetFileInfo info = ParquetFileInfo.read(flat);
        assertEquals(FLAT_COLUMNS, info.topLevelColumns());
        assertEquals(FLAT_COLUMNS, info.primitiveColumns());
    }

    @Test
    void columnChunkStatisticsAreExposedAsStrings() {
        ParquetFileInfo.RowGroupInfo first = ParquetFileInfo.read(flat).rowGroups().get(0);
        ParquetFileInfo.ColumnChunkInfo id = first.columns().stream().filter(c -> c.path().equals("id")).findFirst().orElseThrow();
        assertEquals("INT64", id.primitiveType());
        assertEquals("UNCOMPRESSED", id.codec());
        assertEquals("0", id.min(), "ids are the row index, so the first row group starts at 0");
        assertEquals(Long.valueOf(0), id.nullCount());
        assertEquals(first.rowCount(), id.valueCount());
        assertTrue(id.compressedSize() > 0);
        assertTrue(id.uncompressedSize() > 0);
        assertFalse(id.encodings().isEmpty());
        assertEquals(id.encodings().stream().sorted().toList(), id.encodings(), "encodings are reported sorted");

        ParquetFileInfo.ColumnChunkInfo grp = first.columns().stream().filter(c -> c.path().equals("grp")).findFirst().orElseThrow();
        assertTrue(grp.nullCount() != null && grp.nullCount() > 0, "grp has nulls in the first row group");
    }

    @Test
    void maxOfTheLastRowGroupIsTheLastId() {
        var rgs = ParquetFileInfo.read(flat).rowGroups();
        var last = rgs.get(rgs.size() - 1).columns().stream().filter(c -> c.path().equals("id")).findFirst().orElseThrow();
        assertEquals(String.valueOf(N - 1), last.max());
    }

    @Test
    void nestedColumnsAreTopLevelButNotPrimitive() throws IOException {
        Path nested = RandomParquetFiles.write(dir.resolve("nested.parquet"), RandomParquetFiles.SAMPLE_NESTED_SCHEMA, 200,
                RandomParquetFiles.Config.defaults().seed(1));
        ParquetFileInfo info = ParquetFileInfo.read(nested);
        assertEquals(java.util.List.of("id", "tags", "point"), info.topLevelColumns());
        assertEquals(java.util.List.of("id"), info.primitiveColumns());
        assertEquals(200, info.rowCount());
    }

    @Test
    void readsAnEmptyFile() throws IOException {
        Path empty = ParquetTestData.writeFlat(dir, "empty-info.parquet", 0, SMALL_ROW_GROUP_BYTES, true, 1024);
        ParquetFileInfo info = ParquetFileInfo.read(empty);
        assertEquals(0, info.rowCount());
        assertTrue(info.rowGroups().isEmpty());
    }

    @Test
    void failsWithContextOnMissingOrInvalidFiles() throws IOException {
        Path missing = dir.resolve("missing.parquet");
        ParquetScanException e = assertThrows(ParquetScanException.class, () -> ParquetFileInfo.read(missing));
        assertEquals(missing, e.file());

        Path junk = dir.resolve("junk-info.parquet");
        Files.writeString(junk, "not parquet at all, just a few lines of text to pad it out");
        assertThrows(ParquetScanException.class, () -> ParquetFileInfo.read(junk));
    }
}