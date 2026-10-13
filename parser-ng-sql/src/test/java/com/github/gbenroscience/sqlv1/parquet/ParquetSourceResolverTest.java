package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetSource;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ParquetSourceResolverTest {

    @Test
    void onlySchemeSlashSlashPathsAreRemote() {
        for (String s : new String[] {"s3a://b/k/*.parquet", "gs://b/k", "hdfs://nn/x", "abfss://c@a.dfs.core.windows.net/p"}) {
            assertTrue(ParquetTables.isRemote(s), s);
        }
        for (String s : new String[] {"C:\\data\\x.parquet", "C://x", "/tmp/x.parquet", "data/*.parquet", "x.parquet", ""}) {
            assertFalse(ParquetTables.isRemote(s), s);
        }
    }

    @Test
    void aRemotePathWithoutAResolverIsRejectedWithAClearError() {
        ParquetSqlException e = assertThrows(ParquetSqlException.class,
                () -> ParquetTables.resolveAll("s3a://bucket/key.parquet", ParquetSqlOptions.defaults()));
        assertTrue(e.getMessage().contains("withSourceResolver"), e.getMessage());
    }

    @Test
    void resolvedSourcesAreSortedByName() {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withSourceResolver(spec -> List.of(
                new ParquetSourceResolver.ResolvedSource("s3a://b/k/b.parquet", sourceStub()),
                new ParquetSourceResolver.ResolvedSource("s3a://b/k/a.parquet", sourceStub())));
        List<FileRef> refs = ParquetTables.resolveAll("s3a://b/k/*.parquet", o);
        assertEquals(List.of("s3a://b/k/a.parquet", "s3a://b/k/b.parquet"), refs.stream().map(FileRef::toString).toList());
    }

    @Test
    void anEmptyResultAndAFailingResolverGiveNamedErrors() {
        ParquetSqlOptions empty = ParquetSqlOptions.defaults().withSourceResolver(spec -> List.of());
        assertTrue(assertThrows(ParquetSqlException.class, () -> ParquetTables.resolveAll("gs://b/k", empty))
                .getMessage().contains("matched no files"));

        ParquetSqlOptions boom = ParquetSqlOptions.defaults().withSourceResolver(spec -> { throw new IllegalStateException("denied"); });
        ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> ParquetTables.resolveAll("gs://b/k", boom));
        assertTrue(e.getMessage().contains("gs://b/k") && e.getMessage().contains("denied"), e.getMessage());
        assertNotNull(e.getCause());

        ParquetSqlOptions refuse = ParquetSqlOptions.defaults().withSourceResolver(spec -> { throw new ParquetSqlException("bucket not allowed"); });
        assertEquals("bucket not allowed", assertThrows(ParquetSqlException.class, () -> ParquetTables.resolveAll("gs://b/k", refuse)).getMessage());
    }

    @Test
    void theResolverSurvivesEveryOtherWither() {
        ParquetSourceResolver r = spec -> List.of();
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withSourceResolver(r).withParallelism(2).withBatchSize(10)
                .withMaxRowGroupBytes(100).withMetrics(false).withExactFilter(true).withMaxMaterializedRows(5)
                .withInt64Policy(ParquetSqlOptions.Int64Policy.ROUND_TO_DOUBLE).withAllowedRoot(Path.of("."))
                .withBackend(com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend.CPU_SIMD);
        assertSame(r, o.sourceResolver());
        assertNull(ParquetSqlOptions.defaults().sourceResolver());
    }

    @Test
    void aLocalPathStillGoesThroughTheAllowedRootJail() {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withAllowedRoot(Path.of(System.getProperty("java.io.tmpdir")));
        assertThrows(ParquetSqlException.class, () -> ParquetTables.resolveAll("/definitely/not/under/tmp/x.parquet", o));
    }

    @Test
    void theSingleArgumentFormUsesTheSourceName() {
        ParquetSource src = ParquetSource.of(new org.apache.parquet.io.LocalInputFile(Path.of("x.parquet")), "s3a://b/k/x.parquet");
        assertEquals("s3a://b/k/x.parquet", new ParquetSourceResolver.ResolvedSource(src).name());
        assertThrows(IllegalArgumentException.class, () -> new ParquetSourceResolver.ResolvedSource((ParquetSource) null));
    }

    private static ParquetSource sourceStub() {
        return ParquetSource.of(new org.apache.parquet.io.LocalInputFile(Path.of("unused.parquet")), "unused");
    }
}
