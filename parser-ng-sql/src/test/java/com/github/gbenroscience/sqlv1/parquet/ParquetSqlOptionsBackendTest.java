package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.arrow.tools.box.ArrowExecutionBackend;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ParquetSqlOptionsBackendTest {

    @Test
    void defaultsToCpuSimd() {
        assertEquals(ArrowExecutionBackend.CPU_SIMD, ParquetSqlOptions.defaults().backend());
    }

    @Test
    void withBackendReturnsACopyAndKeepsEveryOtherSetting() {
        ParquetSqlOptions base = ParquetSqlOptions.defaults().withParallelism(3).withBatchSize(500).withMaxRowGroupBytes(1 << 20)
                .withMetrics(false).withExactFilter(true).withMaxMaterializedRows(77)
                .withInt64Policy(ParquetSqlOptions.Int64Policy.ROUND_TO_DOUBLE).withAllowedRoot(Path.of("."));
        ParquetSqlOptions gpu = base.withBackend(ArrowExecutionBackend.GPU_AUTO);
        assertEquals(ArrowExecutionBackend.CPU_SIMD, base.backend(), "the original is immutable");
        assertEquals(ArrowExecutionBackend.GPU_AUTO, gpu.backend());
        assertEquals(base, gpu.withBackend(ArrowExecutionBackend.CPU_SIMD));
    }

    @Test
    void otherWithersKeepTheBackend() {
        ParquetSqlOptions o = ParquetSqlOptions.defaults().withBackend(ArrowExecutionBackend.GPU_AUTO)
                .withParallelism(2).withBatchSize(10).withMaxRowGroupBytes(100).withMetrics(false).withExactFilter(true)
                .withMaxMaterializedRows(5).withInt64Policy(ParquetSqlOptions.Int64Policy.ROUND_TO_DOUBLE).withAllowedRoot(Path.of("."));
        assertEquals(ArrowExecutionBackend.GPU_AUTO, o.backend());
    }

    @Test
    void aNullBackendIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> ParquetSqlOptions.defaults().withBackend(null));
    }

    @Test
    void theEightArgumentConstructorStillWorksAndUsesCpuSimd() {
        ParquetSqlOptions o = new ParquetSqlOptions(null, null, null, true, false, 10L, ParquetSqlOptions.Int64Policy.ERROR, null);
        assertEquals(ArrowExecutionBackend.CPU_SIMD, o.backend());
    }
}
