package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static com.github.gbenroscience.parser.ng.parquet.v1.ParquetTestData.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * One big row group split into many small pages (each column paginates on its own boundaries) plus a narrow
 * predicate: every row of every batch must still be internally consistent. Page-level pruning is off precisely
 * because trimmed pages would let columns of one batch describe different rows.
 */
class PagePruningAlignmentTest {

    @TempDir
    Path dir;

    @Test
    void columnsOfEachRowStayAlignedUnderAPushdownOnManyPages() throws IOException {
        Path f = ParquetTestData.writeFlat(dir, "pages.parquet", N, 64L * 1024 * 1024, false, 512);
        assertEquals(1, ParquetFileInfo.read(f).rowGroups().size(), "fixture: one row group");

        Predicate p = Predicate.and(Predicate.ge("id", 3_000L), Predicate.le("id", 3_100L));
        Set<Long> seen = new HashSet<>();
        try (BufferAllocator a = new RootAllocator();
             ParquetBatchReader r = ParquetScan.scan(f).select("id", "score", "lo", "hi").pushdown(p).open(a)) {
            while (r.next()) {
                VectorSchemaRoot b = r.root();
                BigIntVector id = (BigIntVector) b.getVector("id");
                Float8Vector score = (Float8Vector) b.getVector("score");
                BigIntVector lo = (BigIntVector) b.getVector("lo");
                BigIntVector hi = (BigIntVector) b.getVector("hi");
                for (int k = 0; k < b.getRowCount(); k++) {
                    long i = id.get(k);
                    Row e = ROWS.get((int) i);
                    assertEquals(e.score(), score.get(k), 0.0, "score of row " + i);
                    assertEquals(e.lo(), lo.get(k), "lo of row " + i);
                    assertEquals(e.hi(), hi.get(k), "hi of row " + i);
                    seen.add(i);
                }
            }
        }
        for (long i = 3_000; i <= 3_100; i++) assertTrue(seen.contains(i), "matching row " + i + " missing");
    }
}
