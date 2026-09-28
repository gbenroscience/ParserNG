
package com.github.gbenroscience.parser.ng.parquet.v1;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ParquetScanExceptionTest {

    private static final Path FILE = Path.of("data", "x.parquet");

    @Test
    void messageCarriesFileRowGroupAndColumn() {
        RuntimeException cause = new RuntimeException("root");
        ParquetScanException e = new ParquetScanException("boom", FILE, 3, "price", cause);
        String m = e.getMessage();
        assertTrue(m.startsWith("boom ["), m);
        assertTrue(m.contains("file=" + FILE), m);
        assertTrue(m.contains("rowGroup=3"), m);
        assertTrue(m.contains("column=price"), m);
        assertSame(cause, e.getCause());
        assertEquals(FILE, e.file());
        assertEquals(3, e.rowGroup());
        assertEquals("price", e.column());
    }

    @Test
    void negativeRowGroupAndNullColumnAreOmittedFromTheMessage() {
        ParquetScanException e = new ParquetScanException("boom", FILE);
        assertFalse(e.getMessage().contains("rowGroup"), e.getMessage());
        assertFalse(e.getMessage().contains("column"), e.getMessage());
        assertEquals(-1, e.rowGroup());
        assertNull(e.column());
        assertNull(e.getCause());
    }

    @Test
    void rowGroupZeroIsReported() {
        assertTrue(new ParquetScanException("boom", FILE, 0, null, null).getMessage().contains("rowGroup=0"));
    }

    @Test
    void nullFileIsToleratedInTheMessage() {
        ParquetScanException e = new ParquetScanException("Unknown column", null, -1, "c", null);
        assertTrue(e.getMessage().contains("file=null"));
        assertNull(e.file());
    }

    @Test
    void isUnchecked() {
        assertInstanceOf(RuntimeException.class, new ParquetScanException("x", FILE));
    }
}