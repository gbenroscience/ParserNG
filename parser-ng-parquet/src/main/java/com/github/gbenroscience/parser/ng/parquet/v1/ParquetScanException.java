package com.github.gbenroscience.parser.ng.parquet.v1;

import java.nio.file.Path;

/** Unchecked failure with file / row-group / column context. Never used to signal partial data. */
public final class ParquetScanException extends RuntimeException {

    private final Path file;
    private final int rowGroup;
    private final String column;

    public ParquetScanException(String message, Path file, int rowGroup, String column, Throwable cause) {
        super(format(message, file, rowGroup, column), cause);
        this.file = file;
        this.rowGroup = rowGroup;
        this.column = column;
    }

    public ParquetScanException(String message, Path file) {
        this(message, file, -1, null, null);
    }

    /** Index within the row groups that survived pruning (not the original footer index); -1 if n/a. */
    public int rowGroup() { return rowGroup; }
    public Path file() { return file; }
    public String column() { return column; }

    private static String format(String m, Path f, int rg, String c) {
        StringBuilder sb = new StringBuilder(m);
        sb.append(" [file=").append(f);
        if (rg >= 0) sb.append(", rowGroup=").append(rg);
        if (c != null) sb.append(", column=").append(c);
        return sb.append(']').toString();
    }
}
