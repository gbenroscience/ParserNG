package com.github.gbenroscience.parser.ng.parquet.v1;

import java.nio.file.Path;

/** Unchecked failure with file / row-group / column context. Never used to signal partial data. */
public final class ParquetScanException extends RuntimeException {

    private final String source;
    private final Path file;
    private final int rowGroup;
    private final String column;

    /** For a local file. */
    public ParquetScanException(String message, Path file, int rowGroup, String column, Throwable cause) {
        this(message, file == null ? null : file.toString(), file, rowGroup, column, cause);
    }

    public ParquetScanException(String message, Path file) {
        this(message, file, -1, null, null);
    }

    public ParquetScanException(String message, ParquetSource file) {
        this(message, file, -1, null, null);
    }
    /** For any storage: {@code source} is the name of the file (a path or URI), or null if unknown. */
    public ParquetScanException(String message, ParquetSource source, int rowGroup, String column, Throwable cause) {
        this(message, source == null ? null : source.name(), source == null ? null : source.localPath(),
                rowGroup, column, cause);
    }

    /** For an address that never became a {@link ParquetSource} (e.g. a Hadoop URI that could not be opened). */
    public ParquetScanException(String message, String source, int rowGroup, String column, Throwable cause) {
        this(message, source, null, rowGroup, column, cause);
    }

    private ParquetScanException(String message, String source, Path file, int rowGroup, String column, Throwable cause) {
        super(format(message, source, rowGroup, column), cause);
        this.source = source;
        this.file = file;
        this.rowGroup = rowGroup;
        this.column = column;
    }

    /** Index within the row groups that survived pruning (not the original footer index); -1 if n/a. */
    public int rowGroup() { return rowGroup; }
    /** The local file, or {@code null} if the data source is not a local file (or unknown); see {@link #source()}. */
    public Path file() { return file; }
    /** Name (path or URI) of the data source, or {@code null} if unknown. */
    public String source() { return source; }
    public String column() { return column; }

    private static String format(String m, String f, int rg, String c) {
        StringBuilder sb = new StringBuilder(m);
        sb.append(" [file=").append(f);
        if (rg >= 0) sb.append(", rowGroup=").append(rg);
        if (c != null) sb.append(", column=").append(c);
        return sb.append(']').toString();
    }
}
