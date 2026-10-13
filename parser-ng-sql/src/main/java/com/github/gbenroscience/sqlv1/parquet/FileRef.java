package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetFileInfo;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetSource;

import java.nio.file.Path;

/**
 * One Parquet file to read: a local {@link Path} (the original code path, unchanged) or a remote
 * {@link ParquetSource}. {@link #toString()} is the display name used in plans and errors.
 */
final class FileRef {

    private final Path path;
    private final ParquetSource remote;
    private final String name;

    private FileRef(Path path, ParquetSource remote, String name) {
        this.path = path;
        this.remote = remote;
        this.name = name;
    }

    static FileRef local(Path p) {
        return new FileRef(p, null, p.toString());
    }

    static FileRef remote(String name, ParquetSource s) {
        return new FileRef(null, s, name);
    }

    ParquetFileInfo info() {
        return path != null ? ParquetFileInfo.read(path) : ParquetFileInfo.read(remote);
    }

    ParquetScan scan() {
        return path != null ? ParquetScan.scan(path) : ParquetScan.scan(remote);
    }

    @Override
    public String toString() {
        return name;
    }
}
