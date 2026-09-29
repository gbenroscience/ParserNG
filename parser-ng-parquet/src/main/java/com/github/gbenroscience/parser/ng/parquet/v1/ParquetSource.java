package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.hadoop.conf.Configuration;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.io.InputFile;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.io.SeekableInputStream;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;

/**
 * Where a Parquet file lives: a parquet-java {@link InputFile} plus a human-readable name used in
 * error messages and thread names.
 *
 * <p>Storage is entirely delegated to the {@link InputFile}, so anything parquet-java can read can be
 * scanned here:
 * <ul>
 *   <li>{@link #of(Path)}: a local file (what {@code ParquetScan.scan(Path)} uses).</li>
 *   <li>{@link #of(InputFile)} / {@link #of(InputFile, String)}: any custom implementation.</li>
 *   <li>{@link #hadoop(String, Configuration)}: any Hadoop {@code FileSystem} URI ({@code hdfs://},
 *       {@code s3a://}, {@code gs://}, {@code abfss://}, {@code file://}). The Hadoop filesystem
 *       <em>connector</em> for the scheme (e.g. {@code hadoop-aws} for {@code s3a}) is NOT bundled; add it
 *       to the classpath yourself. Credentials and endpoints come from the {@link Configuration} you
 *       pass, never from anything else.</li>
 * </ul>
 *
 * <p>A scan reads and parses the footer once, then opens one stream per reader (the main reader and, with
 * {@code parallelism > 1}, one per worker), so {@link InputFile#newStream()} must be safe to call
 * repeatedly and concurrently.
 */
public final class ParquetSource {

    private final InputFile input;
    private final String name;
    private final Path localPath; // null for anything that is not a local file

    private ParquetSource(InputFile input, String name, Path localPath) {
        if (input == null) throw new NullPointerException("input");
        if (name == null) throw new NullPointerException("name");
        this.input = input;
        this.name = name;
        this.localPath = localPath;
    }

    /** A file on the local filesystem. */
    public static ParquetSource of(Path file) {
        if (file == null) throw new NullPointerException("file");
        return new ParquetSource(new LocalInputFile(file), file.toString(), file);
    }

    /** Any {@link InputFile}; its {@code toString()} is used as the display name. */
    public static ParquetSource of(InputFile input) {
        if (input == null) throw new NullPointerException("input");
        return new ParquetSource(input, String.valueOf(input), null);
    }

    /** Any {@link InputFile}, with an explicit display name (for example its URI). */
    public static ParquetSource of(InputFile input, String name) {
        return new ParquetSource(input, name, null);
    }

    /**
     * A file addressed by a Hadoop filesystem URI, e.g. {@code s3a://bucket/key.parquet}. The matching
     * Hadoop connector must be on the classpath. The path is resolved (and existence is checked) here.
     *
     * @throws ParquetScanException if the filesystem for the URI's scheme cannot be created or the path cannot be opened
     */
    public static ParquetSource hadoop(String uri, Configuration conf) {
        if (uri == null) throw new NullPointerException("uri");
        if (conf == null) throw new NullPointerException("conf");
        try {
            return new ParquetSource(HadoopInputFile.fromPath(new org.apache.hadoop.fs.Path(uri), conf), uri, null);
        } catch (IOException | RuntimeException e) {
            throw new ParquetScanException("Cannot open Hadoop path", uri, -1, null, e);
        }
    }

    /** As {@link #hadoop(String, Configuration)}. */
    public static ParquetSource hadoop(URI uri, Configuration conf) {
        if (uri == null) throw new NullPointerException("uri");
        return hadoop(uri.toString(), conf);
    }

    /**
     * Reads and parses the footer over one short-lived stream. A scan does this exactly once and hands the
     * result to every reader it opens via {@link #openReader}, so the footer is fetched once per scan rather
     * than once per reader (which matters when each fetch is a network round trip).
     */
    ParquetMetadata readFooter() throws IOException {
        try (SeekableInputStream in = input.newStream()) {
            return ParquetFileReader.readFooter(input, ParquetReadOptions.builder().build(), in);
        }
    }

    /**
     * Opens a reader over its own fresh stream using an already-parsed {@code footer}; the reader owns and
     * closes the stream. Row-group filtering from {@code options} is applied to the footer's blocks by the
     * reader itself, exactly as when it parses the footer on its own.
     */
    ParquetFileReader openReader(ParquetMetadata footer, ParquetReadOptions options) throws IOException {
        SeekableInputStream in = input.newStream();
        try {
            return ParquetFileReader.open(input, footer, options, in);
        } catch (IOException | RuntimeException e) {
            try { in.close(); } catch (IOException suppressed) { e.addSuppressed(suppressed); }
            throw e;
        }
    }

    /** The parquet-java input this scan reads through. */
    public InputFile input() { return input; }

    /** Human-readable identity used in error messages, e.g. a path or URI. */
    public String name() { return name; }

    /** The local path if this is a local file, otherwise {@code null}. */
    public Path localPath() { return localPath; }

    /** Last path segment of {@link #name()}, for thread names and compact logging. */
    public String shortName() {
        String n = name;
        while (n.length() > 1 && (n.endsWith("/") || n.endsWith("\\"))) n = n.substring(0, n.length() - 1);
        int cut = Math.max(n.lastIndexOf('/'), n.lastIndexOf('\\'));
        return cut >= 0 && cut < n.length() - 1 ? n.substring(cut + 1) : n;
    }

    @Override public String toString() { return name; }
}
