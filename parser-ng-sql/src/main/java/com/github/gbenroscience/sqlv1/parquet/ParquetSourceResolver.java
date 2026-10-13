package com.github.gbenroscience.sqlv1.parquet;

import com.github.gbenroscience.parser.ng.parquet.v1.ParquetSource;

import java.util.List;

/**
 * Turns a remote path from SQL text ({@code FROM 's3a://bucket/key/*.parquet'}, {@code read_parquet('gs://...')},
 * or a {@link ParquetSql#register registered} remote path) into the Parquet sources to read.
 *
 * <p>The module does not bundle any storage connector. You supply this resolver, typically over
 * parser-ng-parquet's {@link ParquetSource#hadoop} (with the Hadoop connector for the scheme on your classpath) or
 * {@link ParquetSource#of} over your own parquet-java {@code InputFile}:
 *
 * <pre>{@code
 * Configuration conf = ...; // credentials, endpoint
 * ParquetSqlOptions o = ParquetSqlOptions.defaults().withSourceResolver(spec -> {
 *     org.apache.hadoop.fs.Path p = new org.apache.hadoop.fs.Path(spec);
 *     List<ParquetSourceResolver.ResolvedSource> out = new ArrayList<>();
 *     for (FileStatus st : p.getFileSystem(conf).globStatus(p)) {   // expands globs; a plain key matches itself
 *         out.add(new ParquetSourceResolver.ResolvedSource(ParquetSource.hadoop(st.getPath().toString(), conf)));
 *     }
 *     return out;
 * });
 * }</pre>
 *
 * <p><b>Security.</b> A path in SQL text is an unauthenticated read of whatever this resolver can reach, with the
 * credentials it holds. {@link ParquetSqlOptions#withAllowedRoot} restricts local files only. If user-influenced SQL can
 * reach {@code ParquetSql}, the resolver must enforce its own allowlist (bucket and prefix) and throw otherwise.
 *
 * <p>A path is routed here when it starts with a URI scheme of two or more characters followed by {@code ://}
 * (so {@code C:\data\x.parquet} stays a local path). Implementations must be thread safe.
 */
@FunctionalInterface
public interface ParquetSourceResolver {

    /**
     * @param spec the path exactly as written in the SQL (or registered), possibly containing glob characters
     * @return the sources to read, at least one; they are read in {@link ResolvedSource#name()} order and must all
     *     have identical column names and types for the columns the query reads
     * @throws RuntimeException to refuse or fail; it surfaces as a {@link ParquetSqlException} naming {@code spec}
     */
    List<ResolvedSource> resolve(String spec);

    /**
     * One resolved source.
     *
     * @param name display name used in plan output and errors (typically the full URI). It is printed, so it must
     *     not contain credentials: do not embed {@code user:secret@} in the URI, pass them in the Hadoop {@code Configuration}
     * @param source the parser-ng-parquet source
     */
    record ResolvedSource(String name, ParquetSource source) {
        public ResolvedSource {
            if (name == null || name.isBlank()) throw new IllegalArgumentException("name must not be blank");
            if (source == null) throw new IllegalArgumentException("source must not be null");
        }

        /** Uses {@link ParquetSource#name()} as the display name (for {@code hadoop(uri, conf)} that is the URI). */
        public ResolvedSource(ParquetSource source) {
            this(nameOf(source), source);
        }

        private static String nameOf(ParquetSource source) {
            if (source == null) throw new IllegalArgumentException("source must not be null");
            return source.name();
        }
    }
}
