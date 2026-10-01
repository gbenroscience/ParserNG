package com.github.gbenroscience.sqlv1.parquet;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

/**
 * Turns a path or glob from SQL text (or from {@code register}) into a
 * validated, sorted list of regular files.
 *
 * <h2>Security</h2>
 * A path in SQL text is an <b>unauthenticated file read</b> on the machine
 * running the query: {@code SELECT * FROM '/etc/whatever'} will try to open it
 * (Parquet magic bytes are checked by the reader, but the attempt and its error
 * message still disclose existence and type). If any user-influenced SQL can
 * reach {@link ParquetSql}, configure
 * {@link ParquetSqlOptions#withAllowedRoot(Path)}: every file is then resolved
 * (symlinks included) and must live under that directory, or the query fails.
 */
final class ParquetTables {

    private ParquetTables() { }

    /** Expands {@code spec} (a file or a glob) and validates every match against {@code allowedRoot}. */
    static List<Path> resolve(String spec, Path allowedRoot) {
        if (spec == null || spec.isBlank()) throw new ParquetSqlException("Empty Parquet path");
        List<Path> out;
        if (isGlob(spec)) {
            out = expand(spec);
            if (out.isEmpty()) throw new ParquetSqlException("Glob '" + spec + "' matched no files");
        } else {
            out = new ArrayList<>(List.of(Paths.get(spec)));
        }
        List<Path> checked = new ArrayList<>(out.size());
        for (Path p : out) checked.add(validate(p, allowedRoot));
        Collections.sort(checked);
        return checked;
    }

    static boolean isGlob(String s) {
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '*' || c == '?' || c == '[' || c == '{') return true;
        }
        return false;
    }

    /**
     * Glob syntax treats a backslash as an escape, but on Windows it is also the path separator.
     * With {@code separator == '\\'} every backslash becomes a forward slash (which the Windows
     * glob matcher accepts as a separator); elsewhere the text is untouched.
     */
    static String normalizeGlob(String pattern, char separator) {
        return separator == '\\' ? pattern.replace('\\', '/') : pattern;
    }

    private static List<Path> expand(String rawPattern) {
        String pattern = normalizeGlob(rawPattern, java.io.File.separatorChar);
        int firstGlob = -1;
        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (c == '*' || c == '?' || c == '[' || c == '{') { firstGlob = i; break; }
        }
        int slash = Math.max(pattern.lastIndexOf('/', firstGlob), pattern.lastIndexOf('\\', firstGlob));
        Path base = slash < 0 ? Paths.get(".") : Paths.get(pattern.substring(0, Math.max(slash, 0) + (slash == 0 ? 1 : 0)));
        PathMatcher m = FileSystems.getDefault().getPathMatcher("glob:" + pattern);
        boolean recursive = pattern.contains("**");
        List<Path> hits = new ArrayList<>();
        try (Stream<Path> s = recursive ? Files.walk(base) : Files.list(base)) {
            s.filter(Files::isRegularFile).filter(p -> m.matches(slash < 0 ? p.normalize() : p)).forEach(hits::add);
        } catch (IOException | RuntimeException e) {
            throw new ParquetSqlException("Cannot expand glob '" + pattern + "': " + e.getMessage(), e);
        }
        return hits;
    }

    private static Path validate(Path p, Path allowedRoot) {
        Path abs = p.toAbsolutePath().normalize();
        if (!Files.isRegularFile(abs)) {
            throw new ParquetSqlException("Not a readable file: " + abs);
        }
        if (allowedRoot != null) {
            Path real;
            Path rootReal;
            try {
                real = abs.toRealPath();
                rootReal = allowedRoot.toRealPath();
            } catch (IOException e) {
                throw new ParquetSqlException("Cannot resolve path " + abs + " against allowed root " + allowedRoot, e);
            }
            if (!real.startsWith(rootReal)) {
                throw new ParquetSqlException("Path " + abs + " is outside the allowed root " + allowedRoot);
            }
            return real;
        }
        return abs;
    }
}
