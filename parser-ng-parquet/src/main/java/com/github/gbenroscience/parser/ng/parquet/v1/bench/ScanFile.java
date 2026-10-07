package com.github.gbenroscience.parser.ng.parquet.v1.bench;

import com.github.gbenroscience.parser.ng.parquet.util.RandomParquetFiles;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetBatchReader;
import com.github.gbenroscience.parser.ng.parquet.v1.ParquetScan;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.UnaryOperator;

/**
 * parser-ng-parquet side of the pyarrow comparison.
 *
 * Writes the fixtures with RandomParquetFiles (same schemas, seeds and row-group size as DecodeBenchmark) into
 * <java.io.tmpdir>/parser-ng-pyarrow-compare and KEEPS them, so pyarrow_bench.py reads the very same files
 * from the same default location. Existing fixtures are reused; pass "regen" to rewrite them.
 *
 * Args (all optional): [flatRows=5000000] [nestedRows=1000000] [regen] [rowGroupMB=8]
 * A rowGroupMB other than 8 uses its own folder (parser-ng-pyarrow-compare-rg<N>mb), so pass the same --dir to pyarrow_bench.py.
 */
public class ScanFile {
    public static void main(String[] args) throws Exception {
        int flatRows = args.length > 0 ? Integer.parseInt(args[0]) : 5_000_000;
        int nestedRows = args.length > 1 ? Integer.parseInt(args[1]) : 1_000_000;
        boolean regen = args.length > 2 && args[2].equalsIgnoreCase("regen");
        int rowGroupMb = args.length > 3 ? Integer.parseInt(args[3]) : 8;
        long rowGroupBytes = rowGroupMb * 1024L * 1024L;

        Path dir = Path.of(System.getProperty("java.io.tmpdir"), rowGroupMb == 8 ? "parser-ng-pyarrow-compare" : "parser-ng-pyarrow-compare-rg" + rowGroupMb + "mb");
        Files.createDirectories(dir);
        Path flat = dir.resolve("flat.parquet"), nested = dir.resolve("nested.parquet");
        System.out.println("Fixtures dir: " + dir.toAbsolutePath());

        if (regen || !Files.exists(flat)) {
            RandomParquetFiles.write(flat, RandomParquetFiles.SAMPLE_FLAT_SCHEMA, flatRows,
                    RandomParquetFiles.Config.defaults().seed(42).rowGroupSize(rowGroupBytes));
        }
        if (regen || !Files.exists(nested)) {
            RandomParquetFiles.write(nested, RandomParquetFiles.SAMPLE_NESTED_SCHEMA, nestedRows,
                    RandomParquetFiles.Config.defaults().seed(7).collectionSize(0, 8).rowGroupSize(rowGroupBytes));
        }

        try (BufferAllocator alloc = new RootAllocator()) {
            System.out.printf("%-34s%10s%10s%14s%n", "scenario", "rows", "median s", "rows/s");
            run(alloc, "flat/full-scan/seq", flat, s -> s);
            run(alloc, "flat/projected(2 of 7)/seq", flat, s -> s.select("id", "value"));
            run(alloc, "flat/full-scan/parallel(2)", flat, s -> s.parallelism(2));
            run(alloc, "flat/full-scan/parallel(4)", flat, s -> s.parallelism(4));
            run(alloc, "flat/full-scan/parallel(8)", flat, s -> s.parallelism(8));
            run(alloc, "nested/full-scan/seq", nested, s -> s);
            run(alloc, "nested/full-scan/parallel(4)", nested, s -> s.parallelism(4));
        }
    }

    static void run(BufferAllocator alloc, String label, Path f, UnaryOperator<ParquetScan> cfg) throws Exception {
        for (int i = 0; i < 2; i++) drain(alloc, f, cfg);
        double[] t = new double[5];
        long rows = 0;
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            rows = drain(alloc, f, cfg);
            t[i] = (System.nanoTime() - t0) / 1e9;
        }
        Arrays.sort(t);
        System.out.printf("%-34s%10d%10.3f%,14.0f%n", label, rows, t[2], rows / t[2]);
    }

    static long drain(BufferAllocator alloc, Path f, UnaryOperator<ParquetScan> cfg) throws Exception {
        long rows = 0;
        try (ParquetBatchReader r = cfg.apply(ParquetScan.scan(f)).open(alloc)) {
            while (r.next()) rows += r.root().getRowCount();
        }
        return rows;
    }
}