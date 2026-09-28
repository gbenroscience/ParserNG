# parser-ng-parquet

Read-only storage layer for ParserNG: Parquet file -> projected, row-group-pruned **Arrow
`VectorSchemaRoot`** batches, including nested (struct/list/map) columns, optionally decoded in
parallel across row groups and optionally filtered exactly, row by row. It does not parse SQL
(parser-ng-sql) and does not evaluate expressions (parser-ng-arrow). There is no writer.

* **Coordinates:** `com.github.gbenroscience:parser-ng-parquet:3.0.8`
* **Requires:** JDK 22+ with the incubator Vector API (`--add-modules jdk.incubator.vector`), Arrow 19.0.0,
  parquet-java 1.18.0, Hadoop client 3.4.1
* **Verified on:** JDK 24 (Windows 10) and JDK 26 (CentOS Stream 9 under WSL2). On **JDK 25 and newer** you
  must also pass `-Dio.netty.noUnsafe=false`; see [Build](#build).

## Quick start

```java
try (BufferAllocator alloc = new RootAllocator();
     ParquetBatchReader r = ParquetScan.scan(path)
         .select("id", "value", "name")           // only these columns are decoded
         .pushdown(Predicate.gt("id", 100L))      // prunes row groups/pages; result is a SUPERSET
         .parallelism(4)                          // decode up to 4 row groups concurrently
         .withMetrics(true).open(alloc)) {
    while (r.next()) {
        VectorSchemaRoot batch = r.root();       // valid until next next()/close()
        // hand to parser-ng-arrow / ArrowQuery
    }
    System.out.println(r.metrics());             // includes parallelism used
}
```

`ParquetScan` is an immutable spec: every builder method returns a new instance, and nothing is read
until `open(allocator)` (or `forEachBatch`).

| Builder method | Purpose | Default |
|---|---|---|
| `select(String...)` | Top-level columns to decode (may be nested columns) | all columns |
| `pushdown(Predicate)` | Pruning predicate (row groups + pages) | none |
| `exactFilter()` | Also filter rows exactly; needs `pushdown` first | off |
| `withCustomPredicate(id, impl)` | Register the test behind a `Predicate.custom(id, ...)` leaf | none |
| `batchSize(int)` | Max rows per batch | 32,768 |
| `parallelism(int)` | Row groups decoded concurrently | 1 (sequential) |
| `withMetrics(boolean)` | Collect `ScanMetrics` | off |
| `maxRowGroupBytes(long)` | Reject surviving row groups whose uncompressed size exceeds this (untrusted-input guard) | 2 GiB |

## Usage examples

All examples assume these imports and the sample files created in Example 7:

```java
import com.github.gbenroscience.parser.ng.parquet.v1.*;
import org.apache.arrow.memory.*;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.*;
```

### 1. Inspect a file without decoding any data

`ParquetFileInfo.read` reads only the footer: schema, row/row-group counts, and per-column-chunk
statistics, codec, encodings, and whether the file carries column/offset indexes or Bloom filters.

```java
ParquetFileInfo info = ParquetFileInfo.read(path);
System.out.println("rows=" + info.rowCount() + ", row groups=" + info.rowGroups().size());
System.out.println("top-level columns: " + info.topLevelColumns());

for (ParquetFileInfo.RowGroupInfo rg : info.rowGroups()) {
    for (ParquetFileInfo.ColumnChunkInfo c : rg.columns()) {
        System.out.printf("rg %d  %-10s %-8s min=%s max=%s nulls=%s bloom=%b%n",
                rg.index(), c.path(), c.codec(), c.min(), c.max(), c.nullCount(), c.hasBloomFilter());
    }
}
```

### 2. Projection plus pruning pushdown (with the mandatory re-check)

Pushdown only *prunes*. Row groups and pages whose statistics, dictionary, or Bloom filter prove that
no row can match are skipped, but the batches that come back can still contain non-matching rows, so
apply the same condition yourself. Use literals whose Java type matches the column (`long` for an
`int64` column) since a mismatched leaf is silently not pushed.

```java
try (BufferAllocator alloc = new RootAllocator();
     ParquetBatchReader r = ParquetScan.scan(path)
         .select("id", "value")
         .pushdown(Predicate.and(Predicate.ge("id", 40_000L), Predicate.lt("id", 40_100L)))
         .withMetrics(true)
         .open(alloc)) {

    long matched = 0;
    while (r.next()) {
        VectorSchemaRoot batch = r.root();
        BigIntVector id = (BigIntVector) batch.getVector("id");
        for (int i = 0; i < batch.getRowCount(); i++) {
            long v = id.get(i);
            if (v >= 40_000L && v < 40_100L) matched++;   // the re-check pushdown does NOT replace
        }
    }
    System.out.println("matched=" + matched + ", row groups skipped=" + r.metrics().rowGroupsSkipped());
}
```

### 3. Exact filtering with `exactFilter()`

When this scan is the last stage that filters (ad hoc queries, tests), `exactFilter()` re-tests every
row that survives pruning and emits only true matches. It also unlocks predicate types that can never
be pushed for pruning: `Not`, `Like`/`ilike`, `Regex`, `ColCmp`, and `Custom`. Every column the
predicate touches must be flat, top-level, and part of the projection, otherwise `open` throws a
`ParquetScanException` naming the column.

```java
Predicate p = Predicate.and(
        Predicate.ge("id", 1_000L),
        Predicate.and(
                Predicate.like("name", "ab%"),                  // SQL LIKE: % and _ wildcards
                Predicate.not(Predicate.isNull("value"))));

long rows = 0;
try (BufferAllocator alloc = new RootAllocator();
     ParquetBatchReader r = ParquetScan.scan(path)
         .select("id", "value", "name")
         .pushdown(p)
         .exactFilter()                                          // no downstream re-check needed
         .open(alloc)) {
    while (r.next()) rows += r.root().getRowCount();             // every emitted row matches
}
System.out.println("exact matches: " + rows);
```

Skip `exactFilter()` when a downstream engine (such as `parser-ng-arrow`) re-applies the condition
anyway: exact mode evaluates each row and copies survivors into freshly allocated batches, so it would
just pay for the same filtering twice.

### 4. A custom row-level predicate

For anything `Predicate` has no node for (computed expressions, cross-field business rules), name a
`Predicate.custom` leaf and register the implementation by id. It only runs under `exactFilter()`.
Null handling is yours: check `isNull` explicitly, and exceptions propagate out of the scan.

```java
CustomPredicate valueAboveHalfId = (batch, row) -> {
    BigIntVector id = (BigIntVector) batch.getVector("id");
    Float8Vector value = (Float8Vector) batch.getVector("value");
    return !value.isNull(row) && value.get(row) > id.get(row) / 2.0;
};

try (BufferAllocator alloc = new RootAllocator()) {
    ParquetScan.scan(path)
            .select("id", "value")
            .pushdown(Predicate.custom("value-above-half-id", "id", "value"))  // touched columns
            .withCustomPredicate("value-above-half-id", valueAboveHalfId)
            .exactFilter()
            .forEachBatch(alloc, batch ->                    // batch valid only inside the callback
                    System.out.println("kept " + batch.getRowCount() + " rows"));
}
```

### 5. Parallel scan with metrics

`parallelism(n)` decodes up to `n` row groups at once, each on its own worker thread and file handle.
Batches are still delivered in file order, so the row sequence is identical to a sequential scan.
Pick `n` yourself; nothing adapts it automatically.

```java
int n = Math.min(4, Runtime.getRuntime().availableProcessors());
try (BufferAllocator alloc = new RootAllocator();
     ParquetBatchReader r = ParquetScan.scan(path)
         .parallelism(n)
         .batchSize(65_536)
         .withMetrics(true)
         .open(alloc)) {
    long rows = 0;
    while (r.next()) rows += r.root().getRowCount();

    ScanMetrics m = r.metrics();      // null unless withMetrics(true)
    System.out.println("rows=" + rows + ", parallelism=" + m.parallelism()
            + ", row groups read=" + m.rowGroupsRead() + "/" + m.rowGroupsInFile()
            + ", decode ms (summed over threads)=" + m.decodeNanos() / 1_000_000);
}
```

### 6. Nested columns, and keeping a batch with `detach()`

Structs, lists, and maps come back as Arrow `StructVector`/`ListVector`/`MapVector` with validity
preserved at every level, so a null list, an empty list, and a list containing a null are all
distinguishable. `detach()` hands the current batch to you without copying; you must close it, and the
allocator must outlive it.

```java
List<VectorSchemaRoot> kept = new ArrayList<>();
try (BufferAllocator alloc = new RootAllocator();
     ParquetBatchReader r = ParquetScan.scan(nestedPath).select("id", "tags", "point").open(alloc)) {

    if (r.next()) {
        VectorSchemaRoot batch = r.root();
        ListVector tags = (ListVector) batch.getVector("tags");
        StructVector point = (StructVector) batch.getVector("point");
        for (int i = 0; i < Math.min(5, batch.getRowCount()); i++) {
            String t = tags.isNull(i) ? "null list"
                    : (tags.getElementEndIndex(i) - tags.getElementStartIndex(i)) + " element(s)";
            System.out.println("row " + i + ": point=" + (point.isNull(i) ? "null" : "present") + ", tags=" + t);
        }
        kept.add(r.detach());          // zero-copy handoff; the reader carries on with its next batch
    }
    while (r.next()) { /* ... */ }
    // use `kept` here, while alloc is still open
    kept.forEach(VectorSchemaRoot::close);
}
```

### 7. Generate test files

`util.RandomParquetFiles` writes reproducible random files for the flat and nested sample schemas (or
your own schema, using the supported types), which is handy for trying everything above.

```java
Path dir = Files.createTempDirectory("parquet-demo");
Path path = RandomParquetFiles.write(dir.resolve("flat.parquet"),
        RandomParquetFiles.SAMPLE_FLAT_SCHEMA, 200_000,
        RandomParquetFiles.Config.defaults()
                .seed(42)
                .nullProbability(0.1)
                .rowGroupSize(1 << 20));          // small row groups so there is something to prune/parallelize
Path nestedPath = RandomParquetFiles.write(dir.resolve("nested.parquet"),
        RandomParquetFiles.SAMPLE_NESTED_SCHEMA, 50_000,
        RandomParquetFiles.Config.defaults().seed(7).collectionSize(0, 8));
```

The flat sample has `id` (int64), `value` (double), `ratio` (float), `flag` (boolean), `name` (string),
`day` (date), and `seen_at` (timestamp millis). The nested sample has `id`, `tags` (list of string), and
`point` (struct of `x`, `y`). Note that values are uniformly random, so they carry no correlation with
row position and min/max statistics prune very little; for pruning experiments use a fixture whose
values are a function of the row index (see `DecodeBenchmark.writeSelectivityFixture`).

A runnable walkthrough covering examples 1, 2, 5, and 6 lives in
`v1/examples/ParquetOnlyExample.java`.

## Predicates

`Predicate` is a sealed, dependency-free model (no Arrow or Parquet types). What each node can do:

| Node (factory) | Pruning via `pushdown` | Exact via `exactFilter()` |
|---|---|---|
| `Cmp` (`eq/ne/lt/le/gt/ge`) | yes, flat columns, exact/widening literal type match only; string ranges and NaN are never pushed | yes |
| `In` (`in`) | yes | yes |
| `IsNull` (`isNull/isNotNull`) | yes | yes |
| `And` / `Or` (`and/or`) | yes (`And` drops untranslatable sides; `Or` is dropped entirely if either side is untranslatable) | yes |
| `Not` (`not`) | yes, rewritten with De Morgan's laws, never a raw Parquet `not()` | yes |
| `ColCmp` (`colEq`, `colLt`, ...) | never (statistics describe one column at a time) | yes; both columns must have the same Arrow type |
| `Like` (`like/ilike`), `Regex` (`regex`) | never | yes (Java regex semantics for `Regex`) |
| `Custom` (`custom`) | never (inert without `exactFilter()`) | yes, via `withCustomPredicate` |

A leaf that cannot be pushed simply prunes nothing, which is always sound.

## Contracts
* **Pushdown is pruning, never filtering.** Without `exactFilter()`, emitted batches may contain
  non-matching rows; always re-apply the same condition downstream.
* **Ownership.** Caller owns the allocator and it must outlive the reader and any detached roots.
  `root()` is valid until the next `next()`/`close()`. `detach()` moves the current batch (no copy) to a
  root the caller must close.
* **Nulls.** Validity bits are preserved at every level (leaf, struct, list); a null slot is never
  written as 0/false/""/empty-list.
* **Errors** are `ParquetScanException` carrying file / row group (index among surviving row groups) /
  column. No partial data is ever returned silently.
* **Ordering is unaffected by parallelism.** Row groups are decoded out of order across worker threads
  but always *delivered* in file order, and batches within a row group in order, so parallel and
  sequential scans produce identical row sequences.

## Supported types

| Parquet | Arrow |
|---|---|
| BOOLEAN | `Bool` |
| INT32 (plain, signed 32/16/8) | `Int(32/16/8)` |
| INT32 + DATE | `Date(DAY)` |
| INT64 (plain, signed 64) | `Int(64)` |
| INT64 + TIMESTAMP (millis/micros/nanos) | `Timestamp` (UTC-adjusted or not, as declared) |
| FLOAT / DOUBLE | `FloatingPoint(SINGLE/DOUBLE)` |
| BINARY | `Binary` |
| BINARY + STRING/ENUM/JSON | `Utf8` |

Unsupported leaves (DECIMAL, TIME, UUID, INT96, FIXED_LEN_BYTE_ARRAY, unsigned ints, BSON) fail fast,
naming the column, whether flat or nested. The native decoder also rejects the `DELTA_BINARY_PACKED`,
`DELTA_LENGTH_BYTE_ARRAY`, `DELTA_BYTE_ARRAY`, and `BYTE_STREAM_SPLIT` encodings by name rather than
mis-decoding them.

## Nested types
Handled: structs, 3-level `LIST`/`MAP` (and the legacy 2-level / "tuple" list encodings), bare
`repeated` fields, and any nesting of these over the supported primitive leaves. The mapping from
Parquet's repetition/definition levels to Arrow list offsets and struct validity is Dremel's standard
algorithm, isolated in `internal.LevelWalker` (pure integer arithmetic, testable independently of
Parquet/Arrow). `internal.NodePlan` builds one level-aware plan per projected top-level column;
`RowGroupDecoder` reads each nested leaf's raw level stream, decodes present values straight into the
Arrow leaf vector, and replays the retained levels through `LevelWalker` to build every ancestor list's
offsets and struct validity bitmap.

**Not implemented for nested columns:** predicate pushdown (pruning targets flat leaf columns only; a
predicate naming a path under a nested column is simply never pushed, which is safe, just not an
optimization) and `exactFilter()` (predicate columns must be flat and top-level; only a `Custom`
leaf's touched columns may be nested). Deeply and irregularly nested schemas have been reasoned through and unit-tested at the `LevelWalker`
level (list-of-list, struct-with-null-parent, struct-inside-list). End to end, the benchmarks scan nested
files written by parquet-java (struct plus list of strings); there is no value-level cross-check against
another Parquet reader yet.

## Pruning and parallel reading
Row-group pruning (min/max statistics, dictionary filter, Bloom filter) happens once up front against
the footer. Page-level pruning via column indexes is also enabled in both sequential and parallel mode.
Either way, `pushdown` alone only prunes.

`parallelism(n)` decodes up to `n` row groups concurrently, each on its own worker thread with its own
`ParquetFileReader` (parquet-java's per-codec decompressors aren't safe to share across threads).
Default is `1` (fully sequential, reusing one set of Arrow buffers every batch: zero steady-state
allocation). Parallel mode trades that buffer reuse for concurrency: each row group's batches are
freshly allocated, since they cross a thread boundary. Concurrency is bounded twice over: exactly `n`
decoders exist, and at most `n + 1` row groups are ever in flight, so a slow consumer can't let
decoded-but-unconsumed batches pile up. The caller/planner chooses `n`; nothing adapts it to file size,
CPU count, or I/O characteristics.

## Decode engine
Pages are decoded by a native cursor (`internal.decode.FastColumnCursor`) that reads page bytes
directly instead of going through parquet-java's converter-bound `ColumnReaderImpl`, with an RLE /
bit-packing decoder for levels and dictionary indexes, and a `DictionaryCache` that decodes each
dictionary page once so gathering a dictionary value is an array index rather than a per-row object.
For PLAIN-encoded, little-endian, fixed-width numeric columns (int32/int64/float/double), required or
nullable, `SimdBulkDecode` decodes a whole page's present values at a time using the Vector API, which
is why `jdk.incubator.vector` is needed at compile time and run time. Flat numeric columns are then
written to Arrow one page segment at a time (a tight typed loop, or a single typed gather for dictionary
pages, scattering by definition level when the column is nullable) instead of one virtual call per value.
Bit-unpacking of levels and dictionary indexes bounds-checks once per run rather than once per byte and has
dedicated paths for bit widths 1 and 2-8. The numbers quoted in `SimdBulkDecode`'s source (roughly 2x to 3.7x
faster than the scalar loop for bulk int32/double decode) come from an informal, non-JMH harness and
should be re-measured on your hardware.

## Feature matrix
| Area | Status |
|---|---|
| Projection (top-level columns, undecoded columns skipped) | yes |
| Row-group pruning: min/max stats, dictionary, Bloom (via parquet-java) | yes |
| Page/column-index pruning | yes (sequential and parallel) |
| Exact row filtering (`exactFilter()`), incl. `Not`, `Like`, `Regex`, `ColCmp`, `Custom` | yes (flat top-level columns only) |
| Flat primitive types (see Supported types) | yes |
| Nested: struct, LIST (3-level + legacy), MAP, bare repeated, arbitrary nesting | yes |
| DECIMAL, TIME, UUID, INT96, FLBA, unsigned ints (flat or nested) | clear error, not yet |
| DELTA_* and BYTE_STREAM_SPLIT encodings | clear error, not yet |
| Predicate pushdown / exact filtering on nested columns | not yet |
| Metadata inspection (footer only) | yes (`ParquetFileInfo`) |
| Metrics (row groups in file/skipped/read, rows, batches, decode time, parallelism, bytes decoded / read from disk / Arrow produced) | yes; page counts not available |
| Bounded parallel row-group decoding | yes (`parallelism(n)`), ordering preserved |
| SIMD bulk decode for PLAIN fixed-width numeric columns (required and nullable) | yes |
| JMH benchmarks | yes (`DecodeJmhBenchmark`); results in [Benchmarks](#benchmarks) |
| Writer | **removed from scope** |
| Encryption, checksum verification toggles | not yet |

## What is and isn't zero-copy / allocation-free
* Decode: values are decoded and written once into Arrow buffers, which is a copy, not zero-copy.
  Only `detach()` and downstream transfer are zero-copy.
* Sequential mode (no `exactFilter()`): Arrow buffers are allocated once and reused across batches;
  fixed-width columns decode with no boxing and no per-value allocation from this module.
* Parallel mode: each row group's output batches are freshly allocated, by design.
* `exactFilter()` mode: every emitted batch is freshly allocated by selective copy, regardless of
  parallelism, and every surviving row is evaluated once.
* Nested columns, and BOOLEAN/string/binary columns, still pay per-value handling beyond the flat numeric bulk path.
  No head-to-head comparison against native (C++/Rust) Parquet readers has been run.

## Benchmarks
Two drivers live in `v1.bench`, both over the same eight scenarios (full flat scan, projected scan,
parallel 2/4/8, ~10%-selective predicate, nested sequential, nested parallel):

* `DecodeBenchmark`: a plain `main` with 2 warmup + 5 timed iterations, taking the median. Optional
  args: flat row count, nested row count (defaults 5,000,000 and 1,000,000).
* `DecodeJmhBenchmark`: the same scenarios under JMH with fork isolation and JIT warmup.

```bash
mvn -Pjmh clean package
java --add-modules=jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     --add-opens=java.base/java.nio=ALL-UNNAMED \
     -Dio.netty.tryReflectionSetAccessible=true \
     -Dio.netty.noUnsafe=false \
     -Darrow.allocation.manager.type=Netty \
     -jar target/benchmarks.jar DecodeJmhBenchmark
```

JMH runs each benchmark in forked JVMs, so the runtime flags above must also be present in
`DecodeJmhBenchmark`'s `@Fork(jvmArgsAppend = {...})`; flags given only to the launcher are not enough.

### Results

**Test machine (laptop-class, deliberately modest):** Dell Inspiron 5759, Intel Core i7-6500U (Skylake,
2 cores / 4 threads, 2.5 GHz base, AVX2, 15 W class), 16 GB RAM, Windows 10 Pro (build 19045). The WSL2
column is a CentOS Stream 9 environment on the **same machine**, so the two columns differ in OS layer
and JDK, not in hardware.

| | Windows 10 | WSL2 (CentOS Stream 9) |
|---|---|---|
| JDK | 24 | 26.0.2 |
| JMH | 1.37, 2 forks, 2 x 1 s warmup + 5 x 1 s measurement (10 samples) | same |
| Fixtures | flat: 5,000,000 rows x 7 columns (22 row groups); nested: 1,000,000 rows (7 row groups); selectivity: 5,000,000 rows (3 row groups) | same |

Each cell is the JMH score with its 99.9% confidence interval (10 samples). **Mrows/s** is millions of
rows per second; **decoded MB/s** is `ScanMetrics.uncompressedBytesDecoded()` divided by wall-clock time,
in MiB/s (1 MB = 1,048,576 bytes), i.e. decompressed page bytes actually decoded, respecting projection and
page pruning. Higher is better for both.

| Scenario | Windows 10 (JDK 24) Mrows/s | Windows 10 decoded MB/s | WSL2 (JDK 26) Mrows/s | WSL2 decoded MB/s |
|---|---|---|---|---|
| Flat, full scan, sequential | 4.05 ± 0.11 | 150.8 ± 4.1 | 4.02 ± 0.12 | 149.7 ± 4.3 |
| Flat, full scan, parallelism 2 | 5.76 ± 0.48 | 214.7 ± 17.7 | 5.69 ± 0.52 | 212.2 ± 19.3 |
| Flat, full scan, parallelism 4 | 6.79 ± 0.37 | 253.2 ± 13.9 | 6.73 ± 1.18 | 250.8 ± 44.2 |
| Flat, full scan, parallelism 8 | 5.06 ± 1.84 | 188.5 ± 68.4 | 6.54 ± 0.65 | 243.6 ± 24.4 |
| Flat, projected (2 of 7 columns), sequential | 16.18 ± 1.17 | 230.8 ± 16.7 | 15.36 ± 1.25 | 219.1 ± 17.8 |
| Nested, full scan, sequential | 1.88 ± 0.06 | 101.5 ± 3.1 | 1.68 ± 0.29 | 90.6 ± 15.4 |
| Nested, full scan, parallelism 4 | 2.36 ± 0.52 | 127.2 ± 27.9 | 2.69 ± 0.38 | 145.1 ± 20.4 |
| Flat, selective predicate (~10%), sequential | 11.27 ± 0.52 | 86.0 ± 4.0 | 9.28 ± 0.63 | 70.8 ± 4.8 |

The selective row counts rows *emitted* (about 519,800 per scan: a superset of the ~500,000 that
match, because pushdown prunes at row-group and page granularity), not rows in the file; the scan
skips 2 of its 3 row groups.

The other two byte counters, in MB/s from the same runs (scores only; the confidence intervals are of the
same relative width as the decoded column above). `disk` is `compressedBytesRead()`: as-stored bytes of
the projected column chunks read, an upper bound under page-level pruning. `arrow` is
`arrowBytesProduced()`: Arrow buffer bytes materialized.

| Scenario | Windows 10 disk MB/s | Windows 10 arrow MB/s | WSL2 disk MB/s | WSL2 arrow MB/s |
|---|---|---|---|---|
| Flat, full scan, sequential | 136.3 | 167.4 | 135.3 | 166.1 |
| Flat, full scan, parallelism 2 | 194.1 | 238.3 | 191.9 | 235.6 |
| Flat, full scan, parallelism 4 | 228.9 | 281.0 | 226.8 | 278.4 |
| Flat, full scan, parallelism 8 | 170.4 | 209.3 | 220.2 | 270.4 |
| Flat, projected (2 of 7 columns), sequential | 200.4 | 250.7 | 190.2 | 238.0 |
| Nested, full scan, sequential | 91.8 | 115.6 | 81.9 | 103.1 |
| Nested, full scan, parallelism 4 | 115.0 | 144.8 | 131.2 | 165.2 |
| Flat, selective predicate (~10%), sequential | 71.2 | 174.7 | 58.6 | 143.8 |

**How to read these numbers**
* **The two environments agree closely on most rows.** Sequential, parallelism 2 and parallelism 4 flat
  scans are within about 1.2% of each other; the projected scan is within about 5%. Where they differ, the
  confidence intervals usually overlap: parallelism 8 (Windows 5.06 ± 1.84 against WSL2 6.54 ± 0.65) and the
  two nested scans (Windows is 12% ahead sequentially and 12% behind at parallelism 4, both within the
  wide intervals). The one difference the intervals do not explain is the selective scan: Windows is about
  21% faster (11.27 ± 0.52 against 9.28 ± 0.63 Mrows/s). That scan runs for only tens of milliseconds, so
  fixed per-scan costs (opening the file, reading the footer, filesystem access) weigh more there, which
  makes an OS-layer difference most likely to show; this run does not isolate the cause.
* **Parallel scaling is modest and saturates at the hardware.** On a 2-core / 4-thread chip, flat scans
  gain about 1.42x at parallelism 2 and about 1.68x at parallelism 4 in both environments (rows/s over the
  sequential run). Parallelism 8 is no better than 4: WSL2 measured 1.63x, and Windows measured 1.25x but
  with a relative error of about 36%, so treat that single figure as unreliable rather than as a
  Windows-specific slowdown. Nested scans at parallelism 4 gained 1.25x (Windows) and 1.60x (WSL2), with
  wide intervals in both, so this run only supports "roughly 1.3-1.6x". Choose `parallelism(n)` at or
  below your logical CPU count; more threads only add contention. Parallel results have wider error bars
  than sequential ones.
* **Projection pays off.** Decoding 2 of 7 columns is about 3.8-4.0x faster than the full scan in rows/s,
  a little more than the 3.5x column-count ratio alone would suggest. Decoded MB/s also *rises* (about 150
  to 220-231 MB/s) rather than falling, which is consistent with the skipped columns (strings, booleans,
  nullable columns) being the slower per-byte paths; this run does not measure per-column cost directly.
* **Decoded MB/s is a decode rate, not a scan rate, and it is only comparable within a fixture.** It counts
  decompressed page bytes, so it depends on row width and encoding. The flat fixture decodes about 39 bytes
  per row, the projected scan about 15, the nested fixture about 56, and the selective fixture only about 8
  (its `id` column is 8 plain bytes and its 10-value `bucket` column is dictionary-encoded down to a few
  index bytes). That is why the selective scan has the second-highest rows/s but the lowest decoded MB/s.
  Compare rows/s across scenarios that differ in pruning or projection, and decoded MB/s between runs of
  the same scenario or between environments.
* **Best measured decode throughput is about 250 MB/s** (flat, parallelism 4, both environments); the
  sequential flat scan runs at about 150 MB/s. Parallel scans decode the same bytes per row as sequential
  ones (39.08 bytes/row at every parallelism, derived from these results), so the counters do not
  double-count under parallelism.
* **Arrow bytes exceed decoded bytes when decoding expands data.** About 1.1x for the flat and nested
  scans, and about 2.0x for the selective scan, where the dictionary-encoded `bucket` column is expanded
  to full 8-byte Arrow values (16.25 Arrow bytes per row against 8 decoded).
* **Compression is weak on this data.** Decoded/disk is only 1.1-1.2x because the fixtures come from
  `RandomParquetFiles`, which writes uniformly random values; the disk MB/s figures characterize that
  data, not typical real files. `compressedBytesRead()` counts whole column chunks of surviving row
  groups, so it overstates the bytes touched when page-level pruning skips pages inside them.
* **JMH prints every one of these rows with the unit `ops/s`.** The aux-counter rows (`:rows`,
  `:decodedMB`, `:compressedMB`, `:arrowMB`) are per second of the named quantity, not operations; read
  `:rows` as rows/s and the `MB` rows as MB/s.
* **These are warm-page-cache, end-to-end rates**, from a single machine and one benchmark run per
  environment. Re-run on your hardware before drawing conclusions.
* **Numbers are hardware-specific.** These come from a low-power laptop CPU with AVX2 (256-bit vectors).
  Nothing here was compared against other Parquet readers.

## Use with parser-ng-sql
See the separate `parquet` package of the `parser-ng-sql` module for the bridge (`ScanPlanner`, `PredicateConverter`,
`ParquetSql`), which derives projected columns and a pruning predicate from a parsed `SelectStatement`. `parser-ng-sql` is a test-scope dependency here only; this module never
depends on the SQL or Arrow-query modules at compile time.

## Build
Add `<module>parser-ng-parquet</module>` to the parent POM (`parser-ng-parent`, version 3.0.8), then:

```bash
mvn -pl parser-ng-parquet -am package
```

The compiler is configured for `release 22` with `--add-modules jdk.incubator.vector`. Surefire and any
program that uses this module at runtime need, in addition, the flags below (Arrow's Netty allocator
needs the `java.nio` opens):

```
--add-modules=jdk.incubator.vector
--enable-native-access=ALL-UNNAMED
--add-opens=java.base/java.nio=ALL-UNNAMED
-Dio.netty.tryReflectionSetAccessible=true
-Dio.netty.noUnsafe=false
-Darrow.allocation.manager.type=Netty
```

**JDK 25 and newer:** recent Netty versions disable their `sun.misc.Unsafe` access by default on JDK 25+,
and Arrow's Netty allocator cannot initialize without it. Without `-Dio.netty.noUnsafe=false` the first
`new RootAllocator()` fails with an `ExceptionInInitializerError` whose cause is an
`UnsupportedOperationException` from `EmptyByteBuf.memoryAddress`. The flag is harmless on older JDKs. If
it is not enough on your JDK, also try `--sun-misc-unsafe-memory-access=allow`.

Verify `parquet.version` (1.18.0), `arrow.version` (19.0.0, kept equal to parser-ng-arrow's arrow version),
`hadoop.version` (3.4.1), and `jmh.version` (1.37) before release. JMH is a `provided` dependency, so it
is not pulled in by consumers of the jar.