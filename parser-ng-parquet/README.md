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
For PLAIN-encoded, little-endian, fixed-width numeric columns that are structurally REQUIRED
(`maxDefinitionLevel == 0`), `SimdBulkDecode` decodes a whole page at a time using the Vector API, which
is why `jdk.incubator.vector` is needed at compile time and run time. The numbers quoted in its source
(roughly 2x to 3.7x faster than the scalar loop for bulk int32/double decode) come from an informal,
non-JMH harness and should be re-measured on your hardware.

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
| SIMD bulk decode for PLAIN required fixed-width columns | yes |
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
* Nullable and nested columns pay per-value level handling beyond the flat, REQUIRED fast path.
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

Throughput in **millions of rows per second** (higher is better), JMH score with its 99.9% confidence
interval. The system was otherwise idle for these runs.

| Scenario | Windows 10 (JDK 24) | WSL2 (JDK 26) |
|---|---|---|
| Flat, full scan, sequential | 3.70 ± 0.10 | 3.64 ± 0.12 |
| Flat, full scan, parallelism 2 | 4.81 ± 1.06 | 5.37 ± 0.49 |
| Flat, full scan, parallelism 4 | 6.08 ± 0.47 | 6.60 ± 0.52 |
| Flat, full scan, parallelism 8 | 5.72 ± 0.65 | 5.93 ± 1.18 |
| Flat, projected (2 of 7 columns), sequential | 13.51 ± 3.38 | 14.97 ± 1.86 |
| Nested, full scan, sequential | 1.83 ± 0.03 | 1.79 ± 0.08 |
| Nested, full scan, parallelism 4 | 2.55 ± 0.18 | 2.69 ± 0.28 |
| Flat, selective predicate (~10%), sequential | 10.32 ± 0.39 | 8.85 ± 0.45 |

The selective row counts rows *emitted* (about 519,800 per scan: a superset of the ~500,000 that
match, because pushdown prunes at row-group and page granularity), not rows in the file; the scan
skips 2 of its 3 row groups.

**How to read these numbers**
* **The two environments agree.** Everything except the selective scan is within about 12% between the
  columns, which is inside the error bars for most rows. The selective scan runs for only tens of
  milliseconds, so fixed per-scan costs (opening the file, reading the footer, filesystem access) weigh
  more there and it is the scenario where an OS difference is most likely to show.
* **Parallel scaling is modest and saturates at the hardware.** Speedup over sequential is roughly
  1.3-1.5x at parallelism 2, 1.6-1.8x at 4, and no better at 8, on a 2-core / 4-thread chip. Nested
  columns gain about 1.4-1.5x at parallelism 4. Choose `parallelism(n)` at or below your logical CPU
  count; more threads only add contention. Parallel results also have wider error bars than sequential
  ones.
* **Projection pays off.** Decoding 2 of 7 columns is about 3.7-4.1x faster than the full scan, more than
  the 3.5x you would expect from column count alone, because the skipped columns include the string and
  nullable ones.
* **Rows per second is the figure to compare across scenarios; MB/s is now measured, not estimated.**
  The rows/s table above predates the byte counters. `ScanMetrics` now reports `uncompressedBytesDecoded()`
  (decompressed page bytes actually decoded, respecting projection and page pruning),
  `compressedBytesRead()` (as-stored bytes of the projected column chunks read; an upper bound under
  page-level pruning) and `arrowBytesProduced()`. Both benchmarks divide those by wall-clock time and print
  `decodedMB/s`, `diskMB/s` and `arrowMB/s` alongside rows/s. Quote `decodedMB/s` as decode throughput.
  These are warm-page-cache, end-to-end rates. Re-run on your hardware to fill in an MB/s column; none is
  claimed here.
* **Numbers are hardware-specific.** These come from a low-power laptop CPU with AVX2 (256-bit vectors).
  Re-measure on your target hardware before drawing conclusions; nothing here was compared against other
  Parquet readers.

## Use with parser-ng-sql
See the separate `parser-ng-sql-parquet` module for the bridge (`ScanPlanner`, `PredicateConverter`,
`ParquetSql`), which derives projected columns and a pruning predicate from a parsed `SelectStatement`.
Its planner is unit-tested against the real `sqlv1.ast` classes; the execution path that calls
`ArrowQuery` has not been run. `parser-ng-sql` is a test-scope dependency here only; this module never
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

Verify `parquet.version` (1.18.0), `arrow.version` (19.0.0, kept equal to parser-ng-arrow),
`hadoop.version` (3.4.1), and `jmh.version` (1.37) before release. JMH is a `provided` dependency, so it
is not pulled in by consumers of the jar.