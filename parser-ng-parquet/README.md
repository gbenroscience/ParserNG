# parser-ng-parquet

Read-only storage layer for ParserNG: Parquet file -> projected, row-group-pruned **Arrow
`VectorSchemaRoot`** batches, including nested (struct/list/map) columns, optionally decoded in
parallel across row groups. It does not parse SQL (parser-ng-sql) and does not evaluate expressions
(parser-ng-arrow). There is no writer.

> **Status: written without compiling or running against real dependencies.** The authoring
> environment has no Maven and no Parquet/Arrow jars. `LevelWalker` (the pure Dremel-level arithmetic
> nested types are built on) was compiled and unit-tested standalone — see below — but everything that
> touches parquet-java/Arrow types (`NodePlan`, `RowGroupDecoder`, `SequentialSource`, `ParallelSource`,
> `ParquetBatchReader`) has not been compiled. Expect first-build fixes; run the test suite before
> trusting any of it.

```java
try (BufferAllocator alloc = new RootAllocator();
     ParquetBatchReader r = ParquetScan.scan(path)
         .select("x", "y", "tags")               // only these columns are decoded; 'tags' may be nested
         .pushdown(Predicate.gt("x", 100.0))     // prunes row groups; result is a SUPERSET
         .parallelism(4)                          // decode up to 4 row groups concurrently
         .withMetrics(true).open(alloc)) {
    while (r.next()) {
        VectorSchemaRoot batch = r.root();       // valid until next next()/close()
        // hand to parser-ng-arrow / ArrowQuery
    }
    System.out.println(r.metrics());              // includes parallelism used
}
```

## Contracts
* **Pushdown is pruning, never filtering.** Emitted batches may contain non-matching rows; always
  re-apply the same condition downstream.
* **Ownership.** Caller owns the allocator. `detach()` moves/hands off the current batch (no copy) to a
  root the caller must close.
* **Nulls.** Validity bits are preserved at every level (leaf, struct, list); a null slot is never
  written as 0/false/""/empty-list.
* **Errors** are `ParquetScanException` carrying file / row group / column. No partial data is ever
  returned silently.
* **Ordering is unaffected by parallelism.** Row groups are decoded out of order across worker threads
  but always *delivered* in file order, and batches within a row group are always delivered in order —
  parallel and sequential scans of the same file produce byte-identical row sequences.

## Nested types
Handled: structs, 3-level `LIST`/`MAP` (and the legacy 2-level / "tuple" list encodings), bare
`repeated` fields, and any nesting of these over the supported primitive leaves (see the type table in
the previous release notes). The mapping from Parquet's repetition/definition levels to Arrow list
offsets and struct validity is Dremel's standard algorithm, isolated in `internal.LevelWalker` — pure
integer arithmetic with no Parquet/Arrow types, so it could be (and was) tested independently of the
rest of the stack. `internal.NodePlan` builds one level-aware plan per projected top-level column;
`RowGroupDecoder` reads each nested leaf's raw level stream, decodes present values straight into the
Arrow leaf vector, and replays the retained levels through `LevelWalker` to build every ancestor list's
offsets and every ancestor struct's validity bitmap. A column whose leaf type isn't supported (DECIMAL,
UUID, INT96, FIXED_LEN_BYTE_ARRAY, unsigned ints) fails fast naming that column; the rest of the file
still reads.

**Not implemented for nested columns:** predicate pushdown (pruning targets flat leaf columns only —
see `PredicateTranslator`; a predicate naming a path under a nested column is simply never pushed, which
is always safe, just not an optimization). Deeply and irregularly nested schemas have been reasoned
through and unit-tested at the `LevelWalker` level (list-of-list, struct-with-null-parent,
struct-inside-list) but not against an actual file written by parquet-java or another Parquet
implementation.

## Parallel reading
`ParquetScan.parallelism(n)` decodes up to `n` row groups concurrently, each on its own worker thread
with its own `ParquetFileReader` (parquet-java's per-codec decompressors aren't safe to share across
threads). Default is `1` (fully sequential, reuses one set of Arrow buffers every batch — zero steady-
state allocation). Parallel mode trades that buffer reuse for concurrency: each row group's batches are
freshly allocated, since they cross a thread boundary to reach the caller. Concurrency is bounded twice
over: exactly `n` decoders exist (never one task per row group), and at most `n + 1` row groups are ever
in flight, so a slow consumer can't let decoded-but-unconsumed batches pile up. The caller/planner
chooses `n`; nothing here adapts it automatically to file size, CPU count, or I/O characteristics.

## Feature matrix
| Area | Status |
|---|---|
| Projection (top-level columns, undecoded columns skipped) | yes |
| Row-group pruning: min/max stats, dictionary, Bloom (via parquet-java) | yes |
| Page/column-index pruning | not yet |
| Flat primitive types (see prior notes: ints, floats, STRING/ENUM/JSON, DATE, TIMESTAMP, etc.) | yes |
| Nested: struct, LIST (3-level + legacy), MAP, bare repeated, arbitrary nesting of these | yes (untested against a real file — see above) |
| DECIMAL, TIME, UUID, INT96, FLBA, unsigned ints (flat or nested) | clear error, not yet |
| Metadata inspection (footer only) | yes (`ParquetFileInfo`) |
| Metrics (row groups, rows, batches, decode time, parallelism used) | yes; bytes/pages not available |
| Bounded parallel row-group decoding | yes (`parallelism(n)`), ordering preserved |
| Writer | **removed from scope** |
| JMH benchmarks | not yet |
| Encryption, checksum verification toggles | not yet |

## What is and isn't zero-copy / allocation-free
* Decode: parquet-java decompresses and decodes; values are written once into Arrow buffers — a copy,
  not zero-copy.
* Sequential mode: Arrow buffers are allocated once and reused across batches; fixed-width columns
  decode with no boxing and no per-value allocation from this module (parquet-java's `getBinary()`
  allocates a small wrapper per string/binary value regardless).
* Parallel mode: each row group's output batches are freshly allocated, by design (see above).
* Throughput ceiling: one virtual `ColumnReader` call per value; parquet-java has no public batch-decode
  API. Nested columns pay one extra per-entry branch (against the retained level arrays) beyond the flat
  path. Matching native (C++/Rust) Parquet readers would need a custom page decoder.

## Use with parser-ng-sql
See the separate `parser-ng-sql-parquet` module for the bridge (`ScanPlanner`, `PredicateConverter`,
`ParquetSql`), which derives projected columns and a pruning predicate from a parsed `SelectStatement`.
Its planner is unit-tested against the real `sqlv1.ast` classes; the execution path that calls
`ArrowQuery` has not been run.

## Build
Add `<module>parser-ng-parquet</module>` to the parent POM. Verify `parquet.version` (1.18.0),
`arrow.version` (19.0.0, kept equal to parser-ng-arrow) and `hadoop.version` (3.4.1) before release.
