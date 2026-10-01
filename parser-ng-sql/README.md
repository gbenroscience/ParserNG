# parser-ng-sql

**Make SQL a convenient way to describe vectorized Apache Arrow computations.**

parser-ng-sql is a `SELECT`-only SQL front end for [ParserNG](https://github.com/gbenroscience/ParserNG)'s
Arrow integration, parser-ng-arrow. It compiles a SQL statement once into a
small set of ParserNG expression evaluators, then lets you run that compiled
query against as many `VectorSchemaRoot` batches as you like. It is not, and
is not trying to become, a database: there is no catalog, no `JOIN`, no
`INSERT`/`UPDATE`/`DELETE`, no transactions. There is exactly one job;
turning a `SELECT` statement into a fast, reusable, vectorized computation
over an Arrow batch you already have in hand; and it is built to do that
job well rather than to do many jobs adequately.

```java
ArrowQuery query = ArrowQuery.compile(
        "SELECT sqrt(x*x + y*y) AS distance FROM data WHERE x > 10");

VectorSchemaRoot result = query.execute(root); // run it as many times as you like
query.close();
```

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Why SQL, why this shape](#why-sql-why-this-shape)
- [Language reference](#language-reference)
  - [SELECT list](#select-list)
  - [WHERE](#where)
  - [CASE / WHEN](#case--when)
  - [CAST](#cast)
  - [GROUP BY and aggregates](#group-by-and-aggregates)
  - [HAVING](#having)
  - [ORDER BY and LIMIT](#order-by-and-limit)
  - [Referencing a SELECT-list alias](#referencing-a-select-list-alias)
- [Execution model](#execution-model)
  - [Compile once, execute many](#compile-once-execute-many)
  - [Choosing an execution backend](#choosing-an-execution-backend)
  - [Null handling](#null-handling)
  - [Reusable output buffers (zero-allocation execution)](#reusable-output-buffers-zero-allocation-execution)
  - [Warming up](#warming-up)
- [Performance notes](#performance-notes)
- [Error handling](#error-handling)
- [Querying Parquet files](#querying-parquet-files)
- [What's intentionally out of scope](#whats-intentionally-out-of-scope)
- [Resource ownership and thread-safety](#resource-ownership-and-thread-safety)

## Installation

parser-ng-sql is a module of the main [ParserNG](https://github.com/gbenroscience/ParserNG)
repository and is published to Maven Central alongside it.

**Maven**

```xml
<dependency>
    <groupId>com.github.gbenroscience</groupId>
    <artifactId>parser-ng-sql</artifactId>
    <version>3.0.8</version>
</dependency>
```

**Gradle**

```groovy
implementation("com.github.gbenroscience:parser-ng-sql:3.0.8")
```

parser-ng-sql depends on `parser-ng-arrow`, `parser-ng`, `parser-ng-simd`, and
`parser-ng-gpu-simd` transitively; a plain dependency declaration pulls in
everything needed to compile and run a query, including Apache Arrow's own
`arrow-vector` artifact. Requires JDK 22 or later.

## Quick start

```java
import com.github.gbenroscience.sqlv1.ArrowQuery;
import org.apache.arrow.vector.VectorSchemaRoot;

// root is a VectorSchemaRoot you already have -- from a file, a stream,
// wherever your data comes from.
try (ArrowQuery query = ArrowQuery.compile(
        "SELECT x, y, sqrt(x*x + y*y) AS magnitude "
      + "FROM data "
      + "WHERE magnitude > 60")) {

    try (VectorSchemaRoot result = query.execute(root)) {
        // result has columns x, y, magnitude -- only the rows where
        // magnitude > 60 -- ready to hand off to whatever's next.
    }
}
```

For a single one-off run where you don't need to keep the compiled query
around, `ArrowSql.execute` is a shorter equivalent:

```java
try (VectorSchemaRoot result = ArrowSql.execute(root,
        "SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data WHERE magnitude > 60")) {
    // ...
}
```

## Why SQL, why this shape

ParserNG's Arrow integration already gives you fast, vectorized `filter`,
`project`, and `filterProject` operations driven by expression strings. What
it doesn't give you is a convenient way to *describe* those operations;
you're writing predicate and projection text by hand, keeping track of which
columns feed which computation, and re-deriving the same expression twice if
you need it in both a filter and a projection. SQL is a format almost every
engineer already knows for exactly that kind of description. parser-ng-sql
takes that familiar syntax and compiles it straight down to the same
`ArrowExpressionEvaluator`s you'd have written by hand; nothing about the
underlying computation changes, only how convenient it is to say what you
want.

## Language reference

A query has this shape:

```
SELECT select_list
FROM table_reference
[WHERE boolean_expression]
[GROUP BY expression_list]
[HAVING boolean_expression]
[ORDER BY order_item [, order_item ...]]
[LIMIT integer_literal]
```

Every arithmetic expression anywhere in a query; a `SELECT` item, a `WHERE`
operand, an aggregate's argument, an `ORDER BY` key; is ordinary ParserNG
expression syntax: `+ - * / % ^`, parentheses, and any ParserNG function
(`sqrt`, `sin`, `erf`, `if`, and everything else ParserNG registers).
Column names are looked up directly against the `VectorSchemaRoot` you pass
to `execute`.

### SELECT list

```sql
SELECT * FROM data
```
Every column of the input batch, unchanged.

```sql
SELECT x, y FROM data
```
A column subset; no computation, no filtering.

```sql
SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM data
```
Passthrough columns and a computed column, aliased with `AS`. An unaliased
computed column is named after its own (trimmed) expression text.

### WHERE

```sql
SELECT x, y FROM data WHERE x > 20 AND y < 90
SELECT x, y FROM data WHERE x < 20 OR x > 70
SELECT x, y FROM data WHERE NOT x > 50
SELECT x, y FROM data WHERE x BETWEEN 20 AND 60
SELECT x, y FROM data WHERE x IN (10, 50, 90)
SELECT reading FROM data WHERE reading IS NULL
SELECT reading FROM data WHERE reading IS NOT NULL
```

`AND`/`OR`/`NOT` compose and nest with parentheses as you'd expect:

```sql
SELECT x, y FROM data
WHERE (x > 20 AND y < 90) OR (x < 5 AND y > 50)
```

A `WHERE` operand can itself be a parenthesized boolean condition; useful
for feeding a boolean result into an otherwise-arithmetic position, like a
`CASE` condition built by hand or a nested `if`:

```sql
SELECT x, if(x > 50, 1, 0) AS high_flag FROM data
```

### CASE / WHEN

Both the *searched* and *simple* forms of `CASE` are supported. Every
`CASE` must end in an explicit `ELSE`; there is no numeric `NULL` literal
for a missing branch to fall back to, so a missing `ELSE` is rejected at
compile time with a clear message rather than failing confusingly later.

**Searched form**; a `WHEN` clause per condition, evaluated in order:

```sql
SELECT
    x,
    CASE
        WHEN x < 10 THEN sin(x)
        WHEN x < 50 THEN erf(x)
        ELSE x^3
    END AS result
FROM data
WHERE x > 0
```

**Simple form**; one operand, compared for equality against each `WHEN`
value:

```sql
SELECT
    category,
    CASE category
        WHEN 1 THEN 100
        WHEN 2 THEN 200
        ELSE -1
    END AS label
FROM data
```

`CASE category WHEN 1 THEN ...` is exactly equivalent to hand-writing
`CASE WHEN category = 1 THEN ...`; the simple form is pure convenience,
not a different capability.

### CAST

`CAST(expression AS type)` supports numeric target types; since every
ParserNG value is already a floating-point number under the hood, `CAST`
here means "how should this number be interpreted," not a general type
system:

```sql
SELECT reading, CAST(reading AS INT) AS truncated FROM data
```

`INT`, `INTEGER`, `SMALLINT`, `BIGINT`, and `LONG` all truncate toward
zero; `-2.7` becomes `-2`, not `-3`, matching a Java `(long)` cast, not
mathematical floor. `DOUBLE`, `FLOAT`, `REAL`, `NUMERIC`, and `DECIMAL` are
a no-op identity, since that's already the representation in use. Any other
target type is rejected at compile time with a clear message rather than
silently doing nothing useful.

### GROUP BY and aggregates

`SUM`, `COUNT`, `AVG`, `MIN`, and `MAX` are recognized wherever they appear
as a top-level `SELECT`-item function call. `COUNT(*)` counts every row in
a group; every other aggregate skips `null` inputs, matching standard SQL
aggregate semantics; including that `SUM`/`AVG`/`MIN`/`MAX` over a group
with no non-null values come back `null`, while `COUNT` comes back `0`.

```sql
SELECT
    category,
    SUM(reading)  AS total,
    COUNT(*)      AS n,
    AVG(reading)  AS avg_reading,
    MIN(reading)  AS min_reading,
    MAX(reading)  AS max_reading
FROM data
GROUP BY category
```

An aggregate query needs no explicit `GROUP BY` at all; with none present,
the whole (filtered) input is treated as a single implicit group, exactly
as plain SQL does:

```sql
SELECT SUM(reading) AS total, COUNT(*) AS n
FROM data
WHERE reading > 0
```

A grouped query's non-aggregate `SELECT` items must match a `GROUP BY` key
expression exactly; the usual SQL rule that a plain column in the select
list of a grouped query has to be functionally determined by the grouping
key. This is checked once, at compile time:

```sql
-- rejected: 'reading' is neither aggregated nor a GROUP BY key
SELECT category, reading, SUM(reading) FROM data GROUP BY category
```

`GROUP BY` accepts more than one key, and a key can be any expression, not
just a bare column:

```sql
SELECT category, subcategory, SUM(reading) AS total
FROM data
GROUP BY category, subcategory
ORDER BY category, subcategory
```

### HAVING

`HAVING` filters groups by an aggregate result, after grouping; the same
boolean-expression grammar as `WHERE`, evaluated against the grouped
result rather than individual rows:

```sql
SELECT category, SUM(reading) AS total
FROM data
GROUP BY category
HAVING total > 50
```

`HAVING` only applies to a grouped query (an explicit `GROUP BY`, or at
least one aggregate `SELECT` item); using it otherwise is rejected at
compile time.

### ORDER BY and LIMIT

```sql
SELECT x, y FROM data ORDER BY x DESC LIMIT 3
```

Multiple sort keys, mixed direction, are supported:

```sql
SELECT category, reading FROM data ORDER BY category ASC, reading DESC
```

`NULL`s sort last regardless of `ASC`/`DESC`; a `null` is "unknown," not
the smallest or largest value.

For a query with no `GROUP BY`, `ORDER BY` conceptually sorts the
`FROM`/`WHERE` row set before the `SELECT` list narrows it down to a
smaller list of columns; so it can name a column that isn't even in the
`SELECT` list:

```sql
-- only x is projected, but the sort key is y
SELECT x FROM data ORDER BY y DESC
```

It can also name a `SELECT`-list alias, resolved back to the expression it
stands for, the same way `WHERE` resolves one (see the next section):

```sql
SELECT x, y, sqrt(x*x + y*y) AS magnitude
FROM data
ORDER BY magnitude DESC
LIMIT 3
```

For a grouped query, `ORDER BY` instead sorts the final grouped result, so
it names a `GROUP BY` key or an aggregate's own alias:

```sql
SELECT category, SUM(reading) AS total
FROM data
GROUP BY category
ORDER BY total DESC
LIMIT 2
```

`LIMIT` always applies last, after any `ORDER BY`.

### Referencing a SELECT-list alias

A `WHERE`/`HAVING`/`ORDER BY` clause can name a `SELECT`-list alias
directly instead of repeating its expression:

```sql
SELECT x, y, sqrt(x*x + y*y) AS magnitude
FROM data
WHERE magnitude > 60
```

is exactly equivalent to writing the expression out by hand:

```sql
SELECT x, y, sqrt(x*x + y*y) AS magnitude
FROM data
WHERE (sqrt(x*x + y*y)) > 60
```

A real input column always wins over a same-named alias, and a chain of
aliases (one alias's expression naming a second alias) resolves
transitively.

## Execution model

### Compile once, execute many

`ArrowQuery.compile(sql)` only parses the SQL text; cheap, and independent
of any particular Arrow schema. The actual ParserNG expression evaluators
are compiled lazily on the first call to `execute`, because whether
ParserNG compiles `float64` or `float32` kernels depends on the column
types of the batch you pass in. Once built, that compiled plan is cached
and reused on every subsequent `execute` call against a batch with the
same schema; the SQL parsing and expression compilation genuinely happen
once, no matter how many batches you run through it:

```java
try (ArrowQuery reusable = ArrowQuery.compile(
        "SELECT x, y, x * y AS product FROM data WHERE x > 15")) {

    VectorSchemaRoot result1 = reusable.execute(batch1);
    VectorSchemaRoot result2 = reusable.execute(batch2); // same compiled plan, no recompilation
}
```

The plan is cached by schema shape (column names and types), not by which
`VectorSchemaRoot` instance you pass in; so this works exactly the same,
and just as cheaply, when `batch1` and `batch2` are two different objects
with the same column layout, which is the normal case for a streaming or
per-request workload where a fresh batch arrives on every call. If a later
`execute` call passes a batch with a genuinely different schema (different
column names or types), the plan is transparently recompiled; you don't
need to detect that yourself.

### Choosing an execution backend

By default, compiled expressions target ParserNG's CPU SIMD backend.
`withBackend` selects a different one:

```java
ArrowQuery query = ArrowQuery.compile(sql)
        .withBackend(ArrowExecutionBackend.GPU_AUTO);
```

Available backends: `CPU_SIMD`, `GPU_AUTO`, `GPU_CUDA`, `GPU_OPENCL`,
`GPU_METAL`. Changing the backend after a plan has already been compiled
invalidates and recompiles it on the next `execute`.

### Null handling

`withNullPolicy` controls how a `null` input is treated:

```java
ArrowQuery query = ArrowQuery.compile(sql)
        .withNullPolicy(NullPolicy.PROPAGATE);
```

`NullPolicy.PROPAGATE` (the default) produces a computed value everywhere
the inputs allow one, and `null` only where an input actually was;
standard, predictable null propagation through arithmetic. Unlike
`withBackend`, changing the null policy never requires recompilation.

### Reusable output buffers (zero-allocation execution)

`execute(VectorSchemaRoot)` allocates a fresh, independently-owned result
on every call. That's the right default, and it's what almost every
application should use. But allocating and zeroing a new set of output
vectors on every call is a real, measurable cost once you're running the
same query thousands of times a second over similarly-sized batches; the
kind of workload where you'd otherwise reach for a hand-tuned engine like
Apache Gandiva, which preallocates its output once and writes into it
repeatedly.

For exactly that situation, `ArrowQuery` offers a caller-owned-buffer
overload:

```java
try (ArrowQuery query = ArrowQuery.compile(
        "SELECT x, y, sqrt(x*x + y*y) AS distance FROM data WHERE x > 10");
     VectorSchemaRoot output = query.allocateReusableOutput(schemaTemplate, /* maxRows */ 65_536)) {

    for (VectorSchemaRoot batch : incomingBatches) {
        query.execute(batch, output); // writes into `output`; no allocation once warmed up
        // ... consume `output` before the next call overwrites it ...
    }
}
```

`allocateReusableOutput` builds an output root sized for up to `maxRows`
rows per column, shaped and typed to match what the query produces.
`execute(root, output)` then writes each call's result into that same
buffer instead of allocating a new one. In steady state; once its
internal scratch buffers have grown to the largest batch size you've
passed; this overload performs no heap allocations at all, including for
a `WHERE`-narrowed selection.

This comes with real constraints, by design:

- **Only supported for queries without `GROUP BY`, `HAVING`, `ORDER BY`, or
  `LIMIT`.** Those clauses can change the output row count in ways a
  fixed-capacity buffer can't safely bound; using this overload with one
  of them throws `UnsupportedOperationException` immediately, rather than
  failing confusingly mid-run. Use `execute(root)` for those queries.
- **Not safe for concurrent use on a single `ArrowQuery` instance.** The
  reusable buffer and its scratch state belong to one query, one thread at
  a time; this is the trade-off for the zero-allocation guarantee. If you
  need this on multiple threads, give each thread its own `ArrowQuery`
  (compiled from the same SQL text) and its own reusable output.
- **Every column in `output` must already have capacity for at least
  `root.getRowCount()` rows.** Nothing is resized on your behalf;
  `allocateReusableOutput` exists specifically so you size it once,
  correctly, up front.

Reach for this overload when you're in a tight, latency-sensitive loop;
streaming ingestion, a hot query re-run per incoming micro-batch, or a
benchmark against an engine that itself preallocates its output. For
everything else, `execute(root)` is simpler and just as fast in every way
that matters for typical usage.

### Warming up

The first execution of any compiled expression pays a one-time JIT
warm-up/classload cost, the same as any hot Java code path. If your
service's very first real request needs to already be fast; rather than
paying that cost on live traffic; call `warmup` once, right after
compiling:

```java
try (ArrowQuery query = ArrowQuery.compile(sql)) {
    query.warmup(schemaTemplate); // synthetic data, same schema, same code paths
    // ... query is now primed; the real cached plan is what warmup exercised ...
}
```

`warmup` builds a synthetic batch of the same schema as `schemaTemplate`
(20,000 rows, repeated 8 times, by default; both configurable via the
overload that takes `rows` and `repetitions`), runs the real query against
it the requested number of times, and discards the results. It primes the
exact same cached plan your real traffic will use, so the JIT/classload
cost is paid once, deliberately, at start-up rather than unpredictably on
whichever request happens to be first in line.

## Performance notes

A few design choices worth knowing about if you're deciding whether
parser-ng-sql fits a performance-sensitive path:

- **A passthrough column is never copied when the full column is kept.**
  `SELECT x, y FROM data` returns `x` and `y` as zero-copy references into
  the already-filtered batch, not fresh row-by-row copies; copying only
  happens where it's actually needed (a computed column, a passthrough
  column that's being row-narrowed by a `WHERE` clause, or one that's also
  being renamed while narrowed).
- **`GROUP BY` and aggregate arguments are evaluated in bulk, once, not per
  row and not per group.** Every `GROUP BY` key and every distinct
  aggregate argument is evaluated across the whole filtered batch in a
  single pass before grouping happens, so aggregation cost scales with the
  number of input rows, not with the number of distinct groups.
- **The compiled-plan cache check is allocation-free, and keyed by schema,
  not by object identity.** Confirming a cached plan still matches the
  current batch's schema is a handful of primitive comparisons, not a
  fresh collection or wrapper object built on every call; and this holds
  whether or not you're passing the exact same `VectorSchemaRoot` instance
  back in. A brand-new batch on every call, with the same column layout as
  the last one, still takes the fast, lock-free path.
- **A predicate or projection expression referenced more than once is
  compiled once.** If a `WHERE` clause resolves a `SELECT`-list alias back
  to the same expression a projection already needs, both share a single
  compiled evaluator rather than compiling it twice.
- **For the tightest hot loops, the reusable-output overload adds no
  allocation overhead of its own on top of parser-ng-arrow.** See
  [Reusable output buffers](#reusable-output-buffers-zero-allocation-execution)
  above; this is the overload to reach for when comparing against engines
  that preallocate their output, such as Apache Gandiva, on cheap
  expressions (simple arithmetic, `sqrt`, and similar) where per-call
  fixed overhead matters most.

## Error handling

parser-ng-sql draws a clear line between two kinds of failure, and tries
hard to keep problems on the compile-time side of that line rather than
letting them surface as a confusing runtime failure:

- **`SqlSyntaxException`**; the SQL text itself doesn't parse: a missing
  keyword, an unsupported `CAST` target type, a `CASE` with no `ELSE`. This
  is thrown as early as possible, from `ArrowQuery.compile`, before any
  ParserNG expression is even compiled.
- **`ArrowSqlException`**; the SQL parsed fine, but compiling its
  expressions against a particular batch's schema failed: a grouped
  query's `SELECT` item doesn't match a `GROUP BY` key, a cyclic
  `SELECT`-list alias reference, and similar shape mismatches that can
  only be detected once a real schema is in hand. Thrown from the first
  `execute` call against a given schema.
- **`ArrowBindingException`**; thrown directly by parser-ng-arrow itself
  when a compiled expression can't actually run against the data in front
  of it (a column genuinely missing from the batch, for instance).
- **`UnsupportedOperationException`**; thrown by `execute(root, output)`
  or `allocateReusableOutput` if the query uses `GROUP BY`, `HAVING`,
  `ORDER BY`, or `LIMIT`; see
  [Reusable output buffers](#reusable-output-buffers-zero-allocation-execution).
- **`IllegalArgumentException`**; thrown by `execute(root, output)` if
  `output` doesn't match the query's column shape, or doesn't have enough
  per-column capacity for `root`'s row count.

## Querying Parquet files

`com.github.gbenroscience.sqlv1.parquet` runs the same SQL over Parquet files through
[parser-ng-parquet](../parser-ng-parquet). parser-ng-parquet never depends on this module;
the bridge lives here.

### Quick start

```java
ParquetSql sql = ParquetSql.create(ParquetSqlOptions.defaults().withParallelism(4))
        .register("events", Path.of("data/events.parquet"));

try (BufferAllocator alloc = new RootAllocator();
     ParquetSqlPlan plan = sql.plan(
         "SELECT id, name, x * 2 AS x2 FROM events WHERE x > 100 AND id < 5000")) {
    plan.stream(alloc, batch -> consume(batch));   // bounded memory
    System.out.println(plan.lastMetrics());        // row groups skipped, rows read, ...
    System.out.println(plan.explain());
}

// One-shot, no registration; the path is the source:
try (VectorSchemaRoot r = sql.execute("SELECT COUNT(*) FROM 'data/events.parquet'", alloc)) { ... }
```

`EXPLAIN SELECT ...` returns the plan as a one-column result named `plan`.

### How `FROM` is resolved

| Form | Resolution |
| --- | --- |
| `FROM 'dir/file.parquet'` | The quoted text is the file. Always resolved. |
| `FROM read_parquet('dir/*.parquet', 'other.parquet')` | One or more paths or globs, expanded and sorted. All files must have identical column names and types for the columns the query reads, else a named error. |
| `FROM events` | Looked up among tables passed to `register(name, pathOrGlob)`. An unregistered name is an error. |
| `ParquetSql.plan(file, sql)` with `FROM anything` | The name is ignored and `file` is the source (the original behaviour). Combining an explicit file with a quoted path in the SQL is rejected as ambiguous. |
| `ArrowSql` / `ArrowQuery` (in memory) | The name is never resolved; it is documentation only. |

**Security.** A path in SQL text is an unauthenticated file read on the machine running the
query. If any user-influenced SQL can reach `ParquetSql`, set
`ParquetSqlOptions.withAllowedRoot(dir)`: every file, after symlink resolution, must live under
that directory or the query fails.

### What runs where

* **Numeric work** (expressions, `WHERE`, `GROUP BY`, aggregates, `ORDER BY`) runs in
  `ArrowQuery`'s vectorized double kernels. Only the numeric columns a query actually touches are
  widened to double, into buffers reused across batches.
* **Every column can be selected as-is** in its native Arrow type. Such passthrough columns never
  enter the engine: the engine emits surviving row indices and the native batch is gathered by
  index. A pure projection of plain columns skips the engine and hands the reader's batch through
  with no copy.
* **Row-group pruning** uses a superset predicate derived from the `WHERE`. The full `WHERE` is
  always re-evaluated, unless you opt into `withExactFilter(true)` and the whole `WHERE` converts
  with exactly the SQL semantics (see below).
* **`ORDER BY ... LIMIT n`** (no `GROUP BY`) is a bounded top-N: each batch is sorted and cut to `n`, and the
  bridge keeps at most about `2n` candidate rows however many rows match. Ordering is the engine's: keys
  compare as doubles, nulls last in both directions, ties keep file order. `n` must be at most
  `maxMaterializedRows` and 1,000,000; otherwise, and for `ORDER BY` with no `LIMIT`, every matching row is
  materialized. `explain()` reports `bounded top-n`.
* **`SELECT COUNT(*)`** (only `COUNT(*)` items, no `WHERE`, `GROUP BY`, `HAVING` or `ORDER BY`)
  is answered from the footer row count: no data page is decoded and `lastMetrics()` is `null`.

### Type matrix

| Parquet type | Arrow vector | `SELECT` as-is | In expressions / `WHERE` / `ORDER BY` / `GROUP BY` | Pruning pushed |
| --- | --- | --- | --- | --- |
| INT32, INT(8/16) | Int / SmallInt / TinyInt | native | yes (widened to double, exact) | yes |
| INT64 | BigInt | native, exact | yes; a value beyond 2^53 is an error (see below) | yes |
| UINT8/16/32 | UInt1/2/4 | native | yes (exact) | no |
| UINT64 | UInt8 | native | yes; beyond 2^53 is an error | no |
| FLOAT | Float4 | native | yes (float to double is exact) | yes (bounds rounded outward) |
| DOUBLE | Float8 | native | yes, no copy | yes |
| DECIMAL (INT32/INT64/FLBA/BINARY) | Decimal128 | native, exact | yes, as double (lossy past about 15 significant digits; error beyond 2^53 unscaled) | yes, for columns up to 15 digits (INT32/INT64-backed: ranges; BINARY/FLBA-backed: `=` and `<>` only); never in exactFilter mode |
| BOOLEAN | Bit | native | `WHERE` only: `flag`, `= TRUE/FALSE`, `IS [NOT] TRUE/FALSE`, `IN`, `IS NULL`; not in arithmetic | no |
| STRING / ENUM / JSON | VarChar | native | `WHERE` only: `= <> < <= > >=`, `IN`, `BETWEEN`, `LIKE`, `ILIKE`, `REGEXP`, `IS NULL`; no string functions | yes (`=`, `<>`, `IN` only; no string range pruning) |
| BINARY, UUID | VarBinary / FixedSizeBinary | native | clear error | no |
| DATE | DateDay | native | `WHERE` only: comparisons, `IN`, `BETWEEN`, `IS NULL`, with `DATE '...'` or an ISO string | yes |
| TIMESTAMP (all units, UTC or local), INT96 | TimeStamp | native | `WHERE` only: comparisons, `IN`, `BETWEEN`, `IS NULL`, with `TIMESTAMP '...'` | no (needs a parser-ng-parquet translator change) |
| struct / list / map | nested | native (whole struct decoded) | clear error | no |

A clear error names the column, its type and the clause, for example
`Column 'name' is STRING and cannot be used in an expression, WHERE, GROUP BY or ORDER BY.`

**Precision rule.** A double holds integers exactly up to 2^53. When widening an int64, uint64 or
decimal for arithmetic would round, the query fails and names the column, value and file. Opt in to
rounding with `withInt64Policy(ROUND_TO_DOUBLE)`; `plan.roundedValues()` counts how many values were
rounded. Selecting such a column as-is is always exact. Nulls are preserved exactly, never replaced
by a sentinel.

**Aggregates return `DOUBLE`** (the engine's convention), including `COUNT`. Exact up to 2^53.

**`OR` and nulls.** String, boolean, date, timestamp and `IS NULL` leaves are two-valued (null gives false), so `OR` is correct for them. When the `WHERE` contains an `OR`, simple numeric leaves (a column against a numeric literal, `BETWEEN`, `IN`) are also evaluated as two-valued masks over the widened column, so `a > 5 OR b > 5` keeps a row where `a` is null but `b > 5`. A numeric leaf that is an expression (`a + 1 > 5`) is still evaluated by `ArrowQuery` with null propagation, so under an `OR` it can drop a row whose other branch is true when an input is null. `AND`-only filters keep the fused engine path and are unaffected.

### Predicates on strings, booleans, dates and timestamps

These run in the bridge, directly on the Arrow vectors, and feed the engine a 0/1 mask. Nothing is
allocated per row except `REGEXP` (one `String` per non-null row).

* **Strings.** `=`, `<>`, `<`, `<=`, `>`, `>=`, `IN`, `BETWEEN` and their negations; quotes escape as `''`.
  Ordering is unsigned UTF-8 byte order (code point order), not locale collation.
* **`LIKE` / `NOT LIKE` / `ILIKE`** with `%` (any run) and `_` (exactly one character, a whole UTF-8
  sequence), optional `ESCAPE 'c'` (no escape character unless you give one). Matching is bytewise with fast
  paths for equality, prefix, suffix and substring. **`ILIKE` folds ASCII letters only**; `'É'` does not match `'é'`.
* **`REGEXP` / `RLIKE`**: `java.util.regex` `find()` (matches anywhere). Invalid patterns fail at plan time.
* **Nulls.** A null never matches `LIKE`, `NOT LIKE`, `=`, `<>`, `IN`, `NOT IN`. `x NOT IN (..., NULL)` never matches.
  `IS [NOT] TRUE/FALSE` and `IS [NOT] NULL` are two-valued: `IS NOT TRUE` keeps nulls.
* **Booleans.** `WHERE flag` means `flag = TRUE`; `NOT flag` means `flag <> TRUE`, so nulls stay excluded.
  Only `=` and `<>` are allowed against `TRUE`/`FALSE`.
* **Dates.** `DATE '2024-01-31'`, or an ISO string literal, against a DATE column.
* **Timestamps.** `TIMESTAMP '2024-01-31 12:00:00[.fraction]'`, optional `Z` or `+HH:MM`, or an ISO string.
  A column adjusted to UTC compares by instant (a literal without a zone is read as UTC). A column that is
  not adjusted to UTC holds wall-clock time: a zoned literal is an error (ambiguous). All units (seconds to
  nanoseconds) and INT96 work, and a literal finer than the column unit compares exactly (`=` cannot match).
* **Clear errors** name the column and type: `name = 5`, `id = 'x'`, `flag < TRUE`, two non-numeric columns
  compared with each other, `LIKE` on a non-string column, binary and nested columns in `WHERE`.
* **Not pushed to Parquet:** `LIKE`, `REGEXP`, boolean tests and timestamps. String and date comparisons are
  pushed as a pruning hint only; the predicate is always re-evaluated.
* `WHERE x` on a numeric column now parses (as `x = TRUE`) and then fails with a type error; it used to be a
  syntax error.

### Scan options

`ParquetSqlOptions`: `parallelism`, `batchSize`, `maxRowGroupBytes` (unset means parser-ng-parquet's
own defaults, so the modules cannot drift), `metrics` (default on), `exactFilter` (default off),
`maxMaterializedRows` (default 20,000,000), `int64Policy`, `allowedRoot`.

`exactFilter` lets the scan do the row filtering and drops the `WHERE` from the engine. It applies
only when every leaf converts exactly: numeric columns only, float32 literals exactly representable,
int64 literals within 2^53, no dropped branches. Otherwise it is not applied and `explain()` says so.
The parser-ng-parquet README explains the cost of exact filtering.

### Memory and modes

Queries without `ORDER BY`, `GROUP BY` or aggregates are streamable: `stream()` processes one batch
at a time in bounded memory and stops reading once `LIMIT` is met. `ORDER BY` and grouped queries
materialize the rows that survive row-group pruning (capped by `maxMaterializedRows`, which fails
loudly) and run the engine once. Streaming allocates one small result root per batch; the widened
buffers and row-index buffer are reused.

### Feature status

| Feature | Status |
| --- | --- |
| `SELECT` any column type as-is, across row groups, with nulls | yes |
| Numeric expressions, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT` | yes |
| Row-group pruning for int, long, float, double, date, string equality and short decimals, with correct literal typing | yes |
| `FROM 'path'`, `read_parquet('glob')`, `register(name, path)` | yes |
| Allow-listed root for paths | yes |
| `EXPLAIN` | yes |
| `COUNT(*)` from the footer, no decode | yes |
| Bounded top-N for `ORDER BY ... LIMIT n` | yes |
| Scan options exposed | yes |
| Precision error for int64 beyond 2^53 | yes |
| Boolean, string, date, timestamp `WHERE` predicates, `IS [NOT] TRUE/FALSE`, `IS NULL` on any column | yes |
| Exact decimal predicates | not yet (compared as double) |
| Arithmetic on strings, dates, timestamps | clear error |
| `LIKE`, `ILIKE` (ASCII folding), `REGEXP`, `ESCAPE` | yes |
| String functions (`LOWER`, `UPPER`, `LENGTH`, `TRIM`, `SUBSTRING`, `CONCAT`) | not yet |
| `DATE '...'`, `TIMESTAMP '...'` literals | yes |
| `CAST` to date/timestamp/decimal, `EXTRACT`, `INTERVAL` | not yet |
| Timestamp row-group pruning | not yet (needs a parser-ng-parquet change) |
| Quoted identifiers (`"my col"`) | not yet |
| Nested field access (`s.field`, `items[0]`, `m['k']`) and `CARDINALITY` | not yet (whole struct can be selected) |
| `COUNT(col)`, `COUNT(DISTINCT)`, `SELECT DISTINCT`, `MIN`/`MAX` from statistics | not yet (`MIN`/`MAX`/`COUNT(col)` run through the engine on numeric columns) |
| Bounded top-N for `ORDER BY ... LIMIT` | not yet (materializes surviving rows) |
| `COALESCE`, `NULLIF`, `OFFSET` | not yet |

## What's intentionally out of scope

parser-ng-sql deliberately does not implement:

- `JOIN`, subqueries, or any multi-table concept; every query operates on
  exactly one `VectorSchemaRoot` you already have in hand.
- `INSERT`, `UPDATE`, `DELETE`, transactions, or a catalog of any kind;
  there is no notion of a persistent table to mutate.
- `DISTINCT`.
- Non-numeric `CAST` targets or general string/text processing; every
  value in a ParserNG expression is a floating-point number. Over Parquet,
  string, boolean, date and timestamp predicates are supported; string functions,
  casts to temporal types and decimal predicates are not yet (see the feature status table above).
- Multi-table queries. `read_parquet('glob')` reads several files as one
  table with one schema; it is not a join.

None of these are missing by oversight. The goal is a fast, predictable way
to describe a vectorized Arrow computation, not a general-purpose query
engine; see the tagline at the top of this document.

## Resource ownership and thread-safety

An `ArrowQuery` owns whatever expression evaluators it has compiled and
must be closed when no longer needed; a try-with-resources block, as in
every example above, is the simplest way. Every `VectorSchemaRoot` returned
by `execute(root)` is a fresh, independently-owned batch that the caller
owns and must close in turn; parser-ng-sql never returns a view over, or a
root that shares ownership with, the root you passed in (passthrough
columns may share refcounted buffers with the input, exactly as Arrow's
own `TransferPair` does; closing either side is always safe).

**`execute(VectorSchemaRoot)` is safe to call concurrently from multiple
threads on the same `ArrowQuery`**; including concurrently with
`withBackend`, `withNullPolicy`, and `close()`, and concurrently with the
one-time recompilation that happens the first time a new schema shape is
seen or the backend changes. The plan a given call uses is reference-
counted and lock-free: a plan that's been superseded (by a schema change or
a `withBackend` call) has its evaluators released only once every call
still using it has finished, so no in-flight call is ever disrupted by a
change made from another thread.

**`execute(VectorSchemaRoot, VectorSchemaRoot)` (the reusable-output
overload) and `allocateReusableOutput` are not safe for concurrent use on
one `ArrowQuery` instance**; see
[Reusable output buffers](#reusable-output-buffers-zero-allocation-execution)
above. Use one query instance per thread if you need this overload from
more than one thread at a time; each instance can be compiled from the
same SQL text independently.