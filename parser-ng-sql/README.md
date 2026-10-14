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
  - [`SELECT DISTINCT` in `ArrowQuery`](#select-distinct-in-arrowquery)
  - [Null handling](#null-handling)
  - [Reusable output buffers (zero-allocation execution)](#reusable-output-buffers-zero-allocation-execution)
  - [Warming up](#warming-up)
- [Performance notes](#performance-notes)
- [Error handling](#error-handling)
- [Querying Parquet files](#querying-parquet-files)
  - [Differences between ArrowQuery and ParquetSql](#differences-between-arrowquery-and-parquetsql)
- [Benchmarking and profiling ParquetSql](#benchmarking-and-profiling-parquetsql)
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
    <version>3.0.9</version>
</dependency>
```

**Gradle**

```groovy
implementation("com.github.gbenroscience:parser-ng-sql:3.0.9")
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
SELECT [DISTINCT] select_list
FROM table_reference
[WHERE boolean_expression]
[GROUP BY expression_list]
[HAVING boolean_expression]
[ORDER BY order_item [, order_item ...]]
[LIMIT integer_literal]
```

`SELECT DISTINCT` is described in [its own section](#select-distinct-in-arrowquery) (numeric columns, per batch).

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

> `ParquetSql` also evaluates `CAST(x AS DATE | TIMESTAMP | DECIMAL(p,s) | VARCHAR)`; see the date and time section of
> the Parquet chapter. The numeric targets below are the same in both. Plain `ArrowQuery` / `ArrowSql` rejects the
> other targets at compile time with a `SqlSyntaxException` ("only supported by ParquetSql").

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

That implicit group exists even when no row survives the `WHERE` (or the batch is empty): the query
returns exactly one row, with `COUNT` equal to `0` and `SUM`, `AVG`, `MIN` and `MAX` `null`. A query with
an explicit `GROUP BY` returns one row per group, so an empty input gives no rows.

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

Keys compare by value: `0.0` and `-0.0` fall into one group, and all `null` keys form one group.

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

In `ArrowQuery`, `HAVING` names a `GROUP BY` key or an aggregate's alias. Spelling the aggregate out
(`HAVING SUM(reading) > 50`) parses but is not supported there: it fails when the query executes, with an
`ArrowBindingException`. `ParquetSql` accepts the spelled-out form.

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
types of the batch you pass in: `float64` only when every column is a
`Float8Vector`, otherwise every expression in the query runs as `float32`. Once built, that compiled plan is cached
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
invalidates and recompiles it on the next `execute`. For queries over Parquet
files, set it on the options instead: `ParquetSqlOptions.withBackend(...)`
(see Scan options).

### `SELECT DISTINCT` in `ArrowQuery`

`ArrowQuery.compile(sql)` (and so `ArrowSql`) supports `SELECT DISTINCT` over numeric columns:

```java
try (ArrowQuery q = ArrowQuery.compile("SELECT DISTINCT x FROM t WHERE y > 0 ORDER BY x DESC LIMIT 10");
     VectorSchemaRoot out = q.execute(batch)) { ... }
```

* **Per batch.** Duplicates are removed within the one root passed to `execute`, never across calls. To
  de-duplicate a stream of batches use `ParquetSql`, which merges batches itself.
* **Order and clauses.** Rows keep their first-seen order. `ORDER BY` (select-list columns or aliases only, otherwise
  an `ArrowSqlException`) and `LIMIT` run after de-duplication, so `LIMIT` cannot shortcut the projection. It also
  applies to a grouped result (`SELECT DISTINCT SUM(y) ... GROUP BY x`).
* **Equality.** Two rows are equal when every column is equal: a `NULL` equals only another `NULL` (not `NaN`),
  `-0.0` equals `0.0`, all `NaN`s are equal, and integer columns compare exactly.
* **Numeric output columns only.** A string, boolean, date or other non-numeric output column fails with a clear
  `ArrowSqlException`; use `ParquetSql` for those.
* **Cost.** Boxed row keys held in memory, and every surviving row is projected before de-duplication.
* **Not supported:** `COUNT(DISTINCT x)`, `OFFSET`, `execute(root, reusableOutput)` / `allocateReusableOutput`
  (`UnsupportedOperationException`), and `SELECT DISTINCT` via `ArrowQuery.compile(label, stmt)`.

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

In a `WHERE` clause a null predicate result excludes the row. Without `IS [NOT] NULL` the clause runs as one
fused expression in which a comparison against a null input is itself null, and that null propagates through
`AND`, `OR` and `NOT`. Under `OR` this means a row is dropped when one branch is null even if the other branch
is true: with `a` null and `b = 10`, `a > 5 OR b > 5` does not keep the row, and neither does
`NOT (a > 10 AND b > 5)` (it is evaluated as `a <= 10 OR b <= 5`). SQL's three-valued logic would keep both. A
`WHERE` that contains `IS [NOT] NULL` anywhere is evaluated leaf by leaf instead, with a null leaf counted as
false, so a null branch no longer hides a true one under `OR`. `ParquetSql` evaluates simple numeric leaves
under `OR` as two-valued masks (see "`OR` and nulls"), so the same query can return different rows from the two
entry points; see [Differences between ArrowQuery and ParquetSql](#differences-between-arrowquery-and-parquetsql).

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

- **Only supported for queries without `SELECT DISTINCT`, `GROUP BY`, `HAVING`,
  `ORDER BY`, or `LIMIT`.** Those can change the output row count in ways a
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
  keyword, an unsupported `CAST` target type, a `CASE` with no `ELSE`, `SELECT *`
  combined with `GROUP BY` (this applies to `ParquetSql` too, which lets the
  parser's `SqlSyntaxException` through unwrapped). This
  is thrown as early as possible, from `ArrowQuery.compile`, before any
  ParserNG expression is even compiled.
- **`ArrowSqlException`**; the SQL parsed fine, but compiling its
  expressions against a particular batch's schema failed: a grouped
  query's `SELECT` item doesn't match a `GROUP BY` key, a cyclic
  `SELECT`-list alias reference, a `SELECT DISTINCT` whose `ORDER BY` names a column outside the select
  list, a non-numeric `SELECT DISTINCT` column (raised while executing), and similar shape mismatches that can
  only be detected once a real schema is in hand. Thrown from the first
  `execute` call against a given schema.
- **`ArrowBindingException`**; thrown directly by parser-ng-arrow itself
  when a compiled expression can't actually run against the data in front
  of it (a column genuinely missing from the batch, for instance).
- **`UnsupportedOperationException`**; thrown by `execute(root, output)`
  or `allocateReusableOutput` if the query uses `SELECT DISTINCT`, `GROUP BY`,
  `HAVING`, `ORDER BY`, or `LIMIT`; see
  [Reusable output buffers](#reusable-output-buffers-zero-allocation-execution).
  `ArrowQuery.compile` also throws it for `OFFSET` and `COUNT(DISTINCT ...)`.
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
| `FROM 'dir/file.parquet'` | The quoted text is the file. Always resolved. A `scheme://` path (`s3a://`, `gs://`, ...) is remote and needs a `ParquetSourceResolver`; see Remote storage below. |
| `FROM read_parquet('dir/*.parquet', 'other.parquet')` | One or more paths or globs, expanded and sorted. All files must have identical column names and types for the columns the query reads, else a named error. |
| `FROM events` | Looked up among tables passed to `register(name, pathOrGlob)`. An unregistered name is an error. |
| `ParquetSql.plan(file, sql)` with `FROM anything` | The name is ignored and `file` is the source (the original behaviour). Combining an explicit file with a quoted path in the SQL is rejected as ambiguous. |
| `ArrowSql` / `ArrowQuery` (in memory) | The name is never resolved; it is documentation only. |

**Remote storage.** A path written as `scheme://...` (`s3a://`, `gs://`, `hdfs://`, `abfss://`) is remote.
parser-ng-parquet reads remote data through `ParquetSource`, but this module bundles no connector, so you supply
a `ParquetSourceResolver` that turns the path (glob included) into sources:

```java
Configuration conf = ...;                                   // credentials, endpoint (hadoop-aws etc. on your classpath)
ParquetSqlOptions o = ParquetSqlOptions.defaults().withSourceResolver(spec -> {
    org.apache.hadoop.fs.Path p = new org.apache.hadoop.fs.Path(spec);
    List<ParquetSourceResolver.ResolvedSource> out = new ArrayList<>();
    for (FileStatus st : p.getFileSystem(conf).globStatus(p)) {
        out.add(new ParquetSourceResolver.ResolvedSource(ParquetSource.hadoop(st.getPath().toString(), conf)));
    }
    return out;
});
ParquetSql sql = ParquetSql.create(o);
sql.execute("SELECT region, SUM(revenue) AS total FROM read_parquet('s3a://analytics/sales/*.parquet') "
          + "WHERE year = 2026 GROUP BY region ORDER BY total DESC LIMIT 20", allocator);
```

Without a resolver, a remote path is rejected with a clear error, so remote reads are opt-in. Resolved
sources are read in name order and must have identical columns, as with local globs. Everything else
(pruning, ranged reads, `OFFSET`, grouped folds) works the same; only the projected column chunks of surviving
row groups are fetched, and parser-ng-parquet reads each footer once per scan. Local paths are unchanged.
Not verified against real S3, GCS, ABFS or HDFS (the parser-ng-parquet README says the same): tests use a
local `InputFile` standing in for remote storage. Retry and timeout behaviour is the connector's.

**Security.** `withAllowedRoot` restricts local files only. A remote path in SQL text is an unauthenticated read
with whatever credentials your resolver holds, so if user-influenced SQL can reach `ParquetSql`, the resolver
must enforce its own bucket and prefix allowlist and throw otherwise.

**Security (local files).** A path in SQL text is an unauthenticated file read on the machine running the
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
* **`ORDER BY ... LIMIT n`** (no `GROUP BY`) is a bounded top-N. Per batch the best `n` rows are picked with a
  bounded heap (`TopSelect`), not by sorting the batch: O(rows) when most rows cannot enter, and a row that cannot
  beat the current n-th best over all batches is dropped before it touches the heap. The picked rows are copied
  straight from their source vectors into a running result that holds at most about `2n` candidates however many
  rows match. When there is no `WHERE`, no computed output and the keys are plain numeric columns, the numeric
  engine is not involved at all (the keys are read from the widened columns); otherwise the engine evaluates the
  keys and the filter for the batch, without sorting, and the heap picks among the survivors. Ordering is the
  engine's: keys compare as doubles, nulls last in both directions, ties keep file order. `n` (plus any `OFFSET`)
  must be at most `maxMaterializedRows` and 1,000,000; otherwise, and for `ORDER BY` with no `LIMIT`, every matching
  row is materialized. `explain()` reports `bounded top-n` and whether the engine is involved.
* **Streaming aggregation.** `COUNT`, `SUM`, `AVG`, `MIN` and `MAX`, with or without `GROUP BY`, `HAVING`,
  `ORDER BY` and `LIMIT`, no longer copy the whole scan into one root. The engine aggregates each batch into
  partial values (a count; a sum, minimum or maximum together with the count of values behind it) and the bridge
  merges them (`PartialAgg`, `MergeCore`), so memory is one batch plus one row per group. `AVG` is merged as a sum
  over a count. `HAVING`, `ORDER BY`, `LIMIT` and `OFFSET` are applied to the merged groups. A partial sum, minimum
  or maximum is used only when its count is positive, so whatever the engine returns for an empty batch never
  leaks into the result. Differences from the materializing path, all as SQL defines them: `SUM`, `AVG`, `MIN` and
  `MAX` over no values are null, a `HAVING` that evaluates to null does not keep the group, and groups come out in
  first-seen order unless `ORDER BY` says otherwise. A statement the merge cannot handle exactly (a `DISTINCT`, an
  `ORDER BY` or `HAVING` operand that is not an output or an aggregate) still takes the materializing path.
  `explain()` reports `streaming partial aggregation`.
* **Allocation in the Java folds.** A single string key, `COUNT(DISTINCT string)` and string `MIN`/`MAX` no longer
  create an object per row: strings are hashed from a reused buffer (`ByteKeyTable`), integer, date, timestamp,
  boolean and floating-point keys as `long`s (`LongKeyTable`), and string extremes are compared in place and copied
  only when they become the new best. Several keys that are all strings or long-hashable (integers, dates, timestamps,
  booleans, floating point) use one composite table with a shared byte arena (`CompositeKeyTable`), also without
  per-row allocation; only other key types (decimal, nested, binary) are boxed. Materializing paths pre-size the combined
  vectors from the known row and byte counts, so appending copies each value once.
* **`SELECT DISTINCT`, `COUNT(DISTINCT col)`, `COUNT(col)` on any column, and `MIN`/`MAX` over strings, dates,
  timestamps and booleans** are folded in Java over the native batches of an inner scan that keeps your `FROM`
  and `WHERE` (so row-group pruning and native predicates apply). Rules:
  * Without `GROUP BY` here; with `GROUP BY` see the next bullet. Without `GROUP BY` every aggregate argument
    must be a plain column; `COUNT(*)`, `SUM`, `AVG` and numeric `MIN`/`MAX` may
    appear in the same query. `COUNT`, `SUM`, `AVG` and numeric `MIN`/`MAX` return `Float8` (as the numeric
    engine does); string, date, timestamp and boolean `MIN`/`MAX` keep the column type. Strings compare in
    UTF-8 byte order. Aggregates over no rows give `COUNT` 0 and the others null. A 64-bit integer or decimal
    beyond 2^53 fails loudly.
  * `SELECT DISTINCT` accepts columns and numeric expressions, `ORDER BY` select-list columns or aliases
    (nulls last, ties in file order) and `LIMIT`. Without `ORDER BY`, `LIMIT n` stops the scan after `n`
    distinct rows. Distinct rows are held in memory, capped by `maxMaterializedRows`. `-0.0` equals `0.0` and
    nulls equal each other.
  * `DISTINCT` is a keyword only where a select item can follow it, so a column named `distinct` still works.
    `SUM(DISTINCT x)` and `AVG(DISTINCT x)` are syntax errors; `ArrowQuery` itself rejects both forms.
    `ArrowQuery` also rejects `COUNT(DISTINCT x)`, and rejects `SELECT DISTINCT` when it is compiled from a statement
    (`compile(label, stmt)`, which is what this bridge uses), because the bridge merges batches itself.
  * These queries cannot be streamed (`stream()` throws); string, long-hashable and composite keys of those are looked up
    without allocation; other key types are boxed, so those are correct and bounded but not zero-allocation.
* **`GROUP BY` the numeric engine cannot do** is folded too (`explain()` says `GROUP BY the numeric engine
  cannot do`): a string, boolean, date or timestamp group key or key expression (`GROUP BY name`,
  `GROUP BY LOWER(name)`, `GROUP BY day`, `GROUP BY CAST(ts AS DATE)`), `COUNT(DISTINCT x)`, `MIN`/`MAX` over
  strings, dates, timestamps and booleans, and `SELECT DISTINCT` with `GROUP BY`. A grouped query with only
  numeric keys and numeric aggregates still runs in the numeric engine.
  * The scan is an inner projection of the group keys and aggregate arguments, so each may be any expression
    the bridge understands (`COUNT(DISTINCT LOWER(name))`, `SUM(a + b)`, a nested field, a cast). Groups are
    found by hashing the key values, and nulls form their own group. Keys keep their native type (a date key is
    a `DATE` column), copied from the group's first row. Aggregates have the types and null rules described
    above.
  * `HAVING` combines comparisons, `BETWEEN`, `IN`, `IS NULL`, `AND`, `OR` and `NOT` over numeric aggregates
    (by alias, or spelled out such as `COUNT(*)`, even when not selected) and numbers.
  * `ORDER BY` names a select-list column, alias or aggregate (nulls last, ties in first-seen order). Without it
    the groups come out in first-seen order. `GROUP BY` may name a select alias. `SELECT DISTINCT` removes
    duplicate output rows after grouping. `LIMIT` and `OFFSET` apply to the result.
  * Every select item must be a group key or an aggregate (a named error otherwise), `SELECT *` is rejected,
    and the group count is capped by `maxMaterializedRows`. `SUM`/`AVG` over a string, date, timestamp or
    boolean is a named error. A `HAVING` over a string, date or timestamp `MIN`/`MAX` is not supported.
* **`CAST` of a string to a number.** `CAST(x AS INTEGER | SMALLINT | BIGINT | DOUBLE | FLOAT | REAL | NUMERIC |
  DECIMAL)` where `x` is a string column, string literal, string function, `||` or nested string leaf parses the
  value's bytes strictly into a `Float8` column. Integer targets accept `[+-]digits` and check the target's
  range; the floating targets accept `[+-]digits[.digits][e[+-]digits]` (no `NaN`, `Infinity`, hex or suffixes).
  Surrounding spaces are ignored, a null stays null, and anything else, an empty string included, fails with
  an error naming the expression and the value. An integer beyond 2^53 is an error because a `Float8` cannot
  hold it exactly. A cast of a number is unchanged. `ParquetSql` asks the parser to keep numeric casts as SQL
  text (`SqlParser.parse(sql, true)`) so it can tell the two apart; `ArrowSql` still gets the desugared form.
  Every row of a batch is parsed, including rows the `WHERE` later drops, so one bad value fails the query.
* **`OFFSET`**: `LIMIT n OFFSET m`, `OFFSET m LIMIT n` and a bare `OFFSET m` (optionally followed by `ROW` or
  `ROWS`) are accepted by the bridge. Every execution path produces at most `LIMIT + OFFSET` rows and the
  first `m` are then dropped by a zero-copy slice, so a bounded top-N keeps `n + m` candidates and a
  streamed scan skips `m` rows and stops after `n`. `OFFSET` without `ORDER BY` skips rows in file order,
  which is not a stable order across files, as in any SQL engine. `OFFSET` is a keyword only as a trailing
  clause (`OFFSET`, an integer, then the end or `LIMIT`), so a column named `offset` still works.
  `ArrowQuery` itself rejects `OFFSET` rather than ignore it.
* **String functions.** `LOWER(s)`, `UPPER(s)`, `TRIM(s)`, `LTRIM(s)`, `RTRIM(s)`, `SUBSTRING(s, start [, count])` (alias `SUBSTR`),
  `CONCAT(a, b, ...)`, `a || b`, and `LENGTH(s)` (aliases `CHAR_LENGTH`, `CHARACTER_LENGTH`) and
  `OCTET_LENGTH(s)`. ParserNG evaluates doubles and cannot hold a string, so the bridge evaluates these
  itself, directly over the UTF-8 bytes of the Arrow vectors, and only the results meet the engine.
  * `LENGTH`/`OCTET_LENGTH` return a number (`Float8`, null for a null string), so they work everywhere a
    number does: arithmetic, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING` and inside aggregates
    (`SUM(LENGTH(name))`).
  * The others return a string. They work in the select list and as the operand of a string predicate
    (`WHERE LOWER(name) = 'x'`, `LIKE`, `ILIKE`, `REGEXP`, `IN`, `BETWEEN`, `IS NULL`), and nest
    (`UPPER(TRIM(SUBSTRING(a || b, 2, 5)))`). The output column is `VarChar`, named after the original text
    unless aliased.
  * Arguments are string columns, string literals, numeric literals (taken as their text) and other string
    expressions. A non-string column is an error that names it. `SUBSTRING` start and count must be integer
    literals.
  * Semantics: a null argument gives null, except `CONCAT`, which skips nulls (all null gives the empty
    string), while `||` gives null if any operand is null. `LENGTH` counts characters (UTF-8 code points),
    `OCTET_LENGTH` bytes. `SUBSTRING` is 1-based by character; positions before 1 count as in PostgreSQL
    (`SUBSTRING('abc', 0, 2)` is `'a'`), a start past the end gives the empty string, a negative count is an
    error. `TRIM` removes spaces (U+0020) at both ends, `LTRIM` at the start, `RTRIM` at the end. `LOWER`/`UPPER` fold ASCII by byte and use
    `String.toLowerCase(Locale.ROOT)`/`toUpperCase` only for a value with a non-ASCII byte, so `UPPER('Straße')`
    is `'STRASSE'`.
  * Not supported, with a named error: a string-valued expression as an `ORDER BY` or `GROUP BY` key, as an
    aggregate argument (`COUNT(DISTINCT LOWER(name))`), or mixed with arithmetic (`LOWER(a) + 1`); the
    `SUBSTRING(s FROM n FOR m)` and `TRIM(BOTH ... FROM ...)` forms; numeric columns inside
    `CONCAT`. A string predicate over a function is never pushed to row-group pruning.
  * Cost: each derived column is evaluated for every row of a batch, including rows the `WHERE` later drops,
    into a reusable vector. Buffers are reused, so steady state allocates nothing per row except for a value
    with non-ASCII bytes under `LOWER`/`UPPER`. `explain()` lists the derived columns.
* **Quoted identifiers.** `"my col"` and `` `my col` `` name a column whose name is not a plain identifier
  (spaces, hyphens, a leading digit, a reserved word, non-ASCII letters); a doubled delimiter stands for
  itself (`"a""b"`). They work in the select list, `WHERE`, `GROUP BY`, `ORDER BY`, aggregates, `DISTINCT`
  and inside string functions. A quoted alias (`AS "out name"`) names an output only; referencing it in
  `WHERE` or `ORDER BY` is an error that names it, so repeat the expression instead.
  **Breaking change:** double quotes used to be string literals. String literals are single-quoted only
  now, and `FROM "x"` is the table name `x` (use `FROM 'path'` for a file path or glob). ParserNG only knows
  plain identifiers, so a quoted name that is not one is written into the expression text as a letters-only
  reference (`pngq`, the UTF-8 bytes as letters, `z`) that carries its own meaning; the bridge resolves it
  back to the real column for scanning, pruning and output names. `ArrowSql` alone does not resolve
  references, only `ParquetSql` does. A double column with a quoted name is copied once per batch into the
  engine's input instead of shared by reference.
* **`COALESCE(a, b, ...)` and `NULLIF(a, b)`** work over numeric columns, numeric literals, `NULL` and each
  other (`COALESCE(NULLIF(a, 0), b)`), in the select list, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY` and
  inside aggregates. ParserNG has no notion of null, so each call is evaluated by the bridge per batch over
  the widened columns (nulls preserved) into a derived `Float8` column, and the engine reads that column;
  `explain()` lists the derived columns. `COALESCE` returns the first non-null argument; `NULLIF(a, b)` is
  null when `a` is null or equals `b` (by double value; a null `b` never equals `a`). The output column is
  named after the original text (`COALESCE(a, 0)`) unless aliased, and is `Float8` even for an integer
  column. Not supported, with a named error: string, date, timestamp or boolean arguments, an expression as an
  argument (`COALESCE(a + b, 0)`; write `COALESCE(a, 0) + COALESCE(b, 0)`), and a call as the operand of
  `IS NULL`, `LIKE`, `REGEXP` or a boolean test. Under `OR`, a predicate over a derived column follows the
  engine's null propagation like any numeric expression leaf. A `COALESCE` predicate is never pushed to
  Parquet row-group pruning.
* **Footer statistics.** With no `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY` or `DISTINCT`, a select list made only
  of `COUNT(*)`, `COUNT(col)`, `MIN(col)` and `MAX(col)` over plain flat columns is answered from the
  footer: no data page is decoded and `lastMetrics()` is `null` (`explain()` says `footer statistics`).
  The answer is taken only when it is provably what a scan would give, and anything else is scanned as usual:
  * `COUNT(*)` is the footer row count. `COUNT(col)` is values minus nulls and needs a null count for every
    row group of every file, on any column type.
  * `MIN`/`MAX` use the footer only for signed integers, `DATE` and `BOOLEAN`. `FLOAT` and `DOUBLE` are scanned (a footer
    minimum may be `-0.0` for `0.0`, and `NaN` is left out of statistics). Strings and
    binary (statistics may be truncated, or written in a wrong order by old writers), decimals, unsigned
    integers and timestamps (statistics are stored as writer-dependent text) are always scanned.
  * Every row group must carry usable min and max, or be all nulls. A row group with neither, a 64-bit
    integer beyond 2^53 and a `NaN` statistic all fall back to the scan.
  * Results use the engine's types: `Float8` for counts and numeric `MIN`/`MAX`, the column type for `DATE` and
    `BOOLEAN`. No values gives `COUNT` 0 and a null `MIN`/`MAX`.
* **Nested field access.** `s.field`, `s.inner.deep`, `items[0]` (zero-based), `m['key']`, `items[1].price`
  and `CARDINALITY(list_or_map)`. Each path is resolved against the file's schema and replaced by a derived
  column of the leaf's own type, filled per batch by walking the struct, list and map vectors and copying
  with the vector's own `copyFromSafe`. From then on the leaf is an ordinary column: numeric leaves work in
  expressions, aggregates, `GROUP BY` and `ORDER BY`; string, boolean, date and timestamp leaves work with the
  native kernels, the string functions and the select list. The output column is named after the path
  (`s.field`) unless aliased.
  * A null anywhere along the path, an index past the end and a missing map key give null, never an error.
  * A path through a list needs an index at that step; a map key is a string literal (an integer for an
    integer-keyed map); on a duplicate key the first entry wins. A path must end at a value:
    `CARDINALITY` takes a list or map and returns its size (`Float8`, null for a null list).
  * Unknown fields, indexing a struct, a key on a list and similar mistakes fail with a named error that lists
    the fields that exist. A field whose own name contains `.`, `[` or `]` cannot be addressed.
  * Nested predicates are never pushed to row-group pruning (`explain()` says so). `SELECT s` still returns
    the whole struct. `t.col` is read as a path, so a table alias prefix is not supported.
* **Date and time.** `EXTRACT(field FROM x)`, `CAST(x AS DATE | TIMESTAMP | DECIMAL(p,s) | VARCHAR)` and
  `INTERVAL`, evaluated by the bridge in whole microseconds (a nanosecond column is truncated to microseconds).
  * `EXTRACT` fields: `YEAR QUARTER MONTH WEEK` (ISO) `DAY DOW` (0 = Sunday) `ISODOW` `DOY HOUR MINUTE SECOND`
    (with fraction) `MILLISECOND MICROSECOND` (the seconds field scaled, as in PostgreSQL) and `EPOCH`. The
    result is `Float8`, so it works anywhere a number does. A UTC-adjusted timestamp is read as its UTC wall clock.
  * `CAST` to `DATE`: from a timestamp (truncated), a date, or a string `yyyy-MM-dd`. To `TIMESTAMP`: from a
    date (midnight), a timestamp, or a string `yyyy-MM-dd[ T]HH:mm[:ss[.f]]` with no zone and at most 6 fraction
    digits (more only if the rest are zeros). An invalid string fails with an error naming the expression and the
    value. Results are `DATE` and `TIMESTAMP` (microseconds, no time zone) columns, so compare them with a
    literal that has no zone suffix. `CAST(x AS VARCHAR)` takes a string expression or a signed integer column (other types are a named error) and gives a string. `CAST(n AS DECIMAL(p,s))` rounds half away from zero at scale `s` and fails
    if the result needs more than `p - s` integer digits; it is `Float8`. `CAST(x AS DECIMAL)` with no
    parameters, `INT` and `DOUBLE` are unchanged (see CAST above).
  * `INTERVAL '3 days'`, `INTERVAL '1' MONTH`, `INTERVAL '1 year 2 months'` with units year, month, week, day,
    hour, minute, second, millisecond, microsecond (a day is exactly 24 hours). Year and month steps are
    calendar steps, clamped to the end of the month (`DATE '2024-01-31' + INTERVAL '1 month'` is
    `2024-02-29`), and work on literals only. A column accepts the fixed-length units.
  * Where a predicate operand is built only from literals (`ts >= TIMESTAMP '2024-01-31 12:00:00' - INTERVAL
    '1 month'`, `d = CAST('2024-02-01' AS DATE)`) it is folded at planning time into one literal. Anything that
    involves a column becomes a derived column evaluated per batch and is not pushed to row-group pruning.
  * Not supported, with a named error: a month or year interval on a column, a time zone suffix combined with
    `CAST` or `INTERVAL`, and mixing a date or time expression with arithmetic.
* **`SELECT COUNT(*)`** (only `COUNT(*)` items, no `WHERE`, `GROUP BY`, `HAVING` or `ORDER BY`)
  is answered from the footer row count, as the first case of the footer statistics rule above.

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
| STRING / ENUM / JSON | VarChar | native | `WHERE` and the select list: `= <> < <= > >=`, `IN`, `BETWEEN`, `LIKE`, `ILIKE`, `REGEXP`, `IS NULL`, and the string functions below | yes (`=`, `<>`, `IN` only; no string range pruning) |
| BINARY, UUID | VarBinary / FixedSizeBinary | native | clear error | no |
| DATE | DateDay | native | `WHERE` only: comparisons, `IN`, `BETWEEN`, `IS NULL`, with `DATE '...'` or an ISO string | yes |
| TIMESTAMP (all units, UTC or local), INT96 | TimeStamp | native | `WHERE` only: comparisons, `IN`, `BETWEEN`, `IS NULL`, with `TIMESTAMP '...'` | no (needs a parser-ng-parquet translator change) |
| struct / list / map | nested | native (whole struct decoded); fields by path, see nested field access | a leaf by path (`s.field`, `items[0]`, `m['k']`), `CARDINALITY(list_or_map)`; the whole column is an error | no |

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
`maxMaterializedRows` (default 20,000,000), `int64Policy`, `allowedRoot`, `backend`, `sourceResolver`
(remote paths, see above).

`backend` picks the parser-ng-arrow backend, as `ArrowQuery.withBackend` does (default `CPU_SIMD`;
also `GPU_AUTO`, `GPU_CUDA`, `GPU_OPENCL`, `GPU_METAL`):

```java
ParquetSql sql = ParquetSql.create(ParquetSqlOptions.defaults().withBackend(ArrowExecutionBackend.GPU_AUTO))
        .register("t", "data/*.parquet");
```

It applies to the numeric expression engine only (filters, projections, numeric aggregates). Work the
bridge does itself (string, date and boolean functions, `COUNT(DISTINCT)`, the Java folds, row-group
pruning) always runs on the CPU. A backend the host cannot run fails when the query executes, as in
`ArrowQuery`. `explain()` prints `backend: ...` whenever the engine is involved.

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
| Remote paths (`s3a://`, `gs://`, `hdfs://`, `abfss://`) | yes, through your `ParquetSourceResolver` (no connector bundled; not verified against real cloud storage) |
| Execution backend for the numeric engine (`ParquetSqlOptions.withBackend`) | yes (`CPU_SIMD` default; GPU backends not exercised by the tests) |
| Allow-listed root for paths | yes |
| `EXPLAIN` | yes |
| `COUNT(*)` from the footer, no decode | yes |
| Bounded top-N for `ORDER BY ... LIMIT n` | yes |
| `SELECT DISTINCT`, `COUNT(DISTINCT col)` | yes, with or without `GROUP BY` |
| `SELECT DISTINCT` in plain `ArrowQuery` / `ArrowSql` | numeric columns only, per batch (see "`SELECT DISTINCT` in `ArrowQuery`") |
| `COUNT(col)` any type; `MIN`/`MAX` over string, date, timestamp, boolean | yes, with or without `GROUP BY` |
| `GROUP BY` a string, boolean, date or timestamp key or expression, with `HAVING` and `ORDER BY` | yes (folded in Java) |
| `CAST` of a string to a number | yes (strict parse, named errors) |
| Scan options exposed | yes |
| Precision error for int64 beyond 2^53 | yes |
| Boolean, string, date, timestamp `WHERE` predicates, `IS [NOT] TRUE/FALSE`, `IS NULL` on any column | yes |
| Exact decimal predicates | not yet (compared as double) |
| Arithmetic on strings, dates, timestamps | clear error |
| `LIKE`, `ILIKE` (ASCII folding), `REGEXP`, `ESCAPE` | yes |
| String functions `LOWER`, `UPPER`, `TRIM`, `LTRIM`, `RTRIM`, `SUBSTRING`/`SUBSTR`, `CONCAT`, `||`, `LENGTH`/`CHAR_LENGTH`/`OCTET_LENGTH` | yes (bridge only; see the string functions section) |
| String-valued expressions as `ORDER BY`/`GROUP BY` keys or aggregate arguments, or in arithmetic | not yet (named error) |
| `DATE '...'`, `TIMESTAMP '...'` literals | yes |
| `EXTRACT(field FROM x)`, `CAST(x AS DATE / TIMESTAMP / DECIMAL(p,s) / VARCHAR)`, `INTERVAL` | yes (bridge only; see the date and time section) |
| `CAST ... AS VARCHAR` of a non-integer, `TIMESTAMP WITH TIME ZONE` casts, `date_part`, `AGE`, `DATE_TRUNC` | not yet (named error) |
| Timestamp row-group pruning | not yet (needs a parser-ng-parquet change) |
| Quoted identifiers (`"my col"`, `` `my col` ``) | yes (bridge only; double quotes are no longer string literals, see below) |
| Nested field access (`s.field`, `items[0]`, `m['k']`) and `CARDINALITY` | yes (bridge only; never pushed to row-group pruning) |
| `MIN`/`MAX`/`COUNT(col)` answered from footer statistics | yes, when provably exact (signed integers, dates, booleans; floats are scanned; see below) |
| `HAVING` over a string, date or timestamp `MIN`/`MAX`; `GROUP BY` position (`GROUP BY 1`) | not yet (named error) |
| `LIMIT n OFFSET m`, `OFFSET m [LIMIT n]` | yes (bridge only; `ArrowQuery` rejects `OFFSET`) |
| `COALESCE`, `NULLIF` over numeric columns and literals | yes (bridge only; result is `Float8`) |
| `COALESCE`/`NULLIF` over strings, dates, timestamps, or with expression arguments | not yet (named error) |

### Differences between ArrowQuery and ParquetSql

The same SQL text can behave differently through the two entry points. The first three rows are by design; the
last three are limitations of `ArrowQuery` that `ParquetSql` does not share.

| | `ArrowQuery` / `ArrowSql` | `ParquetSql` |
| --- | --- | --- |
| `OFFSET` | rejected (`UnsupportedOperationException`) | supported |
| `SELECT DISTINCT` | numeric columns, per batch | any column type, across batches |
| `COUNT(DISTINCT x)` | rejected | supported |
| Numeric type of the computation | `float32` kernels unless every column is a `Float8Vector`: a `Float4` column compared with `0.1` is compared as `float32`, so `0.1f > 0.1` is false | every numeric column is widened to double, so the same comparison is evaluated in double and `0.1f > 0.1` is true |
| Null under `OR`, and `NOT (... AND ...)` | a null operand drops the row even when the other branch is true (see Null handling) | simple numeric leaves are two-valued masks, so the row is kept; numeric expression leaves still propagate null |
| `HAVING` | a `GROUP BY` key or an aggregate alias; a spelled-out aggregate fails at execution | aggregates by alias or spelled out |

## Benchmarking and profiling ParquetSql

`com.github.gbenroscience.sqlv1.demo.ParquetSqlBenchmark` measures `ParquetSql` against what it is built on and
says where the time goes. It lives with the other demos and reuses their methods: fixed-seed data written before
any timing and a warm page cache (the parser-ng-parquet fair comparison), warm-up then calibrated reps with the
median, min and max, and every answer consumed so the JIT cannot drop the work (`Benchmark`, `ExecBench`).

**What each workload is.** One SQL query run three ways over the same file:

| column | meaning |
|---|---|
| `sql ms` | `ParquetSql`, planned once and executed repeatedly (median per call) |
| `direct ms` | the same answer from hand-written Java straight on parser-ng-parquet's Arrow batches, with the equivalent `select` and `pushdown` |
| `sql/dir` | the cost of the SQL layer over a hand-written loop on the same scan: 1.0 is no overhead |
| `raw ms`, `raw/sql` | the parquet-java `Group` reader (a few workloads), the baseline of the "3x to 5x" claim, and how many times slower it is than `ParquetSql` |
| `plan ms` | building the plan (footer reads, rewriting, engine setup), timed on its own |
| `alloc/call` | bytes allocated per call by the calling thread (a parallel scan's worker threads are not counted) |
| `ok` | the three answers agree (checksum, relative tolerance 1e-9) and the allocator was empty afterwards (`MISMATCH` or `LEAK` otherwise, and the exit code is 1) |
| `path` | which execution path the plan took (`footer statistics`, `stream`, `materialize`, `streaming partial aggregation`, `bounded top-N`, `fold in Java`) |

The workloads cover each path: footer-only answers (`count-star`, `footer-stats`), a zero-copy projection, row-group
pruning (`filter-pruned`), numeric filters and aggregates through the engine (`filter-numeric`, `agg-sum-avg`,
`group-numeric`), derived columns (`coalesce-sum`, `extract-year`, `upper-eq`), the native string kernel
(`like-native`), the Java folds (`group-string`, `count-distinct`) and bounded top-N (`top-100`). The data is
`id` (increasing int64), `value` (double, 10% null), `qty` (int32, 5% null), `name` (1000 distinct strings, 5% null),
`day` (date) and `flag` (boolean, 20% null).

**Running it.** Use the same JVM flags as the tests. On JDK 24 and later add `-Darrow.allocation.manager.type=Unsafe`.

```
mvn -q -pl parser-ng-sql -am package -DskipTests
mvn -q -pl parser-ng-sql exec:exec -Dexec.executable=java \
  -Dexec.args="--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  --add-opens=java.base/java.nio=ALL-UNNAMED -classpath %classpath \
  com.github.gbenroscience.sqlv1.demo.ParquetSqlBenchmark --rows 3000000"
```

Options: `--rows N`, `--codec UNCOMPRESSED|SNAPPY`, `--rowgroup-mb N`, `--batch N`, `--parallelism N`, `--only TEXT`,
`--no-raw` (the Group baselines are slow), `--warmup N --reps N --min-rep-ms X`, `--explain` (print every plan),
`--csv FILE`, `--keep`. Compare runs by changing one option at a time (`--parallelism 1` against the default shows
what the parallel scan buys; `--codec SNAPPY` shows decompression).

**Profiling.** `--profile TEXT --seconds S` runs the first matching workload under an in-process sampler
(`StackSampler`) and prints self and inclusive time per layer (bridge, SQL engine, scan, expression engine, Arrow,
parquet-java, JDK), the hottest methods and the threads that were running, and the allocation per call. Read it for
"which layer owns the time" and for spotting a hot allocation site. It samples at safepoints, so for line-level
truth attach JFR (`-XX:StartFlightRecording=duration=60s,filename=bench.jfr`) or async-profiler to the same run.

**Streamed or materialized.** A plan that streams (a projection or filter, `isStreamable()`) is measured through
`stream()`, batch by batch, as an application would consume a large result, not through `execute()`, which returns the
whole result as one root and so measures a copy of the table as well.

**A first run, as a baseline.** On a 4-core Windows machine, Java 24.0.1, 3,000,000 rows, 16 MiB row groups
(4 row groups), the numbers before the streaming aggregation, heap top-N and allocation-free folds were:

| workload | sql ms | direct ms | sql/dir | alloc/call | path then |
|---|---:|---:|---:|---:|---|
| scan-project | 351 | 188 | 1.87 | 140 MiB | materialized through `execute()` |
| filter-pruned | 118 | 29 | 4.14 | 21 MiB | materialized through `execute()` |
| filter-numeric | 421 | 197 | 2.14 | 178 MiB | whole scan copied into one root |
| agg-sum-avg | 411 | 177 | 2.32 | 224 MiB | whole scan copied into one root |
| coalesce-sum | 257 | 79 | 3.26 | 131 MiB | whole scan copied into one root |
| group-numeric | 597 | 74 | 8.02 | 267 MiB | whole scan copied into one root |
| group-string | 478 | 485 | 0.98 | 458 MiB | boxed fold |
| count-distinct | 407 | 459 | 0.89 | 320 MiB | boxed fold |
| top-100 | 1243 | 107 | 11.60 | 246 MiB | engine sorted every batch |

The `raw/sql` ratios were 4.33x, 3.74x and 3.67x for the three workloads with a parquet-java baseline, in line with the
3x to 5x claim. Every answer agreed and no allocator leaked. Rerun on your machine after a change and compare against
this table; none of the changes described above has been measured yet.

**Reading the result.** A `sql/dir` close to 1.0 on a scan-bound workload means the layer adds almost nothing there;
a large ratio on a small workload points at per-call fixed cost (`plan ms`, derived-column setup); a large
`alloc/call` on a workload that should be zero-allocation points at the batch loop (the folds box their keys by
design, and `plan.execute` allocates one result root per call). The numbers depend on the machine, the JVM and the
data: run it where you will deploy before drawing conclusions.

## What's intentionally out of scope

parser-ng-sql deliberately does not implement:

- `JOIN`, subqueries, or any multi-table concept; every query operates on
  exactly one `VectorSchemaRoot` you already have in hand.
- `INSERT`, `UPDATE`, `DELETE`, transactions, or a catalog of any kind;
  there is no notion of a persistent table to mutate.
- `COUNT(DISTINCT x)` in `ArrowSql` / `ArrowQuery`, which reject it (`ParquetSql` supports it). `SELECT DISTINCT`
  works in `ArrowQuery` for numeric columns, per batch only (see
  [`SELECT DISTINCT` in `ArrowQuery`](#select-distinct-in-arrowquery)).
- General string/text processing in `ArrowSql`; every value in a ParserNG expression is a
  floating-point number. Over Parquet, string, boolean, date and timestamp predicates, the string
  functions, `CAST` to date, timestamp and decimal, `EXTRACT` and `INTERVAL` are supported by the bridge
  (see the feature status table above).
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