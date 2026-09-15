# parser-ng-sql

**Make SQL a convenient way to describe vectorized Apache Arrow computations.**

parser-ng-sql is a `SELECT`-only SQL front end for [ParserNG](https://github.com/gbenroscience/ParserNG)'s
Arrow integration, parser-ng-arrow. It compiles a SQL statement once into a
small set of ParserNG expression evaluators, then lets you run that compiled
query against as many `VectorSchemaRoot` batches as you like. It is not, and
is not trying to become, a database: there is no catalog, no `JOIN`, no
`INSERT`/`UPDATE`/`DELETE`, no transactions. There is exactly one job —
turning a `SELECT` statement into a fast, reusable, vectorized computation
over an Arrow batch you already have in hand — and it is built to do that
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
- [Performance notes](#performance-notes)
- [Error handling](#error-handling)
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
    <version>3.0.7</version>
</dependency>
```

**Gradle**

```groovy
implementation("com.github.gbenroscience:parser-ng-sql:3.0.7")
```

parser-ng-sql depends on `parser-ng-arrow`, `parser-ng`, `parser-ng-simd`, and
`parser-ng-gpu-simd` transitively — a plain dependency declaration pulls in
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
it doesn't give you is a convenient way to *describe* those operations —
you're writing predicate and projection text by hand, keeping track of which
columns feed which computation, and re-deriving the same expression twice if
you need it in both a filter and a projection. SQL is a format almost every
engineer already knows for exactly that kind of description. parser-ng-sql
takes that familiar syntax and compiles it straight down to the same
`ArrowExpressionEvaluator`s you'd have written by hand — nothing about the
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

Every arithmetic expression anywhere in a query — a `SELECT` item, a `WHERE`
operand, an aggregate's argument, an `ORDER BY` key — is ordinary ParserNG
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
A column subset — no computation, no filtering.

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

A `WHERE` operand can itself be a parenthesized boolean condition — useful
for feeding a boolean result into an otherwise-arithmetic position, like a
`CASE` condition built by hand or a nested `if`:

```sql
SELECT x, if(x > 50, 1, 0) AS high_flag FROM data
```

### CASE / WHEN

Both the *searched* and *simple* forms of `CASE` are supported. Every
`CASE` must end in an explicit `ELSE` — there is no numeric `NULL` literal
for a missing branch to fall back to, so a missing `ELSE` is rejected at
compile time with a clear message rather than failing confusingly later.

**Searched form** — a `WHEN` clause per condition, evaluated in order:

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

**Simple form** — one operand, compared for equality against each `WHEN`
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
`CASE WHEN category = 1 THEN ...` — the simple form is pure convenience,
not a different capability.

### CAST

`CAST(expression AS type)` supports numeric target types — since every
ParserNG value is already a floating-point number under the hood, `CAST`
here means "how should this number be interpreted," not a general type
system:

```sql
SELECT reading, CAST(reading AS INT) AS truncated FROM data
```

`INT`, `INTEGER`, `SMALLINT`, `BIGINT`, and `LONG` all truncate toward
zero — `-2.7` becomes `-2`, not `-3`, matching a Java `(long)` cast, not
mathematical floor. `DOUBLE`, `FLOAT`, `REAL`, `NUMERIC`, and `DECIMAL` are
a no-op identity, since that's already the representation in use. Any other
target type is rejected at compile time with a clear message rather than
silently doing nothing useful.

### GROUP BY and aggregates

`SUM`, `COUNT`, `AVG`, `MIN`, and `MAX` are recognized wherever they appear
as a top-level `SELECT`-item function call. `COUNT(*)` counts every row in
a group; every other aggregate skips `null` inputs, matching standard SQL
aggregate semantics — including that `SUM`/`AVG`/`MIN`/`MAX` over a group
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

An aggregate query needs no explicit `GROUP BY` at all — with none present,
the whole (filtered) input is treated as a single implicit group, exactly
as plain SQL does:

```sql
SELECT SUM(reading) AS total, COUNT(*) AS n
FROM data
WHERE reading > 0
```

A grouped query's non-aggregate `SELECT` items must match a `GROUP BY` key
expression exactly — the usual SQL rule that a plain column in the select
list of a grouped query has to be functionally determined by the grouping
key. This is checked once, at compile time:

```sql
-- rejected: 'reading' is neither aggregated nor a GROUP BY key
SELECT category, reading, SUM(reading) FROM data GROUP BY category
```

`GROUP BY` accepts more than one key, and a key can be any expression, not
just a bare column:

```sql
SELECT category, SUM(reading) AS total
FROM data
GROUP BY category
```

### HAVING

`HAVING` filters groups by an aggregate result, after grouping — the same
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

`NULL`s sort last regardless of `ASC`/`DESC` — a `null` is "unknown," not
the smallest or largest value.

For a query with no `GROUP BY`, `ORDER BY` conceptually sorts the
`FROM`/`WHERE` row set before the `SELECT` list narrows it down to a
smaller list of columns — so it can name a column that isn't even in the
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

`ArrowQuery.compile(sql)` only parses the SQL text — cheap, and independent
of any particular Arrow schema. The actual ParserNG expression evaluators
are compiled lazily on the first call to `execute`, because whether
ParserNG compiles `float64` or `float32` kernels depends on the column
types of the batch you pass in. Once built, that compiled plan is cached
and reused on every subsequent `execute` call against a batch with the
same schema — the SQL parsing and expression compilation genuinely happen
once, no matter how many batches you run through it:

```java
try (ArrowQuery reusable = ArrowQuery.compile(
        "SELECT x, y, x * y AS product FROM data WHERE x > 15")) {

    VectorSchemaRoot result1 = reusable.execute(batch1);
    VectorSchemaRoot result2 = reusable.execute(batch2); // same compiled plan, no recompilation
}
```

If a later `execute` call passes a batch with a genuinely different schema
(different column names or types), the plan is transparently recompiled —
you don't need to detect that yourself.

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
the inputs allow one, and `null` only where an input actually was —
standard, predictable null propagation through arithmetic. Unlike
`withBackend`, changing the null policy never requires recompilation.

## Performance notes

A few design choices worth knowing about if you're deciding whether
parser-ng-sql fits a performance-sensitive path:

- **A passthrough column is never copied.** `SELECT x, y FROM data` returns
  `x` and `y` as zero-copy references into the already-filtered batch, not
  fresh row-by-row copies — copying only happens where it's actually
  needed (a computed column, or a passthrough column that's also being
  renamed).
- **`GROUP BY` and aggregate arguments are evaluated in bulk, once, not per
  row and not per group.** Every `GROUP BY` key and every distinct
  aggregate argument is evaluated across the whole filtered batch in a
  single pass before grouping happens, so aggregation cost scales with the
  number of input rows, not with the number of distinct groups.
- **The compiled-plan cache check itself is allocation-free.** Confirming
  a cached plan still matches the current batch's schema is a handful of
  primitive comparisons, not a fresh collection built on every call.
- **A predicate or projection expression referenced more than once is
  compiled once.** If a `WHERE` clause resolves a `SELECT`-list alias back
  to the same expression a projection already needs, both share a single
  compiled evaluator rather than compiling it twice.

## Error handling

parser-ng-sql draws a clear line between two kinds of failure, and tries
hard to keep problems on the compile-time side of that line rather than
letting them surface as a confusing runtime failure:

- **`SqlSyntaxException`** — the SQL text itself doesn't parse: a missing
  keyword, an unsupported `CAST` target type, a `CASE` with no `ELSE`. This
  is thrown as early as possible, from `ArrowQuery.compile`, before any
  ParserNG expression is even compiled.
- **`ArrowSqlException`** — the SQL parsed fine, but compiling its
  expressions against a particular batch's schema failed: a grouped
  query's `SELECT` item doesn't match a `GROUP BY` key, a cyclic
  `SELECT`-list alias reference, and similar shape mismatches that can
  only be detected once a real schema is in hand. Thrown from the first
  `execute` call against a given schema.
- **`ArrowBindingException`** — thrown directly by parser-ng-arrow itself
  when a compiled expression can't actually run against the data in front
  of it (a column genuinely missing from the batch, for instance).

## What's intentionally out of scope

parser-ng-sql deliberately does not implement:

- `JOIN`, subqueries, or any multi-table concept — every query operates on
  exactly one `VectorSchemaRoot` you already have in hand.
- `INSERT`, `UPDATE`, `DELETE`, transactions, or a catalog of any kind —
  there is no notion of a persistent table to mutate.
- `DISTINCT`.
- Non-numeric `CAST` targets or general string/text processing — every
  value in a ParserNG expression is a floating-point number.

None of these are missing by oversight. The goal is a fast, predictable way
to describe a vectorized Arrow computation, not a general-purpose query
engine — see the tagline at the top of this document.

## Resource ownership and thread-safety

An `ArrowQuery` owns whatever expression evaluators it has compiled and
must be closed when no longer needed — a try-with-resources block, as in
every example above, is the simplest way. Every `VectorSchemaRoot` returned
by `execute` is a fresh, independently-owned batch that the caller owns and
must close in turn; parser-ng-sql never returns a view over, or a root
that shares ownership with, the root you passed in.

Configuring a query (`withBackend`/`withNullPolicy`) and calling `execute`
concurrently from multiple threads is not supported. Calling `execute`
concurrently once a query's configuration has stopped changing is only as
safe as the underlying ParserNG evaluators' own backend.