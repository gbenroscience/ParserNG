# parser-ng-sql

A SQL-shaped front end for `parser-ng-arrow`: `SELECT ... FROM ... [WHERE ...]`
compiles into vectorized Apache Arrow computations, driven entirely by
ParserNG expressions.

```java
ArrowQuery query = ArrowQuery.compile(
        "SELECT sqrt(x*x + y*y) AS distance FROM data WHERE x > 10");
VectorSchemaRoot result = query.execute(root); // call as many times as you like
query.close();

// or, for a one-off query:
VectorSchemaRoot result = ArrowSql.execute(root,
        "SELECT x, y, sqrt(x*x + y*y) AS magnitude FROM points WHERE magnitude > 10");
```

This module intentionally stops at `SELECT`/`WHERE` with column aliasing —
no `JOIN`, `INSERT`, `UPDATE`, `DELETE`, transactions, catalogs, or
subqueries. The goal is *"make SQL a convenient way to describe vectorized
Arrow computations"*, not *"build a database"*.

## Installing into the ParserNG repo

1. Copy the `parser-ng-sql/` folder into the root of your `ParserNG` clone,
   next to `parser-ng-arrow/`.
2. In the root `pom.xml`, add `<module>parser-ng-sql</module>` inside the
   existing `<modules>` block (after `parser-ng-arrow`), if it isn't already
   there.
3. Build: `mvn -pl parser-ng-sql -am install` (or just `mvn install` from the
   repo root to build everything).
4. Run its tests: `mvn -pl parser-ng-sql test`.

Requires JDK 22+ and network access to Maven Central for `arrow-vector` /
`arrow-memory-netty` (19.0.0) and JUnit 5 (5.10.3), matching the versions
already used by `parser-ng-arrow`.

## Grammar (v1)
query ::= SELECT select_list
FROM table_reference
[WHERE boolean_expression]
[GROUP BY expression_list]
[HAVING boolean_expression]
[ORDER BY order_item (',' order_item)]
[LIMIT integer_literal]
select_list ::= '' | select_item (',' select_item)*
select_item ::= expression [AS identifier]
| aggregate_call [AS identifier]
aggregate_call ::= ('COUNT'|'SUM'|'AVG'|'MIN'|'MAX') '(' (expression|'') ')'
-- '' only valid with COUNT
order_item ::= expression ['ASC' | 'DESC']
boolean_expression ::= or_expression
or_expression ::= and_expression ('OR' and_expression)*
and_expression ::= not_expression ('AND' not_expression)*
not_expression ::= 'NOT' not_expression | predicate
predicate ::= comparison
| expression 'IS' ['NOT'] 'NULL'
| expression ['NOT'] 'BETWEEN' expression 'AND' expression
| expression ['NOT'] 'IN' '(' expression_list ')'
| '(' boolean_expression ')'
comparison ::= expression comparison_operator expression
comparison_operator ::= '=' | '!=' | '<>' | '<' | '<=' | '>' | '>='
expression ::= additive_expression (ParserNG's own arithmetic grammar --
+ - * / % ^, parens, function calls,
identifiers, literals, plus
case_expression/cast_expression below)
case_expression ::= 'CASE' expression ('WHEN' expression 'THEN' expression)+
['ELSE' expression] 'END' -- simple CASE
| 'CASE' ('WHEN' boolean_expression 'THEN' expression)+
['ELSE' expression] 'END' -- searched CASE
cast_expression ::= 'CAST' '(' expression 'AS' identifier ')'

Full javadoc for the grammar and its rationale lives on `SqlParser`; the
`ast` package and `BoolExprs` document how `WHERE`/`HAVING` are normalized
(negation-normal form, since ParserNG has no logical-not operator) and
rendered.

### `CASE`/`WHEN`/`THEN`/`ELSE`/`END`

ParserNG has no `CASE` operator, but does have the `if(cond, then, else)`
idiom described below. A `CASE` expression compiles to a right-nested chain
of `if(...)` calls — `CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END` becomes
`if(a, 1, if(b, 2, 3))`, and an omitted `ELSE` falls back to the literal
`NULL`. Both the searched form (a full boolean condition per `WHEN`) and
the simple form (`CASE x WHEN 1 THEN ... END`, compiled via `x == 1`) are
supported. See `SqlParser`'s "CASE/WHEN/THEN/ELSE/END".

### `CAST`

`CAST(expr AS type)` has no ParserNG operator either, and since every value
in this module is already a ParserNG/Arrow `float64`/`float32`, there is no
representation change to make: `CAST(x AS INT/INTEGER/SMALLINT/BIGINT/LONG)`
truncates toward zero (`if(x >= 0, floor(x), -1 * floor(-1 * (x)))` —
assumes ParserNG provides `floor`); `CAST(x AS FLOAT/REAL/DOUBLE/DECIMAL/NUMERIC)`
is a pure no-op. See `SqlParser`'s "CAST".

### `GROUP BY`/`HAVING`/aggregates and `ORDER BY`/`LIMIT`

`COUNT`/`SUM`/`AVG`/`MIN`/`MAX` are recognized only at the top level of a
`select_item` (no nested aggregates, matching standard SQL). Every
non-aggregate `select_item` in a grouped/aggregate query must match a
`GROUP BY` expression verbatim — enforced by `SelectStatement`'s
constructor. Grouping and aggregation are computed client-side (a plain
nested-loop grouped aggregation, not a vectorized one — see `ArrowQuery`'s
"Aggregation strategy"); `HAVING` then reuses `WHERE`'s own fused/leaf-mask
predicate machinery against the resulting one-row-per-group data. Alias an
aggregate item you intend to reference from `HAVING`/`ORDER BY`
(`SUM(x) AS total ... HAVING total > 10`) rather than repeating its raw
call text. `ORDER BY`/`LIMIT` apply after `WHERE`/`GROUP BY`/`HAVING` but
before the final projection, so an `ORDER BY` key may also name a
`SELECT`-list alias; sorting is a stable multi-key sort with `NULL`s last
regardless of direction. See `ArrowQuery`'s "Aggregation strategy" and
"`ORDER BY` and `LIMIT`".

## Package

Everything lives under `com.github.gbenroscience.sqlv1`.
 

## Known v1 limitations (by design, not oversight)

- **`IS [NOT] NULL`** has no ParserNG rendering (ParserNG's bulk evaluators
  never expose a null-test operator), so a `WHERE` clause containing one
  falls back to a slower leaf-by-leaf mask evaluation instead of one fused
  kernel — see `ArrowQuery`'s "Predicate evaluation strategy". For the same
  reason, `IS [NOT] NULL` may only be used directly in a `WHERE` clause: it
  is rejected with a precise `SqlSyntaxException` if used as a function
  argument or inside a grouping paren (see "Embedded boolean conditions"
  below) — there, only comparisons/`AND`/`OR`/`NOT`/`BETWEEN`/`IN` are
  accepted, since only those have a ParserNG rendering.
- **Column aliasing always copies.** Every output column (passthrough or
  computed) is materialized into a fresh vector under its final name,
  rather than zero-copy-transferring a renamed passthrough column, so that
  a source column referenced more than once (`SELECT x, x AS y FROM t`)
  can never be accidentally emptied by a buffer transfer. A future version
  could special-case the single-reference case for a zero-copy rename.
- **No quoted identifiers**, no scientific-notation numeric literals, and
  `table_reference` is a plain identifier that is not resolved against
  anything (there is no table catalog — `ArrowQuery.execute` always takes
  the `VectorSchemaRoot` directly).
- **Expression text is reassembled, not always verbatim.** Reconstructed
  arithmetic text normalizes whitespace to single spaces between tokens
  (`x*x` becomes `x * x` in `SelectItem.exprText()`) — cosmetic only, and
  irrelevant to ParserNG's own whitespace-insensitive scanner, but worth
  knowing if you inspect `exprText()` directly rather than only its
  compiled/evaluated result.
- **Grouped aggregation is client-side, not vectorized.** `GROUP BY`
  partitions rows and computes `COUNT`/`SUM`/`AVG`/`MIN`/`MAX` with a plain
  nested loop (materializing each group's rows into their own small
  sub-batch) rather than a vectorized grouped-aggregation kernel — fine for
  parser-ng-sql's goal of a convenient computation language, not a claim of
  database-grade aggregation performance. See `ArrowQuery`'s "Aggregation
  strategy".
- **`HAVING`/`ORDER BY` should reference an aggregate `SELECT` item by
  alias.** An unaliased aggregate item (`SUM(x)` with no `AS`) is named
  after its own reconstructed call text; referencing that same text from
  `HAVING`/`ORDER BY` only resolves if written identically. Always alias an
  aggregate you plan to reference from either clause.
- **No nested aggregates and no `DISTINCT`.** `COUNT`/`SUM`/`AVG`/`MIN`/
  `MAX` are recognized only at the top level of a `select_item`, matching
  standard SQL's prohibition on nesting them; there is no `COUNT(DISTINCT ...)`
  or other `DISTINCT` support.

## Embedded boolean conditions (`if(sin(x) > 0, tan(x), 0.2)` and friends)

ParserNG's own expression language natively evaluates comparisons and
`&&`/`||` as ordinary truthy (0/1) values — that's exactly how idioms like
`if(sin(x) > 0, tan(x), 0.2)` work in plain ParserNG: the condition argument
is just an expression like any other. The grammar as originally specified
restricted `expression` to pure arithmetic, which meant that idiom could
not be written from SQL text at all (`sin(x) > 0` is a `comparison`, not an
`expression`, one level up in the grammar) — that was an oversight, not an
intended restriction, and is now fixed:

- A `function_call`'s arguments, a parenthesized grouping term, and a
  `select_item`'s own top-level expression may each independently be either
  a plain arithmetic expression **or** a full boolean condition
  (comparisons, `AND`/`OR`/`NOT`, `BETWEEN`, `IN`) — parsed with the same
  machinery as a `WHERE` clause, normalized to negation-normal form, and
  rendered to ParserNG's native `&&`/`||`/`==` text before being spliced
  back in. `SELECT if(sin(x) > 0, tan(x), 0.2) AS y FROM t` and
  `SELECT x > 0 AS flag FROM t` both work.
- This is deliberately **not** extended to a `comparison`'s own operands or
  a `BETWEEN`'s bounds — those are bounded by `AND`/`OR`-adjacent tokens
  that can themselves continue a boolean expression, so a greedy trial
  parse there would swallow tokens meant for the *outer* `WHERE` clause.
  Nesting further inside a paren/argument those positions themselves open
  is still fully supported (`WHERE x = (y > 0)` works fine).
- See `SqlParser`'s "Embedded boolean conditions" section for the full
  rationale and exactly which positions are safe.