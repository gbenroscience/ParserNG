# parser-ng-arrow

Zero-copy Apache Arrow bulk-evaluation bridge for [ParserNG](https://github.com/gbenroscience/ParserNG),
built on `SIMDEngineEvaluator`'s `applyBulk(MemorySegment[], MemorySegment)`
API from `parser-ng-gpu-simd`.

## What this is

`SIMDEngineEvaluator` already speaks `java.lang.foreign.MemorySegment` natively,
including a per-column overload — `applyBulk(MemorySegment[] variables, MemorySegment output)`
— designed specifically so each variable can point at its own independently-allocated
off-heap buffer, rather than requiring one big concatenated segment. That's exactly
Arrow's `VectorSchemaRoot` shape: each `Float8Vector` column owns its own `ArrowBuf`.

`ArrowBulkEvaluator` binds Arrow columns to that API directly:

```java
try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.builder("(x + y) * z")
        .variables("x", "y", "z")   // must match ParserNG's internal slot order — see below
        .build()) {

    try (Float8Vector result = evaluator.evaluate(vectorSchemaRoot, allocator)) {
        // result is a normal Arrow Float8Vector
    }
}
```

No `double[]` copy of any input column happens on the way in. `ArrowSegments.ofData(...)`
constructs a `MemorySegment` directly over each column's existing native buffer via
`MemorySegment.ofAddress(arrowBuf.memoryAddress()).reinterpret(byteSize)`. The output
vector's buffer is bound the same way, so results are written straight into Arrow memory
too.

Whether the *arithmetic itself* touches Arrow memory in place, or stages an operand
into on-heap scratch first, depends on what the expression does — see
`SIMDEngineEvaluator`'s own docs: pure `+ - * /` chains over loaded variables never
materialize their operands into scratch; anything invoking a transcendental function,
`POW`, a comparison, `IF`/`AND`/`OR`, or `VMA` does, for just the operand(s) that op
consumes.

## Variable ordering

`applyBulk(MemorySegment[], MemorySegment)` expects `variables[i]` to be the data for
whichever variable ParserNG assigned to slot `i` when it compiled the expression.
**Confirmed by ParserNG's author:** `MathExpression` does no exotic slot ordering —
variables are discovered and assigned slots on a **first-appearance, left-to-right**
basis as the expression is scanned. `"x + y * z"` binds slot 0 = `x`, slot 1 = `y`,
slot 2 = `z`; `"z * y + x"` binds slot 0 = `z`, slot 1 = `y`, slot 2 = `x` — order
follows the string, not any alphabetic or declaration-list convention. This is also
consistent with what's visible in `MathExpression`'s source: each `Token(Variable v)`
takes `frameIndex = v.getFrameIndex()` from an internal `VariableRegistry` that hands
out slots as `Variable` objects are encountered.

`ArrowBulkEvaluator.Builder.variables(...)` still takes this order as an explicit
parameter rather than auto-deriving it from the expression string — reimplementing
ParserNG's own tokenizer here (to know that `sin` is a function name and not a
variable, for instance) would be a real correctness risk of its own, for a module
that isn't part of ParserNG's core. So: **pass variable names in the same
left-to-right order they first appear in your expression string**, and a wrong
*count* is still caught at `build()` time by a one-row smoke test against the on-heap
path. If ParserNG ever exposes `registry.getSlots()` (or similar) publicly, `Builder`
should gain a `variables(MathExpression)` overload that reads the order directly
instead of relying on the caller to mirror it correctly.

## What's zero-copy, honestly

| Path | Zero-copy? |
|---|---|
| Input column → engine (pure `+ - * /` chains) | Yes — reads straight from `ArrowBuf` via `MemorySegment` |
| Input column → engine (function calls, `POW`, comparisons, `IF`/`AND`/`OR`, `VMA`) | No — that operand is staged into on-heap scratch once, on first use |
| Engine → output column | Yes — writes straight into the output `Float8Vector`'s `ArrowBuf` |
| Non-`Float8Vector` columns (`IntVector`, `BigIntVector`, `Float4Vector`, ...) | No — `VectorCoercion` allocates and copies |
| `NullPolicy.PROPAGATE_NULL`'s validity-bitmap merge | Technically a copy/AND over `MemorySegment`, but it's `rowCount / 8` bytes — negligible next to the `rowCount * 8`-byte data path |

## Null handling

Arrow carries a validity bitmap; `double[]`/raw memory does not. `NullPolicy` makes
you pick:

- **`REJECT_ON_NULL`** (default) — scans every bound column's validity bitmap before
  evaluating; throws `ArrowNullValueException` if any row in range is null. Safe
  default when your pipeline is supposed to guarantee dense batches.
- **`PROPAGATE_NULL`** — computes over every row unconditionally (preserving the
  zero-copy fast path — no per-element branch), then sets the output vector's validity
  bitmap to the bitwise AND of every input column's validity bitmap. The *data* value
  in a resulting null row is whatever the kernel computed from unspecified input bytes
  — not guaranteed `0.0` or `NaN`. Consumers must respect the validity bitmap, as with
  any Arrow null.

## Requirements

- JDK 22+ (finalized Foreign Function & Memory API — `MemorySegment.ofAddress`/`reinterpret`;
  also matches `SIMDEngineEvaluator`'s own JDK22+ requirement for CPU pinning).
- `jdk.incubator.vector` on the module path at compile *and* run time (`--add-modules jdk.incubator.vector`)
  — inherited from `parser-ng-gpu-simd`.
- `--enable-native-access=ALL-UNNAMED` at runtime — `MemorySegment.ofAddress` over an
  arbitrary native address is a restricted method.
- The `--add-opens` flags Arrow itself needs for its off-heap allocator on JDK 9+
  (`java.base/java.nio`, `java.base/java.util`). See `pom.xml`'s `surefire` config for
  the full flag set used in tests; mirror it in your own run scripts / shaded-jar
  manifest args.
- Linux is where `SIMDEngineEvaluator`'s CPU pinning (and therefore its
  `applyBulkParallel` worker efficiency) is strongest, per its own class docs.

## Honest limitations / not done here

- **Not compiled/tested in a live Maven build in this environment** — no Maven Central
  network access was available while writing this, so `pom.xml` and the test suite are
  written carefully against the real Arrow 25.0.0 and `SIMDEngineEvaluator` APIs but
  have not been run through an actual `mvn test`. Run it for real before shipping.
- **Variable ordering follows first-appearance order in the expression string**
  (confirmed by ParserNG's author) but is still supplied by the caller rather than
  auto-derived — get the left-to-right order right, since a wrong order is not
  caught by anything at build or run time (only a wrong count is).
- **Only `Float8Vector` is zero-copy.** Everything else goes through `VectorCoercion`
  (a real copy) or needs an upstream cast.
- **No streaming convenience yet.** Call `evaluateInto(root, output)` once per batch in
  your own `ArrowStreamReader`/`ArrowFileReader` loop.
- **`andValidityInto` is a scalar byte loop**, not SIMD — deliberate v1 tradeoff given
  its tiny size relative to the data path; flagged in `ArrowSegments`' javadoc as a
  reasonable follow-up.
- **`DecimalVector`, `VarCharVector`, and other non-numeric-scalar types** have no
  coercion path at all yet — `VectorCoercion.toFloat8(FieldVector, ...)` will throw
  `UnsupportedVectorTypeException` for them.

## Module layout

```
parser-ng-arrow/
  pom.xml
  src/main/java/com/github/gbenroscience/simdext/arrow/
    ArrowBulkEvaluator.java       — main entry point (Builder + evaluate/evaluateInto)
    ArrowSegments.java            — MemorySegment <-> ArrowBuf zero-copy bridge
    VectorCoercion.java           — non-zero-copy fallback casts to Float8Vector
    NullPolicy.java
    ArrowBindingException.java
    ArrowNullValueException.java
    UnsupportedVectorTypeException.java
    package-info.java
  src/test/java/com/github/gbenroscience/simdext/arrow/
    ArrowBulkEvaluatorTest.java
```
