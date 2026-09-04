# parser-ng-arrow

Zero-copy Apache Arrow bulk-evaluation bridge for [ParserNG](https://github.com/gbenroscience/ParserNG)
expressions, built on `parser-ng-gpu-simd`'s `SIMDCommandSegmentF64` / `SIMDCommandSegmentF32`
`MemorySegment`-native bulk evaluation API.

This README covers the CPU engine, **`ArrowBulkEvaluator`**. For the GPU-backed engine
(`ArrowGpuBulkEvaluator`, CUDA/OpenCL), see **[ARROW-GPU-EVAL.md](ARROW-GPU-EVAL.md)** — it's
covered only briefly below, in the "Switching backends" section.

## What this is

`SIMDCommandSegmentF64` (and its float32 counterpart, `SIMDCommandSegmentF32`) already speak
`java.lang.foreign.MemorySegment` natively, including a per-column overload —
`applyBulk(MemorySegment[] variables, MemorySegment output)` — designed specifically so each
variable can point at its own independently-allocated off-heap buffer, rather than requiring one
big concatenated segment. That's exactly Arrow's `VectorSchemaRoot` shape: each `Float8Vector` /
`Float4Vector` column owns its own `ArrowBuf`.

`ArrowBulkEvaluator` binds Arrow columns to that API directly — no `double[]`/`float[]` copy of
any input column happens on the way in, and results are written straight into the output
vector's Arrow memory too. Internally, this is done via `ArrowMemoryBridge`, which wraps each
column's `ArrowBuf` as a `MemorySegment` over the same native address
(`MemorySegment.ofAddress(arrowBuf.memoryAddress()).reinterpret(byteSize)`) — no allocation, no
element-by-element staging.

On top of that per-row evaluation, the shared `ArrowExpressionEvaluator` interface also provides
`filter` (row selection via a compiled boolean predicate), `project` (append a computed column),
and `filterProject` (a fused filter-then-project pass, so the projection only ever runs over the
rows that survive the filter) — see [Filtering and projection](#filtering-and-projection-filter-project-filterproject)
below.

## Quick start

```java
import com.github.gbenroscience.arrow.tools.box.ArrowBulkEvaluator;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

try (BufferAllocator allocator = new RootAllocator();
     ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile("(x + y) * z")) {

    int rowCount = 1_000_000;

    // ... x, y, z are Float8Vector columns of length >= rowCount,
    // e.g. loaded from a VectorSchemaRoot you read from an Arrow file/stream.
    Float8Vector output = ArrowBulkEvaluator.allocateOutput(allocator, "result", rowCount);

    evaluator.evaluate(java.util.Map.of("x", x, "y", y, "z", z), output);

    // output now holds one (x + y) * z per row, written directly into Arrow memory.
}
```

If your columns already live together in a `VectorSchemaRoot`, skip building the `Map` yourself:

```java
try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile("(x + y) * z")) {
    Float8Vector output = ArrowBulkEvaluator.allocateOutput(allocator, "result", root.getRowCount());
    evaluator.evaluate(root, output); // resolves x, y, z from root by name
}
```

`evaluator.close()` shuts down the evaluator's CPU-pinned worker pool (if one was created);
always use try-with-resources or call it explicitly when done.

## Binding model

Variables in the expression are bound to Arrow columns **by name**, not by position. The
authoritative name-to-slot mapping comes from `MathExpression.getSlotItems()`, which reflects
exactly the variables the compiled expression actually references and the frame index each one
occupies internally — `ArrowBulkEvaluator` does not guess at or reimplement slot ordering itself.

Call `evaluator.requiredVariableNames()` to find out what names an expression needs. Every one of
those names must have a matching entry in the `Map` (or a matching field name in the
`VectorSchemaRoot`) you pass to `evaluate(...)`; extra map entries are simply ignored.

```java
ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile("sin(x) + y * 2");
String[] needed = evaluator.requiredVariableNames(); // -> ["x", "y"], order not guaranteed
```

## Precision: float64 vs float32

`ArrowBulkEvaluator` supports both `Float8Vector` (float64) and `Float4Vector` (float32) columns,
but **a single compiled instance only ever holds one real engine**:

| Compile with               | Engine held | Evaluate with                          |
|-----------------------------|-------------|------------------------------------------|
| `compile(...)` / `compile(expr, numWorkers)` | float64     | the `Float8Vector` `evaluate(...)` overloads |
| `compileF32(...)` / `compileF32(expr, numWorkers)` | float32     | the `Float4Vector` `evaluate(...)` overloads |

Calling a `Float8Vector` overload on an instance compiled with `compileF32(...)` (or vice versa)
throws `IllegalStateException` immediately, for any expression that references variables. The one
exception is a **constant expression** (see below) — those succeed on either overload regardless
of which precision the instance was compiled for, since neither engine is touched to produce the
result.

Columns of other numeric Arrow types (`IntVector`, `BigIntVector`, `DecimalVector`, ...) are
**not** accepted directly — cast/coerce to `Float8Vector` or `Float4Vector` yourself before
binding. `ArrowBulkEvaluator` deliberately does not perform an implicit narrowing/widening copy,
since doing so silently would reintroduce the exact copy this module exists to eliminate.

## Constant expressions

An expression that references no variables at all (e.g. `"42.0"`, or something that fully
constant-folds) compiles to a zero-slot evaluator. `SIMDCommandSegmentF64`/`F32`'s
`applyBulk(MemorySegment[], ...)` treats a zero-length variable array as a no-op by design — left
unhandled, that would silently leave the output buffer untouched. `ArrowBulkEvaluator` detects
this case up front (`isConstantExpression()`) and fills every output row directly via
`MathExpression.solveGeneric()` instead of going through the SIMD engine at all.

```java
try (ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile("2 * 21")) {
    evaluator.isConstantExpression(); // true
    Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "result", 5);
    evaluator.evaluate(java.util.Map.of(), out); // every row == 42.0
}
```

## Null handling

Bulk evaluation reads straight from each column's raw data buffer via `ArrowMemoryBridge` — it
does **not** inspect Arrow validity bitmaps as part of the numeric computation itself, since doing
so would require a per-element branch that defeats the point of the SIMD fast path. `NullPolicy`
governs whether a separate, cheap pass over the (much smaller) validity bitmaps runs afterward:

- **`NullPolicy.IGNORE`** (default, via the 2- and 3-arg `evaluate` overloads) — fastest. Rows
  where a bound input column is null are still evaluated using whatever bit pattern happens to
  occupy that column's data buffer at that position (Arrow does **not** guarantee this is any
  particular value); the output's own validity bitmap is left exactly as the caller supplied it.
  Use this when you can guarantee the bound columns have no nulls, or when null handling is
  performed separately.
- **`NullPolicy.PROPAGATE`** — after evaluation, the output row is marked null if **any** bound
  input column was null at that row (standard SQL/Arrow-style null propagation), by ANDing every
  bound column's validity bitmap into the output's. The data value at a null output row is left as
  whatever the arithmetic happened to produce — only the validity bit is authoritative once this
  policy is used; don't read the data value at a null row without checking validity first. This
  costs one bitwise-AND pass over the validity bitmaps per bound column — cheap relative to the
  data evaluation itself, since validity bitmaps are 64x smaller than the corresponding data
  buffers (1 bit per row vs. 64 bits).

```java
evaluator.evaluate(columns, output, NullPolicy.PROPAGATE);
```

## Parallel dispatch

The full `evaluate(Map, Float8Vector, NullPolicy, boolean parallel)` overload exposes a `parallel`
flag:

- `parallel = true` (the default used by the shorter overloads) dispatches to the evaluator's
  CPU-pinned worker pool via `applyBulkParallel` — recommended for standalone calls on large
  batches.
- `parallel = false` calls `applyBulk` directly on the calling thread — pass this if the call is
  already running inside your own worker thread and you want to avoid nested parallelism.

```java
// Running inside our own worker pool already — avoid nested parallel dispatch.
evaluator.evaluate(columns, output, NullPolicy.IGNORE, false);
```

Control worker count at compile time:

```java
ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile(expr, 8); // 8 pinned workers
ArrowBulkEvaluator evaluator = ArrowBulkEvaluator.compile(expr, 0); // library default (detected physical cores)
```

## Thread safety

A single `ArrowBulkEvaluator` instance may be shared and called concurrently from multiple
threads and will always produce correct results, but the two evaluation modes differ in how much
actual *concurrency* you get:

- Calls with `parallel = false` run fully concurrently — the underlying evaluator uses a
  `ThreadLocal` context per caller thread.
- Calls with `parallel = true` (the default) are internally **serialized against each other** —
  the engine's worker-pool dispatch uses a single shared coordination structure that is only safe
  for one external caller at a time, so concurrent parallel calls queue rather than overlap.

If you need true concurrent *parallel* evaluation from multiple threads, give each thread its own
`ArrowBulkEvaluator` (a separate `compile(...)` call) rather than sharing one.

The one hard exception either way: never call `close()` while another thread may still be inside
`evaluate(...)`.

## Errors

- **`ArrowBindingException`** (unchecked) — thrown when the evaluator can't bind the expression's
  required variables to the columns supplied: a missing column, a bound column shorter than the
  output's row count, or (via the `VectorSchemaRoot` overloads) a column present under the right
  name but the wrong Arrow vector type. Also thrown if `output` hasn't been sized
  (`allocateNew`/`setValueCount`) but a bound column has rows — use `allocateOutput`/
  `allocateOutputF32` to avoid this.
- **`IllegalStateException`** — thrown when you call an `evaluate(...)` overload for the precision
  this instance wasn't compiled for (see "Precision" above), or when you call any method on an
  evaluator that's already been `close()`d.

## Output allocation

`allocateOutput` / `allocateOutputF32` allocate a properly sized vector (`allocateNew` +
`setValueCount`) with every validity bit pre-set to valid — the shape `evaluate(...)` expects:

```java
Float8Vector out64 = ArrowBulkEvaluator.allocateOutput(allocator, "result", rowCount);
Float4Vector out32 = ArrowBulkEvaluator.allocateOutputF32(allocator, "result", rowCount);
```

If you already have a correctly-sized vector from elsewhere, that's fine too — just make sure
`allocateNew(rowCount)` and `setValueCount(rowCount)` were called before passing it in.

## Filtering and projection: `filter`, `project`, `filterProject`

Beyond per-row `evaluate(...)`, every `ArrowExpressionEvaluator` — so `ArrowBulkEvaluator`
included — supports three higher-level, Arrow-batch-in/Arrow-batch-out operations, all driven by
the same compiled expression machinery above. None of these require a new kind of expression;
`filter` just treats this evaluator's result as a boolean predicate (C-style truthiness: `0.0` is
false, anything else — including negatives, `NaN`, infinities — is true).

| Method | Rows | Columns | Analogous SQL |
|---|---|---|---|
| `filter(root)` | keeps only rows where the predicate is true | unchanged | `SELECT * WHERE <expr>` |
| `project(root, name)` | unchanged | adds one new column | `SELECT *, <expr> AS name` |
| `filterProject(root, projection, name)` | keeps only rows where **this** predicate is true | adds one new column, computed by `projection`, only over the surviving rows | `SELECT *, <expr2> AS name WHERE <expr1>` |

### `filter` — row selection

```java
try (BufferAllocator allocator = new RootAllocator();
     ArrowBulkEvaluator isHot = ArrowBulkEvaluator.compile("temperature > 90.0")) {

    // root is a VectorSchemaRoot with float64 columns, e.g. "temperature", "sensor_id", ...
    VectorSchemaRoot hotReadings = isHot.filter(root); // NullPolicy.IGNORE by default

    // hotReadings has the same schema as root, but only the rows that ran hot.
}
```

### `project` — adding a computed column

```java
try (ArrowBulkEvaluator score = ArrowBulkEvaluator.compile("0.5*rsi + 0.3*macd + 0.2*volume_z")) {
    VectorSchemaRoot scored = score.project(root, "score");

    // scored == root's columns + a new "score" column, one value per existing row.
}
```

### `filterProject` — the fused form, and why it's not just `filter` then `project`

`filterProject` runs this evaluator as the predicate, gathers only the surviving rows, and *then*
runs a second, independently-compiled expression as the projection — over just those surviving
rows, not the original batch. That ordering is the entire point: if `projection` is at all
expensive and the predicate is selective, you skip computing it for every row that's about to be
thrown away.

```java
try (ArrowBulkEvaluator liquidAndVolatile =
             ArrowBulkEvaluator.compile("volume > 1_000_000 && atr > 2.5");
     ArrowBulkEvaluator riskAdjustedScore =
             ArrowBulkEvaluator.compile("0.5*rsi + 0.3*macd + 0.2*volume_z")) {

    // root has 1,000,000 rows; say ~5% pass the liquidity/volatility screen.
    VectorSchemaRoot result = liquidAndVolatile.filterProject(
            root, riskAdjustedScore, "risk_adjusted_score");

    // result has ~50,000 rows: root's columns, restricted to the screened rows,
    // plus "risk_adjusted_score" — computed only for those ~50,000 rows, not all 1,000,000.
}
```

`projection` doesn't have to share a backend with the predicate — a `CPU_SIMD` filter can drive a
GPU-evaluated projection (or vice versa), since each stage dispatches through its own
`evaluate(...)` independently. It does, however, need to be compiled for the same precision
(float64/float32) as `root`, same as the predicate.

A second example — flagging and scoring anomalous sensor readings, computing an expensive
correction only for the rows that actually need it:

```java
try (ArrowBulkEvaluator isAnomalous =
             ArrowBulkEvaluator.compile("abs(reading - rolling_mean) > 3 * rolling_stddev");
     ArrowBulkEvaluator correctedValue =
             ArrowBulkEvaluator.compile("reading - sign(reading - rolling_mean) * rolling_stddev")) {

    VectorSchemaRoot corrected = isAnomalous.filterProject(root, correctedValue, "corrected_reading");
    // Only the anomalous rows are kept, each with its correction attached;
    // correctedValue never runs over the (presumably much larger) set of normal readings.
}
```

Both `filter`/`project`/`filterProject` accept an explicit `NullPolicy` overload too
(`filter(root, NullPolicy.PROPAGATE)`, etc.) — semantics match `evaluate(...)`'s null handling
described above, applied to the predicate for `filter`/`filterProject` and to the computed column
for `project`/`filterProject`.

`filter` and `filterProject` copy the surviving rows into a new `VectorSchemaRoot` (there's no way
to select a row subset without copying in Arrow's columnar layout); `project` does not copy
`root`'s existing columns at all — only the new column is freshly allocated — which is what makes
`project` considerably cheaper than `filter` for large batches when you don't also need to drop
rows. `filterProject` pays the row-copy cost once (for the smaller, filtered batch), never for the
original batch.

If no rows survive the predicate, `filter` and `filterProject` both return a valid, empty
(zero-row) `VectorSchemaRoot` with the expected schema — not `null` and not an exception.

## Switching backends (CPU vs GPU)

`ArrowBulkEvaluator` implements the shared `ArrowExpressionEvaluator` interface, alongside the
GPU-backed `ArrowGpuBulkEvaluator` (CUDA/OpenCL). If a call site should stay agnostic about which
engine actually runs — or you want "GPU if available, otherwise CPU" — compile through
`ArrowExpressionEvaluators` instead of calling `ArrowBulkEvaluator.compile(...)` directly:

```java
import com.github.gbenroscience.arrow.tools.box.*;

// Pinned to a specific backend:
ArrowExpressionEvaluator evaluator =
        ArrowExpressionEvaluators.compile(expr, ArrowExecutionBackend.CPU_SIMD);

// "Use the GPU if there is one, otherwise fall back to CPU":
ArrowExpressionEvaluator evaluator = ArrowExpressionEvaluators.compilePreferGpu(expr);
```

`ArrowExpressionEvaluator` intentionally leaves out `ArrowBulkEvaluator`'s `parallel` flag and
`ArrowGpuBulkEvaluator`'s device-selection/introspection methods (`actualBackend()`,
`deviceDescription()`, `listOpenClDevices()`, etc.) — those are backend-specific tuning knobs.
Downcast to `ArrowBulkEvaluator` (guided by `evaluator.backend()`) when you need the `parallel`
flag specifically, or downcast to `ArrowGpuBulkEvaluator` for GPU device selection.

**For everything about the GPU engine itself** — device selection, CUDA vs OpenCL, availability
checks, and its own known limitations — see **[ARROW-GPU-EVAL.md](ARROW-GPU-EVAL.md)**.

## Variable ordering (internals, not a caller concern)

`SIMDCommandSegmentF64`/`F32`'s `applyBulk(MemorySegment[], MemorySegment)` expects
`variables[i]` to be the data for whichever variable ParserNG assigned to slot `i` when it
compiled the expression. `ArrowBulkEvaluator` builds that array itself from
`MathExpression.getSlotItems()` — callers bind by **name** via the `Map`/`VectorSchemaRoot`
overloads and never need to know or supply slot order directly.

(Confirmed by ParserNG's author: variables are discovered and assigned slots on a
first-appearance, left-to-right basis as the expression is scanned — `"x + y * z"` binds slot 0 =
`x`, slot 1 = `y`, slot 2 = `z`. This is purely an internal detail of how `ArrowBulkEvaluator`
constructs the `MemorySegment[]`; it does not affect the public API.)

## What's zero-copy, honestly

| Path | Zero-copy? |
|---|---|
| Input column → engine (`Float8Vector`/`Float4Vector`, any expression) | Yes — reads straight from `ArrowBuf` via `MemorySegment` |
| Engine → output column | Yes — writes straight into the output vector's `ArrowBuf` |
| `NullPolicy.PROPAGATE`'s validity-bitmap merge | Technically a copy/AND over bitmaps, but it's `rowCount / 8` bytes — negligible next to the `rowCount * 8` (or `* 4`)-byte data path |
| Non-`Float8Vector`/`Float4Vector` columns (`IntVector`, `BigIntVector`, `DecimalVector`, ...) | No — not supported; caller must cast/coerce upstream |

Whether the *arithmetic itself* touches Arrow memory in place, or stages an operand into on-heap
scratch first, depends on what the expression does — see `SIMDCommandSegmentF64`/`F32`'s own docs.

## Requirements

- JDK with the Foreign Function & Memory API finalized (`MemorySegment.ofAddress`/`reinterpret`) —
  matches `parser-ng-gpu-simd`'s own JDK requirement for CPU pinning.
- `jdk.incubator.vector` on the module path at compile *and* run time
  (`--add-modules jdk.incubator.vector`) — inherited from `parser-ng-gpu-simd`.
- `--enable-native-access=ALL-UNNAMED` at runtime (or the module-qualified equivalent) —
  `ArrowMemoryBridge` calls `MemorySegment.reinterpret(long)`, a restricted FFM method; calls will
  throw at runtime without this flag.
- The `--add-opens` flags Arrow itself needs for its off-heap allocator (`java.base/java.nio`,
  `java.base/java.util`) — see `pom.xml`'s `surefire` config for the flag set used in tests, and
  mirror it in your own run scripts / shaded-jar manifest args.
- Linux is where the underlying engine's CPU pinning (and therefore `applyBulkParallel` worker
  efficiency) is strongest, per its own class docs.

## Honest limitations / not done here

- **Only `Float8Vector` and `Float4Vector` are zero-copy.** Every other Arrow vector type must be
  cast/coerced to one of these by the caller before binding — `ArrowBulkEvaluator` throws
  `ArrowBindingException` rather than silently copying.
- **`ArrowMemoryBridge` relies on Arrow-internal memory layout guarantees** (`ArrowBuf`'s
  `memoryAddress()`/`capacity()` being stable and meaning what this module assumes) that this
  module cannot independently verify at compile time. Cover it with an integration test against
  the exact Arrow Java version pinned in your `pom.xml` before trusting it in production — ideally
  a round-trip test that writes known values into a real `Float8Vector`, wraps it via
  `ArrowMemoryBridge`, reads the values back through the returned segment, and asserts equality.
- **No streaming convenience yet.** Call `evaluate(root, output, ...)` once per batch in your own
  `ArrowStreamReader`/`ArrowFileReader` loop.

## Module layout

```
parser-ng-arrow/
  pom.xml
  ARROW-GPU-EVAL.md                        # GPU engine (ArrowGpuBulkEvaluator) — separate doc
  src/main/java/com/github/gbenroscience/arrow/tools/box
    ArrowBindingException.java
    ArrowBulkEvaluator.java                # this README's subject — CPU, SIMD-vectorized
    ArrowExecutionBackend.java             # CPU_SIMD / GPU_AUTO / GPU_CUDA / GPU_OPENCL
    ArrowExpressionEvaluator.java          # shared backend-agnostic interface
    ArrowExpressionEvaluators.java         # single entry point for compiling either backend
    ArrowFilterSupport.java                # shared filter()/project() row-select & column-append logic
    ArrowGpuBulkEvaluator.java             # GPU engine — see ARROW-GPU-EVAL.md
    ArrowMemoryBridge.java                 # ArrowBuf <-> MemorySegment, zero-copy
    NullPolicy.java                        # IGNORE / PROPAGATE
    package-info.java
  src/test/java/com/github/gbenroscience/arrow/tools/box
    ArrowBulkEvaluatorTest.java
    ArrowGpuBulkEvaluatorTest.java
```