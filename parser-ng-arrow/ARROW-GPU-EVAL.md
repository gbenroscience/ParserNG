# ArrowGpuBulkEvaluator

Evaluates a compiled [ParserNG](https://github.com/gbenroscience/ParserNG) expression directly over
[Apache Arrow](https://arrow.apache.org/) columns, dispatching the bulk numeric work to a GPU
(CUDA or OpenCL) via `GpuExpressionBridge` instead of the CPU SIMD engine used by
[`ArrowBulkEvaluator`](./ArrowBulkEvaluator.java) (see the main [README](./README.md) for that).

It implements the shared [`ArrowExpressionEvaluator`](./ArrowExpressionEvaluator.java) interface,
so most code should compile against that interface (via
[`ArrowExpressionEvaluators`](./ArrowExpressionEvaluators.java)) rather than depend on this class
directly — see [Switching backends](#switching-backends) below.

## Table of contents

- [Why a separate class from `ArrowBulkEvaluator`](#why-a-separate-class-from-arrowbulkevaluator)
- [Binding model](#binding-model)
- [Quick start](#quick-start)
- [Precision: float64 and float32](#precision-float64-and-float32)
- [Compiling / backend selection](#compiling--backend-selection)
- [Choosing a GPU device](#choosing-a-gpu-device)
- [Null handling](#null-handling)
- [Error handling](#error-handling)
- [Thread safety](#thread-safety)
- [Lifecycle](#lifecycle)
- [Switching backends](#switching-backends)
- [Performance notes](#performance-notes)
- [Testing](#testing)

## Why a separate class from `ArrowBulkEvaluator`

Both classes ultimately bind each Arrow column's data buffer as its own zero-copy
`MemorySegment` and hand an array of them straight to the underlying engine — `ArrowBulkEvaluator`
via `SIMDCommandSegmentF64`/`F32`'s `applyBulk(MemorySegment[], MemorySegment)`,
`ArrowGpuBulkEvaluator` via `GpuCompositeExpression`'s identically-shaped
`applyBulk(MemorySegment[], MemorySegment)` (float64) / `applyBulkF32(MemorySegment[], MemorySegment)`
(float32). Neither path flattens or concatenates columns into one buffer, and neither copies a
bound column's data on the way in — see [Performance notes](#performance-notes) for what does and
doesn't allocate.

So the split isn't about copy cost — it's that the two classes wrap fundamentally different
engines with different operational characteristics worth keeping separate:

- **Dispatch target.** One drives a CPU-pinned worker pool; the other drives a GPU command
  queue/stream (CUDA or OpenCL) with its own bootstrap, device selection, and failure modes —
  see [Compiling / backend selection](#compiling--backend-selection) and
  [Choosing a GPU device](#choosing-a-gpu-device), which have no CPU-side equivalent.
- **Registry-width dispatch array.** `GpuCompositeExpression.applyBulk(MemorySegment[], ...)`
  scatters *every* array element to the device unconditionally, and expects the array sized to
  the expression's full variable **registry** (matching how compiled opcodes address slots by
  absolute registry index) — not just the variables this one expression happens to reference. A
  shared/session registry can outlive any single expression, so `ArrowGpuBulkEvaluator`
  back-fills any registry slot this expression doesn't reference with a small zeroed placeholder,
  scoped to that one `evaluate()` call, drawn from a lazily-grown internal scratch buffer — a
  null entry there would NPE during the device scatter. That placeholder never touches real bound
  column data, and this whole path is skipped (zero cost) when every registry slot is referenced,
  which is the common case. `ArrowBulkEvaluator`'s CPU engine has no equivalent concern.
- **Thread-safety shape.** `ArrowBulkEvaluator` has a fully-concurrent non-parallel path;
  `ArrowGpuBulkEvaluator` does not — see [Thread safety](#thread-safety).
- **Precision model.** One compiled `ArrowGpuBulkEvaluator` instance can evaluate *both*
  `Float8Vector` and `Float4Vector` columns; `ArrowBulkEvaluator` needs a separate `compileF32(...)`
  call for that — see [Precision](#precision-float64-and-float32).

## Binding model

Identical to `ArrowBulkEvaluator`: variables are bound to Arrow columns **by name**. The
authoritative name→slot mapping comes from `MathExpression.getSlotItems()`.

A zero-variable expression (e.g. `"42.0"`, or anything that fully constant-folds) never touches the
GPU: `evaluate()` fills the output directly via the ordinary scalar solver on the CPU. Check this
up front with `isConstantExpression()`.

## Quick start

```java
try (BufferAllocator allocator = new RootAllocator();
     ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile("3*sin(x)-cos(2*x)")) {

    Float8Vector x = new Float8Vector("x", allocator);
    x.allocateNew(1000);
    for (int i = 0; i < 1000; i++) x.set(i, i * 0.01);
    x.setValueCount(1000);

    Float8Vector out = ArrowBulkEvaluator.allocateOutput(allocator, "out", 1000);

    eval.evaluate(Map.of("x", x), out);

    // out now holds one result per row.
}
```

Or bind straight from a `VectorSchemaRoot` whose field names match the expression's variables:

```java
eval.evaluate(root, out, NullPolicy.PROPAGATE);
```

Reuse one compiled `ArrowGpuBulkEvaluator` across many `evaluate()` calls where you can —
repeat calls over the same underlying Arrow buffers are effectively allocation-free after the
first call; see [Performance notes](#performance-notes).

## Precision: float64 and float32

Unlike `ArrowBulkEvaluator`, a single compiled `ArrowGpuBulkEvaluator` instance supports **both**
precisions — there's no `compileF32(...)` equivalent, and no `IllegalStateException` guard against
calling the "other" precision's overload:

```java
try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile(expr)) {
    eval.evaluate(f64Columns, f64Output);   // Float8Vector — dispatches via applyBulk
    eval.evaluate(f32Columns, f32Output);   // Float4Vector — dispatches via applyBulkF32
}
```

Each precision maintains its own zero-copy segment cache internally, so alternating between them
on the same instance doesn't invalidate or re-wrap the other precision's cached segments.

As with `ArrowBulkEvaluator`, columns of other numeric Arrow types (`IntVector`, `BigIntVector`,
`DecimalVector`, ...) are not accepted directly — cast/coerce to `Float8Vector` or `Float4Vector`
yourself before binding.

## Compiling / backend selection

| Method | Behavior |
|---|---|
| `compile(String expr)` / `compile(MathExpression expr)` | Auto-selects a backend: CUDA preferred, OpenCL fallback (see `GpuExpressionBridge#from(VectorTurboEvaluator)`'s javadoc for the exact order and its system-property override). |
| `compile(String expr, GpuBackend backend)` / `compile(MathExpression expr, GpuBackend backend)` | Pins a specific backend. Throws if *that* backend can't be bootstrapped — does not silently fall back to the other one. |
| `isBackendAvailable(GpuBackend backend)` | Probes (by compiling and discarding a trivial constant expression) whether `backend` bootstraps on this JVM, without building a real instance. |
| `isAnyGpuAvailable()` | Same probe, but for auto-selection — true if *either* backend bootstraps. |
| `actualBackend()` (instance method) | Which concrete backend an instance ended up on — the useful one to check after an auto-selected `compile()`. |

```java
if (ArrowGpuBulkEvaluator.isAnyGpuAvailable()) {
    try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile(expr)) {
        System.out.println("Running on " + eval.actualBackend());
        ...
    }
} else {
    // fall back to ArrowBulkEvaluator, or use ArrowExpressionEvaluators.compilePreferGpu instead
}
```

## Choosing a GPU device

Both backends let you pick *which* installed device runs the expression, with the same shape of
API — enumerate, then select by name substring or exact index before compiling. The one real
asymmetry left is OpenCL's extra platform dimension (a device sits under a platform) versus CUDA's
flat `0..N-1` device indexing (no platform layer) — everything else lines up.

**OpenCL:**

```java
List<String> devices = ArrowGpuBulkEvaluator.listOpenClDevices();
devices.forEach(System.out::println);
// [platform 0: Intel(R) OpenCL Graphics] [device 0: Intel(R) Corporation Intel(R) UHD Graphics]
// [platform 1: NVIDIA CUDA] [device 0: NVIDIA Corporation NVIDIA GeForce RTX 4080]

ArrowGpuBulkEvaluator.selectOpenClDevice("NVIDIA");                         // substring match
ArrowGpuBulkEvaluator.selectOpenClDevice(OpenClCompositeExpression.GpuVendor.INTEL); // known vendor
ArrowGpuBulkEvaluator.selectOpenClDevice(1, 0);                            // exact platform/device index
ArrowGpuBulkEvaluator.clearOpenClDeviceSelection();                        // back to "first GPU found"
```

**CUDA:**

```java
List<String> devices = ArrowGpuBulkEvaluator.listCudaDevices();
devices.forEach(System.out::println);
// [cuda device 0] NVIDIA GeForce RTX 4080 (compute capability 8.9)
// [cuda device 1] NVIDIA A100-SXM4-80GB (compute capability 8.0)

ArrowGpuBulkEvaluator.selectCudaDevice("A100");   // substring match
ArrowGpuBulkEvaluator.selectCudaDevice(1);        // exact device index
ArrowGpuBulkEvaluator.clearCudaDeviceSelection();  // back to device 0
```

For both backends, selection only affects instances compiled **afterward** — already-compiled
instances keep the device they were built with, and an in-flight expression's GPU never moves
under it. The first time a given device is selected, its context/program is built and cached;
switching selection back and forth (e.g. across test methods) never rebuilds or recompiles
anything for a device already used once. Confirm what an instance actually landed on with
`deviceDescription()`, which returns a real description for both backends:

```java
try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile(expr, GpuBackend.CUDA)) {
    System.out.println(eval.deviceDescription());
    // [cuda device 1] NVIDIA A100-SXM4-80GB (compute capability 8.0)
}
```

The `cuda.device.index` system property keeps working exactly as before for anyone already using
it — `selectCudaDevice(int)` just sets that same property, read fresh on every compile.

## Null handling

Governed by [`NullPolicy`](./NullPolicy.java), identical semantics to `ArrowBulkEvaluator`:

- `IGNORE` (default) — validity bitmaps are never inspected; the output's own bitmap is left
  exactly as the caller supplied it.
- `PROPAGATE` — the output row is marked null if **any** bound input column is null at that row
  (standard SQL/Arrow-style propagation). This is a cheap bitwise-AND pass over the validity
  bitmaps, run after the GPU dispatch — it does not touch the GPU kernel itself.

## Error handling

All binding problems throw [`ArrowBindingException`](./ArrowBindingException.java) (a
`RuntimeException`), with a message naming exactly what went wrong:

- a required variable has no corresponding entry in the bound column map / `VectorSchemaRoot`
- a bound column is shorter than the output's row count
- the output vector hasn't been sized (`allocateNew`/`setValueCount` never called)
- a bound `VectorSchemaRoot` field exists but isn't the expected vector type for the precision
  you're evaluating (`Float8Vector` or `Float4Vector`)
- the GPU dispatch itself throws (wrapped as `ArrowBindingException`, with the expression text
  and the original throwable as cause, for context)

A `rowCount == 0` output is only accepted if every bound column is *also* empty — otherwise it's
treated as "you forgot to size the output", not "empty batch", and throws.

Calling any method after `close()` throws `IllegalStateException`.

## Thread safety

A single instance may be shared and called concurrently from multiple threads and will always
produce correct results, but **every** `evaluate()` call — across both precisions — is internally
serialized against every other call on the same instance. Both GPU backends dispatch through
shared per-instance device state (a command queue/stream, kernel-arg buffers, and an internal
zero-copy segment cache reused across calls) that isn't safe for concurrent use. Unlike
`ArrowBulkEvaluator`, there is no non-serialized fast path here.

If you need true concurrent GPU evaluation from multiple threads, give each thread its own
instance (a separate `compile()` call) rather than sharing one.

## Lifecycle

Call `close()` when done — this releases the compiled expression's device-side resources (device
buffers and, depending on backend, a staging `Arena`), plus any internal scratch memory allocated
for registry-gap back-filling (see
[Why a separate class](#why-a-separate-class-from-arrowbulkevaluator)). Implements
`AutoCloseable`; use try-with-resources. `close()` is idempotent. Do not call `close()` while
another thread may still be inside `evaluate()`.

## Switching backends

Prefer compiling through `ArrowExpressionEvaluators` and coding against `ArrowExpressionEvaluator`
so the backend is a one-line change:

```java
// Pin a backend explicitly:
ArrowExpressionEvaluator eval = ArrowExpressionEvaluators.compile(expr, ArrowExecutionBackend.GPU_OPENCL);

// Or: use the GPU if there is one, fall back to CPU SIMD otherwise:
ArrowExpressionEvaluator eval = ArrowExpressionEvaluators.compilePreferGpu(expr);
```

Reach for `ArrowGpuBulkEvaluator`'s static methods directly only when you need GPU-specific device
selection or introspection (`listOpenClDevices()`, `selectCudaDevice()`, `actualBackend()`,
`deviceDescription()`) that has no equivalent on the CPU side.

## Performance notes

- Binding is zero-copy for both precisions: each bound column's Arrow data buffer is wrapped as
  its own `MemorySegment` alias (via `ArrowMemoryBridge`) and handed straight to the GPU engine —
  no host-side flatten/concatenation buffer, and no copy of column data, for any slot actually
  bound to a real column.
- Repeat `evaluate()` calls over the **same** underlying Arrow buffers (the common
  benchmark/streaming-batch shape) are effectively allocation-free: the instance caches the last
  wrapped `MemorySegment` per slot and only re-wraps when a column's address or requested element
  count actually changes (e.g. Arrow reallocated the buffer, or a differently-sized batch came
  through).
- Constant expressions never touch the GPU.
- Registry-gap back-filling (see
  [Why a separate class](#why-a-separate-class-from-arrowbulkevaluator)) only allocates once, the
  first time a batch exceeds the previous high-water mark of rows — after that it's reused as-is.
- Prefer reusing one `ArrowGpuBulkEvaluator` instance across many `evaluate()` calls rather than
  recompiling per batch — compilation bootstraps a device context and uploads the compiled opcode
  program, and repeat evaluation is what the internal caching above is optimized for.
- Measure against `ArrowBulkEvaluator` for your actual batch sizes and expressions; which one wins
  depends on host↔device transfer overhead vs. raw compute, not on either path doing unnecessary
  copying.

## Testing

See [`ArrowGpuBulkEvaluatorTest`](./ArrowGpuBulkEvaluatorTest.java) for coverage of
compilation/backend selection, introspection, evaluation correctness (`Map` and
`VectorSchemaRoot` binding, both precisions), error handling, `NullPolicy` behavior, lifecycle,
and thread safety. Like `GpuCompositeExpressionTest`, most of these require an actual GPU device
and are gated behind `-Dgpu.tests=true`.