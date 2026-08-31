# ArrowGpuBulkEvaluator

Evaluates a compiled [ParserNG](https://github.com/gbenroscience) expression directly over
[Apache Arrow](https://arrow.apache.org/) `Float8Vector` columns, dispatching the bulk numeric
work to a GPU (CUDA or OpenCL) via `GpuExpressionBridge` instead of the CPU SIMD engine used by
[`ArrowBulkEvaluator`](./ArrowBulkEvaluator.java).

It implements the shared [`ArrowExpressionEvaluator`](./ArrowExpressionEvaluator.java) interface,
so most code should compile against that interface (via
[`ArrowExpressionEvaluators`](./ArrowExpressionEvaluators.java)) rather than depend on this class
directly — see [Switching backends](#switching-backends) below.

## Table of contents

- [Why a separate class from `ArrowBulkEvaluator`](#why-a-separate-class-from-arrowbulkevaluator)
- [Binding model](#binding-model)
- [Quick start](#quick-start)
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

`SIMDCommandSegmentF64.SIMDVectorCompositeExpression.applyBulk` takes a `MemorySegment[]` — one
independent segment per bound variable — which is exactly what lets `ArrowBulkEvaluator` bind each
Arrow column's data buffer with **zero copying**.

`GpuCompositeExpression.applyBulk(MemorySegment, MemorySegment)` has a different contract: it takes
a *single* input segment holding every variable concatenated column-major (slot `s`'s `rowCount`
values occupy `[s*rowCount, (s+1)*rowCount)`). Arrow's per-column buffers are independent
allocations, not laid out contiguously with each other, so `ArrowGpuBulkEvaluator` **cannot** offer
the same zero-copy binding — every `evaluate()` call stages each bound column's data into its slice
of one flat off-heap buffer before dispatch.

That staging copy is on top of whatever host→device transfer the GPU backend itself performs. For
small/medium batches the GPU's raw throughput usually still wins, but this is not a drop-in
"same cost, more parallelism" swap for `ArrowBulkEvaluator` — measure for your batch sizes before
committing to it.

## Binding model

Identical to `ArrowBulkEvaluator`: variables are bound to Arrow columns **by name**. The
authoritative name→slot mapping comes from `MathExpression.getSlotItems()`. Only `Float8Vector`
(Arrow's float64 column type) columns are supported — this class evaluates in full double
precision on the GPU; there is no float32 path exposed here.

A zero-variable expression (e.g. `"42.0"`, or anything that fully constant-folds) never touches the
GPU: `evaluate()` fills the output directly via the ordinary scalar solver. Check this up front with
[`isConstantExpression()`](./ArrowGpuBulkEvaluator.java).

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
`deviceDescription()`, which now returns a real description for both backends:

```java
try (ArrowGpuBulkEvaluator eval = ArrowGpuBulkEvaluator.compile(expr, GpuBackend.CUDA)) {
    System.out.println(eval.deviceDescription());
    // [cuda device 1] NVIDIA A100-SXM4-80GB (compute capability 8.0)
}
```

The `cuda.device.index` system property keeps working exactly as before for anyone already using
it — `selectCudaDevice(int)` just sets that same property, now read fresh on every compile instead
of once, ever, at JVM startup.

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
- a bound `VectorSchemaRoot` field exists but isn't a `Float8Vector`
- the GPU dispatch itself throws (wrapped with the expression text for context)

A `rowCount == 0` output is only accepted if every bound column is *also* empty — otherwise it's
treated as "you forgot to size the output", not "empty batch", and throws.

## Thread safety

A single instance may be shared and called concurrently from multiple threads and will always
produce correct results, but **every** `evaluate()` call is internally serialized against every
other call on the same instance — both GPU backends dispatch through shared per-instance device
state (a command queue/stream and kernel-arg buffers) that isn't safe for concurrent use. Unlike
`ArrowBulkEvaluator`, there is no non-serialized fast path here.

If you need true concurrent GPU evaluation from multiple threads, give each thread its own instance
(a separate `compile()` call) rather than sharing one.

## Lifecycle

Call `close()` when done — this releases the compiled expression's device-side resources (device
buffers and, depending on backend, a staging `Arena`). Implements `AutoCloseable`; use
try-with-resources. `close()` is idempotent. Do not call `close()` while another thread may still be
inside `evaluate()`. Calling `evaluate()` after `close()` throws `IllegalStateException`.

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

- Every `evaluate()` call stages bound columns into one flat off-heap buffer (see
  [Why a separate class](#why-a-separate-class-from-arrowbulkevaluator)) — this is an extra
  host-side copy the CPU SIMD path doesn't pay.
- Constant expressions never touch the GPU.
- Prefer reusing one `ArrowGpuBulkEvaluator` instance across many `evaluate()` calls rather than
  recompiling per batch — compilation bootstraps a device context and uploads the compiled
  opcode program.
- Measure against `ArrowBulkEvaluator` for your actual batch sizes; the GPU generally wins on
  large batches and complex expressions, not necessarily on small ones.

## Testing

See [`ArrowGpuBulkEvaluatorTest`](./ArrowGpuBulkEvaluatorTest.java) — 33 tests covering
compilation/backend selection, introspection, evaluation correctness (`Map` and
`VectorSchemaRoot` binding), error handling, `NullPolicy` behavior, lifecycle, thread safety, and
device selection. Like `GpuCompositeExpressionTest`, most of these require an actual GPU device and
are gated behind `-Dgpu.tests=true`.