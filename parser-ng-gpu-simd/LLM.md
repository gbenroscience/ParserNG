# GPU Llama Runner; CUDA / OpenCL / Metal

This subsystem runs GGUF-format Llama-family models end-to-end on the
GPU: load weights → tokenize a prompt → embed → run N decoder layers
(attention + FFN) → final norm + LM head → sample → detokenize. It
exists **three times**, once per GPU backend, as three parallel
packages under `com.github.gbenroscience.gpu.llm`:

| Package | Backend | Target hardware |
|---|---|---|
| `com.github.gbenroscience.gpu.llm.cuda`   | CUDA driver API + NVRTC | NVIDIA GPUs |
| `com.github.gbenroscience.gpu.llm.opencl` | OpenCL                  | Any OpenCL 1.2+ device (Intel/AMD/NVIDIA GPUs, and CPUs via an ICD like PoCL) |
| `com.github.gbenroscience.gpu.llm.metal`  | Metal (via the Objective-C runtime) | Apple Silicon Macs (unified memory required) |

All three implement **the same model, the same algorithms, and the
same public shape**; a class in one package has a same-named (or
clearly-corresponding) counterpart in the other two, doing the exact
same thing, differing only in how it talks to the GPU. This README
covers what's shared, what's genuinely backend-specific, and how to
build/run/select between them. Each package's own file-level javadoc
goes into more depth on its specific translation choices; this doc is
the map that ties the three together.

---

## 1. Why three copies instead of one abstraction

The lowest layer of each backend (kernel dispatch, memory allocation,
synchronization) is different enough; CUDA's stream model, OpenCL's
command queues, Metal's encode-then-commit command buffers; that a
single shared implementation would either leak abstraction everywhere
or hide backend-specific correctness/performance details that matter.
Instead, each backend gets a **complete, independent, self-contained
port**, and a thin backend-agnostic layer sits above all three (see
§4). The tradeoff this creates; **three copies of the same algorithm
that must be kept in sync**; is real and is called out explicitly in
§8.

---

## 2. Shared architecture (identical across all three backends)

Every backend implements the same pipeline, with the same file
responsibilities:

```
GGUF file on disk
   │
   ▼
GGUFLoader.GGUFFile  (tensors + metadata, off-heap mapped)
   │
   ├──► LlamaLayer.Config.fromGguf(...)      architecture params (dim, n_layers, heads, RoPE, ...)
   ├──► BpeTokenizer.fromGguf(...)           byte-level BPE tokenizer (GPT-2/Llama-3/Mistral style)
   ├──► EmbeddingTable.fromGguf(...)         host-resident token embedding table
   └──► LlamaLayer.GpuWeights.fromGguf(...)  per-layer weights, uploaded to the GPU (one instance per layer)
   │
   ▼
<Backend>LlamaEngine (implements LlamaGpuEngine)
   │  owns: GpuContext, GpuWeights[], GpuState[], EmbeddingTable, BpeTokenizer
   │
   ▼
<Backend>-flavored engine class (Llama<Backend>Engine)
   │  generate(prompt, GenerationConfig):
   │    1. tokenize prompt
   │    2. processPrompt: batched prefill (LlamaLayer.prefill_layer) for the
   │       first min(T, max_prefill_batch) tokens, then forward_layer
   │       one-at-a-time for any overflow
   │    3. finalLogits → Sampler.sample → next token
   │    4. loop: embed → forward_layer × n_layers → finalLogits → sample,
   │       until EOS or maxNewTokens
   │    5. detokenize the generated ids
   ▼
generated text
```

### Files present in every backend package, doing the identical job

| File | Responsibility (identical algorithm in all three) |
|---|---|
| `<Kernel>Source.java` | The GPU kernel source (CUDA C / OpenCL C / MSL); same 21 kernels, same names, same math, same tolerances. |
| `GpuContext.java` | Bootstraps a device, compiles the kernel source once, resolves every kernel entry point up front, holds them for the process lifetime. |
| `<Backend>DeviceSelector.java` | Enumerates devices, resolves which one to bind to, via a `System.setProperty` "select before construct" contract shared across all three. |
| `LlamaLayer.java` | The actual model math: `Config`, `GpuWeights`, `GpuState`, `forward_layer` (single-token decode), `prefill_layer` (batched prompt processing), and every kernel-dispatch helper. This is the biggest file in each package and the one to read first to understand the model itself. |
| `EmbeddingTable.java` | Host-resident `float[vocabSize*dim]` table; gathers one row (or T rows for prefill) per call rather than keeping the whole table GPU-resident. |
| `BpeTokenizer.java` | Byte-level BPE tokenizer (GPT-2/Llama-3/Mistral/Falcon style). **Byte-identical across all three backends**; this file has no GPU-specific code at all. |
| `Sampler.java` | Host-side temperature / top-k / top-p / repetition-penalty sampling over logits already read back from the GPU. Each backend's `Sampler` is a thin wrapper over the single implementation in `core.Sampler` (order: temperature -> top-k -> renormalise -> top-p -> sample; NaN/+Inf logits rejected; config validated), so the backends cannot sample differently. |
| `Llama<Backend>Engine.java` | Top-level orchestrator: tokenize → prefill → decode loop → detokenize. (`LlamaCudaEngine` / `LlamaOpenCLEngine` / `LlamaMetalEngine`.) |
| `<Backend>LlamaEngine.java` | Implements the backend-agnostic `LlamaGpuEngine` interface; owns GGUF loading and full lifecycle. (`CudaLlamaEngine` / `OpenCLLlamaEngine` / `MetalLlamaEngine`.) |
| `<Backend>Demo.java` | End-to-end wiring example / minimal CLI (`CudaDemo` / `OpenCLDemo` / `MetalDemo`). |
| `ActivationBenchmark.java` | Correctness + speed check for the three FFN activations (SwiGLU/GeGLU/GeLU) in isolation. |
| `CoreKernelBenchmark.java` | The full 12-check correctness suite plus speed sweeps for every other kernel (GEMV/GEMM, RMSNorm, RoPE, attention, both decode and batched-prefill variants). |

### Model-level behavior that's identical regardless of backend

- **Quantization**: weights are Q8_0-quantized (32-value blocks, fp16
  scale + 32 int8 values, 34 bytes/block); activations are quantized
  on the fly before each GEMV/GEMM. `wo`, RMSNorm gammas, and the LM
  head stay F32.
- **Activation types**: `SWIGLU` (default), `GELU` (ungated), `GEGLU`
 ; selected via `Config.activationType`, since GGUF has no metadata
  key for this; set it manually after `Config.fromGguf(...)` if your
  model isn't SwiGLU.
- **RoPE**: standard rotary embeddings plus Llama-3-style NTK-by-parts
  frequency scaling (identity when `rope_scaling_factor == 1.0`).
- **Attention**: GQA-aware (grouped query attention), causal, computed
  per-head as three dispatches (scores → softmax → weighted-sum);
  **not** a fused flash-attention kernel in any backend.
- **Batched prefill**: `prefill_layer` processes up to
  `Config.max_prefill_batch` prompt tokens in one call per layer. It
  is **single-call-only per sequence**; it attends solely within its
  own batch, never the persistent KV cache; so it cannot be chunked
  across multiple calls. Prompts longer than `max_prefill_batch` fall
  back to sequential `forward_layer` decode calls for the overflow;
  `Llama<Backend>Engine.generate(...)` handles this automatically.
- **Tokenizer coverage**: byte-level BPE only (GGUF
  `tokenizer.ggml.model == "gpt2"`). SentencePiece-unigram tokenizers
  (original Llama 1/2, GGUF `"llama"`) are **not implemented**;
  `BpeTokenizer.fromGguf` throws rather than silently mis-tokenizing.
- **Tensor types**: only GGML F32 (type 0) and Q8_0 (type 8) are
  read. Other quantization schemes (Q4_K, Q5_K, etc.) will throw.
- **Threading**: one engine instance owns one sequence's `GpuState`
  (KV cache + position). `generate()` is **not** safe to call
  concurrently on the same instance; construct one engine per
  concurrent conversation; sharing a `GpuContext`/`GpuWeights` across
  multiple engines is fine and intended.

---

## 3. What's genuinely backend-specific

| Concern | CUDA | OpenCL | Metal |
|---|---|---|---|
| Kernel language | CUDA C (`extern "C" __global__`) | OpenCL C (`__kernel`) | Metal Shading Language (`kernel void`) |
| Compile step | NVRTC, JIT to PTX at process start | `clBuildProgram` at process start | `newLibraryWithSource:options:error:` at process start |
| Device pointer representation | raw `long` CUdeviceptr, pointer-arithmetic-able | `MemorySegment` cl_mem handle | `MetalBuffer` (buffer id + byte offset); **not** pointer-arithmetic-able the way the other two are; see that class's javadoc |
| Host↔device transfer | explicit `cuMemcpyHtoD`/`cuMemcpyDtoH` | explicit `clEnqueueWriteBuffer`/`clEnqueueReadBuffer` | none needed for data; `MTLResourceStorageModeShared` buffers are simultaneously host- and device-visible on Apple Silicon's unified memory; only **ordering** (via `waitUntilCompleted`) needs explicit handling |
| Kernel launch grid model | `blockIdx`/`blockDim`/`threadIdx`, host computes grid dimensions | global/local work-size, host computes grid dimensions | `dispatchThreads:threadsPerThreadgroup:`; host supplies the *exact* total thread count; Metal handles the ragged edge itself |
| Sync discipline | one command stream; kernels launched in issued order, no per-kernel sync needed except at genuine host-readback points | one in-order command queue, same idea | **per-kernel** commit + `waitUntilCompleted` (a command buffer must fully finish before you can encode a kernel depending on its output; see `LlamaLayer`'s Metal javadoc for why this is the deliberately simpler, not deliberately fastest, choice) |
| CPU device support | never (no CPU concept in the CUDA driver API) | yes, via a CPU OpenCL ICD (e.g. PoCL) | never (no CPU concept in Metal) |
| Discrete/non-unified-memory GPUs | N/A (CUDA has no unified-memory requirement) | supported | **not** supported by this port; `MetalDeviceSelector.resolve()` rejects a device with `hasUnifiedMemory == false` |

---

## 4. The backend-agnostic layer

Above all three packages sits `com.github.gbenroscience.gpu.llm`:

- **`LlamaGpuEngine`**; the interface all three `<Backend>LlamaEngine`
  classes implement: `generate(prompt, cfg)`, `getDeviceDescription()`,
  `close()`.
- **`LlamaGenerationConfig`**; one flat, backend-agnostic config
  shape (maxNewTokens, sampler settings) that `LlamaGpuBridge`/each
  engine translates into its own backend's nested config type.
- **`LlamaGpuBridge`**; the single entry point most callers should
  use. Auto-detects a working backend (probes each by constructing and
  immediately closing a bare `GpuContext`; no model weights loaded)
  or accepts an explicit `GpuBackend` choice:

  ```java
  // auto-detect: tries backends in gpu.backend.preference order
  // (default "cuda,metal,opencl"), falls through to the next on failure
  try (LlamaGpuEngine llm = LlamaGpuBridge.load(new File("model.gguf"))) {
      String out = llm.generate("Once upon a time,", new LlamaGenerationConfig());
  }

  // explicit backend
  try (LlamaGpuEngine llm = LlamaGpuBridge.load(new File("model.gguf"), GpuBackend.METAL)) {
      String out = llm.generate("Once upon a time,", new LlamaGenerationConfig());
  }
  ```

  Set `-Dgpu.backend.preference=metal,cuda,opencl` (or any subset/order)
  to control auto-detection; this is the **same** property
  `GpuExpressionBridge` (the math-evaluator side of this codebase)
  reads, deliberately shared rather than Llama-specific.

- **`GpuBackend`**; `{ CUDA, OPENCL, METAL }`.

### Device selection (below backend selection)

Each backend has its own device selector, called *before*
`LlamaGpuBridge.load(...)` (or before constructing a `GpuContext`
directly) to target a specific device when more than one is present:

```java
CudaDeviceSelector.selectDevice(1);              // -Dcuda.device.index=1
OpenClDeviceSelector.selectDevice(vendorOrIndex); // -Dopencl.platform.index / .device.index
MetalDeviceSelector.selectDevice(0);              // -Dmetal.device.index=0
```

---

## 5. Building & running

All three packages target Java's FFM API (`java.lang.foreign`) and
need `--enable-preview` (or a JDK version where FFM has graduated,
matching whatever the rest of this codebase targets).

```bash
# CUDA (needs an NVIDIA GPU + driver + CUDA Toolkit for NVRTC)
java --enable-preview -cp <classpath> \
     com.github.gbenroscience.gpu.llm.cuda.CudaDemo model.gguf "Your prompt here"

# OpenCL (needs any OpenCL 1.2+ ICD)
java --enable-preview -cp <classpath> \
     com.github.gbenroscience.gpu.llm.opencl.OpenCLDemo model.gguf "Your prompt here"

# Metal (needs an Apple Silicon Mac)
java --enable-preview -cp <classpath> \
     com.github.gbenroscience.gpu.llm.metal.MetalDemo model.gguf "Your prompt here"
```

### Validating a backend before trusting it

Every backend ships the same two benchmark/correctness harnesses.
**Run these first** on any new device/driver combination; all three
`main()` methods exit non-zero if any correctness check fails, and
explicitly say not to trust generation output until they pass:

```bash
java --enable-preview -cp <classpath> com.github.gbenroscience.gpu.llm.<backend>.ActivationBenchmark
java --enable-preview -cp <classpath> com.github.gbenroscience.gpu.llm.<backend>.CoreKernelBenchmark
```

`CoreKernelBenchmark` checks all 12 kernel families (quantize, both
GEMV variants, F32 GEMV, RMSNorm, RoPE, decode attention, both GEMM
variants, batched RMSNorm/RoPE/attention) against CPU reference
implementations, then benchmarks each at realistic Llama-2-7B-shaped
dimensions.

---

## 6. Known limitations (apply identically to all three backends)

- SentencePiece-unigram tokenizers (`tokenizer.ggml.model == "llama"`)
  are not supported; only byte-level BPE (`"gpt2"`).
- Only F32 and Q8_0 GGUF tensor types are read.
- Attention is unfused (three dispatches per head) in every backend;
  none of the three implement a flash-attention-style single kernel.
- The GEMM kernels (`q8_0_gemm_tiled`, `f32_gemm_tiled`) are a
  straightforward one-thread-per-output-element scheme; not
  cuBLAS/MPS/clBLAS-competitive. A shared-memory-tiled GEMM is flagged
  as future work in every backend, not attempted.
- `prefill_layer` is single-call-only per sequence (see §2) in every
  backend.
- One engine instance = one sequence; not thread-safe for concurrent
  `generate()` calls against the same instance, in every backend.

## 7. Standing caveat

### Verification status (updated by the engineering audit; see `LLM-AUDIT.md`)

| Backend | Java compiles | Kernels executed | Checked against CPU reference | Physical GPU |
|---|---|---|---|---|
| OpenCL | yes | **yes, on a PoCL CPU device** | **yes** (53-test suite, end-to-end logits within 1e-4 abs) | not reproduced by the audit (the author's earlier hardware run predates the fixes below and must be repeated) |
| CUDA | yes (javac only) | **no** | no | **no** |
| Metal | yes (javac only) | **no** | no | **no** |

An OpenCL run on a CPU device proves the OpenCL kernels and the host logic;
it does not prove behaviour on any GPU. Run the suite on real hardware
(`-Dopencl.device.type=GPU`) before calling OpenCL production-verified.
CUDA and Metal received the same source changes as OpenCL (V projection,
Q/K layout, RoPE kernel, fp16 rounding is already RNE there) **without ever
being compiled by NVRTC / Metal or executed**.

The audit found and fixed a decode/prefill KV-layout inconsistency that
affected all three backends: decode projected Q/K/V through the even/odd
"split" GEMV (V should never be permuted) while batched prefill wrote the
natural adjacent-pair layout into the same KV cache. Decode and prefill now
both use the natural GGUF layout. The old `q8_0_gemv_split` kernel is no
longer used by the engines.

### Performance flags (OpenCL)

* `-Dllm.timing=true` prints time-to-first-token, decode tokens/s and the host wall split per token.
* `-Dllm.profile=true` enables the OpenCL event profiler: device time per kernel / phase / layer, launches, synchronisations, allocations, transfers.
* `-Dllm.maxContext=N` caps the context (default 8192) so KV-cache memory is not sized for the model's full context.
* `-Dllm.gemv=legacy`, `-Dllm.attention=legacy`, `-Dllm.rmsnorm=legacy` select the original generic kernels for A/B comparison.
* `-Dllm.lmhead=f32` keeps the LM head dequantised to F32 (the default keeps a Q8_0 head as Q8_0).
* `OpenCLBenchmark --gemv-sweep` / `--memory-plan` (model-free): event-timed projection shapes with a rows-vs-bytes fit, and a categorized device-memory plan. The profile report tags every command with its operation (q proj, attention, lm head, ...).
* `OpenCLBenchmark <model.gguf>` runs a reproducible cold/warm timing run plus a profiled run; `--bandwidth` (or `--bandwidth-only`, no model) adds a streaming-read bandwidth probe.
* `-Dopencl.device.type=CPU|GPU` and `OpenCLDeviceSelector` choose the device.

### Original caveat (kept for history)

Of the three packhges, only OpenCL has been verified on hardware.
**Every file in the CUDA and METAL packages are yet UNVERIFIED**. 
The CUDA and Metal hardware/toolchains were unavailable while writing this
code. Each backend's kernels and dispatch logic were traced carefully
against their respective APIs and against each other for algorithmic
consistency, but none of the two have actually been compiled or run
against real hardware. Run `ActivationBenchmark` and
`CoreKernelBenchmark` on real hardware before trusting any backend's
output, and treat a passing correctness suite on one backend as
evidence about that backend only.

## 8. Maintaining three copies in sync

Because this is a genuine triplication rather than a shared
abstraction (§1), a change to the *model* (a new activation type, a
config field, a fix to an algorithm) needs to be ported to **all
three** `LlamaLayer.java` files, plus the corresponding kernel source
file, plus both benchmark files, to stay consistent. `BpeTokenizer.java`
is the exception (still three identical copies; pure host-side logic), and sampling,
RoPE tables and model validation now live once in `llm/core`
(`Sampler`, `RopeTables`, `ModelSpec`) -- change them there, not per backend. When changing
shared model behavior, grep for the same method/field name across all
three `<backend>` packages before considering the change complete.