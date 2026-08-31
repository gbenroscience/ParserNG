package com.github.gbenroscience.gpu.llm.metal;

/**
 * The Metal counterpart of the {@code long} CUdeviceptr values the CUDA
 * port ({@code com.github.gbenroscience.gpu.llm.cuda}) threads through
 * every method signature, and of the {@code MemorySegment} cl_mem handles
 * the OpenCL port uses. THIS is the single biggest structural difference
 * between this port and its two siblings, so it gets its own class and
 * this up-front explanation rather than being silently absorbed into a
 * {@code long}.
 *
 * <b>WHY A WRAPPER, NOT A BARE LONG:</b> a CUdeviceptr is a raw device
 * VIRTUAL ADDRESS -- host code is free to do pointer arithmetic on it
 * ({@code xBatch + row * dim * 4}) and pass the result straight into
 * another kernel launch's argument list, because CUDA kernel arguments
 * really are just addresses. An {@code id<MTLBuffer>} is an OBJECTIVE-C
 * OBJECT, not an address -- every Metal API that touches a buffer
 * (encoder {@code setBuffer:offset:atIndex:}, {@code contents}, residency
 * tracking) wants the object identity PLUS a separate byte offset, not a
 * moved pointer. Collapsing that pair into one {@code long} would either
 * lose the object identity (if you tried to use the raw {@code contents}
 * host address as "the pointer" -- see below for why that's tempting but
 * wrong) or lose offset arithmetic (if you used the object pointer alone).
 * {@link MetalBuffer} keeps both, and {@link #withOffset(long)} is the
 * direct replacement for the CUDA port's "device pointer + N" arithmetic
 * (see e.g. {@code LlamaCudaEngine.processPrompt}'s
 * {@code xBatch + (prefillLen-1)*dim*4} -- the Metal equivalent is
 * {@code xBatch.withOffset((long)(prefillLen-1)*cfg.dim*Float.BYTES)}).
 *
 * <b>WHY {@link #contents} MAKES THIS CHEAPER THAN THE CUDA PORT, NOT
 * JUST DIFFERENT:</b> every {@link MetalBuffer} here is allocated with
 * {@code MTLResourceStorageModeShared} (see {@link MetalBindings}'s
 * javadoc for why that's the right default on Apple Silicon's unified
 * memory architecture, and the deliberate exception for discrete/eGPU
 * Macs). A shared-storage buffer's {@code contents} pointer IS host
 * memory -- the GPU reads/writes the exact same bytes the CPU does, with
 * no explicit copy command and no driver-mediated staging buffer. That
 * means every host round-trip the CUDA/OpenCL ports need for a genuine
 * PCIe transfer (KV-cache writes via {@code writeIntoCache}, the RMSNorm
 * partial-sum readback, the final logits readback) is, on this backend, a
 * plain {@code MemorySegment.copy} against {@link #contents} -- see
 * {@link LlamaLayer}'s upload/download helpers, which are consequently
 * much shorter than their CUDA counterparts. The one thing a shared
 * buffer does NOT give you for free is ordering: the GPU must actually
 * finish writing before the CPU reads {@link #contents}, which is exactly
 * why {@link GpuContext}'s dispatch helpers still call
 * {@code waitUntilCompleted} at the same points the CUDA port relies on a
 * synchronous {@code cuMemcpyDtoH} for -- see {@link LlamaLayer}'s class
 * javadoc for the full sync-discipline discussion.
 *
 * Immutable and cheap to pass around -- {@link #withOffset(long)} never
 * mutates {@code this}, it returns a new value sharing {@link #id} and
 * {@link #contents} with an adjusted {@link #offset}.
 */
public final class MetalBuffer {

    /** The {@code id<MTLBuffer>} Objective-C object pointer -- pass this to setBuffer:offset:atIndex:. */
    public final long id;

    /** Byte offset into the buffer this value logically points at (the second half of setBuffer:offset:atIndex:). */
    public final long offset;

    /**
     * The buffer's {@code contents} base address (offset 0), i.e. the raw
     * host-visible pointer Metal handed back from {@code -[MTLBuffer contents]}
     * at allocation time. NOT adjusted by {@link #offset} -- callers that
     * need the actual host address this value refers to must add
     * {@link #offset} themselves (see {@link #hostAddress()}).
     */
    public final long contents;

    /** Total capacity of the underlying buffer in bytes (from allocation, not this view's remaining span). */
    public final long capacityBytes;

    public MetalBuffer(long id, long offset, long contents, long capacityBytes) {
        this.id = id;
        this.offset = offset;
        this.contents = contents;
        this.capacityBytes = capacityBytes;
    }

    /** The Metal equivalent of a CUDA {@code devicePtr + extraByteOffset} -- same object, offset advanced. */
    public MetalBuffer withOffset(long extraByteOffset) {
        return new MetalBuffer(id, offset + extraByteOffset, contents, capacityBytes);
    }

    /** The actual host-visible address this value refers to right now ({@link #contents} + {@link #offset}). */
    public long hostAddress() {
        return contents + offset;
    }

    /** Sentinel for "no buffer" (mirrors the CUDA/OpenCL ports' {@code 0L} convention for an unallocated/optional pointer). */
    public static final MetalBuffer NULL = new MetalBuffer(0L, 0L, 0L, 0L);

    public boolean isNull() {
        return id == 0L;
    }

    @Override
    public String toString() {
        return "MetalBuffer[id=0x" + Long.toHexString(id) + ", offset=" + offset + ", cap=" + capacityBytes + "]";
    }
}