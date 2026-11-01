package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * OpenCL counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.EmbeddingTable}.
 * Same design and rationale as the CUDA original -- kept HOST-resident (a
 * flat float[] of vocabSize*dim), one row gathered and uploaded per call
 * rather than keeping the whole table device-resident. See the CUDA
 * original's class javadoc for the full cost argument; nothing about that
 * tradeoff is GPU-API-specific, so it applies unchanged here.
 *
 * The only real change is the upload mechanism: {@code cuMemcpyHtoD} becomes a
 * blocking {@code clEnqueueWriteBuffer} (CL_TRUE for the blocking_write
 * parameter), which -- same as everywhere else in this port -- both
 * performs the transfer and acts as the host-needs-a-result synchronization
 * point against this context's in-order queue.
 */
public final class EmbeddingTable {

    private final float[] flatEmbeddings; // [vocabSize * dim], row-major (null when the table is kept as Q8_0)
    private final byte[] q8Embeddings;    // Q8_0 blocks, row-major (null when the table is F32)
    private final int vocabSize;
    private final int dim;

    public EmbeddingTable(float[] flatEmbeddings, int vocabSize, int dim) {
        if ((long) flatEmbeddings.length != (long) vocabSize * dim) {
            throw new IllegalArgumentException(
                    "flatEmbeddings.length (" + flatEmbeddings.length + ") != vocabSize*dim ("
                            + vocabSize + "*" + dim + "=" + ((long) vocabSize * dim) + ")");
        }
        this.flatEmbeddings = flatEmbeddings;
        this.q8Embeddings = null;
        this.vocabSize = vocabSize;
        this.dim = dim;
    }

    private EmbeddingTable(byte[] q8, int vocabSize, int dim) {
        if (dim % 32 != 0) {
            throw new IllegalArgumentException("Q8_0 embedding row length " + dim + " is not a multiple of 32");
        }
        long expected = (long) vocabSize * (dim / 32) * 34;
        if (q8.length != expected) {
            throw new IllegalArgumentException("Q8_0 embedding bytes (" + q8.length + ") != vocabSize*dim/32*34 (" + expected + ")");
        }
        this.flatEmbeddings = null;
        this.q8Embeddings = q8;
        this.vocabSize = vocabSize;
        this.dim = dim;
    }

    /** Copies row {@code tokenId} into {@code dst[dstOff..dstOff+dim)}, dequantising Q8_0 blocks on the fly. */
    private void rowInto(float[] dst, int dstOff, int tokenId) {
        if (flatEmbeddings != null) {
            System.arraycopy(flatEmbeddings, tokenId * dim, dst, dstOff, dim);
            return;
        }
        final int blocks = dim / 32;
        int src = tokenId * blocks * 34;
        for (int b = 0; b < blocks; b++, src += 34) {
            float scale = Float.float16ToFloat((short) ((q8Embeddings[src] & 0xFF) | ((q8Embeddings[src + 1] & 0xFF) << 8)));
            int o = dstOff + b * 32;
            for (int j = 0; j < 32; j++) {
                dst[o + j] = q8Embeddings[src + 2 + j] * scale;
            }
        }
    }

    public static EmbeddingTable fromGguf(GGUFLoader.GGUFFile gguf, int vocabSize, int dim) {
        GGUFLoader.Tensor t = gguf.tensors.get("token_embd.weight");
        if (t == null) {
            throw new NoSuchElementException("Required GGUF tensor missing: token_embd.weight");
        }
        if (t.type == 8) {
            // keep the quantised bytes (1/4 of the memory, no multi-second dequantise of the whole table at load)
            return new EmbeddingTable(t.loadQ8_0(), vocabSize, dim);
        }
        return new EmbeddingTable(loadAsFloat(t), vocabSize, dim);
    }

    /** GGML type 0 = F32 (read directly), type 8 = Q8_0 (dequantized on the CPU at load time). */
    private static float[] loadAsFloat(GGUFLoader.Tensor t) {
        return switch (t.type) {
            case 0 -> t.loadFloat();
            case 8 -> t.loadQ8_0AsFloat();
            default -> throw new IllegalArgumentException(
                    "Tensor '" + t.name + "' has GGML type " + t.type + " -- only F32 (0) and Q8_0 (8) are handled here.");
        };
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int dim() {
        return dim;
    }

    /** Uploads ONE token's embedding row directly into an already-allocated device buffer (e.g. the engine's decode-path [dim] scratch). xDevice must be a plain [dim]-sized buffer -- this always writes at offset 0. */
    public void embedRow(int tokenId, MemorySegment xDevice, GpuContext ctx) throws Throwable {
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("tokenId " + tokenId + " out of range [0," + vocabSize + ")");
        }
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) dim * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(byteSize);
            float[] row = new float[dim];
            rowInto(row, 0, tokenId);
            MemorySegment.copy(row, 0, host, ValueLayout.JAVA_FLOAT, 0, dim);
            GpuContext.check((int) ctx.cl.clEnqueueWriteBuffer.invoke(
                    ctx.queue, xDevice, OpenCLBindings.CL_TRUE, 0L, byteSize, host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(embedding row)");
        }
    }

    /** Uploads T tokens' embedding rows as one contiguous [T,dim] block into an already-allocated device buffer (e.g. the engine's prefill-path [maxBatchT,dim] scratch), in ONE copy rather than T separate ones. Always writes starting at offset 0. */
    public void embedRows(int[] tokenIds, MemorySegment xBatchDevice, GpuContext ctx) throws Throwable {
        int t = tokenIds.length;
        float[] gathered = new float[t * dim];
        for (int row = 0; row < t; row++) {
            int tokenId = tokenIds[row];
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException("tokenId " + tokenId + " out of range [0," + vocabSize + ") at row " + row);
            }
            rowInto(gathered, row * dim, tokenId);
        }
        try (Arena tmp = Arena.ofConfined()) {
            long byteSize = (long) gathered.length * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment host = tmp.allocate(byteSize);
            MemorySegment.copy(gathered, 0, host, ValueLayout.JAVA_FLOAT, 0, gathered.length);
            GpuContext.check((int) ctx.cl.clEnqueueWriteBuffer.invoke(
                    ctx.queue, xBatchDevice, OpenCLBindings.CL_TRUE, 0L, byteSize, host, 0, MemorySegment.NULL, MemorySegment.NULL),
                    "clEnqueueWriteBuffer(embedding rows)");
        }
    }
}