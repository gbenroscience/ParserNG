package com.github.gbenroscience.gpu.llm.metal;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * Metal counterpart of {@code com.github.gbenroscience.gpu.llm.cuda.EmbeddingTable}.
 * Same design: HOST-resident flat {@code float[vocabSize*dim]} table, one
 * row gathered per token rather than the whole table kept device-resident
 * -- see the CUDA version's class javadoc for the full rationale (a
 * read-once-per-token row lookup is cheap next to the GEMV/GEMM cost of
 * the layer that follows it, so there's no benefit to residency).
 *
 * TRANSLATION NOTE: {@link #embedRow}/{@link #embedRows} write directly
 * into the destination {@link MetalBuffer}'s host-visible
 * {@link MetalBuffer#contents} memory via {@code MemorySegment.copy} --
 * no {@code cuMemcpyHtoD}-equivalent call exists or is needed, because
 * every buffer this port allocates is {@code MTLResourceStorageModeShared}
 * (see {@link MetalBindings}'s class javadoc). This makes both methods
 * here noticeably shorter than their CUDA counterparts, which had to
 * stage through a temporary host arena and issue a real driver copy.
 */
public final class EmbeddingTable {

    private final float[] flatEmbeddings; // [vocabSize * dim], row-major
    private final int vocabSize;
    private final int dim;

    public EmbeddingTable(float[] flatEmbeddings, int vocabSize, int dim) {
        if ((long) flatEmbeddings.length != (long) vocabSize * dim) {
            throw new IllegalArgumentException(
                    "flatEmbeddings.length (" + flatEmbeddings.length + ") != vocabSize*dim ("
                            + vocabSize + "*" + dim + "=" + ((long) vocabSize * dim) + ")");
        }
        this.flatEmbeddings = flatEmbeddings;
        this.vocabSize = vocabSize;
        this.dim = dim;
    }

    public static EmbeddingTable fromGguf(GGUFLoader.GGUFFile gguf, int vocabSize, int dim) {
        GGUFLoader.Tensor t = gguf.tensors.get("token_embd.weight");
        if (t == null) {
            throw new NoSuchElementException("Required GGUF tensor missing: token_embd.weight");
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

    /** Writes ONE token's embedding row directly into an already-allocated device buffer (e.g. the engine's decode-path [dim] scratch). Plain host memcpy -- see class javadoc. */
    public void embedRow(int tokenId, MetalBuffer xDevice, GpuContext ctx) {
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("tokenId " + tokenId + " out of range [0," + vocabSize + ")");
        }
        long byteLen = (long) dim * ValueLayout.JAVA_FLOAT.byteSize();
        MemorySegment dst = MemorySegment.ofAddress(xDevice.hostAddress()).reinterpret(byteLen);
        MemorySegment.copy(flatEmbeddings, tokenId * dim, dst, ValueLayout.JAVA_FLOAT, 0, dim);
    }

    /** Writes T tokens' embedding rows as one contiguous [T,dim] block into an already-allocated device buffer (e.g. the engine's prefill-path [maxBatchT,dim] scratch), gathered host-side then copied in ONE call. */
    public void embedRows(int[] tokenIds, MetalBuffer xBatchDevice, GpuContext ctx) {
        int t = tokenIds.length;
        float[] gathered = new float[t * dim];
        for (int row = 0; row < t; row++) {
            int tokenId = tokenIds[row];
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException("tokenId " + tokenId + " out of range [0," + vocabSize + ") at row " + row);
            }
            System.arraycopy(flatEmbeddings, tokenId * dim, gathered, row * dim, dim);
        }
        long byteLen = (long) gathered.length * ValueLayout.JAVA_FLOAT.byteSize();
        MemorySegment dst = MemorySegment.ofAddress(xBatchDevice.hostAddress()).reinterpret(byteLen);
        MemorySegment.copy(gathered, 0, dst, ValueLayout.JAVA_FLOAT, 0, gathered.length);
    }
}