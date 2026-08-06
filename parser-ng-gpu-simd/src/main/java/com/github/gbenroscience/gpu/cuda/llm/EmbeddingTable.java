/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.gpu.cuda.llm;

/**
 *
 * @author GBEMIRO
 */
  

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.NoSuchElementException;

/**
 * Token embedding lookup table -- kept HOST-resident (a flat float[] of
 * vocabSize*dim), not device-resident. Rationale: this table is read-only
 * and read exactly once per token (one row gather), never computed on;
 * uploading the single relevant row per call (dim floats -- e.g. 4096
 * floats = 16KB for a 4096-dim model) is trivially cheap next to the
 * GEMV/GEMM cost of the layer that follows, so there's no benefit to
 * keeping the full table (which can be hundreds of MB for a large vocab)
 * resident on the GPU. This mirrors the tradeoff GpuLlamaLayerCuda's
 * writeIntoCache already makes for a similar reason (small, infrequent,
 * host-mediated beats a bigger always-resident device allocation with
 * device-side gather logic).
 *
 * Uses GGUFLoader.Tensor's type field to pick the right read path at load
 * time -- loadFloat() for GGML type 0 (F32), loadQ8_0AsFloat() for type 8
 * (Q8_0, dequantized on the CPU once at load time since this table is
 * host-resident anyway) -- rather than assuming one or the other. Real
 * GGUF files vary here: some keep the embedding table (and norm weights)
 * F32, others quantize everything uniformly (this codebase's own
 * ModelLoader.loadLlamaWeights calls .loadQ8_0AsFloat() on every tensor
 * it touches, norms included, implying Q8_0 throughout for at least some
 * of this project's model files) -- auto-detecting avoids hardcoding
 * either convention and throwing against the other.
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

    /** GGML type 0 = F32 (read directly), type 8 = Q8_0 (dequantized on the CPU at load time). Other types aren't handled by GGUFLoader.Tensor's own read methods either. */
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

    /** Uploads ONE token's embedding row directly into an already-allocated device buffer (e.g. the engine's decode-path [dim] scratch). */
    public void embedRow(int tokenId, long xDevice, GpuContext ctx) throws Throwable {
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw new IllegalArgumentException("tokenId " + tokenId + " out of range [0," + vocabSize + ")");
        }
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) dim * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(flatEmbeddings, tokenId * dim, host, ValueLayout.JAVA_FLOAT, 0, dim);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(xDevice, host, host.byteSize()),
                    "cuMemcpyHtoD(embedding row)");
        }
    }

    /** Uploads T tokens' embedding rows as one contiguous [T,dim] block into an already-allocated device buffer (e.g. the engine's prefill-path [maxBatchT,dim] scratch), in ONE copy rather than T separate ones. */
    public void embedRows(int[] tokenIds, long xBatchDevice, GpuContext ctx) throws Throwable {
        int t = tokenIds.length;
        float[] gathered = new float[t * dim];
        for (int row = 0; row < t; row++) {
            int tokenId = tokenIds[row];
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException("tokenId " + tokenId + " out of range [0," + vocabSize + ") at row " + row);
            }
            System.arraycopy(flatEmbeddings, tokenId * dim, gathered, row * dim, dim);
        }
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate((long) gathered.length * ValueLayout.JAVA_FLOAT.byteSize());
            MemorySegment.copy(gathered, 0, host, ValueLayout.JAVA_FLOAT, 0, gathered.length);
            GpuContext.check((int) ctx.cu.cuMemcpyHtoD.invoke(xBatchDevice, host, host.byteSize()),
                    "cuMemcpyHtoD(embedding rows)");
        }
    }
}