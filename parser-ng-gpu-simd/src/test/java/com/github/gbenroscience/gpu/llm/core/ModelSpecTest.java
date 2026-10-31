package com.github.gbenroscience.gpu.llm.core;

import static org.junit.jupiter.api.Assertions.*;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;
import java.io.File;
import java.nio.file.Files;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class ModelSpecTest {

    private static GGUFLoader.GGUFFile load(Consumer<TinyModel> tweak) throws Exception {
        TinyModel m = new TinyModel();
        m.build(1);
        if (tweak != null) {
            tweak.accept(m);
        }
        File f = Files.createTempFile("tiny", ".gguf").toFile();
        f.deleteOnExit();
        m.write(f);
        return GGUFLoader.load(f);
    }

    private static UnsupportedModelException rejected(Consumer<TinyModel> tweak) throws Exception {
        GGUFLoader.GGUFFile g = load(tweak);
        return assertThrows(UnsupportedModelException.class, () -> ModelSpec.fromGguf(g));
    }

    @Test
    void validModelIsAccepted() throws Exception {
        ModelSpec s = ModelSpec.fromGguf(load(null));
        assertEquals("llama", s.architecture);
        assertEquals(2, s.layers);
        assertEquals(64, s.dim);
        assertEquals(16, s.headDim);
        assertEquals(2, s.kvHeads);
        assertEquals(288, s.vocabSize);
        assertEquals(RopeTables.Scaling.NONE, s.rope.scaling);
        assertFalse(s.tiedOutput);
        assertEquals(2L * 64 * 32 * 4, s.kvCacheBytesPerLayer(64, 4));
    }

    @Test
    void tiedOutputIsRecognised() throws Exception {
        assertTrue(ModelSpec.fromGguf(load(m -> m.writer.tensors.remove("output.weight"))).tiedOutput);
    }

    @Test
    void rejectsOtherArchitectures() throws Exception {
        String msg = rejected(m -> m.writer.metadata.put("general.architecture", "qwen2")).getMessage();
        assertTrue(msg.contains("qwen2"), msg);
    }

    @Test
    void rejectsMissingRequiredMetadataInsteadOfDefaulting() throws Exception {
        for (String k : new String[]{"llama.block_count", "llama.embedding_length", "llama.attention.head_count",
            "llama.context_length", "llama.feed_forward_length", "llama.attention.layer_norm_rms_epsilon"}) {
            String msg = rejected(m -> m.writer.metadata.remove(k)).getMessage();
            assertTrue(msg.contains(k), msg);
        }
    }

    @Test
    void rejectsHeadGeometryThatBreaksAttentionBuffers() throws Exception {
        assertTrue(rejected(m -> m.writer.metadata.put("llama.attention.key_length", 32)).getMessage().contains("embedding_length"));
        assertTrue(rejected(m -> m.writer.metadata.put("llama.attention.head_count_kv", 3)).getMessage().contains("multiple"));
    }

    @Test
    void rejectsPartialRotaryAndUnknownScaling() throws Exception {
        assertTrue(rejected(m -> m.writer.metadata.put("llama.rope.dimension_count", 8)).getMessage().contains("partial rotary"));
        assertTrue(rejected(m -> m.writer.metadata.put("llama.rope.scaling.type", "yarn")).getMessage().contains("yarn"));
    }

    @Test
    void ropeScalingModes() throws Exception {
        ModelSpec lin = ModelSpec.fromGguf(load(m -> {
            m.writer.metadata.put("llama.rope.scaling.type", "linear");
            m.writer.metadata.put("llama.rope.scaling.factor", 4.0f);
        }));
        assertEquals(RopeTables.Scaling.LINEAR, lin.rope.scaling);
        assertEquals(4.0, lin.rope.factor);
        ModelSpec legacy = ModelSpec.fromGguf(load(m -> m.writer.metadata.put("llama.rope.scaling.factor", 8.0f)));
        assertEquals(RopeTables.Scaling.LLAMA3, legacy.rope.scaling);
        assertFalse(legacy.warnings.isEmpty(), "the legacy assumption must be surfaced");
    }

    @Test
    void rejectsUnsupportedTensorTypesByName() throws Exception {
        for (int type : new int[]{1, 2}) { // F16, Q4_0: sizes known to the loader
            String msg = rejected(m -> {
                GgufTestWriter.Tensor t = m.writer.tensors.get("blk.0.attn_q.weight");
                int n = t.dims[0] * t.dims[1];
                int bytes = type == 1 ? n * 2 : n / 32 * 18;
                m.writer.tensors.put(t.name, new GgufTestWriter.Tensor(t.name, t.dims, type, new byte[bytes]));
            }).getMessage();
            assertTrue(msg.contains("blk.0.attn_q.weight") && msg.contains(ModelSpec.ggmlTypeName(type)), msg);
        }
    }

    @Test
    void rejectsWrongShapesMissingTensorsBiasesAndVocabMismatch() throws Exception {
        assertTrue(rejected(m -> {
            GgufTestWriter.Tensor t = m.writer.tensors.get("blk.1.attn_k.weight");
            m.writer.tensors.put(t.name, new GgufTestWriter.Tensor(t.name, new int[]{t.dims[0], t.dims[1] * 2},
                    t.type, new byte[t.data.length * 2]));
        }).getMessage().contains("blk.1.attn_k.weight"));
        assertTrue(rejected(m -> m.writer.tensors.remove("blk.0.ffn_down.weight")).getMessage().contains("ffn_down"));
        assertTrue(rejected(m -> m.writer.tensors.put("blk.0.attn_q.bias",
                new GgufTestWriter.Tensor("blk.0.attn_q.bias", new int[]{64}, 0, new byte[256]))).getMessage().contains("bias"));
        assertTrue(rejected(m -> {
            java.util.List<String> t = m.tokens();
            t.remove(t.size() - 1);
            m.writer.metadata.put("tokenizer.ggml.tokens", t);
        }).getMessage().contains("tokenizer.ggml.tokens"));
        assertTrue(rejected(m -> {
            GgufTestWriter.Tensor t = m.writer.tensors.get("output.weight");
            m.writer.tensors.put(t.name, new GgufTestWriter.Tensor(t.name, new int[]{t.dims[0], t.dims[1] - 1}, t.type,
                    new byte[t.data.length - t.dims[0] * 4]));
        }).getMessage().contains("output.weight"));
    }

    @Test
    void readsRopeFreqFactorsTensor() throws Exception {
        ModelSpec s = ModelSpec.fromGguf(load(m -> {
            float[] f = new float[8];
            java.util.Arrays.fill(f, 2f);
            m.writer.tensors.put("rope_freqs.weight", new GgufTestWriter.Tensor("rope_freqs.weight", new int[]{8}, 0, GgufTestWriter.f32Bytes(f)));
        }));
        assertNotNull(s.rope.freqFactors);
        assertEquals(8, s.rope.freqFactors.length);
        assertEquals(2f, s.rope.freqFactors[0]);
    }
}
