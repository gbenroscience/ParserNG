package com.github.gbenroscience.gpu.llm.core;

import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Validated, backend-independent description of a model, built from GGUF
 * metadata AND tensor headers. Construction either returns a spec the engines
 * can execute correctly or throws {@link UnsupportedModelException} naming the
 * exact reason -- before a single byte is uploaded to a device.
 *
 * <h3>What is supported</h3>
 * Architecture {@code llama} (this is also the GGUF architecture of Mistral-7B
 * style dense models): RMSNorm, GQA attention with interleaved-pair RoPE, SwiGLU
 * FFN, no biases, no MoE. Linear weights (q/k/v/gate/up/down) must be Q8_0;
 * output projection, norms and embeddings may be F32 or Q8_0.
 *
 * <h3>What is rejected (non-exhaustive)</h3>
 * other architectures (qwen2, gemma*, phi3, ... use half-split RoPE, biases,
 * different activations or norm conventions), MoE, partial rotary,
 * {@code heads*headDim != dim}, yarn/unknown RoPE scaling, any other tensor type
 * (K-quants, Q4_x, Q5_x, F16, BF16 -- the dequantizing GPU paths for these do
 * not exist yet and are never approximated by another format), tensors whose
 * shape or byte length disagree with the metadata.
 */
public final class ModelSpec {

    public static final int GGML_F32 = 0;
    public static final int GGML_Q8_0 = 8;

    public enum Activation {
        SWIGLU
    }

    public final String architecture;
    public final int layers;
    public final int dim;
    public final int hiddenDim;
    public final int heads;
    public final int kvHeads;
    public final int headDim;
    public final int contextLength;
    public final int vocabSize;
    public final double normEps;
    public final double ropeTheta;
    public final RopeTables.Params rope;
    public final Activation activation;
    public final boolean tiedOutput;
    /** Assumptions the validator made that the user should know about. */
    public final List<String> warnings;

    private ModelSpec(String architecture, int layers, int dim, int hiddenDim, int heads, int kvHeads, int headDim,
            int contextLength, int vocabSize, double normEps, double ropeTheta, RopeTables.Params rope,
            boolean tiedOutput, List<String> warnings) {
        this.architecture = architecture;
        this.layers = layers;
        this.dim = dim;
        this.hiddenDim = hiddenDim;
        this.heads = heads;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.contextLength = contextLength;
        this.vocabSize = vocabSize;
        this.normEps = normEps;
        this.ropeTheta = ropeTheta;
        this.rope = rope;
        this.activation = Activation.SWIGLU;
        this.tiedOutput = tiedOutput;
        this.warnings = Collections.unmodifiableList(warnings);
    }

    public int qDim() {
        return heads * headDim;
    }

    public int kvDim() {
        return kvHeads * headDim;
    }

    /** Bytes of K+V cache for one layer at the given context length and element size. */
    public long kvCacheBytesPerLayer(int ctx, int bytesPerElement) {
        return 2L * ctx * kvDim() * bytesPerElement;
    }

    public static String ggmlTypeName(int t) {
        switch (t) {
            case 0: return "F32";
            case 1: return "F16";
            case 2: return "Q4_0";
            case 3: return "Q4_1";
            case 6: return "Q5_0";
            case 7: return "Q5_1";
            case 8: return "Q8_0";
            case 9: return "Q8_1";
            case 10: return "Q2_K";
            case 11: return "Q3_K";
            case 12: return "Q4_K";
            case 13: return "Q5_K";
            case 14: return "Q6_K";
            case 15: return "Q8_K";
            case 30: return "BF16";
            default: return "type#" + t;
        }
    }

    // ------------------------------------------------------------------ build

    public static ModelSpec fromGguf(GGUFLoader.GGUFFile g) {
        Map<String, Object> md = g.metadata;
        List<String> warnings = new ArrayList<>();

        Object archObj = md.get("general.architecture");
        if (archObj == null) {
            throw new UnsupportedModelException("GGUF has no general.architecture metadata; refusing to guess the model type");
        }
        String arch = String.valueOf(archObj);
        if (!"llama".equals(arch)) {
            throw new UnsupportedModelException("Unsupported architecture '" + arch + "'. Supported: llama (dense Llama/Mistral-style). "
                    + "Other architectures differ in RoPE pairing, biases, activations or normalisation and would silently produce wrong output.");
        }
        if (md.containsKey(arch + ".expert_count") && number(md.get(arch + ".expert_count")) > 0) {
            throw new UnsupportedModelException("Mixture-of-experts models are not supported (" + arch + ".expert_count="
                    + md.get(arch + ".expert_count") + ")");
        }

        int layers = reqInt(md, arch + ".block_count");
        int dim = reqInt(md, arch + ".embedding_length");
        int hidden = reqInt(md, arch + ".feed_forward_length");
        int heads = reqInt(md, arch + ".attention.head_count");
        int kvHeads = md.containsKey(arch + ".attention.head_count_kv") ? reqInt(md, arch + ".attention.head_count_kv") : heads;
        int ctx = reqInt(md, arch + ".context_length");
        double eps = reqDouble(md, arch + ".attention.layer_norm_rms_epsilon");
        double theta = md.containsKey(arch + ".rope.freq_base") ? reqDouble(md, arch + ".rope.freq_base") : 10000.0;
        if (!md.containsKey(arch + ".rope.freq_base")) {
            warnings.add("rope.freq_base missing; using 10000.0");
        }

        require(layers > 0, "block_count must be > 0, got " + layers);
        require(heads > 0 && kvHeads > 0, "head counts must be > 0 (heads=" + heads + ", kvHeads=" + kvHeads + ")");
        require(heads % kvHeads == 0, "head_count (" + heads + ") must be a multiple of head_count_kv (" + kvHeads + ")");
        require(ctx > 0, "context_length must be > 0, got " + ctx);
        require(eps > 0 && !Double.isInfinite(eps), "layer_norm_rms_epsilon must be a positive finite number, got " + eps);
        require(theta > 0 && !Double.isInfinite(theta), "rope.freq_base must be positive and finite, got " + theta);

        int headDim;
        if (md.containsKey(arch + ".attention.key_length")) {
            headDim = reqInt(md, arch + ".attention.key_length");
            if (md.containsKey(arch + ".attention.value_length") && reqInt(md, arch + ".attention.value_length") != headDim) {
                throw new UnsupportedModelException("key_length != value_length is not supported");
            }
        } else {
            require(dim % heads == 0, "embedding_length (" + dim + ") is not divisible by head_count (" + heads + ") and no key_length given");
            headDim = dim / heads;
        }
        require(headDim > 0 && headDim % 2 == 0, "head dimension must be positive and even, got " + headDim);
        if (heads * headDim != dim) {
            throw new UnsupportedModelException("head_count*head_dim (" + heads + "*" + headDim + "=" + heads * headDim
                    + ") != embedding_length (" + dim + "). The attention-output buffers and wo projection assume they are equal.");
        }
        require(dim % 32 == 0, "embedding_length must be a multiple of 32 (Q8_0 block), got " + dim);
        require(hidden > 0 && hidden % 32 == 0, "feed_forward_length must be a positive multiple of 32 (Q8_0 block), got " + hidden);

        if (md.containsKey(arch + ".rope.dimension_count") && reqInt(md, arch + ".rope.dimension_count") != headDim) {
            throw new UnsupportedModelException("rope.dimension_count (" + md.get(arch + ".rope.dimension_count")
                    + ") != head_dim (" + headDim + "): partial rotary embeddings are not supported");
        }
        if (md.containsKey(arch + ".rope.scaling.finetuned") && Boolean.TRUE.equals(md.get(arch + ".rope.scaling.finetuned"))) {
            warnings.add("rope.scaling.finetuned=true is ignored by this runtime");
        }

        RopeTables.Params rope = parseRopeScaling(md, arch, warnings);

        // ------------------------------------------------------------ tensors
        Map<String, GGUFLoader.Tensor> t = g.tensors;
        GGUFLoader.Tensor embd = need(t, "token_embd.weight");
        int vocab = checkMatrix(embd, dim, -1, "token_embd.weight", GGML_F32, GGML_Q8_0);
        Object tokens = md.get("tokenizer.ggml.tokens");
        if (tokens instanceof List && ((List<?>) tokens).size() != vocab) {
            throw new UnsupportedModelException("tokenizer.ggml.tokens has " + ((List<?>) tokens).size()
                    + " entries but token_embd.weight has " + vocab + " rows");
        }
        boolean tied = !t.containsKey("output.weight");
        if (!tied) {
            checkMatrix(t.get("output.weight"), dim, vocab, "output.weight", GGML_F32, GGML_Q8_0);
        }
        checkVector(need(t, "output_norm.weight"), dim, "output_norm.weight");

        for (int l = 0; l < layers; l++) {
            String p = "blk." + l + ".";
            checkMatrix(need(t, p + "attn_q.weight"), dim, heads * headDim, p + "attn_q.weight", GGML_Q8_0);
            checkMatrix(need(t, p + "attn_k.weight"), dim, kvHeads * headDim, p + "attn_k.weight", GGML_Q8_0);
            checkMatrix(need(t, p + "attn_v.weight"), dim, kvHeads * headDim, p + "attn_v.weight", GGML_Q8_0);
            checkMatrix(need(t, p + "attn_output.weight"), heads * headDim, dim, p + "attn_output.weight", GGML_F32, GGML_Q8_0);
            checkMatrix(need(t, p + "ffn_gate.weight"), dim, hidden, p + "ffn_gate.weight", GGML_Q8_0);
            checkMatrix(need(t, p + "ffn_up.weight"), dim, hidden, p + "ffn_up.weight", GGML_Q8_0);
            checkMatrix(need(t, p + "ffn_down.weight"), hidden, dim, p + "ffn_down.weight", GGML_Q8_0);
            checkVector(need(t, p + "attn_norm.weight"), dim, p + "attn_norm.weight");
            checkVector(need(t, p + "ffn_norm.weight"), dim, p + "ffn_norm.weight");
            for (String bias : new String[]{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
                "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"}) {
                if (t.containsKey(p + bias)) {
                    throw new UnsupportedModelException("Tensor " + p + bias + " exists but linear biases are not applied by this runtime");
                }
            }
        }
        if (t.containsKey("rope_freqs.weight")) {
            GGUFLoader.Tensor rf = t.get("rope_freqs.weight");
            checkVector(rf, headDim / 2, "rope_freqs.weight");
            if (rf.type != GGML_F32) {
                throw new UnsupportedModelException("rope_freqs.weight must be F32, found " + ggmlTypeName(rf.type));
            }
            if (rope.scaling == RopeTables.Scaling.LLAMA3) {
                throw new UnsupportedModelException("Both rope_freqs.weight and Llama-3 scaling metadata are present; combining them is ambiguous");
            }
            rope.freqFactors = rf.loadFloat();
        }

        return new ModelSpec(arch, layers, dim, hidden, heads, kvHeads, headDim, ctx, vocab, eps, theta, rope, tied, warnings);
    }

    private static RopeTables.Params parseRopeScaling(Map<String, Object> md, String arch, List<String> warnings) {
        RopeTables.Params p = new RopeTables.Params();
        Object typeObj = md.get(arch + ".rope.scaling.type");
        String type = typeObj == null ? null : String.valueOf(typeObj).toLowerCase();
        boolean hasFactor = md.containsKey(arch + ".rope.scaling.factor");
        double factor = hasFactor ? reqDouble(md, arch + ".rope.scaling.factor") : 1.0;

        if (type == null || type.equals("none")) {
            if (hasFactor && factor != 1.0) {
                // Legacy behaviour of this runtime: factor without a type was treated as Llama-3 NTK-by-parts.
                p.scaling = RopeTables.Scaling.LLAMA3;
                p.factor = factor;
                warnings.add("rope.scaling.factor=" + factor + " present without rope.scaling.type: assuming Llama-3 NTK-by-parts scaling");
            }
        } else if (type.equals("linear")) {
            require(factor > 0, "rope.scaling.factor must be > 0, got " + factor);
            p.scaling = RopeTables.Scaling.LINEAR;
            p.factor = factor;
        } else if (type.equals("llama3")) {
            require(factor > 0, "rope.scaling.factor must be > 0, got " + factor);
            p.scaling = RopeTables.Scaling.LLAMA3;
            p.factor = factor;
        } else {
            throw new UnsupportedModelException("Unsupported rope.scaling.type '" + typeObj + "' (supported: none, linear, llama3)");
        }
        if (p.scaling == RopeTables.Scaling.LLAMA3) {
            p.lowFreqFactor = md.containsKey(arch + ".rope.scaling.low_freq_factor") ? reqDouble(md, arch + ".rope.scaling.low_freq_factor") : 1.0;
            p.highFreqFactor = md.containsKey(arch + ".rope.scaling.high_freq_factor") ? reqDouble(md, arch + ".rope.scaling.high_freq_factor") : 4.0;
            p.origContextLength = md.containsKey(arch + ".rope.scaling.original_context_length")
                    ? reqDouble(md, arch + ".rope.scaling.original_context_length") : 8192.0;
            require(p.highFreqFactor != p.lowFreqFactor, "rope low/high frequency factors must differ");
        }
        return p;
    }

    // ---------------------------------------------------------------- helpers

    private static GGUFLoader.Tensor need(Map<String, GGUFLoader.Tensor> t, String name) {
        GGUFLoader.Tensor x = t.get(name);
        if (x == null) {
            throw new UnsupportedModelException("Required GGUF tensor missing: " + name);
        }
        return x;
    }

    /** Validates a 2-D tensor [cols=in, rows=out]; rows may be -1 (any) and is returned. */
    private static int checkMatrix(GGUFLoader.Tensor x, int cols, int rows, String name, int... allowedTypes) {
        checkType(x, name, allowedTypes);
        if (x.dims.length != 2) {
            throw new UnsupportedModelException("Tensor " + name + " must be 2-D, has " + x.dims.length + " dims");
        }
        if (x.dims[0] != cols || (rows >= 0 && x.dims[1] != rows)) {
            throw new UnsupportedModelException("Tensor " + name + " has dims [" + x.dims[0] + ", " + x.dims[1]
                    + "], expected [" + cols + ", " + (rows >= 0 ? String.valueOf(rows) : "*") + "]");
        }
        if (x.type == GGML_Q8_0 && x.dims[0] % 32 != 0) {
            throw new UnsupportedModelException("Tensor " + name + " row length " + x.dims[0] + " is not a multiple of the Q8_0 block size 32");
        }
        checkBytes(x, name);
        return x.dims[1];
    }

    private static void checkVector(GGUFLoader.Tensor x, int len, String name) {
        checkType(x, name, GGML_F32, GGML_Q8_0);
        if (x.dims.length != 1 || x.dims[0] != len) {
            throw new UnsupportedModelException("Tensor " + name + " has dims " + java.util.Arrays.toString(x.dims) + ", expected [" + len + "]");
        }
        checkBytes(x, name);
    }

    private static void checkType(GGUFLoader.Tensor x, String name, int... allowed) {
        for (int a : allowed) {
            if (x.type == a) {
                return;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int a : allowed) {
            sb.append(sb.length() == 0 ? "" : "/").append(ggmlTypeName(a));
        }
        throw new UnsupportedModelException("Tensor " + name + " is " + ggmlTypeName(x.type) + " (GGML type " + x.type
                + "); this runtime only supports " + sb + " for it. Re-quantize the model to Q8_0 (e.g. llama-quantize ... Q8_0).");
    }

    private static void checkBytes(GGUFLoader.Tensor x, String name) {
        long n = x.elements();
        long expected = (x.type == GGML_F32) ? n * 4 : (n / 32) * 34;
        if (x.nbytes != expected) {
            throw new UnsupportedModelException("Tensor " + name + " byte length " + x.nbytes + " != expected " + expected
                    + " for " + ggmlTypeName(x.type) + " with " + n + " elements");
        }
    }

    private static void require(boolean cond, String msg) {
        if (!cond) {
            throw new UnsupportedModelException(msg);
        }
    }

    private static double number(Object o) {
        if (o instanceof Number) {
            return ((Number) o).doubleValue();
        }
        throw new UnsupportedModelException("expected a numeric metadata value but found: " + o);
    }

    private static int reqInt(Map<String, Object> md, String key) {
        Object o = md.get(key);
        if (!(o instanceof Number)) {
            throw new UnsupportedModelException("Required GGUF metadata key missing or not numeric: " + key);
        }
        long v = ((Number) o).longValue();
        if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
            throw new UnsupportedModelException("Metadata value out of int range for " + key + ": " + v);
        }
        return (int) v;
    }

    private static double reqDouble(Map<String, Object> md, String key) {
        Object o = md.get(key);
        if (!(o instanceof Number)) {
            throw new UnsupportedModelException("Required GGUF metadata key missing or not numeric: " + key);
        }
        return ((Number) o).doubleValue();
    }
}
