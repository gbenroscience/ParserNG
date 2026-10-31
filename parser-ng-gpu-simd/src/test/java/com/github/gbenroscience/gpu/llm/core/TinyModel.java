package com.github.gbenroscience.gpu.llm.core;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * A small random Llama-architecture model with a real byte-level BPE
 * vocabulary (256 byte tokens + fillers, no merges), as GGUF. Weights are
 * kept in F32 ("truth") and quantized to Q8_0 for the file exactly like
 * llama.cpp does for linear layers; wo, norms, embeddings and output stay F32.
 */
public final class TinyModel {

    public int layers = 2;
    public int dim = 64;
    public int hidden = 96;
    public int heads = 4;
    public int kvHeads = 2;
    public int ctx = 64;
    public int vocab = 288; // 256 byte tokens + 32 fillers
    public double eps = 1e-5;
    public double theta = 10000.0;
    public boolean tiedOutput = false;
    public float weightScale = 0.08f;

    public final GgufTestWriter writer = new GgufTestWriter();
    /** F32 originals by tensor name (before any quantization). */
    public final Map<String, float[]> truth = new LinkedHashMap<>();

    public int headDim() {
        return dim / heads;
    }

    public TinyModel build(long seed) {
        Random rnd = new Random(seed);
        GgufTestWriter w = writer;
        w.metadata.put("general.architecture", "llama");
        w.metadata.put("general.alignment", 32);
        w.metadata.put("llama.block_count", layers);
        w.metadata.put("llama.embedding_length", dim);
        w.metadata.put("llama.feed_forward_length", hidden);
        w.metadata.put("llama.attention.head_count", heads);
        w.metadata.put("llama.attention.head_count_kv", kvHeads);
        w.metadata.put("llama.context_length", ctx);
        w.metadata.put("llama.attention.layer_norm_rms_epsilon", (float) eps);
        w.metadata.put("llama.rope.freq_base", (float) theta);
        w.metadata.put("tokenizer.ggml.model", "gpt2");
        w.metadata.put("tokenizer.ggml.tokens", tokens());
        w.metadata.put("tokenizer.ggml.merges", new ArrayList<String>());
        w.metadata.put("tokenizer.ggml.bos_token_id", 256);
        w.metadata.put("tokenizer.ggml.eos_token_id", 257);

        f32("token_embd.weight", new int[]{dim, vocab}, rnd, 0.5f);
        f32("output_norm.weight", new int[]{dim}, rnd, 1f, true);
        if (!tiedOutput) {
            f32("output.weight", new int[]{dim, vocab}, rnd, 0.5f);
        }
        int qd = heads * headDim();
        int kd = kvHeads * headDim();
        for (int l = 0; l < layers; l++) {
            String p = "blk." + l + ".";
            f32(p + "attn_norm.weight", new int[]{dim}, rnd, 1f, true);
            f32(p + "ffn_norm.weight", new int[]{dim}, rnd, 1f, true);
            q8(p + "attn_q.weight", new int[]{dim, qd}, rnd);
            q8(p + "attn_k.weight", new int[]{dim, kd}, rnd);
            q8(p + "attn_v.weight", new int[]{dim, kd}, rnd);
            f32(p + "attn_output.weight", new int[]{qd, dim}, rnd, weightScale);
            q8(p + "ffn_gate.weight", new int[]{dim, hidden}, rnd);
            q8(p + "ffn_up.weight", new int[]{dim, hidden}, rnd);
            q8(p + "ffn_down.weight", new int[]{hidden, dim}, rnd);
        }
        return this;
    }

    public File write(File f) throws IOException {
        return writer.write(f);
    }

    private static int count(int[] dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        return n;
    }

    private void f32(String name, int[] dims, Random rnd, float scale) {
        f32(name, dims, rnd, scale, false);
    }

    private void f32(String name, int[] dims, Random rnd, float scale, boolean aroundOne) {
        float[] v = new float[count(dims)];
        for (int i = 0; i < v.length; i++) {
            v[i] = aroundOne ? 1f + 0.1f * (float) rnd.nextGaussian() : scale * (float) rnd.nextGaussian();
        }
        truth.put(name, v);
        writer.tensors.put(name, new GgufTestWriter.Tensor(name, dims, ModelSpec.GGML_F32, GgufTestWriter.f32Bytes(v)));
    }

    private void q8(String name, int[] dims, Random rnd) {
        float[] v = new float[count(dims)];
        for (int i = 0; i < v.length; i++) {
            v[i] = weightScale * (float) rnd.nextGaussian();
        }
        truth.put(name, v);
        writer.tensors.put(name, new GgufTestWriter.Tensor(name, dims, ModelSpec.GGML_Q8_0, CpuReference.quantizeQ8_0(v)));
    }

    /** GPT-2 byte-to-unicode table, reimplemented independently of BpeTokenizer. */
    public static char[] byteToUnicode() {
        Set<Integer> keep = new HashSet<>();
        for (int i = '!'; i <= '~'; i++) {
            keep.add(i);
        }
        for (int i = 0xA1; i <= 0xAC; i++) {
            keep.add(i);
        }
        for (int i = 0xAE; i <= 0xFF; i++) {
            keep.add(i);
        }
        char[] map = new char[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            map[b] = keep.contains(b) ? (char) b : (char) (256 + n++);
        }
        return map;
    }

    public List<String> tokens() {
        char[] m = byteToUnicode();
        List<String> t = new ArrayList<>();
        for (int b = 0; b < 256; b++) {
            t.add(String.valueOf(m[b]));
        }
        t.add("<s>");
        t.add("</s>");
        while (t.size() < vocab) {
            t.add("<extra" + t.size() + ">");
        }
        return t;
    }
}
