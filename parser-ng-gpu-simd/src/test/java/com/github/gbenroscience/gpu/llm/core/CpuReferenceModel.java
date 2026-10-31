package com.github.gbenroscience.gpu.llm.core;

import java.util.Map;

/**
 * Complete decode-step Llama forward pass built ONLY from {@link CpuReference}
 * primitives and the tensors that were written to the GGUF file. It mirrors the
 * numerical contract of the device engines (Q8_0 activations/weights with fp16
 * scales for q/k/v/gate/up/down, F32 for wo and the output head, double-precision
 * norms/softmax) but shares no code with any kernel or backend.
 */
public final class CpuReferenceModel {

    private final ModelSpec spec;
    private final Map<String, float[]> f32;
    private final Map<String, byte[]> q8;
    private final float[][] kCache;
    private final float[][] vCache;

    public CpuReferenceModel(ModelSpec spec, TinyModel m) {
        this.spec = spec;
        this.f32 = m.truth;
        this.q8 = new java.util.HashMap<>();
        for (GgufTestWriter.Tensor t : m.writer.tensors.values()) {
            if (t.type == ModelSpec.GGML_Q8_0) {
                q8.put(t.name, t.data);
            }
        }
        this.kCache = new float[spec.layers][spec.contextLength * spec.kvDim()];
        this.vCache = new float[spec.layers][spec.contextLength * spec.kvDim()];
    }

    private static float[] add(float[] a, float[] b) {
        float[] o = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            o[i] = a[i] + b[i];
        }
        return o;
    }

    /** Returns the logits after feeding {@code token} at absolute position {@code pos}. */
    public float[] forward(int token, int pos) {
        int dim = spec.dim, hd = spec.headDim, qd = spec.qDim(), kd = spec.kvDim(), hidden = spec.hiddenDim;
        float[] x = java.util.Arrays.copyOfRange(f32.get("token_embd.weight"), token * dim, (token + 1) * dim);
        for (int l = 0; l < spec.layers; l++) {
            String p = "blk." + l + ".";
            float[] xn = CpuReference.rmsNorm(x, f32.get(p + "attn_norm.weight"), spec.normEps);
            byte[] xq = CpuReference.quantizeQ8_0(xn);
            float[] q = CpuReference.q8Gemv(xq, q8.get(p + "attn_q.weight"), qd, dim);
            float[] k = CpuReference.q8Gemv(xq, q8.get(p + "attn_k.weight"), kd, dim);
            float[] v = CpuReference.q8Gemv(xq, q8.get(p + "attn_v.weight"), kd, dim);
            q = CpuReference.ropeInterleaved(q, spec.heads, hd, hd, pos, spec.ropeTheta, spec.rope);
            k = CpuReference.ropeInterleaved(k, spec.kvHeads, hd, hd, pos, spec.ropeTheta, spec.rope);
            System.arraycopy(k, 0, kCache[l], pos * kd, kd);
            System.arraycopy(v, 0, vCache[l], pos * kd, kd);
            float[] a = CpuReference.attentionDecode(q, kCache[l], vCache[l], spec.heads, spec.kvHeads, hd, pos + 1);
            x = add(x, CpuReference.matvecExact(f32.get(p + "attn_output.weight"), a, dim, qd));

            float[] xn2 = CpuReference.rmsNorm(x, f32.get(p + "ffn_norm.weight"), spec.normEps);
            byte[] xq2 = CpuReference.quantizeQ8_0(xn2);
            float[] gate = CpuReference.q8Gemv(xq2, q8.get(p + "ffn_gate.weight"), hidden, dim);
            float[] up = CpuReference.q8Gemv(xq2, q8.get(p + "ffn_up.weight"), hidden, dim);
            byte[] aq = CpuReference.quantizeQ8_0(CpuReference.swiglu(gate, up));
            x = add(x, CpuReference.q8Gemv(aq, q8.get(p + "ffn_down.weight"), dim, hidden));
        }
        float[] xn = CpuReference.rmsNorm(x, f32.get("output_norm.weight"), spec.normEps);
        float[] head = f32.containsKey("output.weight") ? f32.get("output.weight") : f32.get("token_embd.weight");
        return CpuReference.matvecExact(head, xn, spec.vocabSize, dim);
    }
}
