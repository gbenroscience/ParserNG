package com.github.gbenroscience.simd.turbo.tools.llm.loader;

import com.github.gbenroscience.simd.turbo.tools.llm.LlamaLayer;
import java.io.*;
import java.util.NoSuchElementException;

/**
 *
 * @author GBEMIRO
 */
public class ModelLoader {
    public static LlamaLayer.Weights[] loadLlamaWeights(File ggufPath, LlamaLayer.Config cfg) throws IOException {
        GGUFLoader.GGUFFile gguf = GGUFLoader.load(ggufPath);
        LlamaLayer.Weights[] layers = new LlamaLayer.Weights[cfg.n_layers];
        
        for (int i = 0; i < cfg.n_layers; i++) {
            String prefix = "blk." + i + ".";
            
            // Core Weights (Usually Quantized, e.g., Q8_0 or Q4_0)
            GGUFLoader.Tensor qWeight = find(gguf, prefix + "attn_q.weight");
            GGUFLoader.Tensor kWeight = find(gguf, prefix + "attn_k.weight");
            GGUFLoader.Tensor vWeight = find(gguf, prefix + "attn_v.weight");
            GGUFLoader.Tensor oWeight = find(gguf, prefix + "attn_output.weight");
            
            // Feed-Forward Networks SwiGLU Projections
            GGUFLoader.Tensor gateWeight = find(gguf, prefix + "ffn_gate.weight");
            GGUFLoader.Tensor downWeight = find(gguf, prefix + "ffn_down.weight");
            GGUFLoader.Tensor upWeight   = find(gguf, prefix + "ffn_up.weight");
            
            // Layer Normalization Scales (Usually FP32)
            GGUFLoader.Tensor attnNorm = find(gguf, prefix + "attn_norm.weight");
            GGUFLoader.Tensor ffnNorm  = find(gguf, prefix + "ffn_norm.weight");

            // Build layer using zero-allocation off-heap extraction or copy fallback
            layers[i] = new LlamaLayer.Weights(
                qWeight.loadQ8_0AsFloat(),
                kWeight.loadQ8_0AsFloat(),
                vWeight.loadQ8_0AsFloat(),
                oWeight.loadQ8_0AsFloat(),
                gateWeight.loadQ8_0AsFloat(),
                downWeight.loadQ8_0AsFloat(),
                upWeight.loadQ8_0AsFloat(),
                attnNorm.loadQ8_0AsFloat(),
                ffnNorm.loadQ8_0AsFloat()
                    
                // Wire your custom structures here...
            );
        }
        return layers;
    }

    private static GGUFLoader.Tensor find(GGUFLoader.GGUFFile gguf, String name) {
        GGUFLoader.Tensor t = gguf.tensors.get(name);
        if (t == null) {
            throw new NoSuchElementException("Required model weight tensor was missing inside GGUF layout: " + name);
        }
        return t;
    }
}