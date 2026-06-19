package com.github.gbenroscience.simd.turbo.tools;

import java.util.*;
import java.util.concurrent.*;

/**
 * High-Performance, Allocation-Free Multi-Head Attention Engine.
 * Tailored specifically for the ParserNG Turbo hardware stack.
 *
 * @author GBEMIRO
 */
public final class MultiHeadAttention {
    
    private final int numHeads;
    private final int dModel;
    private final int dHead;
    private final FlatMatrix W_Q, W_K, W_V, W_O;
    
    // Persistent workspaces to eliminate runtime heap allocations completely
    private FlatMatrix workspaceQ;
    private FlatMatrix workspaceK;
    private FlatMatrix workspaceV;
    private FlatMatrix workspaceHeadOut;

    public MultiHeadAttention(int dModel, int numHeads, 
                              FlatMatrix W_Q, FlatMatrix W_K, FlatMatrix W_V, FlatMatrix W_O) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dHead = dModel / numHeads;
        this.W_Q = W_Q; this.W_K = W_K; this.W_V = W_V; this.W_O = W_O;
    }
    
    /**
     * Executes the forward attention pass without a single runtime allocation.
     * Keeps execution locked inside the CPU's native cache boundaries.
     */
    public FlatMatrix forward(FlatMatrix X, FlatMatrix out, ExecutorService executor) {
        final int seqLen = X.rows;
        
        // Dynamic re-initialization of cached workspaces ONLY if the input batch sequence length changes
        if (workspaceQ == null || workspaceQ.rows != seqLen) {
            workspaceQ = new FlatMatrix(seqLen, dModel);
            workspaceK = new FlatMatrix(seqLen, dModel);
            workspaceV = new FlatMatrix(seqLen, dModel);
            workspaceHeadOut = new FlatMatrix(seqLen, dModel);
        }

        // 1. Concurrent Q, K, V Projections using pre-allocated matrix buffers
        if (executor == null) {
            FlatMatrix.matmul(X, W_Q, workspaceQ); 
            FlatMatrix.matmul(X, W_K, workspaceK);
            FlatMatrix.matmul(X, W_V, workspaceV);
        } else {
            Future<?> fQ = executor.submit(() -> FlatMatrix.matmul(X, W_Q, workspaceQ));
            Future<?> fK = executor.submit(() -> FlatMatrix.matmul(X, W_K, workspaceK));
            Future<?> fV = executor.submit(() -> FlatMatrix.matmul(X, W_V, workspaceV));
            try {
                fQ.get(); fK.get(); fV.get();
            } catch (ExecutionException e) {
                throw new RuntimeException("QKV projection failed", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted during projections", e);
            }
        }
        
        // 2. Process all attention heads asynchronously
        mhaHeads(workspaceQ, workspaceK, workspaceV, workspaceHeadOut, executor);
        
        // 3. Final projection pass directly out into your destination memory buffer
        FlatMatrix.matmul(workspaceHeadOut, W_O, out); 
        return out;
    } 

    private void mhaHeads(FlatMatrix Q, FlatMatrix K, FlatMatrix V, 
                          FlatMatrix headOut, ExecutorService executor) {
        final int seqLen = Q.rows;
        
        if (executor == null) {
            for (int h = 0; h < numHeads; h++) {
                runHeadFast(h, Q, K, V, headOut, seqLen);
            }
            return;
        }
        
        List<Future<?>> futures = new ArrayList<>(numHeads);
        for (int h = 0; h < numHeads; h++) {
            final int head = h;
            futures.add(executor.submit(() -> runHeadFast(head, Q, K, V, headOut, seqLen)));
        }
        for (Future<?> f : futures) {
            try { f.get(); } 
            catch (ExecutionException e) {
                throw new RuntimeException("MHA head operation failed", e.getCause());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted during head loops", e);
            }
        }
    }
    
    /**
     * Zero-Allocation variant of head extraction utilizing structural sub-views.
     * Since FlatMatrix.view() creates a lightweight container pointer, it is fast,
     * but avoids full duplication.
     */
    private void runHeadFast(int h, FlatMatrix Q, FlatMatrix K, FlatMatrix V, 
                             FlatMatrix headOut, int seqLen) {
        int colOffset = h * dHead;
        
        // Views act as zero-copy windows into our pre-allocated workspace arrays
        FlatMatrix Qh = Q.view(0, colOffset, seqLen, dHead);
        FlatMatrix Kh = K.view(0, colOffset, seqLen, dHead);
        FlatMatrix Vh = V.view(0, colOffset, seqLen, dHead);
        FlatMatrix Oh = headOut.view(0, colOffset, seqLen, dHead);
        
        // Executes the vectorized SIMD attention block directly on your internal data primitive arrays
        FlatMatrix.attention(Qh, Kh, Vh, Oh, null); 
    }
}
