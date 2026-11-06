package com.github.gbenroscience.gpu.llm.opencl;

import com.github.gbenroscience.gpu.llm.core.Fp16;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Host-side helpers for the two Q8_0 device layouts (see {@link KernelSourceSoa} for the SoA byte layout).
 *
 * <ul>
 *   <li><b>AoS</b> -- the GGUF layout: 34-byte blocks (fp16 scale, 32 int8 quants), rows contiguous.</li>
 *   <li><b>SoA</b> -- all quants of the tensor first, then all scales; chosen at load time per device.</li>
 * </ul>
 */
final class Q8Layout {

    static final int GROUP = 32;
    static final int AOS_BLOCK = 34;

    private Q8Layout() {
    }

    /** Device bytes of a quantised ACTIVATION buffer of {@code len} elements (len % 32 == 0), rounded up to 4. */
    static long activationBytes(boolean soa, long len) {
        long b = soa ? len + (len / GROUP) * 4L : (len / GROUP) * AOS_BLOCK;
        return (b + 3) / 4 * 4;
    }

    /** Device bytes of a Q8_0 WEIGHT tensor [n, k] in the given layout. */
    static long weightBytes(boolean soa, long n, long k) {
        long blocks = n * (k / GROUP);
        return soa ? n * k + blocks * 2L : blocks * AOS_BLOCK;
    }

    private static void requireShape(long aosLength, long n, long k) {
        if (k <= 0 || k % GROUP != 0) {
            throw new IllegalArgumentException("Q8_0 row length K=" + k + " is not a positive multiple of " + GROUP);
        }
        if (aosLength != n * (k / GROUP) * AOS_BLOCK) {
            throw new IllegalArgumentException("Q8_0 tensor has " + aosLength + " bytes but N=" + n + ", K=" + k + " needs "
                    + (n * (k / GROUP) * AOS_BLOCK));
        }
    }

    /** Repacks a whole AoS Q8_0 tensor [n, k] into one SoA byte array (tests and small tensors; large ones use {@link #uploadWeightSoa}). */
    static byte[] repackWeightsSoa(byte[] aos, int n, int k) {
        requireShape(aos.length, n, k);
        final int nb = k / GROUP;
        byte[] out = new byte[(int) weightBytes(true, n, k)];
        final long scaleBase = (long) n * k;
        for (int r = 0; r < n; r++) {
            for (int b = 0; b < nb; b++) {
                int src = (int) (((long) r * nb + b) * AOS_BLOCK);
                System.arraycopy(aos, src + 2, out, (int) ((long) r * k + (long) b * GROUP), GROUP);
                int dst = (int) (scaleBase + 2L * ((long) r * nb + b));
                out[dst] = aos[src];
                out[dst + 1] = aos[src + 1];
            }
        }
        return out;
    }

    /**
     * Host reference of the SoA activation quantiser, for tests: {@code x} (length multiple of 32) to quants then
     * FLOAT scales (each fp16-rounded). Mirrors quantize_activation_q8_0_soa.
     */
    static byte[] packActivationSoa(float[] x) {
        final int len = x.length;
        if (len % GROUP != 0) {
            throw new IllegalArgumentException("length " + len + " is not a multiple of " + GROUP);
        }
        byte[] out = new byte[(int) activationBytes(true, len)];
        java.nio.ByteBuffer scales = java.nio.ByteBuffer.wrap(out).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int b = 0; b < len / GROUP; b++) {
            float absmax = 0f;
            for (int j = 0; j < GROUP; j++) {
                absmax = Math.max(absmax, Math.abs(x[b * GROUP + j]));
            }
            float scale = absmax / 127.0f;
            float inv = scale > 0f ? 1.0f / scale : 0f;
            for (int j = 0; j < GROUP; j++) {
                int q = (int) Math.rint((double) (x[b * GROUP + j] * inv));
                out[b * GROUP + j] = (byte) Math.max(-127, Math.min(127, q));
            }
            scales.putFloat(len + 4 * b, Fp16.halfBitsToFloat(Fp16.floatToHalfBits(scale)));
        }
        return out;
    }

    /**
     * Allocates a device buffer for the tensor in the requested layout and uploads it. For SoA the repack is done
     * row by row straight into an off-heap staging segment, so no second multi-hundred-megabyte Java array exists
     * (the LM head of a 128k-vocabulary model is ~280 MB).
     */
    static MemorySegment uploadWeight(GpuContext ctx, byte[] aos, int n, int k) throws Throwable {
        if (!ctx.q8Soa) {
            return LlamaLayer.uploadBytes(ctx, aos);
        }
        return uploadWeightSoa(ctx, aos, n, k);
    }

    static MemorySegment uploadWeightSoa(GpuContext ctx, byte[] aos, int n, int k) throws Throwable {
        requireShape(aos.length, n, k);
        final int nb = k / GROUP;
        final long total = weightBytes(true, n, k);
        MemorySegment dev = LlamaLayer.allocBytes(ctx, total);
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment host = tmp.allocate(total, 16);
            final long scaleBase = (long) n * k;
            byte[] rowQ = new byte[k];
            byte[] rowS = new byte[nb * 2];
            for (int r = 0; r < n; r++) {
                for (int b = 0; b < nb; b++) {
                    int src = (int) (((long) r * nb + b) * AOS_BLOCK);
                    System.arraycopy(aos, src + 2, rowQ, b * GROUP, GROUP);
                    rowS[2 * b] = aos[src];
                    rowS[2 * b + 1] = aos[src + 1];
                }
                MemorySegment.copy(rowQ, 0, host, ValueLayout.JAVA_BYTE, (long) r * k, k);
                MemorySegment.copy(rowS, 0, host, ValueLayout.JAVA_BYTE, scaleBase + 2L * r * nb, nb * 2);
            }
            LlamaLayer.writeBlocking(ctx, dev, host, total);
        } catch (Throwable t) {
            LlamaLayer.freeQuietly(ctx, dev);
            throw t;
        }
        return dev;
    }
}
