package com.github.gbenroscience.gpu.llm.opencl;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.gbenroscience.gpu.llm.core.CpuReference;
import com.github.gbenroscience.gpu.llm.core.Fp16;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Runs the REAL OpenCL kernels (through the project's FFM bindings) against the
 * backend-independent CPU reference. Skipped -- never passed -- when no OpenCL
 * device is visible. The device (and whether it is a CPU or a GPU) is printed
 * so a PoCL/CPU run can never be mistaken for GPU verification.
 */
class OpenCLConformanceTest {

    private static GpuContext ctx;
    private static String skipReason;

    @BeforeAll
    static void open() {
        try {
            ctx = new GpuContext();
            System.out.println("[OpenCLConformanceTest] device = " + ctx.selectedDeviceDescription);
        } catch (Throwable t) {
            skipReason = "no usable OpenCL device: " + t;
        }
    }

    @AfterAll
    static void close() {
        if (ctx != null) {
            ctx.close();
        }
    }

    private static void needDevice() {
        assumeTrue(ctx != null, skipReason);
    }

    private static byte[] runQuantize(float[] x) throws Throwable {
        MemorySegment dx = LlamaLayer.uploadFloats(ctx, x);
        MemorySegment dq = LlamaLayer.allocBytes(ctx, (CpuReference.q8Bytes(x.length) + 3) / 4 * 4L);
        try {
            LlamaLayer.quantizeActivationQ8_0(ctx, dx, dq, x.length);
            LlamaLayer.finish(ctx);
            return download(dq, CpuReference.q8Bytes(x.length));
        } finally {
            LlamaLayer.freeQuietly(ctx, dx);
            LlamaLayer.freeQuietly(ctx, dq);
        }
    }

    private static byte[] download(MemorySegment dev, int bytes) throws Throwable {
        // downloadFloats copies raw 4-byte words; reinterpret them as bytes
        float[] words = LlamaLayer.downloadFloats(ctx, dev, (bytes + 3) / 4);
        byte[] out = new byte[bytes];
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(words.length * 4).order(java.nio.ByteOrder.nativeOrder());
        for (float w : words) {
            bb.putInt(Float.floatToRawIntBits(w));
        }
        System.arraycopy(bb.array(), 0, out, 0, bytes);
        return out;
    }

    @Test
    void quantizeKernelIsByteIdenticalToReferenceOnRandomData() throws Throwable {
        needDevice();
        Random r = new Random(10);
        for (int len : new int[]{32, 4096, 11008}) {
            float[] x = new float[len];
            for (int i = 0; i < len; i++) {
                double mag = Math.pow(10, r.nextInt(9) - 5); // 1e-5 .. 1e3
                x[i] = (float) (r.nextGaussian() * mag);
            }
            assertArrayEquals(CpuReference.quantizeQ8_0(x), runQuantize(x), "len=" + len);
        }
    }

    @Test
    void quantizeKernelRoundsFp16ScaleTiesToEven() throws Throwable {
        needDevice();
        Random r = new Random(11);
        // build blocks whose scale = absmax/127 lands exactly on an fp16 rounding tie
        float[] x = new float[32 * 4000];
        int tieBlocks = 0;
        for (int b = 0; b < 4000; b++) {
            int mant = (r.nextInt(1 << 10) << 13) | 0x1000; // exact halfway between two halves
            int exp = 100 + r.nextInt(60); // float exponent field -> normal half range
            float t = Float.intBitsToFloat((exp << 23) | mant);
            float absmax = t * 127f;
            if (absmax / 127f == t && Float.isFinite(absmax) && Fp16.floatToHalfBits(t) != Fp16.floatToHalfBits(Math.nextUp(t))) {
                tieBlocks++;
            }
            x[b * 32] = absmax;
            for (int j = 1; j < 32; j++) {
                x[b * 32 + j] = (float) (r.nextGaussian() * absmax * 0.1);
            }
        }
        assertTrue(tieBlocks > 500, "test must actually construct ties, got " + tieBlocks);
        assertArrayEquals(CpuReference.quantizeQ8_0(x), runQuantize(x));
    }
}
