package com.github.gbenroscience.gpu.llm.core;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal GGUF v3 writer for tests, written from the format specification and
 * independent of GGUFLoader (the thing under test). Supports uint32/int32/
 * float32/bool/string/array-of-string metadata and F32 / Q8_0 / raw-typed tensors.
 */
public final class GgufTestWriter {

    public static final class Tensor {

        public final String name;
        public final int[] dims; // [cols, rows, ...] fastest first
        public final int type;
        public final byte[] data;

        public Tensor(String name, int[] dims, int type, byte[] data) {
            this.name = name;
            this.dims = dims;
            this.type = type;
            this.data = data;
        }
    }

    public final Map<String, Object> metadata = new LinkedHashMap<>();
    public final Map<String, Tensor> tensors = new LinkedHashMap<>();

    public File write(File f) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        w(out, new byte[]{'G', 'G', 'U', 'F'});
        u32(out, 3);
        u64(out, tensors.size());
        u64(out, metadata.size());
        for (Map.Entry<String, Object> e : metadata.entrySet()) {
            str(out, e.getKey());
            value(out, e.getValue());
        }
        long offset = 0;
        List<Long> offsets = new ArrayList<>();
        for (Tensor t : tensors.values()) {
            str(out, t.name);
            u32(out, t.dims.length);
            for (int d : t.dims) {
                u64(out, d);
            }
            u32(out, t.type);
            u64(out, offset);
            offsets.add(offset);
            offset += align(t.data.length);
        }
        while (out.size() % 32 != 0) {
            out.write(0);
        }
        for (Tensor t : tensors.values()) {
            w(out, t.data);
            for (long i = t.data.length; i < align(t.data.length); i++) {
                out.write(0);
            }
        }
        Files.write(f.toPath(), out.toByteArray());
        return f;
    }

    private static long align(long n) {
        return (n + 31) / 32 * 32;
    }

    private static void value(ByteArrayOutputStream o, Object v) {
        if (v instanceof Integer) {
            u32(o, 5);
            u32(o, (Integer) v);
        } else if (v instanceof Float) {
            u32(o, 6);
            u32(o, Float.floatToIntBits((Float) v));
        } else if (v instanceof Double) {
            u32(o, 6);
            u32(o, Float.floatToIntBits(((Double) v).floatValue()));
        } else if (v instanceof Boolean) {
            u32(o, 7);
            o.write(((Boolean) v) ? 1 : 0);
        } else if (v instanceof String) {
            u32(o, 8);
            str(o, (String) v);
        } else if (v instanceof List) {
            u32(o, 9);
            u32(o, 8); // array of strings
            List<?> l = (List<?>) v;
            u64(o, l.size());
            for (Object s : l) {
                str(o, String.valueOf(s));
            }
        } else {
            throw new IllegalArgumentException("unsupported metadata value " + v);
        }
    }

    private static void str(ByteArrayOutputStream o, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        u64(o, b.length);
        w(o, b);
    }

    private static void u32(ByteArrayOutputStream o, int v) {
        w(o, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
    }

    private static void u64(ByteArrayOutputStream o, long v) {
        w(o, ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array());
    }

    private static void w(ByteArrayOutputStream o, byte[] b) {
        o.write(b, 0, b.length);
    }

    public static byte[] f32Bytes(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putFloat(x);
        }
        return b.array();
    }
}
