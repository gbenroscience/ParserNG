package com.github.gbenroscience.simd.turbo.tools.llm.loader;

import java.io.*;
import java.nio.*;
import java.util.*;
import java.nio.channels.*;
import java.nio.charset.*;
 


/**
 *
 * @author GBEMIRO
 */ 
public final class GGUFLoader {
    private GGUFLoader() {}


public static class Tensor {
    public final String name;
    public final int[] dims; // Ordered [Fastest-varying -> Slowest-varying] (e.g., [cols, rows])
    public final int type;   // GGML Type ID (0 = F32, 2 = Q4_0, 8 = Q8_0)
    public final long offsetRelative; 
    public final long nbytes;
    private MappedByteBuffer bufferWindow; // Isolated off-heap memory view

    Tensor(String name, int[] dims, int type, long offsetRelative, long nbytes) {
        this.name = name;
        this.dims = dims;
        this.type = type;
        this.offsetRelative = offsetRelative;
        this.nbytes = nbytes;
    }

    void setBufferWindow(MappedByteBuffer buf) {
        this.bufferWindow = buf;
        this.bufferWindow.order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Returns a zero-allocation, thread-safe duplicate view of the off-heap tensor storage.
     * Ideal for feeding direct vector pipelines or Project Panama MemorySegments.
     */
    public ByteBuffer buffer() {
        if (bufferWindow == null) throw new IllegalStateException("Tensor window not mapped.");
        return bufferWindow.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    public int rows() { return dims.length > 1 ? dims[1] : (dims.length > 0 ? dims[0] : 1); }
    public int cols() { return dims.length > 0 ? dims[0] : 1; }
    
    public long elements() {
        long ne = 1;
        for (int d : dims) ne *= d;
        return ne;
    }

    public float[] loadFloat() {
        if (type != 0) throw new IllegalArgumentException("Tensor '" + name + "' is not GGML_TYPE_F32 (0). Found: " + type);
        float[] out = new float[(int) (nbytes / 4)];
        buffer().asFloatBuffer().get(out);
        return out;
    }

    public byte[] loadQ8_0() {
        if (type != 8) throw new IllegalArgumentException("Tensor '" + name + "' is not GGML_TYPE_Q8_0 (8). Found: " + type);
        byte[] out = new byte[(int) nbytes];
        buffer().get(out);
        return out;
    }

    /** Bytes per bulk read when streaming a tensor through a reusable array (a whole number of 4-byte floats). */
    private static final int STREAM_CHUNK = 4 << 20;
    /** Q8_0 blocks per dequantisation chunk (34 bytes each, so a chunk is a whole number of blocks): about 2.2 MB. */
    private static final int Q8_BLOCKS_PER_CHUNK = 1 << 16;

    private interface ChunkSink {
        void accept(byte[] chunk, int length);
    }

    /**
     * Walks the tensor's bytes in chunks of at most {@code chunkBytes} through one reused array: one bulk copy per
     * chunk instead of one MappedByteBuffer.get() per element, which is what made the dequantising loaders slow.
     * {@code chunkBytes} must be a multiple of the element/block size the sink works in.
     */
    private void stream(int chunkBytes, ChunkSink sink) {
        ByteBuffer buf = buffer();
        byte[] chunk = new byte[Math.min(chunkBytes, Math.max(1, buf.remaining()))];
        while (buf.hasRemaining()) {
            int n = Math.min(chunk.length, buf.remaining());
            buf.get(chunk, 0, n);
            sink.accept(chunk, n);
        }
    }

    /**
     * Streams and dequantizes Q8_0 blocks straight into a flat float array.
     * Prevents intermediate byte[] allocations to optimize startup throughput.
     */
    public float[] loadQ8_0AsFloat() {
        if (type != 8) {
            throw new IllegalArgumentException("Tensor '" + name + "' is not GGML_TYPE_Q8_0 (8). Found: " + type);
        }
        final float[] out = new float[(int) elements()];
        final int[] idx = {0};
        stream(Q8_BLOCKS_PER_CHUNK * 34, (c, n) -> {
            int i = idx[0];
            for (int p = 0; p + 34 <= n; p += 34) {
                // ggml_fp16_t scale (little endian), then 32 signed quants
                float scale = Float.float16ToFloat((short) ((c[p] & 0xFF) | (c[p + 1] << 8)));
                for (int j = 0; j < 32; j++) {
                    out[i + j] = c[p + 2 + j] * scale;
                }
                i += 32;
            }
            idx[0] = i;
        });
        return out;
    }

    /**
     * Unpacks Q8_0 blocks straight into a flat double precision array 
     * to eliminate conversion steps if executing via a double vector pipeline.
     */
    public double[] loadQ8_0AsDouble() {
        if (type != 8) {
            throw new IllegalArgumentException("Tensor '" + name + "' is not GGML_TYPE_Q8_0 (8). Found: " + type);
        }
        final double[] out = new double[(int) elements()];
        final int[] idx = {0};
        stream(Q8_BLOCKS_PER_CHUNK * 34, (c, n) -> {
            int i = idx[0];
            for (int p = 0; p + 34 <= n; p += 34) {
                double scale = Float.float16ToFloat((short) ((c[p] & 0xFF) | (c[p + 1] << 8)));
                for (int j = 0; j < 32; j++) {
                    out[i + j] = c[p + 2 + j] * scale;
                }
                i += 32;
            }
            idx[0] = i;
        });
        return out;
    }

    /**
     * Fast-bulk read for unquantized FP32 weights upcasted cleanly into a double array.
     */
    public double[] loadFloatAsDouble() {
        if (type != 0) {
            throw new IllegalArgumentException("Tensor '" + name + "' is not GGML_TYPE_F32 (0). Found: " + type);
        }
        final double[] out = new double[(int) elements()];
        final int[] idx = {0};
        stream(STREAM_CHUNK, (c, n) -> {
            java.nio.FloatBuffer fb = ByteBuffer.wrap(c, 0, n).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            int i = idx[0];
            while (fb.hasRemaining()) {
                out[i++] = fb.get();
            }
            idx[0] = i;
        });
        return out;
    }
}

    public static class GGUFFile {
        public final Map<String, Object> metadata = new HashMap<>();
        public final Map<String, Tensor> tensors = new LinkedHashMap<>();
    }

    public static GGUFFile load(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel ch = raf.getChannel()) {
            
            long fileSize = ch.size();
            // Map the initial segment (e.g., 256MB) to safely parsing string layouts and metadata tokens
            long headerMapSize = Math.min(fileSize, 256 * 1024 * 1024);
            MappedByteBuffer headerBuf = ch.map(FileChannel.MapMode.READ_ONLY, 0, headerMapSize);
            headerBuf.order(ByteOrder.LITTLE_ENDIAN);

            int magic = headerBuf.getInt();
            if (magic != 0x46554747) { // "GGUF" in Little Endian ASCII
                throw new IOException("Invalid GGUF magic token: 0x" + Integer.toHexString(magic));
            }

            int version = headerBuf.getInt();
            if (version < 2 || version > 3) {
                throw new IOException("Unsupported GGUF layout version format: " + version);
            }

            long tensorCount = headerBuf.getLong();
            long kvCount = headerBuf.getLong();

            GGUFFile gguf = new GGUFFile();

            // 1. Process Key-Value Hyperparameters
            for (long i = 0; i < kvCount; i++) {
                String key = readString(headerBuf);
                int typeId = headerBuf.getInt();
                Object val = readValue(headerBuf, typeId);
                gguf.metadata.put(key, val);
            }

            // Extract alignment (default according to spec is 32 bytes)
            long alignment = 32;
            Object alignVal = gguf.metadata.get("general.alignment");
            if (alignVal instanceof Number) {
                alignment = ((Number) alignVal).longValue();
            }

            // 2. Process Tensor Configuration Headers
            List<Tensor> parsedTensors = new ArrayList<>();
            for (long i = 0; i < tensorCount; i++) {
                String name = readString(headerBuf);
                int nDims = headerBuf.getInt();
                int[] dims = new int[nDims];
                long numElements = 1;
                for (int d = 0; d < nDims; d++) {
                    dims[d] = (int) headerBuf.getLong();
                    numElements *= dims[d];
                }
                int type = headerBuf.getInt();
                long offsetRelative = headerBuf.getLong();
                long nbytes = calculateTensorBytes(type, numElements);

                Tensor tensor = new Tensor(name, dims, type, offsetRelative, nbytes);
                parsedTensors.add(tensor);
                gguf.tensors.put(name, tensor);
            }

            // Pinpoint structural start boundary of tensor data
            long headerEndPos = headerBuf.position();
            long padding = (alignment - (headerEndPos % alignment)) % alignment;
            long tensorDataStartOffset = headerEndPos + padding;

            // 3. Decentralized Mapping: Break 2GB ceiling by mapping segments directly
            for (Tensor t : parsedTensors) {
                long absoluteOffset = tensorDataStartOffset + t.offsetRelative;
                if (absoluteOffset + t.nbytes > fileSize) {
                    throw new IOException("Malformed GGUF execution window bounds for tensor: " + t.name);
                }
                // Each tensor gets its own perfectly sliced native buffer window from OS virtual memory
                MappedByteBuffer tensorView = ch.map(FileChannel.MapMode.READ_ONLY, absoluteOffset, t.nbytes);
                t.setBufferWindow(tensorView);
            }

            return gguf;
        }
    }

    private static String readString(ByteBuffer buf) {
        int len = (int) buf.getLong();
        byte[] bytes = new byte[len];
        buf.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static Object readValue(ByteBuffer buf, int typeId) {
        switch (typeId) {
            case 0: return buf.get() & 0xFF;         // GGUF_TYPE_UINT8
            case 1: return (int) buf.get();          // GGUF_TYPE_INT8
            case 2: return buf.getShort() & 0xFFFF;  // GGUF_TYPE_UINT16
            case 3: return (int) buf.getShort();     // GGUF_TYPE_INT16
            case 4: return Integer.toUnsignedLong(buf.getInt()); // GGUF_TYPE_UINT32
            case 5: return buf.getInt();             // GGUF_TYPE_INT32
            case 6: return buf.getFloat();           // GGUF_TYPE_FLOAT32
            case 7: return buf.get() != 0;           // GGUF_TYPE_BOOL
            case 8: return readString(buf);          // GGUF_TYPE_STRING
            case 9: {                                // GGUF_TYPE_ARRAY
                int arrTypeId = buf.getInt();
                int len = (int) buf.getLong();
                List<Object> list = new ArrayList<>(len);
                for (int i = 0; i < len; i++) {
                    list.add(readValue(buf, arrTypeId));
                }
                return list;
            }
            case 10: return buf.getLong();           // GGUF_TYPE_UINT64
            case 11: return buf.getLong();           // GGUF_TYPE_INT64
            case 12: return buf.getDouble();         // GGUF_TYPE_FLOAT64
            default: throw new IllegalArgumentException("Unknown GGUF metadata type ID: " + typeId);
        }
    }

    private static long calculateTensorBytes(int typeId, long numElements) {
        switch (typeId) {
            case 0: return numElements * 4; // GGML_TYPE_F32
            case 1: return numElements * 2; // GGML_TYPE_F16
            case 2: return (numElements / 32) * 18; // GGML_TYPE_Q4_0 (2-byte FP16 scale + 16 bytes nibbles)
            case 3: return (numElements / 32) * 20; // GGML_TYPE_Q4_1
            case 6: return (numElements / 32) * 22; // GGML_TYPE_Q5_0
            case 7: return (numElements / 32) * 24; // GGML_TYPE_Q5_1
            case 8: return (numElements / 32) * 34; // GGML_TYPE_Q8_0 (2-byte FP16 scale + 32 bytes values)
            case 9: return (numElements / 32) * 36; // GGML_TYPE_Q8_1
            default:
                throw new UnsupportedOperationException("GGML tensor type encoding unhandled or sparse: " + typeId);
        }
    }
}