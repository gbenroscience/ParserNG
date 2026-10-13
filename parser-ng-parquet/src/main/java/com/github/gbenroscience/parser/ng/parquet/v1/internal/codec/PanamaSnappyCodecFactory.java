package com.github.gbenroscience.parser.ng.parquet.v1.internal.codec;

import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * A {@link CompressionCodecFactory} that serves SNAPPY from the pure-Java FFM {@link SnappyDecompressor}
 * and delegates every other codec (and all compression) to parquet-java's default factory.
 *
 * <p>Installed through {@code ParquetReadOptions.Builder#withCodecFactory}, so parquet-java's own
 * page-reading pipeline calls it: no change to the page-store contract the decode engine relies on.
 * One factory per {@code ParquetReadOptions}; {@code ParquetBatchReader.newReadOptions} builds a fresh
 * one per reader/worker, so instances are never shared across threads.
 */
public final class PanamaSnappyCodecFactory implements CompressionCodecFactory {

    private final CompressionCodecFactory delegate;
    private final BytesInputDecompressor snappy;

    public PanamaSnappyCodecFactory(CompressionCodecFactory delegate) {
        this.delegate = java.util.Objects.requireNonNull(delegate, "delegate");
        this.snappy = new SnappyBytesInputDecompressor(delegate.getDecompressor(CompressionCodecName.SNAPPY));
    }

    @Override
    public BytesInputCompressor getCompressor(CompressionCodecName codecName) {
        return delegate.getCompressor(codecName);
    }

    @Override
    public BytesInputDecompressor getDecompressor(CompressionCodecName codecName) {
        return codecName == CompressionCodecName.SNAPPY ? snappy : delegate.getDecompressor(codecName);
    }

    @Override
    public void release() {
        delegate.release(); // also releases the delegate's SNAPPY decompressor that we hold
    }

    private static final class SnappyBytesInputDecompressor implements BytesInputDecompressor {
        private final BytesInputDecompressor fallback;

        SnappyBytesInputDecompressor(BytesInputDecompressor fallback) {
            this.fallback = fallback;
        }

        @Override
        public BytesInput decompress(BytesInput bytes, int uncompressedSize) throws IOException {
            ByteBuffer in = bytes.toByteBuffer();
            long declared = SnappyDecompressor.uncompressedLength(MemorySegment.ofBuffer(in), 0, in.remaining());
            if (declared != uncompressedSize) {
                throw new IOException("Snappy preamble says " + declared
                        + " bytes but the page header says " + uncompressedSize);
            }
            // Exact-size array: the decode engine can then wrap it directly (see PageBytes) with no further copy.
            byte[] out = new byte[uncompressedSize];
            SnappyDecompressor.decompress(MemorySegment.ofBuffer(in), 0, in.remaining(),
                    MemorySegment.ofArray(out), 0, uncompressedSize);
            return BytesInput.from(out);
        }

        @Override
        public void decompress(ByteBuffer input, int compressedSize, ByteBuffer output, int uncompressedSize)
                throws IOException {
            // Buffer-to-buffer overload: position/limit conventions differ between callers, so keep parquet-java's own.
            fallback.decompress(input, compressedSize, output, uncompressedSize);
        }

        @Override
        public void release() {
            // Stateless; the delegate's decompressor is released by delegate.release().
        }
    }
}
