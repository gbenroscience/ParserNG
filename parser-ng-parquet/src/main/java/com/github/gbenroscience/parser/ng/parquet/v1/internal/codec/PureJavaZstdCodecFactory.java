package com.github.gbenroscience.parser.ng.parquet.v1.internal.codec;

import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.compression.CompressionCodecFactory;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * A {@link CompressionCodecFactory} that serves ZSTD from the pure-Java {@link ZstdDecompressor} and delegates
 * every other codec (and all compression) to the wrapped factory.
 *
 * <p>The ZSTD counterpart of {@link PanamaSnappyCodecFactory}, and built the same way: installed through
 * {@code ParquetReadOptions.Builder#withCodecFactory}, so parquet-java's own page-reading pipeline calls it and the
 * page-store contract the decode engine relies on is untouched. The two stack, because each only intercepts its own
 * codec and forwards the rest: {@code new PureJavaZstdCodecFactory(new PanamaSnappyCodecFactory(base))} serves
 * SNAPPY and ZSTD in pure Java. One factory per {@code ParquetReadOptions};
 * {@code ParquetBatchReader.newReadOptions} builds a fresh one per reader/worker, so instances (and the
 * non-thread-safe decoder inside) are never shared across threads.
 *
 * <p>Unlike the Snappy factory this one never asks the delegate for a ZSTD decompressor, not even as a fallback:
 * both {@code decompress} overloads are implemented here. Constructing parquet-java's ZSTD codec loads
 * {@code zstd-jni}, which is exactly the native dependency this option exists to avoid. The decoder itself is
 * created lazily, on the first ZSTD page, so scans of files that never use ZSTD pay nothing for it.
 */
public final class PureJavaZstdCodecFactory implements CompressionCodecFactory {

    private final CompressionCodecFactory delegate;
    private ZstdBytesInputDecompressor zstd; // created on first use

    public PureJavaZstdCodecFactory(CompressionCodecFactory delegate) {
        this.delegate = java.util.Objects.requireNonNull(delegate, "delegate");
    }

    @Override
    public BytesInputCompressor getCompressor(CompressionCodecName codecName) {
        return delegate.getCompressor(codecName);
    }

    @Override
    public BytesInputDecompressor getDecompressor(CompressionCodecName codecName) {
        if (codecName != CompressionCodecName.ZSTD) return delegate.getDecompressor(codecName);
        ZstdBytesInputDecompressor z = zstd;
        if (z == null) zstd = z = new ZstdBytesInputDecompressor();
        return z;
    }

    @Override
    public void release() {
        zstd = null;          // plain heap arrays: nothing to hand back, just let go of them
        delegate.release();
    }

    private static final class ZstdBytesInputDecompressor implements BytesInputDecompressor {
        private final ZstdDecompressor engine = new ZstdDecompressor();
        /** Staging area for input that has no accessible backing array (direct or read-only buffers). Grows only. */
        private byte[] staging = new byte[0];

        @Override
        public BytesInput decompress(BytesInput bytes, int uncompressedSize) throws IOException {
            requireNonNegative(uncompressedSize);
            // Exact-size array: the decode engine can then wrap it directly (see PageBytes) with no further copy.
            byte[] out = new byte[uncompressedSize];
            decodeInto(bytes.toByteBuffer(), out, 0, uncompressedSize);
            return BytesInput.from(out);
        }

        /**
         * Same contract as parquet-java's own heap decompressor: consumes exactly {@code compressedSize} bytes
         * from {@code input} (position advances by that much, limit is restored) and {@code put}s the
         * decompressed bytes into {@code output} (its position advances by {@code uncompressedSize}).
         */
        @Override
        public void decompress(ByteBuffer input, int compressedSize, ByteBuffer output, int uncompressedSize)
                throws IOException {
            if (input.remaining() < compressedSize) {
                throw new IllegalArgumentException("Not enough bytes available in the input buffer");
            }
            requireNonNegative(uncompressedSize);
            if (output.remaining() < uncompressedSize) {
                throw new IOException("Output buffer has " + output.remaining() + " bytes free but the page needs "
                        + uncompressedSize);
            }
            final int origLimit = input.limit();
            final int origPosition = input.position();
            input.limit(origPosition + compressedSize);
            try {
                if (output.hasArray()) { // decode straight into the caller's array: no temporary, no copy
                    decodeInto(input, output.array(), output.arrayOffset() + output.position(), uncompressedSize);
                    output.position(output.position() + uncompressedSize);
                } else {
                    byte[] tmp = new byte[uncompressedSize];
                    decodeInto(input, tmp, 0, uncompressedSize);
                    output.put(tmp);
                }
            } finally {
                input.limit(origLimit);
            }
            input.position(origPosition + compressedSize);
        }

        @Override
        public void release() {
            // Nothing to release; the factory drops this object in release().
        }

        /**
         * Decodes the frame(s) in {@code in} (position..limit, not modified) into exactly
         * {@code size} bytes at {@code dst[dstOff..]}, or throws {@link IOException}.
         */
        private void decodeInto(ByteBuffer in, byte[] dst, int dstOff, int size) throws IOException {
            final byte[] src;
            final int off, len = in.remaining();
            if (in.hasArray()) {
                src = in.array();
                off = in.arrayOffset() + in.position();
            } else {
                if (staging.length < len) staging = new byte[len];
                in.duplicate().get(staging, 0, len); // duplicate(): leave the caller's position alone
                src = staging;
                off = 0;
            }
            // Fail fast, with a clear message, when the first frame alone already declares more than the page may
            // hold. Only ">" is conclusive: a smaller declared size is legal because a page may be several
            // concatenated frames (libzstd's ZSTD_decompress accepts that too); the final length check covers it.
            long declared = ZstdDecompressor.frameContentSize(src, off, len);
            if (declared > size) {
                throw new IOException("ZSTD frame declares " + declared + " bytes but the page header says " + size);
            }
            final int n;
            try {
                n = engine.decompress(src, off, len, dst, dstOff, size);
            } catch (ZstdDecompressor.ZstdException e) {
                throw new IOException("Corrupt ZSTD page: " + e.getMessage(), e);
            }
            if (n != size) {
                throw new IOException("ZSTD page decoded to " + n + " bytes but the page header says " + size);
            }
        }

        private static void requireNonNegative(int size) throws IOException {
            if (size < 0) throw new IOException("Negative uncompressed page size " + size);
        }
    }
}
