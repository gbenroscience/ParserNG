/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.util.io;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * High-performance, thread-safe byte array builder with runtime endianness
 * toggle.
 *
 * @author GBEMIRO
 */
public final class ByteArrayBuilder implements Cloneable {

    private final Object lock = new Object();

    private static final class Buffer {

        private final List<byte[]> store = new ArrayList<>();
        private final AtomicInteger size = new AtomicInteger();

        private void reset() {
            size.set(0);
            store.clear();
        }
    }

    private final Buffer buffer;
    private byte[] realStore = {};
    private final ByteOrder byteOrder;

    // =====================================================================
    // CONSTRUCTORS
    // =====================================================================
    public ByteArrayBuilder() {
        this(ByteOrder.BIG_ENDIAN);
    }

    public ByteArrayBuilder(boolean littleEndian) {
        this(littleEndian ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
    }

    private ByteArrayBuilder(ByteOrder order) {
        this.byteOrder = order;
        this.buffer = new Buffer();
    }

    public ByteArrayBuilder(byte[] array) {
        this(false);
        append(array.clone(), false);           // deep copy
    }

    public ByteArrayBuilder(byte[] array, boolean littleEndian) {
        this(littleEndian);
        append(array.clone(), false);           // deep copy
    }

    public ByteArrayBuilder(ByteBuffer buffer) {
        this(false);
        append(buffer);
    }

    public ByteArrayBuilder(ByteBuffer buffer, boolean littleEndian) {
        this(littleEndian);
        append(buffer);
    }

    public ByteOrder getByteOrder() {
        return byteOrder;
    }

    // =====================================================================
    // ENDIAN CONVERSIONS
    // =====================================================================
    private byte[] toBytes(short value) {
        byte[] b = new byte[2];
        ByteBuffer.wrap(b).order(byteOrder).putShort(value);
        return b;
    }

    private byte[] toBytes(int value) {
        byte[] b = new byte[4];
        ByteBuffer.wrap(b).order(byteOrder).putInt(value);
        return b;
    }

    private byte[] toBytes(long value) {
        byte[] b = new byte[8];
        ByteBuffer.wrap(b).order(byteOrder).putLong(value);
        return b;
    }

    private byte[] toBytes(short[] shorts) {
        if (shorts == null || shorts.length == 0) {
            return new byte[0];
        }
        byte[] bytes = new byte[shorts.length * 2];
        ByteBuffer.wrap(bytes).order(byteOrder).asShortBuffer().put(shorts);
        return bytes;
    }

    private byte[] toBytes(int[] ints) {
        if (ints == null || ints.length == 0) {
            return new byte[0];
        }
        byte[] bytes = new byte[ints.length * 4];
        ByteBuffer.wrap(bytes).order(byteOrder).asIntBuffer().put(ints);
        return bytes;
    }

    private byte[] toBytes(long[] longs) {
        if (longs == null || longs.length == 0) {
            return new byte[0];
        }
        byte[] bytes = new byte[longs.length * 8];
        ByteBuffer.wrap(bytes).order(byteOrder).asLongBuffer().put(longs);
        return bytes;
    }

    private byte[] toBytes(double[] doubles) {
        if (doubles == null || doubles.length == 0) {
            return new byte[0];
        }
        byte[] bytes = new byte[doubles.length * 8];
        ByteBuffer.wrap(bytes)
                .order(byteOrder)
                .asDoubleBuffer()
                .put(doubles);
        return bytes;
    }

    private byte[] doubleToByteArray(double value) {
        byte[] bytes = new byte[8];
        ByteBuffer.wrap(bytes).order(byteOrder).putDouble(value);
        return bytes;
    }

    private static double readDouble(byte[] data, int offset, ByteOrder order) {
        return ByteBuffer.wrap(data, offset, 8).order(order).getDouble();
    }

    // =====================================================================
    // PUBLIC API
    // =====================================================================
    public ByteArrayBuilder append(long number) {
        return append(toBytes(number), false);
    }

    public ByteArrayBuilder append(double number) {
        return append(doubleToByteArray(number), false);
    }

    public ByteArrayBuilder append(int number) {
        return append(toBytes(number), false);
    }

    public ByteArrayBuilder append(short number) {
        return append(toBytes(number), false);
    }

    public ByteArrayBuilder append(byte data) {
        return append(new byte[]{data}, false);
    }

    public ByteArrayBuilder append(byte[] data) {
        return append(data.clone(), false);
    }

    /**
     * Appends a short array to this builder (respects current endianness).
     */
    public ByteArrayBuilder append(short[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        byte[] bytes = new byte[data.length * 2];
        ByteBuffer.wrap(bytes)
                .order(byteOrder)
                .asShortBuffer()
                .put(data);
        return append(bytes, false);
    }

    /**
     * Appends an int array to this builder (respects current endianness).
     */
    public ByteArrayBuilder append(int[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        byte[] bytes = new byte[data.length * 4];
        ByteBuffer.wrap(bytes)
                .order(byteOrder)
                .asIntBuffer()
                .put(data);
        return append(bytes, false);
    }

    /**
     * Appends a float array to this builder (respects current endianness).
     */
    public ByteArrayBuilder append(float[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        byte[] bytes = new byte[data.length * 4];
        ByteBuffer.wrap(bytes)
                .order(byteOrder)
                .asFloatBuffer()
                .put(data);
        return append(bytes, false);
    }

    /**
     * Appends a char array to this builder (respects current endianness). Each
     * char is stored as 2 bytes.
     */
    public ByteArrayBuilder append(char[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        byte[] bytes = new byte[data.length * 2];
        ByteBuffer.wrap(bytes)
                .order(byteOrder)
                .asCharBuffer()
                .put(data);
        return append(bytes, false);
    }

    /**
     * Appends a boolean array to this builder. Each boolean is stored as 1 byte
     * (0 = false, 1 = true).
     */
    public ByteArrayBuilder append(boolean[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = data[i] ? (byte) 1 : (byte) 0;
        }
        return append(bytes, false);
    }

    /**
     * Appends a long array to this builder (respects current endianness).
     */
    public ByteArrayBuilder append(long[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        byte[] bytes = new byte[data.length * 8];
        ByteBuffer.wrap(bytes)
                .order(byteOrder)
                .asLongBuffer()
                .put(data);
        return append(bytes, false);
    }

    public ByteArrayBuilder append(double[] data) {
        if (data == null || data.length == 0) {
            return this;
        }
        return append(toBytes(data), false);
    }

    public ByteArrayBuilder append(byte[] data, boolean last) {
        synchronized (lock) {
            buffer.store.add(data);
            buffer.size.addAndGet(data.length);
            if (last) {
                reconcile();
            }
            return this;
        }
    }

    public ByteArrayBuilder append(ByteBuffer buf) {
        if (buf == null) {
            throw new NullPointerException("ByteBuffer cannot be null");
        }
        int remaining = buf.remaining();
        if (remaining == 0) {
            return this;
        }
        if (buf.hasArray() && !buf.isReadOnly()) {
            byte[] raw = buf.array();
            int offset = buf.arrayOffset() + buf.position();
            appendItems(raw, offset, remaining, false);
            buf.position(buf.limit());
            return this;
        }
        byte[] chunk = new byte[remaining];
        buf.get(chunk);
        return append(chunk, false);
    }

    public ByteArrayBuilder appendItems(byte[] data, int offset, int numOfItems, boolean last) {
        if (offset < 0 || numOfItems < 0 || offset + numOfItems > data.length) {
            throw new IndexOutOfBoundsException("Invalid offset or length");
        }
        byte[] bits = new byte[numOfItems];
        System.arraycopy(data, offset, bits, 0, numOfItems);
        return append(bits, last);
    }

    public ByteArrayBuilder appendItems(byte[] data, int offset, int numOfItems) {
        return appendItems(data, offset, numOfItems, false);
    }

    public ByteArrayBuilder append(byte[] data, int fromIndex, int toIndex) {
        return append(data, fromIndex, toIndex, false);
    }

    /**
     * Appends bytes from {@code fromIndex} to {@code toIndex} **inclusive**
     * (original API behaviour).
     *
     * @param data
     * @param fromIndex
     * @param toIndex
     * @param last
     * @return
     */
    public ByteArrayBuilder append(byte[] data, int fromIndex, int toIndex, boolean last) {
        if (fromIndex < 0 || fromIndex > data.length) {
            throw new IllegalArgumentException("fromIndex must be >= 0 and <= data.length");
        }
        if (toIndex < fromIndex || toIndex >= data.length) {
            throw new IllegalArgumentException("toIndex must be >= fromIndex and < data.length (inclusive)");
        }
        byte[] bits = new byte[toIndex - fromIndex + 1];
        System.arraycopy(data, fromIndex, bits, 0, bits.length);
        return append(bits, last);
    }

    public ByteArrayBuilder prepend(byte[] data) {
        return insert(0, data);
    }

    public ByteArrayBuilder insert(int index, byte[] data) {
        synchronized (lock) {
            reconcile(); // must happen before bounds check
            if (index < 0 || index > realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Input Index: " + index + " > " + realStore.length);
            }
            if ((index == 0 && realStore.length == 0) || index == realStore.length) {
                return append(data.clone());
            }
            byte[] temp = new byte[realStore.length + data.length];
            System.arraycopy(realStore, 0, temp, 0, index);
            System.arraycopy(data, 0, temp, index, data.length);
            System.arraycopy(realStore, index, temp, index + data.length, realStore.length - index);
            realStore = temp;
            return this;
        }
    }

    public byte get(int index) {
        synchronized (lock) {
            reconcile();
            if (index < 0 || index >= realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Index " + index + " out of bounds. Length = " + realStore.length);
            }
            return realStore[index];
        }
    }

    public void set(int index, byte number) {
        synchronized (lock) {
            reconcile();
            if (index < 0 || index >= realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Index " + index + " out of bounds. Length = " + realStore.length);
            }
            realStore[index] = number;
        }
    }

    public ByteArrayBuilder set(int startIndex, byte[] data) {
        synchronized (lock) {
            reconcile();
            if (startIndex < 0 || startIndex >= realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Index " + startIndex + " out of bounds. Length = " + realStore.length);
            }
            if (startIndex + data.length > realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Not enough space: " + (startIndex + data.length) + " > " + realStore.length);
            }
            System.arraycopy(data, 0, realStore, startIndex, data.length);
            return this;
        }
    }

    public byte[] get(int startIndex, int numberOfItems) {
        synchronized (lock) {
            reconcile();
            if (startIndex < 0 || startIndex >= realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Index " + startIndex + " out of bounds. Length = " + realStore.length);
            }
            if (startIndex + numberOfItems > realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Not enough space: " + (startIndex + numberOfItems) + " > " + realStore.length);
            }
            byte[] data = new byte[numberOfItems];
            System.arraycopy(realStore, startIndex, data, 0, numberOfItems);
            return data;
        }
    }

    public ByteArrayBuilder remove(int start, int numberOfItems) {
        synchronized (lock) {
            reconcile();
            if (start < 0 || start >= realStore.length) {
                throw new ArrayIndexOutOfBoundsException("Index " + start + " out of bounds. Length = " + realStore.length);
            }
            if (numberOfItems <= 0) {
                throw new IllegalArgumentException("numberOfItems must be > 0");
            }
            int end = start + numberOfItems;
            if (end > realStore.length) {
                throw new ArrayIndexOutOfBoundsException("End index " + end + " > length " + realStore.length);
            }
            byte[] data = new byte[realStore.length - numberOfItems];
            System.arraycopy(realStore, 0, data, 0, start);
            System.arraycopy(realStore, end, data, start, realStore.length - end);
            this.realStore = data;
            return this;
        }
    }

    public byte[] getBytes() {
        synchronized (lock) {
            reconcile();
            return realStore;
        }
    }

    public double[] getAsDoubleArray() {
        synchronized (lock) {
            byte[] array = getBytes();
            if (array.length % 8 != 0) {
                throw new IllegalStateException("Byte length must be a multiple of 8 for double array conversion");
            }
            double[] values = new double[array.length / 8];
            for (int i = 0, j = 0; i < array.length; i += 8, j++) {
                values[j] = readDouble(array, i, byteOrder);
            }
            return values;
        }
    }

    public int length() {
        synchronized (lock) {
            return realStore.length + buffer.size.get();
        }
    }

    public int commitedLength() {
        return realStore.length;
    }

    public int nonCommitedLength() {
        return buffer.size.get();
    }

    @Override
    public ByteArrayBuilder clone() {
        synchronized (lock) {
            reconcile();
            ByteArrayBuilder b = new ByteArrayBuilder(byteOrder == ByteOrder.LITTLE_ENDIAN);
            b.realStore = new byte[this.realStore.length];
            System.arraycopy(realStore, 0, b.realStore, 0, realStore.length);
            return b;
        }
    }

    private void reconcile() {
        if (buffer.store.isEmpty()) {
            return;
        }
        int bufferSize = buffer.size.get();
        int existingLength = realStore.length;
        byte[] combined = new byte[existingLength + bufferSize];
        if (existingLength > 0) {
            System.arraycopy(realStore, 0, combined, 0, existingLength);
        }
        int index = existingLength;
        for (byte[] elem : buffer.store) {
            System.arraycopy(elem, 0, combined, index, elem.length);
            index += elem.length;
        }
        realStore = combined;
        buffer.reset();
    }

    @Override
    public String toString() {
        synchronized (lock) {
            reconcile();
            if (realStore.length == 0) {
                return "[]; item-count = 0";
            }
            StringBuilder b = new StringBuilder("[");
            int limit = Math.min(realStore.length, 100);
            for (int i = 0; i < limit; i++) {
                b.append(realStore[i]);
                if (i < limit - 1) {
                    b.append(", ");
                }
            }
            if (realStore.length > limit) {
                b.append(", ... (").append(realStore.length - limit).append(" more bytes)");
            }
            b.append("]; item-count = ").append(length());
            return b.toString();
        }
    }

    public void log() {
        System.out.println(toString());
    }

    public void sync() {
        synchronized (lock) {
            reconcile();
        }
    }

    public void clear() {
        synchronized (lock) {
            realStore = new byte[]{};
            buffer.reset();
        }
    }

}
