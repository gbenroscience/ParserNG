package com.github.gbenroscience.sqlv1.parquet;

/**
 * Picks the best {@code limit} rows of a batch for {@code ORDER BY ... LIMIT n} with a bounded heap, instead of
 * sorting the batch: O(rows) when most rows cannot enter, O(rows log n) at worst, and no allocation per row.
 *
 * <p>Ordering is the engine's: keys compare as doubles, nulls last whatever the direction, a descending key
 * reverses only the non-null comparison, and ties keep arrival order (the lower row index wins). A row that cannot
 * beat the global threshold (the current n-th best row over all batches, supplied by {@link TopN}) is dropped before
 * it touches the heap. The result is sorted best first, which is what {@link TopN#addPicked} expects.
 */
final class TopSelect {

    /** One sort key of a batch, read by row. */
    interface Key {
        boolean isNull(int row);

        double get(int row);
    }

    private final int limit;
    private final boolean[] desc;
    private int[] heap = new int[64];

    TopSelect(int limit, boolean[] desc) {
        if (limit < 1) throw new IllegalArgumentException("limit must be at least 1");
        this.limit = limit;
        this.desc = desc.clone();
    }

    /** Negative if row {@code a} sorts before row {@code b}; ties by row index. */
    private int compare(Key[] keys, int a, int b) {
        for (int k = 0; k < keys.length; k++) {
            boolean na = keys[k].isNull(a), nb = keys[k].isNull(b);
            int c;
            if (na || nb) {
                c = na == nb ? 0 : (na ? 1 : -1);
            } else {
                c = Double.compare(keys[k].get(a), keys[k].get(b));
                if (desc[k]) c = -c;
            }
            if (c != 0) return c;
        }
        return Integer.compare(a, b);
    }

    /**
     * @param rows      rows {@code [0, rows)} are candidates
     * @param keys      one reader per ORDER BY key
     * @param threshold the global n-th best key tuple, or {@code null} if fewer than n rows are known so far
     * @return the picked row indexes, best first (at most {@code limit}); the array is owned by the caller
     */
    int[] select(int rows, Key[] keys, Threshold threshold) {
        int size = 0;
        for (int row = 0; row < rows; row++) {
            if (threshold != null && !threshold.beats(keys, row)) continue;
            if (size < limit) {
                if (size == heap.length) heap = java.util.Arrays.copyOf(heap, Math.min(limit, size * 2));
                heap[size] = row;
                siftUp(keys, size++);
            } else if (compare(keys, row, heap[0]) < 0) {
                heap[0] = row;
                siftDown(keys, 0, size);
            }
        }
        // heap sort: repeatedly take the worst, fill from the back
        int[] out = new int[size];
        int n = size;
        while (n > 0) {
            out[n - 1] = heap[0];
            heap[0] = heap[n - 1];
            n--;
            if (n > 0) siftDown(keys, 0, n);
        }
        return out;
    }

    /** Whether a row beats the n-th best row known so far. */
    interface Threshold {
        boolean beats(Key[] keys, int row);
    }

    private void siftUp(Key[] keys, int i) {
        int row = heap[i];
        while (i > 0) {
            int p = (i - 1) >>> 1;
            if (compare(keys, row, heap[p]) <= 0) break; // the heap keeps the worst row at the root
            heap[i] = heap[p];
            i = p;
        }
        heap[i] = row;
    }

    private void siftDown(Key[] keys, int i, int size) {
        int row = heap[i];
        while (true) {
            int c = 2 * i + 1;
            if (c >= size) break;
            if (c + 1 < size && compare(keys, heap[c + 1], heap[c]) > 0) c++;
            if (compare(keys, heap[c], row) <= 0) break;
            heap[i] = heap[c];
            i = c;
        }
        heap[i] = row;
    }
}
