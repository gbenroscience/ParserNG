/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.parser.ng.parquet.v1.bench;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.ByteReader;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.DeltaBinaryPackedDecoder;
import java.io.ByteArrayOutputStream;
import java.util.Random;

/** Encodes DELTA_BINARY_PACKED the way parquet-java's writer lays it out and decodes it with the module's decoder. */
public class DeltaRepro {
    static void uvar(ByteArrayOutputStream o, long v) { while ((v & ~0x7FL) != 0) { o.write((int) ((v & 0x7F) | 0x80)); v >>>= 7; } o.write((int) v); }
    static long zz(long v) { return (v << 1) ^ (v >> 63); }

    /**
     * staleWidths=true : bit widths of unneeded miniblocks of the LAST block keep the previous block's values (field reused across blocks),
     *                    their data is omitted. staleWidths=false: unneeded widths are 0.
     */
    static byte[] encode(long[] vals, int blockSize, int miniblocks, boolean staleWidths) {
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        uvar(o, blockSize); uvar(o, miniblocks); uvar(o, vals.length); uvar(o, zz(vals[0]));
        int per = blockSize / miniblocks; int[] widths = new int[miniblocks];
        int i = 1;
        while (i < vals.length) {
            int n = Math.min(blockSize, vals.length - i);
            long[] d = new long[n]; long min = Long.MAX_VALUE;
            for (int k = 0; k < n; k++) { d[k] = vals[i + k] - vals[i + k - 1]; min = Math.min(min, d[k]); }
            for (int k = 0; k < n; k++) d[k] -= min;
            int need = (n + per - 1) / per;
            for (int m = 0; m < need; m++) { long mask = 0; for (int k = m * per; k < Math.min(n, (m + 1) * per); k++) mask |= d[k]; widths[m] = 64 - Long.numberOfLeadingZeros(mask); }
            if (!staleWidths) for (int m = need; m < miniblocks; m++) widths[m] = 0;
            uvar(o, zz(min));
            for (int m = 0; m < miniblocks; m++) o.write(widths[m]);
            for (int m = 0; m < need; m++) {
                int w = widths[m]; long acc = 0; int bits = 0;
                for (int k = m * per; k < (m + 1) * per; k++) { long v = k < n ? d[k] : 0; acc |= v << bits; bits += w; while (bits >= 8) { o.write((int) (acc & 0xFF)); acc >>>= 8; bits -= 8; } }
                if (bits > 0) o.write((int) (acc & 0xFF));
            }
            i += n;
        }
        return o.toByteArray();
    }

    static String tryDecode(byte[] page, long[] expect, byte[] trailer) {
        // page = delta sequence #1, followed by delta sequence #2 (same layout), like DELTA_BYTE_ARRAY's prefix/suffix lengths
        byte[] all = new byte[page.length * 2]; System.arraycopy(page, 0, all, 0, page.length); System.arraycopy(page, 0, all, page.length, page.length);
        ByteReader br = new ByteReader(); br.wrap(all, 0, all.length);
        long[] a = new long[expect.length], b = new long[expect.length];
        try { DeltaBinaryPackedDecoder.decode(br, a, expect.length); DeltaBinaryPackedDecoder.decode(br, b, expect.length); }
        catch (RuntimeException e) { return "FAIL: " + e.getMessage(); }
        for (int i = 0; i < expect.length; i++) if (a[i] != expect[i] || b[i] != expect[i]) return "WRONG VALUES at " + i;
        return br.pos == all.length ? "ok" : "ok but ended at " + br.pos + " of " + all.length;
    }

    public static void main(String[] x) {
        Random r = new Random(1);
        for (int n : new int[]{2, 33, 129, 130, 161, 257, 300, 301, 400}) {
            long[] v = new long[n]; long cur = 1000; for (int i = 0; i < n; i++) { cur += r.nextInt(40); v[i] = cur; }   // varying positive deltas -> nonzero widths in every block
            System.out.printf("n=%-4d stale-widths: %-70s zero-widths: %s%n", n, tryDecode(encode(v, 128, 4, true), v, null), tryDecode(encode(v, 128, 4, false), v, null));
        }
    }
}
class Sweep {
    public static void main(String[] a) {
        Random r = new Random(5); int cases = 0, bad = 0;
        int[][] shapes = {{128, 4}, {64, 2}, {256, 8}, {32, 1}, {128, 1}, {64, 4}};
        for (int[] sh : shapes) for (int n = 1; n <= 700; n += (n < 40 ? 1 : 13)) for (boolean stale : new boolean[]{true, false}) for (int mode = 0; mode < 3; mode++) {
            long[] v = new long[n]; long cur = r.nextInt(1000);
            for (int i = 0; i < n; i++) { cur += switch (mode) { case 0 -> r.nextInt(40); case 1 -> r.nextInt(2000001) - 1000000; default -> 1; }; v[i] = cur; }
            String res = DeltaRepro.tryDecode(DeltaRepro.encode(v, sh[0], sh[1], stale), v, null); cases++;
            if (!res.equals("ok")) { bad++; if (bad < 6) System.out.println("BAD shape=" + sh[0] + "/" + sh[1] + " n=" + n + " stale=" + stale + " mode=" + mode + ": " + res); }
        }
        System.out.println("sweep: " + cases + " cases, " + bad + " bad");
    }
}