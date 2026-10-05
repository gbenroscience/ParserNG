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
 
import com.github.gbenroscience.parser.ng.parquet.v1.internal.codec.SnappyDecompressor;
import org.xerial.snappy.Snappy;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import java.util.Arrays;

// Targets the NEW two-phase (fast-loop/tail-loop, SLOP=72) implementation specifically: sizes right at
// the SLOP boundary, large buffers with EVERY offset class (1,2,4, non-power-of-2 periods 3/5/6/7, >=8)
// repeating many times so the fast loop's main body (not just one element) exercises each branch, big
// literals (>=60 bytes, both mid-buffer and at the very end), and both entry points (byte[] AND
// MemorySegment, since heap-backed segments auto-route to the array path -- test native/non-array-backed
// segments too, which must take the MemorySegment loop, not the fast array shortcut).
public class SnappyFuzz2 {
  static int failures = 0;
  static int cases = 0;

  public static void main(String[] args) throws Exception {
    Random r = new Random(987654321L);

    // 1) Every size from 0 to 200 bytes of pure-random data -- walks every SLOP-relative boundary
    //    (srcLimit-SLOP, dstEnd-SLOP going negative, zero, exactly matching sp/dp, etc.) one byte at a time.
    for (int len = 0; len <= 200; len++) {
      byte[] b = new byte[len];
      r.nextBytes(b);
      check(b, "size-walk len=" + len);
    }

    // 2) Buffers built from many repetitions of each offset class, each repetition period chosen to
    //    force copies of varying length (4..64) so both the <=16/offset>=8 fast branch and the >16
    //    word-loop branch in copyMatchFast all fire repeatedly, not just once.
    for (int period : new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 200, 2000}) {
      for (int totalLen : new int[]{period + 1, 100, 500, 5000, 20000}) {
        if (totalLen <= period) continue;
        byte[] seed = new byte[period];
        r.nextBytes(seed);
        byte[] b = new byte[totalLen];
        for (int i = 0; i < totalLen; i++) b[i] = seed[i % period];
        check(b, "period=" + period + " totalLen=" + totalLen);
      }
    }

    // 3) Big literals (forcing the extended literal-length encoding, len 61..5000) placed so they land
    //    squarely inside the fast loop's own big-literal branch, not just the tail.
    for (int litLen : new int[]{60, 61, 64, 65, 100, 500, 4096, 5000}) {
      byte[] prefix = new byte[300]; // compressible prefix so the block doesn't start past SLOP immediately
      Arrays.fill(prefix, (byte) 7);
      byte[] lit = new byte[litLen];
      r.nextBytes(lit); // random -> snappy must emit it as a literal, not a copy
      byte[] suffix = new byte[300];
      Arrays.fill(suffix, (byte) 9);
      byte[] b = new byte[prefix.length + lit.length + suffix.length];
      System.arraycopy(prefix, 0, b, 0, prefix.length);
      System.arraycopy(lit, 0, b, prefix.length, lit.length);
      System.arraycopy(suffix, 0, b, prefix.length + lit.length, suffix.length);
      check(b, "bigLiteral=" + litLen);
    }

    // 4) Mixed realistic-ish data: interleaved compressible runs and random noise, many sizes, to hit
    //    the fast/tail transition at an arbitrary (not hand-picked) point within a single block.
    for (int trial = 0; trial < 3000; trial++) {
      int len = 1 + r.nextInt(3000);
      byte[] b = new byte[len];
      int i = 0;
      while (i < len) {
        int runLen = 1 + r.nextInt(50);
        if (r.nextBoolean()) {
          byte v = (byte) r.nextInt(256);
          for (int k = 0; k < runLen && i < len; k++, i++) b[i] = v;
        } else {
          for (int k = 0; k < runLen && i < len; k++, i++) b[i] = (byte) r.nextInt(256);
        }
      }
      check(b, "mixed trial=" + trial);
    }

    System.out.println("SnappyFuzz2 done: " + cases + " cases, " + failures + " failures");
    if (failures > 0) System.exit(1);
  }

  static void check(byte[] original, String label) throws Exception {
    cases++;
    byte[] compressed = Snappy.compress(original);

    // byte[] entry point
    byte[] gotArr = new byte[original.length];
    int n1 = SnappyDecompressor.decompress(compressed, 0, compressed.length, gotArr, 0, gotArr.length);
    if (n1 != original.length || !Arrays.equals(gotArr, original)) {
      System.out.println("FAIL [byte[]] " + label + ": len mismatch or content mismatch (n=" + n1 + ")");
      failures++;
    }

    // MemorySegment (native, NOT heap-array-backed) entry point
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment src = arena.allocate(Math.max(1, compressed.length));
      MemorySegment.copy(compressed, 0, src, ValueLayout.JAVA_BYTE, 0, compressed.length);
      MemorySegment dst = arena.allocate(Math.max(1, original.length));
      long n2 = SnappyDecompressor.decompress(src, 0, compressed.length, dst, 0, dst.byteSize());
      byte[] gotSeg = new byte[original.length];
      MemorySegment.copy(dst, ValueLayout.JAVA_BYTE, 0, gotSeg, 0, original.length);
      if (n2 != original.length || !Arrays.equals(gotSeg, original)) {
        System.out.println("FAIL [MemorySegment/native] " + label + ": len mismatch or content mismatch (n=" + n2 + ")");
        failures++;
      }
    }

    // MemorySegment.ofArray (heap-array-backed -> should auto-route to the fast array path)
    byte[] dstHeap = new byte[original.length];
    MemorySegment srcHeapSeg = MemorySegment.ofArray(compressed);
    MemorySegment dstHeapSeg = MemorySegment.ofArray(dstHeap);
    long n3 = SnappyDecompressor.decompress(srcHeapSeg, 0, compressed.length, dstHeapSeg, 0, dstHeap.length);
    if (n3 != original.length || !Arrays.equals(dstHeap, original)) {
      System.out.println("FAIL [MemorySegment/heap-array] " + label + ": len mismatch or content mismatch (n=" + n3 + ")");
      failures++;
    }
  }
}