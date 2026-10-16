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
package com.github.gbenroscience.parser.ng.bench;


import jdk.incubator.vector.*; 
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;


/**
 *
 * @author GBEMIRO 
 * Raw sqrt throughput: VectorOperators.SQRT vs scalar Math.sqrt vs memory-copy ceiling.
 * JMH reports ns per invocation (one pass over n elements). Divide by n for ns/element
 * so the numbers are comparable with whatever you measure on the Gandiva side.
 *
 * Sizes are picked for a 32KB L1d / 256KB L2 / 4MB L3 part (i7-6500U):
 *   1024    -> in+out = 16KB   (L1)
 *   8192    -> in+out = 128KB  (L2)
 *   1<<20   -> in+out = 16MB   (DRAM)
 *
 * Run (needs --add-modules jdk.incubator.vector, already in jvmArgs below):
 *   java -jar target/benchmarks.jar SqrtBench -prof gc
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(value = 2, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector"})
public class SqrtBench {

    static final VectorSpecies<Double> D256 = DoubleVector.SPECIES_256;
    static final VectorSpecies<Double> DPREF = DoubleVector.SPECIES_PREFERRED;
    static final VectorSpecies<Float>  F256 = FloatVector.SPECIES_256;

    @Param({"1024", "8192", "1048576"})
    int n;

    double[] in, out;
    float[] fin, fout;

    @Setup
    public void setup() {
        Random r = new Random(42);
        in = new double[n];
        out = new double[n];
        fin = new float[n];
        fout = new float[n];
        for (int i = 0; i < n; i++) {
            in[i] = 1.0 + r.nextDouble() * 1e6;   // positive, no NaN/denormal paths
            fin[i] = (float) in[i];
        }
    }

    /** Scalar baseline: C2 will auto-vectorize this on AVX2, so it is a real contender. */
    @Benchmark
    public double[] scalarMathSqrt() {
        final double[] a = in, o = out;
        for (int i = 0; i < a.length; i++) o[i] = Math.sqrt(a[i]);
        return o;
    }

    /** Straight Vector API loop, 256-bit species. */
    @Benchmark
    public double[] vectorSqrt256() {
        final double[] a = in, o = out;
        final int bound = D256.loopBound(a.length);
        for (int i = 0; i < bound; i += D256.length()) {
            DoubleVector.fromArray(D256, a, i).lanewise(VectorOperators.SQRT).intoArray(o, i);
        }
        return o;
    }

    /** 4x unrolled: hides vsqrtpd latency by keeping several independent sqrts in flight. */
    @Benchmark
    public double[] vectorSqrt256Unrolled4() {
        final double[] a = in, o = out;
        final int L = D256.length();
        final int step = L * 4;
        final int bound = a.length - (a.length % step);
        for (int i = 0; i < bound; i += step) {
            DoubleVector v0 = DoubleVector.fromArray(D256, a, i);
            DoubleVector v1 = DoubleVector.fromArray(D256, a, i + L);
            DoubleVector v2 = DoubleVector.fromArray(D256, a, i + 2 * L);
            DoubleVector v3 = DoubleVector.fromArray(D256, a, i + 3 * L);
            v0.lanewise(VectorOperators.SQRT).intoArray(o, i);
            v1.lanewise(VectorOperators.SQRT).intoArray(o, i + L);
            v2.lanewise(VectorOperators.SQRT).intoArray(o, i + 2 * L);
            v3.lanewise(VectorOperators.SQRT).intoArray(o, i + 3 * L);
        }
        return o;
    }

    /** SPECIES_PREFERRED: should equal 256 on this CPU; here to confirm you are not on a 128-bit path. */
    @Benchmark
    public double[] vectorSqrtPreferred() {
        final double[] a = in, o = out;
        final int bound = DPREF.loopBound(a.length);
        for (int i = 0; i < bound; i += DPREF.length()) {
            DoubleVector.fromArray(DPREF, a, i).lanewise(VectorOperators.SQRT).intoArray(o, i);
        }
        return o;
    }

    /** Hypothesis check: is Gandiva running Float32 sqrt? vsqrtps ymm does 8 lanes ~2x faster than vsqrtpd ymm does 4. */
    @Benchmark
    public float[] vectorSqrtFloat256() {
        final float[] a = fin, o = fout;
        final int bound = F256.loopBound(a.length);
        for (int i = 0; i < bound; i += F256.length()) {
            FloatVector.fromArray(F256, a, i).lanewise(VectorOperators.SQRT).intoArray(o, i);
        }
        return o;
    }

    /** Memory ceiling: if sqrt is within ~1.2x of this at 1<<20, you are bandwidth-bound and no kernel will beat it. */
    @Benchmark
    public double[] copyCeiling() {
        System.arraycopy(in, 0, out, 0, in.length);
        return out;
    }

    // TODO: add the ParserNG path here, same arrays, same n, e.g.
    // @Benchmark public double[] parserNgSqrt() { return ParserNgSimd.sqrt(in, out); }
}