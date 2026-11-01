package com.github.gbenroscience.gpu.llm.opencl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wall-clock breakdown of model loading (GGUF parse, validation, tokenizer, host embedding table, device and
 * program creation, per-layer tensor read vs device upload, state allocation, ...) plus free-text notes such as
 * the device-memory report. Process-wide and cheap; call {@link #reset()} before loading a model and
 * {@link #report()} afterwards.
 */
public final class LoadTimer {

    private static final Map<String, long[]> PHASES = new LinkedHashMap<>(); // name -> {nanos, calls}
    private static final List<String> NOTES = new ArrayList<>();

    private LoadTimer() {
    }

    public static synchronized void reset() {
        PHASES.clear();
        NOTES.clear();
    }

    public static long start() {
        return System.nanoTime();
    }

    public static synchronized void stop(String phase, long startNanos) {
        long[] a = PHASES.computeIfAbsent(phase, k -> new long[2]);
        a[0] += System.nanoTime() - startNanos;
        a[1]++;
    }

    public static synchronized void note(String text) {
        NOTES.add(text);
    }

    public static synchronized String report() {
        StringBuilder sb = new StringBuilder("\n==================== load breakdown ====================\n");
        long total = 0;
        for (Map.Entry<String, long[]> e : PHASES.entrySet()) {
            total += e.getValue()[0];
            sb.append(String.format("%-52s %10.1f ms  (%d call%s)%n", e.getKey(), e.getValue()[0] / 1e6, e.getValue()[1], e.getValue()[1] == 1 ? "" : "s"));
        }
        sb.append(String.format("%-52s %10.1f ms%n", "sum of timed phases", total / 1e6));
        for (String n : NOTES) {
            sb.append(n).append('\n');
        }
        return sb.toString();
    }
}
