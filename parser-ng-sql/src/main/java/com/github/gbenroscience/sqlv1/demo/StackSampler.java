package com.github.gbenroscience.sqlv1.demo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * A small in-process sampling profiler, for when no external profiler is at hand. A daemon thread takes the stack
 * of every runnable thread every few milliseconds and counts, per method, how often it was the top frame (self) and
 * how often it appeared anywhere in a stack (inclusive, each stack counted once).
 *
 * <p>Read the numbers with the usual caution: {@link Thread#getAllStackTraces()} samples at JVM safepoints, so
 * time inside tight, counted loops is attributed to the nearest safepoint poll rather than exactly. It is good for
 * "which layer owns the time" (the parser-ng-sql bridge, parser-ng-parquet, Arrow, the engine) and for spotting a
 * hot allocation site; use JFR or async-profiler for line-level truth.
 */
final class StackSampler implements AutoCloseable {

    private static final Set<String> IDLE_THREADS = Set.of("Reference Handler", "Finalizer", "Signal Dispatcher",
            "Common-Cleaner", "Notification Thread", "Attach Listener", "Monitor Ctrl-Break");

    private final long intervalMs;
    private final Thread thread;
    private volatile boolean running;
    private int samples;
    private int stacks;
    private final Map<String, Integer> self = new HashMap<>();
    private final Map<String, Integer> inclusive = new HashMap<>();
    private final Map<String, Integer> byThread = new HashMap<>();
    private final Map<String, Integer> layerSelf = new HashMap<>();
    private final Map<String, Integer> layerIncl = new HashMap<>();

    /** Package prefixes of the layers of a ParquetSql query, first match wins (more specific prefixes first). */
    private static final String[][] LAYERS = {
            {"com.github.gbenroscience.sqlv1.demo", "0 benchmark harness (baselines, fixture, result checks)"},
            {"com.github.gbenroscience.sqlv1.parquet", "1 bridge: parser-ng-sql ParquetSql (rewriters, kernels, fold, gather)"},
            {"com.github.gbenroscience.sqlv1", "2 SQL engine: parser-ng-sql ArrowQuery"},
            {"com.github.gbenroscience.parser.ng.parquet", "3 scan: parser-ng-parquet"},
            {"com.github.gbenroscience.arrow", "4 expression engine: parser-ng-arrow"},
            {"com.github.gbenroscience.simd", "4 expression engine: parser-ng-simd"},
            {"com.github.gbenroscience.parser", "4 expression engine: ParserNG"},
            {"org.apache.arrow", "5 Arrow (vectors, memory)"},
            {"org.apache.parquet", "6 parquet-java (footer, codecs)"},
            {"jdk.incubator.vector", "7 Vector API"},
            {"java.", "8 JDK"}, {"jdk.", "8 JDK"}, {"sun.", "8 JDK"}, {"io.netty", "9 netty / other"}};

    private static String layerOf(String className) {
        for (String[] l : LAYERS) if (className.startsWith(l[0])) return l[1];
        return "9 other";
    }

    StackSampler(long intervalMs) {
        this.intervalMs = Math.max(1, intervalMs);
        this.thread = new Thread(this::loop, "stack-sampler");
        this.thread.setDaemon(true);
    }

    StackSampler start() {
        running = true;
        thread.start();
        return this;
    }

    private void loop() {
        Thread me = Thread.currentThread();
        while (running) {
            Map<Thread, StackTraceElement[]> all = Thread.getAllStackTraces();
            synchronized (this) {
                samples++;
                for (Map.Entry<Thread, StackTraceElement[]> e : all.entrySet()) {
                    Thread t = e.getKey();
                    StackTraceElement[] st = e.getValue();
                    if (t == me || st.length == 0 || t.getState() != Thread.State.RUNNABLE || IDLE_THREADS.contains(t.getName())) continue;
                    stacks++;
                    byThread.merge(groupOf(t.getName()), 1, Integer::sum);
                    self.merge(name(st[0]), 1, Integer::sum);
                    layerSelf.merge(layerOf(st[0].getClassName()), 1, Integer::sum);
                    Set<String> seen = new HashSet<>();
                    Set<String> seenLayers = new HashSet<>();
                    for (StackTraceElement f : st) {
                        String n = name(f);
                        if (seen.add(n)) inclusive.merge(n, 1, Integer::sum);
                        if (seenLayers.add(layerOf(f.getClassName()))) layerIncl.merge(layerOf(f.getClassName()), 1, Integer::sum);
                    }
                }
            }
            try {
                Thread.sleep(intervalMs);
            } catch (InterruptedException stop) {
                return;
            }
        }
    }

    private static String name(StackTraceElement f) {
        return f.getClassName() + "." + f.getMethodName();
    }

    private static String groupOf(String threadName) {
        return threadName.replaceAll("[-_ ]?\\d+$", "").replaceAll("\\d+", "#");
    }

    @Override
    public void close() {
        running = false;
        thread.interrupt();
        try {
            thread.join(1000);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * @param top    rows per table
     * @param prefix inclusive table keeps only frames of this package prefix (the code you own), or all if empty
     */
    synchronized String report(int top, String prefix) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT, "%d sampling ticks, %d runnable stacks (%.1f per tick)%n", samples, stacks,
                samples == 0 ? 0.0 : stacks / (double) samples));
        sb.append(sortedTable("Self time by layer (the layer of the top frame)", layerSelf, 20, stacks));
        sb.append(sortedTable("Inclusive time by layer (a layer counts if any frame of it is on the stack)", layerIncl, 20, stacks));
        sb.append(table("Where the time is (self: the top frame of a runnable stack)", self, top, "", stacks));
        sb.append(table("Which methods are on the stack (inclusive)" + (prefix.isEmpty() ? "" : ", only " + prefix + "*"),
                inclusive, top, prefix, stacks));
        sb.append(table("Which threads were running", byThread, top, "", stacks));
        return sb.toString();
    }

    private static String sortedTable(String title, Map<String, Integer> m, int top, int total) {
        return table(title, m, top, "", total);
    }

    private static String table(String title, Map<String, Integer> m, int top, String prefix, int total) {
        List<Map.Entry<String, Integer>> rows = new ArrayList<>();
        for (Map.Entry<String, Integer> e : m.entrySet()) if (e.getKey().startsWith(prefix)) rows.add(e);
        rows.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));
        StringBuilder sb = new StringBuilder("\n" + title + "\n");
        for (int i = 0; i < Math.min(top, rows.size()); i++) {
            sb.append(String.format(Locale.ROOT, "  %5.1f%%  %s%n", 100.0 * rows.get(i).getValue() / Math.max(1, total), rows.get(i).getKey()));
        }
        return sb.toString();
    }

    /** Self-test with a made-up workload: {@code java ...StackSampler}. */
    public static void main(String[] args) {
        try (StackSampler s = new StackSampler(2).start()) {
            long end = System.nanoTime() + 600_000_000L;
            double acc = 0;
            while (System.nanoTime() < end) acc += hot(1000) + warm(200);
            System.out.println("checksum " + (acc != 0.0));
            System.out.println(s.report(5, "com.github.gbenroscience"));
        }
    }

    private static double hot(int n) {
        double x = 0;
        for (int i = 1; i < n; i++) x += Math.sqrt(i) * Math.sin(i);
        return x;
    }

    private static double warm(int n) {
        double x = 0;
        for (int i = 1; i < n; i++) x += Math.cos(i);
        return x;
    }
}
