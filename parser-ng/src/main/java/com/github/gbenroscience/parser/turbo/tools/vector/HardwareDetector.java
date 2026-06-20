package com.github.gbenroscience.parser.turbo.tools.vector;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Detects physical CPU cores across Windows/Mac/Linux.
 * Uses OS-native APIs, falls back to logical cores with HT heuristic.
 * Result is cached after first call.
 * * @author GBEMIRO
 */
public final class HardwareDetector {
    
    private static volatile int CACHED_CORES = 0;
    private static final long CMD_TIMEOUT_MS = 2000; // 2s timeout per command

    private HardwareDetector() {} // utility class

    public static int detectPhysicalCores() {
        // Fast path: return cached result
        int cached = CACHED_CORES;
        if (cached > 0) return cached;

        // 1. User override - fastest and most reliable
        String override = System.getProperty("turbo.cores");
        if (override != null) {
            try {
                cached = Math.max(1, Integer.parseInt(override.trim()));
                CACHED_CORES = cached;
                return cached;
            } catch (NumberFormatException ignored) {}
        }

        // 2. OS native detection
        String os = System.getProperty("os.name").toLowerCase();
        int cores = -1;
        try {
            if (os.contains("win")) {
                cores = getWindowsCores();
            } else if (os.contains("mac")) {
                cores = getMacCores();
            } else if (os.contains("nux") || os.contains("nix") || os.contains("aix")) {
                cores = getLinuxCores();
            }
        } catch (Exception e) {
            // SecurityManager, permissions, missing command, or timeout
            // Fall through silently to avoid spamming logs
        }

        // 3. Fallback: logical processors. If >8 and even, assume HT and halve
        if (cores <= 0) {
            cores = Runtime.getRuntime().availableProcessors();
            if (cores > 8 && cores % 2 == 0) {
                cores /= 2; // Heuristic for HT on big CPUs
            }
        }

        cached = Math.max(1, cores);
        CACHED_CORES = cached;
        return cached;
    }

    private static int getWindowsCores() throws Exception {
        // Try PowerShell first - works on Win10+Win11, handles multi-socket natively via sum aggregator
        String output = execToString(new String[]{
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum | Select-Object -ExpandProperty Sum"
        });
        
        if (output != null) {
            int cores = extractFirstInteger(output);
            if (cores > 0) return cores;
        }

        // Fallback to WMIC for older Win10 or restricted PowerShell execution policy
        output = execToString(new String[]{"wmic", "cpu", "get", "NumberOfCores"});
        if (output != null) {
            // Now safely processes all lines from multi-socket systems because the full output was captured
            return sumAllNumbers(output);
        }
        return -1;
    }

    private static int getMacCores() throws Exception {
        String output = execToString(new String[]{"sysctl", "-n", "hw.physicalcpu"});
        return output != null ? extractFirstInteger(output) : -1;
    }

    private static int getLinuxCores() throws Exception {
        String output = execToString(new String[]{"lscpu", "-p=Core,Socket"});
        if (output == null) return -1;

        Set<String> unique = new HashSet<>();
        for (String line : output.split("\\r?\\n")) {
            line = line.trim();
            // Skip comments and empty lines
            if (!line.isEmpty() && !line.startsWith("#")) {
                unique.add(line); // "core,socket" pair = unique physical core
            }
        }
        
        return unique.isEmpty() ? -1 : unique.size();
    }

    /**
     * Executes a command, enforces a true timeout, and reads the entire output safely.
     * Guaranteed never to deadlock on blocking I/O stream loops.
     */
    private static String execToString(String[] cmd) throws Exception {
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true); // Merge stderr to prevent pipe deadlock
        Process p = pb.start();

        // Enforce timeout FIRST. Safe because hardware inventory commands emit < 4KB of text
        // and easily fit inside the operating system's internal pipe buffer.
        boolean finished = p.waitFor(CMD_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        if (!finished) {
            p.destroyForcibly();
            return null; // Hard process kill on hang
        }

        // Process is guaranteed dead/finished here; read all data safely without blocking risk
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = r.readLine()) != null) {
                sb.append(line).append("\n");
            }
        }
        return sb.toString();
    }

    /**
     * Extracts the first isolated integer from a text block.
     */
    private static int extractFirstInteger(String text) {
        Matcher m = Pattern.compile("\\d+").matcher(text);
        if (m.find()) {
            try {
                return Integer.parseInt(m.group());
            } catch (NumberFormatException ignored) {}
        }
        return -1;
    }

    /**
     * Sums all integers found in multi-line output. Used for WMIC multi-socket configurations.
     */
    private static int sumAllNumbers(String multiline) {
        int totalSum = 0;
        for (String line : multiline.split("\\r?\\n")) {
            line = line.trim();
            if (!line.isEmpty() && Character.isDigit(line.charAt(0))) {
                try {
                    totalSum += Integer.parseInt(line);
                } catch (NumberFormatException ignored) {}
            }
        }
        return totalSum > 0 ? totalSum : -1;
    }
    
   
}