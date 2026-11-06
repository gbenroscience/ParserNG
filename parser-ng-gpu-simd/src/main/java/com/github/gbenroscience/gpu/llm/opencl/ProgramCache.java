package com.github.gbenroscience.gpu.llm.opencl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HexFormat;

/**
 * On-disk cache of compiled OpenCL program binaries. Compiling the kernel source (clBuildProgram) took 3.0-4.5 s on an
 * Intel HD 520 and 4.5-13.4 s on an AMD Hainan, on every start; the binary is a pure function of the device, driver and
 * source, so it is built once and reloaded afterwards.
 *
 * <ul>
 *   <li><b>Key</b>: SHA-256 of device name, vendor, driver version, device version and the complete source text, so a
 *       driver update or any kernel edit selects a different file.</li>
 *   <li><b>Integrity</b>: each file is {@code SHA-256(payload) || payload}, written to a temporary file and moved into
 *       place atomically; a short, truncated or altered file is rejected and deleted.</li>
 *   <li><b>Failure policy</b>: every problem here (no permission, bad file, driver refuses the binary) falls back to
 *       building from source. The cache can only make a start faster, never make it fail.</li>
 * </ul>
 * Switches: {@code -Dllm.programCache=false} disables it; {@code -Dllm.programCache.dir=PATH} relocates it (default
 * {@code ~/.parserng/opencl-cache}).
 */
final class ProgramCache {

    private static final int DIGEST_BYTES = 32;

    private ProgramCache() {
    }

    static boolean enabled() {
        String v = System.getProperty("llm.programCache");
        return v == null || !(v.equalsIgnoreCase("false") || v.equals("0") || v.equalsIgnoreCase("off"));
    }

    static Path dir() {
        String d = System.getProperty("llm.programCache.dir");
        return d != null && !d.isBlank() ? Paths.get(d) : Paths.get(System.getProperty("user.home", "."), ".parserng", "opencl-cache");
    }

    static String key(String deviceName, String vendor, String driver, String deviceVersion, String source) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            for (String part : new String[]{deviceName, vendor, driver, deviceVersion, source}) {
                byte[] b = part.getBytes(StandardCharsets.UTF_8);
                md.update(Integer.toString(b.length).getBytes(StandardCharsets.US_ASCII)); // length-prefix: no ambiguity between fields
                md.update((byte) ':');
                md.update(b);
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static Path file(String key) {
        return dir().resolve(key + ".clbin");
    }

    /** The cached binary, or null if there is none or it fails its integrity check (a bad file is deleted). */
    static byte[] read(String key) {
        Path f = file(key);
        try {
            if (!Files.isRegularFile(f)) {
                return null;
            }
            byte[] all = Files.readAllBytes(f);
            if (all.length <= DIGEST_BYTES) {
                delete(key);
                return null;
            }
            byte[] payload = Arrays.copyOfRange(all, DIGEST_BYTES, all.length);
            if (!MessageDigest.isEqual(Arrays.copyOfRange(all, 0, DIGEST_BYTES), MessageDigest.getInstance("SHA-256").digest(payload))) {
                delete(key);
                return null;
            }
            return payload;
        } catch (IOException | java.security.NoSuchAlgorithmException | RuntimeException e) {
            return null;
        }
    }

    /** Best effort; returns whether the file was written. */
    static boolean write(String key, byte[] binary) {
        try {
            Files.createDirectories(dir());
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(binary);
            byte[] all = new byte[DIGEST_BYTES + binary.length];
            System.arraycopy(digest, 0, all, 0, DIGEST_BYTES);
            System.arraycopy(binary, 0, all, DIGEST_BYTES, binary.length);
            Path tmp = Files.createTempFile(dir(), key, ".tmp");
            try {
                Files.write(tmp, all);
                Files.move(tmp, file(key), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                Files.deleteIfExists(tmp);
                throw e;
            }
            return true;
        } catch (IOException | java.security.NoSuchAlgorithmException | RuntimeException e) {
            return false;
        }
    }

    static void delete(String key) {
        try {
            Files.deleteIfExists(file(key));
        } catch (IOException | RuntimeException ignored) {
            // nothing more can be done
        }
    }
}
