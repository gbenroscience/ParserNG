package com.github.gbenroscience.parser.ng.parquet.v1.internal.codec;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Known-answer tests for the pure-Java {@link ZstdDecompressor}, deliberately independent of any native zstd
 * library: the frames below were produced by the reference libzstd and are embedded as hex, and the expected
 * content is regenerated here by a tiny deterministic generator (also implemented in the script that produced the
 * frames). That independence is the point: this option exists for environments without {@code zstd-jni}, so
 * its tests must pass there too. The wide randomized conformance corpus (all levels and strategies, streaming
 * frames, checksums, dictionaries, corrupted input) lives with the standalone decoder this class is derived
 * from; these cover the integration-relevant shapes.
 */
class ZstdDecompressorTest {

    private static final String[] WORDS = {"alpha", "beta", "gamma", "delta", "parquet", "column", "row", "group",
            "page", "zstd", "decode", "java"};

    /** Deterministic text: a 31-bit LCG picks words; roughly every 8th word ends the line. */
    static byte[] textLike(int n, int seed) {
        StringBuilder sb = new StringBuilder();
        long x = seed;
        while (sb.length() < n) {
            x = (x * 1103515245L + 12345L) & 0x7fffffffL;
            sb.append(WORDS[(int) ((x >> 8) % WORDS.length)]).append(((x >> 3) & 7) == 0 ? "\n" : " ");
        }
        return sb.substring(0, n).getBytes(StandardCharsets.US_ASCII);
    }

    /** A period-251 ramp for the first 6000 bytes (long matches at one fixed offset), then zeros (an RLE-like tail). */
    static byte[] patterned(int n) {
        byte[] b = new byte[n];
        for (int i = 0; i < 6000 && i < n; i++) b[i] = (byte) (i % 251);
        return b;
    }

    // text(3000, seed 7), level 3, content checksum on: Huffman literals + FSE-coded sequences
    static final String TEXT_FRAME =
            "28b52ffd64b80a951600c2061316a0a5e90058b3fde993fdd68efb36b1a9aa6af4079e21e8e7fff3cb38ffcbfef264911b8a" +
            "b046894f71c866d91f876a9428408da9914428f038d4c0292deb61aa62807ad8548a50608a72438152a8619b8264d83b3104" +
            "08434196e5560f116020a82493d03685760361cf795efaf055d5ee3da5c501547be6a3821182aab0befc2f6b624a3a567a0c" +
            "a11f4501b4316ac6064c9574c229210b06e72ce94c24e3d99e6a1c5b2f8ebf2aa07a56bbceded74b27eacf6523f4f73e8420" +
            "244224fedfe05749ad8f50cc44d42d937792fc32f07cb4a2ee7aaa44fd05525f686c11721565a4d8474d00d4800614ff928e" +
            "7963e2c50b5e4a3cd2ff590d97f0217e2082ac881cf4988637c5dbb8462ffae099ee0115c518749b96f1958795c1c2a060d0" +
            "1ff5d91cfc008231f85526202bd15e3b227ce6db87f900d645d0a2c4f3cc7e000a83680812ac940c79d2280aa76de3bad017" +
            "1c44d478a08987bf954cd227c740c9444dff3aca2309e444df6453d0b1bdb17ecc5c502fff51e137eed2080231f8311a3430" +
            "592643330709404583633126e6d92ea893f721856c67ba19d52d51f8b21f954d452758864df29f5ac2b388ec685f46167770" +
            "d63263b1997438b569ac0c0bf676005e37f25e7b0630dcd13af55071bc65b01434c0c0c2fbeea495a929464ed0c3cd9f8021" +
            "6d49fbf4931fa8adc9c518ed8e79b777dc2093e225da9564e07abbf4605d83cee505678e771a1066b8c018f3d821c8875e33" +
            "2699d3212d354e6958d17f9a42226a970ef8ab1bd010a45fe38148ca1859b48d20a88d3399dda522ce7b60dc79cd05d4ca49" +
            "fe5b6054e97d599f40902ae0b1b060ebb98266eb4e20934613efae15e0b10140b005c28d0962a67b88003d9cb8d59519c68e" +
            "a37286d84050153ac32add98452ff82cb85b8df80b2a119b0681d67658528fa77703e60840ac1c4e806429a44882f8e5533f" +
            "8fc86b81b980d955b055cfb1f12954a0311e49fce18f127c6ac22944a5f110aa43345342";

    // patterned(8000), level 19: long repeat-offset matches
    static final String PATTERN_FRAME =
            "28b52ffd60401e4d0800c40f000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425" +
            "262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f5051525354555657" +
            "58595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f80818283848586878889" +
            "8a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babb" +
            "bcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebeced" +
            "eeeff0f1f2f3f4f5f6f7f8f9fa000200ccbb5fece5ccdf3f29";

    // frame(text(500, seed 1)) + skippable frame + frame(text(700, seed 2), with checksum)
    static final String CONCAT_FRAMES =
            "28b52ffd60f4006d060032851015b0256d00104bc88aed6ddbdee24d9a982485e94c7505dadb73d9baf5b944735954858606" +
            "88b123681ce1510605531d7d2edb424690479f70805f8c4681769f90573d204042e454da01e9806a99c3ca158a7b801ea0d8" +
            "8088071b692848e06ec0a3482805cc0b801a5843401fa8067833c8031cf784948832f8980cdc558b900ac00e240d5e7747e2" +
            "ff0bd40e802f1009a30cd2b500a160f705ab0fc054e0d801f3370be2864ae06bc0f4528b803ad19711440877095a0c702960" +
            "0608728164a212108a1aedda10bc0b502a4d180500000068656c6c6f28b52ffd64bc0125070012861216b0256d00d826b291" +
            "95c4763b3b926c629214a6131d0178b661828ee866bcd1ad251a14531362ca1f8131d6003cd779af3a2f506571f4662c9e62" +
            "acaf32bc1a20e7ca501b265a591c0152a851df6a65ff3320428320e40d102006194d9b3582cd453e6b27101d365084c73d91" +
            "74429d84276581b4b8a742ce1911a8d50a88025fbb583b1691c18f60a2fcfb0694e7d6f07ff891ef2300994087068337c9de" +
            "762bbe89796f4e4c5107f76264da4e0680602888394445977cfcd0292cb8a5004254e560df3026554ca1c257a0175d4bc443" +
            "f88c369765849fb83cdb37510098ce2a64fba6ae";

    static byte[] hex(String h) {
        byte[] out = new byte[h.length() / 2];
        for (int i = 0; i < out.length; i++) out[i] = (byte) Integer.parseInt(h.substring(2 * i, 2 * i + 2), 16);
        return out;
    }

    private static byte[] decodeWith(ZstdDecompressor d, byte[] frame, int size) {
        byte[] out = new byte[size];
        assertEquals(size, d.decompress(frame, 0, frame.length, out, 0, size), "decoded length");
        return out;
    }

    @Test
    void decodesTextFrameWithChecksum() {
        assertArrayEquals(textLike(3000, 7), decodeWith(new ZstdDecompressor(), hex(TEXT_FRAME), 3000));
    }

    @Test
    void decodesLongMatchesAndRunsAtLevel19() {
        assertArrayEquals(patterned(8000), decodeWith(new ZstdDecompressor(), hex(PATTERN_FRAME), 8000));
    }

    @Test
    void decodesConcatenatedFramesAroundASkippableFrame() {
        byte[] expected = new byte[1200];
        System.arraycopy(textLike(500, 1), 0, expected, 0, 500);
        System.arraycopy(textLike(700, 2), 0, expected, 500, 700);
        assertArrayEquals(expected, decodeWith(new ZstdDecompressor(), hex(CONCAT_FRAMES), 1200));
    }

    @Test
    void reportsDeclaredContentSizeAndMinusOneWhenUnknown() {
        byte[] f = hex(TEXT_FRAME);
        assertEquals(3000, ZstdDecompressor.frameContentSize(f, 0, f.length));
        assertEquals(-1, ZstdDecompressor.frameContentSize(new byte[]{1, 2, 3, 4, 5, 6}, 0, 6), "not a zstd frame");
        assertEquals(-1, ZstdDecompressor.frameContentSize(new byte[2], 0, 2), "too short");
    }

    @Test
    void decodesFromANonZeroOffsetIntoANonZeroOffsetAndStaysInsideTheWindow() {
        byte[] f = hex(TEXT_FRAME);
        byte[] src = new byte[f.length + 37];
        System.arraycopy(f, 0, src, 11, f.length);
        byte[] dst = new byte[3000 + 29];
        Arrays.fill(dst, (byte) 0x55);
        int n = new ZstdDecompressor().decompress(src, 11, f.length, dst, 5, 3000);
        assertEquals(3000, n);
        assertArrayEquals(textLike(3000, 7), Arrays.copyOfRange(dst, 5, 3005));
        for (int i = 0; i < 5; i++) assertEquals((byte) 0x55, dst[i], "byte " + i + " is before the window");
        for (int i = 3005; i < dst.length; i++) assertEquals((byte) 0x55, dst[i], "byte " + i + " is after the window");
    }

    @Test
    void anUndersizedDestinationIsAnErrorNotAnOverrun() {
        byte[] f = hex(TEXT_FRAME);
        byte[] small = new byte[2999];
        assertThrows(ZstdDecompressor.ZstdException.class,
                () -> new ZstdDecompressor().decompress(f, 0, f.length, small, 0, small.length));
    }

    @Test
    void aFlippedByteNeverYieldsWrongDataSilently() {
        // Not "every flip is detected": some bits are inert (padding, redundant encodings) and the reference libzstd
        // accepts those too, with identical output. The invariant that matters is that corruption can never come back
        // as WRONG data without an error. This frame carries an XXH64 content checksum, so a flip that changes the
        // output is caught by it, and a flip that desyncs the entropy streams is caught earlier.
        byte[] f = hex(TEXT_FRAME);
        byte[] expected = textLike(3000, 7);
        int rejected = 0;
        for (int pos = 6; pos < f.length; pos++) {
            byte[] bad = f.clone();
            bad[pos] ^= 0x40;
            byte[] out = new byte[3000];
            try {
                int n = new ZstdDecompressor().decompress(bad, 0, bad.length, out, 0, out.length);
                assertEquals(3000, n);
                assertArrayEquals(expected, out, "flip at byte " + pos + " was accepted AND changed the output");
            } catch (ZstdDecompressor.ZstdException e) {
                rejected++;
            }
        }
        assertTrue(rejected > f.length * 9 / 10, "almost every flip should be rejected, got " + rejected + "/" + f.length);
    }

    @Test
    void truncationAtEveryLengthIsAnErrorNeverAHangOrACrash() {
        byte[] f = hex(TEXT_FRAME);
        for (int len = 1; len < f.length; len++) {
            byte[] out = new byte[3000];
            try {
                new ZstdDecompressor().decompress(f, 0, len, out, 0, out.length);
                fail("a " + len + "-byte prefix of a " + f.length + "-byte frame decoded without error");
            } catch (ZstdDecompressor.ZstdException expected) {
                // good
            }
        }
    }

    @Test
    void garbageIsRejectedAndTheDecoderStaysUsable() {
        ZstdDecompressor d = new ZstdDecompressor();
        byte[] junk = new byte[200];
        new java.util.Random(1).nextBytes(junk);
        assertThrows(ZstdDecompressor.ZstdException.class, () -> d.decompress(junk, 0, junk.length, new byte[1000], 0, 1000));
        // one decoder instance serves every page of a scan: a failure must not poison it
        assertArrayEquals(textLike(3000, 7), decodeWith(d, hex(TEXT_FRAME), 3000));
    }

    @Test
    void oneInstanceDecodesManyFramesBackToBack() {
        ZstdDecompressor d = new ZstdDecompressor();
        for (int i = 0; i < 50; i++) {
            assertArrayEquals(textLike(3000, 7), decodeWith(d, hex(TEXT_FRAME), 3000));
            assertArrayEquals(patterned(8000), decodeWith(d, hex(PATTERN_FRAME), 8000));
        }
    }
}
