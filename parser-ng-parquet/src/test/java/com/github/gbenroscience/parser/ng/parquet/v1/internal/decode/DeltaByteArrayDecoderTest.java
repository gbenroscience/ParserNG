package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.DeltaTestUtil.*;
import static org.junit.jupiter.api.Assertions.*;

class DeltaByteArrayDecoderTest {

    private static String value(MaterializedBinary mb, int i) {
        return new String(mb.data, mb.offsets[i], mb.offsets[i + 1] - mb.offsets[i], StandardCharsets.US_ASCII);
    }

    // ------------------------------------------------------------ DELTA_LENGTH_BYTE_ARRAY

    @Test
    void deltaLengthByteArray_specExample() throws IOException {
        // Encodings.md: "if the data was 'Hello', 'World', 'Foobar', 'ABCDEF': the encoded data would be
        // DeltaEncoding(5, 5, 6, 6) 'HelloWorldFoobarABCDEF'".
        String[] vals = {"Hello", "World", "Foobar", "ABCDEF"};
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        long[] lens = new long[vals.length];
        for (int i = 0; i < vals.length; i++) lens[i] = vals[i].length();
        out.write(encodeDeltaBinaryPacked(8, 1, lens));
        for (String s : vals) out.write(s.getBytes(StandardCharsets.US_ASCII));
        byte[] raw = out.toByteArray();

        ByteReader br = reader(raw);
        MaterializedBinary mb = new MaterializedBinary();
        DeltaByteArrayDecoder.decodeLengthByteArray(br, vals.length, mb);

        for (int i = 0; i < vals.length; i++) assertEquals(vals[i], value(mb, i));
        assertEquals(raw.length, br.pos);
    }

    @Test
    void deltaLengthByteArray_includesEmptyStrings() throws IOException {
        String[] vals = {"", "x", "", "yz", ""};
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        long[] lens = new long[vals.length];
        for (int i = 0; i < vals.length; i++) lens[i] = vals[i].length();
        out.write(encodeDeltaBinaryPacked(8, 1, lens));
        for (String s : vals) out.write(s.getBytes(StandardCharsets.US_ASCII));
        byte[] raw = out.toByteArray();

        ByteReader br = reader(raw);
        MaterializedBinary mb = new MaterializedBinary();
        DeltaByteArrayDecoder.decodeLengthByteArray(br, vals.length, mb);
        for (int i = 0; i < vals.length; i++) assertEquals(vals[i], value(mb, i));
    }

    // ------------------------------------------------------------ DELTA_BYTE_ARRAY (incremental encoding)

    @Test
    void deltaByteArray_incrementalEncodingOnSortedStrings() throws IOException {
        String[] vals = {"apple", "application", "apply", "banana", "band", "bandana"};
        int[] prefixLens = new int[vals.length];
        String[] suffixes = new String[vals.length];
        String prev = "";
        for (int i = 0; i < vals.length; i++) {
            int p = 0, max = Math.min(prev.length(), vals[i].length());
            while (p < max && prev.charAt(p) == vals[i].charAt(p)) p++;
            prefixLens[i] = p;
            suffixes[i] = vals[i].substring(p);
            prev = vals[i];
        }
        long[] pl = new long[vals.length], sl = new long[vals.length];
        for (int i = 0; i < vals.length; i++) { pl[i] = prefixLens[i]; sl[i] = suffixes[i].length(); }

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(encodeDeltaBinaryPacked(8, 1, pl));
        out.write(encodeDeltaBinaryPacked(8, 1, sl));
        for (String s : suffixes) out.write(s.getBytes(StandardCharsets.US_ASCII));
        byte[] raw = out.toByteArray();

        ByteReader br = reader(raw);
        MaterializedBinary mb = new MaterializedBinary();
        DeltaByteArrayDecoder.decodeByteArray(br, vals.length, mb);

        for (int i = 0; i < vals.length; i++) assertEquals(vals[i], value(mb, i));
        assertEquals(raw.length, br.pos);
    }

    @Test
    void deltaByteArray_firstValueHasNoPrefix() throws IOException {
        String[] vals = {"standalone"};
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(encodeDeltaBinaryPacked(8, 1, new long[]{0}));
        out.write(encodeDeltaBinaryPacked(8, 1, new long[]{vals[0].length()}));
        out.write(vals[0].getBytes(StandardCharsets.US_ASCII));
        byte[] raw = out.toByteArray();

        ByteReader br = reader(raw);
        MaterializedBinary mb = new MaterializedBinary();
        DeltaByteArrayDecoder.decodeByteArray(br, 1, mb);
        assertEquals("standalone", value(mb, 0));
    }

    @Test
    void deltaByteArray_prefixLongerThanPreviousValueIsRejected() throws IOException {
        // Corrupt stream: claims a 5-char prefix of a value that was only 3 chars long.
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(encodeDeltaBinaryPacked(8, 1, new long[]{0, 5}));   // prefix lengths: 0, then 5 (invalid)
        out.write(encodeDeltaBinaryPacked(8, 1, new long[]{3, 1}));   // suffix lengths: 3, then 1
        out.write("abc".getBytes(StandardCharsets.US_ASCII));
        out.write("d".getBytes(StandardCharsets.US_ASCII));
        byte[] raw = out.toByteArray();

        ByteReader br = reader(raw);
        MaterializedBinary mb = new MaterializedBinary();
        assertThrows(ByteReader.Corrupt.class, () -> DeltaByteArrayDecoder.decodeByteArray(br, 2, mb));
    }
}
