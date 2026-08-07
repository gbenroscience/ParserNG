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
package com.github.gbenroscience.gpu.llm.opencl;

/**
 *
 * @author GBEMIRO
 */
  
import com.github.gbenroscience.simd.turbo.tools.llm.loader.GGUFLoader;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-level BPE tokenizer -- the GPT-2 / Llama-3 / Mistral / Falcon style
 * (GGUF's "tokenizer.ggml.model" == "gpt2"). Implements the standard
 * byte-to-unicode remapping + iterative lowest-rank-pair merging algorithm
 * GPT-2 introduced and most modern open-weight models still use.
 *
 * WHAT THIS DOES NOT COVER: SentencePiece UNIGRAM tokenizers (GGUF
 * "tokenizer.ggml.model" == "llama", used by the original Llama 1/2
 * family). Unigram is a fundamentally different algorithm --
 * probability-scored Viterbi segmentation against tokenizer.ggml.scores,
 * not merge-rank BPE against tokenizer.ggml.merges -- and is NOT
 * implemented here. fromGguf below checks tokenizer.ggml.model and
 * throws rather than silently mis-tokenizing against the wrong algorithm.
 *
 * fromGguf(GGUFLoader.GGUFFile) reads directly from GGUFFile.metadata --
 * a Map<String,Object> distinct from GGUFFile.tensors -- for
 * tokenizer.ggml.tokens / .merges / .bos_token_id / .eos_token_id /
 * .unknown_token_id. The plain-data constructor (String[] vocab,
 * String[] merges, bosId, eosId, unkId) remains available directly for
 * callers not going through a GGUF file at all (tests, non-GGUF vocab
 * sources).
 */
public final class BpeTokenizer {

    // GPT-2's own pre-tokenization regex, simplified to drop the
    // case-insensitive-contraction variant ((?i:'s|'t|...)) -- this
    // version only matches lowercase contractions ('s, 't, 're, ...).
    // Uppercase contractions (e.g. "IT'S") will fall through to the
    // general punctuation/letter branches instead of being split at the
    // apostrophe the same way GPT-2's reference implementation does;
    // rare in practice and does not break tokenization, just diverges
    // slightly from the reference split for that specific case.
    private static final Pattern PRETOKENIZE_PATTERN = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final String[] idToToken;
    private final Map<String, Integer> tokenToId;
    private final Map<String, Integer> mergeRank; // "left right" -> priority (lower = merge first)
    private final Map<String, List<String>> bpeCache = new HashMap<>();

    private final Map<Integer, Character> byteEncoder; // raw byte -> unicode char
    private final Map<Character, Integer> byteDecoder;  // unicode char -> raw byte

    private final int bosId;
    private final int eosId;
    private final int unkId;

    public BpeTokenizer(String[] vocabTokens, String[] merges, int bosId, int eosId, int unkId) {
        this.idToToken = vocabTokens.clone();
        this.tokenToId = new HashMap<>(vocabTokens.length * 2);
        for (int i = 0; i < vocabTokens.length; i++) {
            tokenToId.put(vocabTokens[i], i);
        }

        this.mergeRank = new HashMap<>(merges.length * 2);
        for (int i = 0; i < merges.length; i++) {
            mergeRank.put(merges[i], i);
        }

        this.byteEncoder = buildByteToUnicode();
        this.byteDecoder = new HashMap<>();
        for (Map.Entry<Integer, Character> e : byteEncoder.entrySet()) {
            byteDecoder.put(e.getValue(), e.getKey());
        }

        this.bosId = bosId;
        this.eosId = eosId;
        this.unkId = unkId;
    }

    /**
     * Builds the tokenizer straight from a loaded GGUF file's metadata
     * section (GGUFFile.metadata, a Map<String,Object> distinct from the
     * tensor map) -- reads tokenizer.ggml.tokens / .merges /
     * .bos_token_id / .eos_token_id / .unknown_token_id, and checks
     * tokenizer.ggml.model to fail loudly rather than silently
     * mis-tokenize if the file is a SentencePiece unigram model (see
     * class javadoc -- that algorithm isn't implemented here).
     *
     * GGUFLoader's array-typed metadata values come back as List<Object>
     * (each element itself an Object -- a String for both tokens and
     * merges here); scalar ids come back as whichever boxed Number type
     * matches their GGUF wire type (UINT32 -> Long, INT32 -> Integer,
     * etc. -- see GGUFLoader.readValue), so both are read via Number
     * rather than assuming Integer specifically.
     */
    public static BpeTokenizer fromGguf(GGUFLoader.GGUFFile gguf) {
        Object modelType = gguf.metadata.get("tokenizer.ggml.model");
        if (modelType != null && !"gpt2".equals(String.valueOf(modelType))) {
            throw new IllegalArgumentException(
                    "GGUF tokenizer.ggml.model=\"" + modelType + "\" -- this BpeTokenizer only implements "
                            + "the byte-level BPE algorithm (GGUF's \"gpt2\" tokenizer type). SentencePiece "
                            + "unigram (\"llama\") models need a different implementation -- see class javadoc.");
        }

        Object tokensRaw = gguf.metadata.get("tokenizer.ggml.tokens");
        if (!(tokensRaw instanceof List)) {
            throw new NoSuchElementException("GGUF metadata missing or malformed key: tokenizer.ggml.tokens");
        }
        List<?> tokensList = (List<?>) tokensRaw;
        String[] vocab = new String[tokensList.size()];
        for (int i = 0; i < vocab.length; i++) {
            vocab[i] = String.valueOf(tokensList.get(i));
        }

        Object mergesRaw = gguf.metadata.get("tokenizer.ggml.merges");
        String[] merges;
        if (mergesRaw instanceof List) {
            List<?> mergesList = (List<?>) mergesRaw;
            merges = new String[mergesList.size()];
            for (int i = 0; i < merges.length; i++) {
                merges[i] = String.valueOf(mergesList.get(i));
            }
        } else {
            merges = new String[0]; // no merges -> every piece stays split at the byte level
        }

        int bosId = intMetadata(gguf, "tokenizer.ggml.bos_token_id", -1);
        int eosId = intMetadata(gguf, "tokenizer.ggml.eos_token_id", -1);
        int unkId = intMetadata(gguf, "tokenizer.ggml.unknown_token_id", -1);

        return new BpeTokenizer(vocab, merges, bosId, eosId, unkId);
    }

    private static int intMetadata(GGUFLoader.GGUFFile gguf, String key, int defaultValue) {
        Object v = gguf.metadata.get(key);
        return (v instanceof Number) ? ((Number) v).intValue() : defaultValue;
    }

    public int getBosId() {
        return bosId;
    }

    public int getEosId() {
        return eosId;
    }

    public int getUnkId() {
        return unkId;
    }

    public int vocabSize() {
        return idToToken.length;
    }

    public int[] encode(String text, boolean addBos) {
        List<Integer> ids = new ArrayList<>();
        if (addBos && bosId >= 0) {
            ids.add(bosId);
        }

        Matcher m = PRETOKENIZE_PATTERN.matcher(text);
        while (m.find()) {
            String piece = m.group();
            StringBuilder byteMapped = new StringBuilder(piece.length() * 2);
            for (byte b : piece.getBytes(StandardCharsets.UTF_8)) {
                byteMapped.append(byteEncoder.get(b & 0xFF));
            }

            for (String token : bpeMerge(byteMapped.toString())) {
                Integer id = tokenToId.get(token);
                if (id != null) {
                    ids.add(id);
                } else if (unkId >= 0) {
                    ids.add(unkId);
                }
                // else: silently drop an unmappable piece rather than
                // inserting a wrong id -- callers relying on exact
                // round-tripping should check vocab coverage up front.
            }
        }

        int[] result = new int[ids.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = ids.get(i);
        }
        return result;
    }

    public String decode(int[] ids) {
        StringBuilder byteMapped = new StringBuilder();
        for (int id : ids) {
            if (id == bosId || id == eosId) {
                continue;
            }
            if (id < 0 || id >= idToToken.length) {
                continue;
            }
            byteMapped.append(idToToken[id]);
        }

        byte[] bytes = new byte[byteMapped.length()];
        for (int i = 0; i < byteMapped.length(); i++) {
            Integer b = byteDecoder.get(byteMapped.charAt(i));
            bytes[i] = (byte) (int) (b != null ? b : 0);
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /** Iterative lowest-merge-rank pairing, standard BPE encode algorithm. Result cached per unique byte-mapped piece. */
    private List<String> bpeMerge(String byteMappedPiece) {
        List<String> cached = bpeCache.get(byteMappedPiece);
        if (cached != null) {
            return cached;
        }

        List<String> word = new ArrayList<>(byteMappedPiece.length());
        for (int i = 0; i < byteMappedPiece.length(); i++) {
            word.add(String.valueOf(byteMappedPiece.charAt(i)));
        }

        while (word.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i < word.size() - 1; i++) {
                Integer rank = mergeRank.get(word.get(i) + " " + word.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx == -1) {
                break; // no more applicable merges
            }
            List<String> next = new ArrayList<>(word.size() - 1);
            int i = 0;
            while (i < word.size()) {
                if (i == bestIdx) {
                    next.add(word.get(i) + word.get(i + 1));
                    i += 2;
                } else {
                    next.add(word.get(i));
                    i += 1;
                }
            }
            word = next;
        }

        bpeCache.put(byteMappedPiece, word);
        return word;
    }

    /** GPT-2's byte<->unicode remapping: printable/visible chars map to themselves, control/whitespace bytes map into a private-use-adjacent range so every byte has a distinct, mergeable, whitespace-free unicode representation. */
    private static Map<Integer, Character> buildByteToUnicode() {
        List<Integer> bytesList = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) {
            bytesList.add(i);
        }
        for (int i = 0xA1; i <= 0xAC; i++) {
            bytesList.add(i);
        }
        for (int i = 0xAE; i <= 0xFF; i++) {
            bytesList.add(i);
        }

        List<Integer> chars = new ArrayList<>(bytesList);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bytesList.contains(b)) {
                bytesList.add(b);
                chars.add(256 + n);
                n++;
            }
        }

        Map<Integer, Character> map = new HashMap<>();
        for (int i = 0; i < bytesList.size(); i++) {
            map.put(bytesList.get(i), (char) (int) chars.get(i));
        }
        return map;
    }
}