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
package com.github.gbenroscience.math.differentialcalculus.equations.coeffextractor.clext.common;

/**
 *
 * @author GBEMIRO
 */
import java.util.HashMap;
import java.util.Map;

/**
 * Resolves a ParserNG function-token name (e.g. {@code "sind"}, {@code
 * "arcsin"}, {@code "sin-¹"}, {@code "asin_grad"}) to a small {@code int}
 * opcode, exactly once, at {@link ExprNode} construction time.
 *
 * <h2>Why this exists</h2>
 * {@link ExprNodeCompiler}'s evaluator is called once per solver step —
 * potentially millions of times per solve — so it cannot afford a {@code
 * String} switch (hashCode + equals per call) on every node visit. The
 * fix mirrors what {@code VectorTurboEvaluator.compileToPrimitiveProgram}
 * already does for the Turbo tile pipeline: resolve every function name to
 * an {@code int} opcode <em>once</em>, up front, then have the actual
 * numeric evaluator (the part that runs per-step) switch on nothing but
 * {@code int}s. Here that one-time resolution happens inside {@link
 * ExprNode#func}, via a single {@code Map.get} — not a switch, and it
 * only ever runs once per node, at tree-build time.
 *
 * <h2>Alias source</h2>
 * The alias lists below are copied from {@code VectorTurboEvaluator}'s own
 * {@code compileToPrimitiveProgram} switch, since that is what defines the
 * literal strings ParserNG actually puts in {@code Token.name} for a given
 * function — word-form ({@code "arcsin"}), short-prefix form ({@code
 * "asin"}), symbolic form ({@code "sin-¹"}), and their {@code _deg}/{@code
 * _grad}/{@code d}/{@code g} suffixed variants. Aliases that compute
 * identical math (e.g. {@code "asin"} and {@code "arcsin"} and {@code
 * "sin-¹"}) are collapsed onto the same opcode here — {@code
 * VectorTurboEvaluator} keeps them as distinct opcodes (OP_ASIN vs
 * OP_ASIN_ALT) purely for its own bookkeeping; a scalar evaluator has no
 * use for that distinction since the computed value is the same.
 *
 * <h2>Two independent opcode namespaces</h2>
 * One-argument and two-argument opcodes are numbered independently
 * starting at 1 (e.g. one-arg {@code SQRT == 1} and two-arg {@code ATAN2
 * == 1} are unrelated numbers). This is safe because {@link ExprNode#func}
 * resolves against the map matching the node's actual arity
 * ({@code children.size()}), and {@link ExprNodeCompiler} always evaluates
 * a function node's opcode through the one-arg or two-arg switch that
 * matches how many children it actually has — the two switches are never
 * consulted for the same node.
 *
 * <h2>Unresolved names</h2>
 * A name with no entry in the relevant map (or a function called with an
 * arity nothing here supports) resolves to {@link #UNRESOLVED} rather than
 * throwing — this class stays a plain, non-throwing lookup table.
 * {@link ExprNodeCompiler}'s validation methods are what turn an
 * unresolved opcode into a compile-time error, keeping "what's supported"
 * and "how it's reported" in the one class that already owned that job.
 */
final class FunctionOpcodes {

    static final int UNRESOLVED = -1;

    // ------------------------------------------------------------------
    // One-argument opcodes
    // ------------------------------------------------------------------
    static final int SQRT = 1;
    static final int CBRT = 2;
    static final int EXP = 3;
    static final int LN = 4;
    static final int LOG10 = 5;
    static final int ABS = 6;
    static final int SINH = 7;
    static final int COSH = 8;
    static final int TANH = 9;
    static final int ASINH = 10;
    static final int ACOSH = 11;
    static final int ATANH = 12;
    static final int SIN = 13;
    static final int SIN_DEG = 14;
    static final int SIN_GRAD = 15;
    static final int COS = 16;
    static final int COS_DEG = 17;
    static final int COS_GRAD = 18;
    static final int TAN = 19;
    static final int TAN_DEG = 20;
    static final int TAN_GRAD = 21;
    static final int ASIN = 22;
    static final int ASIN_DEG = 23;
    static final int ASIN_GRAD = 24;
    static final int ACOS = 25;
    static final int ACOS_DEG = 26;
    static final int ACOS_GRAD = 27;
    static final int ATAN = 28;
    static final int ATAN_DEG = 29;
    static final int ATAN_GRAD = 30;
    static final int SEC = 31;
    static final int SEC_DEG = 32;
    static final int SEC_GRAD = 33;
    static final int CSC = 34;
    static final int CSC_DEG = 35;
    static final int CSC_GRAD = 36;
    static final int COT = 37;
    static final int COT_DEG = 38;
    static final int COT_GRAD = 39;
    static final int ASEC = 40;
    static final int ASEC_DEG = 41;
    static final int ASEC_GRAD = 42;
    static final int ACSC = 43;
    static final int ACSC_DEG = 44;
    static final int ACSC_GRAD = 45;
    static final int ACOT = 46;
    static final int ACOT_DEG = 47;
    static final int ACOT_GRAD = 48;
    static final int DIFF_EQN = 49;
    static final int DIFF_EQN_HO = 50;
    static final int DIFF_EQN_PATH = 51;
    static final int DIFF_EQN_PATH_HO = 52;

    // ------------------------------------------------------------------
    // Two-argument opcodes (independent namespace — see class javadoc)
    // ------------------------------------------------------------------
    static final int ATAN2 = 1;
    static final int LOG_BASE = 2;

    private static final Map<String, Integer> ONE_ARG = new HashMap<>();
    private static final Map<String, Integer> TWO_ARG = new HashMap<>();

    static {
        // --- non-angular ---
        alias(ONE_ARG, SQRT, "sqrt");
        alias(ONE_ARG, CBRT, "cbrt");
        alias(ONE_ARG, EXP, "exp");
        alias(ONE_ARG, LN, "log", "ln");
        alias(ONE_ARG, LOG10, "log10", "lg");
        alias(ONE_ARG, ABS, "abs");

        // --- hyperbolic (forward) ---
        alias(ONE_ARG, SINH, "sinh");
        alias(ONE_ARG, COSH, "cosh");
        alias(ONE_ARG, TANH, "tanh");

        // --- hyperbolic (inverse) — no MethodSack counterpart; Maths.* used at eval time ---
        alias(ONE_ARG, ASINH, "sinh-¹", "arcsinh", "asinh");
        alias(ONE_ARG, ACOSH, "cosh-¹", "arccosh", "acosh");
        alias(ONE_ARG, ATANH, "tanh-¹", "arctanh", "atanh");

        // --- standard trig ---
        alias(ONE_ARG, SIN, "sin", "sin_rad");
        alias(ONE_ARG, COS, "cos", "cos_rad");
        alias(ONE_ARG, TAN, "tan", "tan_rad");
        alias(ONE_ARG, SIN_DEG, "sin_deg", "sind");
        alias(ONE_ARG, COS_DEG, "cos_deg", "cosd");
        alias(ONE_ARG, TAN_DEG, "tan_deg", "tand");
        alias(ONE_ARG, SIN_GRAD, "sin_grad", "sing");
        alias(ONE_ARG, COS_GRAD, "cos_grad", "cosg");
        alias(ONE_ARG, TAN_GRAD, "tan_grad", "tang");

        // --- inverse trig (word-form, symbolic, and short-prefix aliases collapsed together) ---
        alias(ONE_ARG, ASIN, "sin-¹", "sin-¹_rad", "arcsin", "asin", "asin_rad", "arc_sin_alt");
        alias(ONE_ARG, ACOS, "cos-¹", "cos-¹_rad", "arccos", "acos", "acos_rad", "arc_cos_alt");
        alias(ONE_ARG, ATAN, "tan-¹", "tan-¹_rad", "arctan", "atan", "atan_rad", "arc_tan_alt");
        alias(ONE_ARG, ASIN_DEG, "sin-¹_deg", "arcsin_deg", "asin_deg", "asind", "arc_sin_alt_deg");
        alias(ONE_ARG, ACOS_DEG, "cos-¹_deg", "arccos_deg", "acos_deg", "acosd", "arc_cos_alt_deg");
        alias(ONE_ARG, ATAN_DEG, "tan-¹_deg", "arctan_deg", "atan_deg", "atand", "arc_tan_alt_deg");
        alias(ONE_ARG, ASIN_GRAD, "sin-¹_grad", "arcsin_grad", "asin_grad", "asing", "arc_sin_alt_grad");
        alias(ONE_ARG, ACOS_GRAD, "cos-¹_grad", "arccos_grad", "acos_grad", "acosg", "arc_cos_alt_grad");
        alias(ONE_ARG, ATAN_GRAD, "tan-¹_grad", "arctan_grad", "atan_grad", "atang", "arc_tan_alt_grad");

        // --- reciprocal trig ---
        alias(ONE_ARG, SEC, "sec", "sec_rad");
        alias(ONE_ARG, SEC_DEG, "sec_deg", "secd");
        alias(ONE_ARG, SEC_GRAD, "sec_grad");
        alias(ONE_ARG, CSC, "cosec", "csc", "csc_rad");
        alias(ONE_ARG, CSC_DEG, "cosec_deg", "cscd");
        alias(ONE_ARG, CSC_GRAD, "cosec_grad");
        alias(ONE_ARG, COT, "cot", "cot_rad");
        alias(ONE_ARG, COT_DEG, "cot_deg", "cotd");
        alias(ONE_ARG, COT_GRAD, "cot_grad");

        // --- inverse reciprocal trig ---
        alias(ONE_ARG, ASEC, "sec-¹", "sec-¹_rad", "arcsec", "asec", "asec_rad", "arc_sec_alt");
        alias(ONE_ARG, ASEC_DEG, "sec-¹_deg", "arcsec_deg", "asec_deg", "arc_sec_alt_deg");
        alias(ONE_ARG, ASEC_GRAD, "sec-¹_grad", "arcsec_grad", "asec_grad", "arc_sec_alt_grad");
        alias(ONE_ARG, ACSC, "csc-¹", "csc-¹_rad", "arccsc", "acsc", "acsc_rad", "arc_cosec_alt");
        alias(ONE_ARG, ACSC_DEG, "csc-¹_deg", "arccsc_deg", "acsc_deg", "arc_cosec_alt_deg");
        alias(ONE_ARG, ACSC_GRAD, "csc-¹_grad", "arccsc_grad", "acsc_grad", "arc_cosec_alt_grad");
        alias(ONE_ARG, ACOT, "cot-¹", "cot-¹_rad", "arccot", "acot", "acot_rad", "arc_cot_alt");
        alias(ONE_ARG, ACOT_DEG, "cot-¹_deg", "arccot_deg", "acot_deg", "arc_cot_alt_deg");
        alias(ONE_ARG, ACOT_GRAD, "cot-¹_grad", "arccot_grad", "acot_grad", "arc_cot_alt_grad");

        // --- two-argument ---
        alias(TWO_ARG, ATAN2, "atan2");
        alias(TWO_ARG, LOG_BASE, "log");
    }

    private FunctionOpcodes() {
    }

    private static void alias(Map<String, Integer> map, int opcode, String... names) {
        for (String name : names) {
            map.put(name, opcode);
        }
    }

    /** Resolves a one-argument function name to its opcode, or {@link #UNRESOLVED} if unsupported. */
    static int resolveOneArg(String name) {
        Integer opcode = ONE_ARG.get(name);
        return opcode != null ? opcode : UNRESOLVED;
    }

    /** Resolves a two-argument function name to its opcode, or {@link #UNRESOLVED} if unsupported. */
    static int resolveTwoArg(String name) {
        Integer opcode = TWO_ARG.get(name);
        return opcode != null ? opcode : UNRESOLVED;
    }
}