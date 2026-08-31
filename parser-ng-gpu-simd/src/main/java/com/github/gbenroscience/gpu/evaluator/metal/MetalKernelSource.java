package com.github.gbenroscience.gpu.evaluator.metal;

/**
 * A single Metal Shading Language kernel, {@code interpretF32}, compiled from
 * ONE source string via ONE {@code newLibraryWithSource:options:error:} call.
 * Ports VectorTurboEvaluator's / OpenClKernelSource's 101-opcode stack
 * machine, in single precision only.
 *
 * WHY THERE IS NO {@code interpret} (double) KERNEL HERE, UNLIKE
 * OpenClKernelSource: Metal Shading Language has no double-precision
 * floating point support on any Apple GPU. {@code double} is not a usable
 * MSL type for compute -- there is no compiler flag, extension, or
 * {@code #pragma} that turns it on, because the hardware has no fp64 compute
 * units to run it on (this is true across every GPU generation Metal
 * targets: Apple Silicon, and the AMD/Intel GPUs Metal drives on older
 * Intel Macs). This is a hardware/API ceiling, not a gap in this scaffold --
 * see MetalCompositeExpression's class javadoc for how the double-taking
 * methods on {@code GpuCompositeExpression} are handled given that ceiling.
 *
 * DIVERGENCES FROM OpenClKernelSource's F32 kernel body, and why:
 *  - No {@code cbrt} builtin exists in MSL. {@link #OPENCL_SOURCE} below
 *    defines {@code gpu_cbrt_f} as {@code sign(x) * pow(fabs(x), 1/3)} so
 *    negative inputs still produce a real (negative) cube root the way
 *    {@code cbrt} does, rather than the NaN a naive {@code pow(x, 1/3)}
 *    would give for negative x.
 *  - {@code erf} is NOT assumed to be present as an MSL builtin (Apple's
 *    published MSL builtin-function tables are inconsistent about it across
 *    OS/Metal versions). A explicit Abramowitz-Stegun 7.1.26 polynomial
 *    approximation ({@code gpu_erf_f}, max error ~1.5e-7) is used instead so
 *    the kernel compiles and gives consistent results on every Metal
 *    version, matching this scaffold's existing "verify against Maths.java"
 *    caveat below for the ops that are built out of it.
 *  - {@code asinh}/{@code acosh}/{@code atanh} are computed from the same
 *    explicit log-based formulas OpenClKernelSource uses, rather than relying
 *    on MSL builtins, purely for numerical-behavior parity between the two
 *    backends (both derive from the same textbook identities either way).
 *
 * VERIFIED against VectorTurboEvaluator.java source for opcode dispatch,
 * pop/push counts, and formulas for every op EXCEPT the five noted below,
 * which call into a {@code Maths} class not present in what was shared.
 * Those five use the standard/textbook formulas (identical in spirit to
 * OpenClKernelSource's) -- diff against your Maths.java before trusting
 * results for expressions using gelu/geglu/swiglu/erf/gelu_fast.
 *
 * - OP_ERF : polynomial erf() approximation (see gpu_erf_f), NOT exact
 * - OP_GELU : 0.5*x*(1+erf(x/sqrt(2))) [exact-form GELU, using the approx erf]
 * - OP_GELU_FAST : 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715x^3))) [tanh-approx GELU]
 * - OP_SWIGLU : x * sigmoid(x) [self-gated SiLU]
 * - OP_SWIGLU_2 : a * b * sigmoid(b) [SiLU(b) gating a]
 * - OP_GEGLU : x * gelu(x) [self-gated GELU]
 * - OP_GEGLU_2 : a * gelu(b) [GELU(b) gating a]
 */
public final class MetalKernelSource {

    private MetalKernelSource() {
    }

    public static final int MAX_STACK = 64;

    public static final String KERNEL_NAME_F32 = "interpretF32";

    public static final String METAL_SOURCE = """
        #include <metal_stdlib>
        using namespace metal;

        // ---- Shared opcode numbering (identical to OpenClKernelSource) ----
        #define OP_CONST 1
        #define OP_LOAD 2
        #define OP_ADD 3
        #define OP_SUB 4
        #define OP_MUL 5
        #define OP_DIV 6
        #define OP_POW 7
        #define OP_SIN 8
        #define OP_COS 9
        #define OP_TAN 10
        #define OP_SIN_DEG 11
        #define OP_COS_DEG 12
        #define OP_TAN_DEG 13
        #define OP_SIN_GRAD 14
        #define OP_COS_GRAD 15
        #define OP_TAN_GRAD 16
        #define OP_ASIN 17
        #define OP_ACOS 18
        #define OP_ATAN 19
        #define OP_ASIN_ALT 20
        #define OP_ACOS_ALT 21
        #define OP_ATAN_ALT 22
        #define OP_ASIN_DEG 23
        #define OP_ACOS_DEG 24
        #define OP_ATAN_DEG 25
        #define OP_ASIN_DEG_ALT 26
        #define OP_ACOS_DEG_ALT 27
        #define OP_ATAN_DEG_ALT 28
        #define OP_ASIN_GRAD 29
        #define OP_ACOS_GRAD 30
        #define OP_ATAN_GRAD 31
        #define OP_ASIN_GRAD_ALT 32
        #define OP_ACOS_GRAD_ALT 33
        #define OP_ATAN_GRAD_ALT 34
        #define OP_SEC 35
        #define OP_SEC_DEG 36
        #define OP_SEC_GRAD 37
        #define OP_COSEC 38
        #define OP_COSEC_DEG 39
        #define OP_COSEC_GRAD 40
        #define OP_COT 41
        #define OP_COT_DEG 42
        #define OP_COT_GRAD 43
        #define OP_ARC_SEC 44
        #define OP_ARC_SEC_DEG 45
        #define OP_ARC_SEC_GRAD 46
        #define OP_ARC_COSEC 47
        #define OP_ARC_COSEC_DEG 48
        #define OP_ARC_COSEC_GRAD 49
        #define OP_ARC_COT 50
        #define OP_ARC_COT_DEG 51
        #define OP_ARC_COT_GRAD 52
        #define OP_ARC_SIN_ALT 53
        #define OP_ARC_SIN_ALT_DEG 54
        #define OP_ARC_SIN_ALT_GRAD 55
        #define OP_ARC_COS_ALT 56
        #define OP_ARC_COS_ALT_DEG 57
        #define OP_ARC_COS_ALT_GRAD 58
        #define OP_ARC_TAN_ALT 59
        #define OP_ARC_TAN_ALT_DEG 60
        #define OP_ARC_TAN_ALT_GRAD 61
        #define OP_ARC_SEC_ALT 62
        #define OP_ARC_SEC_ALT_DEG 63
        #define OP_ARC_SEC_ALT_GRAD 64
        #define OP_ARC_COSEC_ALT 65
        #define OP_ARC_COSEC_ALT_DEG 66
        #define OP_ARC_COSEC_ALT_GRAD 67
        #define OP_ARC_COT_ALT 68
        #define OP_ARC_COT_ALT_DEG 69
        #define OP_ARC_COT_ALT_GRAD 70
        #define OP_SINH 71
        #define OP_COSH 72
        #define OP_TANH 73
        #define OP_ASINH 74
        #define OP_ACOSH 75
        #define OP_ATANH 76
        #define OP_ASINH_ALT 77
        #define OP_ACOSH_ALT 78
        #define OP_ATANH_ALT 79
        #define OP_ABS 80
        #define OP_EXP 81
        #define OP_SQRT 82
        #define OP_CBRT 83
        #define OP_LOG 84
        #define OP_LOG10 85
        #define OP_VMA 86
        #define OP_REM 87
        #define OP_IF 88
        #define OP_GT 89
        #define OP_LT 90
        #define OP_EQ 91
        #define OP_NE 92
        #define OP_GE 93
        #define OP_LE 94
        #define OP_GELU 95
        #define OP_GELU_FAST 96
        #define OP_GEGLU 97
        #define OP_SWIGLU 98
        #define OP_GEGLU_2 99
        #define OP_SWIGLU_2 100
        #define OP_ERF 101
        #define OP_AND 102
        #define OP_OR 103

        #define MAX_STACK 64

        // All literals carry an explicit f suffix -- MSL has no double, so
        // there is nothing to accidentally widen into even if a literal were
        // left bare, but this keeps the source visually parallel to
        // OpenClKernelSource's float kernel.
        #define DEG_TO_RAD_F 0.017453292f
        #define RAD_TO_DEG_F 57.29577951f
        #define GRAD_TO_RAD_F 0.015707963f
        #define RAD_TO_GRAD_F 63.66197723f

        // No native cbrt in MSL -- sign-preserving real cube root via pow.
        inline float gpu_cbrt_f(float x) {
            float s = (x < 0.0f) ? -1.0f : 1.0f;
            return s * pow(fabs(x), 1.0f / 3.0f);
        }

        // Abramowitz & Stegun 7.1.26 approximation, max abs error ~1.5e-7.
        // Not assumed to be provided by MSL itself -- see class javadoc.
        inline float gpu_erf_f(float x) {
            const float a1 =  0.254829592f;
            const float a2 = -0.284496736f;
            const float a3 =  1.421413741f;
            const float a4 = -1.453152027f;
            const float a5 =  1.061405429f;
            const float p  =  0.3275911f;
            float s = (x < 0.0f) ? -1.0f : 1.0f;
            float ax = fabs(x);
            float t = 1.0f / (1.0f + p * ax);
            float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
            return s * y;
        }

        inline float gpu_gelu_f(float x) {
            return 0.5f * x * (1.0f + gpu_erf_f(x * 0.70710678f));
        }
        inline float gpu_gelu_fast_f(float x) {
            float x3 = x * x * x;
            return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x3)));
        }
        inline float gpu_sigmoid_f(float x) {
            return 1.0f / (1.0f + exp(-x));
        }

        kernel void interpretF32(
            device const int* opcodes            [[buffer(0)]],
            device const int* targetSlots        [[buffer(1)]],
            device const float* literalConstants [[buffer(2)]],
            constant int& instructionCount       [[buffer(3)]],
            device const float* in               [[buffer(4)]],
            constant int& dataSize               [[buffer(5)]],
            constant int& varCount               [[buffer(6)]],
            device float* out                    [[buffer(7)]],
            uint gid                             [[thread_position_in_grid]])
        {
            if ((int)gid >= dataSize) {
                return;
            }

            float stack[MAX_STACK];
            int sp = 0;

            for (int i = 0; i < instructionCount; i++) {
                const int op = opcodes[i];

                switch (op) {
                    case OP_CONST:
                        stack[sp++] = literalConstants[i];
                        break;

                    case OP_LOAD:
                        stack[sp++] = in[targetSlots[i] * dataSize + (int)gid];
                        break;

                    case OP_ADD: { float b = stack[--sp]; stack[sp-1] = stack[sp-1] + b; break; }
                    case OP_SUB: { float b = stack[--sp]; stack[sp-1] = stack[sp-1] - b; break; }
                    case OP_MUL: { float b = stack[--sp]; stack[sp-1] = stack[sp-1] * b; break; }
                    case OP_DIV: { float b = stack[--sp]; stack[sp-1] = stack[sp-1] / b; break; }
                    case OP_POW: { float b = stack[--sp]; stack[sp-1] = pow(stack[sp-1], b); break; }
                    case OP_REM: { float b = stack[--sp]; stack[sp-1] = fmod(stack[sp-1], b); break; }

                    case OP_GT: { float b = stack[--sp]; stack[sp-1] = (stack[sp-1] >  b) ? 1.0f : 0.0f; break; }
                    case OP_LT: { float b = stack[--sp]; stack[sp-1] = (stack[sp-1] <  b) ? 1.0f : 0.0f; break; }
                    case OP_EQ: { float b = stack[--sp]; stack[sp-1] = (stack[sp-1] == b) ? 1.0f : 0.0f; break; }
                    case OP_NE: { float b = stack[--sp]; stack[sp-1] = (stack[sp-1] != b) ? 1.0f : 0.0f; break; }
                    case OP_GE: { float b = stack[--sp]; stack[sp-1] = (stack[sp-1] >= b) ? 1.0f : 0.0f; break; }
                    case OP_LE: { float b = stack[--sp]; stack[sp-1] = (stack[sp-1] <= b) ? 1.0f : 0.0f; break; }

                    case OP_AND: {
                        float right = stack[--sp];
                        stack[sp-1] = (stack[sp-1] != 0.0f && right != 0.0f) ? 1.0f : 0.0f;
                        break;
                    }
                    case OP_OR: {
                        float right = stack[--sp];
                        stack[sp-1] = (stack[sp-1] != 0.0f || right != 0.0f) ? 1.0f : 0.0f;
                        break;
                    }

                    case OP_SWIGLU_2: { float b = stack[--sp]; float a = stack[sp-1]; stack[sp-1] = a * b * gpu_sigmoid_f(b); break; }
                    case OP_GEGLU_2:  { float b = stack[--sp]; float a = stack[sp-1]; stack[sp-1] = a * gpu_gelu_f(b); break; }

                    case OP_VMA: {
                        float c = stack[--sp];
                        float b = stack[--sp];
                        stack[sp-1] = stack[sp-1] * b + c;
                        break;
                    }
                    case OP_IF: {
                        float f = stack[--sp];
                        float t = stack[--sp];
                        stack[sp-1] = (stack[sp-1] != 0.0f) ? t : f;
                        break;
                    }

                    case OP_SIN:  stack[sp-1] = sin(stack[sp-1]);  break;
                    case OP_COS:  stack[sp-1] = cos(stack[sp-1]);  break;
                    case OP_TAN:  stack[sp-1] = tan(stack[sp-1]);  break;
                    case OP_SINH: stack[sp-1] = sinh(stack[sp-1]); break;
                    case OP_COSH: stack[sp-1] = cosh(stack[sp-1]); break;
                    case OP_TANH: stack[sp-1] = tanh(stack[sp-1]); break;
                    case OP_ABS:  stack[sp-1] = fabs(stack[sp-1]); break;
                    case OP_EXP:  stack[sp-1] = exp(stack[sp-1]);  break;
                    case OP_SQRT: stack[sp-1] = sqrt(stack[sp-1]); break;
                    case OP_CBRT: stack[sp-1] = gpu_cbrt_f(stack[sp-1]); break;
                    case OP_LOG:  stack[sp-1] = log(stack[sp-1]);  break;
                    case OP_LOG10: stack[sp-1] = log10(stack[sp-1]); break;

                    case OP_ASIN: case OP_ASIN_ALT: case OP_ARC_SIN_ALT:
                        stack[sp-1] = asin(stack[sp-1]); break;
                    case OP_ACOS: case OP_ACOS_ALT: case OP_ARC_COS_ALT:
                        stack[sp-1] = acos(stack[sp-1]); break;
                    case OP_ATAN: case OP_ATAN_ALT: case OP_ARC_TAN_ALT:
                        stack[sp-1] = atan(stack[sp-1]); break;

                    case OP_SIN_DEG: stack[sp-1] = sin(stack[sp-1] * DEG_TO_RAD_F); break;
                    case OP_COS_DEG: stack[sp-1] = cos(stack[sp-1] * DEG_TO_RAD_F); break;
                    case OP_TAN_DEG: stack[sp-1] = tan(stack[sp-1] * DEG_TO_RAD_F); break;
                    case OP_SIN_GRAD: stack[sp-1] = sin(stack[sp-1] * GRAD_TO_RAD_F); break;
                    case OP_COS_GRAD: stack[sp-1] = cos(stack[sp-1] * GRAD_TO_RAD_F); break;
                    case OP_TAN_GRAD: stack[sp-1] = tan(stack[sp-1] * GRAD_TO_RAD_F); break;

                    case OP_ASIN_DEG: case OP_ASIN_DEG_ALT: case OP_ARC_SIN_ALT_DEG:
                        stack[sp-1] = asin(stack[sp-1]) * RAD_TO_DEG_F; break;
                    case OP_ACOS_DEG: case OP_ACOS_DEG_ALT: case OP_ARC_COS_ALT_DEG:
                        stack[sp-1] = acos(stack[sp-1]) * RAD_TO_DEG_F; break;
                    case OP_ATAN_DEG: case OP_ATAN_DEG_ALT: case OP_ARC_TAN_ALT_DEG:
                        stack[sp-1] = atan(stack[sp-1]) * RAD_TO_DEG_F; break;
                    case OP_ASIN_GRAD: case OP_ASIN_GRAD_ALT: case OP_ARC_SIN_ALT_GRAD:
                        stack[sp-1] = asin(stack[sp-1]) * RAD_TO_GRAD_F; break;
                    case OP_ACOS_GRAD: case OP_ACOS_GRAD_ALT: case OP_ARC_COS_ALT_GRAD:
                        stack[sp-1] = acos(stack[sp-1]) * RAD_TO_GRAD_F; break;
                    case OP_ATAN_GRAD: case OP_ATAN_GRAD_ALT: case OP_ARC_TAN_ALT_GRAD:
                        stack[sp-1] = atan(stack[sp-1]) * RAD_TO_GRAD_F; break;

                    case OP_SEC:  stack[sp-1] = 1.0f / cos(stack[sp-1]); break;
                    case OP_SEC_DEG:  stack[sp-1] = 1.0f / cos(stack[sp-1] * DEG_TO_RAD_F); break;
                    case OP_SEC_GRAD: stack[sp-1] = 1.0f / cos(stack[sp-1] * GRAD_TO_RAD_F); break;
                    case OP_COSEC:  stack[sp-1] = 1.0f / sin(stack[sp-1]); break;
                    case OP_COSEC_DEG:  stack[sp-1] = 1.0f / sin(stack[sp-1] * DEG_TO_RAD_F); break;
                    case OP_COSEC_GRAD: stack[sp-1] = 1.0f / sin(stack[sp-1] * GRAD_TO_RAD_F); break;
                    case OP_COT:  stack[sp-1] = 1.0f / tan(stack[sp-1]); break;
                    case OP_COT_DEG:  stack[sp-1] = 1.0f / tan(stack[sp-1] * DEG_TO_RAD_F); break;
                    case OP_COT_GRAD: stack[sp-1] = 1.0f / tan(stack[sp-1] * GRAD_TO_RAD_F); break;

                    case OP_ARC_SEC: case OP_ARC_SEC_ALT:
                        stack[sp-1] = acos(1.0f / stack[sp-1]); break;
                    case OP_ARC_SEC_DEG: case OP_ARC_SEC_ALT_DEG:
                        stack[sp-1] = acos(1.0f / stack[sp-1]) * RAD_TO_DEG_F; break;
                    case OP_ARC_SEC_GRAD: case OP_ARC_SEC_ALT_GRAD:
                        stack[sp-1] = acos(1.0f / stack[sp-1]) * RAD_TO_GRAD_F; break;
                    case OP_ARC_COSEC: case OP_ARC_COSEC_ALT:
                        stack[sp-1] = asin(1.0f / stack[sp-1]); break;
                    case OP_ARC_COSEC_DEG: case OP_ARC_COSEC_ALT_DEG:
                        stack[sp-1] = asin(1.0f / stack[sp-1]) * RAD_TO_DEG_F; break;
                    case OP_ARC_COSEC_GRAD: case OP_ARC_COSEC_ALT_GRAD:
                        stack[sp-1] = asin(1.0f / stack[sp-1]) * RAD_TO_GRAD_F; break;
                    case OP_ARC_COT: case OP_ARC_COT_ALT:
                        stack[sp-1] = atan(1.0f / stack[sp-1]); break;
                    case OP_ARC_COT_DEG: case OP_ARC_COT_ALT_DEG:
                        stack[sp-1] = atan(1.0f / stack[sp-1]) * RAD_TO_DEG_F; break;
                    case OP_ARC_COT_GRAD: case OP_ARC_COT_ALT_GRAD:
                        stack[sp-1] = atan(1.0f / stack[sp-1]) * RAD_TO_GRAD_F; break;

                    case OP_ASINH: case OP_ASINH_ALT: {
                        float v = stack[sp-1];
                        stack[sp-1] = log(v + sqrt(v * v + 1.0f));
                        break;
                    }
                    case OP_ACOSH: case OP_ACOSH_ALT: {
                        float v = stack[sp-1];
                        stack[sp-1] = log(v + sqrt(v * v - 1.0f));
                        break;
                    }
                    case OP_ATANH: case OP_ATANH_ALT: {
                        float v = stack[sp-1];
                        stack[sp-1] = 0.5f * log((1.0f + v) / (1.0f - v));
                        break;
                    }

                    case OP_ERF:       stack[sp-1] = gpu_erf_f(stack[sp-1]); break;
                    case OP_GELU:      stack[sp-1] = gpu_gelu_f(stack[sp-1]); break;
                    case OP_GELU_FAST: stack[sp-1] = gpu_gelu_fast_f(stack[sp-1]); break;
                    case OP_SWIGLU:    stack[sp-1] = stack[sp-1] * gpu_sigmoid_f(stack[sp-1]); break;
                    case OP_GEGLU:     stack[sp-1] = stack[sp-1] * gpu_gelu_f(stack[sp-1]); break;

                    default:
                        break;
                }
            }

            out[gid] = stack[0];
        }
        """;
}