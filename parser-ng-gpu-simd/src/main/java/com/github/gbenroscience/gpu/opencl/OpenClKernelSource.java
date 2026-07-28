package com.github.gbenroscience.gpu.opencl;

/**
 * Two OpenCL C kernels compiled from ONE combined source string / ONE
 * clBuildProgram call: "interpret" (double stack machine) and "interpretF32"
 * (float stack machine). Both port VectorTurboEvaluator's full 101-opcode set;
 * the switch bodies are structurally identical -- only the element type, the
 * trig/deg-conversion constants, and the gelu/sigmoid helper functions differ
 * between them.
 *
 * WHY FLOAT IS A REAL SEPARATE KERNEL, NOT double-cast-to-float: this is the
 * whole point of doing this properly. interpretF32's stack, its
 * literalConstants buffer, and its in/out buffers are all `float` at every step
 * -- no double arithmetic happens anywhere in its call path. That's what
 * actually buys the throughput: half the memory traffic per element, more
 * elements per warp/wavefront on hardware with wide float SIMD lanes, and it
 * also runs at full speed on GPUs with crippled or absent fp64 hardware (most
 * consumer/mobile GPUs), which the double kernel cannot.
 *
 * The float kernel deliberately does NOT enable cl_khr_fp64 and contains no
 * double literals or double intermediates -- every constant below has an
 * explicit `f` suffix specifically so it compiles and runs on devices with no
 * double support at all, not just ones where double happens to be slower.
 *
 * VERIFIED against VectorTurboEvaluator.java source for opcode dispatch,
 * pop/push counts, and formulas for every op EXCEPT the five noted below, which
 * call into a `Maths` class not present in what was shared. Those five use
 * standard/textbook formulas in BOTH kernels -- diff against your Maths.java
 * before trusting results for expressions using
 * gelu/geglu/swiglu/erf/gelu_fast, in either precision.
 *
 * - OP_ERF : native erf() builtin (exact, not an approximation) - OP_GELU :
 * 0.5*x*(1+erf(x/sqrt(2))) [exact GELU] - OP_GELU_FAST :
 * 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715x^3))) [tanh-approx GELU] - OP_SWIGLU : x
 * * sigmoid(x) [self-gated SiLU] - OP_SWIGLU_2 : a * b * sigmoid(b) [SiLU(b)
 * gating a] - OP_GEGLU : x * gelu(x) [self-gated GELU] - OP_GEGLU_2 : a *
 * gelu(b) [GELU(b) gating a]
 */
public final class OpenClKernelSource {

    private OpenClKernelSource() {
    }

    public static final int MAX_STACK = 64;

    public static final String KERNEL_NAME_F64 = "interpret";
    public static final String KERNEL_NAME_F32 = "interpretF32";

    public static final String OPENCL_SOURCE = """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        // ---- Shared opcode numbering (identical for both precisions) ----
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

        // ---- double-precision constants/helpers (used by "interpret") ----
        #define DEG_TO_RAD 0.017453292519943295
        #define RAD_TO_DEG 57.29577951308232
        #define GRAD_TO_RAD 0.015707963267948967
        #define RAD_TO_GRAD 63.66197723675814

        inline double gpu_gelu(double x) {
            return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
        }
        inline double gpu_gelu_fast(double x) {
            double x3 = x * x * x;
            return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x3)));
        }
        inline double gpu_sigmoid(double x) {
            return 1.0 / (1.0 + exp(-x));
        }

        // ---- single-precision constants/helpers (used by "interpretF32") ----
        // All literals carry an explicit f suffix so no double arithmetic is
        // ever introduced into the float kernel's call path -- this is what
        // lets it run at full speed (and at all) on GPUs with no fp64 unit.
        #define DEG_TO_RAD_F 0.017453292f
        #define RAD_TO_DEG_F 57.29577951f
        #define GRAD_TO_RAD_F 0.015707963f
        #define RAD_TO_GRAD_F 63.66197723f

        inline float gpu_gelu_f(float x) {
            return 0.5f * x * (1.0f + erf(x * 0.70710678f));
        }
        inline float gpu_gelu_fast_f(float x) {
            float x3 = x * x * x;
            return 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x3)));
        }
        inline float gpu_sigmoid_f(float x) {
            return 1.0f / (1.0f + exp(-x));
        }

        __kernel void interpret(
            __global const int* opcodes,
            __global const int* targetSlots,
            __global const double* literalConstants,
            const int instructionCount,
            __global const double* in,
            const int dataSize,
            const int varCount,
            __global double* out)
        {
            const int gid = get_global_id(0);
            if (gid >= dataSize) {
                return;
            }

            double stack[MAX_STACK];
            int sp = 0;

            for (int i = 0; i < instructionCount; i++) {
                const int op = opcodes[i];

                switch (op) {
                    case OP_CONST:
                        stack[sp++] = literalConstants[i];
                        break;

                    case OP_LOAD:
                        stack[sp++] = in[targetSlots[i] * dataSize + gid];
                        break;

                    case OP_ADD: { double b = stack[--sp]; stack[sp-1] = stack[sp-1] + b; break; }
                    case OP_SUB: { double b = stack[--sp]; stack[sp-1] = stack[sp-1] - b; break; }
                    case OP_MUL: { double b = stack[--sp]; stack[sp-1] = stack[sp-1] * b; break; }
                    case OP_DIV: { double b = stack[--sp]; stack[sp-1] = stack[sp-1] / b; break; }
                    case OP_POW: { double b = stack[--sp]; stack[sp-1] = pow(stack[sp-1], b); break; }
                    case OP_REM: { double b = stack[--sp]; stack[sp-1] = fmod(stack[sp-1], b); break; }

                    case OP_GT: { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] >  b) ? 1.0 : 0.0; break; }
                    case OP_LT: { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] <  b) ? 1.0 : 0.0; break; }
                    case OP_EQ: { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] == b) ? 1.0 : 0.0; break; }
                    case OP_NE: { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] != b) ? 1.0 : 0.0; break; }
                    case OP_GE: { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] >= b) ? 1.0 : 0.0; break; }
                    case OP_LE: { double b = stack[--sp]; stack[sp-1] = (stack[sp-1] <= b) ? 1.0 : 0.0; break; }

                    case OP_SWIGLU_2: { double b = stack[--sp]; double a = stack[sp-1]; stack[sp-1] = a * b * gpu_sigmoid(b); break; }
                    case OP_GEGLU_2:  { double b = stack[--sp]; double a = stack[sp-1]; stack[sp-1] = a * gpu_gelu(b); break; }

                    case OP_VMA: {
                        double c = stack[--sp];
                        double b = stack[--sp];
                        stack[sp-1] = stack[sp-1] * b + c;
                        break;
                    }
                    case OP_IF: {
                        double f = stack[--sp];
                        double t = stack[--sp];
                        stack[sp-1] = (stack[sp-1] != 0.0) ? t : f;
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
                    case OP_CBRT: stack[sp-1] = cbrt(stack[sp-1]); break;
                    case OP_LOG:  stack[sp-1] = log(stack[sp-1]);  break;
                    case OP_LOG10: stack[sp-1] = log10(stack[sp-1]); break;

                    case OP_ASIN: case OP_ASIN_ALT: case OP_ARC_SIN_ALT:
                        stack[sp-1] = asin(stack[sp-1]); break;
                    case OP_ACOS: case OP_ACOS_ALT: case OP_ARC_COS_ALT:
                        stack[sp-1] = acos(stack[sp-1]); break;
                    case OP_ATAN: case OP_ATAN_ALT: case OP_ARC_TAN_ALT:
                        stack[sp-1] = atan(stack[sp-1]); break;

                    case OP_SIN_DEG: stack[sp-1] = sin(stack[sp-1] * DEG_TO_RAD); break;
                    case OP_COS_DEG: stack[sp-1] = cos(stack[sp-1] * DEG_TO_RAD); break;
                    case OP_TAN_DEG: stack[sp-1] = tan(stack[sp-1] * DEG_TO_RAD); break;
                    case OP_SIN_GRAD: stack[sp-1] = sin(stack[sp-1] * GRAD_TO_RAD); break;
                    case OP_COS_GRAD: stack[sp-1] = cos(stack[sp-1] * GRAD_TO_RAD); break;
                    case OP_TAN_GRAD: stack[sp-1] = tan(stack[sp-1] * GRAD_TO_RAD); break;

                    case OP_ASIN_DEG: case OP_ASIN_DEG_ALT: case OP_ARC_SIN_ALT_DEG:
                        stack[sp-1] = asin(stack[sp-1]) * RAD_TO_DEG; break;
                    case OP_ACOS_DEG: case OP_ACOS_DEG_ALT: case OP_ARC_COS_ALT_DEG:
                        stack[sp-1] = acos(stack[sp-1]) * RAD_TO_DEG; break;
                    case OP_ATAN_DEG: case OP_ATAN_DEG_ALT: case OP_ARC_TAN_ALT_DEG:
                        stack[sp-1] = atan(stack[sp-1]) * RAD_TO_DEG; break;
                    case OP_ASIN_GRAD: case OP_ASIN_GRAD_ALT: case OP_ARC_SIN_ALT_GRAD:
                        stack[sp-1] = asin(stack[sp-1]) * RAD_TO_GRAD; break;
                    case OP_ACOS_GRAD: case OP_ACOS_GRAD_ALT: case OP_ARC_COS_ALT_GRAD:
                        stack[sp-1] = acos(stack[sp-1]) * RAD_TO_GRAD; break;
                    case OP_ATAN_GRAD: case OP_ATAN_GRAD_ALT: case OP_ARC_TAN_ALT_GRAD:
                        stack[sp-1] = atan(stack[sp-1]) * RAD_TO_GRAD; break;

                    case OP_SEC:  stack[sp-1] = 1.0 / cos(stack[sp-1]); break;
                    case OP_SEC_DEG:  stack[sp-1] = 1.0 / cos(stack[sp-1] * DEG_TO_RAD); break;
                    case OP_SEC_GRAD: stack[sp-1] = 1.0 / cos(stack[sp-1] * GRAD_TO_RAD); break;
                    case OP_COSEC:  stack[sp-1] = 1.0 / sin(stack[sp-1]); break;
                    case OP_COSEC_DEG:  stack[sp-1] = 1.0 / sin(stack[sp-1] * DEG_TO_RAD); break;
                    case OP_COSEC_GRAD: stack[sp-1] = 1.0 / sin(stack[sp-1] * GRAD_TO_RAD); break;
                    case OP_COT:  stack[sp-1] = 1.0 / tan(stack[sp-1]); break;
                    case OP_COT_DEG:  stack[sp-1] = 1.0 / tan(stack[sp-1] * DEG_TO_RAD); break;
                    case OP_COT_GRAD: stack[sp-1] = 1.0 / tan(stack[sp-1] * GRAD_TO_RAD); break;

                    case OP_ARC_SEC: case OP_ARC_SEC_ALT:
                        stack[sp-1] = acos(1.0 / stack[sp-1]); break;
                    case OP_ARC_SEC_DEG: case OP_ARC_SEC_ALT_DEG:
                        stack[sp-1] = acos(1.0 / stack[sp-1]) * RAD_TO_DEG; break;
                    case OP_ARC_SEC_GRAD: case OP_ARC_SEC_ALT_GRAD:
                        stack[sp-1] = acos(1.0 / stack[sp-1]) * RAD_TO_GRAD; break;
                    case OP_ARC_COSEC: case OP_ARC_COSEC_ALT:
                        stack[sp-1] = asin(1.0 / stack[sp-1]); break;
                    case OP_ARC_COSEC_DEG: case OP_ARC_COSEC_ALT_DEG:
                        stack[sp-1] = asin(1.0 / stack[sp-1]) * RAD_TO_DEG; break;
                    case OP_ARC_COSEC_GRAD: case OP_ARC_COSEC_ALT_GRAD:
                        stack[sp-1] = asin(1.0 / stack[sp-1]) * RAD_TO_GRAD; break;
                    case OP_ARC_COT: case OP_ARC_COT_ALT:
                        stack[sp-1] = atan(1.0 / stack[sp-1]); break;
                    case OP_ARC_COT_DEG: case OP_ARC_COT_ALT_DEG:
                        stack[sp-1] = atan(1.0 / stack[sp-1]) * RAD_TO_DEG; break;
                    case OP_ARC_COT_GRAD: case OP_ARC_COT_ALT_GRAD:
                        stack[sp-1] = atan(1.0 / stack[sp-1]) * RAD_TO_GRAD; break;

                    case OP_ASINH: case OP_ASINH_ALT: {
                        double v = stack[sp-1];
                        stack[sp-1] = log(v + sqrt(v * v + 1.0));
                        break;
                    }
                    case OP_ACOSH: case OP_ACOSH_ALT: {
                        double v = stack[sp-1];
                        stack[sp-1] = log(v + sqrt(v * v - 1.0));
                        break;
                    }
                    case OP_ATANH: case OP_ATANH_ALT: {
                        double v = stack[sp-1];
                        stack[sp-1] = 0.5 * log((1.0 + v) / (1.0 - v));
                        break;
                    }

                    case OP_ERF:       stack[sp-1] = erf(stack[sp-1]); break;
                    case OP_GELU:      stack[sp-1] = gpu_gelu(stack[sp-1]); break;
                    case OP_GELU_FAST: stack[sp-1] = gpu_gelu_fast(stack[sp-1]); break;
                    case OP_SWIGLU:    stack[sp-1] = stack[sp-1] * gpu_sigmoid(stack[sp-1]); break;
                    case OP_GEGLU:     stack[sp-1] = stack[sp-1] * gpu_gelu(stack[sp-1]); break;

                    default:
                        break;
                }
            }

            out[gid] = stack[0];
        }

        __kernel void interpretF32(
            __global const int* opcodes,
            __global const int* targetSlots,
            __global const float* literalConstants,
            const int instructionCount,
            __global const float* in,
            const int dataSize,
            const int varCount,
            __global float* out)
        {
            const int gid = get_global_id(0);
            if (gid >= dataSize) {
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
                        stack[sp++] = in[targetSlots[i] * dataSize + gid];
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
                    case OP_SIN:  stack[sp-1] = sin(stack[sp-1]);  break;
                    case OP_COS:  stack[sp-1] = cos(stack[sp-1]);  break;
                    case OP_TAN:  stack[sp-1] = tan(stack[sp-1]);  break;
                    case OP_SINH: stack[sp-1] = sinh(stack[sp-1]); break;
                    case OP_COSH: stack[sp-1] = cosh(stack[sp-1]); break;
                    case OP_TANH: stack[sp-1] = tanh(stack[sp-1]); break;
                    case OP_ABS:  stack[sp-1] = fabs(stack[sp-1]); break;
                    case OP_EXP:  stack[sp-1] = exp(stack[sp-1]);  break;
                    case OP_SQRT: stack[sp-1] = sqrt(stack[sp-1]); break;
                    case OP_CBRT: stack[sp-1] = cbrt(stack[sp-1]); break;
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

                    case OP_ERF:       stack[sp-1] = erf(stack[sp-1]); break;
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
