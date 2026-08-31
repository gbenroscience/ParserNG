package com.github.gbenroscience.gpu.evaluator.metal;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Raw FFM downcalls into the Objective-C runtime (libobjc) and the Metal /
 * Foundation frameworks -- macOS ONLY. Apple never shipped Metal with a flat
 * C ABI the way OpenCL has one; every Metal object (device, command queue,
 * buffer, library, function, pipeline state, command buffer, encoder...) is
 * an Objective-C object reached by sending it a selector through
 * {@code objc_msgSend}, so that is what this class wraps.
 *
 * WHY THIS LOOKS DIFFERENT FROM OpenClBindings: OpenCL exposes one C
 * function per operation (clCreateBuffer, clEnqueueWriteBuffer, ...), so
 * OpenClBindings has one MethodHandle per operation. objc_msgSend is a
 * single, effectively-untyped C function whose real signature depends on
 * which selector you send and what that method's declared argument/return
 * types are -- there is no single correct FunctionDescriptor for it. The
 * standard (and only correct) way to call it from FFM is to obtain a
 * SEPARATE downcall MethodHandle for {@code objc_msgSend}, once per distinct
 * (return type, argument type list) SHAPE actually needed, and reuse that
 * shape's handle for every selector matching it. That is what the
 * {@code idMsgSend*} / {@code voidMsgSend*} / {@code longMsgSend0} families
 * below are: one MethodHandle per call SHAPE, not per Metal method -- still
 * a small, closed set (mirrors OpenClBindings' one-handle-per-call
 * granularity, just sliced along a different axis).
 *
 * MEMORY MANAGEMENT: this binding is used WITHOUT ARC (Automatic Reference
 * Counting is a compiler feature; there is no compiler here). Objects
 * returned by a method whose selector begins with "alloc", "new", "copy", or
 * "mutableCopy" are returned already retained (+1) -- the caller owns them
 * and must eventually send "release". Objects returned by any other selector
 * (typically class convenience constructors) are AUTORELEASED by convention
 * and are only guaranteed to stay alive until the current autorelease pool
 * drains. To avoid depending on autorelease-pool timing entirely, every
 * helper in this class that needs an NSString builds it via
 * {@code alloc}+{@code initWithUTF8String:} (an owned, non-autoreleased
 * reference the caller releases explicitly) rather than the
 * {@code stringWithUTF8String:} convenience constructor. MetalCompositeExpression
 * follows the same alloc/init-and-release discipline for every Metal object
 * it creates directly; the one exception is {@code MTLCreateSystemDefaultDevice()}
 * and {@code MTLCopyAllDevices()}, both plain C functions (not objc_msgSend
 * calls) that Apple's headers document as returning an object the caller
 * must release -- handled the same way here.
 *
 * STRUCT-BY-VALUE ARGUMENTS: the only Metal call in this scaffold's surface
 * that passes a struct by value is {@code dispatchThreads:threadsPerThreadgroup:},
 * which takes two {@code MTLSize} structs ({@code struct { NSUInteger width,
 * height, depth; }}, 24 bytes on 64-bit). Java's FFM Linker supports struct
 * arguments natively via a {@link StructLayout} in the FunctionDescriptor --
 * no {@code objc_msgSend_stret} trampoline is needed for ARGUMENTS (stret
 * variants only ever matter for large STRUCT RETURNS, and nothing in this
 * scaffold's call surface returns a struct by value, so objc_msgSend_stret
 * is deliberately not bound here at all).
 */
public final class MetalBindings {

    // --- MTLResourceOptions bits actually used here (see Apple's
    // MTLResource.h). Only StorageModeShared is used: on Apple Silicon this
    // is the unified-memory mode where CPU and GPU see the same bytes with
    // no explicit copy step, which is what lets MetalCompositeExpression
    // write/read buffer .contents() directly instead of OpenCL-style
    // enqueue-write/enqueue-read calls. On an Intel Mac with a discrete
    // (non-unified-memory) GPU, StorageModeShared still works but the
    // driver, not this code, is doing the copy under the hood -- Managed
    // storage mode plus explicit didModifyRange:/synchronization would be
    // the higher-throughput choice there. Not implemented: this scaffold
    // targets the common case (Shared works correctly everywhere Metal
    // runs) rather than the discrete-GPU fast path.
    public static final long MTLResourceStorageModeShared = 0L << 4;
    public static final long MTLResourceCPUCacheModeDefaultCache = 0L;
    public static final long MTLResourceOptionsDefault = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;

    /** {@code struct MTLSize { NSUInteger width, height, depth; }} -- 3x 8-byte NSUInteger, no padding. */
    public static final StructLayout MTL_SIZE = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("width"),
            ValueLayout.JAVA_LONG.withName("height"),
            ValueLayout.JAVA_LONG.withName("depth")
    ).withName("MTLSize");

    private final Linker linker = Linker.nativeLinker();
    private final SymbolLookup objcLookup;
    private final SymbolLookup metalLookup;
    private final SymbolLookup foundationLookup;

    // ---- objc runtime primitives ----
    private final MethodHandle objc_getClass;
    private final MethodHandle sel_registerName;

    // ---- Metal C functions (NOT objc_msgSend -- these are plain exported
    // C functions from Metal.framework) ----
    public final MethodHandle MTLCreateSystemDefaultDevice;
    public final MethodHandle MTLCopyAllDevices;

    // ---- objc_msgSend, one MethodHandle per call SHAPE (see class javadoc) ----
    private final MethodHandle msgSend_ret_id_args0;
    private final MethodHandle msgSend_ret_id_args1;
    private final MethodHandle msgSend_ret_id_args2;
    private final MethodHandle msgSend_ret_id_args3;
    private final MethodHandle msgSend_ret_id_argL;
    private final MethodHandle msgSend_ret_id_argsLL;
    private final MethodHandle msgSend_ret_id_argsPtrLL;
    private final MethodHandle msgSend_ret_void_args0;
    private final MethodHandle msgSend_ret_void_args1;
    private final MethodHandle msgSend_ret_void_argsPtrLL;
    private final MethodHandle msgSend_ret_void_argsDispatch;
    private final MethodHandle msgSend_ret_long_args0;

    private final ConcurrentHashMap<String, MemorySegment> classCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, MemorySegment> selCache = new ConcurrentHashMap<>();

    public MetalBindings() {
        this.objcLookup = SymbolLookup.libraryLookup("/usr/lib/libobjc.A.dylib", Arena.global());
        this.metalLookup = SymbolLookup.libraryLookup(
                "/System/Library/Frameworks/Metal.framework/Metal", Arena.global());
        this.foundationLookup = SymbolLookup.libraryLookup(
                "/System/Library/Frameworks/Foundation.framework/Foundation", Arena.global());

        ValueLayout.OfInt CI = ValueLayout.JAVA_INT;
        ValueLayout.OfLong CL = ValueLayout.JAVA_LONG;
        AddressLayout PTR = ValueLayout.ADDRESS;

        this.objc_getClass = objcDowncall("objc_getClass", FunctionDescriptor.of(PTR, PTR));
        this.sel_registerName = objcDowncall("sel_registerName", FunctionDescriptor.of(PTR, PTR));

        this.MTLCreateSystemDefaultDevice = metalDowncall("MTLCreateSystemDefaultDevice",
                FunctionDescriptor.of(PTR));
        // Returns an NSArray<id<MTLDevice>>* -- every installed GPU, not
        // just the system default. Only present so multi-GPU machines (an
        // eGPU, or an Intel Mac with both integrated and discrete parts)
        // can be enumerated the same way OpenClCompositeExpression
        // enumerates every OpenCL platform/device rather than trusting a
        // single default.
        this.MTLCopyAllDevices = metalDowncall("MTLCopyAllDevices", FunctionDescriptor.of(PTR));

        this.msgSend_ret_id_args0 = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR));
        this.msgSend_ret_id_args1 = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR, PTR));
        this.msgSend_ret_id_args2 = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR, PTR, PTR));
        this.msgSend_ret_id_args3 = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR, PTR, PTR, PTR));
        // e.g. NSArray's objectAtIndex: -- (NSUInteger) -> id.
        this.msgSend_ret_id_argL = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR, CL));
        this.msgSend_ret_id_argsLL = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR, CL, CL));
        this.msgSend_ret_id_argsPtrLL = objcMsgSend(FunctionDescriptor.of(PTR, PTR, PTR, PTR, CL, CL));
        this.msgSend_ret_void_args0 = objcMsgSend(FunctionDescriptor.ofVoid(PTR, PTR));
        this.msgSend_ret_void_args1 = objcMsgSend(FunctionDescriptor.ofVoid(PTR, PTR, PTR));
        this.msgSend_ret_void_argsPtrLL = objcMsgSend(FunctionDescriptor.ofVoid(PTR, PTR, PTR, CL, CL));
        this.msgSend_ret_void_argsDispatch = objcMsgSend(
                FunctionDescriptor.ofVoid(PTR, PTR, MTL_SIZE, MTL_SIZE));
        this.msgSend_ret_long_args0 = objcMsgSend(FunctionDescriptor.of(CL, PTR, PTR));

        // silence "unused" for CI -- kept for symmetry/documentation with
        // OpenClBindings' local aliasing of its ValueLayouts.
        assert CI != null;
    }

    private MethodHandle objcDowncall(String symbol, FunctionDescriptor fd) {
        MemorySegment addr = objcLookup.find(symbol)
                .orElseThrow(() -> new UnsatisfiedLinkError("objc runtime symbol not found: " + symbol));
        return linker.downcallHandle(addr, fd);
    }

    private MethodHandle metalDowncall(String symbol, FunctionDescriptor fd) {
        MemorySegment addr = metalLookup.find(symbol)
                .or(() -> foundationLookup.find(symbol))
                .orElseThrow(() -> new UnsatisfiedLinkError("Metal/Foundation symbol not found: " + symbol));
        return linker.downcallHandle(addr, fd);
    }

    /** Every {@code objc_msgSend} handle resolves against the same address; only the descriptor (shape) differs. */
    private MethodHandle objcMsgSend(FunctionDescriptor fd) {
        MemorySegment addr = objcLookup.find("objc_msgSend")
                .orElseThrow(() -> new UnsatisfiedLinkError("objc_msgSend not found"));
        return linker.downcallHandle(addr, fd);
    }

    // ================= class / selector lookup (cached) =================

    public MemorySegment cls(Arena arena, String name) {
        return classCache.computeIfAbsent(name, n -> {
            try {
                MemorySegment cName = arena.allocateFrom(n, StandardCharsets.UTF_8);
                MemorySegment c = (MemorySegment) objc_getClass.invoke(cName);
                if (c.equals(MemorySegment.NULL)) {
                    throw new IllegalStateException("Objective-C class not found: " + n);
                }
                return c;
            } catch (Throwable t) {
                throw new RuntimeException("objc_getClass(" + n + ") failed", t);
            }
        });
    }

    public MemorySegment sel(Arena arena, String name) {
        return selCache.computeIfAbsent(name, n -> {
            try {
                MemorySegment sName = arena.allocateFrom(n, StandardCharsets.UTF_8);
                MemorySegment s = (MemorySegment) sel_registerName.invoke(sName);
                if (s.equals(MemorySegment.NULL)) {
                    throw new IllegalStateException("Objective-C selector not found: " + n);
                }
                return s;
            } catch (Throwable t) {
                throw new RuntimeException("sel_registerName(" + n + ") failed", t);
            }
        });
    }

    // ================= objc_msgSend call-shape wrappers =================
    // Thin, checked-exception-free wrappers around the raw MethodHandles so
    // call sites in MetalCompositeExpression read like Objective-C message
    // sends rather than raw invoke() calls.

    public MemorySegment idMsgSend(MemorySegment receiver, MemorySegment selector) {
        return invokeId(msgSend_ret_id_args0, receiver, selector);
    }

    public MemorySegment idMsgSend(MemorySegment receiver, MemorySegment selector, MemorySegment a1) {
        return invokeId(msgSend_ret_id_args1, receiver, selector, a1);
    }

    public MemorySegment idMsgSend(MemorySegment receiver, MemorySegment selector, MemorySegment a1, MemorySegment a2) {
        return invokeId(msgSend_ret_id_args2, receiver, selector, a1, a2);
    }

    public MemorySegment idMsgSend(MemorySegment receiver, MemorySegment selector, MemorySegment a1, MemorySegment a2, MemorySegment a3) {
        return invokeId(msgSend_ret_id_args3, receiver, selector, a1, a2, a3);
    }

    /** e.g. NSArray's {@code objectAtIndex:} -- (NSUInteger) -> id. */
    public MemorySegment idMsgSendL(MemorySegment receiver, MemorySegment selector, long index) {
        try {
            return (MemorySegment) msgSend_ret_id_argL.invoke(receiver, selector, index);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(long) failed", t);
        }
    }

    /** e.g. {@code newBufferWithLength:options:} -- (NSUInteger, MTLResourceOptions). */
    public MemorySegment idMsgSendLL(MemorySegment receiver, MemorySegment selector, long a1, long a2) {
        try {
            return (MemorySegment) msgSend_ret_id_argsLL.invoke(receiver, selector, a1, a2);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(long,long) failed", t);
        }
    }

    /** e.g. {@code newBufferWithBytes:length:options:}. */
    public MemorySegment idMsgSendPtrLL(MemorySegment receiver, MemorySegment selector, MemorySegment ptr, long a2, long a3) {
        try {
            return (MemorySegment) msgSend_ret_id_argsPtrLL.invoke(receiver, selector, ptr, a2, a3);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(ptr,long,long) failed", t);
        }
    }

    public void voidMsgSend(MemorySegment receiver, MemorySegment selector) {
        try {
            msgSend_ret_void_args0.invoke(receiver, selector);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(void) failed", t);
        }
    }

    public void voidMsgSend(MemorySegment receiver, MemorySegment selector, MemorySegment a1) {
        try {
            msgSend_ret_void_args1.invoke(receiver, selector, a1);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(void, ptr) failed", t);
        }
    }

    /** e.g. {@code setBuffer:offset:atIndex:} or {@code setBytes:length:atIndex:} -- both are (ptr, long, long). */
    public void voidMsgSendPtrLL(MemorySegment receiver, MemorySegment selector, MemorySegment ptr, long a2, long a3) {
        try {
            msgSend_ret_void_argsPtrLL.invoke(receiver, selector, ptr, a2, a3);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(void, ptr,long,long) failed", t);
        }
    }

    /** {@code dispatchThreads:threadsPerThreadgroup:} -- the one struct-by-value call in this scaffold. */
    public void voidMsgSendDispatch(MemorySegment receiver, MemorySegment selector,
            MemorySegment threadsPerGrid, MemorySegment threadsPerThreadgroup) {
        try {
            msgSend_ret_void_argsDispatch.invoke(receiver, selector, threadsPerGrid, threadsPerThreadgroup);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(dispatchThreads) failed", t);
        }
    }

    /** e.g. {@code length}, {@code maxTotalThreadsPerThreadgroup} -- NSUInteger-returning property getters. */
    public long longMsgSend(MemorySegment receiver, MemorySegment selector) {
        try {
            return (long) msgSend_ret_long_args0.invoke(receiver, selector);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(long) failed", t);
        }
    }

    private MemorySegment invokeId(MethodHandle h, Object... args) {
        try {
            return (MemorySegment) h.invokeWithArguments(args);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend(id) failed", t);
        }
    }

    // ================= convenience helpers =================

    /**
     * Builds an owned (non-autoreleased) NSString via alloc+initWithUTF8String:,
     * per the class javadoc's memory-management discipline. Caller releases
     * it (send "release") once done -- see {@link #release}.
     */
    public MemorySegment nsString(Arena arena, String value) {
        MemorySegment nsStringClass = cls(arena, "NSString");
        MemorySegment instance = idMsgSend(nsStringClass, sel(arena, "alloc"));
        MemorySegment cString = arena.allocateFrom(value, StandardCharsets.UTF_8);
        return idMsgSend(instance, sel(arena, "initWithUTF8String:"), cString);
    }

    /** Reads an NSString's UTF-8 bytes into a Java String via {@code UTF8String}. */
    public String utf8String(Arena arena, MemorySegment nsString) {
        if (nsString.equals(MemorySegment.NULL)) {
            return null;
        }
        MemorySegment cStr = idMsgSend(nsString, sel(arena, "UTF8String"));
        if (cStr.equals(MemorySegment.NULL)) {
            return null;
        }
        return cStr.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
    }

    public void release(Arena arena, MemorySegment obj) {
        if (obj != null && !obj.equals(MemorySegment.NULL)) {
            voidMsgSend(obj, sel(arena, "release"));
        }
    }

    public MemorySegment mtlSize(Arena arena, long width, long height, long depth) {
        MemorySegment seg = arena.allocate(MTL_SIZE);
        seg.set(ValueLayout.JAVA_LONG, 0, width);
        seg.set(ValueLayout.JAVA_LONG, 8, height);
        seg.set(ValueLayout.JAVA_LONG, 16, depth);
        return seg;
    }
}