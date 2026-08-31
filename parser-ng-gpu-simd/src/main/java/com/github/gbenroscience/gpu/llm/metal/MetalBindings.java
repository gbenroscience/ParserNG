package com.github.gbenroscience.gpu.llm.metal;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * FFM (java.lang.foreign) binding layer for Apple's Metal framework --
 * the Metal counterpart of {@code com.github.gbenroscience.gpu.evaluator.cuda.CudaBindings}
 * / {@code NvrtcBindings}. Distinct in KIND from those two, not just in
 * target platform: CUDA's driver API and NVRTC are plain C ABIs -- one
 * native function per operation, called directly by symbol name
 * ({@code cuMemAlloc}, {@code cuLaunchKernel}, ...). Metal has NO C API at
 * all -- {@code MTLDevice}, {@code MTLBuffer}, {@code MTLLibrary}, etc.
 * are Objective-C protocols/classes, and the ONLY way to drive them from
 * outside Objective-C/Swift is the Objective-C RUNTIME's own C ABI:
 * {@code objc_getClass}, {@code sel_registerName}, and {@code objc_msgSend}
 * (from {@code libobjc.dylib}), plus the two genuinely-C entry points
 * Metal.framework itself exports: {@code MTLCreateSystemDefaultDevice} and
 * {@code MTLCopyAllDevices}.
 *
 * <b>DESIGN CHOICE -- SEMANTIC METHODS, NOT RAW HANDLES:</b> CudaBindings
 * exposes one {@code MethodHandle} field per CUDA function because CUDA's
 * surface really does have that shape (a flat function table). Objective-C
 * messaging has no such per-operation surface -- "call a method" is always
 * the SAME two or three {@code objc_msgSend} shapes (varying only in
 * argument/return types), dispatched by selector string. Exposing raw
 * {@code objc_msgSend} handles here and making every call site build its
 * own selector + argument marshaling would just relocate CudaBindings'
 * simplicity problem into every caller. Instead this class exposes small,
 * named, semantic Java methods (`newBufferWithLength`, `dispatchThreads`,
 * `compileLibrary`, ...) that internally pick the right {@code objc_msgSend}
 * variant and selector -- {@link GpuContext} and {@link LlamaLayer} call
 * THOSE, the same way CUDA call sites call {@code cu.cuMemAlloc.invoke(...)}.
 *
 * <b>ARM64 SIMPLIFICATION USED HERE:</b> on Apple Silicon (the only
 * realistic Metal target for a from-scratch 2026 port -- Intel Macs with
 * discrete AMD GPUs are out of scope, flagged rather than silently
 * mishandled: {@link GpuContext} fails loudly if
 * {@code MTLCreateSystemDefaultDevice} returns a device without unified
 * memory), the Objective-C ABI does NOT use a separate
 * {@code objc_msgSend_stret} for struct returns the way x86_64 does --
 * plain {@code objc_msgSend} handles struct returns too, and small structs
 * passed BY VALUE (like the two {@code MTLSize} args to
 * {@code dispatchThreads:threadsPerThreadgroup:}) go through the normal
 * argument path. This lets every call site here use one of a handful of
 * typed {@code objc_msgSend} variants below rather than needing per-call
 * custom descriptors.
 *
 * <b>MEMORY MODEL -- WHY EVERY BUFFER IS
 * {@code MTLResourceStorageModeShared}:</b> Apple Silicon GPUs share
 * physical memory with the CPU (unified memory architecture) --
 * {@code StorageModeShared} buffers are simultaneously host- and
 * device-visible with no explicit copy step, which is what lets
 * {@link MetalBuffer#contents} be treated as ordinary host memory
 * throughout {@link LlamaLayer} (see that wrapper's javadoc). This is
 * deliberately NOT {@code StorageModeManaged} (explicit
 * {@code didModifyRange:}/{@code synchronize} calls the way a discrete-GPU
 * Mac would need) or {@code StorageModePrivate} (GPU-only, requires a
 * blit-encoder copy for every host round trip, i.e. reintroducing exactly
 * the PCIe-style staging the unified-memory architecture exists to avoid)
 * -- {@code Shared} is correct and fastest for every integrated
 * Apple-Silicon GPU, which is the only device class this binding targets.
 *
 * UNVERIFIED, same standing caveat as its CUDA/OpenCL siblings in this
 * codebase: no Apple Silicon Mac with the Metal frameworks was available
 * while writing this. The objc_msgSend argument/return shapes, selector
 * names, and MTLSize/MTLResourceOptions layouts are all traced against
 * Apple's published Objective-C ABI and Metal API surface, but this exact
 * call sequence has not been run against a real Metal device.
 */
public final class MetalBindings {

    // =====================================================================
    // ===================== NATIVE LIBRARY LOOKUP ========================
    // =====================================================================

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup OBJC = SymbolLookup.libraryLookup("/usr/lib/libobjc.A.dylib", Arena.global());
    private static final SymbolLookup METAL = SymbolLookup.libraryLookup(
            "/System/Library/Frameworks/Metal.framework/Metal", Arena.global());
    private static final SymbolLookup FOUNDATION = SymbolLookup.libraryLookup(
            "/System/Library/Frameworks/Foundation.framework/Foundation", Arena.global());

    private static final ValueLayout.OfLong PTR = ValueLayout.JAVA_LONG; // pointers carried as raw 64-bit values throughout
    private static final MemoryLayout MTLSIZE_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("width"),
            ValueLayout.JAVA_LONG.withName("height"),
            ValueLayout.JAVA_LONG.withName("depth"));

    // ---- objc runtime entry points ----
    private final MethodHandle h_objc_getClass;
    private final MethodHandle h_sel_registerName;

    // objc_msgSend variants, named by (argument shape) -> (return shape).
    private final MethodHandle h_send_0_ptr;                       // id recv, SEL           -> id
    private final MethodHandle h_send_1ptr_ptr;                    // id recv, SEL, id        -> id
    private final MethodHandle h_send_2ptr_ptr;                    // id recv, SEL, id, id    -> id
    private final MethodHandle h_send_1ptr_void;                   // id recv, SEL, id        -> void
    private final MethodHandle h_send_2long_ptr;                   // id recv, SEL, long,long -> id  (newBufferWithLength:options:)
    private final MethodHandle h_send_ptr2long_ptr;                // id recv, SEL, id,long,long -> id (newBufferWithBytes:length:options:)
    private final MethodHandle h_send_0_long;                      // id recv, SEL            -> long (length, etc.)
    private final MethodHandle h_send_3long_void;                  // id recv, SEL, long,long,long -> void (setBuffer:offset:atIndex:)
    private final MethodHandle h_send_ptrlonglong_void;            // id recv, SEL, ptr,long,long  -> void (setBytes:length:atIndex:)
    private final MethodHandle h_send_longptr_void;                // id recv, SEL, long,id   -> void (setThreadgroupMemoryLength:atIndex: uses long,long actually)
    private final MethodHandle h_send_2long_void;                  // id recv, SEL, long,long -> void
    private final MethodHandle h_send_2struct_void;                // id recv, SEL, MTLSize, MTLSize -> void (dispatchThreads:threadsPerThreadgroup:)
    private final MethodHandle h_send_void;                        // id recv, SEL            -> void
    private final MethodHandle h_send_3ptr_ptr;                    // id recv, SEL, id,id,id  -> id (newLibraryWithSource:options:error:)
    private final MethodHandle h_send_1ptr_ptr_out;                // id recv, SEL, id,ptr    -> id (newComputePipelineStateWithFunction:error:)

    // ---- Metal / Foundation C entry points ----
    private final MethodHandle h_MTLCreateSystemDefaultDevice;
    private final MethodHandle h_MTLCopyAllDevices;

    public MetalBindings() {
        this.h_objc_getClass = LINKER.downcallHandle(OBJC.find("objc_getClass").orElseThrow(),
                FunctionDescriptor.of(PTR, PTR));
        this.h_sel_registerName = LINKER.downcallHandle(OBJC.find("sel_registerName").orElseThrow(),
                FunctionDescriptor.of(PTR, PTR));

        MemorySegment msgSend = OBJC.find("objc_msgSend").orElseThrow();

        this.h_send_0_ptr = LINKER.downcallHandle(msgSend, FunctionDescriptor.of(PTR, PTR, PTR));
        this.h_send_1ptr_ptr = LINKER.downcallHandle(msgSend, FunctionDescriptor.of(PTR, PTR, PTR, PTR));
        this.h_send_2ptr_ptr = LINKER.downcallHandle(msgSend, FunctionDescriptor.of(PTR, PTR, PTR, PTR, PTR));
        this.h_send_1ptr_void = LINKER.downcallHandle(msgSend, FunctionDescriptor.ofVoid(PTR, PTR, PTR));
        this.h_send_2long_ptr = LINKER.downcallHandle(msgSend,
                FunctionDescriptor.of(PTR, PTR, PTR, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        this.h_send_ptr2long_ptr = LINKER.downcallHandle(msgSend,
                FunctionDescriptor.of(PTR, PTR, PTR, PTR, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        this.h_send_0_long = LINKER.downcallHandle(msgSend, FunctionDescriptor.of(ValueLayout.JAVA_LONG, PTR, PTR));
        this.h_send_3long_void = LINKER.downcallHandle(msgSend,
                FunctionDescriptor.ofVoid(PTR, PTR, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        this.h_send_ptrlonglong_void = LINKER.downcallHandle(msgSend,
                FunctionDescriptor.ofVoid(PTR, PTR, PTR, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        this.h_send_longptr_void = LINKER.downcallHandle(msgSend,
                FunctionDescriptor.ofVoid(PTR, PTR, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        this.h_send_2long_void = h_send_longptr_void;
        this.h_send_2struct_void = LINKER.downcallHandle(msgSend,
                FunctionDescriptor.ofVoid(PTR, PTR, MTLSIZE_LAYOUT, MTLSIZE_LAYOUT));
        this.h_send_void = LINKER.downcallHandle(msgSend, FunctionDescriptor.ofVoid(PTR, PTR));
        this.h_send_3ptr_ptr = LINKER.downcallHandle(msgSend, FunctionDescriptor.of(PTR, PTR, PTR, PTR, PTR, PTR));
        this.h_send_1ptr_ptr_out = LINKER.downcallHandle(msgSend, FunctionDescriptor.of(PTR, PTR, PTR, PTR, PTR));

        this.h_MTLCreateSystemDefaultDevice = LINKER.downcallHandle(
                METAL.find("MTLCreateSystemDefaultDevice").orElseThrow(), FunctionDescriptor.of(PTR));
        this.h_MTLCopyAllDevices = LINKER.downcallHandle(
                METAL.find("MTLCopyAllDevices").orElseThrow(), FunctionDescriptor.of(PTR));
    }

    // =====================================================================
    // ===================== SELECTOR / CLASS CACHING =====================
    // =====================================================================

    private final Map<String, Long> classCache = new HashMap<>();
    private final Map<String, Long> selCache = new HashMap<>();
    private final Arena stringArena = Arena.ofShared(); // backs every C string this instance ever interns; freed when the process exits

    private MemorySegment cstr(String s) {
        return stringArena.allocateFrom(s, StandardCharsets.UTF_8);
    }

    public long getClass(String name) {
        return classCache.computeIfAbsent(name, n -> {
            try {
                return (long) h_objc_getClass.invoke(cstr(n));
            } catch (Throwable t) {
                throw new RuntimeException("objc_getClass(" + n + ") failed", t);
            }
        });
    }

    public long sel(String name) {
        return selCache.computeIfAbsent(name, n -> {
            try {
                return (long) h_sel_registerName.invoke(cstr(n));
            } catch (Throwable t) {
                throw new RuntimeException("sel_registerName(" + n + ") failed", t);
            }
        });
    }

    // =====================================================================
    // ===================== GENERIC MESSAGE SEND ==========================
    // =====================================================================

    public long send0(long recv, String selector) {
        try {
            return (long) h_send_0_ptr.invoke(recv, sel(selector));
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend[" + selector + "] failed", t);
        }
    }

    public long send0Long(long recv, String selector) {
        try {
            return (long) h_send_0_long.invoke(recv, sel(selector));
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend[" + selector + "] failed", t);
        }
    }

    public long send1(long recv, String selector, long arg) {
        try {
            return (long) h_send_1ptr_ptr.invoke(recv, sel(selector), arg);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend[" + selector + "] failed", t);
        }
    }

    public void send1Void(long recv, String selector, long arg) {
        try {
            h_send_1ptr_void.invoke(recv, sel(selector), arg);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend[" + selector + "] failed", t);
        }
    }

    public long send2(long recv, String selector, long arg1, long arg2) {
        try {
            return (long) h_send_2ptr_ptr.invoke(recv, sel(selector), arg1, arg2);
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend[" + selector + "] failed", t);
        }
    }

    public void sendVoid(long recv, String selector) {
        try {
            h_send_void.invoke(recv, sel(selector));
        } catch (Throwable t) {
            throw new RuntimeException("objc_msgSend[" + selector + "] failed", t);
        }
    }

    /** newBufferWithLength:options: -- allocates an uninitialized MTLBuffer of `length` bytes with the given MTLResourceOptions. */
    public long newBufferWithLength(long device, long length, long options) {
        try {
            return (long) h_send_2long_ptr.invoke(device, sel("newBufferWithLength:options:"), length, options);
        } catch (Throwable t) {
            throw new RuntimeException("newBufferWithLength:options: failed", t);
        }
    }

    /** newBufferWithBytes:length:options: -- allocates an MTLBuffer and copies `length` bytes from the given host pointer into it. */
    public long newBufferWithBytes(long device, MemorySegment src, long length, long options) {
        try {
            return (long) h_send_ptr2long_ptr.invoke(device, sel("newBufferWithBytes:length:options:"), src, length, options);
        } catch (Throwable t) {
            throw new RuntimeException("newBufferWithBytes:length:options: failed", t);
        }
    }

    /** -[MTLBuffer contents] -- host-visible base pointer for a shared-storage-mode buffer. */
    public long bufferContents(long buffer) {
        return send0(buffer, "contents");
    }

    /** -[MTLBuffer length] */
    public long bufferLength(long buffer) {
        return send0Long(buffer, "length");
    }

    /** setBuffer:offset:atIndex: on a compute command encoder. */
    public void setBuffer(long encoder, long buffer, long offset, long index) {
        try {
            h_send_3long_void.invoke(encoder, sel("setBuffer:offset:atIndex:"), buffer, offset, index);
        } catch (Throwable t) {
            throw new RuntimeException("setBuffer:offset:atIndex: failed", t);
        }
    }

    /** setBytes:length:atIndex: on a compute command encoder -- used for scalar kernel constants (the Metal analogue of a CUDA kernel's plain int/float args). */
    public void setBytes(long encoder, MemorySegment src, long length, long index) {
        try {
            h_send_ptrlonglong_void.invoke(encoder, sel("setBytes:length:atIndex:"), src, length, index);
        } catch (Throwable t) {
            throw new RuntimeException("setBytes:length:atIndex: failed", t);
        }
    }

    /** setThreadgroupMemoryLength:atIndex: -- the Metal analogue of a CUDA kernel's dynamic extern __shared__ allocation size. */
    public void setThreadgroupMemoryLength(long encoder, long lengthBytes, long index) {
        try {
            h_send_2long_void.invoke(encoder, sel("setThreadgroupMemoryLength:atIndex:"), lengthBytes, index);
        } catch (Throwable t) {
            throw new RuntimeException("setThreadgroupMemoryLength:atIndex: failed", t);
        }
    }

    public void setComputePipelineState(long encoder, long pipelineState) {
        send1Void(encoder, "setComputePipelineState:", pipelineState);
    }

    /**
     * dispatchThreads:threadsPerThreadgroup: -- dispatches EXACTLY
     * (gx,gy,gz) total threads, letting Metal handle any non-uniform edge
     * threadgroup itself. This is the direct replacement for the CUDA
     * port's manual "ceil(workItems/blockSize)" grid-dimension rounding
     * (see LlamaLayer.launch1D/launch2D there) -- Metal does that
     * rounding internally for this selector, so callers here just supply
     * the true total work size, same as a CUDA kernel's logical N.
     * Requires the pipeline's kernel to have been compiled with Metal 2.1+
     * semantics (true for every macOS/iOS version this port targets).
     */
    public void dispatchThreads(long encoder, long gx, long gy, long gz, long tx, long ty, long tz) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment gridSize = tmp.allocate(MTLSIZE_LAYOUT);
            gridSize.set(ValueLayout.JAVA_LONG, 0, gx);
            gridSize.set(ValueLayout.JAVA_LONG, 8, gy);
            gridSize.set(ValueLayout.JAVA_LONG, 16, gz);
            MemorySegment groupSize = tmp.allocate(MTLSIZE_LAYOUT);
            groupSize.set(ValueLayout.JAVA_LONG, 0, tx);
            groupSize.set(ValueLayout.JAVA_LONG, 8, ty);
            groupSize.set(ValueLayout.JAVA_LONG, 16, tz);
            h_send_2struct_void.invoke(encoder, sel("dispatchThreads:threadsPerThreadgroup:"), gridSize, groupSize);
        } catch (Throwable t) {
            throw new RuntimeException("dispatchThreads:threadsPerThreadgroup: failed", t);
        }
    }

    /**
     * -[NSObject release] -- every {@code new*}/{@code alloc}/{@code copy}
     * Objective-C call in this file hands back an object already owned by
     * the caller (Cocoa's "NARC" convention: New/Alloc/Retain/Copy methods
     * transfer +1 ownership), so callers that are done with a long-lived
     * object (a compiled library, a pipeline state, a command queue, a
     * device buffer) balance that +1 with exactly one {@code release}
     * call. This requires no ARC bridge -- manual retain/release is the
     * runtime's native model; ARC is a compiler feature layered on top of
     * it, not a runtime requirement.
     */
    public void release(long obj) {
        if (obj != 0L) {
            sendVoid(obj, "release");
        }
    }

    public void endEncoding(long encoder) {
        sendVoid(encoder, "endEncoding");
    }

    public void commit(long commandBuffer) {
        sendVoid(commandBuffer, "commit");
    }

    public void waitUntilCompleted(long commandBuffer) {
        sendVoid(commandBuffer, "waitUntilCompleted");
    }

    /** -[MTLCommandBuffer status] -- 0=NotEnqueued,1=Enqueued,2=Committed,3=Scheduled,4=Completed,5=Error. Checked after waitUntilCompleted to surface GPU-side kernel failures instead of silently returning garbage. */
    public long commandBufferStatus(long commandBuffer) {
        return send0Long(commandBuffer, "status");
    }

    /** -[MTLCommandBuffer error] -- non-null id<NSError> when status==Error (5). */
    public long commandBufferError(long commandBuffer) {
        return send0(commandBuffer, "error");
    }

    // =====================================================================
    // ===================== DEVICE / QUEUE / LIBRARY ======================
    // =====================================================================

    public long createSystemDefaultDevice() {
        try {
            return (long) h_MTLCreateSystemDefaultDevice.invoke();
        } catch (Throwable t) {
            throw new RuntimeException("MTLCreateSystemDefaultDevice failed", t);
        }
    }

    /** MTLCopyAllDevices() -> NSArray<id<MTLDevice>>, unpacked into a Java array via -count/-objectAtIndex:. */
    public long[] copyAllDevices() {
        try {
            long arr = (long) h_MTLCopyAllDevices.invoke();
            if (arr == 0L) {
                return new long[0];
            }
            long count = send0Long(arr, "count");
            long[] out = new long[(int) count];
            long selObjAt = sel("objectAtIndex:");
            for (int i = 0; i < out.length; i++) {
                out[i] = (long) invokeSend1(arr, selObjAt, i);
            }
            return out;
        } catch (Throwable t) {
            throw new RuntimeException("MTLCopyAllDevices failed", t);
        }
    }

    private Object invokeSend1(long recv, long selector, long idx) throws Throwable {
        return h_send_1ptr_ptr.invoke(recv, selector, idx);
    }

    /** -[MTLDevice name] as a Java String, via NSString -UTF8String. */
    public String deviceName(long device) {
        long nsName = send0(device, "name");
        return nsStringToJava(nsName);
    }

    /** -[MTLDevice hasUnifiedMemory] (BOOL, returned in the low byte of the register on arm64). */
    public boolean hasUnifiedMemory(long device) {
        return (send0Long(device, "hasUnifiedMemory") & 0xFF) != 0;
    }

    /** -[MTLDevice newCommandQueue] */
    public long newCommandQueue(long device) {
        return send0(device, "newCommandQueue");
    }

    /** -[MTLCommandQueue commandBuffer] */
    public long commandBuffer(long queue) {
        return send0(queue, "commandBuffer");
    }

    /** -[MTLCommandBuffer computeCommandEncoder] */
    public long computeCommandEncoder(long commandBuffer) {
        return send0(commandBuffer, "computeCommandEncoder");
    }

    /**
     * Compiles MSL source into an {@code id<MTLLibrary>} via
     * {@code -[MTLDevice newLibraryWithSource:options:error:]} -- the
     * Metal analogue of NVRTC's {@code nvrtcCompileProgram}. `options` may
     * be 0 (nil, default compile options). Throws with the compiler's
     * NSError message on failure instead of returning a null library.
     */
    public long compileLibrary(long device, String mslSource) {
        try (Arena tmp = Arena.ofConfined()) {
            long nsSource = javaStringToNSString(mslSource);
            MemorySegment errorOut = tmp.allocate(ValueLayout.JAVA_LONG);
            long library = (long) h_send_3ptr_ptr.invoke(device, sel("newLibraryWithSource:options:error:"),
                    nsSource, 0L, errorOut);
            if (library == 0L) {
                long error = errorOut.get(ValueLayout.JAVA_LONG, 0);
                throw new IllegalStateException("MSL compile failed: " + describeNSError(error));
            }
            return library;
        } catch (Throwable t) {
            if (t instanceof IllegalStateException ise) {
                throw ise;
            }
            throw new RuntimeException("newLibraryWithSource:options:error: failed", t);
        }
    }

    /** -[MTLLibrary newFunctionWithName:] */
    public long newFunction(long library, String name) {
        long nsName = javaStringToNSString(name);
        return send1(library, "newFunctionWithName:", nsName);
    }

    /** -[MTLDevice newComputePipelineStateWithFunction:error:] -- the Metal analogue of resolving a CUfunction, but here it also performs the equivalent of PTX-to-SASS finalization for this specific device. */
    public long newComputePipelineState(long device, long function) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment errorOut = tmp.allocate(ValueLayout.JAVA_LONG);
            long pso = (long) h_send_1ptr_ptr_out.invoke(device, sel("newComputePipelineStateWithFunction:error:"),
                    function, errorOut);
            if (pso == 0L) {
                long error = errorOut.get(ValueLayout.JAVA_LONG, 0);
                throw new IllegalStateException("newComputePipelineStateWithFunction:error: failed: " + describeNSError(error));
            }
            return pso;
        } catch (Throwable t) {
            if (t instanceof IllegalStateException ise) {
                throw ise;
            }
            throw new RuntimeException("newComputePipelineStateWithFunction:error: failed", t);
        }
    }

    // =====================================================================
    // ===================== NSString / NSError HELPERS ====================
    // =====================================================================

    private long nsStringClass = 0L;

    /** [NSString stringWithUTF8String:] */
    public long javaStringToNSString(String s) {
        if (nsStringClass == 0L) {
            nsStringClass = getClass("NSString");
        }
        MemorySegment cstr = stringArena.allocateFrom(s, StandardCharsets.UTF_8);
        return send1(nsStringClass, "stringWithUTF8String:", cstr.address());
    }

    /** -[NSString UTF8String], copied into a Java String. */
    public String nsStringToJava(long nsString) {
        if (nsString == 0L) {
            return "";
        }
        long cstrPtr = send0(nsString, "UTF8String");
        if (cstrPtr == 0L) {
            return "";
        }
        MemorySegment seg = MemorySegment.ofAddress(cstrPtr).reinterpret(1L << 20);
        return seg.getString(0, StandardCharsets.UTF_8);
    }

    private String describeNSError(long error) {
        if (error == 0L) {
            return "(no NSError provided)";
        }
        long desc = send0(error, "localizedDescription");
        return nsStringToJava(desc);
    }

    // =====================================================================
    // ===================== MTLResourceOptions constants =================
    // =====================================================================

    /** MTLResourceStorageModeShared << MTLResourceStorageModeShift (shift=4) -- see class javadoc for why this is the only storage mode used in this port. */
    public static final long MTL_RESOURCE_STORAGE_MODE_SHARED = 0L << 4;

    /** MTLCPUCacheModeDefaultCache | MTLResourceStorageModeShared -- the full options value passed to newBufferWith*:options: everywhere in this port. */
    public static final long DEFAULT_BUFFER_OPTIONS = MTL_RESOURCE_STORAGE_MODE_SHARED;

    /** MTLCommandBufferStatusCompleted */
    public static final long MTL_COMMAND_BUFFER_STATUS_COMPLETED = 4L;
    /** MTLCommandBufferStatusError */
    public static final long MTL_COMMAND_BUFFER_STATUS_ERROR = 5L;
}