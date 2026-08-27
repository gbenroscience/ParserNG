package com.github.gbenroscience.arrow.tools.box;

import com.github.gbenroscience.gpu.GpuBackend;
import com.github.gbenroscience.gpu.evaluator.GpuCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.GpuExpressionBridge;
import com.github.gbenroscience.gpu.evaluator.cuda.CudaCompositeExpression;
import com.github.gbenroscience.gpu.evaluator.opencl.OpenClCompositeExpression;
import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.simd.turbo.tools.VectorTurboEvaluator;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.arrow.vector.Float4Vector;

/**
 * Evaluates a compiled ParserNG expression directly over Apache Arrow columnar
 * batches, using {@link GpuExpressionBridge}'s CUDA/OpenCL-backed bulk
 * evaluation path instead of {@code SIMDEngineEvaluator}'s CPU SIMD path.
 *
 * <h2>Why this is a separate class, not a mode flag on
 * {@link ArrowBulkEvaluator}</h2>
 * {@code SIMDEngineEvaluator.SIMDVectorCompositeExpression.applyBulk} takes a
 * {@code MemorySegment[]} — one independent segment per bound variable — which
 * is exactly what lets {@link ArrowBulkEvaluator} bind each Arrow column's data
 * buffer with zero copying, and {@link GpuCompositeExpression} exposes the
 * identical shape of entry point:
 * {@link GpuCompositeExpression#applyBulk(MemorySegment[], MemorySegment)}.
 * This class dispatches through THAT overload, not the single-flat-segment
 * {@link GpuCompositeExpression#applyBulk(MemorySegment, MemorySegment)} (which
 * takes one input segment holding every variable pre-concatenated column-major
 * — see that method's own javadoc). Each bound column's Arrow data buffer is
 * wrapped as its own {@link MemorySegment} — an alias via
 * {@link ArrowMemoryBridge}, never a copy — and handed straight to the GPU
 * backend in registry-slot order. No {@code Arena}, no
 * {@code MemorySegment.copy}, for any slot actually bound to a column: see
 * {@link GpuCompositeExpression#applyBulk(MemorySegment[], MemorySegment)}'s
 * own javadoc for why this is the true zero-copy path ("no host-side
 * flatten/staging buffer here at all... each slot's segment is transferred
 * straight into its slice of the device input buffer").
 * <p>
 * One narrow edge case still allocates, and it's worth naming rather than
 * hiding: {@code applyBulk(MemorySegment[], ...)} scatters <i>every</i>
 * array element to the device unconditionally, and the array must be sized to
 * {@code expression.getRegistry()}'s full width (matching how
 * {@code VectorTurboEvaluator}'s opcodes address slots by absolute registry
 * index) — not just the variables this one expression happens to reference. A
 * shared/session registry can outlive any single expression, so if this
 * expression doesn't reference every registry slot, the unreferenced ones are
 * back-filled with a small zeroed placeholder scoped to that one
 * {@link #evaluate} call (a null entry there would NPE during the device
 * scatter). That placeholder never touches real bound column data, and is
 * skipped entirely — no allocation at all — when every registry slot is
 * referenced, which is the common case.
 *
 * <h2>Binding model</h2>
 * Identical to {@link ArrowBulkEvaluator}: variables are bound to Arrow columns
 * by name, with the authoritative name-to-slot mapping coming from
 * {@link MathExpression#getSlotItems()}.
 *
 * <h2>Type support</h2>
 * Only {@link Float8Vector} (Arrow's float64 column type) columns are
 * supported, same as {@link ArrowBulkEvaluator} — this class evaluates in full
 * double precision on the GPU (via
 * {@link GpuCompositeExpression#applyBulk(MemorySegment[], MemorySegment)}),
 * not the native float32 kernel path
 * ({@link GpuCompositeExpression#applyBulkF32}). There is currently no float32
 * counterpart here since Arrow columns bound by this class are float64 to begin
 * with.
 *
 * <h2>Constant expressions</h2>
 * Same handling as {@link ArrowBulkEvaluator}: a zero-slot expression skips the
 * GPU entirely and fills the output via the ordinary scalar solver.
 *
 * <h2>Backend selection</h2> {@link #compile(MathExpression)} compiles a
 * {@code VectorTurboEvaluator} for {@code expression} and hands it to
 * {@link GpuExpressionBridge#from(VectorTurboEvaluator)}, which auto-selects
 * CUDA or OpenCL depending on what's actually bootstrapable on this machine
 * (CUDA preferred, OpenCL fallback — see that method's javadoc for the exact
 * order). {@link #compile(MathExpression, GpuBackend)} pins a specific backend
 * instead, via
 * {@link GpuExpressionBridge#from(VectorTurboEvaluator, GpuBackend)} —
 * compilation fails if that exact backend can't be bootstrapped, rather than
 * silently trying the other one. Call {@link #actualBackend()} after compiling
 * to find out which concrete backend an auto-selected instance ended up on. Use {@link #isBackendAvailable(GpuBackend)} /
 * {@link #isAnyGpuAvailable()} to probe before committing — e.g. to decide at
 * startup whether to build an {@link ArrowGpuBulkEvaluator} at all or fall back
 * to {@link ArrowBulkEvaluator} on machines with no GPU (or just call
 * {@link ArrowExpressionEvaluators#compilePreferGpu}, which does exactly that).
 *
 * <h2>Choosing a GPU device</h2>
 * Both backends now let you pick <i>which</i> installed GPU device runs the
 * expression with the same shape of API — enumerate, then select by name
 * substring or exact index before compiling. The one real asymmetry left is
 * OpenCL's extra platform dimension (a device sits under a platform) versus
 * CUDA's flat 0..N-1 device indexing (no platform layer) — everything else
 * lines up:
 * <ul>
 * <li><b>OpenCL</b> — call {@link #listOpenClDevices()} to see every (platform,
 * device) pair on this machine, then {@link #selectOpenClDevice(String)}
 * (vendor/name substring),
 * {@link #selectOpenClDevice(OpenClCompositeExpression.GpuVendor)} (a known
 * vendor, alias-matched), or {@link #selectOpenClDevice(int, int)} (exact
 * platform/device index) before compiling the next
 * {@link ArrowGpuBulkEvaluator}. {@link #clearOpenClDeviceSelection()} reverts
 * to the default (first GPU found).
 * <li><b>CUDA</b> — call {@link #listCudaDevices()} to see every device this
 * machine's driver exposes, then {@link #selectCudaDevice(String)} (name
 * substring, e.g. "4080", "A100") or {@link #selectCudaDevice(int)} (exact
 * device index) before compiling. {@link #clearCudaDeviceSelection()} reverts
 * to the default (device 0).
 * </ul>
 * For both backends: selection only affects instances compiled
 * <i>afterward</i> — an already-compiled instance's device is fixed for its
 * whole lifetime and never changes under it. The first time a given device is
 * selected its context/program is built and cached; switching selection back
 * and forth (e.g. across test methods) never rebuilds or recompiles anything
 * for a device already used once. Once compiled, call
 * {@link #deviceDescription()} on the instance to confirm which device it
 * actually landed on — this now returns a real description for both backends,
 * not just OpenCL.
 *
 * <h2>Thread safety</h2>
 *
 * A single instance may be shared and called concurrently from multiple threads
 * and will always produce correct results, but unlike
 * {@link ArrowBulkEvaluator} (which has a fully-concurrent {@code
 * parallel=false} path), every {@code evaluate} call here is internally
 * serialized against every other call on the same instance — both GPU backends
 * dispatch through shared per-instance device state (a command queue/stream and
 * kernel-arg buffers) that is not safe for concurrent use from multiple threads
 * (see {@code GpuCompositeExpressionTest}'s BUG#3 regression note). If you need
 * concurrent GPU evaluation from multiple threads, give each thread its own
 * instance (a separate {@link #compile} call) rather than sharing one. As with
 * {@link ArrowBulkEvaluator}, do not call {@link #close()} while another thread
 * may still be inside {@link #evaluate}.
 *
 * <h2>Lifecycle</h2>
 * Call {@link #close()} when done — this releases the compiled expression's
 * device-side resources ({@link GpuCompositeExpression} owns a device buffer
 * and, depending on backend, a staging {@code Arena}). Implements
 * {@link AutoCloseable} for try-with-resources use.
 *
 * <h2>Switching backends</h2>
 * This class implements {@link ArrowExpressionEvaluator}, the surface shared
 * with the CPU-backed {@link ArrowBulkEvaluator}. Prefer compiling through
 * {@link ArrowExpressionEvaluators#compile} with an
 * {@link ArrowExecutionBackend} GPU value (or
 * {@link ArrowExpressionEvaluators#compilePreferGpu}) over calling
 * {@link #compile(String)} directly when the call site should stay
 * backend-agnostic; reach for this class's static methods directly only for the
 * GPU-specific device selection and introspection they expose.
 */
public final class ArrowGpuBulkEvaluator implements ArrowExpressionEvaluator {

    private final MathExpression expression;
    private final GpuCompositeExpression compiled;
    private final MathExpression.Slot[] requiredSlots;
    private final String[] requiredVariableNames;
    private final int slotCount;
    private final boolean constantExpression;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // Every GPU dispatch on this instance goes through shared per-instance
    // device state (kernel args, command queue/stream) that only one caller
    // may touch at a time -- see the class javadoc's Thread safety section.
    // Unlike ArrowBulkEvaluator's parallelLock (which only guards the
    // parallel=true path), this guards every evaluate() call, since there is
    // no non-serialized GPU dispatch path to fall back to.
    private final Object dispatchLock = new Object();

    private ArrowGpuBulkEvaluator(MathExpression expression, GpuCompositeExpression compiled) {
        this.expression = expression;
        this.compiled = compiled;
        this.requiredSlots = expression.getSlotItems();
        this.slotCount = expression.getRegistry().size();
        this.constantExpression = requiredSlots.length == 0;

        String[] names = new String[requiredSlots.length];
        for (int i = 0; i < requiredSlots.length; i++) {
            names[i] = requiredSlots[i].getName();
        }
        this.requiredVariableNames = names;
    }

    // =========================================================================
    // Compilation entry points
    // =========================================================================
    /**
     * Compiles {@code expr}, auto-selecting a GPU backend (CUDA preferred,
     * OpenCL fallback). Throws whatever
     * {@link GpuExpressionBridge#from(VectorTurboEvaluator)} throws if no
     * usable backend is found on this machine.
     *
     * @param expr
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(String expr) throws Throwable {
        return compile(new MathExpression(expr));
    }

    /**
     * Compiles an already-constructed {@link MathExpression}, auto-selecting a
     * GPU backend.
     *
     * @param expression
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(MathExpression expression) throws Throwable {
        return new ArrowGpuBulkEvaluator(expression, bridgeFrom(expression, null));
    }

    /**
     * Compiles {@code expr} pinned to a specific GPU backend.
     *
     * @param expr
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(String expr, GpuBackend backend) throws Throwable {
        return compile(new MathExpression(expr), backend);
    }

    /**
     * Compiles an already-constructed {@link MathExpression} pinned to a
     * specific GPU backend.
     *
     * @param expression
     * @param backend
     * @return
     * @throws java.lang.Throwable
     */
    public static ArrowGpuBulkEvaluator compile(MathExpression expression, GpuBackend backend) throws Throwable {
        if (backend == null) {
            throw new NullPointerException("backend must not be null; use compile(MathExpression) to auto-select");
        }
        return new ArrowGpuBulkEvaluator(expression, bridgeFrom(expression, backend));
    }

    /**
     * Builds the {@code VectorTurboEvaluator} for {@code expression} and hands
     * it to {@link GpuExpressionBridge}. {@code vte.compile()} is called before
     * bridging (producing, and discarding, the CPU
     * {@code SIMDCompositeExpression}) because every call site of
     * {@code GpuExpressionBridge.from(vte)} in
     * {@code GpuCompositeExpressionTest} hands over a
     * {@code VectorTurboEvaluator} that has already been compiled this way —
     * either directly, or indirectly via
     * {@code SIMDEngineEvaluator.getEvaluator(...)}, which compiles internally
     * before returning. This method mirrors that pattern rather than relying on
     * undocumented lazy-compilation behavior inside the bridge.
     */
    private static GpuCompositeExpression bridgeFrom(MathExpression expression, GpuBackend backend) throws Throwable {
        VectorTurboEvaluator vte = new VectorTurboEvaluator(expression);
        vte.compile();
        return backend == null ? GpuExpressionBridge.from(vte) : GpuExpressionBridge.from(vte, backend);
    }

    /**
     * Whether {@code backend} bootstraps successfully on this JVM, probed right
     * now by compiling and immediately discarding a trivial constant expression
     * against it. Safe to call speculatively before deciding whether to build
     * an {@link ArrowGpuBulkEvaluator} at all, or to fall back to
     * {@link ArrowBulkEvaluator} instead — though for the common "use GPU if
     * available" case, {@link ArrowExpressionEvaluators#compilePreferGpu} is
     * simpler than probing first and compiling separately.
     *
     * @param backend must not be null; use {@link #isAnyGpuAvailable()} to
     * probe auto-selection instead of a specific backend
     * @return
     */
    public static boolean isBackendAvailable(GpuBackend backend) {
        if (backend == null) {
            throw new NullPointerException("backend must not be null; use isAnyGpuAvailable() to probe auto-selection");
        }
        return probe(backend);
    }

    /**
     * Whether ANY GPU backend (CUDA or OpenCL, whichever
     * {@link GpuExpressionBridge#from(VectorTurboEvaluator)} would auto-select)
     * bootstraps successfully on this JVM, probed right now.
     *
     * @return
     */
    public static boolean isAnyGpuAvailable() {
        return probe(null);
    }

    private static boolean probe(GpuBackend backend) {
        try {
            GpuCompositeExpression probe = bridgeFrom(new MathExpression("0"), backend);
            probe.close();
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    // =========================================================================
    // Backend / device introspection and selection
    // =========================================================================
    /**
     * Which concrete GPU backend this instance actually compiled against. For
     * an instance built via {@link #compile(MathExpression)} (auto selection),
     * this is the answer to "which one did it pick".
     *
     * @return
     */
    public GpuBackend actualBackend() {
        if (compiled instanceof CudaCompositeExpression) {
            return GpuBackend.CUDA;
        }
        if (compiled instanceof OpenClCompositeExpression) {
            return GpuBackend.OPENCL;
        }
        throw new IllegalStateException(
                "Unrecognized GpuCompositeExpression implementation: " + compiled.getClass().getName());
    }

    @Override
    public ArrowExecutionBackend backend() {
        return actualBackend() == GpuBackend.CUDA ? ArrowExecutionBackend.GPU_CUDA : ArrowExecutionBackend.GPU_OPENCL;
    }

    /**
     * A human-readable description of the exact device this instance is bound
     * to — e.g. {@code "[cuda device 0] NVIDIA GeForce RTX 4080
     * (compute capability 8.9)"} or {@code "[platform 0: NVIDIA CUDA]
     * [device 0: NVIDIA Corporation NVIDIA GeForce RTX 4080]"}. Fixed for this
     * instance's whole lifetime — confirms which device a prior
     * {@link #selectCudaDevice} / {@link #selectOpenClDevice} call actually
     * resolved to. Both backends expose this
     * ({@link CudaCompositeExpression#getDeviceDescription()} and
     * {@link OpenClCompositeExpression#getDeviceDescription()}).
     *
     * @return
     */
    public String deviceDescription() {
        if (compiled instanceof CudaCompositeExpression cuda) {
            return cuda.getDeviceDescription();
        }
        if (compiled instanceof OpenClCompositeExpression ocl) {
            return ocl.getDeviceDescription();
        }
        throw new IllegalStateException(
                "Unrecognized GpuCompositeExpression implementation: " + compiled.getClass().getName());
    }

    /**
     * Every (platform, GPU device) pair OpenCL can see on this machine, as
     * human-readable descriptions — see
     * {@link OpenClCompositeExpression#listAvailableDevices()}. Call this
     * first, before guessing at a substring to pass to
     * {@link #selectOpenClDevice(String)}.
     *
     * @return
     */
    public static List<String> listOpenClDevices() {
        return OpenClCompositeExpression.listAvailableDevices();
    }

    /**
     * Selects, by vendor/name substring match, which OpenCL device the NEXT
     * compiled {@link ArrowGpuBulkEvaluator} (with an OpenCL-selecting backend)
     * will use. See {@link OpenClCompositeExpression#selectDevice(String)} for
     * the exact matching rules. Already-compiled instances are unaffected.
     *
     * @param vendorOrNameSubstring
     */
    public static void selectOpenClDevice(String vendorOrNameSubstring) {
        OpenClCompositeExpression.selectDevice(vendorOrNameSubstring);
    }

    /**
     * Selects, by known vendor (alias-matched), which OpenCL device the NEXT
     * compiled instance will use. See
     * {@link OpenClCompositeExpression#selectDevice(OpenClCompositeExpression.GpuVendor)}.
     *
     * @param vendor
     */
    public static void selectOpenClDevice(OpenClCompositeExpression.GpuVendor vendor) {
        OpenClCompositeExpression.selectDevice(vendor);
    }

    /**
     * Selects an exact (platform, device) index pair for the NEXT compiled
     * instance. See {@link OpenClCompositeExpression#selectDevice(int, int)}.
     *
     * @param platformIndex
     * @param deviceIndex
     */
    public static void selectOpenClDevice(int platformIndex, int deviceIndex) {
        OpenClCompositeExpression.selectDevice(platformIndex, deviceIndex);
    }

    /**
     * Reverts OpenCL device selection to the default (first GPU device found),
     * for instances compiled after this call. See
     * {@link OpenClCompositeExpression#clearDeviceSelection()}.
     */
    public static void clearOpenClDeviceSelection() {
        OpenClCompositeExpression.clearDeviceSelection();
    }

    /**
     * Every CUDA device this machine's driver can see, as human-readable
     * descriptions — see
     * {@link CudaCompositeExpression#listAvailableDevices()}. Call this first,
     * before guessing at a substring to pass to
     * {@link #selectCudaDevice(String)}.
     *
     * @return
     */
    public static List<String> listCudaDevices() {
        return CudaCompositeExpression.listAvailableDevices();
    }

    /**
     * Selects, by case-insensitive name substring match (e.g. "4080", "A100",
     * "RTX"), which CUDA device the NEXT compiled {@link ArrowGpuBulkEvaluator}
     * (with a CUDA-selecting backend) will use. See
     * {@link CudaCompositeExpression#selectDevice(String)} for the exact
     * matching rules. Already-compiled instances are unaffected.
     *
     * @param nameSubstring
     */
    public static void selectCudaDevice(String nameSubstring) {
        CudaCompositeExpression.selectDevice(nameSubstring);
    }

    /**
     * Selects an exact device index for the NEXT compiled instance. See
     * {@link CudaCompositeExpression#selectDevice(int)}.
     *
     * @param deviceIndex
     */
    public static void selectCudaDevice(int deviceIndex) {
        CudaCompositeExpression.selectDevice(deviceIndex);
    }

    /**
     * Reverts CUDA device selection to the default (device 0), for instances
     * compiled after this call. See
     * {@link CudaCompositeExpression#clearDeviceSelection()}.
     */
    public static void clearCudaDeviceSelection() {
        CudaCompositeExpression.clearDeviceSelection();
    }

    // =========================================================================
    // Introspection
    // =========================================================================
    /**
     * The variable names this expression requires, in no particular order.
     * Every one of these must have a corresponding Arrow column bound at
     * evaluation time, or {@link #evaluate} throws
     * {@link ArrowBindingException}.
     *
     * @return
     */
    @Override
    public String[] requiredVariableNames() {
        return requiredVariableNames.clone();
    }

    /**
     * True if this expression references no variables at all. Such expressions
     * still evaluate correctly via {@link #evaluate} — the output is filled
     * with the single constant value on the CPU, and the GPU is never touched.
     *
     * @return
     */
    @Override
    public boolean isConstantExpression() {
        return constantExpression;
    }

    @Override
    public String getExpressionText() {
        return expression.getExpression();
    }

    // =========================================================================
    // Evaluation — Map<String, Float8Vector> binding
    // =========================================================================
    @Override
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output) {
        evaluate(columns, output, NullPolicy.IGNORE);
    }

    /**
     * Evaluates the compiled expression on the GPU, writing one result per row
     * into {@code output}.
     *
     * <p>
     * <b>Precondition:</b> {@code output} must already be sized —
     * {@code output.allocateNew(rowCount)} and
     * {@code output.setValueCount(rowCount)} must have been called before this
     * method. Use {@link ArrowBulkEvaluator#allocateOutput} if you don't
     * already have an output vector prepared.
     *
     * @param columns Arrow columns, keyed by the variable name they bind to.
     * Must contain an entry for every name in {@link #requiredVariableNames()};
     * extra entries are ignored. Every bound column's {@code getValueCount()}
     * must be at least {@code output.getValueCount()}.
     * @param output pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled — see
     * {@link NullPolicy}
     * @throws ArrowBindingException if a required column is missing, a bound
     * column is shorter than the output, the output has not been sized, or the
     * GPU dispatch itself throws
     */
    @Override
    public void evaluate(Map<String, Float8Vector> columns, Float8Vector output, NullPolicy nullPolicy) {
        ensureOpen();

        int rowCount = output.getValueCount();
        if (rowCount == 0) {
            for (MathExpression.Slot slot : requiredSlots) {
                Float8Vector col = columns.get(slot.getName());
                if (col != null && col.getValueCount() > 0) {
                    throw new ArrowBindingException(
                            "Output vector has not been sized (valueCount=0) but bound column '"
                            + slot.getName() + "' has " + col.getValueCount() + " rows. Call "
                            + "output.allocateNew(rowCount) and output.setValueCount(rowCount) "
                            + "before evaluate(), or use ArrowBulkEvaluator.allocateOutput(...).");
                }
            }
            return; // legitimately empty batch, nothing further to do
        }

        if (constantExpression) {
            fillConstant(output, rowCount);
            return;
        }

        // TRUE zero-copy: each bound column's Arrow data buffer is wrapped as
        // its own MemorySegment -- an alias via ArrowMemoryBridge, never a
        // copy -- and handed straight to GpuCompositeExpression.applyBulk(
        // MemorySegment[], MemorySegment), the multi-segment entry point
        // built specifically to avoid the caller-side flatten this class
        // used to do unconditionally. See the class javadoc for the one
        // narrow edge case (unreferenced registry slots) that still
        // allocates, and why it's safe/necessary.
        MemorySegment[] variableSegments = new MemorySegment[slotCount];
        int filledSlots = 0;
        for (MathExpression.Slot slot : requiredSlots) {
            Float8Vector col = columns.get(slot.getName());
            if (col == null) {
                throw new ArrowBindingException(
                        "Missing Arrow column for variable '" + slot.getName() + "'. Required variables: "
                        + Arrays.toString(requiredVariableNames));
            }
            if (col.getValueCount() < rowCount) {
                throw new ArrowBindingException(
                        "Column '" + slot.getName() + "' has " + col.getValueCount()
                        + " rows, but output expects " + rowCount + " rows.");
            }
            variableSegments[slot.getSlot()] = ArrowMemoryBridge.wrapDoubles(col.getDataBuffer(), rowCount);
            filledSlots++;
        }

        MemorySegment outSeg = ArrowMemoryBridge.wrapDoubles(output.getDataBuffer(), rowCount);

        if (filledSlots == slotCount) {
            // Common case: every registry slot is referenced by this
            // expression -- nothing left to allocate, dispatch directly on
            // column-backed segments only.
            dispatch(variableSegments, outSeg);
        } else {
            // Rare case: expression.getRegistry() is wider than the set of
            // variables THIS expression actually references (a shared/
            // session registry can outlive any one expression).
            // applyBulk(MemorySegment[], ...) scatters every array element
            // to the device unconditionally, so a null entry here would NPE
            // even though the compiled kernel itself never addresses that
            // slot. Back-fill just the unreferenced slots with a small
            // zeroed placeholder, scoped to this one call -- this never
            // touches actual bound column data, and this whole branch is
            // skipped when filledSlots == slotCount.
            try (Arena gapArena = Arena.ofConfined()) {
                for (int i = 0; i < slotCount; i++) {
                    if (variableSegments[i] == null) {
                        variableSegments[i] = gapArena.allocate(ValueLayout.JAVA_DOUBLE, rowCount);
                    }
                }
                dispatch(variableSegments, outSeg);
            }
        }

        if (nullPolicy == NullPolicy.PROPAGATE) {
            propagateNulls(columns, output, rowCount);
        }
    }

    private void dispatch(MemorySegment[] variableSegments, MemorySegment outSeg) {
        synchronized (dispatchLock) {
            try {
                compiled.applyBulk(variableSegments, outSeg);
            } catch (ArrowBindingException e) {
                throw e;
            } catch (Throwable t) {
                throw new ArrowBindingException(
                        "GPU evaluation failed for expression '" + expression.getExpression() + "'", t);
            }
        }
    }

    // =========================================================================
    // Evaluation — VectorSchemaRoot convenience binding
    // =========================================================================
    @Override
    public void evaluate(VectorSchemaRoot root, Float8Vector output) {
        evaluate(root, output, NullPolicy.IGNORE);
    }

    /**
     * Convenience overload that resolves each required variable's column by
     * name from {@code root} instead of a caller-built {@code Map}.
     *
     * @param root
     * @param output
     * @param nullPolicy
     * @throws ArrowBindingException if a required field is missing from
     * {@code root}, or is present but is not a {@link Float8Vector}
     */
    @Override
    public void evaluate(VectorSchemaRoot root, Float8Vector output, NullPolicy nullPolicy) {
        ensureOpen();

        Map<String, Float8Vector> columns = new HashMap<>(Math.max(4, requiredSlots.length * 2));
        for (MathExpression.Slot slot : requiredSlots) {
            FieldVector fv = root.getVector(slot.getName());
            if (fv == null) {
                continue; // reported uniformly by evaluate(Map, ...) above
            }
            if (!(fv instanceof Float8Vector)) {
                throw new ArrowBindingException(
                        "Column '" + slot.getName() + "' must be a Float8Vector (float64) for GPU "
                        + "evaluation; found " + fv.getClass().getSimpleName()
                        + ". Cast this column to float64 before binding.");
            }
            columns.put(slot.getName(), (Float8Vector) fv);
        }

        evaluate(columns, output, nullPolicy);
    }

    /**
     * Convenience overload that resolves each required variable's column by
     * name from {@code root} instead of a caller-built {@code Map}.
     *
     * @param root
     * @param output
     * @param nullPolicy
     * @throws ArrowBindingException if a required field is missing from
     * {@code root}, or is present but is not a {@link Float8Vector}
     */
    @Override
    public void evaluate(VectorSchemaRoot root, Float4Vector output, NullPolicy nullPolicy) {
        ensureOpen();

        Map<String, Float4Vector> columns = new HashMap<>(Math.max(4, requiredSlots.length * 2));
        for (MathExpression.Slot slot : requiredSlots) {
            FieldVector fv = root.getVector(slot.getName());
            if (fv == null) {
                continue; // reported uniformly by evaluate(Map, ...) above
            }
            if (!(fv instanceof Float4Vector)) {
                throw new ArrowBindingException(
                        "Column '" + slot.getName() + "' must be a Float4Vector (float32) for GPU "
                        + "evaluation; found " + fv.getClass().getSimpleName()
                        + ". Cast this column to float32 before binding.");
            }
            columns.put(slot.getName(), (Float4Vector) fv);
        }

        evaluate(columns, output, nullPolicy);
    }

    // =========================================================================
    // Internals
    // =========================================================================
    private void fillConstant(Float8Vector output, int rowCount) {
        double value = expression.solveGeneric().scalar;
        for (int i = 0; i < rowCount; i++) {
            output.set(i, value);
        }
    }

    private void propagateNulls(Map<String, Float8Vector> columns, Float8Vector output, int rowCount) {
        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf outValidity = output.getValidityBuffer();

        for (int i = 0; i < validityBytes; i++) {
            outValidity.setByte(i, (byte) 0xFF);
        }
        for (MathExpression.Slot slot : requiredSlots) {
            Float8Vector col = columns.get(slot.getName());
            ArrowBuf colValidity = col.getValidityBuffer();
            for (int i = 0; i < validityBytes; i++) {
                byte combined = (byte) (outValidity.getByte(i) & colValidity.getByte(i));
                outValidity.setByte(i, combined);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
       /**
     * Evaluates the compiled expression on the GPU, writing one result per
     * row into {@code output}.
     *
     * <p><b>Precondition:</b> {@code output} must already be sized —
     * {@code output.allocateNew(rowCount)} and {@code output.setValueCount(rowCount)}
     * must have been called before this method. Use
     * {@link ArrowBulkEvaluator#allocateOutput} if you don't already have an
     * output vector prepared.
     *
     * @param columns    Arrow columns, keyed by the variable name they bind
     *                   to. Must contain an entry for every name in
     *                   {@link #requiredVariableNames()}; extra entries are
     *                   ignored. Every bound column's {@code getValueCount()}
     *                   must be at least {@code output.getValueCount()}.
     * @param output     pre-sized destination vector
     * @param nullPolicy how Arrow validity bitmaps are handled — see {@link NullPolicy}
     * @throws ArrowBindingException if a required column is missing, a bound
     *                                column is shorter than the output, the
     *                                output has not been sized, or the GPU
     *                                dispatch itself throws
     */
@Override
    public void evaluate(Map<String, Float4Vector> columns, Float4Vector output, NullPolicy nullPolicy) {
        ensureOpen();

        int rowCount = output.getValueCount();
        if (rowCount == 0) {
            for (MathExpression.Slot slot : requiredSlots) {
                Float4Vector col = columns.get(slot.getName());
                if (col != null && col.getValueCount() > 0) {
                    throw new ArrowBindingException(
                            "Output vector has not been sized (valueCount=0) but bound column '"
                            + slot.getName() + "' has " + col.getValueCount() + " rows. Call "
                            + "output.allocateNew(rowCount) and output.setValueCount(rowCount) "
                            + "before evaluate(), or use ArrowBulkEvaluator.allocateOutput(...).");
                }
            }
            return;
        }

        if (constantExpression) {
            fillConstant(output, rowCount);
            return;
        }

        MemorySegment[] variableSegments = new MemorySegment[slotCount];
        int filledSlots = 0;
        for (MathExpression.Slot slot : requiredSlots) {
            Float4Vector col = columns.get(slot.getName());
            if (col == null) {
                throw new ArrowBindingException(
                        "Missing Arrow column for variable '" + slot.getName() + "'. Required variables: "
                        + Arrays.toString(requiredVariableNames));
            }
            if (col.getValueCount() < rowCount) {
                throw new ArrowBindingException(
                        "Column '" + slot.getName() + "' has " + col.getValueCount()
                        + " rows, but output expects " + rowCount + " rows.");
            }
            // FIX 1: Wrap as Floats (4-byte strides) instead of Doubles (8-byte strides)
            variableSegments[slot.getSlot()] = ArrowMemoryBridge.wrapFloats(col.getDataBuffer(), rowCount);
            filledSlots++;
        }

        // FIX 1: Wrap output as Floats
        MemorySegment outSeg = ArrowMemoryBridge.wrapFloats(output.getDataBuffer(), rowCount);

        if (filledSlots == slotCount) {
            dispatchF32(variableSegments, outSeg);
        } else {
            try (Arena gapArena = Arena.ofConfined()) {
                for (int i = 0; i < slotCount; i++) {
                    if (variableSegments[i] == null) {
                        // FIX 1: Allocate JAVA_FLOAT for unreferenced slots
                        variableSegments[i] = gapArena.allocate(ValueLayout.JAVA_FLOAT, rowCount);
                    }
                }
                dispatchF32(variableSegments, outSeg);
            }
        }

        if (nullPolicy == NullPolicy.PROPAGATE) {
            propagateNulls(columns, output, rowCount);
        }
    }

    // FIX 2: Dedicated Float32 dispatch path calling applyBulkF32
    private void dispatchF32(MemorySegment[] variableSegments, MemorySegment outSeg) {
        synchronized (dispatchLock) {
            try {
                compiled.applyBulkF32(variableSegments, outSeg);
            } catch (ArrowBindingException e) {
                throw e;
            } catch (Throwable t) {
                throw new ArrowBindingException(
                        "GPU evaluation failed for expression '" + expression.getExpression() + "'", t);
            }
        }
    }

    // FIX 3: Safe Null Propagation without NPEs
    private void propagateNulls(Map<String, Float4Vector> columns, Float4Vector output, int rowCount) {
        int validityBytes = (rowCount + 7) / 8;
        ArrowBuf outValidity = output.getValidityBuffer();

        for (int i = 0; i < validityBytes; i++) {
            outValidity.setByte(i, (byte) 0xFF);
        }
        for (MathExpression.Slot slot : requiredSlots) {
            Float4Vector col = columns.get(slot.getName());
            if (col == null || col.getNullCount() == 0) {
                continue; // Skip columns with zero nulls or missing validity buffers
            }
            ArrowBuf colValidity = col.getValidityBuffer();
            if (colValidity == null) {
                continue;
            }
            for (int i = 0; i < validityBytes; i++) {
                byte combined = (byte) (outValidity.getByte(i) & colValidity.getByte(i));
                outValidity.setByte(i, combined);
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    private void fillConstant(Float4Vector output, int rowCount) {
        float value = (float) expression.solveGeneric().scalar;
        for (int i = 0; i < rowCount; i++) {
            output.set(i, value);
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("ArrowGpuBulkEvaluator has been closed: " + expression.getExpression());
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            compiled.close();
        }
    }
}
