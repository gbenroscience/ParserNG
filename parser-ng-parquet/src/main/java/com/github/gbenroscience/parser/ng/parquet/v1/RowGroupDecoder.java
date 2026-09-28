package com.github.gbenroscience.parser.ng.parquet.v1;

import com.github.gbenroscience.parser.ng.parquet.v1.internal.LevelWalker;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.NodePlan;
import com.github.gbenroscience.parser.ng.parquet.v1.internal.decode.FastColumnCursor;
import org.apache.arrow.vector.FieldVector; 
import org.apache.arrow.vector.complex.ListVector;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnReader;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ColumnChunkMetaData;
import org.apache.parquet.hadoop.metadata.ColumnPath;
import org.apache.parquet.schema.MessageType;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.arrow.vector.complex.StructVector;

/**
 * Decodes row groups into Arrow vectors. Owns one {@link ParquetFileReader} (its own stream and codec
 * state), so ONE INSTANCE PER THREAD. Not thread-safe; never share an instance.
 *
 * <p>Flat top-level columns take the direct primitive loop. Nested columns are read entry by entry into
 * the leaf vectors while the level streams of one representative leaf per container are kept, then the
 * {@link LevelWalker} rebuilds list offsets and struct validity from them.
 *
 * <h2>Per-column decode engine: {@link FastColumnCursor}, not {@code ColumnReadStoreImpl}</h2>
 * {@link #begin} constructs one {@link FastColumnCursor} per projected column directly against the
 * row group's {@link PageReadStore} — no {@code ColumnReadStoreImpl}, no {@code GroupConverter} tree,
 * no record-assembly layer at all. This replaces an earlier revision that used parquet-column's own
 * {@code ColumnReadStoreImpl} (driven by a hand-built no-op converter tree) purely to obtain
 * {@code ColumnReader} instances whose values were then read through the typed getters below anyway,
 * never through a {@code Converter}. {@link FastColumnCursor} genuinely implements
 * {@code ColumnReader} (see its own class Javadoc), so this file — {@link #readLeaf}, {@link #fill},
 * {@code internal.ColumnPlan}, {@code internal.NodePlan}, {@code internal.LevelWalker} — did not need
 * to change at all beyond {@link #begin}'s construction of {@link #readers}: everything downstream is
 * already typed against the {@code ColumnReader} interface, not a concrete class, which is exactly
 * what makes this swap possible without touching the nested-type assembly logic.
 *
 * <h2>Page-level (column-index) pruning</h2>
 * Both {@link #nextFilteredRowGroup()} and {@link #readRowGroup} go through parquet-java's
 * <em>filtered</em> row-group entry points ({@code readNextFilteredRowGroup()} /
 * {@code readFilteredRowGroup(int)}), not the unfiltered ones: a row group that survives
 * row-group-level pruning (stats/dictionary/Bloom filter, decided once up front) can still have
 * individual pages skipped inside it via the file's column indexes, when the reader was opened with
 * {@code useColumnIndexFilter(true)} and a record filter. The row count this class ever sees
 * ({@code pages.getRowCount()} in {@link #begin}, {@code cr.getTotalValueCount()} in
 * {@link #readLeaf}) is read directly off the {@link PageReadStore}/{@link ColumnReader} that
 * pruning already produced, never off static footer metadata — so this decode loop needs no special
 * case for "some pages were skipped": a page-pruned row group simply presents fewer values to begin
 * with, through the exact same calls as an unpruned one. This holds identically for
 * {@link FastColumnCursor}, since it reads its own value/level counts off the {@link PageReadStore}
 * it was constructed against, not off static footer metadata either.
 *
 * <p><b>One interaction worth a targeted test before relying on it under real load:</b> Parquet's
 * column-index page-skip range computation is defined and verified against flat, non-repeated
 * columns; this module has not been able to confirm (no compiler, no test run available while
 * writing this) that parquet-java computes equally precise skip ranges for a row group that also
 * contains a {@code LIST} or {@code struct} column in the read set, where a page boundary does not
 * necessarily align with a record boundary the same way it does for a flat column. The safe case
 * either way: parquet-java either computes a correct (possibly less precise) row range for such a
 * row group, or does not attempt page pruning for it — both leave this class's own bookkeeping
 * intact. What is <em>not</em> yet verified is whether pruning is as <em>effective</em> in that
 * mixed case as it is for an all-flat row group.
 */
final class RowGroupDecoder implements AutoCloseable {

    private final Path file;
    private final ParquetFileReader reader;
    private final MessageType projected;
    private final NodePlan nodePlan;
    private final ColumnDescriptor[] descs;
    private final long maxRowGroupBytes;
    private final ScanMetrics metrics; // may be null

    private final ColumnReader[] readers;
    private final FastColumnCursor[] cursors;   // same objects as readers[]; typed so page-byte counters are reachable
    private final Set<ColumnPath> projectedPaths; // leaf paths of the projection; null when metrics are off
    private final long[] consumed;
    private final int[][] repBuf;
    private final int[][] defBuf;
    private final int[] entryCounts;
    private final int[] slotCounts;
    private final int[] tmp = new int[2];
    private final ListAdapter listAdapter = new ListAdapter();
    private final StructAdapter structAdapter = new StructAdapter();

    private long remaining;
    private int rgIndex = -1;
    private long pageBytesReported; // portion of the current row group's cursors' pageBytesLoaded already added to metrics

    RowGroupDecoder(Path file, ParquetFileReader reader, MessageType projected, NodePlan nodePlan,
                    ColumnDescriptor[] descs, long maxRowGroupBytes, ScanMetrics metrics) {
        this.file = file;
        this.reader = reader;
        this.projected = projected;
        this.nodePlan = nodePlan;
        this.descs = descs;
        this.maxRowGroupBytes = maxRowGroupBytes;
        this.metrics = metrics;
        this.readers = new ColumnReader[descs.length];
        this.cursors = new FastColumnCursor[descs.length];
        if (metrics != null) {
            this.projectedPaths = new HashSet<>();
            for (ColumnDescriptor d : descs) projectedPaths.add(ColumnPath.get(d.getPath()));
        } else {
            this.projectedPaths = null;
        }
        this.consumed = new long[descs.length];
        this.repBuf = new int[descs.length][];
        this.defBuf = new int[descs.length][];
        this.entryCounts = new int[descs.length];
        this.slotCounts = new int[nodePlan.nodeCount()];
        try {
            reader.setRequestedSchema(projected);
        } catch (RuntimeException e) {
            throw new ParquetScanException("Cannot apply column projection", file, -1, null, e);
        }
    }

    long remaining() { return remaining; }

    /**
     * Sequential mode: next row group that survived the reader's own row-group-level filter, with
     * page-level (column-index) pruning applied within it if the reader was opened with
     * {@code useColumnIndexFilter(true)} and a record filter.
     */
    boolean nextFilteredRowGroup() throws IOException {
        PageReadStore pages = reader.readNextFilteredRowGroup();
        if (pages == null) return false;
        rgIndex++;
        begin(pages, reader.getRowGroups().get(rgIndex));
        return true;
    }

    /**
     * Parallel mode: a specific row group by its ordinal in the footer, with the same page-level
     * pruning as {@link #nextFilteredRowGroup()} — the reader was opened with the record filter and
     * {@code useColumnIndexFilter(true)} exactly like the sequential path; only the access pattern
     * (random by ordinal, for a pooled/reused reader) differs.
     */
    void readRowGroup(int footerOrdinal, int survivorIndex) throws IOException {
        PageReadStore pages = reader.readFilteredRowGroup(footerOrdinal);
        rgIndex = survivorIndex;
        begin(pages, reader.getFooter().getBlocks().get(footerOrdinal));
    }

    private void begin(PageReadStore pages, BlockMetaData block) {
        if (block.getTotalByteSize() > maxRowGroupBytes) {
            throw new ParquetScanException("Row group declares " + block.getTotalByteSize()
                    + " uncompressed bytes, above the configured limit " + maxRowGroupBytes, file, rgIndex, null, null);
        }
        for (int c = 0; c < descs.length; c++) {
            String colName = String.join(".", descs[c].getPath());
            try {
                cursors[c] = new FastColumnCursor(descs[c], pages, file, colName);
                readers[c] = cursors[c];
            } catch (RuntimeException e) {
                // Constructing a cursor decodes the column's dictionary page (if any) and its first
                // data page eagerly (see FastColumnCursor's constructor), so a corrupt/desynced page
                // can throw here, before fill() is ever called -- this call site had no file/column/
                // row-group context attached until now, which is why a page-level decode failure here
                // previously surfaced as a raw, unattributed exception (see ByteReader.Corrupt).
                throw new ParquetScanException("Failed to begin decoding column '" + colName
                        + "' (row group " + rgIndex + ")", file, rgIndex, colName, e);
            }
        }
        Arrays.fill(consumed, 0L);
        remaining = pages.getRowCount();
        if (metrics != null) {
            metrics.rowGroupsRead.increment();
            // On-disk (compressed) size of exactly the projected column chunks of this row group, from the footer.
            // Upper bound when column-index pruning skips pages inside it -- see ScanMetrics#compressedBytesRead.
            long compressed = 0;
            for (ColumnChunkMetaData cc : block.getColumns()) {
                if (projectedPaths.contains(cc.getPath())) compressed += cc.getTotalSize();
            }
            metrics.compressedBytesRead.add(compressed);
            pageBytesReported = 0L;
            reportPageBytes(); // dictionary pages + each column's eagerly-loaded first data page
        }
    }

    /** Allocates or clears vectors so each can hold {@code cap} top-level rows. */
    static void prepare(FieldVector[] vecs, int cap) {
        for (FieldVector v : vecs) {
            // Capacity check also covers buffers moved away by detach() or a downstream transfer.
            if (v.getValueCapacity() < cap) {
                v.clear();
                v.setInitialCapacity(cap);
                v.allocateNew();
            } else {
                v.reset();
            }
        }
    }

    /** Decodes the next {@code n} records of the current row group into {@code vecs} (already prepared). */
    void fill(FieldVector[] vecs, int n) {
        if (n <= 0 || n > remaining) throw new IllegalArgumentException("bad batch size " + n + " (remaining " + remaining + ")");
        long t0 = metrics != null ? System.nanoTime() : 0L;
        NodePlan.Node[] tops = nodePlan.tops();
        for (int t = 0; t < tops.length; t++) {
            NodePlan.Node node = tops[t];
            FieldVector v = vecs[t];
            try {
                if (node instanceof NodePlan.Leaf leaf) {
                    leaf.plan.fill(readers[leaf.leafIndex], n, v); // flat: direct primitive loop
                    v.setValueCount(n);
                } else {
                    readTree(node, v, n);
                    finishTree(node, v);
                    if (slotCounts[node.id] != n) {
                        throw corrupt(node.name, "produced " + slotCounts[node.id] + " rows, expected " + n);
                    }
                }
            } catch (ParquetScanException e) {
                throw e;
            } catch (RuntimeException e) {
                throw new ParquetScanException("Failed decoding column", file, rgIndex, node.name, e);
            }
        }
        remaining -= n;
        if (metrics != null) {
            metrics.decodeNanos.add(System.nanoTime() - t0);
            metrics.rowsRead.add(n);
            metrics.batches.increment();
            reportPageBytes(); // pages pulled in lazily while filling this batch
            long arrow = 0L;
            for (FieldVector v : vecs) arrow += v.getBufferSize();
            metrics.arrowBytesProduced.add(arrow);
        }
    }

    /** Adds the page bytes the current row group's cursors have loaded since the last call (delta, so nothing is double counted). */
    private void reportPageBytes() {
        long loaded = 0L;
        for (FastColumnCursor c : cursors) loaded += c.pageBytesLoaded();
        metrics.uncompressedBytesDecoded.add(loaded - pageBytesReported);
        pageBytesReported = loaded;
    }

    // ------------------------------------------------------------ nested reading

    private void readTree(NodePlan.Node node, FieldVector v, int n) {
        if (node instanceof NodePlan.Leaf leaf) {
            readLeaf(leaf, v, n);
        } else if (node instanceof NodePlan.Struct s) {
            List<FieldVector> kids = v.getChildrenFromFields();
            for (int i = 0; i < s.children.length; i++) readTree(s.children[i], kids.get(i), n);
        } else {
            readTree(((NodePlan.Listy) node).element, v.getChildrenFromFields().get(0), n);
        }
    }

    /**
     * Reads {@code nRecords} records of one leaf. A record is the run of entries up to (excluding) the next
     * entry with repetition level 0. Writes present values straight into {@code v}; keeps the levels only
     * if a container above needs them.
     */
    private void readLeaf(NodePlan.Leaf leaf, FieldVector v, int nRecords) {
        final int li = leaf.leafIndex;
        final ColumnReader cr = readers[li];
        final long total = cr.getTotalValueCount();
        final boolean retain = leaf.retainLevels();
        final int reach = leaf.reach, dmax = leaf.def;
        int[] rb = null, db = null;
        if (retain) {
            if (repBuf[li] == null) { repBuf[li] = new int[1024]; defBuf[li] = new int[1024]; }
            rb = repBuf[li];
            db = defBuf[li];
        }
        long done = consumed[li];
        int entries = 0, slots = 0;
        for (int rec = 0; rec < nRecords; rec++) {
            if (done >= total) throw corrupt(leaf.name, "column ended before record " + rec);
            boolean first = true;
            while (true) {
                final int rep = cr.getCurrentRepetitionLevel();
                final int def = cr.getCurrentDefinitionLevel();
                if (first) {
                    if (rep != 0) throw corrupt(leaf.name, "record does not start with repetition level 0");
                    first = false;
                } else if (rep == 0) {
                    break; // next record begins
                }
                if (retain) {
                    if (entries == rb.length) {
                        rb = Arrays.copyOf(rb, entries * 2);
                        db = Arrays.copyOf(db, entries * 2);
                        repBuf[li] = rb;
                        defBuf[li] = db;
                    }
                    rb[entries] = rep;
                    db[entries] = def;
                }
                entries++;
                if (def >= reach) {
                    if (def >= dmax) leaf.plan.writeSafe(cr, slots, v);
                    slots++;
                }
                cr.consume();
                done++;
                if (done >= total) break;
            }
        }
        consumed[li] = done;
        entryCounts[li] = entries;
        slotCounts[leaf.id] = slots;
        v.setValueCount(slots);
    }

    private void finishTree(NodePlan.Node node, FieldVector v) {
        if (node instanceof NodePlan.Leaf) return; // done in readLeaf
        List<FieldVector> kids = v.getChildrenFromFields();
        int fl = node.firstLeaf();
        if (node instanceof NodePlan.Struct s) {
            for (int i = 0; i < s.children.length; i++) finishTree(s.children[i], kids.get(i));
            structAdapter.target = v instanceof StructVector sv ? sv : null;
            int slots = LevelWalker.walkStruct(repBuf[fl], defBuf[fl], entryCounts[fl], s.rep, s.reach, s.def, structAdapter);
            structAdapter.target = null;
            for (NodePlan.Node c : s.children) {
                if (slotCounts[c.id] != slots) throw corrupt(s.name, "child '" + c.name + "' has " + slotCounts[c.id] + " slots, struct has " + slots);
            }
            slotCounts[s.id] = slots;
            v.setValueCount(slots);
        } else {
            NodePlan.Listy l = (NodePlan.Listy) node;
            finishTree(l.element, kids.get(0));
            listAdapter.target = (ListVector) v; // MapVector extends ListVector
            LevelWalker.walkList(repBuf[fl], defBuf[fl], entryCounts[fl], l.rep, l.reach, l.def, l.repR, l.defR, listAdapter, tmp);
            listAdapter.target = null;
            if (tmp[1] != slotCounts[l.element.id]) {
                throw corrupt(l.name, "list holds " + tmp[1] + " elements but its element column has " + slotCounts[l.element.id] + " slots");
            }
            slotCounts[l.id] = tmp[0];
            v.setValueCount(tmp[0]);
        }
    }

    private ParquetScanException corrupt(String column, String why) {
        return new ParquetScanException("Inconsistent repetition/definition levels: " + why, file, rgIndex, column, null);
    }

    private static final class ListAdapter implements LevelWalker.ListEvents {
        ListVector target;
        @Override public void nullList(int i) { target.setNull(i); }
        @Override public void list(int i, int size) { target.startNewValue(i); target.endValue(i, size); }
    }

    private static final class StructAdapter implements LevelWalker.StructEvents {
        StructVector target; // null for a non-nullable struct vector: no validity to write
        @Override public void slot(int i, boolean valid) {
            if (target == null) return;
            if (valid) target.setIndexDefined(i); else target.setNull(i);
        }
    }

    @Override
    public void close() {
        try { reader.close(); } catch (IOException | RuntimeException ignored) { }
    }
}