package com.github.gbenroscience.parser.ng.parquet.v1;

import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.column.statistics.Statistics;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ColumnChunkMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.schema.Type;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Footer-only metadata inspection: reads the footer, never a data page. */
public record ParquetFileInfo(
        Path file, long fileSizeBytes, long rowCount, String schema, String createdBy,
        Map<String, String> keyValueMetadata, List<RowGroupInfo> rowGroups,
        List<String> topLevelColumns, List<String> primitiveColumns) {

    /**
     * @param topLevelColumns every top-level field name, in file order
     * @param primitiveColumns the subset that are non-repeated primitives (the ones a flat scan can read)
     */

    public record RowGroupInfo(int index, long rowCount, long totalByteSize, List<ColumnChunkInfo> columns) { }

    /**
     * @param min/max/nullCount null when the writer recorded no (usable) statistics
     * @param hasColumnIndex/hasOffsetIndex/hasBloomFilter whether the file carries those structures
     */
    public record ColumnChunkInfo(String path, String primitiveType, String codec, List<String> encodings,
                                  long valueCount, long compressedSize, long uncompressedSize,
                                  String min, String max, Long nullCount,
                                  boolean hasColumnIndex, boolean hasOffsetIndex, boolean hasBloomFilter) { }

    public static ParquetFileInfo read(Path file) {
        try (ParquetFileReader r = ParquetFileReader.open(new LocalInputFile(file), ParquetReadOptions.builder().build())) {
            ParquetMetadata md = r.getFooter();
            List<RowGroupInfo> rgs = new ArrayList<>();
            int i = 0;
            for (BlockMetaData b : md.getBlocks()) {
                List<ColumnChunkInfo> cols = new ArrayList<>();
                for (ColumnChunkMetaData c : b.getColumns()) {
                    Statistics<?> s = c.getStatistics();
                    boolean usable = s != null && !s.isEmpty() && s.hasNonNullValue();
                    cols.add(new ColumnChunkInfo(
                            c.getPath().toDotString(), c.getPrimitiveType().getPrimitiveTypeName().name(),
                            c.getCodec().name(),
                            c.getEncodings().stream().map(Enum::name).sorted().collect(Collectors.toList()),
                            c.getValueCount(), c.getTotalSize(), c.getTotalUncompressedSize(),
                            usable ? s.minAsString() : null, usable ? s.maxAsString() : null,
                            (s != null && s.isNumNullsSet()) ? s.getNumNulls() : null,
                            c.getColumnIndexReference() != null, c.getOffsetIndexReference() != null,
                            c.getBloomFilterOffset() >= 0));
                }
                rgs.add(new RowGroupInfo(i++, b.getRowCount(), b.getTotalByteSize(), List.copyOf(cols)));
            }
            long rows = md.getBlocks().stream().mapToLong(BlockMetaData::getRowCount).sum();
            return new ParquetFileInfo(file, Files.size(file), rows, md.getFileMetaData().getSchema().toString(),
                    md.getFileMetaData().getCreatedBy(), Map.copyOf(md.getFileMetaData().getKeyValueMetaData()),
                    List.copyOf(rgs),
                    md.getFileMetaData().getSchema().getFields().stream().map(Type::getName).toList(),
                    md.getFileMetaData().getSchema().getFields().stream()
                            .filter(t -> t.isPrimitive() && t.getRepetition() != Type.Repetition.REPEATED)
                            .map(Type::getName).toList());
        } catch (IOException | RuntimeException e) {
            throw new ParquetScanException("Cannot read Parquet footer", file, -1, null, e);
        }
    }
}
