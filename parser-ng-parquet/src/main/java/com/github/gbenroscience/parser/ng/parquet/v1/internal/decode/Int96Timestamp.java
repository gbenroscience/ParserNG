package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

/**
 * Converts Parquet's deprecated INT96 physical type to epoch nanoseconds, using the convention every
 * major INT96-writing engine (Impala, Hive, old Spark) actually wrote: 12 bytes, PLAIN-encoded as
 * &lt;8-byte little-endian nanoseconds-of-day&gt; &lt;4-byte little-endian Julian day number&gt;. There is no
 * logical-type annotation on INT96 (it predates {@code LogicalTypeAnnotation} entirely) -- this
 * convention is the only reason INT96 is readable as a timestamp at all, and is exactly what
 * {@code ColumnPlan.forPrimitive} assumes when it maps INT96 to {@code ArrowType.Timestamp(NANOSECOND,
 * "UTC")} (see its Javadoc).
 *
 * <p>Not a general calendar utility: only handles the one conversion this column type needs.
 */
final class Int96Timestamp {

    private Int96Timestamp() { }

    /** Julian day number of 1970-01-01 (the Unix epoch), i.e. this day number converts to epoch day 0. */
    private static final long JULIAN_EPOCH_DAY = 2_440_588L;
    private static final long NANOS_PER_DAY = 86_400_000_000_000L;

    /** {@code nanosOfDay} in [0, NANOS_PER_DAY); {@code julianDay} the Julian day number. Returns epoch nanoseconds (UTC). */
    static long toEpochNanos(long nanosOfDay, int julianDay) {
        return (julianDay - JULIAN_EPOCH_DAY) * NANOS_PER_DAY + nanosOfDay;
    }
}
