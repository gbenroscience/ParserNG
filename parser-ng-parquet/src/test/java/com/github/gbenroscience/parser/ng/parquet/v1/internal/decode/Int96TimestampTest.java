package com.github.gbenroscience.parser.ng.parquet.v1.internal.decode;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Int96TimestampTest {

    /** Splits an {@link Instant} into (nanos-of-day, Julian day) exactly the way an INT96 writer would,
     *  independently of {@link Int96Timestamp}, so this is a genuine cross-check rather than a tautology. */
    private static long[] toNanosOfDayAndJulianDay(Instant instant) {
        long epochSecond = instant.getEpochSecond();
        long epochDay = Math.floorDiv(epochSecond, 86400L);
        long secondOfDay = Math.floorMod(epochSecond, 86400L);
        long nanosOfDay = secondOfDay * 1_000_000_000L + instant.getNano();
        long julianDay = epochDay + 2_440_588L; // Julian day number of 1970-01-01
        return new long[]{nanosOfDay, julianDay};
    }

    @Test
    void knownInstantRoundTrips() {
        Instant instant = Instant.parse("2024-01-15T12:34:56.789012345Z");
        long[] split = toNanosOfDayAndJulianDay(instant);
        long got = Int96Timestamp.toEpochNanos(split[0], (int) split[1]);
        long want = instant.getEpochSecond() * 1_000_000_000L + instant.getNano();
        assertEquals(want, got);
    }

    @Test
    void unixEpochItself() {
        Instant instant = Instant.EPOCH;
        long[] split = toNanosOfDayAndJulianDay(instant);
        assertEquals(0L, Int96Timestamp.toEpochNanos(split[0], (int) split[1]));
    }

    @Test
    void beforeTheEpoch() {
        Instant instant = Instant.parse("1969-07-20T20:17:00Z"); // Apollo 11 landing
        long[] split = toNanosOfDayAndJulianDay(instant);
        long got = Int96Timestamp.toEpochNanos(split[0], (int) split[1]);
        long want = instant.getEpochSecond() * 1_000_000_000L + instant.getNano();
        assertEquals(want, got);
        assertTrue(got < 0);
    }

    @Test
    void midnightHasZeroNanosOfDay() {
        LocalDate date = LocalDate.of(2030, 3, 1);
        long julianDay = date.toEpochDay() + 2_440_588L;
        long got = Int96Timestamp.toEpochNanos(0L, (int) julianDay);
        assertEquals(date.toEpochDay() * 86_400_000_000_000L, got);
    }
}
