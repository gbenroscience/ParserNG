package com.github.gbenroscience.sqlv1.parquet;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.IsoFields;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Calendar arithmetic, checked against {@code java.time}. */
class CivilTimeTest {

    private static final long DAY = CivilTime.DAY_MICROS;

    @Test
    void civilDatesRoundTripAndEveryExtractFieldMatchesJavaTime() {
        Random r = new Random(11);
        long[] sc = new long[3];
        for (int i = 0; i < 30_000; i++) {
            long days = r.nextInt(400_000) - 100_000;
            LocalDate d = LocalDate.ofEpochDay(days);
            CivilTime.civilFromDays(days, sc);
            assertEquals(d.getYear(), sc[0]);
            assertEquals(d.getMonthValue(), sc[1]);
            assertEquals(d.getDayOfMonth(), sc[2]);
            assertEquals(days, CivilTime.daysFromCivil(d.getYear(), d.getMonthValue(), d.getDayOfMonth()));
            long micros = days * DAY + Math.floorMod(r.nextLong(), DAY);
            LocalDateTime t = LocalDateTime.ofEpochSecond(Math.floorDiv(micros, 1_000_000L), (int) Math.floorMod(micros, 1_000_000L) * 1000, ZoneOffset.UTC);
            assertEquals(t.getYear(), CivilTime.extract(CivilTime.Field.YEAR, micros, sc));
            assertEquals(t.getMonthValue(), CivilTime.extract(CivilTime.Field.MONTH, micros, sc));
            assertEquals((t.getMonthValue() - 1) / 3 + 1, CivilTime.extract(CivilTime.Field.QUARTER, micros, sc));
            assertEquals(t.getDayOfMonth(), CivilTime.extract(CivilTime.Field.DAY, micros, sc));
            assertEquals(t.getDayOfYear(), CivilTime.extract(CivilTime.Field.DOY, micros, sc));
            assertEquals(t.getHour(), CivilTime.extract(CivilTime.Field.HOUR, micros, sc));
            assertEquals(t.getMinute(), CivilTime.extract(CivilTime.Field.MINUTE, micros, sc));
            assertEquals(t.getSecond() + t.getNano() / 1e9, CivilTime.extract(CivilTime.Field.SECOND, micros, sc), 1e-9);
            assertEquals(t.getDayOfWeek().getValue(), CivilTime.extract(CivilTime.Field.ISODOW, micros, sc));
            assertEquals(t.getDayOfWeek().getValue() % 7, CivilTime.extract(CivilTime.Field.DOW, micros, sc));
            assertEquals(t.get(IsoFields.WEEK_OF_WEEK_BASED_YEAR), CivilTime.extract(CivilTime.Field.WEEK, micros, sc), "week of " + t);
            assertEquals(micros / 1e6, CivilTime.extract(CivilTime.Field.EPOCH, micros, sc), 1e-6);
        }
    }

    @Test
    void intervalAdditionMatchesJavaTimeAndClampsMonthEnds() {
        Random r = new Random(5);
        for (int i = 0; i < 20_000; i++) {
            LocalDateTime base = LocalDateTime.ofEpochSecond((r.nextInt(60_000) - 10_000) * 86_400L + r.nextInt(86_400), r.nextInt(1_000_000) * 1000, ZoneOffset.UTC);
            long micros = base.toEpochSecond(ZoneOffset.UTC) * 1_000_000L + base.getNano() / 1000;
            int mo = r.nextInt(60) - 30, dd = r.nextInt(100) - 50, hh = r.nextInt(200) - 100;
            LocalDateTime e = base.plusMonths(mo).plusDays(dd).plusHours(hh);
            long want = e.toEpochSecond(ZoneOffset.UTC) * 1_000_000L + e.getNano() / 1000;
            assertEquals(want, CivilTime.addInterval(micros, new CivilTime.Interval(mo, dd * DAY + hh * 3_600_000_000L)), base + " " + mo + "m " + dd + "d " + hh + "h");
        }
        byte[] jan31 = "2024-01-31".getBytes();
        byte[] feb29 = "2024-02-29".getBytes();
        assertEquals(CivilTime.parse(feb29, 10, true, "x"),
                CivilTime.addInterval(CivilTime.parse(jan31, 10, true, "x"), CivilTime.interval("1 month", null)));
    }

    @Test
    void intervalTextForms() {
        assertEquals(new CivilTime.Interval(14, 3 * DAY), CivilTime.interval("1 year 2 months 3 days", null));
        assertEquals(new CivilTime.Interval(0, 5 * DAY), CivilTime.interval("5", "day"));
        assertEquals(-3 * DAY, CivilTime.interval("-3 days", null).micros());
        assertEquals(5_400_000_000L, CivilTime.interval("1.5 hours", null).micros());
        assertTrue(assertThrows(ParquetSqlException.class, () -> CivilTime.interval("1.5 months", null)).getMessage().contains("whole"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> CivilTime.interval("3 fortnights", null)).getMessage().contains("not supported"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> CivilTime.interval("3", null)).getMessage().contains("pairs"));
    }

    @Test
    void parsingIsStrictAndNamesTheValue() {
        for (String g : new String[] {"2024-02-29", "1999-12-31", "0999-01-01"}) {
            assertEquals(LocalDate.parse(g).toEpochDay() * DAY, CivilTime.parse(g.getBytes(), g.length(), true, "x"), g);
        }
        Random r = new Random(3);
        for (int i = 0; i < 5_000; i++) {
            LocalDateTime x = LocalDateTime.ofEpochSecond(r.nextInt(2_000_000_000) - 500_000_000L, r.nextInt(1_000_000) * 1000, ZoneOffset.UTC);
            String s = x.toLocalDate() + (r.nextBoolean() ? " " : "T") + String.format("%02d:%02d:%02d", x.getHour(), x.getMinute(), x.getSecond())
                    + (x.getNano() == 0 ? "" : "." + String.format("%06d", x.getNano() / 1000));
            assertEquals(x.toEpochSecond(ZoneOffset.UTC) * 1_000_000L + x.getNano() / 1000, CivilTime.parse(s.getBytes(), s.length(), false, "x"), s);
        }
        for (String bad : new String[] {"2024-02-30", "2024-13-01", "24-01-01", "2024-1-01", "", "abc", "2024-01-01 25:00:00", "2024-01-01 10:61", "2024-01-01 10:00:00.1234567"}) {
            ParquetSqlException e = assertThrows(ParquetSqlException.class, () -> CivilTime.parse(bad.getBytes(), bad.length(), false, "CAST(c AS TIMESTAMP)"), bad);
            assertTrue(e.getMessage().contains("CAST(c AS TIMESTAMP)") && e.getMessage().contains("not a valid"), e.getMessage());
        }
        byte[] timePart = "2024-01-01 10:00".getBytes();
        assertTrue(assertThrows(ParquetSqlException.class, () -> CivilTime.parse(timePart, timePart.length, true, "x")).getMessage().contains("not a valid DATE"));
        byte[] padded = "  2024-03-04  ".getBytes();
        assertEquals(LocalDate.of(2024, 3, 4).toEpochDay() * DAY, CivilTime.parse(padded, padded.length, true, "x"));
        byte[] zeros = "2024-01-01 10:00:00.1234560".getBytes();
        assertEquals(CivilTime.parse("2024-01-01 10:00:00.123456".getBytes(), 26, false, "x"), CivilTime.parse(zeros, zeros.length, false, "x"),
                "trailing zero digits beyond microseconds lose nothing");
    }

    @Test
    void extractFieldNamesAndAliases() {
        assertSame(CivilTime.Field.YEAR, CivilTime.Field.of("year"));
        assertSame(CivilTime.Field.DOW, CivilTime.Field.of("dayofweek"));
        assertTrue(assertThrows(ParquetSqlException.class, () -> CivilTime.Field.of("fortnight")).getMessage().contains("supported"));
    }
}
