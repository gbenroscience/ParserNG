package com.github.gbenroscience.sqlv1.parquet;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Calendar arithmetic for {@link TemporalFns}, with no Arrow dependency: civil dates from and to epoch days,
 * the {@code EXTRACT} fields, {@code INTERVAL} parsing and addition, and strict parsing of date and timestamp
 * text from bytes. Every temporal value is microseconds since the epoch. Nothing here allocates per call
 * except on the error path.
 */
final class CivilTime {

    private CivilTime() { }

    static final long SEC_MICROS = 1_000_000L;

    static final long DAY_MICROS = 86_400_000_000L;

    // ------------------------------------------------------------------ calendar arithmetic (no allocation)

    /** Days since 1970-01-01 of the proleptic Gregorian date. */
    static long daysFromCivil(long y, int m, int d) {
        y -= m <= 2 ? 1 : 0;
        long era = Math.floorDiv(y, 400);
        long yoe = y - era * 400;
        long doy = (153L * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097 + doe - 719468;
    }

    /** {@code {year, month, day}} of an epoch day, written into {@code out}. */
    static void civilFromDays(long days, long[] out) {
        long z = days + 719468;
        long era = Math.floorDiv(z, 146097);
        long doe = z - era * 146097;
        long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        long y = yoe + era * 400;
        long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        long mp = (5 * doy + 2) / 153;
        long d = doy - (153 * mp + 2) / 5 + 1;
        long m = mp < 10 ? mp + 3 : mp - 9;
        out[0] = y + (m <= 2 ? 1 : 0);
        out[1] = m;
        out[2] = d;
    }

    private static boolean leap(long y) {
        return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    }

    private static int monthLength(long y, int m) {
        return switch (m) {
            case 2 -> leap(y) ? 29 : 28;
            case 4, 6, 9, 11 -> 30;
            default -> 31;
        };
    }

    private static int isoWeeksInYear(long y) {
        long p = Math.floorMod(y + Math.floorDiv(y, 4) - Math.floorDiv(y, 100) + Math.floorDiv(y, 400), 7);
        long q = Math.floorMod(y - 1 + Math.floorDiv(y - 1, 4) - Math.floorDiv(y - 1, 100) + Math.floorDiv(y - 1, 400), 7);
        return 52 + (p == 4 || q == 3 ? 1 : 0);
    }

    enum Field {
        YEAR, QUARTER, MONTH, WEEK, DAY, DOW, ISODOW, DOY, HOUR, MINUTE, SECOND, MILLISECOND, MICROSECOND, EPOCH;

        static Field of(String name) {
            String n = name.toUpperCase(Locale.ROOT);
            switch (n) {
                case "MILLISECONDS": n = "MILLISECOND"; break;
                case "MICROSECONDS": n = "MICROSECOND"; break;
                case "DAYOFWEEK": n = "DOW"; break;
                case "DAYOFYEAR": n = "DOY"; break;
                case "ISOWEEK": n = "WEEK"; break;
                default: break;
            }
            try {
                return valueOf(n);
            } catch (IllegalArgumentException e) {
                throw new ParquetSqlException("EXTRACT does not support the field '" + name + "'; supported: "
                        + java.util.Arrays.toString(values()));
            }
        }
    }

    /** The {@code field} of the instant {@code micros} (microseconds since the epoch), as PostgreSQL's EXTRACT defines it. */
    static double extract(Field f, long micros, long[] scratch) {
        long days = Math.floorDiv(micros, DAY_MICROS);
        long inDay = Math.floorMod(micros, DAY_MICROS);
        switch (f) {
            case EPOCH:
                return micros / 1e6;
            case HOUR:
                return inDay / 3_600_000_000L;
            case MINUTE:
                return (inDay / 60_000_000L) % 60;
            case SECOND:
                return (inDay % 60_000_000L) / 1e6;
            case MILLISECOND:
                return (inDay % 60_000_000L) / 1e3;
            case MICROSECOND:
                return (double) (inDay % 60_000_000L);
            case DOW:
                return Math.floorMod(days + 4, 7);
            case ISODOW: {
                long dow = Math.floorMod(days + 4, 7);
                return dow == 0 ? 7 : dow;
            }
            default:
                break;
        }
        civilFromDays(days, scratch);
        long y = scratch[0];
        int m = (int) scratch[1];
        switch (f) {
            case YEAR:
                return y;
            case QUARTER:
                return (m - 1) / 3 + 1;
            case MONTH:
                return m;
            case DAY:
                return scratch[2];
            case DOY:
                return days - daysFromCivil(y, 1, 1) + 1;
            case WEEK: {
                long doy = days - daysFromCivil(y, 1, 1) + 1;
                long isodow = Math.floorMod(days + 4, 7);
                if (isodow == 0) isodow = 7;
                long w = (doy - isodow + 10) / 7;
                if (w < 1) return isoWeeksInYear(y - 1);
                if (w > isoWeeksInYear(y)) return 1;
                return w;
            }
            default:
                throw new IllegalStateException(f.name());
        }
    }

    // ------------------------------------------------------------------ intervals

    /** A parsed interval: calendar months plus a fixed number of microseconds. */
    record Interval(long months, long micros) {
        Interval negate() {
            return new Interval(-months, -micros);
        }
    }

    private static final Map<String, long[]> UNITS = new LinkedHashMap<>();

    static {
        // {months, micros}
        for (String s : new String[] {"year", "years", "yr", "yrs"}) UNITS.put(s, new long[] {12, 0});
        for (String s : new String[] {"month", "months", "mon", "mons"}) UNITS.put(s, new long[] {1, 0});
        for (String s : new String[] {"week", "weeks"}) UNITS.put(s, new long[] {0, 7 * DAY_MICROS});
        for (String s : new String[] {"day", "days"}) UNITS.put(s, new long[] {0, DAY_MICROS});
        for (String s : new String[] {"hour", "hours", "hr", "hrs"}) UNITS.put(s, new long[] {0, 3_600_000_000L});
        for (String s : new String[] {"minute", "minutes", "min", "mins"}) UNITS.put(s, new long[] {0, 60_000_000L});
        for (String s : new String[] {"second", "seconds", "sec", "secs"}) UNITS.put(s, new long[] {0, SEC_MICROS});
        for (String s : new String[] {"millisecond", "milliseconds", "ms"}) UNITS.put(s, new long[] {0, 1_000L});
        for (String s : new String[] {"microsecond", "microseconds", "us"}) UNITS.put(s, new long[] {0, 1L});
    }

    /** {@code INTERVAL '<spec>' [unit]}: {@code spec} is {@code "n unit [n unit ...]"}, or just {@code n} when {@code unit} is given. */
    static Interval interval(String spec, String unit) {
        long months = 0, micros = 0;
        String[] w = spec.trim().split("\\s+");
        if (unit != null) {
            if (w.length != 1) throw new ParquetSqlException("INTERVAL '" + spec + "' " + unit + ": expected one number before the unit");
            long[] u = unitOf(unit);
            return scaled(w[0], u, spec);
        }
        if (w.length == 0 || w.length % 2 != 0 || w[0].isEmpty()) {
            throw new ParquetSqlException("INTERVAL '" + spec + "': expected pairs like '3 days' or '1 year 2 months'");
        }
        for (int i = 0; i < w.length; i += 2) {
            Interval part = scaled(w[i], unitOf(w[i + 1]), spec);
            months = Math.addExact(months, part.months());
            micros = Math.addExact(micros, part.micros());
        }
        return new Interval(months, micros);
    }

    private static long[] unitOf(String name) {
        long[] u = UNITS.get(name.toLowerCase(Locale.ROOT));
        if (u == null) {
            throw new ParquetSqlException("INTERVAL unit '" + name + "' is not supported; supported: year, month, week, day, hour, minute, second, millisecond, microsecond");
        }
        return u;
    }

    private static Interval scaled(String number, long[] unit, String spec) {
        BigDecimal n;
        try {
            n = new BigDecimal(number);
        } catch (NumberFormatException e) {
            throw new ParquetSqlException("INTERVAL '" + spec + "': '" + number + "' is not a number");
        }
        try {
            long months = unit[0] == 0 ? 0 : n.multiply(BigDecimal.valueOf(unit[0])).longValueExact();
            long micros = unit[1] == 0 ? 0 : n.multiply(BigDecimal.valueOf(unit[1])).longValueExact();
            return new Interval(months, micros);
        } catch (ArithmeticException e) {
            throw new ParquetSqlException("INTERVAL '" + spec + "': months must be whole, and the fixed part must be a whole number of microseconds");
        }
    }

    /** {@code micros} plus the interval, with calendar months clamped to the end of the month. */
    static long addInterval(long micros, Interval iv) {
        long t = micros;
        if (iv.months() != 0) {
            long days = Math.floorDiv(t, DAY_MICROS), inDay = Math.floorMod(t, DAY_MICROS);
            long[] c = new long[3];
            civilFromDays(days, c);
            long total = c[0] * 12 + (c[1] - 1) + iv.months();
            long y = Math.floorDiv(total, 12);
            int m = (int) Math.floorMod(total, 12) + 1;
            int d = (int) Math.min(c[2], monthLength(y, m));
            t = Math.addExact(Math.multiplyExact(daysFromCivil(y, m, d), DAY_MICROS), inDay);
        }
        return Math.addExact(t, iv.micros());
    }


    /** {@code yyyy-MM-dd} (and, for a timestamp, {@code [ T]HH:mm[:ss[.f]]}) from ASCII bytes, to micros; throws on anything else. */
    static long parse(byte[] b, int len, boolean toDate, String text) {
        int a = 0, e = len;
        while (a < e && b[a] == ' ') a++;
        while (e > a && b[e - 1] == ' ') e--;
        try {
            int p = a;
            boolean neg = false;
            if (p < e && b[p] == '-') { neg = true; p++; }
            int ys = p;
            while (p < e && b[p] >= '0' && b[p] <= '9') p++;
            if (p - ys < 4 || p - ys > 6 || p >= e || b[p] != '-') throw new IllegalArgumentException();
            long y = 0;
            for (int i = ys; i < p; i++) y = y * 10 + (b[i] - '0');
            if (neg) y = -y;
            p++;
            int m = two(b, p, e); p += 2;
            if (p >= e || b[p] != '-') throw new IllegalArgumentException();
            p++;
            int d = two(b, p, e); p += 2;
            if (m < 1 || m > 12 || d < 1 || d > monthLength(y, m)) throw new IllegalArgumentException();
            long micros = daysFromCivil(y, m, d) * DAY_MICROS;
            if (p == e) return micros;
            if (toDate) throw new IllegalArgumentException();
            if (b[p] != ' ' && b[p] != 'T') throw new IllegalArgumentException();
            p++;
            int hh = two(b, p, e); p += 2;
            if (p >= e || b[p] != ':') throw new IllegalArgumentException();
            p++;
            int mi = two(b, p, e); p += 2;
            int ss = 0;
            long frac = 0;
            if (p < e && b[p] == ':') {
                p++;
                ss = two(b, p, e); p += 2;
                if (p < e && b[p] == '.') {
                    p++;
                    int digits = 0;
                    while (p < e && b[p] >= '0' && b[p] <= '9') {
                        if (digits < 6) frac = frac * 10 + (b[p] - '0');
                        else if (b[p] != '0') throw new IllegalArgumentException(); // would lose precision
                        digits++;
                        p++;
                    }
                    if (digits == 0) throw new IllegalArgumentException();
                    for (int k = digits; k < 6; k++) frac *= 10;
                }
            }
            if (p != e || hh > 23 || mi > 59 || ss > 59) throw new IllegalArgumentException();
            return micros + (hh * 3600L + mi * 60L + ss) * SEC_MICROS + frac;
        } catch (IllegalArgumentException | ArrayIndexOutOfBoundsException bad) {
            throw new ParquetSqlException(text + ": '" + new String(b, 0, len, StandardCharsets.UTF_8) + "' is not a valid "
                    + (toDate ? "DATE (expected yyyy-MM-dd)" : "TIMESTAMP (expected yyyy-MM-dd[ HH:mm[:ss[.ffffff]]], no time zone)"));
        }
    }

    private static int two(byte[] b, int p, int e) {
        if (p + 2 > e) throw new IllegalArgumentException();
        int x = b[p] - '0', y = b[p + 1] - '0';
        if (x < 0 || x > 9 || y < 0 || y > 9) throw new IllegalArgumentException();
        return x * 10 + y;
    }

}
