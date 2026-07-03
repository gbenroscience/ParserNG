package com.github.gbenroscience.parser.ng.bench.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author GBEMIRO
 */
public class Stats {
    
    public final static int[] splitLongIntoDigits(long n) {
        if (n == 0) {
            return new int[]{0}; // Special case for zero
        }

        boolean isNegative = n < 0;
        long temp = Math.abs(n); // Work with the absolute value
        List<Integer> digitList = new ArrayList<>();

        while (temp > 0) {
            // Get the last digit using modulo 10
            digitList.add((int) (temp % 10));
            // Remove the last digit using integer division
            temp /= 10;
        }

        // The digits are in reverse order, so reverse the list
        Collections.reverse(digitList);

        // Convert the ArrayList<Integer> to a primitive int[] array
        int[] digits = new int[digitList.size()];
        for (int i = 0; i < digitList.size(); i++) {
            digits[i] = digitList.get(i);
        }

        // Handle the sign if necessary (e.g. if the original number was negative, you might need 
        // to handle the representation of the sign explicitly, depending on requirements)
        // For simply getting the sequence of digits, the absolute value is sufficient.
        return digits;
    }

    public static double findAverage(double[] data) {
        if (data == null || data.length == 0) {
            return Double.NaN;
        }
        if (data.length <= 3) {
            // Too few points for meaningful outlier removal – return simple average
            double sum = 0.0;
            for (double v : data) {
                sum += v;
            }
            return sum / data.length;
        }

        // Work on a copy to avoid modifying the original array
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        int n = sorted.length;

        // More accurate quartile calculation with linear interpolation (Type 7 / "default" method)
        double q1 = percentile(sorted, 25.0);
        double q3 = percentile(sorted, 75.0);
        double iqr = q3 - q1;

        // Use standard 1.5 * IQR fences (Tukey method)
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;

        // Count inliers and compute sum for average
        double sum = 0.0;
        int count = 0;
        for (double v : sorted) {
            if (v >= lowerBound && v <= upperBound) {
                sum += v;
                count++;
            }
        }

        // Fallback if all values were outliers (rare but possible with extreme noise)
        if (count == 0) {
            // Return overall average as safe fallback
            sum = 0.0;
            for (double v : data) {
                sum += v;
            }
            return sum / data.length;
        }

        return sum / count;
    }

// Helper: percentile with linear interpolation (matches R's default, Python's numpy.percentile, etc.)
    private static double percentile(double[] sorted, double p) {
        if (sorted.length == 0) {
            return Double.NaN;
        }
        if (p <= 0) {
            return sorted[0];
        }
        if (p >= 100) {
            return sorted[sorted.length - 1];
        }

        double index = (p / 100.0) * (sorted.length - 1);
        int lower = (int) Math.floor(index);
        double fraction = index - lower;

        if (lower + 1 < sorted.length) {
            return sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]);
        } else {
            return sorted[lower];
        }
    }

    public static double round(double val, int n) {
        double scale = Math.pow(10, n);
        return Math.round(val * scale) / scale;
    }
}
