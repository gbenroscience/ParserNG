package com.github.gbenroscience.math.parser.methods.ext;

import com.github.gbenroscience.parser.methods.ext.Utils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class UtilsTest {

    @Test
    void stringsToBigDecimalsTest() {
        List<BigDecimal> bl = Utils.stringsToBigDecimals(Arrays.asList("1", "2"));
        Assertions.assertEquals(Arrays.asList(new BigDecimal("1"), new BigDecimal("2")), bl);
    }

    @Test
    void stringsToBigDecimalsTest2() {
        List<BigDecimal> bl = Utils.stringsToBigDecimals(Arrays.asList("1", "2"), 0);
        Assertions.assertEquals(Arrays.asList(new BigDecimal("1"), new BigDecimal("2")), bl);
        bl = Utils.stringsToBigDecimals(Arrays.asList("1", "2"), 1);
        Assertions.assertEquals(Arrays.asList(new BigDecimal("2")), bl);
        bl = Utils.stringsToBigDecimals(Arrays.asList("1", "2"), 2);
        Assertions.assertEquals(new ArrayList<>(), bl);
        bl = Utils.stringsToBigDecimals(Arrays.asList("1", "2"), 3);
        Assertions.assertEquals(new ArrayList<>(), bl);
        bl = Utils.stringsToBigDecimals(Arrays.asList("1", "2"), 4);
        Assertions.assertEquals(new ArrayList<>(), bl);
    }


    @Test
    void trimListTestOddShort() {
        List l = Utils.trimList(Arrays.asList(1), 0, 0);
        Assertions.assertEquals(Arrays.asList(1), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1)), 1, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1)), 2, 0);
        Assertions.assertEquals(Arrays.asList(), l);

        l = Utils.trimList(Arrays.asList(1), 0, 1);
        Assertions.assertEquals(Arrays.asList(1), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1)), 1, 1);
        Assertions.assertEquals(Arrays.asList(1), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1)), 2, 1);
        Assertions.assertEquals(Arrays.asList(1), l);

        l = Utils.trimList(Arrays.asList(1), 0, 2);
        Assertions.assertEquals(Arrays.asList(1), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1)), 1, 2);
        Assertions.assertEquals(Arrays.asList(1), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1)), 2, 2);
        Assertions.assertEquals(Arrays.asList(1), l);
    }

    @Test
    void trimListTestEvenShort() {
        List l = Utils.trimList(Arrays.asList(), 0, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList()), 1, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList()), 2, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(Arrays.asList(), 0, 1);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList()), 1, 1);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList()), 2, 1);
        Assertions.assertEquals(Arrays.asList(), l);

        l = Utils.trimList(Arrays.asList(1, 2), 0, 0);
        Assertions.assertEquals(Arrays.asList(1, 2), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2)), 1, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2)), 2, 0);
        Assertions.assertEquals(Arrays.asList(), l);

        l = Utils.trimList(Arrays.asList(1, 2), 0, 1);
        Assertions.assertEquals(Arrays.asList(1, 2), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2)), 1, 1);
        Assertions.assertEquals(Arrays.asList(1, 2), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2)), 2, 1);
        Assertions.assertEquals(Arrays.asList(1, 2), l);
    }

    @Test
    void trimListTestOdd() {
        List l = Utils.trimList(Arrays.asList(1, 2, 3), 0, 0);
        Assertions.assertEquals(Arrays.asList(1, 2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 1, 0);
        Assertions.assertEquals(Arrays.asList(2), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 2, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 3, 0);
        Assertions.assertEquals(Arrays.asList(), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3), 0, 1);
        Assertions.assertEquals(Arrays.asList(1, 2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 1, 1);
        Assertions.assertEquals(Arrays.asList(2), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(2)), 2, 1);
        Assertions.assertEquals(Arrays.asList(2), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(2)), 3, 1);
        Assertions.assertEquals(Arrays.asList(2), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3), 0, 2);
        Assertions.assertEquals(Arrays.asList(1, 2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 1, 2);
        Assertions.assertEquals(Arrays.asList(1, 2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 2, 2);
        Assertions.assertEquals(Arrays.asList(1, 2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3)), 3, 2);
        Assertions.assertEquals(Arrays.asList(1, 2, 3), l);
    }


    @Test
    void trimListTestEven() {
        List l = Utils.trimList(Arrays.asList(1, 2, 3, 4), 0, 0);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 1, 0);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 2, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 3, 0);
        Assertions.assertEquals(Arrays.asList(), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 4, 0);
        Assertions.assertEquals(Arrays.asList(), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3, 4), 0, 1);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 1, 1);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 2, 1);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 3, 1);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 4, 1);
        Assertions.assertEquals(Arrays.asList(2, 3), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3, 4), 0, 2);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 1, 2);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 2, 2);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 3, 2);
        Assertions.assertEquals(Arrays.asList(2, 3), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 4, 2);
        Assertions.assertEquals(Arrays.asList(2, 3), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3, 4), 0, 3);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 1, 3);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 2, 3);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 3, 3);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 4, 3);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3, 4), 0, 4);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 1, 4);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 2, 4);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 3, 4);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 4, 4);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);

        l = Utils.trimList(Arrays.asList(1, 2, 3, 4), 0, 5);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 1, 5);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 2, 5);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 3, 5);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);
        l = Utils.trimList(new ArrayList<>(Arrays.asList(1, 2, 3, 4)), 4, 5);
        Assertions.assertEquals(Arrays.asList(1, 2, 3, 4), l);

    }

    @Test
    void evaluateSingleTokenTest() {
        List<BigDecimal> l;
        l = Utils.evaluateSingleToken(new ArrayList<String>(0));
        Assertions.assertEquals(new ArrayList<BigDecimal>(0), l);
        l = Utils.evaluateSingleToken(Arrays.asList("1"));
        Assertions.assertEquals(Arrays.asList(new BigDecimal("1")), l);
        l = Utils.evaluateSingleToken(Arrays.asList("1+1"));
        Assertions.assertEquals(Arrays.asList(new BigDecimal("2.0")), l);
        l = Utils.evaluateSingleToken(Arrays.asList("1", "2"));
        Assertions.assertEquals(Arrays.asList(new BigDecimal("1"), new BigDecimal("2")), l);
        l = Utils.evaluateSingleToken(Arrays.asList("1+1", "2"));
        Assertions.assertEquals(Arrays.asList(new BigDecimal("13.0")), l); //this is artificial value which should never happen, but demonstrate possible issue
        l = Utils.evaluateSingleToken(Arrays.asList("1", "+", "2"));
        Assertions.assertEquals(Arrays.asList(new BigDecimal("3.0")), l);
        l = Utils.evaluateSingleToken(null);
        Assertions.assertEquals(null, l);
    }

    @Test
    void getFirstTokenAsInt() {
        int i = Utils.getFirstStringTokenAsInt(Arrays.asList("11.1"));
        Assertions.assertEquals(11, i);
    }

    @Test
    void decimalAndFractionalParts() {
        int[] r = Utils.decimalAndFractionalParts("1.1");
        Assertions.assertArrayEquals(new int[]{1,1}, r);
        r = Utils.decimalAndFractionalParts("1.12");
        Assertions.assertArrayEquals(new int[]{1,2}, r);
        r = Utils.decimalAndFractionalParts("12.1");
        Assertions.assertArrayEquals(new int[]{2,1}, r);
        r = Utils.decimalAndFractionalParts("12.12");
        Assertions.assertArrayEquals(new int[]{2,2}, r);

        r = Utils.decimalAndFractionalParts("12");
        Assertions.assertArrayEquals(new int[]{2,0}, r);
        r = Utils.decimalAndFractionalParts("12.");
        Assertions.assertArrayEquals(new int[]{2,0}, r);

        r = Utils.decimalAndFractionalParts("1234567890.1234567890000");
        Assertions.assertArrayEquals(new int[]{10,9}, r);

        r = Utils.decimalAndFractionalParts("-1.1");
        Assertions.assertArrayEquals(new int[]{1,1}, r);
        r = Utils.decimalAndFractionalParts("-1.12");
        Assertions.assertArrayEquals(new int[]{1,2}, r);
        r = Utils.decimalAndFractionalParts("-12.1");
        Assertions.assertArrayEquals(new int[]{2,1}, r);
        r = Utils.decimalAndFractionalParts("-12.12");
        Assertions.assertArrayEquals(new int[]{2,2}, r);
    }
}
