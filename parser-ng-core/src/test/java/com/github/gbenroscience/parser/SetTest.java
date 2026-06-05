package com.github.gbenroscience.math.parser;

import com.github.gbenroscience.parser.Set;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Arrays;

class SetTest {

    @Test
    void gsumTest() {
        Set s = new Set(Arrays.asList("2","3","4"));
        Assertions.assertEquals(0, new BigDecimal("24").compareTo(s.gsum()));
    }

    @Test
    void sizeTest() {
        Set s = new Set(Arrays.asList("2","3","4"));
        Assertions.assertEquals(3, s.size());
    }

    @Test
    void geomTest() {
        Set s = new Set(Arrays.asList("2","8", "4"));
        Assertions.assertEquals(0, new BigDecimal("4").compareTo(s.geom()));
    }
}