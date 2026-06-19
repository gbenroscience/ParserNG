package com.github.gbenroscience.math;

import com.github.gbenroscience.math.BigDecimalNthRootCalculation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

class BigDecimalNthRootCalculationTest {

    @Test
    void geomTest() {
        Assertions.assertEquals(0, new BigDecimal("2").compareTo(BigDecimalNthRootCalculation.nthRoot(2, new BigDecimal("4"))));
        Assertions.assertEquals(0, new BigDecimal("3").compareTo(BigDecimalNthRootCalculation.nthRoot(3, new BigDecimal("27"))));
        Assertions.assertEquals(0, new BigDecimal("2").compareTo(BigDecimalNthRootCalculation.nthRoot(8, new BigDecimal("256"))));
        Assertions.assertEquals(0, new BigDecimal("2").compareTo(BigDecimalNthRootCalculation.nthRoot(6, new BigDecimal("64"))));
        Assertions.assertEquals(0, new BigDecimal("3.36").compareTo(BigDecimalNthRootCalculation.nthRoot(3, new BigDecimal("38"), new MathContext(3, RoundingMode.HALF_DOWN))));
    }

}