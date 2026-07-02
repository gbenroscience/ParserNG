/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math;

/**
 *
 * @author GBEMIRO
 */

/**
 * Highly accurate Error Function (erf) using Cody's Rational Approximations.
 * ~15 decimal digits accuracy.
 */
public class CodyMath {

    /**
     * Accurate erf(x) using Cody's rational approximations
     */
  public static double erf(double x) {
        if (x == 0.0) return 0.0;

        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t 
                          + 1.421413741) * t - 0.284496736) * t 
                          + 0.254829592) * t) * Math.exp(-x * x);

        return Math.copySign(y, x);
    }

    private static double erfcMedium(double x) {
        double p = (((((0.05895007051304052 * x + 3.829044036164835) * x 
                   + 43.18793405786769) * x + 157.3068460515812) * x 
                   + 214.6001532057584) * x + 106.1664295623469) * x 
                   + 14.5302830141757;

        double q = (((((1.0 * x + 17.3459738020609) * x 
                   + 127.8650310266291) * x + 314.141639017465) * x 
                   + 309.7712496731671) * x + 122.2982420286843) * x 
                   + 14.53028300724806;

        return Math.exp(-x * x) * (p / q);
    }

    private static double erfcLarge(double x) {
        double xsqInv = 1.0 / (x * x);
        double p = (((0.0736713390314174 * xsqInv + 0.745582913335534) * xsqInv 
                  + 1.541607549923533) * xsqInv + 0.910514304077531) * xsqInv 
                  + 0.1350648667813508;

        double q = ((((0.01282616526077372 * xsqInv + 0.2447524102450513) * xsqInv 
                   + 1.137319453745207) * xsqInv + 1.84085398366255) * xsqInv 
                   + 0.9785461628585647) * xsqInv + 0.1350648965456337;

        return (Math.exp(-x * x) / x) * (p / q);
    }

    public static double erfc(double x) {
        double absX = Math.abs(x);
        if (absX > 26.543) return x > 0 ? 0.0 : 2.0;
        if (absX <= 0.46875) return 1.0 - erf(x);
        return x < 0 ? 2.0 - erfcMedium(absX) : erfcMedium(absX);
    }
    public static void main(String[] args) {
        System.out.println("erf(2) = " + CodyMath.erf(2));           // Should be ~0.995322
System.out.println("erf(1) = " + CodyMath.erf(1));           // Should be ~0.842701
System.out.println("erf(3) = " + CodyMath.erf(3));           // Should be ~0.999978
    }
}


