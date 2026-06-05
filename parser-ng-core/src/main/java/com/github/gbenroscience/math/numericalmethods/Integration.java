/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.numericalmethods;

import com.github.gbenroscience.parser.Function;
import java.lang.invoke.MethodHandle;
import java.util.*;

public class Integration {

// Create the full 8-point arrays (mirrored)
    private static final double[] fullNodes8 = new double[8];
    private static final double[] fullWeights8 = new double[8];

// Create the full 64-point arrays (mirrored)
    private static final double[] fullNodes = new double[64];
    private static final double[] fullWeights = new double[64];

    private static final double[] fullNodes16 = new double[16];
    private static final double[] fullWeights16 = new double[16];

    private static final double[] fullNodes32 = new double[32];
    private static final double[] fullWeights32 = new double[32];

    //////n=8
    // Define the 4 positive nodes and weights for n=8
private static final double[] nodes4 = {
        0.1834346424956498, 0.5255324099163290,
        0.7966664774136267, 0.9602898564975362
    };

    private static final double[] weights4 = {
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533445, 0.1012285362903763
    };

    ////16 point
private static final double[] nodes8 = {
        0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
        0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499
    };
    private static final double[] weights8 = {
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
        0.1246289712569339, 0.0951585116824475, 0.0622535239386479, 0.0271524594117541
    };

    /////32 point
private static final double[] nodes16 = {
        0.0483076656877383, 0.1444719615827965, 0.2392873622521371, 0.3318686022953256,
        0.4213512761306353, 0.5068999089322294, 0.5877157572407623, 0.6630442669302152,
        0.7321821187402897, 0.7944837959679424, 0.8493676137325700, 0.8963211557602521,
        0.9349060759377397, 0.9647622555875064, 0.9856115115452683, 0.9972638618494816
    };
    private static final double[] weights16 = {
        0.0965445133110442, 0.0956387200792749, 0.0938443990808046, 0.0911738786957639,
        0.0876520930044038, 0.0833119242269468, 0.0781938957870703, 0.0723457941088485,
        0.0658222227763618, 0.0586840934785355, 0.0510008305922203, 0.0428483214667525,
        0.0343103221349885, 0.0254746746194153, 0.0164416613033565, 0.0073015540131196
    };

    ///64 point

// Define the 32 positive nodes and weights
private static final double[] nodes32 = {
        0.0243502926634244, 0.0729931217877990, 0.1214628192961206, 0.1696444204239928,
        0.2174236437400071, 0.2646871622087674, 0.3113228719902110, 0.3572201583376681,
        0.4022701579639916, 0.4463660172534641, 0.4894031457070530, 0.5312794640198945,
        0.5718956462026340, 0.6111553551723933, 0.6489654712546573, 0.6852363130542332,
        0.7198818501716108, 0.7528199072605319, 0.7839723589433414, 0.8132653151227976,
        0.8406292962525804, 0.8659993981540928, 0.8893154459951141, 0.9105221370785028,
        0.9295691721319396, 0.9464113748584028, 0.9610087996520537, 0.9733268277899110,
        0.9833362538846260, 0.9910133714767443, 0.9963401167719553, 0.9993050417357721
    };

    private static final double[] weights32 = {
        0.0486909570091397, 0.0485754674415034, 0.0483447622348030, 0.0479993885964583,
        0.0475401657148303, 0.0469681828162100, 0.0462847965813144, 0.0454916279274181,
        0.0445905581637566, 0.0435837245293235, 0.0424735151236536, 0.0412625632426235,
        0.0399537411327203, 0.0385501531786156, 0.0370551285402400, 0.0354722132568824,
        0.0338051618371416, 0.0320579283548516, 0.0302346570724025, 0.0283396726142595,
        0.0263774697150547, 0.0243527025687109, 0.0222701738083833, 0.0201348231535302,
        0.0179517157756973, 0.0157260304760247, 0.0134630478967186, 0.0111681394601311,
        0.0088467598263639, 0.0065044579689784, 0.0041470332605625, 0.0017832807216964
    };

    private final double[] preAllocatedVars = new double[256];

    static {

        for (int i = 0; i < 4; i++) {
            // Fill negative side (indices 0 to 3)
            fullNodes8[3 - i] = -nodes4[i];
            fullWeights8[3 - i] = weights4[i];

            // Fill positive side (indices 4 to 7)
            fullNodes8[4 + i] = nodes4[i];
            fullWeights8[4 + i] = weights4[i];
        }

        for (int i = 0; i < 8; i++) {
            fullNodes16[7 - i] = -nodes8[i];
            fullWeights16[7 - i] = weights8[i];
            fullNodes16[8 + i] = nodes8[i];
            fullWeights16[8 + i] = weights8[i];
        }

        for (int i = 0; i < 16; i++) {
            fullNodes32[15 - i] = -nodes16[i];
            fullWeights32[15 - i] = weights16[i];
            fullNodes32[16 + i] = nodes16[i];
            fullWeights32[16 + i] = weights16[i];
        }

        for (int i = 0; i < 32; i++) {
            // Fill negative side (0 to 31)
            fullNodes[31 - i] = -nodes32[i];
            fullWeights[31 - i] = weights32[i];

            // Fill positive side (32 to 63)
            fullNodes[32 + i] = nodes32[i];
            fullWeights[32 + i] = weights32[i];
        }

//end 64 point
    }

    private Function function = null;   // Function to be integrated
    private boolean setFunction = false;        // = true when Function set
    private double lowerLimit = Double.NaN;     // Lower integration limit
    private double upperLimit = Double.NaN;     // Upper integration limit
    private boolean setLimits = false;          // = true when limits set

    private int glPoints = 0;                   // Number of points in the Gauss-Legendre integration
    private boolean setGLpoints = false;        // = true when glPoints set
    private int nIntervals = 0;                 // Number of intervals in the rectangular rule integrations
    private boolean setIntervals = false;       // = true when nIntervals set

    private double integralSum = 0.0D;          // Sum returned by the numerical integration method
    private boolean setIntegration = false;     // = true when integration performed

    // ArrayLists to hold Gauss-Legendre Coefficients saving repeated calculation
    private static ArrayList<Integer> gaussQuadIndex = new ArrayList<>();         // Gauss-Legendre indices
    private static ArrayList<double[]> gaussQuadDistArrayList = new ArrayList<>();  // Gauss-Legendre distances
    private static ArrayList<double[]> gaussQuadWeightArrayList = new ArrayList<>();// Gauss-Legendre weights

    // Iterative trapezium rule
    private double requiredAccuracy = 0.0D;     // required accuracy at which iterative trapezium is terminated
    private double trapeziumAccuracy = 0.0D;    // actual accuracy at which iterative trapezium is terminated as instance variable
    private static double trapAccuracy = 0.0D;  // actual accuracy at which iterative trapezium is terminated as class variable
    private int maxIntervals = 0;               // maximum number of intervals allowed in iterative trapezium
    private int trapeziumIntervals = 1;         // number of intervals in trapezium at which accuracy was satisfied as instance variable
    private static int trapIntervals = 1;       // number of intervals in trapezium at which accuracy was satisfied as class variable
    MethodHandle targetHandle;
    private int varIndex = 0; // Default slot

    // CONSTRUCTORS
    // Default constructor
    public Integration() {
    }

    // Constructor taking function to be integrated
    public Integration(Function intFunc) {
        this.function = intFunc;
        this.setFunction = true;
    }

    // Constructor taking function to be integrated and the limits
    public Integration(Function intFunc, double lowerLimit, double upperLimit) {
        this.function = intFunc;
        this.setFunction = true;
        this.lowerLimit = lowerLimit;
        this.upperLimit = upperLimit;
        this.setLimits = true;
    }

    // SET METHODS
    // Set function to be integrated
    public void setFunction(Function intFunc) {
        this.function = intFunc;
        this.setFunction = true;
    }

    // Set limits
    public void setLimits(double lowerLimit, double upperLimit) {
        this.lowerLimit = lowerLimit;
        this.upperLimit = upperLimit;
        this.setLimits = true;
    }

    // Set lower limit
    public void setLowerLimit(double lowerLimit) {
        this.lowerLimit = lowerLimit;
        if (!Fmath.isNaN(this.upperLimit)) {
            this.setLimits = true;
        }
    }

    // Set lower limit
    public void setlowerLimit(double lowerLimit) {
        this.lowerLimit = lowerLimit;
        if (!Fmath.isNaN(this.upperLimit)) {
            this.setLimits = true;
        }
    }

    // Set upper limit
    public void setUpperLimit(double upperLimit) {
        this.upperLimit = upperLimit;
        if (!Fmath.isNaN(this.lowerLimit)) {
            this.setLimits = true;
        }
    }

    // Set upper limit
    public void setupperLimit(double upperLimit) {
        this.upperLimit = upperLimit;
        if (!Fmath.isNaN(this.lowerLimit)) {
            this.setLimits = true;
        }
    }

    // Set number of points in the Gaussian Legendre integration
    public void setGLpoints(int nPoints) {
        this.glPoints = nPoints;
        this.setGLpoints = true;
    }

    // Set number of intervals in trapezoidal, forward or backward rectangular integration
    public void setNintervals(int nIntervals) {
        this.nIntervals = nIntervals;
        this.setIntervals = true;
    }

    // GET METHODS
    // Get the sum returned by the numerical integration
    public double getIntegralSum() {
        if (!this.setIntegration) {
            throw new IllegalArgumentException("No integration has been performed");
        }
        return this.integralSum;
    }

    private double gaussQuadWithoutMethodHandle() {
        // 1. Validations
        if (!this.setGLpoints) {
            throw new IllegalArgumentException("Number of points not set");
        }
        if (!this.setLimits) {
            throw new IllegalArgumentException("One limit or both limits not set");
        }
        if (!this.setFunction) {
            throw new IllegalArgumentException("No integral function has been set");
        }

        double[] gaussQuadDist;
        double[] gaussQuadWeight;

        // 2. Direct switch for O(1) table access (The Bypass)
        switch (this.glPoints) {
            case 8:
                gaussQuadDist = fullNodes8;
                gaussQuadWeight = fullWeights8;
                break;
            case 16:
                gaussQuadDist = fullNodes16;
                gaussQuadWeight = fullWeights16;
                break;
            case 32:
                gaussQuadDist = fullNodes32;
                gaussQuadWeight = fullWeights32;
                break;
            case 64:
                gaussQuadDist = fullNodes;
                gaussQuadWeight = fullWeights;
                break;
            default:
                // Dynamic fallback for non-standard point counts
                int kn = -1;
                for (int k = 0; k < this.gaussQuadIndex.size(); k++) {
                    if (this.gaussQuadIndex.get(k) == this.glPoints) {
                        kn = k;
                        break;
                    }
                }

                if (kn == -1) {
                    gaussQuadDist = new double[glPoints];
                    gaussQuadWeight = new double[glPoints];
                    Integration.gaussQuadCoeff(gaussQuadDist, gaussQuadWeight, glPoints);
                    Integration.gaussQuadIndex.add(glPoints);
                    Integration.gaussQuadDistArrayList.add(gaussQuadDist);
                    Integration.gaussQuadWeightArrayList.add(gaussQuadWeight);
                } else {
                    gaussQuadDist = gaussQuadDistArrayList.get(kn);
                    gaussQuadWeight = gaussQuadWeightArrayList.get(kn);
                }
                break;
        }

        // 3. High-Precision Setup
        double sum = 0.0D;
        double c = 0.0D; // Kahan compensation
        double xplus = 0.5D * (upperLimit + lowerLimit);
        double xminus = 0.5D * (upperLimit - lowerLimit);

        // LOCAL REFERENCE trick (Prevents repeated heap access via 'this')
        double[] dist = gaussQuadDist;
        double[] weight = gaussQuadWeight;

        // Cache functional reference locally
        com.github.gbenroscience.parser.Function func = this.function;

        // 4. Perform summation with Kahan algorithm
        for (int i = 0; i < glPoints; i++) {
            // Calculate point and update function arguments
            func.updateArgs(xplus + (xminus * dist[i]));

            // Precise addition
            double val = weight[i] * func.calc();
            double y = val - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }

        this.integralSum = sum * xminus;
        this.setIntegration = true;
        return this.integralSum;
    }

    private double gaussQuadWithMethodHandle() {
        if (!this.setGLpoints) {
            throw new IllegalArgumentException("Number of points not set");
        }
        if (!this.setLimits) {
            throw new IllegalArgumentException("Limits not set");
        }
        if (this.targetHandle == null && !this.setFunction) {
            throw new IllegalArgumentException("No integral function or MethodHandle set");
        }

        double[] gaussQuadDist;
        double[] gaussQuadWeight;

        // Direct switch for O(1) table access
        switch (this.glPoints) {
            case 8:
                gaussQuadDist = fullNodes8;
                gaussQuadWeight = fullWeights8;
                break;
            case 16:
                gaussQuadDist = fullNodes16;
                gaussQuadWeight = fullWeights16;
                break;
            case 32:
                gaussQuadDist = fullNodes32;
                gaussQuadWeight = fullWeights32;
                break;
            case 64:
                gaussQuadDist = fullNodes;
                gaussQuadWeight = fullWeights;
                break;
            default:
                // Dynamic fallback: Search existing cache or compute via Newton-Raphson
                int kn = -1;
                for (int k = 0; k < gaussQuadIndex.size(); k++) {
                    if (gaussQuadIndex.get(k) == this.glPoints) {
                        kn = k;
                        break;
                    }
                }
                if (kn != -1) {
                    gaussQuadDist = gaussQuadDistArrayList.get(kn);
                    gaussQuadWeight = gaussQuadWeightArrayList.get(kn);
                } else {
                    gaussQuadDist = new double[glPoints];
                    gaussQuadWeight = new double[glPoints];
                    Integration.gaussQuadCoeff(gaussQuadDist, gaussQuadWeight, glPoints);
                    Integration.gaussQuadIndex.add(glPoints);
                    Integration.gaussQuadDistArrayList.add(gaussQuadDist);
                    Integration.gaussQuadWeightArrayList.add(gaussQuadWeight);
                }
                break;
        }

        double sum = 0.0D;
        double c = 0.0D; // Kahan compensation variable
        double xplus = 0.5D * (upperLimit + lowerLimit);
        double xminus = 0.5D * (upperLimit - lowerLimit);

        // LOCAL REFERENCE trick (Drains the CPU)
        double[] dist = gaussQuadDist;
        double[] weight = gaussQuadWeight;

        if (this.targetHandle != null) {
            // REUSE an existing array if possible to avoid allocation
            double[] vars = (this.preAllocatedVars != null) ? this.preAllocatedVars : new double[256];
            int vIdx = this.varIndex;

            try {
                for (int i = 0; i < glPoints; i++) {
                    vars[vIdx] = xplus + (xminus * dist[i]);

                    // 1. Get value to add
                    double val = weight[i] * (double) this.targetHandle.invokeExact(vars);

                    // 2. Kahan Summation (Extreme Precision)
                    double y = val - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
            } catch (Throwable t) {
                throw new RuntimeException("Turbo Integration failed at slot " + vIdx, t);
            }
        } else {
            // Legacy path with Kahan
            for (int i = 0; i < glPoints; i++) {
                this.function.updateArgs(xplus + (xminus * dist[i]));
                double val = weight[i] * this.function.calc();
                double y = val - c;
                double t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }

        this.integralSum = sum * xminus;
        this.setIntegration = true;
        return this.integralSum;
    }

    // Numerical integration using n point Gaussian-Legendre quadrature (instance method)
    // All parameters except the number of points in the Gauss-Legendre integration preset
    public double gaussQuad(int glPoints) {
        this.glPoints = glPoints;
        this.setGLpoints = true;
        return this.targetHandle == null ? this.gaussQuadWithoutMethodHandle() : this.gaussQuadWithMethodHandle();
    }

    // Numerical integration using n point Gaussian-Legendre quadrature (static method)
    // All parametes provided
    public static double gaussQuad(Function intFunc, double lowerLimit, double upperLimit, int glPoints) {
        Integration intgrtn = new Integration(intFunc, lowerLimit, upperLimit);
        return intgrtn.gaussQuad(glPoints);
    }

    private static void clearCache() {
        Integration.gaussQuadIndex.clear();
        Integration.gaussQuadDistArrayList.clear();
        Integration.gaussQuadWeightArrayList.clear();
    }

    // Numerical integration using n point Gaussian-Legendre quadrature (static method)
    // All parameters provided
    // Updated static method to include the variable slot index
    public static double gaussQuad(MethodHandle targetHandle, int varIndex, Function intFunc, double lowerLimit, double upperLimit, int glPoints) {
        Integration intgrtn = new Integration(intFunc, lowerLimit, upperLimit);
        intgrtn.targetHandle = targetHandle;
        intgrtn.varIndex = varIndex; // Pass the slot here
        return intgrtn.gaussQuad(glPoints);
    }

    // Returns the distance (gaussQuadDist) and weight coefficients (gaussQuadCoeff)
    // for an n point Gauss-Legendre Quadrature.
    // The Gauss-Legendre distances, gaussQuadDist, are scaled to -1 to 1
    // See Numerical Recipes for details
    public static void gaussQuadCoeff(double[] gaussQuadDist, double[] gaussQuadWeight, int n) {

        double z = 0.0D, z1 = 0.0D;
        double pp = 0.0D, p1 = 0.0D, p2 = 0.0D, p3 = 0.0D;

        double eps = 1e-15;	// set required precision
        double x1 = -1.0D;		// lower limit
        double x2 = 1.0D;		// upper limit

        //  Calculate roots
        // Roots are symmetrical - only half calculated
        int m = (n + 1) / 2;
        double xm = 0.5D * (x2 + x1);
        double xl = 0.5D * (x2 - x1);

        // Loop for  each root
        for (int i = 1; i <= m; i++) {
            // Approximation of ith root
            z = Math.cos(Math.PI * (i - 0.25D) / (n + 0.5D));

            // Refinement on above using Newton's method
            do {
                p1 = 1.0D;
                p2 = 0.0D;

                // Legendre polynomial (p1, evaluated at z, p2 is polynomial of
                //  one order lower) recurrence relationsip
                for (int j = 1; j <= n; j++) {
                    p3 = p2;
                    p2 = p1;
                    p1 = ((2.0D * j - 1.0D) * z * p2 - (j - 1.0D) * p3) / j;
                }
                pp = n * (z * p1 - p2) / (z * z - 1.0D);    // Derivative of p1
                z1 = z;
                z = z1 - p1 / pp;			            // Newton's method
            } while (Math.abs(z - z1) > eps);

            gaussQuadDist[i - 1] = xm - xl * z;		    // Scale root to desired interval
            gaussQuadDist[n - i] = xm + xl * z;		    // Symmetric counterpart
            gaussQuadWeight[i - 1] = 2.0 * xl / ((1.0 - z * z) * pp * pp);	// Compute weight
            gaussQuadWeight[n - i] = gaussQuadWeight[i - 1];		// Symmetric counterpart
        }
    }

    // TRAPEZIUM METHODS
    // Numerical integration using the trapeziodal rule (instance method)
    // all parameters preset
    public double trapezium() {
        if (!this.setIntervals) {
            throw new IllegalArgumentException("Number of intervals not set");
        }
        if (!this.setLimits) {
            throw new IllegalArgumentException("One limit or both limits not set");
        }
        if (!this.setFunction) {
            throw new IllegalArgumentException("No integral function has been set");
        }

        double y1 = 0.0D;
        double interval = (this.upperLimit - this.lowerLimit) / this.nIntervals;
        double x0 = this.lowerLimit;
        double x1 = this.lowerLimit + interval;
        this.function.updateArgs(x0);
        double y0 = this.function.calc();
        this.integralSum = 0.0D;

        for (int i = 0; i < nIntervals; i++) {
            // adjust last interval for rounding errors
            if (x1 > this.upperLimit) {
                x1 = this.upperLimit;
                interval -= (x1 - this.upperLimit);
            }
            this.function.updateArgs(x1);
            // perform summation
            y1 = this.function.calc();
            this.integralSum += 0.5D * (y0 + y1) * interval;
            x0 = x1;
            y0 = y1;
            x1 += interval;
        }
        this.setIntegration = true;
        return this.integralSum;
    }

    // Numerical integration using the trapeziodal rule (instance method)
    // all parameters except the number of intervals preset
    public double trapezium(int nIntervals) {
        this.nIntervals = nIntervals;
        this.setIntervals = true;
        return this.trapezium();
    }

    // Numerical integration using the trapeziodal rule (static method)
    // all parameters to be provided
    public static double trapezium(Function intFunc, double lowerLimit, double upperLimit, int nIntervals) {
        Integration intgrtn = new Integration(intFunc, lowerLimit, upperLimit);
        return intgrtn.trapezium(nIntervals);
    }

    // Numerical integration using an iteration on the number of intervals in the trapeziodal rule
    // until two successive results differ by less than a predetermined accuracy times the penultimate result
    public double trapezium(double accuracy, int maxIntervals) {
        this.requiredAccuracy = accuracy;
        this.maxIntervals = maxIntervals;
        this.trapeziumIntervals = 1;

        double summ = this.trapezium(this.function, this.lowerLimit, this.upperLimit, 1);
        double oldSumm = summ;
        int i = 2;
        for (i = 2; i <= this.maxIntervals; i++) {
            summ = this.trapezium(this.function, this.lowerLimit, this.upperLimit, i);
            this.trapeziumAccuracy = Math.abs((summ - oldSumm) / oldSumm);
            if (this.trapeziumAccuracy <= this.requiredAccuracy) {
                break;
            }
            oldSumm = summ;
        }

        if (i > this.maxIntervals) {
            System.out.println("accuracy criterion was not met in Integration.trapezium - current sum was returned as result.");
            this.trapeziumIntervals = this.maxIntervals;
        } else {
            this.trapeziumIntervals = i;
        }
        Integration.trapIntervals = this.trapeziumIntervals;
        Integration.trapAccuracy = this.trapeziumAccuracy;
        return summ;
    }

    // Numerical integration using an iteration on the number of intervals in the trapeziodal rule (static method)
    // until two successive results differ by less than a predtermined accuracy times the penultimate result
    // All parameters to be provided
    public static double trapezium(Function intFunc, double lowerLimit, double upperLimit, double accuracy, int maxIntervals) {
        Integration intgrtn = new Integration(intFunc, lowerLimit, upperLimit);
        return intgrtn.trapezium(accuracy, maxIntervals);
    }

    // Get the number of intervals at which accuracy was last met in trapezium if using the instance trapezium call
    public int getTrapeziumIntervals() {
        return this.trapeziumIntervals;
    }

    // Get the number of intervals at which accuracy was last met in trapezium if using static trapezium call
    public static int getTrapIntervals() {
        return Integration.trapIntervals;
    }

    // Get the actual accuracy acheived when the iterative trapezium calls were terminated, using the instance method
    public double getTrapeziumAccuracy() {
        return this.trapeziumAccuracy;
    }

    // Get the actual accuracy acheived when the iterative trapezium calls were terminated, using the static method
    public static double getTrapAccuracy() {
        return Integration.trapAccuracy;
    }

    // BACKWARD RECTANGULAR METHODS
    // Numerical integration using the backward rectangular rule (instance method)
    // All parameters preset
    public double backward() {
        if (!this.setIntervals) {
            throw new IllegalArgumentException("Number of intervals not set");
        }
        if (!this.setLimits) {
            throw new IllegalArgumentException("One limit or both limits not set");
        }
        if (!this.setFunction) {
            throw new IllegalArgumentException("No integral function has been set");
        }

        double interval = (this.upperLimit - this.lowerLimit) / this.nIntervals;
        double x = this.lowerLimit + interval;
        this.function.updateArgs(x);
        double y = this.function.calc();
        this.integralSum = 0.0D;

        for (int i = 0; i < this.nIntervals; i++) {
            // adjust last interval for rounding errors
            if (x > this.upperLimit) {
                x = this.upperLimit;
                interval -= (x - this.upperLimit);
            }
            this.function.updateArgs(x);
            // perform summation
            y = this.function.calc();
            this.integralSum += y * interval;
            x += interval;
        }

        this.setIntegration = true;
        return this.integralSum;
    }

    // Numerical integration using the backward rectangular rule (instance method)
    // all parameters except number of intervals preset
    public double backward(int nIntervals) {
        this.nIntervals = nIntervals;
        this.setIntervals = true;
        return this.backward();
    }

    // Numerical integration using the backward rectangular rule (static method)
    // all parameters must be provided
    public static double backward(Function intFunc, double lowerLimit, double upperLimit, int nIntervals) {
        Integration intgrtn = new Integration(intFunc, lowerLimit, upperLimit);
        return intgrtn.backward(nIntervals);
    }

    // FORWARD RECTANGULAR METHODS
    // Numerical integration using the forward rectangular rule
    // all parameters preset
    public double forward() {

        double interval = (this.upperLimit - this.lowerLimit) / this.nIntervals;
        double x = this.lowerLimit;
        this.function.updateArgs(x);
        double y = this.function.calc();
        this.integralSum = 0.0D;

        for (int i = 0; i < this.nIntervals; i++) {
            // adjust last interval for rounding errors
            if (x > this.upperLimit) {
                x = this.upperLimit;
                interval -= (x - this.upperLimit);
            }
            this.function.updateArgs(x);
            // perform summation
            y = this.function.calc();
            this.integralSum += y * interval;
            x += interval;
        }
        this.setIntegration = true;
        return this.integralSum;
    }

    // Numerical integration using the forward rectangular rule
    // all parameters except number of intervals preset
    public double forward(int nIntervals) {
        this.nIntervals = nIntervals;
        this.setIntervals = true;
        return this.forward();
    }

    // Numerical integration using the forward rectangular rule (static method)
    // all parameters provided
    public static double forward(Function integralFunc, double lowerLimit, double upperLimit, int nIntervals) {
        Integration intgrtn = new Integration(integralFunc, lowerLimit, upperLimit);
        return intgrtn.forward(nIntervals);
    }

    public static double foreward(Function integralFunc, double lowerLimit, double upperLimit, int nIntervals) {
        Integration intgrtn = new Integration(integralFunc, lowerLimit, upperLimit);
        return intgrtn.forward(nIntervals);
    }

    public static void main(String[] args) {

        Function func = new Function("y=@(x)sin(x)-cos(x)");
        Integration intg = new Integration(func, 2, 3);
        intg.gaussQuad(20);
        System.err.println("The value " + intg.getIntegralSum());

        //0.84887248854057823517082799192315
    }

}
