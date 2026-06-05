/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.geom;

import static java.lang.Math.*;

/**
 *
 * @author GBEMIRO
 */
public class Plane {

    /**
     * The direction indices of the normal line vector to the plane
     */
    private Direction directionIndices;
    /**
     * The constant term in the plane equation. e.g in 2x+3y-9z-12
     * planeConstant=-12.
     */
    private double planeConstant;

    /**
     * Creates a Plane object given 3 points on it.
     *
     * @param p1 a point on the plane
     * @param p2 another point on the plane
     * @param p3 another point on the plane
     */
    public Plane(Point p1, Point p2, Point p3) {
        double a = (p3.z - p1.z) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.z - p1.z);
        double b = (p3.x - p1.x) * (p2.z - p1.z) - (p3.z - p1.z) * (p2.x - p1.x);
        double c = (p3.y - p1.y) * (p2.x - p1.x) - (p3.x - p1.x) * (p2.y - p1.y);
        directionIndices = new Direction(a, b, c);
        this.planeConstant = -1 * (a * p1.x + b * p1.y + c * p1.z);
    }

    /**
     * Creates a Plane object given the direction indices of its normal vector
     * (stored in a gameMath.Point object)
     *
     * @param directionIndices the direction indices of its normal vector
     * @param planeConstant the constant of the plane.
     */
    public Plane(Direction directionIndices, double planeConstant) {
        this.directionIndices = directionIndices;
        this.planeConstant = planeConstant;
    }//end constructor

    /**
     * Creates the plane given 2 lines on it.
     *
     * @param line1 A line on the plane.
     * @param line2 Another line on the plane.
     */
    public Plane(Line3D line1, Line3D line2) {

        double a = line2.getXyLine().getM() * line1.getXzLine().getM() - line1.getXyLine().getM() * line2.getXzLine().getM();
        double b = line2.getXzLine().getM() - line1.getXyLine().getM();
        double c = line1.getXyLine().getM() - line2.getXyLine().getM();
        double d = line2.getXzLine().getC() * -1 * c - line2.getXyLine().getC() * -1 * b;

        this.directionIndices = new Direction(a, b, c);
        this.planeConstant = d;
    }//end constructor

    /**
     *
     * @param line a line that is normal to the Plane object.
     * @param point a Point on the Plane object
     */
    public Plane(Line3D line, Point point) {
        Direction dir = line.direction();

        double a = dir.a;
        double b = dir.b;
        double c = dir.c;
        double d = -1 * (a * point.x + b * point.y + c * point.z);

        this.directionIndices = new Direction(a, b, c);
        this.planeConstant = d;
    }//end constructor

    /**
     *
     * @param dir the direction vector of the plane.This is an object of class
     * Direction.
     * @param point a Point on the Plane object
     */
    public Plane(Direction dir, Point point) {
        double a = dir.a;
        double b = dir.b;
        double c = dir.c;
        double d = -1 * (a * point.x + b * point.y + c * point.z);
        this.directionIndices = new Direction(a, b, c);
        this.planeConstant = d;
    }//end constructor

    /**
     *
     * @return this Plane object's Direction attribute.
     */
    public Direction getDirectionIndices() {
        return directionIndices;
    }//end method

    /**
     *
     * @param directionIndices sets this Plane object's Direction attribute
     */
    public void setDirectionIndices(Direction directionIndices) {
        this.directionIndices = directionIndices;
    }//end method

    /**
     *
     * @return the constant attribute of this Plane object
     */
    public double getPlaneConstant() {
        return planeConstant;
    }//end method

    /**
     *
     * @param planeConstant sets the constant attribute of this Plane object
     */
    public void setPlaneConstant(double planeConstant) {
        this.planeConstant = planeConstant;
    }//end method

    /**
     *
     * @return the angle between this Plane object and the XY plane as an Array
     * object of 2 double values.
     */
    public double[] angleXY() {
        double cosAng = directionIndices.c / sqrt(pow(directionIndices.a, 2) + pow(directionIndices.b, 2) + pow(directionIndices.c, 2));
        return new double[]{acos(cosAng), acos(-cosAng)};
    }//end method

    /**
     *
     * @return the angle between this Plane object and the XZ plane as an Array
     * object of 2 double values.
     */
    public double[] angleXZ() {
        double cosAng = directionIndices.b / sqrt(pow(directionIndices.a, 2) + pow(directionIndices.b, 2) + pow(directionIndices.c, 2));
        return new double[]{acos(cosAng), acos(-cosAng)};
    }//end method

    /**
     *
     * @return the angle between this Plane object and the YZ plane as an Array
     * object of 2 double values.
     */
    public double[] angleYZ() {
        double cosAng = directionIndices.a / sqrt(pow(directionIndices.a, 2) + pow(directionIndices.b, 2) + pow(directionIndices.c, 2));
        return new double[]{acos(cosAng), acos(-cosAng)};
    }//end method

    /**
     *
     * @param plane The angle this Plane object makes with another Plane object.
     * @return the angle between the 2 Plane objects.
     */
    public double angle(Plane plane) {
        Direction dir1 = directionIndices;
        Direction dir2 = plane.directionIndices;

        double a1 = dir1.a;
        double a2 = dir2.a;
        double b1 = dir1.b;
        double b2 = dir2.b;
        double c1 = dir1.c;
        double c2 = dir2.c;

        double num = a1 * a2 + b1 * b2 + c1 * c2;
        double den = sqrt((a1 * a1 + b1 * b1 + c1 * c1) * (a2 * a2 + b2 * b2 + c2 * c2));
        return acos(num / den);
    }//end method

    /**
     *
     * @param line the Line3D object whose angle with this Plane object is
     * desired.
     * @return the angle between this Plane object and the Line3D object passed
     * as a parameter to this method
     */
    public double angle(Line3D line) {
        return line.angle(this);
    }//end method

    /**
     *
     * @param x the x coordinate of a Point object on this Plane object.
     * @param y the y coordinate of a Point object on this Plane object.
     * @return the z coordinate of that Point object.
     */
    public double getZ(double x, double y) {
        return -1 * (directionIndices.a * x + directionIndices.b * y + planeConstant) / directionIndices.c;
    }//end method

    /**
     *
     * @param x the x coordinate of a Point object on this Plane object.
     * @param z the z coordinate of a Point object on this Plane object.
     * @return the y coordinate of that Point object.
     */
    public double getY(double x, double z) {
        return -1 * (directionIndices.a * x + directionIndices.c * z + planeConstant) / directionIndices.b;
    }//end method

    /**
     *
     * @param y the y coordinate of a Point object on this Plane object.
     * @param z the z coordinate of a Point object on this Plane object.
     * @return the x coordinate of that Point object.
     */
    public double getX(double y, double z) {
        return -1 * (directionIndices.b * y + directionIndices.c * z + planeConstant) / directionIndices.a;
    }//end method

    /**
     *
     * @param point The Point object in consideration
     * @return true if the Point object lies on this Plane object.
     */
    public boolean containsPoint(Point point) {
        double eval = directionIndices.a * point.x + directionIndices.b * point.y + directionIndices.c * point.z + planeConstant;
        return approxEquals(eval, 0);
    }//end method

    /**
     *
     * @param lin the Line3D object.
     * @return true if the Line3D object lies on this Plane object.
     */
    public boolean containsLine(Line3D lin) {
        return approxEquals(angle(lin), 0);
    }//end method

    /**
     *
     * @param line The Line3D object under consideration
     * @return a Point object that contains the point of intersection of this
     * Plane object and the given Line3D object.
     */
    public Point intersectionWith(Line3D line) {
        double a = directionIndices.a;
        double b = directionIndices.b;
        double c = directionIndices.c;
        double d = planeConstant;

        double num = -1 * (d + (c * line.getLINEXZIntercept()) + (b * line.getLINEXYIntercept()));
        double den = a + (b * line.getLINEXYGrad()) + (c * line.getLINEXZGrad());

        double x = num / den;

        double y = line.getLINEXYGrad() * x + line.getLINEXYIntercept();
        double z = line.getLINEXZGrad() * x + line.getLINEXZIntercept();

        return new Point(x, y, z);
    }//end method

    /**
     *
     * @param plane The Plane object under consideration
     * @return a Line3D object that contains the point of intersection of this
     * Plane object and the given Line3D object.
     */
    public Line3D intersectionWith(Plane plane) {
        double a1 = directionIndices.a;
        double b1 = directionIndices.b;
        double c1 = directionIndices.c;
        double d1 = planeConstant;

        double a2 = plane.directionIndices.a;
        double b2 = plane.directionIndices.b;
        double c2 = plane.directionIndices.c;
        double d2 = plane.planeConstant;

        Line lineXY = new Line((a2 - a1) / (c1 - c2), (d2 - d1) / (c1 - c2));
        Line lineXZ = new Line((a2 - a1) / (b1 - b2), (d2 - d1) / (b1 - b2));

        return new Line3D(lineXY, lineXZ);
    }//end method

    /**
     *
     * @return the equation of this Plane object as a function of x,y,z.
     */
    @Override
    public String toString() {
        return directionIndices.a + "*x + " + directionIndices.b + "*y + " + directionIndices.c + "*z + " + planeConstant + " = 0";
    }//end method

    /**
     * Compares two numbers to see if they are close enough to be almost the
     * same It checks if the values deviate by 1.0E-14 or lesser.
     *
     * @param val1 the first value to compare
     * @param val2 the second value to compare
     * @return true if the values deviate by 1.0E-14 or lesser.
     */
    public boolean approxEquals(double val1, double val2) {
        return abs(abs(val1) - abs(val2)) <= 1.0E-14;
    }

    /**
     * Compares two numbers to see if they are close enough to be almost the
     * same It checks if the values deviate by 1.0E-14 or lesser.
     *
     * @param val1 the first value to compare
     * @param val2 the second value to compare
     * @param minDeviation the minimum difference they must have to be
     * acceptably equal.
     * @return true if the values deviate by 1.0E-14 or lesser.
     */
    public boolean approxEquals(double val1, double val2, double minDeviation) {
        return abs(abs(val1) - abs(val2)) <= abs(minDeviation);
    }

    public static void main(String args[]) {
        Plane plane = new Plane(new Direction(2, 3, 4), 2);
        Plane planar = new Plane(new Direction(-1, -3, 2), 3);
        System.out.println(plane.intersectionWith(planar));
    }//end method

}//end class
