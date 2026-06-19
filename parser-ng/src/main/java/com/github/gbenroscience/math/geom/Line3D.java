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
public class Line3D extends Line {

    /**
     * The Line object that represents the image objects of this class produce
     * on the XY plane
     */
    private Line xyLine;
    /**
     * The Line object that represents the image objects of this class produce
     * on the XZ plane
     */
    private Line xzLine;

    /**
     * Creates a 3D line having 2 line attributes.
     *
     * The lines are the images produced by objects of this class on the x,y,z
     * planes.
     *
     * @param xyLine The Line object that represents the image objects of this
     * class produce on the XY plane
     *
     *
     * @param xzLine The Line object that represents the image objects of this
     * class produce on the XZ plane
     */
    public Line3D(Line xyLine, Line xzLine) {
        this.xyLine = xyLine;
        this.xzLine = xzLine;
    }

    /**
     * Creates a new Line object given a point through which it passes and the
     * direction coordinates of the Line object.
     *
     *
     * @param direction A Point object that stores the direction coordinates of
     * this object.
     * @param point A point through which this Line passes
     */
    public Line3D(Direction direction, Point point) {
        this.xyLine = new Line((direction.b / direction.a), point.y - ((direction.b * point.x) / direction.a));
        this.xzLine = new Line((direction.c / direction.a), point.z - ((direction.c * point.x) / direction.a));
    }

    /**
     *
     * @param p1 A point through which the Line3D object passes.
     * @param p2 Another point through which the Line3D object passes.
     */
    public Line3D(Point p1, Point p2) {

        double m = (p2.y - p1.y) / (p2.x - p1.x);
        double c = p1.y - m * p1.x;

        double n = (p2.z - p1.z) / (p2.x - p1.x);
        double d = p1.z - n * p1.x;

        this.xyLine = new Line(m, c);
        this.xzLine = new Line(n, d);

    }

    /**
     *
     * @return the XY plane image of objects of this class.
     */
    public Line getXyLine() {
        return xyLine;
    }

    /**
     *
     * @param xyLine sets the XY plane image of objects of this class.
     */
    public void setXyLine(Line xyLine) {
        this.xyLine = xyLine;
    }

    /**
     * ]
     *
     * @return the XZ plane image of objects of this class.
     */
    public Line getXzLine() {
        return xzLine;
    }

    /**
     *
     * @param xzLine sets the XZ plane image of objects of this class.
     */
    public void setXzLine(Line xzLine) {
        this.xzLine = xzLine;
    }

    /**
     *
     * @param p1 the first Point object
     * @param p2 the second Point object
     * @return the distance between the points.
     */
    @Override
    public double distance(Point p1, Point p2) {
        return sqrt(pow((p2.x - p1.x), 2) + pow((p2.y - p1.y), 2) + pow((p2.z - p1.z), 2));
    }

    /**
     * @return the square of distance between the points.
     */
    @Override
    public double distanceSquared(Point p1, Point p2) {
        return pow((p2.x - p1.x), 2) + pow((p2.y - p1.y), 2) + pow((p2.z - p1.z), 2);
    }

    /**
     *
     * @return the Direction object that specifies this line's direction in
     * space. the coordinates of the Direction object are eexpressed relative to
     * the x coordinate so the x coordinate is always 1.
     */
    public Direction direction() {
        return new Direction(1, xyLine.getM(), xzLine.getM());
    }

    /**
     *
     * @return the angle this Line object makes with another Line object
     * @param line the Line object whose angle with this Line object is needed
     */
    public double angle(Line3D line) {
        double m1 = this.xyLine.getM();
        double n1 = this.xzLine.getM();
        double m2 = line.xyLine.getM();
        double n2 = line.xzLine.getM();

        return acos((1 + (m1 * m2) + (n1 * n2))
                / sqrt((1 + pow(m1, 2) + pow(n1, 2)) * (1 + pow(m2, 2) + pow(n2, 2)))
        );

    }

    /**
     *
     * @return the gradient of this object's image line on the xy plane
     */
    public double getLINEXYGrad() {
        return xyLine.getM();
    }

    /**
     *
     * @return the y intercept of this object's image line on the xy plane
     */
    public double getLINEXYIntercept() {
        return xyLine.getC();
    }

    /**
     *
     * @return the gradient of this object's image line on the xz plane
     */
    public double getLINEXZGrad() {
        return xzLine.getM();
    }

    /**
     *
     * @return the z intercept of this object's image line on the xz plane
     */
    public double getLINEXZIntercept() {
        return xzLine.getC();
    }

    /**
     *
     * @return the angle this Line object makes with another Plane object
     * @param plane the Line object whose angle with this Line object is needed
     */
    public double angle(Plane plane) {
        Direction dir = plane.getDirectionIndices();

        double a = dir.a;
        double b = dir.b;
        double c = dir.c;
        double m = this.getXyLine().getM();
        double n = this.getXzLine().getM();

        return asin((a + b * m + c * n)
                / sqrt((1 + pow(m, 2) + pow(n, 2)) * (pow(a, 2) + pow(b, 2) + pow(c, 2))));
    }

    /**
     * Ascertains that a Line3D object passes through a given Point object
     *
     * @param p1 the Point object.
     * @return true if the Point object lies on this Line3D object
     */
    @Override
    public boolean passesThroughPoint(Point p1) {
        boolean truly = xyLine.passesThroughPoint(new Point(p1.x, p1.y));
        boolean verily = xzLine.passesThroughPoint(new Point(p1.x, p1.z));
        return truly && verily;
    }

    /**
     *
     * @param line the Line3D object to check whether or not it intersects with
     * this Line object
     * @return true if this Line3D object interferes with the Line3D object
     * passed to its method as a parameter.
     */
    @Override
    public boolean intersectsLine(Line line) {
        Line3D lin = (Line3D) line;
        boolean truly = this.xyLine.intersectsLine(lin.xyLine);
        boolean verily = this.xzLine.intersectsLine(lin.xzLine);
        return truly && verily;
    }

    /**
     *
     * @param line the Line3D object whose intersection with this Line3D object
     * is desired.
     * @return a Point object containing the point of intersection of both
     * Lines.
     */
    @Override
    public Point intersectionWithLine(Line line) {

        Line3D lin = (Line3D) line;
        if (intersectsLine(line)) {
            Point p1 = this.xyLine.intersectionWithLine(lin.xyLine);
            Point p2 = this.xzLine.intersectionWithLine(lin.xzLine);

            return new Point(p1.x, p1.y, p2.y);
        } else {
            return new Point(Double.NaN, Double.NaN, Double.NaN);
        }
    }//end method

    @Override
    public boolean isParallelTo(Line line) {
        Line3D lin = (Line3D) line;
        Direction dir = direction();
        Direction dir1 = lin.direction();
        return approxEquals(dir.a, dir1.a) && approxEquals(dir.b, dir1.b) && approxEquals(dir.c, dir1.c);
    }

    /**
     *
     * @param plane the Plane object.
     * @return true if this Line3D object lies on the given Plane object.
     */
    public boolean liesOn(Plane plane) {
        return approxEquals(plane.angle(this), 0);
    }

    @Override
    public String toString() {
        double m = xyLine.getM();
        double e = xyLine.getC();
        double n = xzLine.getM();
        double f = xzLine.getC();

        return "y = " + m + "*x + " + e + "\nz = " + n + "*x + " + f;
    }

    public static void main(String args[]) {
        Plane plane = new Plane(new Direction(0.1, 2, 0.5), 5);
        Line3D line = new Line3D(new Line(2, 3, 20, 13), new Line(-4, 12, 45, 23.65));
        Line3D lin = new Line3D(new Line(12, 33, 5, 13), new Line(3, 12, 45, 23.65));
        System.out.println(line.intersectionWithLine(lin));

    }

}
