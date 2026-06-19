/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.geom;

import java.awt.Graphics;
import java.awt.Rectangle;
import static java.lang.Math.*;
import java.awt.Polygon;
import java.awt.geom.Area;
import java.awt.geom.Line2D;

/**
 *
 * @author GBEMIRO
 */
public class Line {

    private double m;
    private double c;
    private double xValue = Double.NaN;

    /**
     * Creates a straight line parallel to the x axis
     */
    public Line() {
        this.m = 0;
        this.c = 0;
    }

    /**
     * Creates a new Line object given the gradient and the y intercept
     *
     * @param m the gradient of the Line object
     * @param c the y intercept of the Line object
     */
    public Line(double m, double c) {
        this.m = m;
        this.c = c;
 
    }

    /**
     * Creates a new Line object give the gradient and a point on the line.
     *
     * @param m the gradient of the Line object
     * @param p a Point object that lies on the Line object.
     */
    public Line(double m, Point p) {
        this.m = m;
        this.c = p.y - (m * p.x);

        if (isVertical()) {
            xValue = p.x;
        }

    }
    

    /**
     * Creates a line between points joining: the points x1,y1 and x2,y2
     *
     * @param x1 the x coordinate of the first point
     * @param y1 the y coordinate of the first point
     * @param x2 the x coordinate of the second point
     * @param y2 the y coordinate of the second point
     */
    public Line(double x1, double y1, double x2, double y2) {
        this.m = new Point(x1, y1).findXYGrad(new Point(x2, y2));
        this.c = y1 - this.m * x1;

        if (isVertical()) {
            xValue = x1;
        }
    }

    /**
     * Creates a line between points joining: the points p1 and p2
     *
     * @param p1 the first point
     * @param p2 the second point
     */
    public Line(Point p1, Point p2) {
        this.m = p1.findXYGrad(p2);
        this.c = p1.y - this.m * p1.x;

        if (isVertical()) {
            xValue = p1.x;
        }
    }

    /**
     *
     * @param m sets the gradient of this Line.
     */
    public void setM(double m) {
        this.m = m;
    }

    /**
     *
     * @return the gradient of this Line.
     */
    public double getM() {
        return m;
    }

    /**
     *
     * @param c sets the intercept of this Line.
     */
    public void setC(double c) {
        this.c = c;
    }

    /**
     *
     * @return the intercept of this Line.
     */
    public double getC() {
        return c;
    }

    /**
     *
     * @param y the y coordinate of a given point on a Line object.
     * @return the x coordinate of that point.
     */
    public double getX(double y) {
        return (y - this.c) / this.m;
    }

    /**
     *
     * @param x the x coordinate of a given point on a Line object.
     * @return the y coordinate of that point.
     */
    public double getY(double x) {
        return (this.m * x) + this.c;
    }

    /**
     *
     * @return true if this Line object is a vertical one.
     */
    public boolean isVertical() {
        return this.m == Double.POSITIVE_INFINITY || this.m == Double.NEGATIVE_INFINITY;
    }

    /**
     * This method tells the constant value that a vertical line, one whose
     * equation is of the form x = constant has.
     *
     * @return the constant value that x has along this line, if and only if
     * this line is vertical.
     */
    public double getxValue() {
        if (isVertical()) {
            return xValue;
        }
        throw new ArithmeticException(" THE LINE IS OF THE FORM Y = M*X + C\n"
                + "SO IT HAS NO X VALUE.\n"
                + "THE X VALUE IS THE CONSTANT VALUE SUCH THAT X = CONSTANT, AND IS ONLY FOR "
                + "VERTICAL LINES.");
    }

    /**
     *
     * Finds the distance between 2 Point objects lying on this Line object They
     * must lie on this Line object, else the method will return 0;
     *
     * @param p1 the first Point object to consider
     * @param p2 the second Point object to consider
     * @return the distance along this Line object between the 2 given Point
     * objects lying on it
     */
    public double distance(Point p1, Point p2) {
        double dist = 0;
        if (passesThroughPoint(p1) && passesThroughPoint(p2)) {
            dist = p2.calcDistanceTo(p1);
        }
        return dist;
    }

    /**
     *
     * Finds the square of the distance between 2 Point objects lying on this
     * Line object They must lie on this Line object, else the method will
     * return 0;
     *
     * @param p1 the first Point object to consider
     * @param p2 the second Point object to consider
     * @return the distance along this Line object between the 2 given Point
     * objects lying on it
     */
    public double distanceSquared(Point p1, Point p2) {
        double dist = 0;
        if (passesThroughPoint(p1) && passesThroughPoint(p2)) {
            dist = pow((p2.x - p1.x), 2) + pow((p2.y - p1.y), 2);
        }
        return dist;
    }

    /**
     *
     * @param d The distance.
     * @return the equation of a parallel line at a given perpendicular distance
     * from this Line object.
     */
    public Line getParallelLineAtDistance(double d) {
        double ang = atan(m) - PI / 2;
        double intercept = c + d / sin(ang);
        return new Line(m, intercept);

    }

    /**
     *
     * @param horDisp The horizontal displacement of the new line from this
     * line.
     * @param verDisp The vertical displacement of the new line from this line.
     * @return the equation of a parallel line at a given horizontal
     * displacement and a given vertical displacement from this object.
     */
    public Line getParallelLineAtDistance(int horDisp, int verDisp) {
        double ang = atan(m) - PI / 2;
        double theta = atan(-1 / m);
        double d = sqrt(horDisp * horDisp + verDisp * verDisp);
        double intercept = c + d / sin(ang);
        return new Line(m, intercept);

    }

    /**
     *
     * @param line the Line object to be checked if or not it intersects with
     * this one.
     * @return true if the 2 Line objects intersect.
     */
    public boolean intersectsLine(Line line) {
        return !isParallelTo(line);
    }

    /**
     * Checks if this Line object is parallel to another.
     *
     * @param line the Line object to be checked against this one for
     * parallelism
     * @return true if it is parallel to the other Line object
     */
    public boolean isParallelTo(Line line) {
        //both are not vertical, so check their gradients to
        //see if they are the same.
        if (!isVertical() && !line.isVertical()) {
            return approxEquals(this.m, line.m);
        } //both are vertical, means that they are parallel
        else if (isVertical() && line.isVertical()) {
            return true;
        } //one vertical and the other not vertical means that they are not parallel
        else {
            return false;
        }
    }

    /**
     *
     * @param p1 the Point object that we wish to check if or not it lies on
     * this Line object.
     * @return true if it lies on this Line object
     */
    public boolean passesThroughPoint(Point p1) {
        return approxEquals(p1.y, (this.m * p1.x + this.c));
    }

    /**
     *
     * @param line the Line object whose point of intersection with this Line
     * object is required
     * @return the point of intersection of both Line objects
     */
    public Point intersectionWithLine(Line line) {
        if (!isVertical() && !line.isVertical()) {
            double x = (-1 * (this.c - line.c) / (this.m - line.m));
            double y = this.m * x + this.c;
            return new Point(x, y);
        } else if (isVertical() && !line.isVertical()) {
            return new Point(xValue, line.m * xValue + line.c);
        } else if (!isVertical() && line.isVertical()) {
            return new Point(line.xValue, this.m * line.xValue + this.c);
        } //both are vertical, so throw an exception since no intersection can occur.
        else {
            throw new ArithmeticException("BOTH LINES ARE VERTICAL, NO INTERSECTION CAN OCCUR.");
        }

    }

    /**
     * Draws this Line object for the interval between x1 and x2.
     *
     * @param g The Graphics object used to draw this Line object.
     * @param x1 The x coordinate of the first point on this Line object where
     * drawing is to start
     * @param x2 The x coordinate of the second point on this Line object where
     * drawing is to start
     */
    public void draw(Graphics g, double x1, double x2) {
        g.drawLine((int) x1, (int) getY(x1), (int) x2, (int) getY(x2));

    }

    /**
     * Compares two numbers to see if they are close enough to be almost the
     * same It checks if the values deviate by 1.0E-14 or lesser.
     *
     * @param val1 the first value to compare
     * @param val2 the second value to compare
     * @return true if the values deviate by 1.0E-14 or lesser.
     */
    public static boolean approxEquals(double val1, double val2) {
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
    public static boolean approxEquals(double val1, double val2, double minDeviation) {
        return abs(abs(val1) - abs(val2)) <= abs(minDeviation);
    }

    /**
     *
     * @param ang The angle of rotation in rads.
     * @param p The point about which to rotate this Line object,
     * @return this Line object after rotating it through theta rads about
     * point, p.
     */
    public void rotate(double ang, Point p) {
        double x1 = 0;
        double y1 = getY(x1);
        Point p1 = new Point(x1, y1);
        double y2 = 0;
        double x2 = getX(y2);

        Point p2 = new Point(x2, y2);

        Point p11 = ROTOR.planarXYRotate(p1, p, ang);
        Point p22 = ROTOR.planarXYRotate(p2, p, ang);

        Line l = new Line(p11, p22);
        setM(l.m);
        setC(l.c);
    }

    /**
     *
     * @param ang The angle of rotation in rads.
     * @param p The point about which to rotate this Line object,
     * @return a new Line object after rotating it through theta rads about
     * point, p.
     */
    public Line rotateLine(double ang, Point p) {
        double x1 = 0;
        double y1 = getY(x1);
        Point p1 = new Point(x1, y1);
        double y2 = 0;
        double x2 = getX(y2);

        Point p2 = new Point(x2, y2);

        Point p11 = ROTOR.planarXYRotate(p1, p, ang);
        Point p22 = ROTOR.planarXYRotate(p2, p, ang);

        return new Line(p11, p22);
    }

    /**
     *
     * @param p1 The first polygon.
     * @param p2 The second polygon.
     * @return true if the 2 polygons intersect.
     */ 
    public static boolean polygonIntersects(java.awt.Polygon p1, java.awt.Polygon p2) {

        int x1[] = p1.xpoints;
        int y1[] = p1.ypoints;

        int x2[] = p2.xpoints;
        int y2[] = p2.ypoints;
//Check for edge collisions
        for (int j = 0; j < x1.length - 1; j++) {
                Line l = new Line(x1[j], y1[j], x1[j + 1], y1[j + 1]);
                for (int i = 0; i < x2.length - 1; i++) {
                        Line ll = new Line(x2[i], y2[i], x2[i + 1], y2[i + 1]);
                        if (l.intersectsLine(ll)) {
                            Point p = l.intersectionWithLine(ll);
                            if (p2.contains(p.getAWTPoint()) && p1.contains(p.getAWTPoint())) {
                                return true;
                            }
                        }//end if
                }//end inner for
        }//end outer for
        /**
         * [x0,x1,x2,x3]--4 entries
         * [y0,y1,y2,y3]--4 entries
         * 
         * L1 =[x0,y0]-[x1,y1]
         * L2 =[x1,y1]-[x2,y2]
         * L3 =[x2,y2]-[x3,y3]
         * L4 =[x3,y3]-[x1,y1]
         * 
         * 
         * 
         *         /**
         * [x0,x1,x2,x3]--4 entries
         * [y0,y1,y2,y3]--4 entries
         * 
         * L1 =[x0,y0]-[x1,y1]
         * L2 =[x1,y1]-[x2,y2]
         * L3 =[x2,y2]-[x3,y3]
         * L4 =[x3,y3]-[x1,y1]
         * 
         * 
         * 
         * 
         */

        //Now check for containment cases
        for (int j = 0; j < x1.length; j++) {
            if (p2.contains(x1[j], y1[j])) {
                return true;
            }
        }

        for (int j = 0; j < x2.length; j++) {
            if (p1.contains(x2[j], y2[j])) {
                return true;
            }
        }

        return false;
    }

    public static boolean polygonIntersectsNew(Polygon p1, Polygon p2) {
        Area area1 = new Area(p1);
        Area area2 = new Area(p2);
        area1.intersect(area2);
        return !area1.isEmpty();
    }

    /**
     *
     * @param polygon
     * @param rect
     * @return true if the polygon and the rectangle intersect.
     */
    public static boolean polygonIntersects(java.awt.Polygon polygon, java.awt.Rectangle rect) {
        java.awt.Polygon p = Line.convertRectangleToPolygon(rect);
        return polygonIntersects(polygon, p);
    }

    /**
     *
     * @param p A polygon whose area is required.
     */
    public static double polygonArea(java.awt.Polygon p) {

        int[] x = p.xpoints;
        int[] y = p.ypoints;
        double det = 0;
        for (int i = 0; i < x.length; i++) {
            try {
                det += (x[i + 1] * y[i] - x[i] * y[i + 1]);
            } catch (IndexOutOfBoundsException indexException) {
                return (det / 2);
            }
        }//end

        return (det / 2);
    }

    /**
     *
     * @param rectangle
     * @return a java.awt.Polygon object that has the same properties as the
     * rectangle
     */
    public static java.awt.Polygon convertRectangleToPolygon(Rectangle rectangle) {

        int w = rectangle.width;
        int h = rectangle.height;
        int x1 = rectangle.x;
        int y1 = rectangle.y;

        int[] x = new int[]{x1, x1, x1 + w, x1 + w, x1};
        int[] y = new int[]{y1, y1 + h, y1 + h, y1, y1};
        return new java.awt.Polygon(x, y, x.length);
    }

    @Override
    public String toString() {
        if (isVertical()) {
            return "x = " + xValue;
        }//end if
        else {
            return "y = " + this.m + "*x + " + this.c;
        }//end else
    }

    public static void main(String args[]) {
        java.awt.Polygon p = new java.awt.Polygon(new int[]{10, 6, 8, 12, 14, 10}, new int[]{12, 7, 2, 2, 7, 12},
                6);
        java.awt.Polygon p1 = new java.awt.Polygon(new int[]{9, 7, 8, 10, 11, 9}, new int[]{10, 8, 0, 0, 8, 10},
                6);

        System.out.println(Line.polygonIntersects(p, p1));

    }

}//end class Line
