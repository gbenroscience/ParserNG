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
public class Point {
public double x;
public double y;
public double z;
/**
 * ICreates a Point object at the origin of
 * 3D space.
 */
public Point(){
    this.x=0;
    this.y=0;
    this.z=0;
}
/**
 * Initializes a 1D Point object
 * @param x the x coordinate of the Point object
 */
public Point(double x){
    this.x=x;
    this.y=0;
    this.z=0;
}
/**
 * Initializes a 2D Point object
 * @param x the x coordinate of the Point object
 * @param y the y coordinate of the Point object
 */
public Point(double x,double y){
    this.x=x;
    this.y=y;
    this.z=0;
}
/**
 * Creates an object of this class from an awt Point object.
 * @param pt The awt Point object
 */
public Point(java.awt.Point pt){
    this.x=pt.x;
    this.y=pt.y;
    this.z=0;
}


/**
 * Initializes a 2D Point object
 * @param x the x coordinate of the Point object
 * @param y the y coordinate of the Point object
 * @param z the z coordinate of the Point object
 */

    public Point(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }


/**
 * Creates a new Point object
 * similar to this one but not referring
 * to the same object
 * @param point The point to mutate
 */
    public Point(Point point){
        this.x=point.x;
        this.y=point.y;
        this.z=point.z;
    }

/**
 *
 * @param x sets x the x coordinate of the Point object
 */
    public void setX(double x) {
        this.x = x;
    }
/**
 *
 * @return x the x coordinate of the Point object
 */
    public double getX() {
        return x;
    }
/**
 *
 * @param y x the y coordinate of the Point object
 */
    public void setY(double y) {
        this.y = y;
    }
/**
 *
 * @return x the y coordinate of the Point object
 */
    public double getY() {
        return y;
    }
/**
 *
 * @param z sets the z coordinate of the Point object
 */
    public void setZ(double z) {
        this.z = z;
    }
/**
 *
 * @return the z coordinate of the Point object
 */
    public double getZ() {
        return z;
    }
/***
 *
 * @param pt the Point object whose distance to this Point object is required
 * @return the distance between this Point object and Point pt
 */

public double calcDistanceTo(Point pt){
    return sqrt(  pow( (x-pt.x),2)+pow( (y-pt.y),2)+pow( (z-pt.z),2) );
}
/**
 *
 * @param x1 The x coordinate of the first point.
 * @param y1 The y coordinate of the first point.
 * @param x2 The x coordinate of the second point.
 * @param y2 The y coordinate of the second point.
 * @return the distance between both points.
 */
public static double distance(double x1 , double y1 , double x2, double y2){
    double dx = (x2 - x1);
    double dy = (y2 - y1);
    return sqrt( dx*dx + dy*dy  );
}
/**
 *
 * @param pt the point between which an imaginary line runs
 * @return the gradient of the projection of the line joining
 * these points on the XY plane
 */
public double findXYGrad(Point pt){
    return (y-pt.y)/(x-pt.x);
}

/**
 *
 * @param pt the point between which an imaginary line runs
 * @return the gradient of the projection of the line joining
 * these points on the XZ plane
 */
public double findXZGrad(Point pt){
    return (z-pt.z)/(x-pt.x);
}

/**
 *
 * @param pt the point between which an imaginary line runs
 * @return the gradient of the projection of the line joining
 * these points on the YZ plane
 */
public double findYZGrad(Point pt){
    return (z-pt.z)/(y-pt.y);
}
/**
 *
 * @param horDisp The horizontal displacement from the current point.
 * @param verDisp The vertical displacement from the current point.
 * @return a new point having the given horizontal and vertical
 * displacements from this object.
 */
public Point pointAtDisplacement( double horDisp, double verDisp ){
return new Point( x+horDisp, y+verDisp );
}
/**
 * @param line The reference line that this point object lies on.
 * @param distance The perpendicular distance from this point
 * to a point in space.
 * @return a new point that is the point, distance "distance"
 * from this object.
 */
public Point pointAtDisplacement( Line line,double distance ){
    double ang = atan( -1/line.getM() );
return new Point( x+distance*cos(ang), y+distance*sin(ang) );
}

/**
 * Converts objects of this class to
 * the normal Point object.
 * @return a java.awt.Point object from
 * this Point object
 */
public java.awt.Point getAWTPoint(){
    return new java.awt.Point(  (int) this.x, (int) this.y  );
}
/**
 * Converts objects of this class to
 * the normal Point object.
 * @param point
 * @return a java.awt.Point object from
 * an object of this class.
 */
public static java.awt.Point getAWTPoint(Point point){
    return new java.awt.Point(  (int)point.x,(int)point.y  );
}
/**
 *
 * @param p1 the
 * @return an object of type gameMath.Point
 * from an object of type java.awt.Point
 */
public static Point getGameMathPoint( java.awt.Point p1){
    return new Point(  p1.x,p1.y );
}

/**
 *
 * @param p1 The first Point object.
 * @param p2 The second Point object.
 * @return The Point object that contains the coordinates
 * of the midpoint of the line joining p1 and p2
 */
public static Point midPoint(Point p1,Point p2){
    return new Point(0.5*(p1.x+p2.x),0.5*(p1.y+p2.y)  );
}

/**
 *
 * @param p1
 * @param p2
 * @return true if the 2 points and this one
 * lie on the same straight line.
 */
public boolean isCollinearWith(Point p1,Point p2){
    Line line=new Line(p1,p2);
    return line.passesThroughPoint(this);
}
public void setLocation( double x , double y){
   this.x = x;
   this.y = y;
}

/**
 *
 * @param p1
 * @param p2
 * @return true if this Point object lies on the same
 * straight line with p1 and p2 and it lies in between them.
 */
public boolean liesBetween(Point p1,Point p2){
    boolean truly1 = ( (p1.x<=x&&p2.x>=x)||(p2.x<=x&&p1.x>=x)  );
boolean truly2 = ( (p1.y<=y&&p2.y>=y)||(p2.y<=y&&p1.y>=y)  );
boolean truly3 = ( (p1.z<=z&&p2.z>=z)||(p2.z<=z&&p1.z>=z)  );

return truly1&&truly2&&truly3&&isCollinearWith(p1, p2); 
}


    @Override
public String toString(){
    return "("+x+" "+y+" "+z+")" ;
}

    @Override
    public boolean equals(Object obj){
        return this.toString().equals(obj.toString());
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.x) ^ (Double.doubleToLongBits(this.x) >>> 32));
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.y) ^ (Double.doubleToLongBits(this.y) >>> 32));
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.z) ^ (Double.doubleToLongBits(this.z) >>> 32));
        return hash;
    }


}
