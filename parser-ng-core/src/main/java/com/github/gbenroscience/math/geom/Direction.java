/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package com.github.gbenroscience.math.geom;

/**
 * Models Direction vectors.
 * They determine the nature of many objects
 * e.g Line objects and plane objects.
 * @author GBEMIRO
 */
public class Direction {
    /**
     * The  direction coordinate along the x axis
     */
double a;
    /**
     * The  direction coordinate along the y axis
     */
double b;
    /**
     * The  direction coordinate along the z axis
     */
double c;
      /**
       *
       * @param a The direction coordinate along the x axis
       * @param b The direction coordinate along the y axis
       * @param c The direction coordinate along the z axis
       */
    public Direction(double a, double b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
    }
/**
 *
 * @return the direction coordinate along the x axis
 */
    public double getA() {
        return a;
    }
/**
 *
 * @param a sets the direction coordinate along the x axis
 */
    public void setA(double a) {
        this.a = a;
    }
/**
 *
 * @return the direction coordinate along the y axis
 */
    public double getB() {
        return b;
    }
/**
 *
 * @param b sets the direction coordinate along the y axis
 */
    public void setB(double b) {
        this.b = b;
    }
/**
 *
 * @return the direction coordinate along the z axis
 */
    public double getC() {
        return c;
    }
/**
 *
 * @param c sets the direction coordinate along the z axis
 */
    public void setC(double c) {
        this.c = c;
    }

    @Override
    public String toString() {
        return "D = "+a+" i + "+b+" j + "+c+" k";
    }





}
