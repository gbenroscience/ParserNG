package com.github.gbenroscience.parser.pro.turbo.tools;




/**
 *
 * @author GBEMIRO
 */
public interface TileOperator {
    // Processes an entire tile in one go with ZERO internal branching
    void evaluate(double[] src, double[] dest, int tileSize);
}