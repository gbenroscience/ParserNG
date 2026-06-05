/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package com.github.gbenroscience.math.graph;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public enum GraphType {

    FunctionPlot,VerticePlot,MixedInstructionsPlot;


    public boolean isFunctionPlot(){
        return this==FunctionPlot;
    }

    public boolean isVerticePlot(){
        return this==VerticePlot;
    }
    public boolean isMixedInstructionsPlot(){
        return this==MixedInstructionsPlot;
    }
}
