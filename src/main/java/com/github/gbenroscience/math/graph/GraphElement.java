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
package com.github.gbenroscience.math.graph;

import com.github.gbenroscience.parser.Scanner;
import com.github.gbenroscience.parser.Function;
import com.github.gbenroscience.util.FunctionManager;
import java.util.Arrays;
import java.util.List;

/**
 * Created by JIBOYE Oluwagbemiro Olaoluwa on 6/28/2016.
 */
public class GraphElement {

    private String function;
    /**
     * Store the f name to help with quick retrieval of the f later on.
     */
    private String functionName;
    double[] horizontalCoordinates = new double[0];
    double[] verticalCoordinates = new double[0];

    private GraphType graphType;

    public GraphElement(String function, GraphType graphType) {
        this.function = function;
        this.graphType = graphType;
    }

    public String getFunctionName() {
        return functionName;
    }

    public void setFunctionName(String functionName) {
        this.functionName = functionName;
    }

    public void setGraphType(GraphType graphType) {
        this.graphType = graphType;
    }

    public GraphType getGraphType() {
        return graphType;
    }

    public void setFunction(String function) {
        this.function = function;
    }

    public String getFunction() {
        return function;
    }

    public double[] getHorizontalCoordinates() {
        return horizontalCoordinates;
    }

    public void setHorizontalCoordinates(double[] horizontalCoordinates) {
        this.horizontalCoordinates = horizontalCoordinates;
    }

    public void setVerticalCoordinates(double[] verticalCoordinates) {
        this.verticalCoordinates = verticalCoordinates;
    }

    public double[] getVerticalCoordinates() {
        return verticalCoordinates;
    }

    public void fillCoords(double xLower, double xUpper, double xStep, double yStep, int drg) {

        if (graphType.isFunctionPlot()) {

            Function f = new Function(this.function);
            this.functionName = f.getName();
            FunctionManager.add(f);
            String horVariableName = f.getIndependentVariables().get(0).getName();
            double[][] values = f.evalRange(xLower, xUpper, xStep, horVariableName, drg);
            this.verticalCoordinates = values[0];
            this.horizontalCoordinates = values[1];
 //System.err.println("functionPlot: "+toString());
        } //[-200,200,300,-200:][1,3,-2,1:]
        else if (graphType.isVerticePlot()) {
            this.function = this.function.trim();
            String horizontal = this.function.substring(this.function.indexOf("["), this.function.indexOf("]")).trim();
            String vertical = this.function.substring(this.function.lastIndexOf("[") + 1, this.function.lastIndexOf("]")).trim();
            Scanner horCs = new Scanner(horizontal, false, ":", ",", "[","]");
            Scanner verCs = new Scanner(vertical, false, ":", ",", "[","]");

            List<String> horValues = horCs.scan();
            List<String> verValues = verCs.scan();
           
            int sz = horValues.size();
            if (sz > 0 && sz == verValues.size()) {
            this.horizontalCoordinates = new double[sz];
            this.verticalCoordinates = new double[sz];
                for (int i = 0; i < sz; i++) {
                    this.horizontalCoordinates[i] = Double.parseDouble(horValues.get(i));
                    this.verticalCoordinates[i] = Double.parseDouble(verValues.get(i));
                }
            }
           // System.err.println("verticesPlot: "+toString());

        }

    }

    /**
     *
     * @param searchValue The item whose index we are trying to estimate.
     * @param array The array to conduct binarySearch on
     * @return the index in the array where this item would be best placed. This
     * index is such that any element to its left would be less than
     * {@code searchValue} and any element to its right or at {@code index}
     * would be greater than or equal to it.
     */
    public static int binarySearch(double searchValue, double[] array) {
//3.1,3.21,4.13,4.4,5.0,5.2,5.21,5.32,6.0,6.01
        boolean notFound = true;
        int startLen = array.length;
        int indexShift = 0;
        while (notFound) {
            int midIndex = indexShift + startLen / 2;

            if (midIndex >= array.length - 1) {
                return array.length - 1;
            }
//Item is in the left partition.
            if (searchValue < array[midIndex]) {
                if (midIndex == 0) {
                    return midIndex;
                }
                if (searchValue >= array[midIndex - 1]) {
                    return midIndex;
                }
//System.out.println("1......[indexShift, midIndex] = ["+indexShift+", "+midIndex+"].");
                startLen /= 2;

            } else if (searchValue == array[midIndex]) {
                //  System.out.println("Elem found");
                return midIndex;
            } //Item is in the right partition
            else {
                indexShift = midIndex + 1;
                if (indexShift >= array.length - 1) {
                    return array.length - 1;
                }
                //System.out.println("2......[indexShift, midIndex] = ["+indexShift+", "+midIndex+"].");
                startLen /= 2;
            }

        }
        return -1;
    }

    @Override
    public String toString() {
        return String.format("{\n"
                + "  \"function\": \"%s\",\n"
                + "  \"functionName\": \"%s\",\n"
                + "  \"horizontalCoordinates\": \"%s\",\n"
                + "  \"verticalCoordinates\": \"%s\"\n"
                + "  }", function, functionName, Arrays.toString(horizontalCoordinates), Arrays.toString(verticalCoordinates));
    }

    
  
    public static void main(String[] args) {
        Grid.GraphDataSharer dataSharer = new Grid.GraphDataSharer();
        dataSharer.drg = 1;
        dataSharer.xLower = -100;
        dataSharer.xUpper = 100;
        dataSharer.yStep = 0.1;
        dataSharer.xStep = 0.1; 
        
        GridExpressionParser gridExpressionParser = new GridExpressionParser("[-3,-1,0,4,-3:][0,3,4,-1,0:];y(x)=sin(x-3*pi);v(x,y)=3*x!-10*y", dataSharer);
       
    }
    
}
