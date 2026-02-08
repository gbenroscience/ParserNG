/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.math.graph;

import java.util.List;
import com.github.gbenroscience.parser.Scanner;

import java.util.ArrayList;
import static com.github.gbenroscience.parser.STRING.*;
import java.util.Objects;

/**
 * Objects of this class take a compound plot instruction, scan it into all
 * smaller plot instructions, and then analyze each instruction to determine its
 * type. They store a record of each smaller plot instruction in an ArrayList of
 * {@link GraphElement} objects
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class GridExpressionParser {

    public final Object lock = new Object();
    private String input;

    private ArrayList<GraphElement> graphElements = new ArrayList<>();

    /**
     * Set to false if it cannot be plotted.
     */
    private boolean canPlot = true;

    public GridExpressionParser(String input, Grid.GraphDataSharer dataSharer) {
        this.input = input;
        scan(dataSharer);
    }

    public void setInput(String input, Grid.GraphDataSharer dataSharer) {
        this.input = input;
        scan(dataSharer);
    }

    public String getInput() {
        return input;
    }

    public void setCanPlot(boolean canPlot) {
        this.canPlot = canPlot;
    }

    public boolean isCanPlot() {
        return canPlot;
    }

    /**
     *
     * @param input The input plot instruction.
     * @return the type of GraphType that input represents.
     * @throws NullPointerException
     */
    public static GraphType determineType(String input) throws NullPointerException {
        input = purifier(input);
        if (input.contains("[") && input.contains("]")) {
            return GraphType.VerticePlot;
        } else if (input.contains("=") && !input.contains("[") && !input.contains("]")) {
            return GraphType.FunctionPlot;
        } else {
            return null;
        }
    }

    public void setGraphElements(ArrayList<GraphElement> graphElements) {
        this.graphElements = graphElements;
    }

    public ArrayList<GraphElement> getGraphElements() {
        return graphElements;
    }

    /**
     *
     * @return the first {@link GraphElement} that is of type
     * {@link GraphType#FunctionPlot}
     */
    public GraphElement getFirstFunctionPlotTypeGraphElement() {
        for (GraphElement elem : graphElements) {
            if (elem != null && elem.getGraphType() != null && elem.getGraphType().isFunctionPlot()) {
                return elem;
            }
        }
        return null;
    }

    /**
     *
     * @param gep Another GridExpressionParser object. Appends the function
     * definitions in the other GridExpressionParser object to the functions in
     * this one.
     */
    public void addFunctions(GridExpressionParser gep) {
        if (gep.canPlot) {
            graphElements.addAll(gep.getGraphElements());
        }
    }

    /**
     * Isolates each complete plottable instruction.
     *
     * @param dataSharer
     */
    public final void scan(Grid.GraphDataSharer dataSharer) {
        graphElements.clear();
//[-200,200,300,-200:][1,3,-2,1:];y=@(x)3x+1;
        try {
            List<String> instructionTokens = new Scanner(purifier(input), false, ";").scan();
            int sz = instructionTokens.size();
            for (int i = 0; i < sz; i++) {
                try {
                    String instructionToken = instructionTokens.get(i);

                    final GraphElement elem = new GraphElement(instructionToken, determineType(instructionToken));
                    elem.fillCoords(dataSharer.xLower, dataSharer.xUpper, dataSharer.xStep, dataSharer.yStep, dataSharer.drg);
                    graphElements.add(elem);
                } catch (NullPointerException nol) {
                    nol.printStackTrace();
                    setCanPlot(false);
                }
            }
        }//end try
        catch (IndexOutOfBoundsException indexErr) {
            indexErr.printStackTrace();
        }

    }//end method

}//end class GridExpressionParser
