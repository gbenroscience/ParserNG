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

import com.github.gbenroscience.math.graph.tools.GraphColor;
import com.github.gbenroscience.math.graph.tools.GraphFont;
import com.github.gbenroscience.parser.*;
import com.github.gbenroscience.math.*;
import static java.lang.Math.*;

import static com.github.gbenroscience.parser.STRING.*;  
import com.github.gbenroscience.util.Dimension;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class Grid {

    public static final int MIN_GRID_SIZE = 6;
    public static final int MAX_GRID_SIZE = 200;

    /**
     * The function to be plotted. This function can be homogeneous i.e an
     * instruction to plot a single curve or set of points, or heterogeneous i.e
     * instructions to plot a group of curves.
     *
     * For instance, an homogeneous instruction that plots a set of points is
     * [x1,x2,x3,.....:][y1,y2,y3,.......:] The array of numbers...x1,x2.. are a
     * set of x coordinates, and the array of numbers y1,y2....are the
     * corresponding set of y coordinates.
     *
     */
    //private String function;
    /**
     * Object that handles the input expression and scans it into any lesser
     * instructions, determines their types and stores information about all
     * these in one of its attributes.
     */
    private GridExpressionParser gridExpressionParser;

    /**
     * sets whether the grid lines will be visible or not.
     */
    private boolean showGridLines;
    /**
     * sets whether the major axes are labeled with numbers or not
     */
    private boolean labelAxis;

    /**
     * If true,then the object will attempt to draw the graph based on a scale
     * that it generates automatically.
     */
    private boolean autoScaleOn = true;

    /**
     * The coefficient of x at the leftmost visible point on the graph.
     */
    private double lowerVisibleX;
    /**
     * The coefficient of x at the rightmost visible point on the graph.
     */
    private double upperVisibleX;

    /**
     * The coefficient of y at the bottom-most(lol) visible point on the graph.
     */
    private double lowerVisibleY;
    /**
     * The coefficient of y at the top-most visible point on the graph.
     */
    private double upperVisibleY;

    /**
     * determine the size of the small boxes that make up the grid.
     */
    private Dimension gridSize = new Dimension(0, 0);
    /**
     * The color of the grid.
     */
    private GraphColor gridColor;
    /**
     * The color of the major axes, x and y.
     */
    private GraphColor majorAxesColor;
    /**
     * The color of the ticks.
     */
    private GraphColor tickColor;
    /**
     * The color used to generate the plot.
     */
    private GraphColor plotColor;
    /**
     * The upper value of x up to which the graph will be plotted.
     */
    private double upperXLimit;
    /**
     * The lower value of x up to which the graph will be plotted.
     */
    private double lowerXLimit;
    /**
     * The resolution of the graph along x. xStep is the equivalent in
     * calculation of the size (width or height) of each grid box. So if
     * gridSize=2 screen pixels and xStep =0.01. Then a box of length 2 units
     * corresponds to 0.01 units in the user plot along x.
     */
    private double xStep;
    /**
     * The resolution of the graph along y. yStep is the equivalent in
     * calculation of the size (width or height) of each grid box. So if
     * gridSize=2 and yStep =0.01 Then a distance of 2 units on the graph
     * corresponds to 0.01 units in the user plot along y.
     */
    private double yStep;
    /**
     * The length of the longer ticks used to mark off the graph.
     */
    private int majorTickLength;
    /**
     * The maximum number of iterations allowed per plot.
     */
    private static final double MAX_ITERATIONS = 500.0;

    /**
     * The length of the shorter ticks used to mark off the graph
     */
    private int minorTickLength;
    /**
     * The point where the origin of the graph will reside on the screen.
     */
    private Point locationOfOrigin;

    /**
     * The text used to label the horizontal axis. By default , its value is X.
     */
    private Variable horizontalAxisLabel = new Variable("X");
    /**
     * The text used to label the vertical axis. By default , its value is Y.
     */
    private Variable verticalAxisLabel = new Variable("Y");
    /**
     * The GraphFont object used to write on the graph.
     */
    private GraphFont font;
    /**
     * The scale suggested by the programmer. Cane be modified by the user to
     * see more detail.
     */
    private Size defaultScale;
    /**
     * determines the mode in which trig operations will be carried out on
     * numbers.If DRG==0,it is done in degrees if DRG==1, it is done in radians
     * and if it is 2, it is done in grads.By default, it is done in grads.
     */
    private int DRG = 1;
    /**
     * The panel on which this grid will be laid.
     */
    private JPanel component;

    private volatile boolean lockedForMathCalc = false;
    /**
     * If updates happened while the math-calc-lock was in place, register it
     * here, so that we can call the replot function again. This helps us incur
     * only one extra call instead of a flurry of calls
     */
    private volatile boolean updateHappenedDuringLock = false;
    private volatile boolean refreshingIndices = false;

    private volatile GraphDataSharer dataSharer = new GraphDataSharer();

    /**
     * @param function The function to be plotted on this Grid object. It may be
     * an algebraic one or a geometric or vertices based one. The algebraic one
     * is entered as y=f(x)..while the geometric one is entered in the form:
     * [2,3,4,5]:[-90,3,15,9]
     * @param showGridLines sets whether the grid lines will be visible or not.
     * @param labelAxis
     * @param gridSize determine the size of the small boxes that make up the
     * grid.
     * @param gridColor The color of the grid.
     * @param tickColor The color of the ticks used to mark off the coordinate
     * axes.
     * @param plotColor
     * @param majorTickLength The length of the longer ticks used to mark off
     * the graph.
     * @param minorTickLength The length of the shorter ticks used to mark off
     * the graph
     * @param majorAxesColor The color of the major axes, x and y.
     * @param upperXLimit The upper value of x up to which the graph will be
     * plotted.
     * @param lowerXLimit The lower value of x up to which the graph will be
     * plotted.
     * @param xStep The resolution of the graph along x.
     * @param yStep
     * @param font
     * @param component The component on which this grid will be laid.
     */
    public Grid(String function, boolean showGridLines, boolean labelAxis,
            int gridSize, GraphColor gridColor, GraphColor majorAxesColor, GraphColor tickColor, GraphColor plotColor,
            int majorTickLength, int minorTickLength,
            double lowerXLimit, double upperXLimit,
            double xStep, double yStep, GraphFont font, JPanel component) {
        /**
         * identifies the kind of input, be it a vertices based plot or an
         * algebraic one. Based on the recommendations of this section, it then
         * sets the graph-type.
         */
        this.component = component;
        this.showGridLines = showGridLines;
        this.labelAxis = labelAxis;
        this.gridSize = new Dimension(gridSize, gridSize);
        this.gridColor = gridColor;
        this.majorAxesColor = majorAxesColor;
        this.tickColor = tickColor;
        this.plotColor = plotColor;
        setMajorTickLength(majorTickLength);
        setMinorTickLength(minorTickLength);

        this.locationOfOrigin = new Point(component.getSize().width / 2, component.getSize().height / 2);

        computeXMinBoundPossibleOnScreen();
        computeXMaxBoundPossibleOnScreen();
        computeYMinBoundPossibleOnScreen();
        computeYMaxBoundPossibleOnScreen();

        this.xStep = xStep;
        this.yStep = yStep;

        this.upperXLimit = upperXLimit;
        this.lowerXLimit = lowerXLimit;

        setRefreshingIndices(true);
        this.font = font;
    }//end constructor

    public static class GraphDataSharer {

        public double xStep = 1.0;
        public double yStep = 1.0;
        public int drg = 0;
        public double xLower = 0.0;
        public double xUpper = 0.0;

        public GraphDataSharer() {

        }

    }

    /**
     * validates the {@link Grid#xStep} attribute. It checks if the number of
     * iterations is not greater.
     */
    public void validateMaxIterations() {
        double validateIterations = (upperXLimit - lowerXLimit) / xStep;
        if (validateIterations > MAX_ITERATIONS) {
            xStep = (upperXLimit - lowerXLimit) / MAX_ITERATIONS;
        }
    }

    public void setComponent(JPanel component) {
        this.component = component;
    }

    public JPanel getComponent() {
        return component;
    }

    public void setDRG(int DRG) {
        this.DRG = DRG;
        this.dataSharer.drg = this.DRG;
        setRefreshingIndices(true);
    }

    public int getDRG() {
        return DRG;
    }

    public Variable getHorizontalAxisLabel() {
        return horizontalAxisLabel;
    }

    public void setHorizontalAxisLabel(Variable horizontalAxisLabel) {
        this.horizontalAxisLabel = horizontalAxisLabel;
    }

    public Variable getVerticalAxisLabel() {
        return verticalAxisLabel;
    }

    public void setVerticalAxisLabel(Variable verticalAxisLabel) {
        this.verticalAxisLabel = verticalAxisLabel;
    }

    public void setPlotColor(GraphColor plotColor) {
        this.plotColor = plotColor;
    }

    public void setDefaultScale(Size defaultScale) {
        this.defaultScale = defaultScale;
        setRefreshingIndices(true);
    }

    public boolean isAutoScaleOn() {
        return autoScaleOn;
    }

    public void setAutoScaleOn(boolean autoScaleOn) {
        this.autoScaleOn = autoScaleOn;
        setRefreshingIndices(true);
    }

    public Size getDefaultScale() {
        return defaultScale;
    }

    public GraphColor getPlotColor() {
        return plotColor;
    }

    public void setFont(GraphFont font) {
        this.font = font;
    }

    public GraphFont getFont() {
        return font;
    }

    public GridExpressionParser getGridExpressionParser() {
        return gridExpressionParser;
    }

    public void setGridExpressionParser(GridExpressionParser gridExpressionParser) {
        this.gridExpressionParser = gridExpressionParser;
    }

    public double getLowerVisibleX() {
        return lowerVisibleX;
    }

    public double getUpperVisibleX() {
        return upperVisibleX;
    }

    public double getLowerVisibleY() {
        return lowerVisibleY;
    }

    public double getUpperVisibleY() {
        return upperVisibleY;
    }

    public GraphColor getGridColor() {
        return gridColor;
    }

    public void setGridColor(GraphColor gridColor) {
        this.gridColor = gridColor;
    }

    public void setGridSize(Dimension gridSize) {
        setGridSize(gridSize.width, gridSize.height);
    }

    public void setGridSize(int wid, int hei) {

        wid = wid <= MIN_GRID_SIZE ? MIN_GRID_SIZE : wid;
        wid = wid >= MAX_GRID_SIZE ? MAX_GRID_SIZE : wid;
        hei = hei <= MIN_GRID_SIZE ? MIN_GRID_SIZE : hei;
        hei = hei >= MAX_GRID_SIZE ? MAX_GRID_SIZE : hei;

        if (this.gridSize != null) {
            this.gridSize.width = wid;
            this.gridSize.height = hei;
        } else {
            this.gridSize = new Dimension(wid, hei);
        }
    }

    public Dimension getGridSize() {
        return gridSize;
    }

    public static double getMAX_ITERATIONS() {
        return MAX_ITERATIONS;
    }

    public Point getLocationOfOrigin() {
        return locationOfOrigin;
    }

    public void setLocationOfOrigin(Point locationOfOrigin) {
        this.locationOfOrigin = locationOfOrigin;
        setRefreshingIndices(true);
    }

    public GraphColor getMajorAxesColor() {
        return majorAxesColor;
    }

    public void setMajorAxesColor(GraphColor majorAxesColor) {
        this.majorAxesColor = majorAxesColor;
    }

    public int getMajorTickLength() {
        return majorTickLength;
    }

    public final void setMajorTickLength(int majorTickLength) {
        this.majorTickLength = majorTickLength;
    }

    public int getMinorTickLength() {
        return minorTickLength;
    }

    public final void setMinorTickLength(int minorTickLength) {
        this.minorTickLength = (minorTickLength < (majorTickLength / 2.0)) ? minorTickLength : majorTickLength / 2;
    }

    public double getLowerXLimit() {
        return lowerXLimit;
    }

    public final synchronized void setRefreshingIndices(boolean refreshingIndices) {
        this.refreshingIndices = refreshingIndices;
        /**
         * Only the replot function can set the refreshingIndices field to
         * false. Other setters and functions may set it to true. If a function
         * sets it to true, it means it wants the ui to run its math intensive
         * 'replotOnLimitChangeFunction' and update the graph. But that method
         * is not only expensive, it also spawns a new thread every time. So we
         * want only one instance of it to run at any given time.
         *
         * So, check if a lock exist(lockedForMathCalc == true) if so notify the
         * system that another update has happened by setting
         * 'updateHappenedDuringLock' to true and exit. Else if a lock does not
         * exist, check if the call is an update needed
         * call(refreshingIndices==true) immediately call the
         * 'replotOnLimitChangeFunction' to update the UI.
         */
        if (this.refreshingIndices) {
            if (this.lockedForMathCalc) {
                // A thread is already running. Just set the "reminder" flag.
                this.updateHappenedDuringLock = true;
            } else {
                // No thread is running. Start the calculation.
                this.replotOnLimitChange();
            }
        } else {
            // This part is called when a thread FINISHES (refreshingIndices = false).
            // Check if we need to start a new update because one was requested while we were busy.
            if (this.updateHappenedDuringLock) {
                this.updateHappenedDuringLock = false; // Reset the reminder
                this.replotOnLimitChange();
            }
        }
    }

    public boolean isRefreshingIndices() {
        return refreshingIndices;
    }

    public void replotOnLimitChange() {
        this.updateHappenedDuringLock = false;
        this.lockedForMathCalc = true;

        computeXMinBoundPossibleOnScreen();
        computeXMaxBoundPossibleOnScreen();
        computeYMinBoundPossibleOnScreen();
        computeYMaxBoundPossibleOnScreen();

        final Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Determine if limits need updating
                    boolean needsUpdate = false;
                    if (lowerVisibleX <= lowerXLimit || upperXLimit <= upperVisibleX) {
                        needsUpdate = true;

                        // Update limits
                        Grid.this.lowerXLimit = Grid.this.lowerVisibleX - 50 * gridSize.width * xStep;
                        Grid.this.upperXLimit = Grid.this.upperVisibleX + 50 * gridSize.width * xStep;

                        // SYNCHRONIZED LOOP: Prevents ConcurrentModificationException
                        synchronized (gridExpressionParser.lock) {
                            for (GraphElement gr : gridExpressionParser.getGraphElements()) {
                                if (gr.getGraphType() == GraphType.FunctionPlot) {
                                    synchronized (gr) {
                                        gr.fillCoords(Grid.this.lowerXLimit, Grid.this.upperXLimit, Grid.this.xStep, Grid.this.yStep, Grid.this.DRG);
                                    }
                                }
                            }
                        }

                        String direction = (lowerVisibleX <= lowerXLimit) ? "LEFT" : "RIGHT";
                        System.out.println("Now scrolled to extreme " + direction + ", rebuilding graph values...");
                    }

                } catch (Exception ex) {
                    ex.printStackTrace();
                } finally {
                    // ALWAYS release the locks here, no matter what happens in the try block
                    lockedForMathCalc = false;

                    // Request a repaint on the EDT
                    SwingUtilities.invokeLater(() -> {
                        component.repaint();
// Only trigger if an update was requested while we were locked
                        if (updateHappenedDuringLock) {
                            setRefreshingIndices(true);
                        }
                    });
                }
            }
        });

        t.setPriority(Thread.NORM_PRIORITY); // Avoid starving the UI thread
        t.start();
    }

    
    public boolean isShowGridLines() {
        return showGridLines;
    }

    public void setShowGridLines(boolean showGridLines) {
        this.showGridLines = showGridLines;
    }

    /**
     *
     * @param lowerXLimit The lowest value up to which x should be plotted This
     * value is re-computed in the method to ensure that it is always a unit
     * number of {@link Grid#xStep}s away from the x coordinate(cartesian
     * coordinate, not screen coordinates) of the left side of the graph
     */
    public void setLowerXLimit(double lowerXLimit) {
        this.lowerXLimit = (abs(lowerXLimit) != Double.POSITIVE_INFINITY) ? lowerXLimit : -100;
        this.dataSharer.xLower = this.lowerXLimit;
        setRefreshingIndices(true);
    }

    public double getUpperXLimit() {
        return upperXLimit;
    }

    /**
     *
     * @param upperXLimit The highest value up to which x should be plotted This
     * value is re-computed in the method to ensure that it is always a unit
     * number of {@link Grid#xStep}s away from the x coordinate(cartesian
     * coordinate, not screen coordinates) of the left side of the graph
     */
    public void setUpperXLimit(double upperXLimit) {
        this.upperXLimit = (abs(upperXLimit) != Double.POSITIVE_INFINITY) ? upperXLimit : 100;
        this.dataSharer.xUpper = this.upperXLimit;
        setRefreshingIndices(true);
    }

    public double getxStep() {
        return xStep;
    }

    public void setxStep(double xStep) {
        this.xStep = Math.abs(xStep);
        this.dataSharer.xStep = this.xStep;
    }

    public void setyStep(double yStep) {
        this.yStep = Math.abs(yStep);
        this.dataSharer.yStep = this.yStep;
    }

    public double getyStep() {
        return yStep;
    }

    public void setLabelAxis(boolean labelAxis) {
        this.labelAxis = labelAxis;
    }

    public boolean isLabelAxis() {
        return labelAxis;
    }

    public void setTickColor(GraphColor tickColor) {
        this.tickColor = tickColor;
    }

    public GraphColor getTickColor() {
        return tickColor;
    }

    /**
     *
     * @param function A command string consisting of functions (geometric and
     * algebraic) to be added to the graph. The functions are separated by a
     * semicolon(;) Calling this method will ensure that the functions contained
     * in the command will be added to the ones already plotted on the graph. An
     * example of the format of function is:
     * [-200,200,300,-200:][1,3,-2,1:];y=@(x)3x+1;y1=@(x)sin(3*x-1)
     */
    public void addFunction(String function) {
        try {
            if (gridExpressionParser != null) {
                GridExpressionParser gep = new GridExpressionParser(function, dataSharer);
                gridExpressionParser.addFunctions(gep);
            }//end if
            else {
                gridExpressionParser = new GridExpressionParser(function, dataSharer);
            }//end else
        }//end try
        catch (NullPointerException exception) {

        }//end catch
    }//end method.

    /**
     *
     * @param function A command string consisting of functions (geometric and
     * algebraic) to be added to the graph. The functions are separated by a
     * semicolon(;) Calling this method will ensure that the functions contained
     * in the command alone will be the functions to be plotted on the graph.
     */
    public void setFunction(String function) {
        try {
            if (gridExpressionParser != null) {
                gridExpressionParser.setInput(function, dataSharer);
            }//end if
            else {
                gridExpressionParser = new GridExpressionParser(function, dataSharer);
            }//end else
        }//end try
        catch (NullPointerException exception) {

        }//end catch
    }

    /**
     * Draws the 2 major axes, the x and the y and labels them.
     *
     * @param g The Graphics object used to draw.
     */
    public void drawMajorAxes(DrawingContext g) {

        Point compLoc = new Point(0, 0);//component.getLocation();
        Dimension compSize = new Dimension(component.getWidth(), component.getHeight());//component.getSize();
        int xLocOfRightSideOfComponent = (int) compLoc.x + compSize.width;
        int yLocOfBottomSideOfComponent = (int) compLoc.y + compSize.height;

        int xLocOfLeftSideOfComponent = (int) compLoc.x;
        int yLocOfTopSideOfComponent = (int) compLoc.y;

        g.setFont(font);
        drawXAxes:
        {
            g.setColor(majorAxesColor);
            g.drawLine(0, (int) locationOfOrigin.y, xLocOfRightSideOfComponent, (int) locationOfOrigin.y);
            g.drawLine(0, (int) locationOfOrigin.y + 1, xLocOfRightSideOfComponent, (int) locationOfOrigin.y + 1);//make the line thick

            drawTicksOnXaxis:
            {
                int x = (int) locationOfOrigin.x;

                int countTicks = 0;

//draw to the right of the y axis
                while (x < xLocOfRightSideOfComponent) {

                    //indicate the name of the horizontal axis.
                    if (labelAxis) {
                        g.drawText(horizontalAxisLabel.getName() + "→", compSize.width - 80, (int) locationOfOrigin.y + majorTickLength + 40);
                    }

//draw the major tick lengths
                    if (countTicks == 0) {
                        g.setColor(tickColor);
                        g.drawLine(x, (int) locationOfOrigin.y, x, (int) locationOfOrigin.y + majorTickLength);

                        String numLabel = String.valueOf(convertScreenPointToGraphCoords(x, (int) locationOfOrigin.y)[0]);
                        if (!numLabel.equals("0.0")) {
                            if (labelAxis) {
                                g.drawText(numLabel, x, (int) locationOfOrigin.y + majorTickLength + 10);
                            }
                        }
                    }//end if
                    else {
                        g.setColor(tickColor);
                        g.drawLine(x, (int) locationOfOrigin.y, x, (int) locationOfOrigin.y + minorTickLength);
                    }//end if
                    x += (gridSize.width * 2);
                    ++countTicks;
                    if (countTicks == 5) {
                        countTicks = 0;
                    }
                }//end while

                countTicks = 0;
                x = (int) locationOfOrigin.x;

//now draw to the left of the y axis
                while (x + compLoc.x > xLocOfLeftSideOfComponent) {
//draw the major tick lengths
                    if (countTicks == 0) {
                        g.setColor(tickColor);
                        g.drawLine(x, (int) locationOfOrigin.y, x, (int) locationOfOrigin.y + majorTickLength);
                        String numLabel = String.valueOf(convertScreenPointToGraphCoords(x, (int) locationOfOrigin.y)[0]);
                        if (!numLabel.equals("0.0")) {
                            if (labelAxis) {
                                g.drawText(numLabel, x, (int) locationOfOrigin.y + majorTickLength + 10);
                            }
                        }
                    }//end if
                    else {
                        g.setColor(tickColor);
                        g.drawLine(x, (int) locationOfOrigin.y, x, (int) locationOfOrigin.y + minorTickLength);
                    }//end if
                    x -= (gridSize.width * 2);
                    ++countTicks;
                    if (countTicks == 5) {
                        countTicks = 0;
                    }
                }//end while

            }//end inner label

        }//end outer label

        drawYAxes:
        {

            g.setColor(majorAxesColor);
            g.drawLine((int) locationOfOrigin.x, 0, (int) locationOfOrigin.x, yLocOfBottomSideOfComponent);
            g.drawLine((int) locationOfOrigin.x + 1, 0, (int) locationOfOrigin.x + 1, yLocOfBottomSideOfComponent);//make the line thick

            int y = (int) (locationOfOrigin.y + compLoc.y);

            int countTicks = 0;
//draw to the right of the y axis
            while (y > yLocOfTopSideOfComponent) {
                //label the vertical axis.
                if (labelAxis) {
                    g.drawText(verticalAxisLabel.getName() + "↑", (int) locationOfOrigin.x + majorTickLength + 50, yLocOfTopSideOfComponent + 50);
                }

//draw the major tick lengths
                if (countTicks == 0) {
                    g.setColor(tickColor);
                    g.drawLine((int) locationOfOrigin.x, y, (int) locationOfOrigin.x + majorTickLength, y);
                    String numLabel = String.valueOf(convertScreenPointToGraphCoords((int) locationOfOrigin.x, y)[1]);
                    if (!numLabel.equals("0.0")) {
                        if (labelAxis) {
                            g.drawText(numLabel, (int) locationOfOrigin.x + majorTickLength + 10, y);//thicken the ticks
                        }
                    }
                }//end if
                else {
                    g.setColor(tickColor);
                    g.drawLine((int) locationOfOrigin.x, y, (int) locationOfOrigin.x + minorTickLength, y);
                }//end if
                y -= (gridSize.height * 2);
                ++countTicks;
                if (countTicks == 5) {
                    countTicks = 0;
                }
            }//end while

            countTicks = 0;
            y = (int) (locationOfOrigin.y + compLoc.y);
//now draw to the left of the y axis
            while (y < yLocOfBottomSideOfComponent) {

//draw the major tick lengths
                if (countTicks == 0) {
                    g.setColor(tickColor);
                    g.drawLine((int) locationOfOrigin.x, y, (int) locationOfOrigin.x + majorTickLength, y);
                    String numLabel = String.valueOf(convertScreenPointToGraphCoords((int) locationOfOrigin.x, y)[1]);
                    if (!numLabel.equals("0.0") && labelAxis) {
                        g.drawText(numLabel, (int) locationOfOrigin.x + majorTickLength + 10, y);
                    }
                }//end if
                else {
                    g.setColor(tickColor);
                    g.drawLine((int) locationOfOrigin.x, y, (int) locationOfOrigin.x + minorTickLength, y);

                }//end if
                y += (gridSize.height * 2);
                ++countTicks;
                if (countTicks == 5) {
                    countTicks = 0;
                }

            }//end while

        }
        // art.fillOval(g, locationOfOrigin.x, locationOfOrigin.y, gridSize.width, gridSize.height + 3);

        // component.drawBitmap(g.getBitmap(),0,0,null);
    }//end method drawMajorAxes

    /**
     * draws the grid
     *
     * @param g the Graphics object used to draw.
     */
    public void draw(DrawingContext g) {

        try {
            g.setFont(font);
            
            drawHorizontalLines:
            {
                if (showGridLines) {

                    g.setColor(gridColor);
                    int y = 0;

                    while (y < component.getHeight()) {
                        g.drawLine(0, y, component.getWidth(), y);
                        y += gridSize.height;
                    }

                }
            }

            drawVerticalLines:
            {
                int x = 0;
                if (showGridLines) {
                    while (x < component.getWidth()) {
                        g.drawLine(x, 0, x, component.getHeight());
                        x += this.gridSize.width;
                    }
                }

            }

            g.setColor(this.plotColor);
            int x = 0;/* the left side of the screen is the first possible x location. Get this as an xcoordinate in the graph's cooordinate space*/


            if (this.gridExpressionParser != null && !this.gridExpressionParser.getGraphElements().isEmpty()) {

                double graphXPosOfLeftSideOfScreen = this.convertScreenPointToGraphCoords(x - this.gridSize.width, 0)[0];
                double graphXPosOfRightSideOfScreen = this.convertScreenPointToGraphCoords(this.component.getWidth(), 0)[0];

                GraphElement graphElementOne = this.gridExpressionParser.getGraphElements().get(0);
                int count = graphElementOne.getHorizontalCoordinates().length;

                double firstXCoordinateInArray = graphElementOne.getHorizontalCoordinates()[0];
                double lastXCoordinateInArray = graphElementOne.getHorizontalCoordinates()[count - 1];

                double startIndex = graphXPosOfLeftSideOfScreen == firstXCoordinateInArray ? 0
                        : GraphElement.binarySearch(graphXPosOfLeftSideOfScreen, graphElementOne.getHorizontalCoordinates());

                double endIndex = graphXPosOfRightSideOfScreen == lastXCoordinateInArray ? count - 1
                        : GraphElement.binarySearch(graphXPosOfRightSideOfScreen, graphElementOne.getHorizontalCoordinates());

                synchronized (gridExpressionParser.lock) {
                    for (GraphElement gr : this.gridExpressionParser.getGraphElements()) {
                        synchronized (gr) {
                            if (gr.getGraphType() == GraphType.FunctionPlot) {
                                int beginIndex = (int) startIndex;
                                double xx = 0;
                                while (beginIndex <= endIndex - 1) {

                                    double[] horCoords = gr.getHorizontalCoordinates();
                                    double[] verCoords = gr.getVerticalCoordinates();

                                    xx = horCoords[beginIndex];
                                    double yy = verCoords[beginIndex];

                                    double xx_next = horCoords[beginIndex + 1];
                                    double yy_next = verCoords[beginIndex + 1];

                                    int[] screenPoint = this.convertGraphPointToScreenCoords(xx, yy);
                                    int[] screenPoint1 = this.convertGraphPointToScreenCoords(xx_next, yy_next);

                                    g.drawLine(screenPoint[0], screenPoint[1], screenPoint1[0], screenPoint1[1]);

                                    beginIndex += 1;
                                }

                            } else {
                                int i = 0;
                                while (i < gr.getHorizontalCoordinates().length - 1) { 
                                    double[] horCoords = gr.getHorizontalCoordinates();
                                    double[] verCoords = gr.getVerticalCoordinates();
                                    
                                    double xx = horCoords[i];
                                    double yy = verCoords[i];
                                    double xx_next = horCoords[i + 1];
                                    double yy_next = verCoords[i + 1];
                                    int[] screenPoint = this.convertGraphPointToScreenCoords(xx, yy);
                                    int[] screenPoint1 = this.convertGraphPointToScreenCoords(xx_next, yy_next);
                                    g.drawLine(screenPoint[0], screenPoint[1], screenPoint1[0], screenPoint1[1]);
                                    i += 1;
                                }
                            }
                        }
                    }
                }
            }

            g.setColor(GraphColor.black);
            g.fillOval(((int) locationOfOrigin.x) - 5, ((int) locationOfOrigin.y) - 5, 10, 10);
            drawMajorAxes(g);

        }//end try
        catch (NumberFormatException numErr) {
            gridExpressionParser.setCanPlot(false);
        } catch (NullPointerException nullErr) {
            nullErr.printStackTrace();
        }

    }//end method

    /**
     *
     * @param gridDistance The horizontal distance along the grid.
     * @return the equivalent horizontal distance in user terms.
     */
    public double convertGridSizeToUserDistanceAlongX(int gridDistance) {
        return gridDistance * xStep / ((double) gridSize.width);
    }

    /**
     *
     * @param gridDistance The vertical distance along the grid.
     * @return the vertical distance in user terms.
     */
    public double convertGridSizeToUserDistanceAlongY(int gridDistance) {
        return gridDistance * yStep / ((double) gridSize.height);
    }

    /**
     *
     * @param userX The horizontal distance along the grid in user terms.
     * @return the equivalent horizontal distance in screen terms.
     */
    public long convertUserDistanceAlongX_ToGridSize(double userX) {
        String value = String.valueOf(round((userX * gridSize.width) / (xStep)));
        int startIndex = value.length();
        try {
            if (value.substring(startIndex - 2).equals(".0")) {
                value = delete(value, startIndex - 2, startIndex);
            }
        }//end try
        catch (IndexOutOfBoundsException indexErr) {

        }
        return Long.parseLong(value);
    }

    /**
     *
     * @param userY The vertical distance along the grid in user terms.
     * @return the equivalent vertical distance in screen terms .
     */
    public long convertUserDistanceAlongY_ToGridSize(double userY) {
        String value = String.valueOf(round((userY * gridSize.height) / (yStep)));
        int startIndex = value.length();
        try {
            if (value.substring(startIndex - 2).equals(".0")) {
                value = delete(value, startIndex - 2, startIndex);
            }
        }//end try
        catch (IndexOutOfBoundsException indexErr) {

        }

        return Long.parseLong(value);
    }

    /**
     * Converts a point on the screen to its equivalent point in mathematics
     * relative to the specified origin. It is useful basically in labeling the
     * axes during its construction.
     *
     * @param screenPoint The point on the screen.
     * @return the point on the graph.
     */
    public com.github.gbenroscience.math.Point convertScreenPointToGraphPointzaa(Point screenPoint) {
        double xGraph = convertGridSizeToUserDistanceAlongX((int) (screenPoint.x - locationOfOrigin.x));
        double yGraph = convertGridSizeToUserDistanceAlongY((int) (locationOfOrigin.y - screenPoint.y));
        return new com.github.gbenroscience.math.Point(xGraph, yGraph);
    }

    /**
     * This method does basically what
     * {@link Grid#convertScreenPointToGraphCoords(int, int)} does. But it
     * returns its result as an array of size 2. Converts the x and y
     * coordinates of a point on the screen to its equivalent point in
     * mathematics relative to the specified origin. It is useful basically in
     * labeling the axes during its construction.
     *
     * @param screenX The x coordinate of the point on the screen.
     * @param screenY The y coordinate of the point on the screen.
     *
     * @return an array of size 2 having the graph equivalent of the screen's x
     * coordinate in index 0 and the graph equivalent of the screen's y
     * coordinate in index 1
     */
    public double[] convertScreenPointToGraphCoords(int screenX, int screenY) {
        double xGraph = convertGridSizeToUserDistanceAlongX(screenX - (int) locationOfOrigin.x);
        double yGraph = convertGridSizeToUserDistanceAlongY((int) locationOfOrigin.y - screenY);
        return new double[]{xGraph, yGraph};
    }

    /**
     * Say the user is about to identify plot Point p = [2,4] on the screen, He
     * passes it in to the draw method as convertGraphPointToScreenPoint(p). and
     * gets his plot.This method takes care of all conversions from the math
     * coordinates to the screen coordinates.
     *
     * @param userPoint The point on the graph to be drawn on the screen.
     * @return the screen equivalent of the point.
     */
    public Point convertGraphPointToScreenPoint(com.github.gbenroscience.math.Point userPoint) {
        double xScreen = locationOfOrigin.x + convertUserDistanceAlongX_ToGridSize(userPoint.x);
        double yScreen = locationOfOrigin.y - convertUserDistanceAlongY_ToGridSize(userPoint.y);//in contrast to screen coordinates math coordinates go up as y increases.
        return new Point((int) round(xScreen), (int) round(yScreen));
    }

    /**
     * This does the same thing as
     * {@link Grid#convertGraphPointToScreenPoint(math.Point)} but it returns
     * the result in an array of size two instead.
     *
     * @param graphX The y coordinate of the point to be drawn on the screen.
     * @param graphY The y coordinate of the point to be drawn on the screen.
     * @return an array containing the screen equivalent of coordinate x in
     * index 0 and the the screen equivalent of coordinate y in index 1.
     */
    public int[] convertGraphPointToScreenCoords(double graphX, double graphY) {
        double xScreen = locationOfOrigin.x + convertUserDistanceAlongX_ToGridSize(graphX);
        double yScreen = locationOfOrigin.y - convertUserDistanceAlongY_ToGridSize(graphY);//in contrast to screen coordinates math coordinates go up as y increases.
        return new int[]{(int) round(xScreen), (int) round(yScreen)};
    }

    /**
     * *
     *
     */
    public final void computeXMinBoundPossibleOnScreen() {
        /*  int width = component.getWidth();
        int gridWidth = gridSize.width;
        int originX = locationOfOrigin.getX();

        double xMinBound = -(1.0 * locationOfOrigin.getX() ) / xStep;

        return xMinBound;
         */
        this.lowerVisibleX = convertScreenPointToGraphCoords(0, 0)[0];
    }

    /**
     *
     */
    public final void computeXMaxBoundPossibleOnScreen() {
        /* int width = component.getWidth();
        int gridWidth = gridSize.width;
        int originX = locationOfOrigin.getX();
        double widthOfNegativeXAxis = width - originX;

        double xMaxBound = widthOfNegativeXAxis / xStep;

        return xMaxBound;
         */
        this.upperVisibleX = convertScreenPointToGraphCoords(component.getWidth(), 0)[0];
    }

    /**
     *
     */
    public final void computeYMinBoundPossibleOnScreen() {
        /*   int width = component.getHeight();
        int gridHeight = gridSize.height;
        int originX = locationOfOrigin.getY();

        double yMinBound = -(1.0 * locationOfOrigin.getY() ) / yStep;
         */
        this.lowerVisibleY = convertScreenPointToGraphCoords(0, component.getHeight())[1];
    }

    /**
     *
     */
    public final void computeYMaxBoundPossibleOnScreen() {
        /* int width = component.getHeight();
        int gridHeight = gridSize.height;
        int originX = locationOfOrigin.getY();
        double widthOfNegativeXAxis = width - originX;

        double yMaxBound = widthOfNegativeXAxis / yStep;

        return yMaxBound;*/

        this.upperVisibleY = convertScreenPointToGraphCoords(0, 0)[1];
    }

    /**
     * Generates automatically the numeric drawing parameters for the function.
     *
     *
     */
    public void generateAutomaticScale() {
        if (isAutoScaleOn()) {
            setxStep(0.1);
            setyStep(0.1);
        }//end if isAutoscaleOn

    }//end method

}//end class

