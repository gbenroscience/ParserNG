# Graphing with ParserNG

**ParserNG** now enables easy graphing in any Java UI framework, including Swing, JavaFX, Android, and others.

The library provides `DrawingContext` and `AbstractView` interfaces that can be implemented to render graphs on the target UI component (e.g., `javax.swing.JPanel` in Swing, `android.view.View` in Android, or `javafx.scene.canvas.Canvas` in JavaFX).

Key packages:
- `com.github.gbenroscience.math.graph` – Core graphing classes
- `com.github.gbenroscience.math.graph.tools` – Supporting tools (colors, fonts, etc.)

The main class for rendering graphs is `Grid` in `com.github.gbenroscience.math.graph`. To use it, implement the `DrawingContext` interface and the `AbstractView` for your target framework and pass the implementation to the `Grid` instance.

## Table of Contents

- [Introduction](#introduction)
- [DrawingContext Interface](#drawingcontext-interface)
- [Framework Adapters](#framework-adapters)
  - [Swing Adapter](#swing-adapter)
  - [Android Adapter](#android-adapter)
  - [JavaFX Adapter](#javafx-adapter)
- [Example Usage: GraphPanel in Swing](#example-usage-graphpanel-in-swing)

## Introduction

**ParserNG** allows developers to draw mathematical graphs with ease across multiple Java UI frameworks. The core idea is abstraction: the graphing logic is framework-agnostic and delegates all drawing operations to a `DrawingContext` implementation specific to the target UI toolkit.

This makes it simple to render the same graph on Swing, JavaFX, Android, or any other framework that can provide basic 2D drawing primitives.

## DrawingContext Interface

The `DrawingContext` interface defines the drawing operations needed by the graphing library:
- Setting color and stroke width
- Drawing/filling ovals, rectangles, lines
- Setting font and drawing text
- Measuring text dimensions
- Getting scale factor

You must implement this interface for your chosen framework.

## Framework Adapters

### Swing Adapter

```java
package math.graph.alpha;

import com.github.gbenroscience.math.graph.DrawingContext;
import com.github.gbenroscience.math.graph.tools.FontStyle;
import com.github.gbenroscience.math.graph.tools.GraphColor;
import com.github.gbenroscience.math.graph.tools.GraphFont;
import com.github.gbenroscience.math.graph.tools.TextDimensions;

import java.awt.*;

import static com.github.gbenroscience.math.graph.tools.FontStyle.*;

public class SwingDrawingContext implements DrawingContext {
    private final Graphics2D g;
    private final float scale = 1.0f;

    public SwingDrawingContext(Graphics2D g) {
        this.g = g;
    }

    @Override
    public void setColor(GraphColor c) {
        g.setColor(new Color(c.r, c.g, c.b, c.a));
    }

    @Override
    public void setStrokeWidth(float w) {
        g.setStroke(new BasicStroke(w));
    }

    @Override
    public void drawOval(int x, int y, int width, int height) {
        g.drawOval(x, y, width, height);
    }

    @Override
    public void fillOval(int x, int y, int width, int height) {
        g.fillOval(x, y, width, height);
    }

    @Override
    public void setFont(GraphFont f) {
        int awtStyle;
        switch (f.getStyle()) {
            case BOLD -> awtStyle = Font.BOLD;
            case ITALIC -> awtStyle = Font.ITALIC;
            case BOLD_ITALIC -> awtStyle = Font.BOLD | Font.ITALIC;
            default -> awtStyle = Font.PLAIN;
        }
        g.setFont(new Font(f.getFamily(), awtStyle, (int) f.getSize()));
    }

    @Override
    public void drawLine(float x1, float y1, float x2, float y2) {
        g.drawLine((int) x1, (int) y1, (int) x2, (int) y2);
    }

    @Override
    public void drawRect(float x, float y, float w, float h) {
        g.drawRect((int) x, (int) y, (int) w, (int) h);
    }

    @Override
    public void fillRect(float x, float y, float w, float h) {
        g.fillRect((int) x, (int) y, (int) w, (int) h);
    }

    @Override
    public void drawText(String text, float x, float y) {
        g.drawString(text, (int) x, (int) y);
    }

    @Override
    public TextDimensions measureText(String text) {
        FontMetrics m = g.getFontMetrics(g.getFont());
        return new TextDimensions(m.stringWidth(text), m.getHeight());
    }

    @Override
    public float getScale() {
        return scale;
    }

    // Helper methods (optional)
    public GraphFont getGraphFont(Font f) {
        FontStyle fontStyle = switch (f.getStyle()) {
            case Font.BOLD -> FontStyle.BOLD;
            case Font.ITALIC -> FontStyle.ITALIC;
            case (Font.BOLD | Font.ITALIC) -> FontStyle.BOLD_ITALIC;
            default -> FontStyle.PLAIN;
        };
        return new GraphFont(f.getFamily(), fontStyle, f.getSize2D());
    }
}
```

### Android Adapter

```java
import android.graphics.*;
import com.github.gbenroscience.math.graph.DrawingContext;
import com.github.gbenroscience.math.graph.tools.FontStyle;
import com.github.gbenroscience.math.graph.tools.GraphColor;
import com.github.gbenroscience.math.graph.tools.GraphFont;
import com.github.gbenroscience.math.graph.tools.TextDimensions;

 package math.graph.gui.adapter;

import android.graphics.*;
import com.github.gbenroscience.math.graph.DrawingContext;
import com.github.gbenroscience.math.graph.tools.FontStyle;
import com.github.gbenroscience.math.graph.tools.GraphColor;
import com.github.gbenroscience.math.graph.tools.GraphFont;
import com.github.gbenroscience.math.graph.tools.TextDimensions;

public class AndroidDrawingContext implements DrawingContext {
    private final Canvas canvas;
    private final Paint paint = new Paint();
    private final float scale;
    // Reuse this object to avoid memory churn/GC overhead
    private final RectF rectBuffer = new RectF();

    public AndroidDrawingContext(Canvas canvas, float density) {
        this.canvas = canvas;
        this.scale = density; // pixels per dp
    }

    @Override
    public void setColor(GraphColor c) {
        paint.setARGB(c.a, c.r, c.g, c.b);
    }

    @Override
    public void setStrokeWidth(float w) {
        paint.setStrokeWidth(w);
    }


    @Override
    public void setFont(GraphFont f) {

        int face = Typeface.NORMAL;
        switch (f.getStyle()){
            case PLAIN:
                face = Typeface.NORMAL;
                break;
            case ITALIC:
                face = Typeface.ITALIC;
                break;
            case BOLD:
                face = Typeface.BOLD;
                break;
            case BOLD_ITALIC:
                face = Typeface.BOLD_ITALIC;
                break;
            default:
                break;
        }

        paint.setTypeface(Typeface.create(f.getFamily(), face));
        paint.setTextSize(f.getSize() * scale);
    }
/// ////////////////////////////////////////////////////

@Override
public void drawLine(float x1, float y1, float x2, float y2) {
    canvas.drawLine(x1 * scale, y1 * scale, x2 * scale, y2 * scale, paint);
}

    @Override
    public void drawRect(float x, float y, float w, float h) {
        paint.setStyle(Paint.Style.STROKE);
        canvas.drawRect(x * scale, y * scale, (x + w) * scale, (y + h) * scale, paint);
    }

    @Override
    public void fillRect(float x, float y, float w, float h) {
        paint.setStyle(Paint.Style.FILL);
        canvas.drawRect(x * scale, y * scale, (x + w) * scale, (y + h) * scale, paint);
    }

    @Override
    public void drawOval(int x, int y, int width, int height) {
        paint.setStyle(Paint.Style.STROKE);
        updateRectBuffer(x, y, width, height); // already applies scale
        canvas.drawOval(rectBuffer, paint);
    }

    @Override
    public void fillOval(int x, int y, int width, int height) {
        paint.setStyle(Paint.Style.FILL);
        updateRectBuffer(x, y, width, height); // already applies scale
        canvas.drawOval(rectBuffer, paint);
    }

    @Override
    public void drawText(String text, float x, float y) {
        float offsetY = Math.abs(paint.ascent());
        canvas.drawText(text, x * scale, (y * scale) + offsetY, paint);
    }




    //////////////////////////////






    private void updateRectBuffer(float x, float y, float w, float h) {
        // Replaces 'new RectF()' to keep the heap clean
        rectBuffer.set(
                x * scale,
                y * scale,
                (x + w) * scale,
                (y + h) * scale
        );
    }


    @Override
    public TextDimensions measureText(String text) {
        float pixelWidth = paint.measureText(text);
        Paint.FontMetrics m = paint.getFontMetrics();

        // Logical height is the distance from the very top to the very bottom
        float pixelHeight = m.descent - m.ascent;

        return new TextDimensions(pixelWidth / scale, pixelHeight / scale);
    }

    @Override
    public float getScale() {
        return scale;
    }
}
```

### JavaFX Adapter

```java
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.*;
import com.github.gbenroscience.math.graph.DrawingContext;
import com.github.gbenroscience.math.graph.tools.FontStyle;
import com.github.gbenroscience.math.graph.tools.GraphColor;
import com.github.gbenroscience.math.graph.tools.GraphFont;
import com.github.gbenroscience.math.graph.tools.TextDimensions;

class JavaFXDrawingContext implements DrawingContext {
    private final GraphicsContext gc;
    private final float scale = 1.0f;

    public JavaFXDrawingContext(GraphicsContext gc) {
        this.gc = gc;
    }

    @Override
    public void setColor(GraphColor c) {
        Color fxColor = Color.rgb(c.r, c.g, c.b, c.a / 255.0);
        gc.setStroke(fxColor);
        gc.setFill(fxColor);
    }

    @Override
    public void setStrokeWidth(float w) {
        gc.setLineWidth(w);
    }

    @Override
    public void drawOval(int x, int y, int width, int height) {
        gc.strokeOval(x, y, width, height);
    }

    @Override
    public void fillOval(int x, int y, int width, int height) {
        gc.fillOval(x, y, width, height);
    }

    @Override
    public void setFont(GraphFont f) {
        FontWeight weight = FontWeight.NORMAL;
        FontPosture posture = FontPosture.REGULAR;

        if (f.getStyle() == FontStyle.BOLD) weight = FontWeight.BOLD;
        if (f.getStyle() == FontStyle.ITALIC) posture = FontPosture.ITALIC;
        if (f.getStyle() == FontStyle.BOLD_ITALIC) {
            weight = FontWeight.BOLD;
            posture = FontPosture.ITALIC;
        }

        gc.setFont(Font.font(f.getFamily(), weight, posture, f.getSize()));
    }

    @Override
    public void drawLine(float x1, float y1, float x2, float y2) {
        gc.strokeLine(x1, y1, x2, y2);
    }

    @Override
    public void drawRect(float x, float y, float w, float h) {
        gc.strokeRect(x, y, w, h);
    }

    @Override
    public void fillRect(float x, float y, float w, float h) {
        gc.fillRect(x, y, w, h);
    }

    @Override
    public void drawText(String text, float x, float y) {
        gc.fillText(text, x, y);
    }

    @Override
    public TextDimensions measureText(String text) {
        Text helper = new Text(text);
        helper.setFont(gc.getFont());
        return new TextDimensions(
            (float) helper.getLayoutBounds().getWidth(),
            (float) helper.getLayoutBounds().getHeight()
        );
    }

    @Override
    public float getScale() {
        return scale;
    }
}
```


### AbstractView

The `com.github.gbenroscience.math.graph.AbstractView` is a generic way by which **ParserNG** defines the  Views that it draws on.
For `javax.swing` it could be a `JPanel`, for Android it may be a `View` or a `SurfaceView` or even an `ImageView` etc.

 
#### Usage of AbstractView
Make your view implement `AbstracView` and then pass the instance of your view to the `Grid` constructor as in the `GraphPanel` example below.

## Example Usage: GraphPanel in Swing

Below is a complete example of a `JPanel` subclass that uses the `Grid` class with the Swing adapter to display an interactive graph:

```java
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.awt.print.*;
import javax.swing.Timer;
import com.github.gbenroscience.math.Point;
import com.github.gbenroscience.math.graph.Grid;
import com.github.gbenroscience.math.graph.tools.FontStyle;
import com.github.gbenroscience.math.graph.tools.GraphColor;
import com.github.gbenroscience.math.graph.tools.GraphFont;

public class GraphPanel extends javax.swing.JPanel implements Printable, AbstractView {
    private Point startCoords = new Point();
    private Dimension shiftCoords = new Dimension();

    private Grid grid;
    private String function = "p(x)=3*x^2";

    private int gridSize = 8;
    private boolean showGridLines = true;
    private boolean labelAxis = true;

    private GraphColor gridColor = GraphColor.gray;
    private GraphColor majorAxesColor = GraphColor.orange;
    private GraphColor tickColor = GraphColor.pink;
    private GraphColor plotColor = GraphColor.black;

    private int majorTickLength = 8;
    private int minorTickLength = 4;

    private double lowerXLimit = -200;
    private double upperXLimit = 200;
    private double xStep = 0.1;
    private double yStep = 0.1;

    private GraphFont font = new GraphFont("Times New Roman", FontStyle.BOLD, 14);
    private Point locationOfOrigin = new Point();

    private SwingDrawingContext context;
    private boolean reloadGraphics = false;

    public GraphPanel() {
        initComponents();

        grid = new Grid(function, showGridLines, labelAxis, gridSize, gridColor,
                majorAxesColor, tickColor, plotColor, majorTickLength,
                minorTickLength, lowerXLimit, upperXLimit, xStep, yStep, font, this);
    }

    // ... (mouse listeners, tooltip, dragging for panning, etc.)

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (context == null || reloadGraphics) {
            context = new SwingDrawingContext((Graphics2D) g);
            reloadGraphics = true;
        }
        grid.draw(context);
    }

    // Setter methods for customizing the graph (function, colors, limits, etc.)
    // Each calls repaint() after updating the Grid

    // Example setters omitted for brevity – see full original code for details

    @Override
    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        if (pageIndex > 0) return Printable.NO_SUCH_PAGE;

        BufferedImage img = createSwingObjectImage(this);
        Graphics2D g2d = (Graphics2D) graphics;
        g2d.drawImage(img, 0, 0, getWidth(), getHeight(), this);
        return Printable.PAGE_EXISTS;
    }

    private BufferedImage createSwingObjectImage(JComponent obj) {
        int w = obj.getWidth() > 0 ? obj.getWidth() : obj.getPreferredSize().width;
        int h = obj.getHeight() > 0 ? obj.getHeight() : obj.getPreferredSize().height;
        if (w <= 0 || h <= 0) { w = 1; h = 1; }

        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = img.createGraphics();
        obj.setSize(w, h);
        obj.doLayout();
        obj.paint(g2d);
        g2d.dispose();
        return img;
    }
}
```
<br><br>
This `GraphPanel` can be added to any Swing application and supports interactive features like panning, tooltips showing coordinates, customizable appearance, and printing.


We provide simple implementations of the AbstractView for various platforms

#### Swing - AbstractView implementation
```Java
import javax.swing.JComponent;

class SwingView implements AbstractView {
    private final JComponent component;

    public SwingView(JComponent component) {
        this.component = component;
    }

    @Override
    public int getWidth() {
        return component.getWidth();
    }

    @Override
    public int getHeight() {
        return component.getHeight();
    }

    @Override
    public void repaint() {
        component.repaint();
    }

    @Override
    public void repaint(int x, int y, int width, int height) {
        component.repaint(x, y, width, height);
    }
}
```

#### Android - AbstractView implementation
```Java
import android.view.View;

class AndroidView implements AbstractView {
    private final View view;

    public AndroidView(View view) {
        this.view = view;
    }

    @Override
    public int getWidth() {
        return view.getWidth();
    }

    @Override
    public int getHeight() {
        return view.getHeight();
    }

    @Override
    public void repaint() {
        view.invalidate();
    }

    @Override
    public void repaint(int x, int y, int width, int height) {
        view.invalidate(x, y, x + width, y + height);
    }
}
```


#### JavFX - AbstractView implementation
```Java
import javafx.scene.canvas.Canvas;

class JavaFXView implements AbstractView {
    private final Canvas canvas;

    public JavaFXView(Canvas canvas) {
        this.canvas = canvas;
    }

    @Override
    public int getWidth() {
        return (int) canvas.getWidth();
    }

    @Override
    public int getHeight() {
        return (int) canvas.getHeight();
    }

    @Override
    public void repaint() {
        canvas.getGraphicsContext2D().clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        // trigger redraw logic here if needed
    }

    @Override
    public void repaint(int x, int y, int width, int height) {
        canvas.getGraphicsContext2D().clearRect(x, y, width, height);
        // trigger the partial redraw logic here if needed
    }
}
```

### Grid
Since `Grid` is what you will add to your UI to make the graph, here is its API:

### Public API of `com.github.gbenroscience.math.graph.Grid`

#### Public Constants
```java
public static final int MIN_GRID_SIZE = 6;
public static final int MAX_GRID_SIZE = 200;
```

#### Public Constructor
```java
public Grid(
    String function,
    boolean showGridLines,
    boolean labelAxis,
    int gridSize,
    GraphColor gridColor,
    GraphColor majorAxesColor,
    GraphColor tickColor,
    GraphColor plotColor,
    int majorTickLength,
    int minorTickLength,
    double lowerXLimit,
    double upperXLimit,
    double xStep,
    double yStep,
    GraphFont font,
    AbstractView component
)
```
Initializes the grid with a function string (or empty for later addition), visual settings, scale limits, resolution steps, font, and the Swing panel to draw on.

#### Public Methods

**Component and Basic Settings**
```java
public void setComponent(AbstractView component)
public AbstractView getComponent()

public void setDRG(int DRG)                  // 0=degrees, 1=radians, 2=grads (default: 1)
public int getDRG()

public void setShowGridLines(boolean showGridLines)
public boolean isShowGridLines()

public void setLabelAxis(boolean labelAxis)
public boolean isLabelAxis()
```

**Axis Labels**
```java
public Variable getHorizontalAxisLabel()     // Default: "X"
public void setHorizontalAxisLabel(Variable horizontalAxisLabel)

public Variable getVerticalAxisLabel()       // Default: "Y"
public void setVerticalAxisLabel(Variable verticalAxisLabel)
```

**Colors**
```java
public GraphColor getGridColor()
public void setGridColor(GraphColor gridColor)

public GraphColor getMajorAxesColor()
public void setMajorAxesColor(GraphColor majorAxesColor)

public GraphColor getTickColor()
public void setTickColor(GraphColor tickColor)

public GraphColor getPlotColor()
public void setPlotColor(GraphColor plotColor)
```

**Grid Size**
```java
public Dimension getGridSize()
public void setGridSize(Dimension gridSize)
public void setGridSize(int width, int height)   // Clamped to [MIN_GRID_SIZE, MAX_GRID_SIZE]
```

**Ticks**
```java
public int getMajorTickLength()
public final void setMajorTickLength(int majorTickLength)

public int getMinorTickLength()
public final void setMinorTickLength(int minorTickLength)   // Ensures minor ≤ major/2
```

**Origin**
```java
public Point getLocationOfOrigin()
public void setLocationOfOrigin(Point locationOfOrigin)    // Triggers replot
```

**Scaling and Limits**
```java
public boolean isAutoScaleOn()
public void setAutoScaleOn(boolean autoScaleOn)            // Triggers replot

public Size getDefaultScale()
public void setDefaultScale(Size defaultScale)             // Triggers replot

public double getLowerXLimit()
public void setLowerXLimit(double lowerXLimit)             // Triggers replot

public double getUpperXLimit()
public void setUpperXLimit(double upperXLimit)             // Triggers replot

public double getxStep()
public void setxStep(double xStep)                         // Absolute value taken

public double getyStep()
public void setyStep(double yStep)                         // Absolute value taken

public double getLowerVisibleX()
public double getUpperVisibleX()
public double getLowerVisibleY()
public double getUpperVisibleY()
// These are computed view bounds (read-only)
```

**Font**
```java
public GraphFont getFont()
public void setFont(GraphFont font)
```

**Function Management**
```java
public void addFunction(String function)     // Appends new plots (semicolon-separated)
public void setFunction(String function)     // Replaces all plots

public GridExpressionParser getGridExpressionParser()
public void setGridExpressionParser(GridExpressionParser parser)
```

**Drawing and Computation**
```java
public void draw(DrawingContext g)                     // Main rendering method
public void drawMajorAxes(DrawingContext g)             // Axes, ticks, labels

public void generateAutomaticScale()                    // Sets xStep/yStep = 0.1 if auto-scale on

public void validateMaxIterations()                    // Caps iterations at ~500

public void replotOnLimitChange()                      // Internal, but public — triggers recompute on extreme scroll
```

**Coordinate Conversions**
```java
public double convertGridSizeToUserDistanceAlongX(int gridDistance)
public double convertGridSizeToUserDistanceAlongY(int gridDistance)

public long convertUserDistanceAlongX_ToGridSize(double userX)
public long convertUserDistanceAlongY_ToGridSize(double userY)

public com.github.gbenroscience.math.Point convertScreenPointToGraphPointzaa(Point screenPoint)
public double[] convertScreenPointToGraphCoords(int screenX, int screenY)

public Point convertGraphPointToScreenPoint(com.github.gbenroscience.math.Point userPoint)
public int[] convertGraphPointToScreenCoords(double graphX, double graphY)
```

**Bound Computation (View Limits)**
```java
public final void computeXMinBoundPossibleOnScreen()
public final void computeXMaxBoundPossibleOnScreen()
public final void computeYMinBoundPossibleOnScreen()
public final void computeYMaxBoundPossibleOnScreen()
```

#### Public Inner Class
```java
public static class GraphDataSharer {
    public double xStep = 1.0;
    public double yStep = 1.0;
    public int drg = 0;
    public double xLower = 0.0;
    public double xUpper = 0.0;
}
```
Used for sharing scale/DRG data with the parser.

This covers the entire public interface. All fields are private except the two static constants. Most mutating operations that affect scaling or origin trigger background replotting for responsiveness.
