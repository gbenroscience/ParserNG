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
package com.github.gbenroscience.math.graph.tools;

/**
 *
 * @author GBEMIRO
 */
public class GraphColor {

    public final int r;
    public final int g;
    public final int b;
    public final int a;
    
    
    
    /**
     * The color white.  In the default sRGB space.
     */
    public static final GraphColor white     = new GraphColor(255, 255, 255);

    /**
     * The color white.  In the default sRGB space.
     *
     */
    public static final GraphColor WHITE = white;

    /**
     * The color light gray.  In the default sRGB space.
     */
    public static final GraphColor lightGray = new GraphColor(192, 192, 192);

    /**
     * The color light gray.  In the default sRGB space.
     *
     */
    public static final GraphColor LIGHT_GRAY = lightGray;

    /**
     * The color gray.  In the default sRGB space.
     */
    public static final GraphColor gray      = new GraphColor(128, 128, 128);

    /**
     * The color gray.  In the default sRGB space.
     *
     */
    public static final GraphColor GRAY = gray;

    /**
     * The color dark gray.  In the default sRGB space.
     */
    public static final GraphColor darkGray  = new GraphColor(64, 64, 64);

    /**
     * The color dark gray.  In the default sRGB space.
     *
     */
    public static final GraphColor DARK_GRAY = darkGray;

    /**
     * The color black.  In the default sRGB space.
     */
    public static final GraphColor black     = new GraphColor(0, 0, 0);

    /**
     * The color black.  In the default sRGB space.
     *
     */
    public static final GraphColor BLACK = black;

    /**
     * The color red.  In the default sRGB space.
     */
    public static final GraphColor red       = new GraphColor(255, 0, 0);

    /**
     * The color red.  In the default sRGB space.
     *
     */
    public static final GraphColor RED = red;

    /**
     * The color pink.  In the default sRGB space.
     */
    public static final GraphColor pink      = new GraphColor(255, 175, 175);

    /**
     * The color pink.  In the default sRGB space.
     *
     */
    public static final GraphColor PINK = pink;

    /**
     * The color orange.  In the default sRGB space.
     */
    public static final GraphColor orange    = new GraphColor(255, 200, 0);

    /**
     * The color orange.  In the default sRGB space.
     *
     */
    public static final GraphColor ORANGE = orange;

    /**
     * The color yellow.  In the default sRGB space.
     */
    public static final GraphColor yellow    = new GraphColor(255, 255, 0);

    /**
     * The color yellow.  In the default sRGB space.
     *
     */
    public static final GraphColor YELLOW = yellow;

    /**
     * The color green.  In the default sRGB space.
     */
    public static final GraphColor green     = new GraphColor(0, 255, 0);

    /**
     * The color green.  In the default sRGB space.
     *
     */
    public static final GraphColor GREEN = green;

    /**
     * The color magenta.  In the default sRGB space.
     */
    public static final GraphColor magenta   = new GraphColor(255, 0, 255);

    /**
     * The color magenta.  In the default sRGB space.
     *
     */
    public static final GraphColor MAGENTA = magenta;

    /**
     * The color cyan.  In the default sRGB space.
     */
    public static final GraphColor cyan      = new GraphColor(0, 255, 255);

    /**
     * The color cyan.  In the default sRGB space.
     *
     */
    public static final GraphColor CYAN = cyan;

    /**
     * The color blue.  In the default sRGB space.
     */
    public static final GraphColor blue      = new GraphColor(0, 0, 255);

    /**
     * The color blue.  In the default sRGB space.
     *
     */
    public static final GraphColor BLUE = blue;

    public GraphColor() {
        this(255,255,255,255);
    }

    public GraphColor(int red, int green, int blue) {
        this(red, green, blue, 255);
    }

    public GraphColor(int red, int green, int blue, int alpha) {
        this.r = red;
        this.g = green;
        this.b = blue;
        this.a = alpha;
    }

    public int getRedComponent() {
        return r;
    }

    

    public int getGreenComponent() {
        return g;
    }

    

    public int getBlueComponent() {
        return b;
    }

    

    public int getAlpha() {
        return a;
    }
 
    
    
    
    

}
