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
import com.github.gbenroscience.math.graph.tools.TextDimensions;


/**
 *
 * @author GBEMIRO
 */  
public interface DrawingContext {
    void setColor(GraphColor c);
    void setStrokeWidth(float width);
    void setFont(GraphFont font); 
    void drawLine(float x1, float y1, float x2, float y2);
    void drawRect(float x, float y, float width, float height);
    void fillRect(float x, float y, float width, float height);
    void drawOval(int x, int y, int width, int height);
    void fillOval(int x, int y, int width, int height);
    void drawText(String text, float x, float y);
    TextDimensions measureText(String text);
    float getScale();
}
 