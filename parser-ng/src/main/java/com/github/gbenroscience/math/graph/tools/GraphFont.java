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
public class GraphFont {
    private String family;
    private FontStyle style;
    private float size;

    public GraphFont(String family, FontStyle style, float size) {
        this.family = family;
        this.style = style;
        this.size = size;
    }

    public String getFamily() {
        return family;
    }

    public void setFamily(String family) {
        this.family = family;
    }

    public FontStyle getStyle() {
        return style;
    }

    public void setStyle(FontStyle style) {
        this.style = style;
    }

    public float getSize() {
        return size;
    }

    public void setSize(float size) {
        this.size = size;
    }
    
    
    
    
}
