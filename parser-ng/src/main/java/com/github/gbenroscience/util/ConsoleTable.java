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
package com.github.gbenroscience.util;

import com.github.gbenroscience.parser.STRING;

/**
 *
 * @author GBEMIRO
 */ 
public class ConsoleTable {
    private final String title;
    private final String[] headers;
    private final String[][] data;
    private int[] columnWidths;

    public ConsoleTable(String title, String[] headers, String[][] data) {
        this.title = title;
        this.headers = headers;
        this.data = data;
        calculateWidths();
    }

    private void calculateWidths() {
        columnWidths = new int[headers.length];
        for (int i = 0; i < headers.length; i++) {
            columnWidths[i] = headers[i].length();
            for (String[] row : data) {
                if (row[i] != null && row[i].length() > columnWidths[i]) {
                    columnWidths[i] = row[i].length();
                }
            }
        }
    }

    private String divider() {
        StringBuilder sb = new StringBuilder("+");
        for (int width : columnWidths) {
            sb.append(STRING.repeating("-", width + 2)).append("+");
        }
        return sb.toString();
    }

    private String row(String[] row) {
        StringBuilder sb = new StringBuilder("|");
        for (int i = 0; i < row.length; i++) {
            String format = " %-" + columnWidths[i] + "s |";
            sb.append(String.format(format, (row[i] == null ? "null" : row[i])));
        }
        return sb.toString();
    }

    /**
     * Lays the table out as text: the title, then the header and the rows between
     * dividers. Lines are separated by {@code \n}, with no trailing newline. Use this
     * when the table has to go somewhere other than {@code System.out}, e.g. a GUI console.
     *
     * @return the table as text
     */
    public String render() {
        StringBuilder sb = new StringBuilder(title).append('\n');
        sb.append(divider()).append('\n');
        sb.append(row(headers)).append('\n');
        sb.append(divider()).append('\n');
        for (String[] r : data) {
            sb.append(row(r)).append('\n');
        }
        sb.append(divider());
        return sb.toString();
    }

    public void display() {
        System.out.println("\n" + render());
    }

    public static void main(String[] args) {
        // 1. Define Title
        String title = "RESULTS for [(sin(3)+cos(4-sin(2)))^(3*sin(4))]";

        // 2. Define Headers
        String[] headers = {"n", "Library1 ID", "Library2 ID", "Library3 ID", "Values(Lib1)", "Values(Lib2)", "Values(Lib3)"};

        // 3. Define 20 Rows of Data (2D Array)
        String[][] data = new String[20][7];
        int[] nValues = {1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 15000000, 20000000};

        for (int i = 0; i < 20; i++) {
            data[i][0] = String.valueOf(nValues[i]);
            data[i][1] = "ParserNG";
            data[i][2] = "JavaMEP";
            data[i][3] = "MathLib";
            data[i][4] = (i < 5) ? "NaN" : String.format("%.15f", Math.random());
            data[i][5] = (i < 5) ? "NaN" : String.format("%.15f", Math.random());
            data[i][6] = (i < 5) ? "NaN" : String.format("%.15f", Math.random());
        }

        // 4. Create and Display Table
        ConsoleTable table = new ConsoleTable(title, headers, data);
        table.display();
    }
}