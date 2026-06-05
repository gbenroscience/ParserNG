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
package com.github.gbenroscience.parser.benchmarks;

import com.github.gbenroscience.parser.MathExpression;
import com.github.gbenroscience.parser.turbo.QuickTime;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author GBEMIRO
 */
public class MeCopyVsConstructor {

    private static volatile boolean toggle = true;
    public static final ArrayList<MathExpression> TRASH = new ArrayList<>();
    private static AtomicInteger crossCount = new AtomicInteger();
    private static AtomicInteger deleteCount = new AtomicInteger();

    public static void main(String[] args) {
        final Thread t = new Thread(() -> {
            while (toggle) {
                if (TRASH.isEmpty()) {
                    continue;
                }
                MathExpression m = TRASH.remove(0);
                if (m != null) {
                    m.solve();
                    deleteCount.incrementAndGet();
                }
            }
            //drain the trash
            while (!TRASH.isEmpty()) {
                MathExpression m = TRASH.remove(0);
                if (m != null) {
                    m.solve();
                    deleteCount.incrementAndGet();
                }
            }
        });
        t.start();
        new Thread(() -> {
            final String expr = "(38*x+29*sin(x)^2+3*cos(x^2)^2)^2.82";
            QuickTime.benchmarkNano("MathExpression-Constructor", 1000, 10000, () -> {
                MathExpression m = new MathExpression(expr);
                TRASH.add(m);
            });
            crossCount.incrementAndGet();
            if (crossCount.get() == 3) {
                toggle = false;
                if (!t.isInterrupted()) {
                    t.interrupt();
                    System.out.println("INTERRUPT FIRES! ---By MathExpression-Constructor");
                }
            }
        }).start();
        new Thread(() -> {
            final String expr = "(38*x+29*sin(x)^2+3*cos(x^2)^2)^2.82";
            final MathExpression m = new MathExpression(expr);
            QuickTime.benchmarkNano("MathExpression-Copy", 1000, 10000, () -> {
                MathExpression me = m.copy();
                TRASH.add(me);
            });
            crossCount.incrementAndGet();
            if (crossCount.get() == 3) {
                toggle = false;
                if (!t.isInterrupted()) {
                    t.interrupt();
                    System.out.println("INTERRUPT FIRES! ---By MathExpression-Copy");
                }
            }
        }).start();

        new Thread(() -> {
            final String expr = "(38*x+29*sin(x)^2+3*cos(x^2)^2)^2.82";
            final MathExpression m = new MathExpression(expr);
            QuickTime.benchmarkNano("MathExpression-Clone", 1000, 10000, () -> {
                try {
                    MathExpression me = m.clone();
                    TRASH.add(me);
                } catch (CloneNotSupportedException ex) {
                    Logger.getLogger(MeCopyVsConstructor.class.getName()).log(Level.SEVERE, null, ex);
                }
            });
            crossCount.incrementAndGet();
            if (crossCount.get() == 3) {
                toggle = false;
                if (!t.isInterrupted()) {
                    t.interrupt();
                    System.out.println("INTERRUPT FIRES! ---By MathExpression-Clone");
                }
            }
        }).start();

        try {
            t.join();
            System.out.println("INTERRUPT DETECTED! Thread join broken!");
        } catch (InterruptedException i) {
            i.printStackTrace();
        }

        System.out.println("deleteCount: " + deleteCount.get());

    }

}
