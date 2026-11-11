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

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.management.ManagementFactory;

/**
 * Reads the JVM's per-thread allocation counter ({@code com.sun.management.ThreadMXBean#getThreadAllocatedBytes})
 * WITHOUT allocating or generating code while a measurement is running.
 * <p>
 * Why this exists: the tests used {@code Method.invoke} for every reading. On JDK 8 - 17 reflection is "inflated" on the
 * 16th invocation of a {@code Method}: the JVM generates a bytecode accessor class, which allocates roughly 11 KB on the
 * calling thread. Because the counter is shared by all tests in the JVM, that 16th call can land inside any measurement
 * window of an allocation-contract test, which then reports a fictitious "36 bytes per step" even though the code under
 * test allocates nothing (JDK 18+ reflection uses method handles and never inflates, which is why only JDK &lt; 21 CI
 * jobs failed and why the failing test depended on test order). A bound {@link MethodHandle} invoked with
 * {@code invokeExact} returns a primitive {@code long} (no boxing) and is fully warmed here, before the first reading.
 */
public final class AllocationMeter {

    private static final MethodHandle READER;

    static {
        MethodHandle mh = null;
        try {
            Object bean = ManagementFactory.getThreadMXBean();
            Class<?> sun = Class.forName("com.sun.management.ThreadMXBean");
            if (sun.isInstance(bean)) {
                mh = MethodHandles.lookup()
                        .findVirtual(sun, "getThreadAllocatedBytes", MethodType.methodType(long.class, long.class))
                        .bindTo(bean);
                // warm-up: resolve and compile every lambda form the measuring path uses, long before a window opens
                long id = Thread.currentThread().getId();
                long sink = 0;
                for (int i = 0; i < 200; i++) {
                    sink += (long) mh.invokeExact(id);
                }
                if (sink < 0) {
                    mh = null;
                }
            }
        } catch (Throwable t) {
            mh = null;
        }
        READER = mh;
    }

    private AllocationMeter() {
    }

    /** @return true if this JVM exposes per-thread allocation counters. */
    public static boolean available() {
        return READER != null;
    }

    /** @return bytes allocated so far by the thread with the given id, or -1 if the JVM cannot tell. */
    public static long allocatedBytes(long threadId) {
        if (READER == null) {
            return -1;
        }
        try {
            return (long) READER.invokeExact(threadId);
        } catch (Throwable t) {
            return -1;
        }
    }

    /** @return bytes allocated so far by the calling thread, or -1 if the JVM cannot tell. */
    public static long allocatedByCurrentThread() {
        return allocatedBytes(Thread.currentThread().getId());
    }
}
