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
package com.github.gbenroscience.math.differentialcalculus.pde;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * Small reusable fork-join helper with STATIC partitioning and no per-dispatch
 * allocation. The calling thread acts as worker 0, so a pool of size 1 simply
 * runs the task inline and owns no threads.
 * <p>
 * Dispatch uses a volatile epoch counter and spin-then-park waiting; it never
 * creates task, future or lambda objects after construction.
 */
final class WorkerPool implements AutoCloseable {

    /** Work unit; implementations keep their parameters in fields set before {@link #run}. */
    interface Task {

        void run(int worker, int workers);
    }

    private static final int SPINS = 4000;
    private static final long PARK_NANOS = 20_000L;

    private final int size;
    private final Thread[] threads;
    private volatile int epoch;
    private volatile boolean shutdown;
    private volatile Task task;
    private volatile Thread owner;
    private final AtomicInteger done = new AtomicInteger();
    private volatile Throwable failure;

    WorkerPool(int size) {
        this.size = Math.max(1, size);
        this.threads = new Thread[this.size - 1];
        for (int i = 0; i < threads.length; i++) {
            final int id = i + 1;
            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    loop(id);
                }
            }, "pde-worker-" + id);
            t.setDaemon(true);
            threads[i] = t;
            t.start();
        }
    }

    int size() {
        return size;
    }

    private void loop(int id) {
        int seen = 0;
        while (true) {
            int spins = 0;
            int e;
            while ((e = epoch) == seen) {
                if (shutdown) {
                    return;
                }
                if (++spins > SPINS) {
                    LockSupport.parkNanos(PARK_NANOS);
                }
            }
            seen = e;
            if (shutdown) {
                return;
            }
            try {
                task.run(id, size);
            } catch (Throwable ex) {
                failure = ex;
            }
            if (done.incrementAndGet() == size - 1) {
                Thread o = owner;
                if (o != null) {
                    LockSupport.unpark(o);
                }
            }
        }
    }

    /** Runs {@code t} on all workers and returns when every worker has finished. */
    void run(Task t) {
        if (size == 1) {
            t.run(0, 1);
            return;
        }
        owner = Thread.currentThread();
        failure = null;
        task = t;
        done.set(0);
        epoch = epoch + 1;
        Throwable local = null;
        try {
            t.run(0, size);
        } catch (Throwable ex) {
            local = ex;
        }
        int spins = 0;
        while (done.get() < size - 1) {
            if (++spins > SPINS) {
                LockSupport.parkNanos(PARK_NANOS);
            }
        }
        Throwable f = local != null ? local : failure;
        if (f != null) {
            if (f instanceof RuntimeException) {
                throw (RuntimeException) f;
            }
            if (f instanceof Error) {
                throw (Error) f;
            }
            throw new PDEException("Worker failed", f);
        }
    }

    @Override
    public void close() {
        shutdown = true;
        epoch = epoch + 1;
        for (Thread t : threads) {
            LockSupport.unpark(t);
        }
    }
}
