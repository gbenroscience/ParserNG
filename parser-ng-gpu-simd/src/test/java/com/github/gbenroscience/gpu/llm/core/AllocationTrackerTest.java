package com.github.gbenroscience.gpu.llm.core;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class AllocationTrackerTest {

    @Test
    void tracksBytesCategoriesAndPeak() {
        AllocationTracker t = new AllocationTracker();
        t.add(1, 100, "a");
        t.add(2, 50, "a");
        t.add(3, 7, "b");
        assertEquals(157, t.liveBytes());
        assertEquals(157, t.peakBytes());
        assertEquals(150, t.byCategory().get("a")[0]);
        assertEquals(2, t.byCategory().get("a")[1]);
        assertTrue(t.remove(2));
        assertFalse(t.remove(2), "double free is a no-op");
        assertEquals(107, t.liveBytes());
        assertEquals(157, t.peakBytes());
        assertEquals(100, t.byCategory().get("a")[0]);
        assertEquals(2, t.drain().length);
        assertEquals(0, t.liveBytes());
        assertEquals(0, t.size());
    }

    @Test
    void rejectsDuplicateHandles() {
        AllocationTracker t = new AllocationTracker();
        t.add(5, 1, "x");
        assertThrows(IllegalStateException.class, () -> t.add(5, 1, "x"));
    }
}
