package com.github.gbenroscience.parser.ng.parquet.v1.internal;

/**
 * Pure Dremel-level arithmetic: turns the (repetition, definition) level stream of ONE leaf column into
 * the slot structure of the Arrow containers above it. No Parquet or Arrow types, so it is fully
 * unit-testable.
 *
 * <h2>The rule</h2>
 * Every Arrow node N has two numbers: {@code r} (how many repeated Parquet nodes enclose N's slots) and
 * {@code reach} (the definition level an entry needs before N has a slot at all). An entry
 * {@code (rep, def)} gives N a new slot iff {@code rep <= r && def >= reach}.
 * <ul>
 *   <li>Struct S: one slot per such entry; valid iff {@code def >= d(S)}. Children of a null struct still
 *       get (null) slots, exactly like Arrow.</li>
 *   <li>List L with repeated group R: a list slot per entry with {@code rep <= rL && def >= reachL};
 *       valid iff {@code def >= dL}; the entry adds an element iff {@code rep <= rR && def >= dR}.
 *       A null list and an empty list both add no elements, and are told apart by {@code dL}.</li>
 * </ul>
 * All containers under a leaf can be derived from any single leaf beneath them, because sibling leaves
 * carry identical structure levels.
 */
public final class LevelWalker {

    private LevelWalker() { }

    public interface StructEvents {
        void slot(int index, boolean valid);
    }

    public interface ListEvents {
        void nullList(int index);
        void list(int index, int size);
    }

    /** @return number of struct slots */
    public static int walkStruct(int[] rep, int[] def, int n, int r, int reach, int d, StructEvents ev) {
        int slots = 0;
        for (int i = 0; i < n; i++) {
            if (rep[i] <= r && def[i] >= reach) {
                ev.slot(slots++, def[i] >= d);
            }
        }
        return slots;
    }

    /**
     * @param out receives {@code out[0] = number of list slots}, {@code out[1] = number of element slots}
     */
    public static void walkList(int[] rep, int[] def, int n,
                                int rL, int reachL, int dL, int rR, int dR,
                                ListEvents ev, int[] out) {
        int slots = 0, children = 0;
        int cur = -1, size = 0;
        boolean valid = false;
        for (int i = 0; i < n; i++) {
            final int r = rep[i], d = def[i];
            if (r <= rL && d >= reachL) {
                if (cur >= 0) flush(ev, cur, valid, size);
                cur = slots++;
                valid = d >= dL;
                size = 0;
            }
            if (r <= rR && d >= dR) {
                children++;
                size++;
            }
        }
        if (cur >= 0) flush(ev, cur, valid, size);
        out[0] = slots;
        out[1] = children;
    }

    private static void flush(ListEvents ev, int idx, boolean valid, int size) {
        if (valid) ev.list(idx, size); else ev.nullList(idx);
    }
}
