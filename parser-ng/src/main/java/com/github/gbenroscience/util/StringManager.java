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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Global registry for String values used by ParserNG.
 *
 * <p>The registry maps variable/handle names to their String values.
 * Anonymous strings are assigned names of the form {@code anon_strN}.</p>
 *
 * <p>The map is synchronized because ParserNG may create and resolve
 * String variables from multiple execution contexts.</p>
 *
 * @author GBEMIRO
 */
public final class StringManager {

    private StringManager() {
        // Utility class.
    }

    public static final String ANON_PREFIX = "anon_str";

    /**
     * Total number of anonymous string handles created during this JVM session.
     */
    public static final AtomicInteger ANON_CURSOR = new AtomicInteger();

    /**
     * Global String registry.
     *
     * <p>A null value is permitted because {@link #lockDown(String, String...)}
     * uses a null value to represent a reserved handle whose actual String
     * value will be supplied later.</p>
     */
    public static final Map<String, String> STRINGS =
            Collections.synchronizedMap(new HashMap<>());

    /**
     * Returns the String associated with the supplied name.
     *
     * @param name variable/handle name
     * @return associated String, or null if no value has been assigned
     */
    public static String getString(String name) {
        return STRINGS.get(name);
    }

    /**
     * Looks up a String by name.
     *
     * @param name variable/handle name
     * @return associated String, or null if no value has been assigned
     */
    public static String lookUp(String name) {
        return STRINGS.get(name);
    }
    
    public static boolean hasValue(String txt){
        return STRINGS.containsValue(txt);
    }

    /**
     * Reserves a named String handle if it does not already exist.
     *
     * <p>A reservation is represented by a null value. The operation is
     * atomic with respect to other StringManager operations.</p>
     *
     * @param name name of the String handle
     * @param independentVars retained for API compatibility
     * @return the String currently associated with the handle, or null
     */
    public static String lockDown(String name, String... independentVars) {
        synchronized (STRINGS) {
            if (!STRINGS.containsKey(name)) {
                STRINGS.put(name, null);
            }
            return STRINGS.get(name);
        }
    }

    /**
     * Creates and reserves an anonymous String handle.
     *
     * @param independentVars retained for API compatibility
     * @return anonymous handle name
     */
    public static String lockDownAnon(String... independentVars) {
        String name = ANON_PREFIX + ANON_CURSOR.incrementAndGet();
        lockDown(name, independentVars);
        return name;
    }

    /**
     * Adds a String under an automatically generated anonymous name.
     *
     * @param text String value
     * @return generated anonymous handle
     */
    public static String add(String text) {
        String name = ANON_PREFIX + ANON_CURSOR.incrementAndGet();
        add(name, text);
        return name;
    }

    /**
     * Adds or replaces a String under the supplied name.
     *
     * @param varName variable/handle name
     * @param text String value
     * @return the supplied variable/handle name
     */
    public static String add(String varName, String text) {
        STRINGS.put(varName, text);
        return varName;
    }

    /**
     * Loads String variables into the registry.
     *
     * @param strings Strings to load
     */
    public static void load(Map<String, String> strings) {
        load(strings, false);
    }

    /**
     * Loads String variables into the registry.
     *
     * @param strings Strings to load
     * @param clearFirst whether existing entries should first be removed
     */
    public static void load(Map<String, String> strings, boolean clearFirst) {
        synchronized (STRINGS) {
            if (clearFirst) {
                STRINGS.clear();
            }

            STRINGS.putAll(strings);
        }
    }

    /**
     * Removes a String variable from the registry.
     *
     * @param name variable/handle name
     */
    public static void delete(String name) {
        STRINGS.remove(name);
        update();
    }

    /**
     * Renames a String variable.
     *
     * <p>If the old name does not exist, nothing is changed.</p>
     *
     * @param oldVarName existing variable/handle name
     * @param newName new variable/handle name
     */
    public static void update(String oldVarName, String newName) {
        synchronized (STRINGS) {
            if (!STRINGS.containsKey(oldVarName)) {
                return;
            }

            String value = STRINGS.remove(oldVarName);
            STRINGS.put(newName, value);
        }
    }

    /**
     * Removes all anonymous String variables.
     */
    public static void clearAnonymousStrings() {
        synchronized (STRINGS) {
            Set<Map.Entry<String, String>> entries = STRINGS.entrySet();

            entries.removeIf(entry -> isAnonymousFormat(entry.getKey()));
        }
    }

    /**
     * Removes all registered Strings.
     */
    public static void clear() {
        STRINGS.clear();
    }

    /**
     * Returns the number of anonymous String variables currently registered.
     *
     * @return number of anonymous Strings
     */
    public static int countAnonymousStrings() {
        int count = 0;

        synchronized (STRINGS) {
            for (String name : STRINGS.keySet()) {
                if (isAnonymousFormat(name)) {
                    count++;
                }
            }
        }

        return count;
    }

    /**
     * Determines whether a name has the anonymous String format.
     *
     * <p>Valid examples include {@code anon_str1} and {@code anon_str123}.</p>
     *
     * @param name name to test
     * @return true if the name is an anonymous String handle
     */
    public static boolean isAnonymousFormat(String name) {
        if (name == null) {
            return false;
        }

        int prefixLength = ANON_PREFIX.length();

        if (name.length() <= prefixLength ||
                !name.startsWith(ANON_PREFIX)) {
            return false;
        }

        for (int i = prefixLength; i < name.length(); i++) {
            char c = name.charAt(i);

            if (c < '0' || c > '9') {
                return false;
            }
        }

        return true;
    }

    /**
     * Returns the values of all named String variables.
     *
     * <p>Anonymous handles are excluded.</p>
     *
     * @return list of defined String values
     */
    public static ArrayList<String> getDefinedStrings() {
        ArrayList<String> strings = new ArrayList<>();

        synchronized (STRINGS) {
            for (Map.Entry<String, String> entry : STRINGS.entrySet()) {
                if (!isAnonymousFormat(entry.getKey())) {
                    String value = entry.getValue();

                    if (value != null) {
                        strings.add(value);
                    }
                }
            }
        }

        return strings;
    }

    /**
     * Notification hook for clients/UI integrations.
     *
     * <p>Currently intentionally empty.</p>
     */
    public static void update() {
        // Reserved for listeners/UI notification.
    }
}