/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
package com.github.gbenroscience.parser;

import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.util.FunctionManager;
import com.github.gbenroscience.util.VariableManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Caches the {@link Scanner.TokenTable}s used by {@link MathScanner} and
 * {@link DataSetFormatter}.
 * <p>
 * Previously every expression rebuilt its token vocabulary from scratch: it
 * copied every function and variable name into arrays, hashed them into
 * per-first-character lists and sorted each list. That work depends only on the
 * set of names, which almost never changes between two expressions.
 * <p>
 * Correctness does not rely on anyone telling us when a name is added or
 * removed. Each lookup re-validates the cached table against the live
 * {@link FunctionManager#FUNCTIONS} and {@link VariableManager#VARIABLES} key
 * sets (same size and every cached key still present, which is exactly key-set
 * equality) and against the identity of the inbuilt-method snapshot. A stale
 * table is rebuilt on the spot, so direct writes to those public maps are still
 * honoured. Validation is allocation-free.
 *
 * @author GBEMIRO
 */
final class TokenTables {

    private TokenTables() {
    }

    private static final class Entry {

        final Object inbuiltVersion;
        final String[] functionKeys;
        final String[] variableKeys; // empty for the data-set table
        final Scanner.TokenTable table;
        final Scanner template; // fully configured scanner, re-targeted per input

        Entry(Object inbuiltVersion, String[] functionKeys, String[] variableKeys, Scanner.TokenTable table, Scanner template) {
            this.inbuiltVersion = inbuiltVersion;
            this.functionKeys = functionKeys;
            this.variableKeys = variableKeys;
            this.table = table;
            this.template = template;
        }
    }

    /**
     * The scanner that splits a script into statements on ';'. Its vocabulary is
     * the constant {@link VariableManager#endOfLine}, so one immutable template
     * serves every {@code MathExpression} instead of rebuilding a table each time.
     */
    private static final Scanner STATEMENTS = new Scanner("", false, VariableManager.endOfLine);

    static Scanner statementScanner(String input) {
        return STATEMENTS.withInput(input);
    }

    private static volatile Entry dataSetEntry;
    private static volatile Entry mathEntry;

    /**
     * Vocabulary for {@link DataSetFormatter}: all function names plus the comma
     * and bracket characters.
     */
    static Scanner dataSetScanner(String input) {
        return dataSetEntry().template.withInput(input);
    }

    private static Entry dataSetEntry() {
        Entry e = dataSetEntry;
        Object inbuilt = Declarations.inbuiltMethodsVersion();
        if (e != null && e.inbuiltVersion == inbuilt && sameKeys(FunctionManager.FUNCTIONS, e.functionKeys)) {
            return e;
        }
        String[] fnKeys = keysOf(FunctionManager.FUNCTIONS);
        List<String> tokens = new ArrayList<>(fnKeys.length + 160);
        addAll(tokens, fnKeys);
        addAll(tokens, com.github.gbenroscience.parser.methods.Method.getInbuiltNames());
        tokens.add(Operator.COMMA);
        tokens.add(Operator.OPEN_CIRC_BRAC);
        tokens.add(Operator.CLOSE_CIRC_BRAC);
        tokens.add(Operator.OPEN_SQUARE_BRAC);
        tokens.add(Operator.CLOSE_SQUARE_BRAC);
        Scanner.TokenTable table = Scanner.TokenTable.of(tokens);
        Scanner template = new Scanner.Builder("").includeTokens(true).tokenTable(table).build();
        e = new Entry(inbuilt, fnKeys, new String[0], table, template);
        dataSetEntry = e;
        return e;
    }

    /**
     * Vocabulary for {@link MathScanner}: variable names, function names and
     * every operator.
     */
    static Scanner mathScanner(String input) {
        return mathEntry().template.withInput(input);
    }

    private static Entry mathEntry() {
        Entry e = mathEntry;
        Object inbuilt = Declarations.inbuiltMethodsVersion();
        if (e != null && e.inbuiltVersion == inbuilt
                && sameKeys(FunctionManager.FUNCTIONS, e.functionKeys)
                && sameKeys(VariableManager.VARIABLES, e.variableKeys)) {
            return e;
        }
        String[] fnKeys = keysOf(FunctionManager.FUNCTIONS);
        String[] varKeys = keysOf(VariableManager.VARIABLES);
        List<String> tokens = new ArrayList<>(fnKeys.length + varKeys.length + 200);
        addAll(tokens, varKeys);
        addAll(tokens, fnKeys);
        addAll(tokens, com.github.gbenroscience.parser.methods.Method.getInbuiltNames());
        addAll(tokens, Operator.operators);
        Scanner.TokenTable table = Scanner.TokenTable.of(tokens);
        Scanner template = new Scanner.Builder("")
                .includeTokens(true)
                .ignoreWhitespace(false)
                .addDelimitedRegion("\"", "\"")
                .addDelimitedRegion("'", "'")
                .withDynamicMatcher(MathScanner.DYNAMIC_RULES)
                .tokenTable(table)
                .matchDynamicFirst(true)
                .identifierPartExtra('[', ']')
                .build();
        e = new Entry(inbuilt, fnKeys, varKeys, table, template);
        mathEntry = e;
        return e;
    }

    private static void addAll(List<String> into, String[] items) {
        for (String s : items) {
            into.add(s);
        }
    }

    private static String[] keysOf(Map<String, ?> map) {
        synchronized (map) { // the mutex of a Collections.synchronizedMap is the wrapper itself
            return map.keySet().toArray(new String[0]);
        }
    }

    private static boolean sameKeys(Map<String, ?> map, String[] keys) {
        synchronized (map) {
            if (map.size() != keys.length) {
                return false;
            }
            for (String k : keys) {
                if (!map.containsKey(k)) {
                    return false;
                }
            }
            return true;
        }
    }
}
