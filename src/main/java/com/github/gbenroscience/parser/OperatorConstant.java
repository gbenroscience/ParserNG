package com.github.gbenroscience.parser;

/**
 *
 * @author GBEMIRO
 */
import java.util.HashMap;
import java.util.Map;

public enum OperatorConstant {
    // Basic Arithmetic
    PLUS("+"), 
    MINUS("-"), 
    DIVIDE("/"), 
    MULTIPLY("*"), 
    REMAINDER("%"),
    
    // Powers and Roots
    POWER("^"), 
    ROOT("√"), 
    CUBE_ROOT("³√"), 
    SQUARE("²"), 
    CUBE("³"),
    INVERSE("-¹"),
    
    // Logic and Comparison
    LESS_THAN("<"), 
    GREATER_THAN(">"), 
    LESS_OR_EQUALS("≤"), 
    GREATER_OR_EQUALS("≥"), 
    EQUALS("=="), 
    ASSIGN("="),
    AND("&"), 
    OR("|"),
    
    // Brackets and Delimiters
    OPEN_CIRC_BRAC("("), 
    CLOSE_CIRC_BRAC(")"), 
    OPEN_SQUARE_BRAC("["), 
    CLOSE_SQUARE_BRAC("]"),
    COMMA(","), 
    COLON(":"), 
    SEMI_COLON(";"),
    
    // Specialized Math
    FACTORIAL("!"), 
    PERMUTATION("Р"), 
    COMBINATION("Č"),
    
    // Constants and Commands
    CONST("const"), 
    STORE("store:"), 
    EXIT("exit:"),
    
    // Miscellaneous
    EN_DASH("–"), 
    SPACE(" "), 
    AT("@");

    private final String symbol;

    // Internal Map for O(1) "isOperator" lookup
    private static final Map<String, OperatorConstant> SYMBOL_MAP = new HashMap<>();

    static {
        for (OperatorConstant op : OperatorConstant.values()) {
            SYMBOL_MAP.put(op.symbol, op);
        }
    }

    OperatorConstant(String symbol) {
        this.symbol = symbol;
    }

    public String getSymbol() {
        return symbol;
    }

    /**
     * Replaces your old isOperatorString(String op)
     * Constant time lookup O(1)
     * @param s 
     */
    public static boolean isOperatorString(String s) {
        return s != null && SYMBOL_MAP.containsKey(s);
    }

    /**
     * Handy method to get the actual Enum from a string
     * @param s 
     * @return 
     */
    public static OperatorConstant fromString(String s) {
        return SYMBOL_MAP.get(s);
    }
}