/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.github.gbenroscience.parser;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Arrays;

/**
 *
 * @author GBENRO
 */
public class Bracket extends Operator {

    /**
     * The index of the bracketChar in the ArrayList containing the scanned
     * function
     */
    private int index;
    /**
     * objects of this class keep a record of their counterpart or complementing
     * bracketChar.
     *
     */
    private transient Bracket complement;

    private final BracketMode mode;

    public static enum BracketMode {
        CIRCULAR_OPEN('('), CIRCULAR_CLOSE(')'), SQUARE_OPEN('['), SQUARE_CLOSE(']'), CURVED_OPEN('{'), CURVED_CLOSE('}'), ANGULAR_OPEN('<'), ANGULAR_CLOSE('>');
        private final char bracketChar;

        private BracketMode(char bracketChar) {
            this.bracketChar = bracketChar;
        }

        public final char getBracket() {
            return bracketChar;
        }

        public static final BracketMode fromChar(char op) {
            switch (op) {
                case '(':
                    return CIRCULAR_OPEN;
                case ')':
                    return CIRCULAR_CLOSE;
                case '[':
                    return SQUARE_OPEN;
                case ']':
                    return SQUARE_CLOSE;
                case '{':
                    return CURVED_OPEN;
                case '}':
                    return CURVED_CLOSE;
                case '<':
                    return ANGULAR_OPEN;
                case '>':
                    return ANGULAR_CLOSE;
                default:
                    throw new RuntimeException("Invalid bracket char spotted: " + Character.toString(op));
            }
        }

        public BracketMode getComplement() {
            return getComplement(this);
        }

        public static final BracketMode getComplement(BracketMode bm) {
            switch (bm.bracketChar) {
                case '(':
                    return CIRCULAR_CLOSE;
                case ')':
                    return CIRCULAR_OPEN;
                case '[':
                    return SQUARE_CLOSE;
                case ']':
                    return SQUARE_OPEN;
                case '{':
                    return CURVED_CLOSE;
                case '}':
                    return CURVED_OPEN;
                case '<':
                    return ANGULAR_CLOSE;
                case '>':
                    return ANGULAR_OPEN;
                default:
                    throw new RuntimeException("Invalid bracket mode spotted: " + Character.toString(bm.bracketChar));
            }
        }

    }

    /**
     * Constructor of this class for creating its objects and initializing their
     * names with either a (,[,{,< or ),],},>
     *
     * @param op
     */
    public Bracket(String op) {
        super(op);
        if (op.length() == 1) {
            this.mode = BracketMode.fromChar(op.charAt(0));
        }else{
              throw new RuntimeException("Invalid bracket entry spotted: " + op);
        }
    }

    /**
     * Constructor of this class for creating its objects and initializing their
     * names with either a ( or a ) and initial
     *
     * @param op
     */
    public Bracket(char op) {
        this(Character.toString(op));
    }

    /**
     *
     * @param mode The BracketMode
     */
    public Bracket(BracketMode mode) {
        this(mode.bracketChar);
    }

    public static final Bracket fromMode(BracketMode bm) {
        Bracket b = new Bracket(bm.bracketChar);
        Bracket comp = new Bracket(BracketMode.getComplement(bm).bracketChar);
        b.setComplement(comp);
        return b;
    }

    /**
     * Used to create similar objects that are not equal The object created by
     * this class is similar to the parameter because it contains the same data
     * as the parameter. However,its address in memory is different because it
     * refers to an entirely different object of the same class,but having
     * similar attributes.
     *
     * How can this method be of any use? Imagine an Array of Brackets say array
     * bracs filled with Bracket objects.
     *
     * If we create another Bracket array, say array moreBracs and copy the
     * objects in bracs into moreBracs.Now, both bracs and moreBracs will hold
     * references to these Bracket objects in memory.Java will not create new,
     * similar objects at another address in memory and store in the new array.
     * The command was most likely moreBracs=bracs; or in a loop, it would look
     * like:
     *
     * for(int i=0;i&lt;bracs.length;i++){ moreBracs=bracs[i]; }
     *
     * These statements will only ensure that both arrays will hold a reference
     * to the same objects in memory,i.e RAM.
     *
     * Hence whenever an unsuspecting coder modifies the contents of bracs,
     * thinking He/She has a backup in moreBracs,Java is effecting the
     * modification on the objects referred to by moreBracs, too.This can cause
     * a serious logical error in applications. To stop this, we use this method
     * in this way:
     *
     * for(int i=0;i&lt;bracs.length;i++){
     * moreBracs[i]=createTwinBracket(bracs[i]); }
     *
     * Note that this can be applied to all storage objects too e.g Collection
     * objects and so on.
     *
     * @param brac The object whose twin we wish to create.
     * @return a Bracket object that manifests exactly the same attributes as
     * brac but is a distinct object from brac.
     */
    public static Bracket createTwinBracket(Bracket brac) {
        Bracket newBrac = new Bracket(brac.getName());
        newBrac.setComplement(brac.getComplement());
        newBrac.setIndex(brac.getIndex());
        return newBrac;
    }

    /**
     * non-static version of the above method. This one creates a twin for this
     * Bracket object. The one above creates a twin for the specified bracket
     * object.
     *
     * @return a Bracket object that manifests exactly the same attributes as
     * brac but is a distinct object from brac.
     */
    public Bracket createTwinBracket() {
        Bracket newBrac = new Bracket(getName());
        newBrac.setComplement(getComplement());
        newBrac.setIndex(getIndex());
        return newBrac;
    }

    public BracketMode getMode() {
        return mode;
    }
    
    

    /**
     *
     * @return the index of this Bracket object in a scanned function
     */
    public int getIndex() {
        return index;
    }

    /**
     *
     * @param index the ne w index to assign to this Bracket object in a scanned
     * Function
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     *
     * @return the Bracket object which is the complement of this Bracket object
     */
    public Bracket getComplement() {
        return complement;
    }

    /**
     *
     * @param complement sets the Bracket object which is to be the complement
     * to this one in the scanned Function
     */
    public void setComplement(Bracket complement) {
        this.complement = complement;
    }

    /**
     * checks if the Bracket object argument below is the sane as the complement
     * to this Bracket object.
     *
     * @param brac The Bracket object whose identity is to be checked whether or
     * not it complements this Bracket object.
     * @return true if the parameter is the complement to this one.
     */
    public boolean isComplement(Bracket brac) {
        return brac == getComplement();
    }

    /**
     *
     * @param brac the bracketChar to be checked if or not it is enclosed by
     * this bracketChar and its complement.
     * @return true if the bracketChar is enclosed by this bracketChar and its
     * counterpart.
     */
    public boolean encloses(Bracket brac) {
        boolean truth = false;

        if (this.getIndex() < brac.getIndex() && this.getComplement().getIndex() > brac.getIndex()) {
            truth = true;
        } else if (this.getIndex() > brac.getIndex() && this.getComplement().getIndex() < brac.getIndex()) {
            truth = true;
        }

        return truth;
    }

    /**
     *
     * @param brac an ArrayList object containing all brackets found in a
     * function
     * @return the number of bracketChar pairs contained between this Bracket
     * object and its complement
     */
    public int getNumberOfInternalBrackets(ArrayList<Bracket> brac) {
        int num = 0;
        int i = 0;
        while (i < brac.size()) {
            if (encloses(brac.get(i))) {
                num++;
            }

            i++;
        }
        return (num / 2);
    }

    /**
     * @param scan The ArrayList object containing the scanned function.
     * @param openMode [, (, { or <
     * @return true if this Bracket object forms with its complement, a single
     * bracketChar pair that is a bracketChar pair containing no other
     * bracketChar pairs.
     */
    public boolean isSBP(ArrayList<String> scan, BracketMode openMode) {
        int i = this.index;
        int j = this.complement.index;

        char open = openMode.bracketChar;
        char close = openMode.getComplement().bracketChar;

        if (i < j) {
            ++i;//step away from current bracketChar and start searching for other brackets.
            for (; i < j; i++) {
                String stTkn = scan.get(i);
                char token = stTkn.length() == 1 ? stTkn.charAt(0) : '\u0000';
                if (token == open || token == close) {
                    return false;
                }
            }//end for
        } else if (i > j) {
            ++j;//step away from current bracketChar and start searching for other brackets.
            for (; j < i; j++) {

                String stTkn = scan.get(j);
                char token = stTkn.length() == 1 ? stTkn.charAt(0) : '\u0000';
                if (token == open || token == close) {
                    return false;
                }

            }//end for
        } else if (i == j) {
            throw new InputMismatchException("Open MBracket Cannot Be On The Same Index As Closing MBracket");
        }
        return true;
    }//end method

    /**
     * @param isOpenBracket boolean variable that should be true if this
     * bracketChar object whose complement we seek is an opening bracketChar i.e
     * (, and should be set to false if this bracketChar object whose complement
     * we seek is a closing bracketChar i.e )
     * @param mode The {@link BracketMode}
     * @param start the index of the given bracketChar.
     * @param scan the ArrayList containing the scanned function.
     * @return the index of the enclosing or complement bracketChar of this
     * bracketChar object
     */
    public static int getComplementIndex(boolean isOpenBracket, BracketMode mode, int start, List<String> scan) {

        char openBrac = mode.bracketChar;
        char closeBrac = BracketMode.getComplement(mode).bracketChar;

        int open = 0;
        int close = 0;
        int stop = 0;
        if (isOpenBracket) {
            try {
                for (int i = start; i < scan.size(); i++) {
                    String s = scan.get(i);
                    if (s.charAt(0) == openBrac) {
                        open++;
                    } else if (s.charAt(0) == closeBrac) {
                        close++;
                    }
                    if (open == close) {
                        stop = i;
                        break;
                    }

                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }
        }//end if
        else if (!isOpenBracket) {
            try {
                for (int i = start; i >= 0; i--) {
                    try {
                        String s = scan.get(i);
                        if (s.charAt(0) == openBrac) {
                            open++;
                        } else if (s.charAt(0) == closeBrac) {
                            close++;
                        }
                        if (open == close) {
                            stop = i;
                            break;
                        }
                    }//end try
                    catch (IndexOutOfBoundsException ind) {
                    }
                }//end for
            }//end try
            catch (IndexOutOfBoundsException ind) {
            }

        }
        return stop;
    }

    /**
     * @param isOpenBracket boolean variable that should be true if this
     * bracketChar object whose complement we seek is an opening bracketChar i.e
     * (, and should be set to false if this bracketChar object whose complement
     * we seek is a closing bracketChar i.e )
     * @param mode The {@link BracketMode}
     * @param start the index of the given bracketChar.
     * @param expr the function string containing the brackets.
     * @return the index of the enclosing or complement bracketChar of this
     * bracketChar object
     */
    public static int getComplementIndex(boolean isOpenBracket, BracketMode mode, int start, String expr) {

        char openBrac = mode.bracketChar;
        char closeBrac = BracketMode.getComplement(mode).bracketChar;
        int open = 0;
        int close = 0;
        int stop = 0;
        if (expr == null) {
            return -1;
        }
        final int n = expr.length();
        if (isOpenBracket) {
            if (start < 0 || start >= n) {
                return stop;
            }
            for (int i = start; i < n; i++) {
                char c = expr.charAt(i);
                if (c == openBrac) {
                    open++;
                } else if (c == closeBrac) {
                    close++;
                }
                if (open == close) {
                    stop = i;
                    break;
                }
            }//end for
        }//end if
        else if (!isOpenBracket) {
            if (start < 0 || start >= n) {
                return stop;
            }
            for (int i = start; i >= 0; i--) {
                char c = expr.charAt(i);
                if (c == openBrac) {
                    open++;
                } else if (c == closeBrac) {
                    close++;
                }
                if (open == close) {
                    stop = i;
                    break;
                }
            }//end for 
        }
        return stop;
    }

    /**
     *
     * @param list The list containing the scanned math expression.
     * @param start The point in the list where this algorithm should start
     * checking the bracketChar syntax.(inclusive)
     * @param end The point in the list where this algorithm should stop
     * checking the bracketChar syntax.(exclusive — same convention as
     * {@link List#subList(int, int)}, which is what this method actually used
     * under the hood before this rewrite).
     * @return true if the bracketChar syntax of the scanned expression in the
     * given range is valid or the expression in the given range is devoid of
     * brackets.
     */
    public static boolean checkBracketStructure(List<String> list, int start, int end) {
        return validateBracketStructure(list, start, end,
                BracketMode.CIRCULAR_OPEN, BracketMode.SQUARE_OPEN, BracketMode.CURVED_OPEN, BracketMode.ANGULAR_OPEN);
    }//end method

    /**
     *
     * @param list The list containing the scanned math expression.
     * @param start The point in the list where this algorithm should start
     * checking for brackets.(inclusive)
     * @param end The point in the list where this algorithm should stop
     * @param openMode checking for brackets.(inclusive)
     * @return true if the scanned expression contains no brackets in the given
     * range.
     */
    public static boolean hasBracketsInRange(List<String> list, int start, int end, BracketMode openMode) {
        int sz = list.size();
        if (start >= 0 && end < sz) {

            char open = openMode.bracketChar;
            char close = openMode.getComplement().bracketChar;
            for (int i = start; i <= end; i++) {
                String stTkn = list.get(i);
                char token = stTkn.length() == 1 ? stTkn.charAt(0) : '\u0000';
                if (token == open || token == close) {
                    return true;
                }
            }
        }//end if

        return false;
    }//end method

    /**
     *
     * @param bracket The String object.
     * @return true if the String object represents an open bracketChar
     */
    public static boolean isOpenBracket(String bracket) {
        return bracket.equals("(");
    }

    /**
     *
     * @param bracket The String object.
     * @return true if the String object represents a close bracketChar
     */
    public static boolean isCloseBracket(String bracket) {
        return bracket.equals(")");
    }

    /**
     *
     * @param scan The ArrayList containing the scanned function inside which
     * this Bracket exists.
     * @param registry The {@link MathExpression.VariableRegistry} of the
     * {@link MathExpression} that owns the scanned output
     * @return true if between this Bracket and its complement, a Variable
     * object is found.
     */
    public boolean simpleBracketPairHasVariables(List<String> scan, MathExpression.VariableRegistry registry) {

        if (isOpenBracket(name)) {
            int i = this.index;
            int j = this.complement.index;

            for (; i <= j; i++) {
                String var = scan.get(i);
                if (Variable.isVariableString(var)) {
                    try {
                        Variable v = registry.lookUp(var, false);
                        return v != null;
                    }//end try
                    catch (NullPointerException exception) {
                    }//end catch
                }//end if
            }//end for
        }//end if
        else {
            int i = this.index;
            int j = this.complement.index;
            for (; j <= i; j++) {
                String var = scan.get(j);
                if (Variable.isVariableString(var)) {
                    try {
                        Variable v = registry.lookUp(var, false);
                        return v != null;
                    }//end try 
                    catch (NullPointerException exception) {
                    }//end catch
                }//end if

            }//end for
        }//end else

        return false;
    }//end method

    /**
     *
     * @param scan The ArrayList object containing the scanned function.
     * @return The contents of this bracketChar and its complement as a string,
     * the bracketChar and its complement are also returned. e.g in
     * 5+(2+3-sin2).. This method will return (2+3-sin2).
     */
    public String getDomainContents(List<String> scan) {

        StringBuilder contents = new StringBuilder();
        if (isOpenBracket(name)) {
            int i = this.index;
            int j = this.complement.index;

            for (; i <= j; i++) {
                contents.append(scan.get(i));
            }//end for
        }//end if
        else {
            int i = this.index;
            int j = this.complement.index;
            for (; j <= i; j++) {
                contents.append(scan.get(j));
            }//end for
        }//end else
        return contents.toString();
    }//end method

    /**
     * returns a List containing the contents of a bracketChar pair,including
     * the bracketChar pair itself.
     *
     * @param scan the ArrayList containing the scanner output for a Function
     * @param openMode 
     * @return the bracketChar pair and its contents.
     */
    public List<String> getBracketDomainContents(List<String> scan, BracketMode openMode) {
        char open = openMode.bracketChar;
        char close = openMode.getComplement().bracketChar;
        if ( open == this.name.charAt(0)) {
            return scan.subList(this.getIndex(), this.getComplement().getIndex() + 1);
        } else if (close == this.name.charAt(0)) {
            return scan.subList(this.getComplement().getIndex(), this.getIndex() + 1);
        }
        return null;
    }

    /**
     * Fast, single-pass validation of possibly-interleaved bracket structure
     * across any number of bracket kinds at once.
     *
     * Given a set of OPEN {@link BracketMode}s (e.g. CIRCULAR_OPEN,
     * SQUARE_OPEN, CURVED_OPEN, ANGULAR_OPEN), this scans
     * {@code list[start, end)} exactly once with a single combined stack,
     * pushing the expected closing char whenever it meets one of the given
     * opening chars, and popping/checking it whenever it meets any of the
     * corresponding closing chars.
     *
     * <p>
     * This replaces validating each bracket kind independently in separate
     * passes (the original approach used by
     * {@link #checkBracketStructure(List, int, int)}), which was strictly worse
     * on two counts:</p>
     * <ul>
     * <li><b>Slower</b>: one O(n) sweep over the token range here, instead of
     * one O(n) sweep per bracket kind — 4x fewer token visits for the 4-kind
     * case in {@code checkBracketStructure}, and no
     * {@link List#subList(int, int)} view is allocated per kind.</li>
     * <li><b>Less correct</b>: validating each bracket kind in isolation cannot
     * catch brackets of different kinds that are individually balanced but
     * incorrectly interleaved with each other, e.g. {@code "(3+[2)]"} — one '('
     * matches one ')', and one '[' matches one ']', so 4 separate single-kind
     * passes would wrongly accept it. A single shared stack correctly rejects
     * it, since the ')' does not match the ']' that the stack expects to close
     * next.</li>
     * </ul>
     *
     * <p>
     * Only the OPEN mode of each bracket kind should be supplied — the matching
     * CLOSE char for each is derived automatically via
     * {@link BracketMode#getComplement(BracketMode)}, since the closing bracket
     * is entirely a consequence of whichever opening bracket it must match.
     * (Passing a CLOSE mode by mistake is tolerated — it is normalized to its
     * OPEN complement — but only the OPEN modes need ever be passed.)</p>
     *
     * @param list the list containing the scanned math expression.
     * @param start the index in {@code list} to start checking from
     * (inclusive).
     * @param end the index in {@code list} to stop checking at (exclusive,
     * matching {@link List#subList(int, int)} semantics).
     * @param modes the OPEN bracket modes to validate together in one sweep,
     * e.g. {@code BracketMode.CIRCULAR_OPEN,
     *              BracketMode.SQUARE_OPEN}. Passing none is trivially valid, since there is
     * then nothing to check.
     * @return true if the bracket syntax across all given bracket kinds is
     * valid — properly matched AND properly nested/interleaved — in the given
     * range, or the range contains none of the given bracket kinds at all.
     */
    private static boolean validateBracketStructure(List<String> list, int start, int end, BracketMode... modes) {
        if (list == null) {
            return false;
        }
        if (start < 0 || end > list.size() || start > end) {
            throw new IndexOutOfBoundsException(
                    "Invalid range [" + start + ", " + end + ") for a list of size " + list.size());
        }
        if (modes == null || modes.length == 0) {
            return true;
        }

        int m = modes.length;
        char[] opens = new char[m];
        char[] closes = new char[m];
        for (int k = 0; k < m; k++) {
            // Normalize to the OPEN mode regardless of what was passed, so
            // it is always the opening char that gets pushed and its
            // complement that gets matched against on close.
            BracketMode open = isCloseMode(modes[k]) ? BracketMode.getComplement(modes[k]) : modes[k];
            opens[k] = open.getBracket();
            closes[k] = BracketMode.getComplement(open).getBracket();
        }

        Deque<Character> stack = new ArrayDeque<>(Math.max(16, (end - start) / 4));

        for (int i = start; i < end; i++) {
            String token = list.get(i);
            if (token == null || token.isEmpty()) {
                continue;
            }
            char c = token.charAt(0);

            int openIdx = indexOfChar(opens, c);
            if (openIdx >= 0) {
                stack.push(closes[openIdx]);
                continue;
            }

            int closeIdx = indexOfChar(closes, c);
            if (closeIdx >= 0) {
                if (stack.isEmpty() || stack.pop() != c) {
                    // Either an unmatched closing bracket, or it closes a
                    // different bracket kind than the one the stack expects
                    // to close next -- e.g. "(3+[2)]".
                    return false;
                }
            }
        }

        // valid only if no unmatched opening brackets remain
        return stack.isEmpty();
    }

    /**
     * True if {@code mode} is one of the four CLOSE variants.
     */
    private static boolean isCloseMode(BracketMode mode) {
        switch (mode) {
            case CIRCULAR_CLOSE:
            case SQUARE_CLOSE:
            case CURVED_CLOSE:
            case ANGULAR_CLOSE:
                return true;
            default:
                return false;
        }
    }

    /**
     * Linear search over a small (bracket-kind-count-sized) char array.
     */
    private static int indexOfChar(char[] arr, char c) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == c) {
                return i;
            }
        }
        return -1;
    }

    /**
     * @param scanner The ArrayList containing the scanner output for a Function
     * Multiplies the contents of this List by -1.
     */
    public void multiplyContentsByMinusOne(List<String> scanner, BracketMode openMode) {
        List<String> domain = getBracketDomainContents((ArrayList<String>) scanner, openMode);

        for (int i = 0; i < domain.size(); i++) {
            if (domain.get(i).equals("+")) {
                domain.set(i, "-");
            }//end if
            else if (domain.get(i).equals("-")) {
                domain.set(i, "+");
            }//end if

        }//end for loop

        if (Number.isNumber(domain.get(1))) {
            domain.set(1, "" + (-1 * Double.parseDouble(domain.get(1))));
        } else if (Variable.isVariableString(domain.get(1))) {
            domain.add(1, "*");
            domain.add(1, "-1");
            complement.setIndex(complement.index + 2);
        }

    }//end method

    /**
     * @param scanner The ArrayList containing the scanner output for a Function
     * @param index The index at which the token is to be retrieved. The first
     * and elements are compulsorily always an open bracketChar and a close
     * bracketChar respectively.
     */
    public String domainTokenAt(List<String> scanner, int index, BracketMode openMode) {
        List<String> domain = getBracketDomainContents((ArrayList<String>) scanner, openMode);
        return domain.get(index);
    }

    public String toString() {
        return String.format(
                "{\"name\": \"%s\", \"index\": %d, \"c_name\": \"%s\", \"c_index\": %d}",
                name, index, complement.name, complement.index
        );
    }

    public static void main(String... args) {

        MathExpression me = new MathExpression("4*x^3*sin(x^2)");
        System.out.println("scanner: " + me.scanner);
        System.out.println("rpn-tokens: " + Arrays.deepToString(me.getCachedPostfix()));

        String s1 = "sin(1)+cos(1)+tan(1)+log(10)+sqrt(16)+exp(1)+pow(2,8)+abs(-42)+sum(1,2,3,4,5)+sin(3*12+cos(55))-(4+5)*(2*(9-2)+12*(4-7));";

        MathExpression m = new MathExpression(s1);

        double N = 10000;
        double start = System.nanoTime();
        boolean s = false;
        for (int i = 0; i < N; i++) {
            s = Bracket.checkBracketStructure(m.scanner, 0, m.scanner.size());
        }
        double interval = (System.nanoTime() - start) / N;
        System.out.println("soln: " + s + ", " + (interval / 1000) + " microns");

        start = System.nanoTime();
        s = false;
        for (int i = 0; i < N; i++) {
            s = Bracket.checkBracketStructure(m.scanner, 0, m.scanner.size());
        }
        interval = (System.nanoTime() - start) / N;
        System.out.println("soln: " + s + ", " + (interval / 1000) + " microns");

    }//end method
}//end class Bracket
