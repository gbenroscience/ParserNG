package parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import static java.lang.Math.*;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author JIBOYE OLUWAGBEMIRO OLAOLUWA. This class has many methods. Method
 * delete provides 2 constructors.The at takes 2 arguments,the string to
 * manipulate and the index to be deleted from the string.It returns the
 * original string without the element at the specified index.
 *
 * Method delete takes 3 arguments.The string,the index from which deletion is
 * to begin and the index where deletion ends.The returned string is the
 * original string without the characters from the starting index to one index
 * behind the ending index.So the character at the ending index is not deleted.
 * So if we write At( u,0,u.length() ),the whole string is deleted.
 *
 * Method isDouble1 takes a single argument;that is the number string which we
 * wish to know if it is of type double or not. It is a bold attempt by the
 * author to devise a means of checking if a number entry qualifies as a double
 * number string. This checking is done at much higher speeds in comparison to
 * method isDouble.
 */
public class STRING {

    /**
     * contains allows you to verify if an input string contains a given
     * substring.
     *
     * @param str : the string to check
     * @param substr : the substring to check for
     * @return true or false depending on if or not the input consists of digits
     * alone or otherwise.
     */
    public static boolean contains(String str, String substr) {
        return str != null && str.contains(substr);
    }

    /**
     * isOnlyDigits allows you to verify if an input string consists of just
     * digits and the exponential symbol. So the method will return true for
     * 32000 but not for 3.2E4.It will also return true for 32E3.No decimal
     * points are allowed
     *
     * @param h ..The string to be analyzed.
     * @return true or false depending on if or not the input consists of digits
     * alone or otherwise.
     */
    public static boolean isOnlyDigits(String h) {
        // 1. Safety check: An empty or null string contains no digits
        if (h == null || h.isEmpty()) {
            return false;
        }

        int len = h.length();
        for (int i = 0; i < len; i++) {
            char c = h.charAt(i);
            // 2. Fast primitive check (faster than calling a separate method)
            if (c < '0' || c > '9') {
                return false; // 3. Early exit: stop as soon as we know it's not a digit
            }
        }
        return true;
    }

    /**
     *
     * @param s The single character string to be examined if it is a letter of
     * the alphabet or not.
     * @return true if the parameter is a valid letter of the English alphabet.
     */
    public static boolean isLetter(String s) {
        // 1. Safety check to avoid IndexOutOfBoundsException
        if (s == null || s.isEmpty()) {
            return false;
        }

        // 2. Access the character directly without copying the array
        return Character.isLetter(s.charAt(0));
    }

    /**
     *
     * @param input The string from which we wish to remove all new line
     * characters.
     * @return a string like the input but free of all new line characters.
     */
    public static String removeNewLineChar(String input) {
        if (input == null) {
            return null;
        }

        int len = input.length();
        // Pre-sizing the builder to the input length avoids resizing mid-loop
        StringBuilder sb = new StringBuilder(len);

        for (int i = 0; i < len; i++) {
            char c = input.charAt(i);
            // Only append if it's NOT a newline
            if (c != '\n' && c != '\r') { // Added \r check just in case of Windows line endings
                sb.append(c);
            }
        }
        return sb.toString();
    }

//abcdabcde
//  234567
    /**
     * isFullyDouble allows you to verify if an input string is in double number
     * format or not.
     *
     * @param num ..The string to be analyzed.
     * @return true or false depending on if or not the input represents a valid
     * double number
     */
    public static boolean isFullyDouble(String num) {
        if (num == null || num.isEmpty()) {
            return false;
        }

        int n = num.length();
        // quick reject: cannot end with E/e or + or -
        char last = num.charAt(n - 1);
        if (last == 'E' || last == 'e' || last == '+' || last == '-') {
            return false;
        }

        boolean seenDigit = false;   // any digit seen (before or after point, before or after E)
        boolean seenPoint = false;   // decimal point seen (only allowed before E)
        boolean seenExp = false;     // exponent marker seen
        boolean expSignSeen = false; // sign immediately after E seen
        int signCount = 0;           // total signs encountered (leading and exponent sign)

        for (int i = 0; i < n; i++) {
            char c = num.charAt(i);

            if (Character.isDigit(c)) {
                seenDigit = true;
                continue;
            }

            if (c == '+' || c == '-') {
                // Leading sign allowed at index 0
                if (i == 0) {
                    signCount++;
                    continue;
                }
                // Sign after exponent is allowed only immediately after E/e and only once
                if (seenExp && !expSignSeen && num.charAt(i - 1) == 'E' || (seenExp && !expSignSeen && num.charAt(i - 1) == 'e')) {
                    expSignSeen = true;
                    signCount++;
                    continue;
                }
                // Any other sign placement is invalid
                return false;
            }

            if (c == '.') {
                // decimal point not allowed after exponent or more than once
                if (seenPoint || seenExp) {
                    return false;
                }
                seenPoint = true;
                continue;
            }

            if (c == 'E' || c == 'e') {
                // E must come after at least one digit and only once
                if (!seenDigit || seenExp) {
                    return false;
                }
                seenExp = true;
                // reset digit tracking for exponent digits (must have at least one digit after E)
                seenDigit = false;
                expSignSeen = false;
                continue;
            }

            // any other character invalidates the string
            return false;
        }

        // final check: ensure at least one digit was seen (and if E was present, digits after E exist)
        return seenDigit;
    }

    public static char firstChar(String u) {
        return u.charAt(0);
    }

    public static char lastChar(String u) {
        return u.charAt(u.length() - 1);
    }

    /**
     * method firstElement takes one argument and that is the string to be
     * modified.
     *
     * @param u the string to be modified
     * @return the first element of the string
     * @throws StringIndexOutOfBoundsException
     */
    public static String firstElement(String u) {
        if (u == null || u.isEmpty()) {
            return "";
        }
        return Character.toString(u.charAt(0));
    }

    /**
     * method lastElement takes one argument and that is the string to be
     * modified.
     *
     * @param u the string to be modified
     * @return the last element of the string
     * @throws StringIndexOutOfBoundsException
     */
    public static String lastElement(String u) {
        if (u == null || u.isEmpty()) {
            return "";
        }
        return Character.toString(u.charAt(u.length() - 1));
    }

    /**
     * nextIndexOf allows you to get the next occurence of a substring searching
     * from a point in the string forwards
     *
     * @param str ..The String being analyzed.
     * @param index ..The starting point of the search.
     * @param str2 ..The string whose next occurence we desire.
     * @return the next index of the str2 if found or -1 if not found
     */
    public static int nextIndexOf(String str, int index, String str2) {
        // 1. Basic safety check
        if (str == null || str2 == null) {
            return -1;
        }

        // 2. Index logic: Start search at index + 1
        int startSearchAt = index + 1;

        // 3. Simple bounds check to avoid exceptions
        if (startSearchAt < 0 || startSearchAt >= str.length()) {
            return -1;
        }

        // 4. Native performance: searches forward without creating a substring
        return str.indexOf(str2, startSearchAt);
    }

    /**
     * prevIndexOf allows you to get the previous occurence of a substring
     * searching from a point in the string backwards
     *
     * @param str ..The String being analyzed.
     * @param index ..The starting point of the search.
     * @param str2 ..The string whose previous occurence we desire.
     * @return the previous index of the str2 if found or -1 if not found
     */
    public static int prevIndexOf(String str, int index, String str2) {
        // 1. Basic null/empty safety
        if (str == null || str2 == null) {
            return -1;
        }

        // 2. Logical safety checks (Simplified)
        // We search up to (index - 1) because the current 'index' is usually 
        // the character we are already looking at.
        if (index <= 0 || index > str.length()) {
            return -1;
        }

        // 3. The high-performance way: No substrings!
        // lastIndexOf(string, fromIndex) searches starting from fromIndex backwards.
        return str.lastIndexOf(str2, index - 1);
    }

    /**
     *
     * @param val the entry being analyzed.
     * @return true if the entry is a + sign or a minus sign
     */
    public static boolean isSign(String val) {
        return val != null && val.length() == 1 && (val.charAt(0) == '+' || val.charAt(0) == '-');
    }

    /**
     *
     * @param val the entry being analyzed.
     * @return true if the entry is a floating point
     */
    public static boolean isDecimalPoint(String val) {
        return val != null && val.length() == 1 && val.charAt(0) == '.';
    }

    /**
     *
     * @param val the entry being analyzed.
     * @return true if the entry is the power of 10 symbol i.e E
     */
    public static boolean isExponentOf10(String val) {
        return val != null && val.length() == 1 && val.charAt(0) == 'E';
    }

    /**
     * method getFirstOccurenceOfDigit takes a String object and returns the
     * first occurence of a number character in the String.
     *
     * @param value the string to analyze
     * @return the first occurence of a number character in the sequence.
     * @throws StringIndexOutOfBoundsException
     */
    public static String getFirstOccurenceOfDigit(String value) {
        if (value == null) {
            return "";
        }

        int len = value.length();
        for (int i = 0; i < len; i++) {
            char c = value.charAt(i);
            // Fast primitive check
            if (c >= '0' && c <= '9') {
                // Converts a single char to String efficiently
                return Character.toString(c);
            }
        }
        return "";
    }

    /**
     * method getFirstIndexOfDigit takes a String object and returns the index
     * of the first occurence of a number character in the String.
     *
     * @param val the string to analyze
     * @return the index of the first occurence of a number character in the
     * sequence.
     * @throws StringIndexOutOfBoundsException
     */
    public static int getFirstIndexOfDigit(String val) {
        if (val == null) {
            return -1;
        }

        int len = val.length();
        for (int i = 0; i < len; i++) {
            char c = val.charAt(i);
            // Direct character comparison is much faster than substring + method call
            if (c >= '0' && c <= '9') {
                return i;
            }
        }
        return -1;
    }

    /**
     * method getFirstIndexOfDigit takes a String object and returns the index
     * of the first occurence of a number character in the String.
     *
     * @param val the string to analyze
     * @return the index of the first occurence of a number character in the
     * sequence.
     * @throws StringIndexOutOfBoundsException
     */
    public static int getFirstIndexOfDigitOrPoint(String val) throws StringIndexOutOfBoundsException {
        int i;
        for (i = 0; i < val.length(); i++) {
            if (isDigit(val.substring(i, i + 1)) || val.substring(i, i + 1).equals(".")) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Counts how many times a given single character string occurs in a given
     * String object.
     *
     * @param c A single character String value whose occurences in another
     * string we wish to count.
     * @param str The String object that we will search through.
     * @return The number of times c occurs in str
     */
    public static int countOccurences(String c, String str) {
        // Validate inputs without expensive concatenation in the exception message
        if (c == null || c.length() != 1 || str == null) {
            throw new IllegalArgumentException("Target string must be length 1 and source must not be null.");
        }

        char target = c.charAt(0);
        int count = 0;
        int len = str.length();

        for (int i = 0; i < len; i++) {
            if (str.charAt(i) == target) {
                count++;
            }
        }
        return count;
    }

    /**
     * Returns the first number-like substring starting at the first digit found
     * in the input. Supports optional decimal point and exponent (E or e) with
     * optional sign. Examples it will extract: "123", "0.45", ".78",
     * "3.14E-10".
     *
     * Note: relies on getFirstIndexOfDigit(val) to return the index of the
     * first digit in val, or -1 if none.
     *
     * @param val the string to analyze
     * @return the index of the first occurrence of a number character in the
     * sequence.
     * @throws StringIndexOutOfBoundsException
     */
    public static String getFirstNumberSubstring(String val) {
        if (val == null || val.isEmpty()) {
            return "";
        }

        int n = val.length();
        int ind = getFirstIndexOfDigit(val);
        if (ind < 0 || ind >= n) {
            return "";
        }

        StringBuilder sb = new StringBuilder(16);

        // If the character before the first digit is a dot, treat as leading "."
        boolean seenPoint = false;
        boolean seenExp = false;
        boolean expSignSeen = false;
        int expDigits = 0;

        if (ind > 0 && val.charAt(ind - 1) == '.') {
            sb.append('.');
            seenPoint = true;
        }

        // Walk forward from the first digit and build the number token
        for (int i = ind; i < n; i++) {
            char c = val.charAt(i);

            // Digit
            if (Character.isDigit(c)) {
                sb.append(c);
                if (seenExp) {
                    expDigits++;
                }
                continue;
            }

            // Decimal point (only allowed once and not after exponent)
            if (c == '.') {
                if (seenPoint || seenExp) {
                    break;
                }
                seenPoint = true;
                sb.append('.');
                continue;
            }

            // Exponent marker (E or e)
            if (c == 'E' || c == 'e') {
                if (seenExp) {
                    break; // only one exponent allowed
                }            // If no digits have been collected yet (e.g., "." followed by "E"), stop
                if (sb.length() == 0 || (sb.length() == 1 && sb.charAt(0) == '.')) {
                    break;
                }
                seenExp = true;
                sb.append('E'); // normalize to 'E'
                continue;
            }

            // Sign inside exponent (only immediately after E and only once)
            if ((c == '+' || c == '-') && seenExp && !expSignSeen && expDigits == 0) {
                expSignSeen = true;
                sb.append(c);
                continue;
            }

            // Any other character ends the number token
            break;
        }

        // Trim trailing incomplete exponent/sign markers:
        // If last char is '+' or '-' (dangling sign) or 'E' (dangling exponent), remove them.
        while (sb.length() > 0) {
            char last = sb.charAt(sb.length() - 1);
            if (last == '+' || last == '-' || last == 'E') {
                sb.setLength(sb.length() - 1);
                // If we removed the sign, we should continue checking for a dangling 'E'
                // (e.g., "123E+" -> remove '+' then remove 'E')
                continue;
            }
            break;
        }

        return sb.toString();
    }

    /**
     * Allows you to remove commas from a string.
     *
     * @param h
     * @return a string in which the contents of the original have been freed
     * from all commas.
     */
    public static String removeCommas(String s) {
        if (s == null) {
            return null;
        }

        int firstComma = s.indexOf(',');
        if (firstComma == -1) {
            return s; // no commas — return original string
        }
        StringBuilder sb = new StringBuilder(s.length());
        // copy prefix up to first comma
        sb.append(s, 0, firstComma);

        // append remaining characters skipping commas
        for (int i = firstComma + 1, n = s.length(); i < n; i++) {
            char c = s.charAt(i);
            if (c != ',') {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * purifier allows you to remove all white spaces in a given string,which it
     * receives as its argument
     *
     * @param h The string to free from white space.
     * @return a string devoid of all white space.
     */
    public static String purifier(String h) {
        if (h == null || h.isEmpty()) {
            return h == null ? null : "";
        }
        StringBuilder b = new StringBuilder(h.length());
        for (int i = 0, n = h.length(); i < n; i++) {
            char c = h.charAt(i);
            if (!Character.isWhitespace(c)) {
                b.append(c);
            }
        }
        return b.toString();
    }

    /**
     * Returns a string where characters from index `fromIndex` onward are
     * purified (spaces and newlines removed), while the prefix [0, fromIndex)
     * is preserved exactly.
     *
     * Semantics match the original: only ' ' and '\n' are removed.
     *
     * @param h The string to remove white space from.
     * @param fromIndex
     * @return a string from which white space has been removed starting from
     * index fromIndex.
     */
    public static String purifier(String h, int fromIndex) {
        if (h == null) {
            return null;
        }

        int n = h.length();
        if (fromIndex <= 0) {
            // Purify whole string
            return purifier(h);
        }
        if (fromIndex >= n) {
            // Nothing to purify; return original string
            return h;
        }

        StringBuilder out = new StringBuilder(n);
        // append prefix without creating a substring
        out.append(h, 0, fromIndex);

        // append purified suffix directly
        for (int i = fromIndex; i < n; i++) {
            char c = h.charAt(i);
            if (c != ' ' && c != '\n') {
                out.append(c);
            }
        }
        return out.toString();
    }

    /**
     * removeAll allows you to remove all occurrences of an unwanted string from
     * a given string, which it receives as its argument
     *
     * @param str The string to remove the unwanted substring from.
     * @param unwanted The unwanted substring.
     * @return a string from which all instance of unwanted has been removed.
     */
    public static String removeAll(String str, String unwanted) throws IndexOutOfBoundsException {
        if (unwanted.length() <= str.length()) {
            String[] splitted = splits(str, unwanted);
            String acc = "";
            for (int i = 0; i < splitted.length; i++) {
                acc += splitted[i];
            }

            return acc;
        }//end if
        throw new IndexOutOfBoundsException("Substring to be removed cannot be longer than the String "
                + (unwanted.length() > str.length()));
    }

    /**
     * Replace all literal occurrences of `replaced` in `str` with `replacing`.
     * This preserves the original behavior for occurrences at the start or end.
     *
     * @param str the input string (must not be null)
     * @param replaced the substring to replace (must not be null)
     * @param replacing the replacement string (must not be null)
     * @return a new string with all literal occurrences replaced
     * @throws NullPointerException if any argument is null
     * @throws IndexOutOfBoundsException if replaced is longer than str
     */
    public static String replaceAll(String str, String replaced, String replacing) {
        if (str == null || replaced == null || replacing == null) {
            throw new NullPointerException("Arguments must not be null");
        }

        int strLen = str.length();
        int repLen = replaced.length();

        if (repLen > strLen) {
            throw new IndexOutOfBoundsException(
                    "Substring to be removed cannot be longer than the String"
            );
        }

        // Special-case empty 'replaced' to avoid infinite loop (match at every index).
        if (repLen == 0) {
            return str;
        }

        StringBuilder sb = new StringBuilder(strLen + Math.max(0, replacing.length() - repLen) * 4);

        int from = 0;
        int idx = str.indexOf(replaced, from);
        while (idx != -1) {
            // append the chunk before the match
            sb.append(str, from, idx);
            // append replacement
            sb.append(replacing);
            // move past the matched substring
            from = idx + repLen;
            idx = str.indexOf(replaced, from);
        }
        // append the remainder
        sb.append(str, from, strLen);

        return sb.toString();
    }

    /**
     * method delete takes 2 arguments,the string to be modified and the index
     * of the character to be deleted
     *
     * @param u
     * @param i
     * @return the string after the character at index i has been removed
     * @throws StringIndexOutOfBoundsException
     */
    public static String delete(String u, int i) {
        // Basic safety check (replaces the expensive try-catch)
        if (u == null || i < 0 || i >= u.length()) {
            throw new StringIndexOutOfBoundsException("Index: " + i + ", Length: " + (u == null ? 0 : u.length()));
        }

        return new StringBuilder(u)
                .deleteCharAt(i)
                .toString();
    }

    /**
     * method delete takes 2 arguments,the string to be modified and the
     * substring to be removed from it.
     *
     * @param u the string to be modified
     * @param str the substring to be removed from u
     * @return the string after the first instance of the specified substring
     * has been removed
     * @throws StringIndexOutOfBoundsException
     */
    public static String delete(String u, String str) {
        if (u == null || str == null || str.isEmpty()) {
            return u;
        }
        int idx = u.indexOf(str);
        if (idx == -1) {
            return u;
        }
        StringBuilder sb = new StringBuilder(u);
        sb.delete(idx, idx + str.length());
        return sb.toString();
    }

    /**
     * method delete takes 3 arguments,the string to be modified, the index at
     * which deleting is to begin and the index one after the point where
     * deletion is to end. STRING.delete("Corinthian", 2,4)returns "Conthian"
     *
     * @param u
     * @param i
     * @param j
     * @return
     * @throws StringIndexOutOfBoundsException
     */
    public static String delete(String u, int i, int j) {
        if (u == null) {
            throw new NullPointerException("Input string is null");
        }
        if (i < 0 || j < i || j > u.length()) {
            throw new StringIndexOutOfBoundsException(
                    "Invalid range: i=" + i + ", j=" + j + ", length=" + u.length()
            );
        }
        StringBuilder sb = new StringBuilder(u);
        sb.delete(i, j); // delete characters in range [i, j)
        return sb.toString();
    }

    /**
     * method deleteCharsAfter takes 3 arguments,the string to be modified, the
     * index at which deleting is to begin and the index one after the point
     * where deletion is to end. STRING.delete("Corinthian", 2,4)returns
     * "Conthian"
     *
     * @param str..The original string
     * @param i..The index at which we start deleting
     * @param num The number of characters to delete after the index.
     * @return the resulting string after the chars between i and i+num-1 have
     * been deleted.
     * @throws StringIndexOutOfBoundsException
     */
    public static String deleteCharsAfter(String str, int i, int num) {
        if (str == null) {
            throw new NullPointerException("Input string is null");
        }
        if (i < 0 || num < 0 || i + num > str.length()) {
            throw new StringIndexOutOfBoundsException(
                    "Invalid range: i=" + i + ", num=" + num + ", length=" + str.length()
            );
        }

        StringBuilder sb = new StringBuilder(str);
        sb.delete(i, i + num); // delete characters in range [i, i+num)
        return sb.toString();
    }

    /**
     * Checks if a string can be parsed as a double.
     *
     * @param s the string to check
     * @return true if parsable as double, false otherwise
     * @throws NumberFormatException
     */
    public static boolean isDouble(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        try {
            Double.parseDouble(s); // avoids boxing
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * Checks if a string is a valid number component. Valid components: digits,
     * '.', 'E', '-', '+'
     */
    public static boolean isNumberComponent(String comp) {
        if (comp == null || comp.length() != 1) {
            return false;
        }
        char c = comp.charAt(0);
        return Character.isDigit(c) || c == '.' || c == 'E' || c == '-' || c == '+';
    }

    /**
     * This is an optimized specialized substitute for String.split in the Java
     * API. It runs about 10-20 times or more faster than the split method in
     * the Java API and it returns an ArrayList of values, instead of an array.
     *
     * Splits a string on all instances of the splitting object
     *
     * @param splittingObject The substring on which the string is to be split.
     * @param stringTosplit The string to be split.
     * @return an array containing substrings that the string has been split
     * into.
     */
    public static ArrayList<String> split(String stringTosplit, String splittingObject) throws ArrayIndexOutOfBoundsException {
        ArrayList<String> split = new ArrayList<String>();
        String[] splitted = splits(stringTosplit, splittingObject);
        split.addAll(Arrays.asList(splitted));

        return split;
    }//end method

    /**
     * This is an highly optimized specialized substitute for String.split in
     * the Java API. It runs about 40-100 times faster than the split method in
     * the Java API and it returns an array, too.
     *
     * Splits a string on all instances of the splitting object
     *
     * @param stringToSplit The string to be split.
     * @param splittingObject The substring on which the string is to be split.
     * @return an array containing substrings that the string has been split
     * into.
     */
    public static String[] splits(String stringToSplit, String splittingObject) {
        if (stringToSplit == null) {
            return new String[0];
        }
        if (splittingObject == null) {
            return new String[]{stringToSplit};
        }

        // Case 1: empty delimiter → split into characters
        if (splittingObject.isEmpty()) {
            int len = stringToSplit.length();
            String[] chars = new String[len];
            for (int i = 0; i < len; i++) {
                chars[i] = String.valueOf(stringToSplit.charAt(i));
            }
            return chars;
        }

        int len = stringToSplit.length();
        int delimLen = splittingObject.length();

        // Trim leading/trailing delimiter
        if (len >= delimLen && stringToSplit.startsWith(splittingObject)) {
            stringToSplit = stringToSplit.substring(delimLen);
            len = stringToSplit.length();
        }
        if (len >= delimLen && stringToSplit.endsWith(splittingObject)) {
            stringToSplit = stringToSplit.substring(0, len - delimLen);
            len = stringToSplit.length();
        }

        int firstIndex = stringToSplit.indexOf(splittingObject);
        if (delimLen < len && firstIndex != -1) {
            // Preallocate with max possible size
            String[] temp = new String[len];
            int index = 0;
            int lastIndex = 0;

            while (true) {
                int pos = stringToSplit.indexOf(splittingObject, lastIndex);
                if (pos == -1) {
                    // last chunk
                    temp[index++] = stringToSplit.substring(lastIndex);
                    break;
                }
                temp[index++] = stringToSplit.substring(lastIndex, pos);
                lastIndex = pos + delimLen;
            }

            // Copy to correctly sized array
            String[] result = new String[index];
            System.arraycopy(temp, 0, result, 0, index);
            return result;
        } else if (firstIndex == -1) {
            return new String[]{stringToSplit};
        } else {
            throw new ArrayIndexOutOfBoundsException(
                    stringToSplit + " must have more characters than " + splittingObject
            );
        }
    }

    /**
     * This is an highly optimized specialized substitute for String.split in
     * the Java API. It runs about 5-10 times faster than the split method in
     * the Java API and it returns an Array object, too. If the array of
     * splitting objects contains nothing or contains elements that are not
     * found in the string, then an Array object containing the original string
     * is returned. But if the array of splitting objects contains an empty
     * string, it chops or slices the string into pieces, and so returns an
     * Array object containing the individual characters of the string.
     *
     *
     * Splits a string on all instances of the splitting object
     *
     * @param stringTosplit The string to be split.
     * @param splittingObjects An array containing substrings on which the
     * string should be split.
     * @return an array containing substrings that the string has been split
     * into.
     * @throws NullPointerException if the array of splitting objects contains a
     * null value or is null.
     */
    public static String[] splits(String stringTosplit, String[] splittingObjects) throws NullPointerException {

        ArrayList<String> split = new ArrayList<String>();
        split.add(stringTosplit);
        for (int j = 0; j < splittingObjects.length; j++) {
            for (int i = 0; i < split.size(); i++) {
                ArrayList<String> sub = split(split.get(i), splittingObjects[j]);
                split.remove(i);
                split.addAll(i, sub);
                i += sub.size() - 1;
            }//end for
        }//end for
        /**
         * Remove any indiscriminate white spaces produced.
         */
        for (int i = 0; i < split.size(); i++) {
            if (split.get(i).equals("")) {
                split.remove(i);
                i--;
            }
        }
        return split.toArray(new String[]{});
    }//end method

    ///////////////////////////////////////////////////////////////////////////////
// Fast char-based digit check
public static boolean isDigitChar(char c) {
        return c >= '0' && c <= '9';
    }

    /**
     * boolean Method isDigit can be used to check for single digits i.e 0-9.
     *
     * @param a the String to check.
     * @return true if the string is a valid digit.
     */
    public static boolean isDigit(String s) {
        return s != null && s.length() == 1 && isDigitChar(s.charAt(0));
    }

    ////////////////////////////////////////////////////////////////////////////////



 
/**
* Method isDouble1 is specially built for parsing number strings in my parser.Please do not use it for your own purposes.
 * Its purpose is to detect already scanned number strings in a List object that contains many kinds of string objects
 * @param a
 * @return true if the string represents a number string( this in reality is true if the number has already being scanned and proven to be a valid number)
 */

public static boolean isDouble1(String a) {

        int L = a.length();
        boolean truth = false;
        int s = 1 / L;
        if (L > 3 && a.charAt(0) == 'E') {
            truth = false;
        } else if (L > 3 && STRING.isDigitChar(a.charAt(0))) {
            truth = true;
        } else if (L > 3 && a.charAt(0) == '.') {
            truth = true;
        } else if (L > 3 && a.charAt(0) == '-') {
            truth = true;
        }
        if (L == 3 && a.charAt(0) == 'E') {
            truth = false;
        } else if (L == 3 && a.charAt(0) == '.') {
            truth = true;
        } else if (L == 3 && a.charAt(0) == '-') {
            truth = true;
        } else if (L == 3 && STRING.isDigitChar(a.charAt(0))) {
            truth = true;
        }
        if (L == 2 && STRING.isDigitChar(a.charAt(0))) {
            truth = true;
        } else if (L == 2 && a.charAt(0) == '.') {
            truth = true;
        } else if (L == 2 && a.substring(1, 2).equals("-")) {
            truth = true;

        } else if (L == 2 && a.charAt(0) == 'E') {
            truth = false;
        } else if (L == 1 && a.charAt(0) == '-') {
            truth = false;
        } else if (L == 1 && STRING.isDigitChar(a.charAt(0))) {
            truth = true;
        } else if (L == 1 && a.charAt(0) == '.') {
            truth = false;
        } else if (L == 1 && a.charAt(0) == 'E') {
            truth = false;
        } else {
            truth = false;
        }

        return truth;
    }

    public static boolean isWord(String a) {
        if (a == null || a.isEmpty()) {
            return false;
        }

        for (int i = 0; i < a.length();) {
            int cp = a.codePointAt(i);
            if (!Character.isLetter(cp)) {
                return false;
            }
            i += Character.charCount(cp);
        }
        return true;
    }

    /**
     * method isLowerCaseWord takes one argument and that is the string whose
     * case we wish to know.
     *
     * @param s The string to manipulate.
     * @return true if the string is a lowercase letter
     */
    public static boolean isLowerCaseWord(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }

        for (int i = 0; i < s.length();) {
            int cp = s.codePointAt(i);
            if (!Character.isLowerCase(cp)) {
                return false;
            }
            i += Character.charCount(cp); // advance by 1 or 2 chars depending on surrogate pair
        }
        return true;
    }

    /**
     * method isUpperCaseWord takes one argument and that is the string whose
     * case we wish to know.
     *
     * @param s The string to manipulate.
     * @return true if the string is an uppercase letter
     */
    public static boolean isUpperCaseWord(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }

        for (int i = 0; i < s.length();) {
            int cp = s.codePointAt(i);
            if (!Character.isUpperCase(cp)) {
                return false;
            }
            i += Character.charCount(cp);
        }
        return true;
    }

    /**
     * method toUpperCase takes one argument only and that is the input string
     * to be converted to upper case
     *
     * @param m the single length string to be converted to upper case
     * @return the upper case version of the input
     * @throws StringIndexOutOfBoundsException
     */
    public static String toUpperCase(String m) {
        if (m == null || m.length() != 1) {
            throw new StringIndexOutOfBoundsException();
        }
        if (!STRING.isWord(m)) {
            throw new InputMismatchException();
        }

        char c = m.charAt(0);
        // If lowercase a–z, convert to uppercase
        if (c >= 'a' && c <= 'z') {
            c = (char) (c - ('a' - 'A'));
        }
        return String.valueOf(c);
    }

    /**
     * method toLowerCase takes one argument only and that is input string to be
     * converted to lower case
     *
     * @param m the single length string to be converted to lower case
     * @return the lower case version of the input
     * @throws StringIndexOutOfBoundsException
     */
    public static String toLowerCase(String m) {
        if (m == null || m.length() != 1) {
            throw new StringIndexOutOfBoundsException();
        }
        if (!STRING.isWord(m)) {
            throw new InputMismatchException();
        }

        char c = m.charAt(0);
        // If uppercase A–Z, convert to lowercase
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c + ('a' - 'A'));
        }
        return String.valueOf(c);
    }

    /**
     * method reverse takes 1 argument and that is the string to be reversed
     *
     * @param s The string to be reversed.
     * @return The reversed string.
     */
    public static String reverse(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        }

        char[] chars = s.toCharArray();
        int left = 0, right = chars.length - 1;
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
        return new String(chars);
    }

    /**
     * method replace takes 4 arguments,the string to be modified, the replacing
     * string, the index at which replacement is to begin and the index where
     * replacement ends plus one i.e replacement ends at index j-1.
     * STRING.replace("Corinthian","The" 3,6)returns "CorThehians"
     *
     * @param u The string to manipulate.
     * @param u1 The replacing string
     * @param i Index at which to begin replacing
     * @param j Index at which replacement ends plus one
     * @return A string in which the characters between index i and j-1 have
     * been replaced with the replacing string
     * @throws StringIndexOutOfBoundsException
     */
    public static String replace(String u, String u1, Integer i, Integer j) throws StringIndexOutOfBoundsException {
        u = u.substring(0, i) + u1 + u.substring(j);
        return u;
    }

    public static boolean isEven(String s) {
        if (s == null) {
            return false; // or throw IllegalArgumentException
        }
        return (s.length() & 1) == 0; // bitwise check for even
    }

    private static double convertdoubleDigitToString(String num) {
        double ans = -1;
        if (num.equals("0")) {
            ans = 0;
        } else if (num.equals("1")) {
            ans = 0;
        } else if (num.equals("2")) {
            ans = 3;
        } else if (num.equals("3")) {
            ans = 4;
        } else if (num.equals("4")) {
            ans = 4;
        } else if (num.equals("5")) {
            ans = 5;
        } else if (num.equals("6")) {
            ans = 6;
        } else if (num.equals("7")) {
            ans = 7;
        } else if (num.equals("8")) {
            ans = 8;
        } else if (num.equals("9")) {
            ans = 9;
        } else {
            throw new NoSuchElementException("Only 0 through 9 can be converted");
        }
        return ans;
    }

    private static String convertdoubleDigitToString(double a) {
        String ans = "";
        if (a == 0) {
            ans = "0";
        } else if (a == 1) {
            ans = "1";
        } else if (a == 2) {
            ans = "2";
        } else if (a == 3) {
            ans = "3";
        } else if (a == 4) {
            ans = "4";
        } else if (a == 5) {
            ans = "5";
        } else if (a == 6) {
            ans = "6";
        } else if (a == 7) {
            ans = "7";
        } else if (a == 8) {
            ans = "8";
        } else if (a == 9) {
            ans = "9";
        } else {
            throw new NoSuchElementException("Only 0 through 9 can be converted");
        }
        return ans;
    }
    //45.89===4589000000000000000 

    public static String doubleToString(double num) {
        // target: 17 significant digits in 1.xxxxx...E<exp> form
        if (Double.isNaN(num)) {
            return "NaN";
        }
        if (Double.isInfinite(num)) {
            return (num < 0 ? "-" : "") + "Infinity";
        }
        if (num == 0.0) {
            return "0.0E0";
        }

        boolean negative = num < 0;
        double absVal = Math.abs(num);

        int exp = 0;
        // normalize to [1,10)
        if (absVal >= 10.0) {
            while (absVal >= 10.0) {
                absVal *= 0.1;
                exp++;
            }
        } else if (absVal < 1.0) {
            while (absVal < 1.0) {
                absVal *= 10.0;
                exp--;
                // if it becomes zero due to denormals, break
                if (absVal == 0.0) {
                    break;
                }
            }
        }

        // collect 17 digits into a char array
        final int DIGITS = 17;
        char[] digits = new char[DIGITS];
        for (int i = 0; i < DIGITS; i++) {
            int d = (int) absVal;            // fast floor for positive values
            digits[i] = (char) ('0' + d);
            absVal = (absVal - d) * 10.0;
        }

        // rounding: look at next digit
        int nextDigit = (int) absVal;
        if (nextDigit >= 5) {
            // round up with carry
            for (int i = DIGITS - 1; i >= 0; i--) {
                if (digits[i] == '9') {
                    digits[i] = '0';
                } else {
                    digits[i]++;
                    break;
                }
                // if we reach i==0 and carried past '9', handle below
                if (i == 0) {
                    // carried past most significant digit -> make it "10..." and adjust exponent
                    // set digits to '1' followed by zeros
                    digits[0] = '1';
                    for (int j = 1; j < DIGITS; j++) {
                        digits[j] = '0';
                    }
                    exp++;
                }
            }
        }

        // build final string without substring allocations
        StringBuilder sb = new StringBuilder(24 + (negative ? 1 : 0));
        if (negative) {
            sb.append('-');
        }
        sb.append(digits[0]);
        sb.append('.');
        for (int i = 1; i < DIGITS; i++) {
            sb.append(digits[i]);
        }
        sb.append('E');
        sb.append(exp);

        return sb.toString();
    }

    /**
     *
     * @param s The String object to check.
     * @return true if it contains at least one non-white space characters.
     */
    public static boolean hasChars(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length();) {
            int cp = s.codePointAt(i);
            if (!Character.isWhitespace(cp)) {
                return true;
            }
            i += Character.charCount(cp);
        }
        return false;
    }

    /**
     * Checks if a given number is even
     *
     * @param a the number
     * @return true if a is even
     */
    public static boolean isEven(double a) {
        return (a % 2) == 0;
    }

    /**
     * Checks if a given number is even
     *
     * @param a the number
     * @return true if a is even
     */
    public static boolean isEven(float a) {
        return (a % 2) == 0;
    }

    /**
     * Checks if a given number is even
     *
     * @param a the number
     * @return true if a is even
     */
    public static boolean isEven(int a) {
        return (a % 2) == 0;
    }

    public static void main(String args[]) {//tester method for STRING methods

        String num = "3.187";
        double start = System.nanoTime();
        double N = 1000;
        boolean s = false;
        for (int i = 0; i < N; i++) {
            s = isFullyDouble(num);
        }
        double interval = (System.nanoTime() - start) / N;
        //System.out.println("soln: " + s + ", " + (interval / 1000) + " microns");
        System.out.println("soln: " + s + ", " + (interval / 1000) + " microns");
        start = System.nanoTime();
        for (int i = 0; i < N; i++) {
            s = isFullyDouble(num);
        }
        interval = (System.nanoTime() - start) / N;
        //System.out.println("soln: " + s + ", " + (interval / 1000) + " microns");
        System.out.println("soln: " + s + ", " + (interval / 1000) + " microns");

        /*
System.out.println( "Intent: TO REPLACE ALL OCCURRENCES\n"
        + "OF "+replaced+" IN "+str+"\n WITH "+replacing );
String aaa = replaceAll(str, replaced, replacing);
System.out.println( aaa );
         */
    }//end main
}//end class STRING

//Shohkhohlhohkhohbanhghohsheh
//A+B+356.523E-235+4C
//Nebuchadrezzar
