package parser.logical;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public abstract class AbstractSplittingParser {

    private final Pattern pattern1;
    private final Pattern pattern2;
    private final String original;
    protected final List<String> split;
    protected final ExpressionLogger log;

    /**
     * Dummy constructor to make more easy overriding
     **/
    public AbstractSplittingParser() {
        pattern1 = Pattern.compile(".*");
        pattern2 = pattern1;
        original = pattern1.toString();
        split = new ArrayList<>(0);
        log = ExpressionLogger.DEV_NULL;
    }

    public AbstractSplittingParser(String expression, ExpressionLogger log) {
        this.log = log;
        pattern1 = toPattern(getPrimaryChars1(), getPrimaryChars2());
        if (getSecondaryChars1().length > 0 || getSecondaryChars2().length > 0) {
            pattern2 = toPattern(getSecondaryChars1(), getSecondaryChars2());
        } else {
            pattern2 = null;
        }
        this.original = expression;
        split = split(original);
    }

    protected Pattern toPattern(String[] chars1, String[] chars2) {
        String p1 = null;
        if (chars1.length > 0) {
            p1 = "\\s*(" + joinAsOr1n(escape(chars1)) + ")\\s*";
        }
        String p2 = null;
        if (chars2.length > 0) {
            p2 = "\\s+(" + joinAsOr1n(escape(chars2)) + ")\\s+";
        }
        if (p1 == null) {
            return Pattern.compile(p2);
        }
        if (p2 == null) {
            return Pattern.compile(p1);
        }
        return Pattern.compile("(" + p1 + ")|(" + p2 + ")");
    }

    List<String> split(String expression) {
        List<String> r = new ArrayList();
        while (true) {
            Matcher m = pattern1.matcher(expression);
            String[] chars = concatWithArrayCopy(getPrimaryChars1(), getPrimaryChars2());
            boolean found = m.find();
            if (!found && concatWithArrayCopy(getSecondaryChars1(), getSecondaryChars2()).length > 0) {
                m = pattern2.matcher(expression);
                chars = concatWithArrayCopy(getSecondaryChars1(), getSecondaryChars2());
                found = m.find();
            }
            if (!found) {
                r.add(expression);
                break;
            }
            String group = m.group();
            int start = m.start();
            int end = m.end();
            String part = expression.substring(0, start);
            String sanitizedGroup = group.trim();
            for (String ch : chars) {
                sanitizedGroup = sanitizedGroup.replaceAll("(" + escape(ch) + ")+", ch);
            }
            r.addAll(split(part));
            r.add(sanitizedGroup);
            expression = expression.substring(end);
        }
        return r;
    }

    private static String joinAsOr1n(String[] chars) {
        return Arrays.stream(chars).collect(Collectors.joining("+|")) + "+";
    }

    static String[] concatWithArrayCopy(String[] array1, String[] array2) {
        String[] result = Arrays.copyOf(array1, array1.length + array2.length);
        System.arraycopy(array2, 0, result, array1.length, array2.length);
        return result;
    }

    private static String[] escape(String[] chars) {
        String[] r = new String[chars.length];
        for (int i = 0; i < chars.length; i++) {
            r[i] = escape(chars[i]);
        }
        return r;
    }

    private static String escape(String s) {
        if ("&".equals(s) || "|".equals(s)) {
            return "\\" + s;
        } else {
            return s;
        }
    }

    public String getOriginal() {
        return original;
    }

    public abstract boolean evaluate();

    /**
     * Primary characters are processed first. Seondary second.
     * The reason is, if some char is substring of another. Then first msut go the longer ones (which may contain secondary as substring)
     * the goes secondary. Yah. it can be done bette.. by sorting by lenght and so on... but maybe next tim
     *
     * PrimaryChars1 are allowed without spaces
     *
     * @return strings which are substituted first
     */
    public abstract String[] getPrimaryChars1();

    /**
     * Primary characters are processed first. Seondary second.
     * The reason is, if some char is substring of another. Then first msut go the longer ones (which may contain secondary as substring)
     * the goes secondary. Yah. it can be done bette.. by sorting by lenght and so on... but maybe next tim
     *
     * PrimaryChars2 are NOT allowed without spaces
     *
     * @return strings which are substituted first
     */
    public abstract String[] getPrimaryChars2();

    /**
     * Primary characters are processed first. Seondary second.
     * The reason is, if some char is substring of another. Then first msut go the longer ones (which may contain secondary as substring)
     * the goes secondary. Yah. it can be done bette.. by sorting by lenght and so on... but maybe next tim
     *
     * SecondaryChars1 are allowed without spaces
     *
     * @return characters which are subsituted second
     */

    public abstract String[] getSecondaryChars1();

    /**
     * Primary characters are processed first. Seondary second.
     * The reason is, if some char is substring of another. Then first msut go the longer ones (which may contain secondary as substring)
     * the goes secondary. Yah. it can be done bette.. by sorting by lenght and so on... but maybe next tim
     *
     * SecondaryChars2 are NOT allowed without spaces
     *
     * @return characters which are subsituted second
     */
    public abstract String[] getSecondaryChars2();

    public String getHelp() {
        return this.getName() + " - " +
                "allowed with spaces:" + Arrays.stream(getPrimaryChars1()).collect(Collectors.joining(", ")) + ", " +
                "" + Arrays.stream(getSecondaryChars1()).collect(Collectors.joining(", ")) +
                "; not allowed with spaces:" + Arrays.stream(getPrimaryChars2()).collect(Collectors.joining(", ")) + ", " +
                "" + Arrays.stream(getSecondaryChars2()).collect(Collectors.joining(", "));
    }

    public abstract String getName();
}
