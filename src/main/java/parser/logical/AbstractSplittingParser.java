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

    public AbstractSplittingParser(String expression, ExpressionLogger log) {
        this.log = log;
        pattern1 = toPattern(getPrimaryChars());
        if (getSecondaryChars().length > 0) {
            pattern2 = toPattern(getSecondaryChars());
        } else {
            pattern2 = null;
        }
        this.original = expression;
        split = split(original);
    }

    private static Pattern toPattern(String[] chars) {
        return Pattern.compile("\\s*(" + joinAsOr1n(escape(chars)) + ")\\s*");
    }

    List<String> split(String expression) {
        List<String> r = new ArrayList();
        while (true) {
            Matcher m = pattern1.matcher(expression);
            String[] chars = getPrimaryChars();
            boolean found = m.find();
            if (!found && getSecondaryChars().length > 0) {
                m = pattern2.matcher(expression);
                chars = getSecondaryChars();
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
            r.add(part);
            r.add(sanitizedGroup);
            expression = expression.substring(end);
        }
        return r;
    }

    private static String joinAsOr1n(String[] chars) {
        return Arrays.stream(chars).collect(Collectors.joining("+|")) + "+";
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
     * @return strings which are substituted first
     */
    public abstract String[] getPrimaryChars();

    /**
     * Primary characters are processed first. Seondary second.
     * The reason is, if some char is substring of another. Then first msut go the longer ones (which may contain secondary as substring)
     * the goes secondary. Yah. it can be done bette.. by sorting by lenght and so on... but maybe next tim
     *
     * @return characters which are subsituted second
     */
    public abstract String[] getSecondaryChars();

    public String getHelp() {
        return this.getName() + ": " +
                Arrays.stream(getPrimaryChars()).collect(Collectors.joining(", ")) + ", " +
                Arrays.stream(getSecondaryChars()).collect(Collectors.joining(", "));
    }

    public abstract String getName();
}
