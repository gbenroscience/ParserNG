package com.github.gbenroscience.parser.expanding;

import com.github.gbenroscience.parser.LogicalExpression;
import com.github.gbenroscience.parser.logical.ComparingExpressionParser;
import com.github.gbenroscience.parser.logical.ExpressionLogger;
import com.github.gbenroscience.parser.logical.LogicalExpressionMemberFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class  ExpandingExpressionParser {

    private static final int MAX = 100;
    private static final Pattern downRange = Pattern.compile("\\.\\.L\\d+");
    private static final Pattern upRange = Pattern.compile("L\\d+\\.\\.");
    private static final Pattern bothRange = Pattern.compile("L\\d+\\.\\.L\\d+");
    private final String originalExpression;
    private final List<String> points;
    private final LogicalExpression logicalExpressionParser;
    private final LogicalExpressionMemberFactory logicalExpressionMemberFactory;
    private final ExpressionLogger log;
    private final String expanded;

    public ExpandingExpressionParser(String expression, List<String> points, ExpressionLogger log) {
        this(expression, points, log, true, new ComparingExpressionParser.ComparingExpressionParserFactory());
    }

    public ExpandingExpressionParser(String expression, List<String> points, ExpressionLogger log, LogicalExpressionMemberFactory logicalExpressionMemberFactory) {
        this(expression, points, log, true, logicalExpressionMemberFactory);
    }

    private ExpandingExpressionParser(String expression, List<String> points, ExpressionLogger log, boolean details, LogicalExpressionMemberFactory logicalExpressionMemberFactory) {
        this.log = log;
        this.points = points;
        this.originalExpression = expression;
        this.logicalExpressionMemberFactory = logicalExpressionMemberFactory;
        log.log("Expression : " + originalExpression.replaceAll("\\s*L\\s*", "L"));
        List<String> naturalisedOrder = new ArrayList<>(points);
        Collections.reverse(naturalisedOrder);
        if (details) {
            log.log("Upon       : " + naturalisedOrder.stream().collect(Collectors.joining(",")));
            log.log("As         : Ln...L1,L0");
            log.log("MN         = " + points.size());
        }
        expanded = expandALL(expression);
        log.log("Expanded as: " + expanded);
        logicalExpressionParser = new LogicalExpression(expanded, new ExpressionLogger.InheritingExpressionLogger(log), logicalExpressionMemberFactory);
    }

    String expandALL(String expression) {
        //order metters!
        String expanded = expression;
        expanded = expandMN(expression);
        expanded = expandCurlyIndexes(expanded);
        expanded = expanded.replaceAll("L\\s*", "L");
        expanded = expanded.replaceAll("\\s*\\.\\.\\s*", "..");
        expanded = expandLL(expanded);
        expanded = expandLd(expanded);
        expanded = expandLu(expanded);
        if (points.size() > 0) {
            //maybe better to throw and die?
            expanded = expandL(expanded);
        } else {
            log.log("Warning! no points in input!");
        }
        return expanded;
    }

    String expandCurlyIndexes(String expression) {
        return evalBrackets(expression, new ExpressionLogger.InheritingExpressionLogger(log), new int[]{0});
    }

    String expandMN(String expression) {
        return expression.replace("MN", "" + points.size());
    }

    String expandLL(String expression) {
        while (true) {
            Matcher m = bothRange.matcher(expression);
            boolean found = m.find();
            if (!found) {
                break;
            }
            String group = m.group();
            expression = expression.replace(group, createRange(group.replace("L", "").replaceAll("\\.\\..*", ""), group.replace("L", "").replaceAll(".*\\.\\.", "")));
        }
        return expression;
    }

    String expandLd(String expression) {
        while (true) {
            Matcher m = downRange.matcher(expression);
            boolean found = m.find();
            if (!found) {
                break;
            }
            String group = m.group();
            expression = expression.replace(group, createRange((points.size() - 1) + "", group.replace("..L", "")));
        }
        return expression;
    }

    String expandLu(String expression) {
        while (true) {
            Matcher m = upRange.matcher(expression);
            boolean found = m.find();
            if (!found) {
                break;
            }
            String group = m.group();
            expression = expression.replace(group, createRange(group.replace("L", "").replaceAll("\\.\\..*", ""), "0"));
        }
        return expression;
    }

    String expandL(String expression) {
        for (int x = Math.min(9, MAX); x >= 0; x--) {
            expression = expression.replace("L0" + x, points.get(limit(x)));
        }
        for (int x = MAX; x >= 0; x--) {
            expression = expression.replace("L" + x, points.get(limit(x)));
        }
        return expression;
    }

    private String createRange(String from, String to) {
        return createRange(Integer.parseInt(from), Integer.parseInt(to));
    }

    private String createRange(int from, int to) {
        from = limit(from);
        to = limit(to);
        int ffrom = Math.min(from, to);
        boolean revert = false;
        if (ffrom != from) {
            revert = true;
        }
        int tto = Math.max(from, to);
        if (tto >= points.size()) {
            tto = points.size() - 1;
        }
        List<String> l = new ArrayList(tto - ffrom + 5);
        for (int i = ffrom; i <= tto; i++) {
            l.add(points.get(i));
        }
        if (revert) {
            Collections.reverse(l);
        }
        return l.stream().collect(Collectors.joining(","));
    }

    private int limit(int i) {
        if (i < 0) {
            return 0;
        }
        if (i >= points.size()) {
            return points.size() - 1;
        }
        return i;
    }

    public String solve() {
        String rs = logicalExpressionParser.solve();
        log.log("is: " + rs);
        return rs;
    }

    public boolean evaluate() {
        Boolean r = Boolean.parseBoolean(solve());
        return r;
    }

    public String getExpanded() {
        return expanded;
    }


    private String evalBrackets(String ex, ExpressionLogger logger, int[] depth) {
        depth[0] = depth[0] + 1;
        if (ex.contains("{")) {
            logger.log("L indexes brackets: " + ex);
        }
        for (int x = 0; x < ex.length(); x++) {
            if (ex.charAt(x) == '{') {
                for (int z = x - 1; z >= 0; z--) {
                    if (ex.charAt(z) != ' ' && ex.charAt(z) != '\n' && ex.charAt(z) != '\t') {
                        break;
                    }
                }
                int c = 1;
                for (int y = x + 1; y < ex.length(); y++) {
                    if (ex.charAt(y) == '{') {
                        c++;
                    }
                    if (ex.charAt(y) == '}') {
                        c--;
                        if (c == 0) {
                            String s = ex.substring(x + 1, y);
                            String eval = null;
                            if (s.contains("}")) {
                                eval = evalBrackets(s, new ExpressionLogger.InheritingExpressionLogger(logger), depth);
                            } else {
                                eval = evalDirect(s, new ExpressionLogger.InheritingExpressionLogger(logger));
                            }
                            String s1 = ex.substring(0, x);
                            String s2 = ex.substring(y + 1);
                            ex = s1 + " " + eval + " " + s2;
                            logger.log("to: " + ex);
                            break;
                        }
                    }
                }
            }
        }
        depth[0] = depth[0] - 1;
        if (depth[0] > 0) {
            String r = evalDirect(ex, new ExpressionLogger.InheritingExpressionLogger(logger));
            return r;
        } else {
            return ex;
        }

    }

    private String evalDirect(String s, ExpressionLogger logger) {
        ExpandingExpressionParser lex = new ExpandingExpressionParser(s, points, logger, false, logicalExpressionMemberFactory);
        String r = lex.solve();
        int rr = Double.valueOf(r).intValue();//L2.0 is not valid value, but L2 is
        logger.log(s + " = " + rr + " (" + r + ")");
        return "" + rr;
    }
}
