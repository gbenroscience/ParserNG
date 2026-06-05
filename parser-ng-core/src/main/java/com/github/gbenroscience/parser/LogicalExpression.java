package com.github.gbenroscience.parser;

import com.github.gbenroscience.interfaces.Solvable;
import com.github.gbenroscience.math.Main;
import com.github.gbenroscience.parser.logical.ComparingExpressionParser;
import com.github.gbenroscience.parser.logical.ExpressionLogger;
import com.github.gbenroscience.parser.logical.LogicalExpressionMemberFactory;
import com.github.gbenroscience.parser.logical.LogicalExpressionMemberFactory.LogicalExpressionMember;
import com.github.gbenroscience.parser.logical.LogicalExpressionParser;
import com.github.gbenroscience.parser.methods.Declarations;
import com.github.gbenroscience.util.VariableManager;

import java.util.Arrays;

public class LogicalExpression implements Solvable {

    private final String originalExpression;
    private final ExpressionLogger mainLogger;
    private final LogicalExpressionMemberFactory logicalExpressionMemberFactory;

    public static final ExpressionLogger verboseStderrLogger = new ExpressionLogger() {
        @Override
        public void log(String s) {
            if (Main.isVerbose()) {
                System.err.println(s);
            }
        }
    };

    public LogicalExpression(String s, ExpressionLogger logger) {
        this(s, logger, new ComparingExpressionParser.ComparingExpressionParserFactory());
    }

    public LogicalExpression(String s, ExpressionLogger logger, LogicalExpressionMemberFactory logicalExpressionMemberFactory) {
        this.originalExpression = s;
        this.mainLogger = logger;
        this.logicalExpressionMemberFactory = logicalExpressionMemberFactory;
        String varInitStatements[] = ExpressionInitParser.extractInitializers(s);
        for(String stmt : varInitStatements){
            Variable v = new Variable(stmt);
            VariableManager.add(v);
        }
    }

    public static void main(String args[]) {
        String in = Main.joinArgs(Arrays.asList(args), true);
        verboseStderrLogger.log(in);
        System.out.println(new LogicalExpression(in, verboseStderrLogger).solve());
        
        quickTestForInitializerExtractor(args);
    }//end method

    @Override
    public String solve() {
        if (originalExpression.trim().equalsIgnoreCase(Declarations.HELP)) {
            return getHelp();
        }
        if (new LogicalExpressionParser("", ExpressionLogger.DEV_NULL, logicalExpressionMemberFactory).isLogicalExpressionMember(originalExpression)) {
            return evalBrackets(originalExpression, mainLogger);
        } else {
            return new MathExpression(originalExpression).solve();
        }
    }

    private String evalBrackets(String ex, ExpressionLogger logger) {
        logger.log("brackets: " + ex);
        for (int x = 0; x < ex.length(); x++) {
            if (ex.charAt(x) == '[') {
                int neg = -1;
                int steps = 0;
                for (int z = x - 1; z >= 0; z--) {
                    steps++;
                    if (ex.charAt(z) == '!') {
                        neg = steps;
                        break;
                    }
                    if (ex.charAt(z) != ' ' && ex.charAt(z) != '\n' && ex.charAt(z) != '\t') {
                        break;
                    }
                }
                int c = 1;
                for (int y = x + 1; y < ex.length(); y++) {
                    if (ex.charAt(y) == '[') {
                        c++;
                    }
                    if (ex.charAt(y) == ']') {
                        c--;
                        if (c == 0) {
                            String s = ex.substring(x + 1, y);
                            String eval = null;
                            if (s.contains("[")) {
                                eval = evalBrackets(s, new ExpressionLogger.InheritingExpressionLogger(logger));
                            } else {
                                eval = evalDirect(s, new ExpressionLogger.InheritingExpressionLogger(logger));
                            }
                            if (neg >= 0) {
                                x = x - neg;//!!!!
                                ExpressionLogger tmpl = new ExpressionLogger.InheritingExpressionLogger(logger);
                                tmpl.log("!" + eval);
                                boolean b = !ComparingExpressionParser.parseBooleanStrict(eval.trim(), logger);
                                tmpl.log("..." + b);
                                eval = "" + b;
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
        String r = evalDirect(ex, new ExpressionLogger.InheritingExpressionLogger(logger));
        logger.log(r);
        return r;
    }

    private String evalDirect(String s, ExpressionLogger logger) {
        LogicalExpressionParser lex = new LogicalExpressionParser(s, new ExpressionLogger.InheritingExpressionLogger(logger), logicalExpressionMemberFactory);
        return "" + lex.evaluate();
    }

    public String getHelp() {
        LogicalExpressionParser l = new LogicalExpressionParser(" 1 == 1", ExpressionLogger.DEV_NULL, logicalExpressionMemberFactory);
        return l.getSubexpressionFactory().createLogicalExpressionMember(" 1 == 1", ExpressionLogger.DEV_NULL).getHelp() + "\n"
                + l.getHelp() + "\n"
                + "As Mathematical parts are using () as brackets, Logical parts must be grouped by [] eg: " + "\n"
                + "1+1 < (2+0)*1 impl [ [5 == 6 || 33<(22-20)*2 ]xor [ [  5-3 < 2 or 7*(5+2)<=5 ] and 1+1 == 2]] eq [ true && false ]" + "\n"
                + "Note, that logical parsser supports only dual operators, so where true|false|true is valid, 1<2<3  is invalid!" + "\n"
                + "Thus:  [1<2]<3   is necessary and  even  [[true|false]|true]is recomeded to be used, For 1<2<3  exception is thrown. " + "\n"
                + "Single letter can logical operands can be used in row. So eg | have same meaning as ||. But also unluckily also eg < is same as <<" + "\n"
                + "Negation can be done by single ! strictly close attached to [; eg ![true]  is ... false. Some spaces like ! [ are actually ok too\n"
                + "Note, that variables works, but must be included in first evaluated expression. Which is obvious for \"r=3;r<r+1\"\n"
                + "But much less for [r=3;r<r+1 || [r<5]]\", which fails and must be declared as \"[r<r+1 || [r=3;r<5]]\" \n"
                + "To avoid this, you can declare all in first dummy expression: \"[r=3;r<1] || [r<r+1 || [r<5]]\" which ensure theirs allocation ahead of time and do not affect the rest\n"
                + "If you modify the variables, in the subseqet calls, results maybe funny. Use verbose mode to debug order";

    }
 
public static final class ExpressionInitParser {

    public static String[] extractInitializers(String expr) {
        if (expr == null || expr.isEmpty()) {
            return new String[0];
        }

        int len = expr.length();
        boolean hasOuterBrackets = expr.charAt(0) == '[' && expr.charAt(len - 1) == ']';
        
        int start = hasOuterBrackets ? 1 : 0;
        int end = hasOuterBrackets ? len - 1 : len;
        
        int targetBracketDepth = hasOuterBrackets ? 1 : 0;
        int bracketDepth = targetBracketDepth;
        int parenDepth = 0;
        
        int lastMarker = start;
        String[] results = new String[8]; // Start with a reasonable capacity
        int count = 0;

        for (int i = start; i < end; i++) {
            char c = expr.charAt(i);
            
            if (c == '[') {
                bracketDepth++;
            } else if (c == ']') {
                bracketDepth--;
            } else if (c == '(') {
                parenDepth++;
            } else if (c == ')') {
                parenDepth--;
            } else if (c == ';' && bracketDepth == targetBracketDepth && parenDepth == 0) {
                
                // 1. Isolate the segment (e.g., " a,b,c = 3.14 ")
                int s = lastMarker;
                int e = i;
                
                // Fast outer trim
                while (s < e && expr.charAt(s) <= ' ') s++;
                while (e > s && expr.charAt(e - 1) <= ' ') e--;
                
                if (s < e) {
                    // 2. Locate the '=' sign
                    int eqIdx = -1;
                    for (int k = s; k < e; k++) {
                        if (expr.charAt(k) == '=') {
                            eqIdx = k;
                            break;
                        }
                    }
                    
                    if (eqIdx != -1) {
                        // 3. Extract the Right-Hand Value
                        int vStart = eqIdx + 1;
                        int vEnd = e;
                        while (vStart < vEnd && expr.charAt(vStart) <= ' ') vStart++;
                        String valStr = expr.substring(vStart, vEnd);
                        
                        // 4. Sweep the Left-Hand Side for commas
                        int varStart = s;
                        for (int k = s; k <= eqIdx; k++) {
                            // Trigger extraction on comma OR the equals sign
                            if (k == eqIdx || expr.charAt(k) == ',') {
                                int vs = varStart;
                                int ve = k;
                                
                                // Fast inner trim for the variable name
                                while (vs < ve && expr.charAt(vs) <= ' ') vs++;
                                while (ve > vs && expr.charAt(ve - 1) <= ' ') ve--;
                                
                                // If we found a valid variable name, store it
                                if (vs < ve) {
                                    if (count == results.length) {
                                        String[] temp = new String[results.length * 2];
                                        System.arraycopy(results, 0, temp, 0, count);
                                        results = temp;
                                    }
                                    // Construct the new string directly
                                    results[count++] = expr.substring(vs, ve) + "=" + valStr;
                                }
                                varStart = k + 1; // Move pointer past the comma
                            }
                        }
                    } else {
                        // Fallback: No '=' found, just add the segment as-is
                        if (count == results.length) {
                            String[] temp = new String[results.length * 2];
                            System.arraycopy(results, 0, temp, 0, count);
                            results = temp;
                        }
                        results[count++] = expr.substring(s, e);
                    }
                }
                lastMarker = i + 1;
            }
        }

        if (count == 0) {
            return new String[0];
        }

        String[] finalResults = new String[count];
        System.arraycopy(results, 0, finalResults, 0, count);
        return finalResults;
    }

   
}
    
    // Quick test
        public static void quickTestForInitializerExtractor(String[] args) {
            String e1 = "[s=3;s<s+1 || [s<5]]";
            String e2 = "[s=3;m=-4;s>s-m || [s<5*m]]";
            String e3 = "[s<s+1 || [s<5]]";

            System.out.println(java.util.Arrays.toString(ExpressionInitParser.extractInitializers(e1))); // [s=3]
            System.out.println(java.util.Arrays.toString(ExpressionInitParser.extractInitializers(e2))); // [s=3, m=-4]
            System.out.println(java.util.Arrays.toString(ExpressionInitParser.extractInitializers(e3))); // []
            
              String test = "[s=3; m=-4; a, b, c, d = 3.14159; s>s-m || [s<5*m]]";
        String[] inits = ExpressionInitParser.extractInitializers(test);
        for (String init : inits) {
            System.out.println(init);
        }
        }
}
