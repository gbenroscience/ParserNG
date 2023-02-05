package parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import interfaces.Solvable;
import math.Main;
import parser.expanding.ExpandingExpressionParser;
import parser.logical.ComparingExpressionParser;
import parser.logical.ExpressionLogger;
import parser.logical.LogicalExpressionMemberFactory;
import parser.methods.Declarations;

public class ExpandingExpression implements Solvable {

    private final String originalExpression;
    private final List<String> points;
    private final ExpressionLogger mainLogger;
    private final LogicalExpressionMemberFactory logicalExpressionMemberFactory;
    public static final String VALUES_PNG = "VALUES_PNG";
    public static final String VALUES_IPNG = "VALUES_IPNG";

    public static final ExpressionLogger verboseStderrLogger = new ExpressionLogger() {
        @Override
        public void log(String s) {
            if (Main.isVerbose()) {
                System.err.println(s);
            }
        }
    };

    public ExpandingExpression(String s, List<String> points, ExpressionLogger logger) {
        this(s, points, logger, new ComparingExpressionParser.ComparingExpressionParserFactory());
    }

    public ExpandingExpression(String s, List<String> points, ExpressionLogger logger, LogicalExpressionMemberFactory logicalExpressionMemberFactory) {
        this.originalExpression = s;
        this.mainLogger = logger;
        this.points = points;
        this.logicalExpressionMemberFactory = logicalExpressionMemberFactory;
    }

    public static void main(String args[]) {
        String in = Main.joinArgs(Arrays.asList(args), true);
        verboseStderrLogger.log(in);
        if (in.trim().equalsIgnoreCase(Declarations.HELP)) {
            System.out.println(getHelp());
            return;
        }
        List<String> values = getValuesFromVariables();
        System.out.println(new ExpandingExpression(in, values, verboseStderrLogger).solve());
    }//end method

    public static List<String> getValuesFromVariables() {
        String values_png = System.getenv(VALUES_PNG);
        String values_ipng = System.getenv(VALUES_IPNG);
        if (values_png != null && values_ipng != null) {
            throw new RuntimeException("Both " + VALUES_PNG + " and " + VALUES_IPNG + " are declared. That is no go");
        }
        List<String> values;
        if (values_png != null) {
            if (values_png.trim().isEmpty()) {
                values = new ArrayList<>();
            } else {
                values = new ArrayList<>(Arrays.asList(values_png.split("\\s+")));
                Collections.reverse(values);
            }
        } else if (values_ipng != null) {
            if (values_ipng.trim().isEmpty()) {
                values = new ArrayList<>();
            } else {
                values = new ArrayList<>(Arrays.asList(values_ipng.split("\\s+")));
            }
        } else {
            throw new RuntimeException("None of " + VALUES_PNG + " or " + VALUES_IPNG + " declared. Try help");
        }
        if (values.isEmpty()) {
            verboseStderrLogger.log("Warning, " + VALUES_PNG + " or " + VALUES_IPNG + " declared, but are empty!");
        }
        return values;
    }

    @Override
    public String solve() {
        if (originalExpression.trim().equalsIgnoreCase(Declarations.HELP)) {
            return getHelp();
        }
        ExpandingExpressionParser eep = new ExpandingExpressionParser(originalExpression, points, mainLogger,  logicalExpressionMemberFactory);
        mainLogger.log(eep.getExpanded());
        String r = eep.solve();
        return r;
    }


    public static String getHelp() {
        //longterm todo, repalce this static help by better help using delegated help methods from logical parser
        return "This is abstraction which allows to set with slices, rows and subset of immutable known numbers." + "\n" +
                "Instead of numbers, you can use literalls L0, L1...L99, which you can then call by:" + "\n" +
                "Ln - vlaue of Nth number" + "\n" +
                "L2..L4 - will expand to values of L2,L3,L4 - order is hnoured" + "\n" +
                "L2.. - will expand to values of L2,L3,..Ln-1,Ln" + "\n" +
                "..L5 - will expand to values of  L0,L1...L4,L5" + "\n" +
                "When used as standalone, " + VALUES_PNG + " xor " + VALUES_IPNG + "  are used to pass in the space separated numbers (the I is inverted order)" + "\n" +
                "Assume " + VALUES_PNG + "='5 9 3 8', then it is the same as " + VALUES_IPNG + "='8 3 9 5'; BUt be aware, with I the L.. and ..L are a bit oposite then expected" + "\n" +
                "L0 then expand to 8; L2.. expands to 9,3,8; ' ..L2 expands to 5,9 " + "\n" +
                "L2..L4 expands to 9,5; L4..L2 expands to 5,9" + "\n" +
                "There is a special element MN, which represents count of input points, so you do not need to call `count(..L0)` arround and arround. So..." + "\n" +
                "Expression : avg(..L1)*1.1-MN <  L0 | L1*1.3 + MN<  L0 \n" +
                "Upon       : 60,20,80,70\n" +
                "As         : Ln...L1,L0\n" +
                "MN         = 4\n" +
                "Expanded as: avg(60,20,80)*1.1-4 <  70 | 80*1.3 + 4<  70\n" +
                "...indeed\n" +
                "Dynamic calculation of L's indexes\n" +
                "Sometimes, Lx, as number is not enough, and you need to calcualte it dynamically. To do so, you can use L{}\n" +
                "Inisde {} can be mathematical formula (including Ls, MN. or even nested {}, which will evaluate itself as number, which will be used as Lx. Eg:\n" +
                "'avg(..L{MN/2}) < avg(L{MN/2}..)' will go to 'avg(..L1) < avg(L1..)' will go on 1 2 3 to 'avg(1,2 ) < avg(2,3) will go to 1.5<2.5 ... true'\n" +
                "For fun try eg: VALUES_PNG='1 2 3' on 'avg(..L{L{MN/2}}) < avg(L{L{MN/2}}..)'\n" +
                "This parser by default uses LogicalExpression interpreter, but should work directly in" + "\n" +
                "In verbose mode, the expanded expression is always printed";

    }
}
