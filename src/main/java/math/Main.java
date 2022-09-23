package math;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import interfaces.Solvable;
import parser.LogicalExpression;
import parser.MathExpression;
import parser.cmd.ParserCmd;

public class Main {

    public static class MultiSwitch {
        private final String[] switches;

        public MultiSwitch(String... switches) {
            this.switches = switches;
        }

        public void removeFrom(List<String> l) {
            for (String s : switches) {
                l.remove(s);
            }
        }

        public boolean isIn(String l) {
            for (String s : switches) {
                if (l.equals(s)) {
                    return true;
                }
            }
            return false;
        }

        public boolean isContained(List<String> l) {
            for (String s : switches) {
                if (l.contains(s)) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int x = 0; x < switches.length; x++) {
                sb.append(switches[x]);
                if (x != switches.length - 1) {
                    sb.append("/");
                }
            }
            return sb.toString();
        }

        public String getSwitch(int i) {
            return switches[i];
        }
    }

    private static final MultiSwitch verboseSwitch = new MultiSwitch("-v", "-V", "--verbose");
    private static final MultiSwitch trimSwitch = new MultiSwitch("-t", "-T", "--trim");
    private static final MultiSwitch helpSwitch = new MultiSwitch("-h", "-H", "--help");
    private static final MultiSwitch interactiveSwitch = new MultiSwitch("-i", "-I", "--interactive");
    private static final MultiSwitch logcalSwitch = new MultiSwitch("-l", "-L", "--logic");

    private static boolean trim = false;
    private static boolean verbose = false;
    private static boolean logic = false;

    public static void main(String... args) throws IOException {
        List<String> aargs = new ArrayList<>(Arrays.asList(args));
        if (verboseSwitch.isContained(aargs)) {
            //todo pass, to debug in ParserNG engine
            verbose = true;
            verboseSwitch.removeFrom(aargs);
        }
        if (trimSwitch.isContained(aargs)) {
            trim = true;
            trimSwitch.removeFrom(aargs);
        }
        if (logcalSwitch.isContained(aargs)) {
            logic = true;
            logcalSwitch.removeFrom(aargs);
        }
        if (helpSwitch.isContained(aargs)) {
            help();
            if (isVerbose()) {
                examples();
            }
        } else if (interactiveSwitch.isContained(aargs)) {
            interactiveSwitch.removeFrom(aargs);
            if (aargs.size() > 0) {
                System.err.println(interactiveSwitch.toString() + " is interactive mode, commandline expression omitted");
            }
            ParserCmd.main(new String[0]);
        } else {
            String[] exs = joinArgs(aargs, trim).split("\n");
            for (String ex : exs) {
                if (verbose) {
                    System.out.flush();
                    System.err.println(ex);
                    System.err.flush();
                }
                String r = null;
                try {
                    Solvable exp = logic ? new LogicalExpression(ex, LogicalExpression.verboseStderrLogger) : new MathExpression(ex);
                    r  = exp.solve();
                }catch(Exception fatal){
                    if (verbose){
                        throw fatal;
                    } else {
                        r = fatal.getMessage();
                    }
                }
                System.out.println(r);
            }
        }
    }

    static void help() {
        System.out.println("  ParserNG " + getVersion() + " " + Main.class.getName());
        System.out.println(helpSwitch + "         this text; do not change for help (witout dashes), which lists functions");
        System.out.println(verboseSwitch + "      output is reprinted to stderr with some inter-steps");
        System.out.println(logcalSwitch + "        will add logical expression wrapper around the expression");
        System.out.println("                     Logical expression parser is much less evolved and slow. Do not use it if you don't must");
        System.out.println("                     If you use logical parse, result is always true/false. If it is not understood, it reuslts to false");
        System.out.println(trimSwitch + "         by default, each line is one expression,");
        System.out.println("                     however for better redability, sometimes it is worthy to");
        System.out.println("                     to split the expression to multiple lines. and evaluate as one.");
        System.out.println(interactiveSwitch + "  instead of evaluating any input, interactive prompt is opened");
        System.out.println("                     If you lunch interactive mode wit TRIM, the expression is");
        System.out.println("                     evaluated once you exit (done, quit, exit...)");
        System.out.println("                     it is the same as launching " + ParserCmd.class.getName() + " main class");
        System.out.println("           To read stdin, you have to set INTERACTIVE mode on");
        System.out.println("           To list all known functions,  type `help` as MathExpression");
        System.out.println("  Without any parameter, input is considered as math expression and calculated");
        System.out.println("  without trim, it would be the same as launching " + MathExpression.class.getName() + " main class");
        System.out.println("  run help in verbose mode (" + helpSwitch.getSwitch(0) + " " + verboseSwitch.getSwitch(0) + ") to get examples");
    }

    static void examples() {
        System.out.println("Examples:");
        System.out.println("  java -jar parser-ng-" + getVersion() + ".jar -h");
        System.out.println("    this help");
        System.out.println("  java -jar parser-ng-" + getVersion() + ".jar 1+1");
        System.out.println("    2.0");
        System.out.println("  java -jar parser-ng-" + getVersion() + ".jar \"1+1\n"
                + "                                 +2+2\"");
        System.out.println("    2.0");
        System.out.println("    4.0");
        System.out.println("  java -jar parser-ng-" + getVersion() + ".jar -t \"1+1\n"
                + "                                    +2+2\"");
        System.out.println("    6.0");
        System.out.println("  java -jar parser-ng-" + getVersion() + ".jar -i  1+1");
        System.out.println("    nothing, will expect manual output, and calculate line by line");
        System.out.println("  java -jar parser-ng-" + getVersion() + ".jar -i -t  1+1");
        System.out.println("    nothing, will expect manual output and calcualte it all as one expression");
        System.out.println("  echo 2+2 | java -jar parser-ng-" + getVersion() + ".jar  1+1");
        System.out.println("    2.0");
        System.out.println("  echo \"1+1 \n"
                + "        +2+2 | java -jar parser-ng-" + getVersion() + ".jar -i");
        System.out.println("    2.0");
        System.out.println("    4.0");
        System.out.println("  echo \"1+1 \n"
                + "        +2+2 | java -jar parser-ng-" + getVersion() + ".jar -i -t");
        System.out.println("    6.0");
        System.out.println("  java -cp parser-ng-" + getVersion() + ".jar " + ParserCmd.class.getName() + " \"1+1");
        System.out.println("    will ask for manual imput en evaluate per line");
        System.out.println("  echo \"1+1 \n"
                + "        +2+2 | java -cp parser-ng-" + getVersion() + ".jar " + ParserCmd.class.getName() + " 2>/dev/null");
        System.out.println("    2.0");
        System.out.println("    4.0");
        System.out.println("  java -cp parser-ng-" + getVersion() + ".jar " + MathExpression.class.getName() + " \"1+1\n"
                + "                                                      +2+2\"");
        System.out.println("    6.0");
        System.out.println("  java -cp parser-ng-" + getVersion() + ".jar " + LogicalExpression.class.getName() + " \"true or false\"");
        System.out.println("    true");
        System.out.println("  Note, that " + MathExpression.class.getName() + " nor " + LogicalExpression.class.getName() + " classes do not take any parameters except expressions");
        System.out.println("  Note, that " + ParserCmd.class.getName() + " class takes single parameter " + logcalSwitch + " to contorl its evaluation");
        System.out.println(LogicalExpression.getHelp());
    }

    public static String joinArgs(List<String> filteredArgs, boolean trim) {
        StringBuilder sb = new StringBuilder();
        for (String arg : filteredArgs) {
            if (trim) {
                sb.append(arg.replace("\n", " "));
            } else {
                sb.append(arg);
            }
            sb.append(" ");
        }
        return sb.toString();
    }

    public static boolean isTrim() {
        return trim;
    }

    public static boolean isVerbose() {
        return verbose;
    }

    public static boolean isLogic() {
        return logic;
    }

    public static void setLogic(boolean logic) {
        Main.logic = logic;
    }

    public static String getVersion() {
        //todo, read from pom. See JRD how we did it
        return "0.1.8";
    }

    public static MultiSwitch getLogcalSwitch() {
        return logcalSwitch;
    }
}
