/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parser.cmd;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import interfaces.Solvable;
import math.Main;
import parser.ExpandingExpression;
import parser.LogicalExpression;
import parser.MathExpression;
import parser.logical.ExpressionLogger;

/**
 * @author GBEMIRO JIBOYE <gbenroscience@gmail.com>
 */
public class ParserCmd {

    private static final String divider = "\n______________________________________________________\n";

    public static void main(String[] args) throws IOException {

        if (args.length > 0) {
            if (Main.getLogcalSwitch().isIn(args[0])) {
                Main.setLogic(true);
            }
            if (Main.getExpandableSwitch().isIn(args[0])) {
                Main.setExpandable(true);
            }
        }
        BufferedReader sc = new BufferedReader(new InputStreamReader(System.in));

        System.err.println("Welcome To ParserNG Command Line");
        String title = Main.isTrim() ? Main.isLogic() ? "Logical Line" : "Math Line" : Main.isLogic() ? "Logical Question" : "Math Question";
        int i = 0;
        List<String> batch = new ArrayList<>();
        while (true) {
            System.err.printf("\n" + title + " %d:%s", (++i), divider);
            String cmd = sc.readLine();
            if (cmd == null || cmd.equals("exit") || cmd.equals("quit") || cmd.equals("done")) {
                if (Main.isTrim()) {
                    String all = Main.joinArgs(batch, true);
                    if (Main.isVerbose()) {
                        System.err.println(all);
                    }
                    calWitOutput(all);
                }
                break;
            }
            batch.add(cmd);
            if (Main.isVerbose()) {
                System.err.println(cmd);
            }
            System.err.flush();
            if (!Main.isTrim()) {
                calWitOutput(cmd);
            }

        }


    }

    private static void calWitOutput(String cmd) {
        String ans = null;
        try {

            Solvable expression = Main.isLogic() ?
                    new LogicalExpression(cmd, LogicalExpression.verboseStderrLogger) :
                    Main.isExpandable() ?
                            new ExpandingExpression(cmd, ExpandingExpression.getValuesFromVariables(), ExpandingExpression.verboseStderrLogger) :
                            new MathExpression(cmd);
            ans = expression.solve();
        }catch (Exception ex) {
            ans = ex.getMessage();
        }
        System.err.printf("Answer%s\n", divider);
        System.err.flush();
        System.out.println(ans);
    }


}
