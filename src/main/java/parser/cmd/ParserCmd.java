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
import java.util.Scanner;

import interfaces.Solvable;
import math.Main;
import parser.MathExpression;

/**
 * @author GBEMIRO JIBOYE <gbenroscience@gmail.com>
 */
public class ParserCmd {

    private static final String divider = "\n______________________________________________________\n";

    public static void main(String[] args) throws IOException {

        BufferedReader sc = new BufferedReader(new InputStreamReader(System.in));

        System.err.println("Welcome To ParserNG Command Line");
        String title = Main.isTrim() ? "Line" : "Question";
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
        Solvable expression = new MathExpression(cmd);
        String ans = expression.solve();
        System.err.printf("Answer%s\n", divider);
        System.err.flush();
        System.out.println(ans);
    }


}
