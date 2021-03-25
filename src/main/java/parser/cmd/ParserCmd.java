/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package parser.cmd;

import java.util.Scanner;
import parser.MathExpression;

/**
 *
 * @author GBEMIRO JIBOYE <gbenroscience@gmail.com>
 */
public class ParserCmd {
    
    static MathExpression expression;
    public static void main(String[] args) {
        
        Scanner sc = new Scanner(System.in);
        
        System.out.println("Welcome To ParserNG Command Line");
        String divider = "\n______________________________________________________\n";
        
        int i = 0;
        
        while(true){
            System.out.printf("\nQuestion %d:%s", (++i), divider);
            String cmd = sc.nextLine();
             expression = new MathExpression(cmd);
             String ans = expression.solve();
             System.out.printf("Answer%s%s\n",divider, ans);
            
        }
        
        
        
        
    }
    
    
    
    
}
