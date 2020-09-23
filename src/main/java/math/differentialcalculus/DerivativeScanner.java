/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import parser.Bracket;
import parser.CustomScanner;
import parser.Operator;
import parser.Variable;
import static parser.Operator.*;
import static math.differentialcalculus.Methods.*;
import parser.STRING;
import static parser.Variable.*;
import java.util.ArrayList;
/**
 *
 * @author GBEMIRO
 */
public class DerivativeScanner {
    
    String expression;
    private ArrayList<String> scanner;
    private boolean syntaxValid;

    public DerivativeScanner(String expression) throws Exception {

        expression = STRING.replaceAll(expression, "³√", "\\cbrt//_\\temp//");//capture and change all ³√ to cbrt_temp to avoid confusion with ³ below

        expression = STRING.replaceAll(expression, "²", "^2");
        expression = STRING.replaceAll(expression, "³", "^3");
        expression = STRING.replaceAll(expression, "+-", "-");
        expression = STRING.replaceAll(expression, "-+", "-");
        expression = STRING.replaceAll(expression, "--", "+");
        //Now change back cbrt_temp to ³√
        if(expression.contains("\\cbrt//_\\temp//")){
            expression = expression.replace("\\cbrt//_\\temp//", "³√");
        }


        for(int i=0;i<expression.length();i++) {
            try {
                if (expression.substring(i, i + 1).equals("π")) {
                    expression = STRING.replace(expression, "pi", i, i + 1);
                } else if (expression.substring(i, i + 1).equals("×")) {
                    expression = STRING.replace(expression, Operator.MULTIPLY, i, i + 1);
                } else if (expression.substring(i, i + 1).equals("÷")) {
                    expression = STRING.replace(expression, Operator.DIVIDE, i, i + 1);
                }
            }
catch (IndexOutOfBoundsException e){

}

        }
        StringBuilder expr = new StringBuilder(expression);

        for(int i=0;i<expr.length();i++){
if(expr.substring(i, i+1).equals("-")&&expr.substring(i+1, i+2).equals("-")){
    expr.replace(i, i+2,  "+");	
    i-=1;
        	}
if(expr.substring(i, i+1).equals("-")&&expr.substring(i+1, i+2).equals("+")){
	  expr.replace(i, i+2,  "-");		
	  i-=1;
}
if(expr.substring(i, i+1).equals("+")&&expr.substring(i+1, i+2).equals("+")){
	  expr.replace(i, i+2,  "+");		
	  i-=1;
}
if(expr.substring(i, i+1).equals("+")&&expr.substring(i+1, i+2).equals("-")){
	  expr.replace(i, i+2,  "-");		
	  i-=1;
}

        }
        expression = expr.toString();

        this.expression = expression;
        
        CustomScanner cs = new CustomScanner(expression, true, Methods.inbuiltMethods,Methods.inbuiltOperators);
        scanner = (ArrayList<String>) cs.scan();
        orderNumberTokens();

        syntaxAnalyzer();
    }
/**
 * 
 * @param token The token to examine.
 * @return true if it is found to be a number.
 */
public static boolean isNumber( String token){
    try{
        double val = Double.parseDouble(token);
        return true;
    }
    catch(Exception numErr){
        return false;
    }
}
/**
 * 
 * @param op The token to examine.
 * @return true if it a valid operator or method
 * in a differentiable function.
 */
private static boolean isValidOperatorOrMethod(String op){
    for(String value:Methods.inbuiltMethods){
        if(op.equals(value)){
            return true;
        }
    }
    for(String value:Methods.inbuiltOperators){
        if(op.equals(value)){
            return true;
        }
    }

return false;    
}//end method    


/**
 * 
 * @return true if the tokens found in the array are either of type
 * operator,variable,number or method...e.g sin,cos...
 * At this stage also, we disallow the user from using inbuilt method names in
 * user-defined variable names.
 * @throws Exception if an invalid token is found during scanning,
 */
    private boolean syntaxAnalyzer() throws Exception{
      int count=0;
      String token="";
        for(;count<scanner.size();count++){
        	token = scanner.get(count);
            if( !isNumber(token) && !Variable.isVariableString(token) &&!isValidOperatorOrMethod(token)   ){
                System.out.println("Invalid "+token+" found!");
                scanner.clear();
                throw new Exception("Syntax Error Found During Scanning");
            }
            if(count<scanner.size()-1&&isInbuiltMethodName(token)&&!isOpeningBracket(scanner.get(count+1))){
                System.out.println("Cannot use an inbuilt method name as a variable name.");
                scanner.clear();
                throw new Exception("Syntax Error Found During Scanning");            	
            }
        }
        return syntaxValid = true;
    }//end method

    public void setSyntaxValid(boolean syntaxValid) {
        this.syntaxValid = syntaxValid;
    }

    public boolean isSyntaxValid() {
        return syntaxValid;
    }

    public void setScanner(ArrayList<String> scanner) {
        this.scanner = scanner;
    }

    public ArrayList<String> getScanner() {
        return scanner;
    }

    public void setExpression(String expression) {
        this.expression = expression;
    }

    public String getExpression() {
        return expression;
    }
    
    
    /**
     * Scans the list and arranges scattered parts of a single number token.
     * CustomScanner is not built to leave numbers intact..but rather splits
     * according to the supplied tokens. So since - and + were supplied
     * as splitting tokens here, a number such as 3.2E9 would be left intact,
     * whereas 3.2E-9 would be split into 3.2E,-,9
     */
    private void orderNumberTokens(){
        int sz = scanner.size();
        
        for(int i=0;i<(sz=scanner.size());i++){
      
            if(i>0&&i<sz-1){
        String token = scanner.get(i);
             String prevToken = scanner.get(i-1);
             String nextToken = scanner.get(i+1);
              
            if(token.equals("+")||token.equals("-")){
                int len = prevToken.length();
             if(len>1 && isNumber(prevToken.substring(0,len-1))&& !isNumber(prevToken.substring(0,len))){  
                if(prevToken.substring(len-1).equalsIgnoreCase("E")&&isNumber(nextToken)){
                    StringBuilder sb =new StringBuilder(prevToken);
                    sb.append(token);
                    sb.append(nextToken);
                    scanner.set(i-1,sb.substring(0, sb.length()));
                    scanner.subList(i, i+2).clear();
                    i-=1;
                }//end if
            }//end if
            
            }//end if
            
        }//end if
            /**
             * Convert the pattern (-x...) to (-1*x....)
             */
            if(isOpeningBracket(scanner.get(i)) && scanner.get(i+1).equals("-") &&isVariableString(scanner.get(i+2))){
                scanner.set(i+1,"-1");
                scanner.add(i+2,"*");
            }
            //x,^,-,x convert to x,^,(,-1,*,x,)...or if x,^,-,sin,(,...,) convert to x,^,(,-1,*,sin,(,.....,),)
            else if(isPower(scanner.get(i))&&scanner.get(i+1).equals("-")&&isVariableString(scanner.get(i+2))){
                if(isOpeningBracket(scanner.get(i+3))){
                    int close = Bracket.getComplementIndex(true, i+3, scanner);
                    scanner.add(close,")");
                    scanner.add(i+2,"*");
                    scanner.set(i+1,"-1");
                    scanner.add(i+1,"(");
                }
                else{
                    scanner.add(i+3,")");
                    scanner.add(i+2,"*");
                    scanner.set(i+1,"-1");
                    scanner.add(i+1,"(");                    
                }
            }
          //...,^,-,2---becomes....,^,-2,
            else if(isPower(scanner.get(i))&&scanner.get(i+1).equals("-")&&isNumber(scanner.get(i+2))){
                    scanner.set(i+1,"-"+scanner.get(i+2));
                    scanner.remove(i+2);
            }
          //...,*|/,-,2---becomes....,*|/,-2,
            else if(isMulOrDiv(scanner.get(i))&&scanner.get(i+1).equals("-")&&isNumber(scanner.get(i+2))){
                    scanner.set(i+1,"-"+scanner.get(i+2));
                    scanner.remove(i+2);
            }
          //...,+|-,-,2---becomes....,+|-,-2,
            else if(isPlusOrMinus(scanner.get(i))&&scanner.get(i+1).equals("-")&&isNumber(scanner.get(i+2))){
            	if(scanner.get(i).equals("+")){
            		  scanner.set(i,"-");
                      scanner.remove(i+1);	
            	}
            	else{
            		  scanner.set(i,"+");
                      scanner.remove(i+1);	
            	}
                  
            }
            
                      
        }//end for loop
        /*
                for(int i=0;i<(sz=scanner.size());i++){
      
      //Convert the pattern (-x...) to (-1*x....) or *-x
            
            if(scanner.get(i).equals("-") &&isVariableString(scanner.get(i+1))){
                scanner.add(i+1,"*");
                scanner.add(i+1,"-1");
                scanner.remove(i);
            }
            
        }//end for loop
        
        
        */
        
    }//end method


    
    
public static void main(String[]args){
    try{
    String expression="1/x^-2.22+sin(myDiff_3)/(myDiff_2)*diff(myDiff_2)";
    DerivativeScanner ds = new DerivativeScanner(expression);
     System.out.println(ds.scanner);
    }//end try
    catch(Exception e){
        
    }
String a = "abcdratatouille";
a=a.replace("rat", "cat");

       System.out.println(a);
    
       
    
    
    
    
}//end main method
    
    
}
