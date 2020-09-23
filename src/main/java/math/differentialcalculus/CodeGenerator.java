
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import parser.Bracket;
import static parser.Number.*;
import static parser.Operator.*;
import parser.STRING;
import static parser.Variable.*;

import static parser.methods.Method.*;
import java.util.ArrayList;
import java.util.List;

import static math.differentialcalculus.Utilities.*;

/**
 *
 * Objects of this class take the output of the semantic
 * analyzer and uses it to generate code that the derivative engine can
 * work with.
 *
 * @author GBEMIRO
 */
public class CodeGenerator {

  private ArrayList<String>scanner;
    /**
     * @param expression
     * @throws Exception
     */
    public CodeGenerator(String expression) throws Exception {
        try {
            SemanticAnalyzer semanticAnalyzer =
          new SemanticAnalyzer( (expression.startsWith("(")&&expression.endsWith(")"))?expression: "("+expression+")");
            scanner = semanticAnalyzer.getScanner();    
        } catch (Exception ex) {
            ex.printStackTrace();
            throw new Exception("Bad Input!---"+expression);
        }
    }//end constructor.
public ArrayList<String> getScanner() {
	return scanner;
}
   
/**
 * Coordinating method to generate math code almost ready for calculus.
 * @param scan An ArrayList object containing math tokens.
 */
    public static void simplifyTokens(ArrayList<String> scan) {

        openUpUnnecessaryBrackets(scan);
        bracketVariableProductsAndQuotients(scan);
        tokenRearranger(scan);
        quicksimplify(scan);
         
        for (int i = 0; i < scan.size(); i++) {
try{

if(isOpeningBracket(scan.get(i))){
       while(isOpeningBracket(scan.get(i))){
        ++i;
    }
       int start = i;
    Object[] obj = utilitySimplifier(scan, start);
      i= (int) Integer.parseInt(obj[0].toString());
    ArrayList<String>vars = (ArrayList<String>) obj[1];
    
      List<String>view = scan.subList(start, i);
  int size = view.size();

   if(size>0&&!vars.isEmpty()){
   boolean startsWithMul = view.get(0).equals("*");
   boolean startsWithDiv = view.get(0).equals("/");
   boolean endsWithMul =   view.get(size-1).equals("*");
   boolean endsWithDiv =   view.get(size-1).equals("/");
   
   
   view.clear();
   view.addAll(vars);
if(startsWithMul){
    view.add(0,"*");
}
else if(startsWithDiv){
    view.add(0,"/");
}
if(endsWithMul){
    view.add("*");
}
else if(endsWithDiv){
    view.add("/");
}


   }//end if
    
   i=start+vars.size()-1;
   
  
}//end if
else if(isPlusOrMinus(scan.get(i))){
++i;
       int start = i;
    Object[] obj = utilitySimplifier(scan, start);
      i= Integer.parseInt( obj[0].toString() );
    ArrayList<String>vars = (ArrayList<String>) obj[1];
    
      List<String>view = scan.subList(start, i);
  int size = view.size();

   if(size>0&&!vars.isEmpty()){
   boolean startsWithMul = view.get(0).equals("*");
   boolean startsWithDiv = view.get(0).equals("/");
   boolean endsWithMul =   view.get(size-1).equals("*");
   boolean endsWithDiv =   view.get(size-1).equals("/");
   
   
   view.clear();
   view.addAll(vars);
if(startsWithMul){
    view.add(0,"*");
}
else if(startsWithDiv){
    view.add(0,"/");
}
if(endsWithMul){
    view.add("*");
}
else if(endsWithDiv){
    view.add("/");
}


   }//end if
    
    
   i=start+vars.size()-1;
   
}
else if(isClosingBracket(scan.get(i))){
    int siz = scan.size();
    while(i<siz&&isClosingBracket(scan.get(i))){
        ++i;
    }
    if(isMulOrDiv(scan.get(i))||isPlusOrMinus(scan.get(i))){
       int start = i;
    Object[] obj = utilitySimplifier(scan, start+1);
      i= (int) Integer.parseInt(obj[0].toString());
    ArrayList<String>vars = (ArrayList<String>) obj[1];
     
      List<String>view = scan.subList(start+1, i);
  int size = view.size();

   if(size>0&&!vars.isEmpty()){
   boolean startsWithMul = view.get(0).equals("*");
   boolean startsWithDiv = view.get(0).equals("/");
   boolean endsWithMul =   view.get(size-1).equals("*");
   boolean endsWithDiv =   view.get(size-1).equals("/");
   
   
   view.clear();
   view.addAll(vars);
if(startsWithMul){
    view.add(0,"*");
}
else if(startsWithDiv){
    view.add(0,"/");
}
if(endsWithMul){
    view.add("*");
}
else if(endsWithDiv){
    view.add("/");
}


   }//end if
   
   i=start+vars.size()-1;
   
    }//end if
    
}//end else if



}//end try
catch(IndexOutOfBoundsException boundsException){
    break;
}

        }//end for loop
////print("Before simplifyTokensStage1...scan = "+scan);

unbracketNumbers(scan);
tokenRearranger(scan);
quicksimplify(scan);
multiplyOutProductBrackets(scan);

    
    
/**
 * 
 * 
 * Help methods acquire double brackets so that
 * the pattern..sin(4*x-7)...sin((4*x-7)):u=(4*x-7)....sin(u)
 * 
 * 
 */ 
    for(int i=0;i<scan.size();i++){
        try{
            
            if(isMethodName(scan.get(i))&&isOpeningBracket(scan.get(i+1))){
                int initialCloseIndex = Bracket.getComplementIndex(true, i+1, scan); 
                //Check for double brackets.
                if(isOpeningBracket(scan.get(i+2))){
                    /**
                     * Check if the inside bracket is really double to the outer one...
                     * Because the pattern is possible...sin((4*x-7)+1) and this is not really double.
                     * The pattern we wish to build is sin(((4*x-7)+1))
                     */
                    int closeIndex = Bracket.getComplementIndex(true, i+2, scan);
                    //method has double bracket already..sin((expr))..so do not add extra.
                    if(closeIndex+1==initialCloseIndex){
                        
                    }
                    else{
          scan.add(initialCloseIndex,")");
          scan.add(i+1,"("); 
                    }
                }//end if
                else{
           scan.add(initialCloseIndex,")");
          scan.add(i+1,"(");                    
                }
                
                
                

          
            }//end if
            

        }//end try
        catch(IndexOutOfBoundsException boundsException){
            
        }//end catch
    }//end for loop 


    freeSpaces(scan);
        ////print("Free spaces optimized.");
cleanUp(scan);
    ////print("Clean-up successful! Code Generator Worked! Thank you.");
    }//end method.
/**
 * Open up unnecessary brackets..e.g. (5) becomes 5
 * 
 * @param scan The scanner output.
 */
private static void unbracketNumbers(ArrayList<String>scan){
  ////print("Before unbracketNumbers...scan = "+scan);
    for(int i=0;i<scan.size();i++){  
        try{
        if(isNumber(scan.get(i))&&(isOpeningBracket(scan.get(i-1))&&!isMethodName(scan.get(i-2)) )&&isClosingBracket(scan.get(i+1))){
            scan.remove(i+1);
            scan.remove(i-1);
            }
        }//end try
        catch(IndexOutOfBoundsException boundsException){
            
        }
    } //end for loop
      ////print("Before unbracketNumbers...scan = "+scan);
}//end method.    
/**
 * Open up unnecessary brackets..e.g. (5) or 3+(2-9+8+cos(4))-7*x..should become
 * 3+2-9+8+cos(4)-7*x
 * 
 * @param scan The scanner output.
 */    
public static void openUpUnnecessaryBrackets(ArrayList<String>scan){
    ////print("Before openUpUnnecessaryBrackets...scan = "+scan);
     
    for(int i=0;i<scan.size();i++){
       
   
        if(isClosingBracket(scan.get(i))){
            int close = i;
            int open = Bracket.getComplementIndex(false, close, scan);
            if(open>0){
                if(isOpeningBracket(scan.get(open-1))){
                    int outerOpen = open-1;
                    int outerClose = Bracket.getComplementIndex(true, outerOpen, scan);
                  if(outerClose == close+1){
                      scan.remove(outerClose);
                      scan.remove(outerOpen);
                      i-=2;
                  }  
                }
                
            }
          }

    }//end for loop 
  
    //print("After openUpUnnecessaryBrackets...scan = "+scan);
}//end method
    
/**
 * Introduces brackets around 
 * products of variables with themselves and
 * with numbers.
 * @param scan The scanner output.
 */    
private static void bracketVariableProductsAndQuotients(ArrayList<String>scan){
   //print("Before bracketing products...scan = "+scan);
    for(int i=0;i<scan.size();i++){
        try{
           
            if(isMulOrDiv(scan.get(i))){
                
                String previousToken = scan.get(i-1);
                String token = scan.get(i);
                String nextToken = scan.get(i+1);
                
            int start = i;    
           while(isNumber(token)||isMulOrDiv(token)||(isVariableString(token))){ 
        
           if(isVariableString(token)&&isOpeningBracket(nextToken)){
               i=Bracket.getComplementIndex(true, i+1, scan);
           }
           ++i;
          

try{
            previousToken = scan.get(i-1);
            token = scan.get(i);    
            nextToken = scan.get(i+1);
}
catch(IndexOutOfBoundsException boundsException){
     
}
             }//end while loop   
           /*
            * Before inserting brackets..check that the discovered pattern is not a simple
            * quantity like a number or a variable...e.g. this method will attempt
            * to bracket 5*sin(x)..to give...(5)*sin(x). This is correct but adds the overhead of having to
            * evaluate the brackets around the 5 again later.
            * Also, check that the expression is not already bracketed...e.g this method will try to 
            * convert sin(1/x) into sin((1/x))..which though correct, adds the overhead of having to
            * evaluate the extra brackets.
            * So even though you find that the sub-expression needs to be bracketed, but has 
            * 
            */ 
           if((isOpeningBracket(scan.get(start-2))&&scan.subList(start, i).size()<4)||(scan.subList(start, i).size()==1)){
               
           }
           else{
            //Insert the closing bracket.
           //The terminating pattern is *,( or similar stuff...terminated on the (
            if(isMulOrDiv( scan.get(i-1) ) ){
               scan.add(i-1,")"); 
           }    
            //The terminating pattern is *,2,+..or similar stuff, terminated on the +.
           else{
                scan.add(i,")");   
           }  
           
           
               //situations like ..3+4*x*5/9...Insert the opening bracket starting from 4.
               if(isNumber(scan.get(start-1))|| isVariableString(scan.get(start-1))){
               scan.add(start-1,"("); 
               }
               //situations like ...)*x*5/9..Insert the opening bracket starting from x
               else{
                 scan.add(start+1,"(");   
               }
               i+=2;
           }//end else.
                
        } 
            
            
            
        }//end try
        catch(IndexOutOfBoundsException boundsException){
           // boundsException.printStackTrace();
        }//end catch
        
    }//end for loop
       //print("After bracketing products...scan = "+scan);
}//end method    
    




    
/**
 * Runs through the scanner output and
 * simplifies expressions like
 * 2,*,3..2,+,3...5,^,2 and so on.
 * It does this in readiness for token re-arranging.
 * It also removes some unnecessary brackets.
 * @param scan The scanner output.
 */    
private static void quicksimplify(ArrayList<String>scan){
    
        //print("Quick Simplify input = "+scan);
   //Simplify powers
    for(int i=0;i<scan.size();i++){
        try{
            
      
   if(isPower(scan.get(i)) && isNumber(scan.get(i-1)) && isNumber(scan.get(i+1)) ){
                  scan.set(i+1,""+(Math.pow(Double.parseDouble(scan.get(i-1)), Double.parseDouble(scan.get(i+1)))) );
                  scan.set(i-1,"");
                  scan.set(i,"");//(,3,^,2,)
       //Open the brackets if not belonging to a function.
             if(isOpeningBracket(scan.get(i-2))&&isClosingBracket(scan.get(i+2))){
                 if(i-3>=0 && !isMethodName(scan.get(i-3))){
                 scan.set(i-2,"");
                 scan.set(i+2,"");
                 i+=2;
                 }
             }     
              }//end if
        }//end try
        catch(IndexOutOfBoundsException boundsException){
            
        }
     }//end for loop
   freeSpaces(scan);
    //Simplify and unbracket products and quotients of pattern (2*3) or (2/3).
    for(int i=0;i<scan.size();i++){
        try{
            
            
            
   if(scan.get(i).equals("*") && isNumber(scan.get(i-1)) && isNumber(scan.get(i+1)) ){
                  scan.set(i+1,""+(Double.parseDouble(scan.get(i-1)) * Double.parseDouble(scan.get(i+1))) );
                  scan.set(i-1,"");
                  scan.set(i,"");//(,3,*,2,)   
                  
                     //If product or quotient was self-contained in a bracket,open the brackets if not belonging to a function.
   //e.g. if sin(,3,/,2) do not open. But if just (3,/,2,)..then open.
             if(isOpeningBracket(scan.get(i-2))&&isClosingBracket(scan.get(i+2))){
                 if(i-3>=0 && !isMethodName(scan.get(i-3))){
                 scan.set(i-2,"");
                 scan.set(i+2,"");
                 i+=2;
                 }
             }//end if
              }//end if
   else if(scan.get(i).equals("/") && isNumber(scan.get(i-1)) && isNumber(scan.get(i+1)) ){
                  scan.set(i+1,""+(Double.parseDouble(scan.get(i-1)) / Double.parseDouble(scan.get(i+1))) );
                  scan.set(i-1,"");
                  scan.set(i,"");//(,3,/,2,)   
                  
                     //If product or quotient was self-contained in a bracket,open the brackets if not belonging to a function.
   //e.g. if sin(,3,/,2) do not open. But if just (3,/,2,)..then open.
             if(isOpeningBracket(scan.get(i-2))&&isClosingBracket(scan.get(i+2))){
                 if(i-3>=0 && !isMethodName(scan.get(i-3))){
                 scan.set(i-2,"");
                 scan.set(i+2,"");
                 i+=2;
                 }
             }//end if
              }//end if

        }//end try
        catch(IndexOutOfBoundsException boundsException){
            
        }
             
     }//end for loop
    freeSpaces(scan);
//Simplify adds and subtracts.
    for(int i=0;i<scan.size();i++){
        try{
   if(scan.get(i).equals("+") && isNumber(scan.get(i-1)) && isNumber(scan.get(i+1)) ){
                  scan.set(i+1,""+(Double.parseDouble(scan.get(i-1)) + Double.parseDouble(scan.get(i+1))) );
                  scan.set(i-1,"");
                  scan.set(i,"");//(,3,+,2,) 
                     //If product or quotient was self-contained in a bracket,open the brackets if not belonging to a function.
   //e.g. if sin(,3,+,2) do not open. But if just (3,+,2,)..then open.
             if(isOpeningBracket(scan.get(i-2))&&isClosingBracket(scan.get(i+2))){
                 if(i-3>=0 && !isMethodName(scan.get(i-3))){
                 scan.set(i-2,"");
                 scan.set(i+2,"");
                 }
             }
              }//end if
   else if(scan.get(i).equals("-") && isNumber(scan.get(i-1)) && isNumber(scan.get(i+1)) ){
                  scan.set(i+1,""+(Double.parseDouble(scan.get(i-1)) - Double.parseDouble(scan.get(i+1))) );
                  scan.set(i-1,"");
                  scan.set(i,"");//(,3,-,2,)    
                     //If product or quotient was self-contained in a bracket,open the brackets if not belonging to a function.
   //e.g. if sin(,3,+,2) do not open. But if just (3,-,2,)..then open.
             if(isOpeningBracket(scan.get(i-2))&&isClosingBracket(scan.get(i+2))){
                 if(i-3>=0 && !isMethodName(scan.get(i-3))){
                 scan.set(i-2,"");
                 scan.set(i+2,"");
                 }
             }
              }//end if

                }//end try
        catch(IndexOutOfBoundsException boundsException){
            
        }
     }//end for loop
    freeSpaces(scan);
        //print("Quick Simplify output = "+scan);
    

}    


/**
 * Attempts to reduce output complexity by multiplying out bracketed expressions
 * that are separated by the <code>*</code> operator.
 * 
 * @param scan The scanner output.
 */
private static void multiplyOutProductBrackets(ArrayList<String>scan){

    for(int i=0;i<scan.size();i++){
        try{
       /**
        * Deal with the pattern...),*,(
        * 
        */
            if(isClosingBracket(scan.get(i-1))&&scan.get(i).equals("*") &&isOpeningBracket(scan.get(i+1))){
                int start = Bracket.getComplementIndex(false, i-1, scan);
                int end = Bracket.getComplementIndex(true, i+1, scan);
                
                List<String>leftBrac  = new ArrayList<String>(scan.subList(start, i));
                List<String>rightBrac = new ArrayList<String>(scan.subList(i+1,end+1));
                boolean canReduce = true;
                ///pattern sin(....)*(...)..the sin disqualifies expansion.
                if(isMethodName(scan.get(start-1))){
                    canReduce=false;
                }
                
                
                for(int j=0;j<leftBrac.size()&&canReduce;j++){
                    try{
                    String str = leftBrac.get(j);
                    if(j>0&&isClosingBracket(leftBrac.get(j-1))&&isPower(leftBrac.get(j))&&isOpeningBracket(leftBrac.get(j+1))){
                        canReduce = false;
                        break;
                    }//end if
                    if(j>0&&isClosingBracket(leftBrac.get(j-1))&&isPower(leftBrac.get(j))&&isOpeningBracket(leftBrac.get(j+1))){
                        canReduce = false;
                        break;
                    }//end if
                    
                    if(isNumber(str)||(isVariableString(str)&&!isOpeningBracket(leftBrac.get(j+1)))||isMulOrDiv(str)
                            ||isBracket(str)||isPower(str)  ){
                        canReduce=true;
                    }//end if
                    
                    else{
                           canReduce = false;
                           break;
                    }//end else
                    }//end try
                    catch(IndexOutOfBoundsException boundsException){
                        boundsException.printStackTrace();
                    }
                }//end for loop.
 if(canReduce){
                for(int j=0;j<rightBrac.size()&&canReduce;j++){
                    try{
                    String str = rightBrac.get(j);
                    if(j>0&&isClosingBracket(rightBrac.get(j-1))&&isPower(rightBrac.get(j))&&isOpeningBracket(rightBrac.get(j+1))){
                        canReduce = false;
                           break;
                    }//end if
                    
                    if(isNumber(str)||(isVariableString(str)&&!isOpeningBracket(rightBrac.get(j+1)))||isMulOrDiv(str)
                            ||isBracket(str)||isPower(str)  ){
                        canReduce=true;
                    }//end if
                    else{
                           canReduce = false;
                           break;
                    }
                    }//end try
                    catch(IndexOutOfBoundsException boundsException){
                        boundsException.printStackTrace();
                    }
                }//end for loop.     
 }//end if
         
if(canReduce){
    ArrayList<String> vars = new ArrayList<String>();
   
    
    for(int k=0;k<leftBrac.size();k++){
        try{
        if(isNumber(leftBrac.get(k))&&leftBrac.get(k+1).equals("*")){
           if(vars.isEmpty()){
               vars.add(leftBrac.get(k));
           }//end if
           else{
            if(isNumber(vars.get(0))){
                vars.set(0,""+(Double.parseDouble(vars.get(0))*Double.parseDouble(leftBrac.get(k))));
            }//end if
            else{
                vars.add(0,""+(Double.parseDouble(leftBrac.get(k))));
            }//end else
           }//end else
        }//end if
        else if(isPower(leftBrac.get(k))&&isVariableString(leftBrac.get(k-1))&&isNumber(leftBrac.get(k+1))){
            int index = vars.indexOf(leftBrac.get(k-1));
            
            if(index == -1){
                if(vars.isEmpty()){
                    vars.add("(");
                    vars.add(leftBrac.get(k-1));
                    vars.add("^");
                    vars.add(leftBrac.get(k+1));
                    vars.add(")");
                }//end if
                else{
                    vars.add("*");
                    vars.add("(");
                    vars.add(leftBrac.get(k-1));
                    vars.add("^");
                    vars.add(leftBrac.get(k+1));
                    vars.add(")");
                }//end else
            }///end if
            else{
                vars.set(index+2,""+(Double.parseDouble(vars.get(index+2))+Double.parseDouble(leftBrac.get(k+1))));
            }//end else
            
        }//end else if
        }//end try
        catch(IndexOutOfBoundsException boundsException){
         
        }//end catch
    }//end for loop
    for(int k=0;k<rightBrac.size();k++){
        try{
        if(isNumber(rightBrac.get(k))&&rightBrac.get(k+1).equals("*")){
           if(vars.isEmpty()){
               vars.add(rightBrac.get(k));
           }//end if
           else{
            if(isNumber(vars.get(0))){
                vars.set(0,""+(Double.parseDouble(vars.get(0))*Double.parseDouble(rightBrac.get(k))));
            }//end if
            else{
                vars.add(0,""+(Double.parseDouble(rightBrac.get(k))));
            }//end else
           }//end else
        }//end if
        else if(isPower(rightBrac.get(k))&&isVariableString(rightBrac.get(k-1))&&isNumber(rightBrac.get(k+1))){
            int index = vars.indexOf(rightBrac.get(k-1));
            
            if(index == -1){
                if(vars.isEmpty()){
                    vars.add("(");
                    vars.add(rightBrac.get(k-1));
                    vars.add("^");
                    vars.add(rightBrac.get(k+1));
                    vars.add(")");
                }//end if
                else{
                    vars.add("*");
                    vars.add("(");
                    vars.add(rightBrac.get(k-1));
                    vars.add("^");
                    vars.add(rightBrac.get(k+1));
                    vars.add(")");
                }//end else
            }///end if
            else{
                vars.set(index+2,""+(Double.parseDouble(vars.get(index+2))+Double.parseDouble(rightBrac.get(k+1))));
            }//end else
            
        }//end else if
   }//end try
        catch(IndexOutOfBoundsException boundsException){
            
        }//end catch      
    }//end for loop    
    
//vars.add(0,"(");
//vars.add(")");
List<String> temp = scan.subList(start, end+1);
temp.clear();
temp.addAll(vars);
   i=start+vars.size()-1;
}//end if          
 
 
 
 
                
            }//end if
        
        }//end try
        catch(IndexOutOfBoundsException boundsException){
        
    }//end catch
    }//end for
       
}//end method   
    
/**
 * Condenses numbers and variables to be added within
 * a bracket to the end of the bracket.
 * For instance,(3+x+sin(3*x)+5-7) becomes
 * (sin(3*x)+3+x+5-7)
 * @param scan The scanner output. 
 */
private static void tokenRearranger(ArrayList<String>scan){
    //print("Before re-arranging tokens--- scan = "+scan);
    for(int i=0;i<scan.size();i++){
   try{
        if(isClosingBracket(scan.get(i))){
            int close = i;
            int open = Bracket.getComplementIndex(false, close, scan);
            List<String>temp= new ArrayList<String>();
            int j = open+1;
          while(j<close){
              
              
              if(isOpeningBracket(scan.get(j))){
               int closeIndex = Bracket.getComplementIndex(true, j, scan);
               j = closeIndex;
              }//end if
              
             else if(isNumber(scan.get(j)) && isPlusOrMinus(scan.get(j-1)) && isPlusOrMinus(scan.get(j+1))){
                  temp.add(scan.get(j-1));
                  temp.add(scan.get(j));
                  scan.set(j-1,"");
                  scan.set(j,"");
              }//end if
              else if( isNumber(scan.get(j)) && isOpeningBracket(scan.get(j-1))&& isPlusOrMinus(scan.get(j+1))){
                  temp.add("+");
                  temp.add(scan.get(j));
                  scan.set(j,"");
              }//end else if
              
           ++j;
          }//end while  
            scan.addAll(close, temp);  
            
            freeSpaces(scan.subList(open, Bracket.getComplementIndex(true, open, scan)));
if(scan.get(open+1).equals("+")){
                scan.set(open+1,"");
                //print("open = "+open);
            }
        }//end if
        
   }//end try
   catch(Exception e){
       e.printStackTrace();
   }
    }//end for loop
    
    //print("After re-arranging tokens--- scan = "+scan);
}//end method    

/**
 * Removes unconventional code generated and introduces final optimization 
 * modifications before the output goes to the differentiator stages.
 * 
 * @param scan The scanner output.
 */
private static void cleanUp(ArrayList<String>scan){
    
    for(int i=1;i<scan.size();i++){
        try{
       /**
        * Deal with the pattern...(,+,(...or (,-,(
        * This is generated during token rearrangement...e.g. when numbers sink to the
        * bottom of the bracket pair.For example,(,3,+,(,sin,(,x,),..),)..becomes (,+,(sin,(,x,),+,3,),)..
        * 
        */
            if(isOpeningBracket(scan.get(i-1))&&isPlusOrMinus(scan.get(i))&&isOpeningBracket(scan.get(i+1))){
                if(scan.get(i).equals("+")){
                    scan.remove(i);
                }
                else if(scan.get(i).equals("-")){
                    scan.set(i,"-1");
                    scan.add(i+1,"*");
                }
                
            }
            //The pattern...(,+,..or (,-,...
            else if(isOpeningBracket(scan.get(i))&&isPlusOrMinus(scan.get(i+1))){
                if(scan.get(i+1).equals("+")){
                    scan.remove(i+1);
                }
                else if(scan.get(i+1).equals("-")){
                    scan.set(i+1,"-1");
                    scan.add(i+2,"*");
                    
                }
                
            }
        
        }//end try
        catch(IndexOutOfBoundsException boundsException){
        
    }//end catch
    }//end for
    
}//end method
    
/**
 * 
 * @param scan The ArrayList object containing the scanned tokens.
 * @param i The starting index in the ArrayList.
 * @return an object array containing in the first index,
 * the ending index after simplifying as far as possible
 * and in the second index, the ArrayList object that stores the simplified
 * version of the scanner output in the given range.
 * 
 * This stage simplifies portions of the expression
 * that have strings of multiplications and divisions of numbers and variables,
 * generating number products and variable powers.
 * <b color='green'>
 * Come here to allow the parser deal with the pattern
 * x^constant_name also. For now, it deals with x^constant_value...e.g.
 * x^3 but not x^a. Once this is done, update the other parser methods as
 * relevant all through the parser.
 * 
 * </b>
 * 
 */
    private static Object[] utilitySimplifier(ArrayList<String>scan, int i){
           int start = i;
 
ArrayList<String>vars=new ArrayList<String>();

while(i+1<=scan.size() && (isNumber(scan.get(i))||(isVariableString(scan.get(i))&&!isOpeningBracket(scan.get(i+1)))||isMulOrDiv(scan.get(i))||
        STRING.purifier(scan.get(i)).isEmpty()) ){

    /**
     * Handle Numbers
     */  
    if(isNumber(scan.get(i))&&scan.get(i-1).equals("/")){
        
        if(vars.isEmpty()){
            vars.add(""+(1.0/Double.parseDouble(scan.get(i))));
        }
        else{
         if(!isNumber(vars.get(0))){
           vars.add(0,""+(1.0/Double.parseDouble(scan.get(i)))); 
           vars.add(1,"*");
        }
        else if(isNumber(vars.get(0)) ){
        vars.set(0,""+(Double.parseDouble(vars.get(0))/Double.parseDouble(scan.get(i))) );
        }
        }
           //   scan.set(i,"");
    }    
    else if(isNumber(scan.get(i))&&scan.get(i-1).equals("*")){
        if(vars.isEmpty()){
            vars.add(scan.get(i));
        }
        else{
        if(!isNumber(vars.get(0))){ 
           vars.add(0,scan.get(i)); 
           vars.add(1,"*");
        }
        else if(isNumber(vars.get(0)) ){ 
        vars.set(0,""+(Double.parseDouble(vars.get(0))*Double.parseDouble(scan.get(i))) );
        }
    }
         //scan.set(i,"");
    }
    else if(isNumber(scan.get(i))&&scan.get(i+1).equals("*")){
        if(vars.isEmpty()){
            vars.add(scan.get(i));
        }
        else{
        if(!isNumber(vars.get(0))){
           vars.add(0,scan.get(i)); 
           vars.add(1,"*");
        }
        else if(isNumber(vars.get(0)) ){
        vars.set(0,""+(Double.parseDouble(vars.get(0))*Double.parseDouble(scan.get(i))) );
        }
}
         //scan.set(i,"");
    }  
    else if(isNumber(scan.get(i))&&isPlusOrMinus(scan.get(i-1))){//e.g. 2*3+(
        if(vars.isEmpty()){
            vars.add(scan.get(i));
        }
        else{
        if(!isNumber(vars.get(0))){
           vars.add(0,scan.get(i)); 
           vars.add(1,"*");
        }
        else if(isNumber(vars.get(0)) ){
        vars.set(0,""+(Double.parseDouble(vars.get(0))*Double.parseDouble(scan.get(i))) );
        }
    }
         //scan.set(i,"");
    }//end else if  
    else if(isNumber(scan.get(i))&&isPlusOrMinus(scan.get(i+1))){//e.g. 2*3+(
        if(vars.isEmpty()){
            vars.add(scan.get(i));
        }
        else{
        if(!isNumber(vars.get(0))){
           vars.add(0,scan.get(i)); 
           vars.add(1,"*");
        }
        else if(isNumber(vars.get(0)) ){
        vars.set(0,""+(Double.parseDouble(vars.get(0))*Double.parseDouble(scan.get(i))) );
        }
    }
         //scan.set(i,"");
    }//end else if  
        
   
    
    /**
     * Handle Variables.
     * 
     */
    else if( isVariableString(scan.get(i))&&scan.get(i-1).equals("/") ){
        int index = vars.indexOf(scan.get(i));
        //Variable not encountered yet.
        if(index==-1){
       if( !vars.isEmpty()&&!vars.get(vars.size()-1).equals("*") ){
           vars.add("*");
       }
            vars.add("(");
            vars.add(scan.get(i));
            vars.add("^");
            vars.add("-1");
            vars.add(")");
        }
        else{
        vars.set(index+2, ""+(Integer.parseInt(vars.get(index+2))-1) );
        }
scan.set(i,"");  
    }//end if      
    else if( isVariableString(scan.get(i))&& (scan.get(i-1).equals("*")||scan.get(i+1).equals("*")) ){
        int index = vars.indexOf(scan.get(i));
        //Variable not encountered yet.
        if(index==-1){
       if( !vars.isEmpty()&&!vars.get(vars.size()-1).equals("*") ){
           vars.add("*");
       }
            vars.add("(");
            vars.add(scan.get(i));
            vars.add("^");
            vars.add("1");
            vars.add(")");
        }
        else{
        vars.set(index+2, ""+(Integer.parseInt(vars.get(index+2))+1) );
        }
scan.set(i,"");  
    }//end if
  

    
    

      ++i;
    }//end while loop

    return new Object[]{i,vars};
    }//end method
    
    


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("");
        ArrayList<String>scan = getScanner();
        for(String token:scan){
            sb.append(token);
        }
       return sb.substring(0, sb.length());
    }

    
    
    
    public static void main(String[] args) {
        try {
            String expression = 
                    "2^(3+2)-sin(2)+((((7))))-3*x/(5-sin(4*x^2-7))";
//"x*y*2*x*(3*x*5*y)+8-(5+2+6)-2+6+1*2-x*y*3*(2*x*x*2*x*x*8*x*7*x*4*y+(3*2*8+4/21*x)-¹-(((5*2)))-3*x*x^2*2*3*4^x-cos(4/x)+5*sin(1/x))+5*3/6*8+4*x*3";
            CodeGenerator dsb = new CodeGenerator(expression);
            System.out.println("Initially: "+expression);
            System.out.println("Finally: "+dsb.getScanner());
            System.out.println("Generated expression =  "+dsb);
            
        } 
        catch (Exception e) {
        }


    }//end main method   
}
