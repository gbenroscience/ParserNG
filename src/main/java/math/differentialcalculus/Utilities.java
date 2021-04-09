/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import parser.Bracket;
import static parser.Bracket.*;

import java.util.ArrayList;
import java.util.List;
import static parser.Number.*;
import static parser.Variable.*;
import static parser.methods.Method.*;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 *
 * @author GBEMIRO
 */
public class Utilities {
    public static final ArrayList<String>whitespaceremover;
    
    static {
        whitespaceremover = new ArrayList<>();
        whitespaceremover.add("");
        whitespaceremover.add(" ");
    }
    
    public static void freeSpaces(List<String>scan){
         scan.removeAll(whitespaceremover);
    }
  public static void print(Object obj) {
      //  System.out.println(obj);
    }
    /**
     * 
     * @param name The name to check.
     * @return true if the name is automatically generated and
     * so, most likely refers to a stored Differentiable.
     */
    public static boolean isAutoGenNameFormat(String name){
          int indexOfnumber = name.indexOf("_")+1;
    if(indexOfnumber!=-1){
     String numberpart = name.substring(indexOfnumber);
     return name.startsWith("myDiff_")&&isNumber(numberpart);
    }//end if
    return false;
    }//end method
 
    /**
     * 
     * @param data An ArrayList of Strings.
     * return the Strings in the List concatenated together.
     */
    public static String getText(ArrayList<String> data){
        StringBuilder sb = new StringBuilder("");
        for(String token:data){
            sb.append(token);
        }
        return sb.substring(0, sb.length());
    }//end method
    /**
     * 
     * @param value The item to check
     * Always double check in the scanner where this item exists
     * that the next token to <code>value</code> is not an open bracket
     * if value is a valid variable name.
     * @return true if it can be differentiated to get a value.
     */
    public static boolean isDifferentiable(String value){
        return isNumber(value)||isVariableString(value);
    }
    
    public static boolean isFormula(String name){
        int indexOfnumber = name.indexOf("_")+1;
  if(indexOfnumber!=-1){
   String numberpart = name.substring(indexOfnumber);
   return name.startsWith("myForm_")&&isNumber(numberpart);
  }//end if
  return false;
  }//end method   
    
/**
 * Simplifies portions of math code that involve
 * products or quotients of -1.
 * @param list The list of math tokens.
 */    
public static void simplifyNegOneProducts(ArrayList<String>list){
    
 for(int i=0;i<list.size();i++){   
    try{
    if( list.get(i).equals("-1")||list.get(i).equals("-1.0") ){
        //....,*|/,-1
    if(isMulOrDiv(list.get(i-1))){
         //....,),*|/,-1
      
int j=i-1;
   while(true){
    if(isMulOrDiv(list.get(j))){
    --j;    
    }//end if
    else if(isClosingBracket(list.get(j))){
        j = Bracket.getComplementIndex(false, j, list);
        --j;
    }
    else if(isNumber(list.get(j))){
      list.set(j,""+(-1*Double.parseDouble(list.get(j))));
      list.remove(i-1);
      i-=1;
      break;
    }
    else if(isVariableString(list.get(j))){
      --j;
    }
    else if(isPlusOrMinus(list.get(j))){
    if(list.get(j).equals("+")){
        list.set(j,"-");
    }
    else{
       list.set(j,"+");   
    }
      list.remove(i-1);
      i-=1;
      break;    
    }//end else if
    //...,(,(......)....,*,-1
    else if(isOpeningBracket(list.get(j))){
      list.add(j+1,"-");
      list.remove(i);
      i-=1;
      break;
    }
   }//end while        
    }
    //now look to the right.
    if(isMulOrDiv(list.get(i+1))){
int j=i+1;
   while(true){
    if(isMulOrDiv(list.get(j))){
    ++j;    
    }//end if
    else if(isOpeningBracket(list.get(j))){
        j = Bracket.getComplementIndex(true, j, list);
        ++j;
    }
    else if(isNumber(list.get(j))){
      list.set(j,""+(-1*Double.parseDouble(list.get(j))));
      list.remove(i+1);
      break;
    }
    else if(isVariableString(list.get(j))){
      ++j;
    }
    else if(isPlusOrMinus(list.get(j))){
    if(list.get(j).equals("+")){
        list.set(j,"-");
    }
    else{
       list.set(j,"+");   
    }
      list.remove(i+1);
      break;    
    }//end else if
    //...,-1,*....,(......,),),
    else if(isClosingBracket(list.get(j))){
      list.add(j+1,"-");
      list.remove(i);
      break;
    }
   }//end while        
        
        
        
        
        
        
        
        
    }//end if
        
        
        
    }//end if
 }//end try
 catch(IndexOutOfBoundsException boundsException){
    
}//end catch
    
 }//end for loop
    
    
}//end method.
    
    
    
/**
 * 
 * @param array The scanned list containing math tokens.
 * 
 * This method seeks to apply the algebraic effect of adding
 * or subtracting or multiplying zeroes to a math expression.
 * e.g. 3*x*0+4*1 becomes 4*1.
 * 
 */
    public static void simplifyZeroes(ArrayList<String>array){
  //do the zeroes
  for(int i=0;i<array.size();i++){
      try{
    if(array.get(i).equals("0") ||array.get(i).equals("0.0")){
simplifyZeroesAt(i, array);
   }//end if      
      }//end try
      catch(IndexOutOfBoundsException boundsException){
          
      }//end catch
      
  }//end for loop 
    }//end method
    
/**
 * 
 * @param index The index at which the zero is.
 * @param list The scanned list containing math tokens.
 * 
 * Method that recursively applies the effect of zero 
 * at its point of occurrence to a math expression.
 * 
 */
public static void simplifyZeroesAt(int index,ArrayList<String>list){
    
   if( list.get(index).equals("0")||list.get(index).equals("0.0")){
       //(,0,+|-
       if(isOpeningBracket(list.get(index-1))&&isPlusOrMinus(list.get(index+1))){
        if(list.get(index+1).equals("+")){
            list.remove(index+1);
            list.remove(index);
        }
        else if(list.get(index+1).equals("-")){
            list.remove(index);
        }
        
       }//end if
       //..,+|-,0,)
       else if(isPlusOrMinus(list.get(index-1))){
           list.subList(index-1, index+1).clear();
       }//end else if
       //unbracket the pattern...(,0,)...avoid the pattern sin,(,0,)
       else if(isOpeningBracket( list.get(index-1) ) && isClosingBracket(list.get(index+1)) &&!isMethodName(list.get(index-2))){
           list.remove(index+1);
           list.remove(index-1);
           --index;
           simplifyZeroesAt(index, list);
       }//end if
       //0,*|/
       else if(isMulOrDiv(list.get(index+1))){
            //0,*,num|var  
        if( isNumber(list.get(index+2)) || (isVariableString(list.get(index+2))&&!isOpeningBracket(list.get(index+3))  ) ){
            list.subList(index+1, index+3).clear();
        }//end if
        //0,*,sin,(,....,)
        else if( isVariableString(list.get(index+2))&&isOpeningBracket(list.get(index+3)) ){
  int close = Bracket.getComplementIndex(true, index+3, list);
            list.subList(index+1, close+1).clear();
        }//end else if
        //0,*,(,....,)
        else if( isOpeningBracket(list.get(index+2)) ){
  int close = Bracket.getComplementIndex(true, index+2, list);
            list.subList(index+1, close+1).clear();
        }//end else if
         simplifyZeroesAt(index, list);
       }//end else if
       
       
       //...*,0...
       else if(list.get(index-1).equals("*")){
            //num|var,*,0,...  
        if( isNumber(list.get(index-2)) || isVariableString(list.get(index-2))   ){
            //*|/,num|var,*,0
            if(isMulOrDiv(list.get(index-3))){
                //dont remove the * before the 0.
            list.subList(index-3, index-1).clear();
            index-=2;
            }
            //+|-,num|var,*,0
            else if(isPlusOrMinus(list.get(index-3))){
            list.subList(index-2, index).clear();
            index-=2;
            }
            //(,num|var,*,0
            else if(isOpeningBracket(list.get(index-3))){
            list.subList(index-2, index).clear();  
            index-=2;
            }
            
        }//end if
       //..,),*,0,
        else if( isClosingBracket(list.get(index-2)) ){
  int open = Bracket.getComplementIndex(false, index-2, list);
  //sin,(,....,),*,0 
  if(isMethodName(list.get(open-1)) ){
      //*|/,sin,(,....,),*,0
 if(isMulOrDiv(list.get(open-2))){
  List temp = list.subList(open-2, index-1);
  int sz = temp.size();  
  temp.clear();
  index-=sz;     
 }
 //+|-,sin,(,....,),*,0
 else if(isPlusOrMinus(list.get(open-2))){
  List temp = list.subList(open-2, index);
  int sz = temp.size();  
  temp.clear();
  index-=sz;     
 }
 //(,sin,(,.....,),*,0
 else if(isOpeningBracket(list.get(open-2))){
  List temp = list.subList(open-1, index);
  int sz = temp.size();  
  temp.clear();
  index-=sz;      
 }
      

  }//end if
    //(,....,),*,0
  else{
      //*|/,(,....,),*,0
 if(isMulOrDiv(list.get(open-1))){
  List temp = list.subList(open-1, index-1);
  int sz = temp.size();  
  temp.clear();
  index-=sz;     
 }
 //+|-,(,....,),*,0
 else if(isPlusOrMinus(list.get(open-1))){
  List temp = list.subList(open-1, index);
  int sz = temp.size();  
  temp.clear();
  index-=sz;     
 }
 //(,(,.....,),*,0
 else if(isOpeningBracket(list.get(open-1))){
  List temp = list.subList(open-1, index);
  int sz = temp.size();  
  temp.clear();
  index-=sz;      
 }      
  }//end else
  
  
        }//end else if
      
  
         simplifyZeroesAt(index, list);
       }//end else if
       
       
       
       
       
   }//end if
    
}//end method


/**
 * 
 * @param index The index of the 1
 * @param list The ArrayList containing it.
 * 
 * Seeks to calculate the effect of multiplying 
 * 1 with quantities in the list...e.g 1*(....) becomes (.....),
 * 1*sin(...) becomes sin(...)
 */
private static void simplifyOneProducts(int index,ArrayList<String>list){
    
    if( list.get(index).equals("1")||list.get(index).equals("1.0")){
        if(list.get(index-1).equals("(")&&list.get(index+1).equals(")") &&!isMethodName(list.get(index-2))){
            list.remove(index+1);
            list.remove(index-1);
            --index;
             simplifyOneProducts(index, list);
        }//end if
        else if(isMulOrDiv(list.get(index-1))){
          list.subList(index-1, index+1).clear(); 
           simplifyOneProducts(index, list);    
     
        }//end else if
        else if(list.get(index+1).equals("*")){
          list.subList(index, index+2).clear();  
           simplifyOneProducts(index, list);    
        }//end else if
        else{
            
        }
    
        
    }//end if

    
    
    
}//end method
/**
 * 
 * @param list The ArrayList containing it.
 * 
 * Seeks to calculate the effect of multiplying 
 * 1 with quantities in the list...e.g 1*(....) becomes (.....),
 * 1*sin(...) becomes sin(...)
 */
public static void simplifyOneProducts(ArrayList<String>list){
      for(int i=0;i<list.size();i++){
      try{
          
    if(list.get(i).equals("1")||list.get(i).equals("1.0")){
simplifyOneProducts(i, list);
   }//end if
    
}//end try 
      catch(IndexOutOfBoundsException boundsException){
          
      }//end catch
      
  }//end for loop
}//end method



/**
 * Checks for the pattern..
 * ...+,(,.......,)+|- and 
 * removes the bracket..if possible.
 * @param scan The list of math tokens
 */
public void openBrackets(ArrayList<String>scan){
    
}



/**
 * Evaluates products of numbers or divisions
 * of numbers.
 * @param scan The scanner output. 
 */
public static void evaluateTokens(ArrayList<String>scan){
    
    for(int i=0;i<scan.size();i++){
   try{
        if(isClosingBracket(scan.get(i))){
            int close = i;
            int open = Bracket.getComplementIndex(false, close, scan);
            int j = open+1;
            double value=0;
          while(j<close){
              
         if( isPlusOrMinus(scan.get(j)) ){
             
         }     
              
              
              
              
              if(isOpeningBracket(scan.get(j))){
               int closeIndex = Bracket.getComplementIndex(true, j, scan);
               j = closeIndex;
              }//end if
              else if(isVariableString(scan.get(j))){
              //do nothing
              }//end if 
              
              //...,(,num,*|/,
             else if(isNumber(scan.get(j)) && isOpeningBracket(scan.get(j-1))&& isMulOrDiv(scan.get(j+1))){
                  value*= Double.parseDouble(scan.get(j));   
                  scan.set(j,"");
              }//end if
             //...,*,num,...
              else if( isNumber(scan.get(j)) && isMulOrDiv(scan.get(j-1))){
 if(scan.get(j-1).equals("*")){
                   value*= Double.parseDouble(scan.get(j));   
                  }
                  else{
                    value/= Double.parseDouble(scan.get(j));    
                  }
                  scan.set(j-1,"");
                  scan.set(j,"");
              }//end else if
             //...,*|/,num,),...
              else if( isNumber(scan.get(j)) && isMulOrDiv(scan.get(j-1)) && isClosingBracket(scan.get(j+1))){
 if(scan.get(j-1).equals("*")){
                   value*= Double.parseDouble(scan.get(j));   
                  }
                  else{
                    value/= Double.parseDouble(scan.get(j));    
                  }
                  scan.set(j-1,"");
                  scan.set(j,"");
              }//end else if              
              
           ++j;
          }//end while  
          if(value!=0){
              String val = (""+value);
              val = val.endsWith(".0")?val.substring(0,val.indexOf(".")):val;
            scan.add(close, val);
            scan.add(close, "+");
          }//end if
           
            freeSpaces(scan.subList(open, Bracket.getComplementIndex(true, open, scan)));
if(scan.get(open+1).equals("+")){
                scan.remove(open+1);
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
 * Condenses numbers and variables to be added within
 * a bracket to the end of the bracket.
 * For instance,(3+x+sin(3*x)+5-7) becomes
 * (sin(3*x)+3+x-2)
 * @param scan The scanner output. 
 */
public static void tokenRearranger(ArrayList<String>scan){
    //print("Before re-arranging tokens--- scan = "+scan);
    for(int i=0;i<scan.size();i++){
   try{
        if(isClosingBracket(scan.get(i))){
            int close = i;
            int open = Bracket.getComplementIndex(false, close, scan);
            int j = open+1;
            double value=0;
          while(j<close){
              
              
              if(isOpeningBracket(scan.get(j))){
               int closeIndex = Bracket.getComplementIndex(true, j, scan);
               j = closeIndex;
              }//end if
              //...+|-,num,+|-
             else if(isNumber(scan.get(j)) && isPlusOrMinus(scan.get(j-1))&& isPlusOrMinus(scan.get(j+1))){
 if(scan.get(j-1).equals("-")){
                   value+= -1*Double.parseDouble(scan.get(j));   
                  }
                  else{
                    value+= Double.parseDouble(scan.get(j));    
                  }
                  scan.set(j-1,"");
                  scan.set(j,"");
                 
              }//end if
             //...,(,num,+|-,...
              else if( isNumber(scan.get(j)) && isOpeningBracket(scan.get(j-1))&& isPlusOrMinus(scan.get(j+1))){
  value+= Double.parseDouble(scan.get(j));    
                  
                  scan.set(j,"");
              }//end else if
             //...,+|-,num,),...
              else if( isNumber(scan.get(j)) && isPlusOrMinus(scan.get(j-1)) && isClosingBracket(scan.get(j+1))){
 if(scan.get(j-1).equals("-")){
                   value+= -1*Double.parseDouble(scan.get(j));   
                  }
                  else{
                    value+= Double.parseDouble(scan.get(j));    
                  }
                  scan.set(j-1,"");
                  scan.set(j,"");
              }//end else if              
              
           ++j;
          }//end while  
          if(value!=0){
              String val = (""+value);
              val = val.endsWith(".0")?val.substring(0,val.indexOf(".")):val;
            scan.add(close, val);
            scan.add(close, "+");
          }//end if
           
            freeSpaces(scan.subList(open, Bracket.getComplementIndex(true, open, scan)));
if(scan.get(open+1).equals("+")){
                scan.remove(open+1);
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
 * 
 * @param list The list containing the scanned math tokens.
 */
public static void multipleBracketRemover(ArrayList<String>list){

for(int i=0;i<list.size();i++){
	
if(isCloseBracket(list.get(i))){
	int close = i;
	//int open = Bracket.getComplementIndex(true, start, scan);
	
	
}




}//end for loop


	
	
}//end method
    
    

/**
 * 
 * @param list The list containing the scanned math tokens.
 */
public static void multipleBracketRemover_(ArrayList<String>list){

for(int i=0;i<list.size();i++){
	
	
	if(isOpenBracket(list.get(i)) && i>0){
		
		if(isMethodName(list.get(i-1))) {
			int open = i;
			int close = Bracket.getComplementIndex(true, i, list);
			try {
			while( isOpenBracket(list.get(open+1)) && isCloseBracket(list.get(close-1) ) 
					&& close-1==Bracket.getComplementIndex(true, open+1, list)){
				
			list.remove(close);
			list.remove(open);
			close-=2;
			}//end while
			}//end try ty
			catch (Exception e) {
				break;
			}
		}//end if		
		//...^(((stuff)))
		else if(isPower(list.get(i-1)) ) {
			int open = i;
			int close = Bracket.getComplementIndex(true, i, list);
			List<String>sub=list.subList(open, close+1);
			//...^(num|var)))
			if(sub.size()==3){
				list.remove(close);list.remove(open);
				continue;
			}
			//^(((var|num+|-var|num)))
			if(sub.contains("+")||sub.contains("-")){
			try {
			while( isOpenBracket(list.get(open+1)) && isCloseBracket(list.get(close-1) ) 
					&& close-1==Bracket.getComplementIndex(true, open+1, list)){
			list.remove(close);
			list.remove(open);
			close-=2;
			
			}//end while
			}//end try ty
			catch (Exception e) {
				break;
			}
			}//end else if
			//^(((var|num*|/|...var|num)))
			else if(!sub.contains("+")&&!sub.contains("-")){
			try {
			while( isOpenBracket(list.get(open+1)) && isCloseBracket(list.get(close-1) ) 
					&& close-1==Bracket.getComplementIndex(true, open+1, list)){
			list.remove(close);
			list.remove(open);
			close-=2;
			
			}//end while
			list.remove(close);
			list.remove(open);
			
			}//end try ty
			catch (Exception e) {
				break;
			}//end catch
			}//end else if
		}//end else if	
		//...(((...)))
		else if(!isMethodName(list.get(i-1)) && ! isPower(list.get(i-1))){
			int open = i;
			int close = Bracket.getComplementIndex(true, i, list);
			List<String>sub=list.subList(open, close+1);
			//...^(num|var)))
			if(sub.size()==3){
				list.remove(close);list.remove(open);
				
				continue;
			}
			//^(((var|num+|-var|num)))
			if(sub.contains("+")||sub.contains("-")){
			try {
				
			while( isOpenBracket(list.get(open+1)) && isCloseBracket(list.get(close-1) ) 
					&& close-1==Bracket.getComplementIndex(true, open+1, list) ){
			list.remove(close-1);
			list.remove(open+1);
			close-=2;
			
			}//end while
			
			}//end try ty
			catch (Exception e) {
				e.printStackTrace();
				break;
			}
			}//end else if
			//^(((var|num*|/|...var|num)))
			else if(!sub.contains("+")&&!sub.contains("-")){
			try {
			while( isOpenBracket(list.get(open+1)) && isCloseBracket(list.get(close-1) ) 
					&& close-1==Bracket.getComplementIndex(true, open+1, list)){
			list.remove(close);
			list.remove(open);
			close-=2;
			
			}//end while
			list.remove(close);
			list.remove(open);
			}//end try ty
			catch (Exception e) {
				break;
			}//end catch
			}//end else if		
		}//end else if
	}
	
	




}//end for loop


	
	
}//end method
    
    

    
    
    
    public static void main(String s[]){
        try {
            ArrayList<String>scan = new ArrayList<String>();
            scan = new DerivativeScanner("(1/((x-x^2))*5^sin(((((((((x)))+7))))))-7+((2^(x)))+5*cos(((x-2)))+2*((((((x-1))))))-1)").getScanner();
            
            multipleBracketRemover(scan);
            print("scan = "+scan);
            
        } catch (Exception ex) {
            Logger.getLogger(Utilities.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
    }
    
    
}//end class