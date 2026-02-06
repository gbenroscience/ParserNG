/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package util;

/**
 *
 * @author GBEMIRO
 */

import java.util.EmptyStackException;
import java.util.Stack;



public class HistoryManager{


    public static Stack<String> upStack = new Stack<>();

    public static Stack<String> downStack = new Stack<>();

    public HistoryManager(){
    }



    public void recordHistory(String expr){
      try {
         String peekVal = upStack.peek();
         if(expr!=null && !expr.isEmpty() ) {
             if (!peekVal.equals(expr)) {
                 upStack.push(expr);
             }
         }
     }
     catch (EmptyStackException e){
         if(expr!=null && !expr.isEmpty() ) {
             upStack.push(expr);
         }
     }





    }
    public  String getNext( String currDisplayingInput){
       try {


           String history = upStack.pop();

           if (history != null) {
               downStack.push(history);
           }

            if(history.equals(currDisplayingInput)) {
                history = upStack.pop();
                if (history != null) {
                    downStack.push(history);
                }
           }


           return history;
       }
       catch (EmptyStackException ex){
           return null;
       }
    }
    public String getPrevious( String currDisplayingInput){
        try {


            String history = downStack.pop();

            if (history != null) {
                upStack.push(history);
            }

            if(history.equals(currDisplayingInput)) {
                history = downStack.pop();
                if (history != null) {
                    upStack.push(history);
                }
            }


            return history;
        }
        catch (EmptyStackException ex){
            return null;
        }
    }


    public void clearHistory(){
        upStack.clear();downStack.clear();
    }
 
}//end class