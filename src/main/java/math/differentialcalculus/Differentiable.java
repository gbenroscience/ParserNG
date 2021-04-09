/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package math.differentialcalculus;

import java.util.ArrayList;
import static parser.Number.*;
import static parser.Operator.*;
import static parser.Variable.*;

import static parser.methods.Method.*;
import java.util.List;
import static math.differentialcalculus.Utilities.*;


/**
 * Anything that can be mathematically differentiated..
 * e.g a number, a variable or an expression.
 * @author GBEMIRO
 */
public class Differentiable {
    /**
     * The representation of the
     * Differentiable.
     */
    private String name;
    /**
     * The information contained by the Differentiable.
     */
    private ArrayList<String>data;
 
    


    /**
     *
     * @param name The representation of the Differentiable.
     */
    public Differentiable(String name) {
        this.name = name;
        this.data = new ArrayList<>();
    }

    /**
     *
     * @param name The representation of the Differentiable.
     * @param data The information contained by the Differentiable.
     */
    public Differentiable(String name, ArrayList<String> data) {

        //Check simple format...( number|var,*|/,number|var,)
        this.name = name;
        this.data = data;

    }//end constructor
    /**
     *
     * @return true if this name is automatically generated..
     * which implies that it was created to represent a Differentiable
     * that is a part of the differentiation chain.
     */
    public boolean isChain(){
        return isAutoGenNameFormat(name);
    }//end method

    public void setName(String name){
        this.name = name;
    }
    public String getName(){
        return this.name;
    }

    public void setData(ArrayList<String> data) {
        this.data = data;
    }

    public ArrayList<String> getData() {
        return data;
    }



    /**
     * Simplifies the contents of the <code>derivedData</code>
     * ArrayList.
     * @param derivedData
     */
    public void simplifyDerivedData(ArrayList<String>derivedData){



        for(int i=0;i<derivedData.size();i++){
            if(isOpeningBracket(derivedData.get(i)) ){
                int j=i+1;

                while( !isClosingBracket(derivedData.get(j)) && !isOpeningBracket(derivedData.get(j)) ){
                    j++;
                }
                if(isClosingBracket(derivedData.get(j))){//Good deal..you have a (...) pattern with no bracket inside! simplify its contents.
                    try{
                        //Check if the bracket pattern is enclosed directly in another bracket pair. If so unwrap.
                        if(isOpeningBracket(derivedData.get(i-1)) && isClosingBracket(derivedData.get(j+1))){
                            derivedData.remove(j+1);
                            derivedData.remove(i-1);
                            i--;//
                        }
                    }
                    catch(Exception e){}

                }
            }
        }
        //Look for a imple bracket pair and simplify its contents using the Formula class
        for(int i=0;i<derivedData.size();i++){
            if(isOpeningBracket(derivedData.get(i)) ){
                int j=i+1;
                while( !isClosingBracket(derivedData.get(j)) && !isOpeningBracket(derivedData.get(j)) ){
                    j++;
                }
                if(isClosingBracket(derivedData.get(j))){//Good deal..you have a (...) pattern with no bracket inside! simplify its contents.
                    List<String>l = derivedData.subList(i+1, j);
                    //Formula.simplify(l);

                }
            }
        }
        for(int i=0;i<derivedData.size();i++){
            String token = derivedData.get(i);

            if( i+3 < derivedData.size() && ( token.equals("+")||token.equals("-")||token.equals("*")||token.equals("/") )){
                String nextToken = derivedData.get(i+1);
                String foundToken = derivedData.get(i+2);

                if(isOpeningBracket(nextToken) && (isNumber(foundToken) || isVariableString(foundToken) )
                        && isClosingBracket(derivedData.get(i+3))){
                    List subList = derivedData.subList(i+1, i+4);
                    subList.clear();
                    subList.add(foundToken);
                }

            }

            if( i-1 > 0 && token.equals("*") && isNumber(derivedData.get(i-1)) ){
                if(Double.parseDouble(derivedData.get(i-1))==0.0){//what to do when 0, *, patterns are discovered.

                }
                else if(Double.parseDouble(derivedData.get(i-1))==1.0){//clear 1, *, patterns
                    derivedData.subList(i-1, i+1).clear();
                }

            }


        }



    }//end method




    /**
     *
     * @param d The parent Derivative object.
     * @return the List containing the tokens of the derivative.
     */
    @SuppressWarnings("unused")
    public ArrayList<String> differentiate(Derivative d){
        ArrayList<String>derivedData = new ArrayList<>();
        handleStrangeFormats:{
            //turn (,-,var,) to (,-1,*,var,)
            if(data.get(0).equals("(")&&data.get(1).equals("-")&&isVariableString(data.get(2))&&isClosingBracket(data.get(3))){
                data.set(1, "-1");
                data.add(2, "*");
            }
            if(data.get(0).equals("-") && isVariableString(data.get(1))){
                data.set(0, "-1");
                data.add(1, "*");
            }






        }//end block


        boolean autoWrapWithBrackets=false;



        int sz = data.size();
        if(sz==3&&!data.get(0).equals("(")&&!data.get(2).equals(")")){
            data.add(0,"(");data.add(")");
            autoWrapWithBrackets=true;
            sz = data.size();
        }
        else if(sz==1){
            data.add(0,"(");data.add(")");
            sz = data.size();
        }



        if( sz == 3 ){
            if( isOpeningBracket(data.get(0)) && isVariableString(data.get(1)) && isClosingBracket(data.get(2)) ){
                if(d.isBaseVariable(data.get(1))){
                    derivedData.add("1");
                }
                else if(isAutoGenNameFormat(data.get(1))){
                    derivedData.add("diff");
                    derivedData.add("(");
                    derivedData.add(data.get(1));
                    derivedData.add(")");
                }


            }//end if
            else if( isOpeningBracket(data.get(0)) && isNumber(data.get(1)) && isClosingBracket(data.get(2)) ){
                derivedData.add("0");
            }//end if

        }
        else if( sz == 4 ){
            //pattern sin,(,var,)
            if(isMethodName(data.get(0))&&isOpeningBracket(data.get(1))&&isVariableString(data.get(2))&&isClosingBracket(data.get(3))){
                derivedData.addAll(Methods.getMethodDifferential(data.get(0), data.get(2),d));
            }//end if
        }//end else if

        //pattern (,num|var,operator,num|var,)
        else if( sz == 5 ){


            if( isMethodName(data.get(0)) && isOpeningBracket(data.get(1)) && data.get(2).equals("-") && isVariableString(data.get(3))
                    && isClosingBracket(data.get(4)) ){
                derivedData.add("-1");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.addAll(Methods.getMethodDifferential(data.get(0), data.get(3),d));
                derivedData.add(")");
            }

            else if( isNumber(data.get(1)) && isPower(data.get(2)) && isNumber(data.get(3)) ){
                derivedData.add("0");
            }//end if
            //(,num,^,var,)
            else if( isNumber(data.get(1)) && isPower(data.get(2)) && isVariableString(data.get(3)) ){

                derivedData.add("(");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(data.get(2));
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("ln");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            //(,var,^,num,)
            else if( isVariableString(data.get(1)) && isPower(data.get(2)) && isNumber(data.get(3)) ){
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add("^");
                derivedData.add(""+(Double.parseDouble(data.get(3))-1));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            //(,var, ^, var,)
            else if( isVariableString(data.get(1)) && isPower(data.get(2)) && isVariableString(data.get(3)) ){
                derivedData.add("(");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(data.get(2));
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("ln");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("+");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("/");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add(")");
                derivedData.add(")");

            }//end else if
            //(,num, *, var,)
            else if( isNumber(data.get(1)) && data.get(2).equals("*") && isVariableString(data.get(3)) ){
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            // (,var,*,num,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("*") && isNumber(data.get(3)) ){
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            //(,var,*,var,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("*") && isVariableString(data.get(3)) ){
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("+");
                derivedData.add(data.get(1));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if

            //(,num, /, var,)
            else if( isNumber(data.get(1)) && data.get(2).equals("/") && isVariableString(data.get(3)) ){
                derivedData.add("(");
                derivedData.add(""+(-1*Double.parseDouble(data.get(1))) );
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("^");
                derivedData.add("-2");
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            // (,var,/,num,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("/") && isNumber(data.get(3)) ){
                derivedData.add("(");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("/");
                derivedData.add(data.get(3));
                derivedData.add(")");
            }//end else if
            //(,var,/,var,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("/") && isVariableString(data.get(3)) ){

                derivedData.add("(");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("^");
                derivedData.add("-1");
                derivedData.add(")");
                derivedData.add("-");
                derivedData.add(data.get(1));
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add("*");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add("^");
                derivedData.add("-2");
                derivedData.add(")");
                derivedData.add(")");
            }//end else if

            //(,num, +, var,)
            else if( isNumber(data.get(1)) && data.get(2).equals("+") && isVariableString(data.get(3)) ){
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
            }//end else if
            // (,var,+,num,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("+") && isNumber(data.get(3)) ){
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
            }//end else if
            //(,var,+,var,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("+") && isVariableString(data.get(3)) ){
                derivedData.add("(");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("+");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            //(,num, -, var,)
            else if( isNumber(data.get(1)) && data.get(2).equals("-") && isVariableString(data.get(3)) ){

                derivedData.add("(");
                derivedData.add("-1");
                derivedData.add("*");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if
            // (,var,-,num,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("-") && isNumber(data.get(3)) ){
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
            }//end else if
            //(,var,-,var,)
            else if( isVariableString(data.get(1)) && data.get(2).equals("-") && isVariableString(data.get(3)) ){
                derivedData.add("(");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(1));
                derivedData.add(")");
                derivedData.add("-");
                derivedData.add("diff");
                derivedData.add("(");
                derivedData.add(data.get(3));
                derivedData.add(")");
                derivedData.add(")");
            }//end else if

        }//end if


        /**
         *
         * Solve differential forms...pattern diff,(,var,).
         * This is where the CHAIN RULE comes in.
         * Having differentiated the function at the top of the chain,
         * This stage differentiates all other functions connected to
         * the chain.
         */
        for(int i=0;i<derivedData.size();i++){
            try{
                if( derivedData.get(i).equals("diff")&&isOpeningBracket(derivedData.get(i+1))&&
                        isVariableString(derivedData.get(i+2))&&isClosingBracket(derivedData.get(i+3)) ){

                    //print("data---"+data);
                    if(isAutoGenNameFormat(derivedData.get(i+2))){
                        Differentiable diff = d.builder.getManager().lookUp(derivedData.get(i+2));
                        ArrayList<String> derived = diff.differentiate(d);
                        List<String> temp = derivedData.subList(i, i+4);
                        temp.clear();
                        temp.addAll(derived);
                        --i;
                    }//end if
                    else if(d.isBaseVariable(derivedData.get(i+2))){
                        List<String> temp = derivedData.subList(i, i+4);
                        temp.clear();
                        temp.add("1");
                    }
                }//end if
                else if( derivedData.get(i).equals("diff")&&isOpeningBracket(derivedData.get(i+1))&& derivedData.get(i+2).equals("-") &&
                        isVariableString(derivedData.get(i+3))&&isClosingBracket(derivedData.get(i+4)) ){

                    //print("data---"+data);
                    if(isAutoGenNameFormat(derivedData.get(i+3))){
                        Differentiable diff = d.builder.getManager().lookUp(derivedData.get(i+2));
                        ArrayList<String> derived = diff.differentiate(d);
                        List<String> temp = derivedData.subList(i, i+4);
                        temp.clear();
                        temp.add("-1");
                        temp.add("*");
                        temp.add("(");
                        temp.addAll(derived);
                        temp.add(")");

                        --i;
                    }//end if
                    else if(d.isBaseVariable(derivedData.get(i+3))){
                        List<String> temp = derivedData.subList(i, i+4);
                        temp.clear();
                        temp.add("-1");
                    }
                }//end if

            }//end try
            catch(IndexOutOfBoundsException boundsException){

            }//end catch
        }//end for loop


        if(autoWrapWithBrackets && (data.get(0).equals("(")&&data.get(data.size()-1).equals(")")) ){
            data.remove(data.size()-1);
            data.remove(0);
            autoWrapWithBrackets=false;
        }
        if(data.contains("+")||data.contains("-")){
            if(!data.get(0).equals("(") && !data.get(data.size()-1).equals(")")){
                data.add(0,"(");data.add(")");
            }
        }

        if(derivedData.contains("+")||derivedData.contains("-")){
            if(!derivedData.get(0).equals("(")&&!derivedData.get(derivedData.size()-1).equals(")")){
                derivedData.add(0,"(");derivedData.add(")");
            }
        }

        print("data---"+data+"---Diff of data = derivedData---"+derivedData);
        return derivedData;
    }//end method











    /**
     * @param derivedData 
     * @return the derivative as a String object.
     */
    public String getDerivativeExpression(ArrayList<String>derivedData){
        return getText(derivedData);
    }//end method

    /**
     *
     * @return The string format of the data.
     */
    public String getExpression(){
        return getText(data);
    }//end method



//    @Override
//    public String toString() {
//        return "<name,basevariable> = <"+name+","+basevariable+">,data = "+data;
//    }



}

