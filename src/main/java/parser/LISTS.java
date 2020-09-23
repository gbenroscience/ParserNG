/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package parser;
import parser.methods.Method;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
/**
 *
 * @author GBENRO
 * this class contain utility methods that define various operations on
 * storage structures such as lists and arrays.
 */
public class LISTS{









    /**
     *
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search backwards for the first occurrence
     * of  a string that represents a statistical operator( e.g sum(,sort(,med(,st_err( e.t.c ) or a logtoanybase
     * operator e.g log(a,b) where a and b are numbers or an antilogtoanybaseoperator
     * that index itself been not included
     * @return the index of the first occurrence of the object behind index start or -1 if the object is not found.
     */
    public static int firstoccurrenceOfStatsOrLogOrAntilogBehind(int start,List<String>list){
        int index=-1;
        if( start>=0 ){
            for(int i=start-1;i>=0;i--){
                if(Method.isStatsMethod(list.get(i))|| Method.isAntiLogToAnyBase((list.get(i)))
                        || Method.isLogToAnyBase(list.get(i))){
                    index=i;
                    break;
                }//end if
            }//end for

        }//end if
        else{
            throw new IndexOutOfBoundsException("Attempt to access index less than 0");
        }


        return index;
    }//end method firstoccurrenceOfStatsOrLogOrAntilogBehind

    /**
     *
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the first occurrence
     * of  a string that represents a statistical operator( e.g sum(,sort(,med(,st_err( e.t.c ) or a logtoanybase
     * operator e.g log(a,b) where a and b are numbers or an antilogtoanybaseoperator
     * that index itself been not included
     * @return the index of the first occurrence of the object beyond index start or -1 if the object is not found.
     */
    public static int firstoccurrenceOfStatsOrLogOrAntilogBeyond( int start,List<String>list){
        int index=-1;
        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Method.isStatsMethod(list.get(i))|| Method.isAntiLogToAnyBase(list.get(i))
                        || Method.isLogToAnyBase(list.get(i))){
                    index=i;
                    break;
                }//end if
            }//end for

        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }



        return index;
    }//end method firstoccurrenceOfStatsOrLogOrAntilogBeyond

    /**
     *
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search backwards for the object
     * that index itself been not included
     * @param sought the object that we seek
     * @return the index of the first occurrence of the object behind index start or -1 if the object is not found.
     */
    public static int prevIndexOf(List list,int start,Object sought){
        int index=-1;
        if(sought.getClass() == list.get(0).getClass() ){
            if( start>=0 ){
                for(int i=start-1;i>=0;i--){
                    if(list.get(i).equals(sought)){
                        index=i;
                        break;
                    }//end if
                }//end for

            }//end if
            else{
                throw new IndexOutOfBoundsException("Attempt to access index less than 0");
            }
        }//end if
        else{
            throw new ClassCastException("The object types must be the same for the list" +
                    " and the object been searched for in the list");
        }

        return index;
    }//end method prevIndexOf


    /**
     *
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index included. So if the search item is found at the <code>start</code>, the search returns instantly.
     * @param sought the object that we seek
     * @return the index of the first occurrence of the object after start or -1 if the object is not found.
     */
    public static int nextIndexOf(List list,int start,Object sought){
        int index=-1;
        if(sought.getClass() == list.get(0).getClass() ){
            if( start < list.size()  && !list.isEmpty()){
                for(int i=start+1;i<list.size();i++){
                    if(list.get(i).equals(sought)){
                        index=i;
                        break;
                    }//end if
                }//end for

            }//end if
            else{
                throw new IndexOutOfBoundsException("Empty list or end of list reached");
            }
        }//end if
        else{
            throw new ClassCastException("The object types must be the same for the list" +
                    " and the object been searched for in the list");
        }

        return index;
    }//end method



    /**
     *Searches a List of String objects forwards for the first occurrence of a * or / or % operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfMulOrDivOrRem(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isMulOrDiv(list.get(i))||Operator.isRemainder(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfMulOrDivOrRem


    /**
     *Searches a List of String objects backwards for the first occurrence of a * or / or % operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfMulOrDivOrRem(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isMulOrDiv(list.get(i))||Operator.isRemainder(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfMulOrDivOrRem





    /**
     *Searches a List of String objects forwards for the first occurrence of a power operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfPowerOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isPower(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfPowerOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a power operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfPowerOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isPower(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfPowerOperator
    /**
     *Searches a List of String objects forwards for the first occurrence of a Variable String
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int  nextIndexOfVariable(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Variable.isVariableString(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfVariable

    /**
     *Searches a List of String objects backwards for the first occurrence of a Variable String
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfVariable(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Variable.isVariableString(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfVariable

    /**
     *Searches a List of String objects forwards for the first occurrence of a permutation or combination operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int  nextIndexOfPermOrCombOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isPermOrComb(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfPermOrCombOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a permutation or combination operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfPermOrCombOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isPermOrComb(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfPermOrCombOperator













    /**
     *Searches a List of String objects forwards for the first occurrence of an In Between Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfInBetweenOperatorOrComma(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isBinaryOperator(list.get(i))||Operator.isComma(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfInBetweenOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of an In Between Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfInBetweenOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isBinaryOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfInBetweenOperator



    /**
     *Searches a List of String objects forwards for the first occurrence of a Pre Number Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfPreNumberOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isUnaryPreOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfPreNumberOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a Pre-Number Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfPreNumberOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isUnaryPreOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfPreNumberOperator



    /**
     *Searches a List of String objects forwards for the first occurrence of a Post Number Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfPostNumberOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isUnaryPostOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfPostNumberOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a Post Number Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfPostNumberOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isUnaryPostOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfPostNumberOperator



    /**
     *Searches a List of String objects forwards for the first occurrence of a Number Type Stats Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfNumberReturningStatsOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Method.isNumberReturningStatsMethod(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfNumberReturningStatsOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a NumberReturningStatsOperator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfNumberReturningStatsOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Method.isNumberReturningStatsMethod(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfNumberReturningStatsOperator





    /**
     *Searches a List of String objects forwards for the first occurrence of a ListTypeStats Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfListReturningStatsOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Method.isListReturningStatsMethod(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfListReturningStatsOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a ListReturningStatsOperator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfListReturningStatsOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Method.isListReturningStatsMethod(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfListReturningStatsOperator






    /**
     *Searches a List of String objects forwards for the first occurrence of a LogOrAntiLogToAnyBase Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfLogOrAntiLogToAnyBase(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Method.isLogOrAntiLogToAnyBase(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfLogOrAntiLogToAnyBase


    /**
     *Searches a List of String objects backwards for the first occurrence of a LogOrAntiLogToAnyBase Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfLogOrAntiLogToAnyBase(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Method.isLogOrAntiLogToAnyBase(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfLogOrAntiLogTOAnyBase







    /**
     *Searches a List of String objects forwards for the first occurrence of a Logic Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfLogicOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isLogicOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfLogicOperator


    /**
     *Searches a List of String objects backwards for the first occurrence of a Logic Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfLogicOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isLogicOperator(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfLogicOperator





    /**
     *Searches a List of String objects forwards for the first occurrence of a Comma String object
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfComma(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isComma(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfComma


    /**
     *Searches a List of String objects backwards for the first occurrence of a Comma String Object
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfComma(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isComma(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfComma







    /**
     *Searches a List of String objects forwards for the first occurrence of a Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int nextIndexOfOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start+1;i<list.size();i++){
                if(Operator.isOperatorString(list.get(i))){
                    index=i;
                    break;
                }//end if

            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method nextIndexOfOperator
    /**
     *
     * @param list The List from which the copy is to be done
     * @param start The index to start the copy from
     * @param end The index where the copy is to end(excluded).
     * @return a String that contains the concatenation of all
     * items between index start and index end(excluded.) of the List.
     */
    public static String createStringFrom(List<String>list,int start,int end){
        if( (start >= 0 && end <= list.size() && end > start)  ){
            StringBuilder copy = new StringBuilder();
            for(int i=start;i<end;i++){
                copy=copy.append(list.get(i));
            }//end for
            return copy.substring(0,copy.length());
        }//end if

        throw new  IndexOutOfBoundsException(" The start index and end index must be greater than or equal to zero\n"
                + "and the end index must be greater than the start index!---start: "+start+"----end: "+end);
    }//end method


    /**
     *Searches a List of String objects backwards for the first occurrence of a Operator
     * from the point where the search commences.
     * @param list the collection of objects that the search is to be carried out on
     * @param start the starting index of the search from where we search forwards for the object
     * that index itself been not included
     * @return the index of the first occurrence of the String object after start or -1 if the object is not found.
     */
    public static int prevIndexOfOperator(List<String> list,int start){
        int index=-1;


        if( start<=list.size()-2 ){
            for(int i=start-1;i>=0;i--){
                if(Operator.isOperatorString(list.get(i))){
                    index=i;
                    break;
                }//end if


            }//end for
        }//end if
        else{
            throw new IndexOutOfBoundsException("Cannot Search Beyond The Last Item Of This List");
        }




        return index;
    }//end method prevIndexOfOperator




    /**
     *
     * @param list the collection of objects that we wish to modify
     * @param start the index at which we start removing items from the list(start inclusive)
     * @param end   the index at which we stop removing items (end exclusive)
     * @return the list without all elements between start(start inclusive) and end(not inclusive)
     * e.g for list L=[0,1,2,3,4,5,6,7,8,9],cutPortionOfList( L,6,8) = [0,1,2,3,4,5,8,9]
     */
    public static List cutPortionOfList(List list,int start, int end){

        if(start>=0&&start<=end&&end<=list.size()){
            list.subList(start, end).clear();
        }
        else if(start<0){
            throw new IndexOutOfBoundsException("The start index cannot be less than 0.");
        }
        else if(end>list.size()){
            throw new IndexOutOfBoundsException("The end index cannot be same as or more than the list's size.");
        }
        else if(start>end){
            throw new IndexOutOfBoundsException("The starting index cannot be more than the ending index.");
        }

        return list;
    }//end method prevIndexOf

    public static void main(String[] args) {
        System.err.println(createStringFrom(new ArrayList<>(Arrays.asList("diff","(","@","(","x",")", "3*x+1","2",")")), 0, 9));
    }

}
