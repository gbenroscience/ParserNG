/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import java.util.ArrayList;

/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class HalfDet extends MatrixOperator{
private int index;
private HalfDet complement;
    public HalfDet(int index,ArrayList<String>scan){
        super("|");
              if(this.getName().equals("")){
            throw new IndexOutOfBoundsException("Invalid Name For Half-Determinant MOperator."  );
        }//end if

        else{
            this.index=(index>=0&&scan.get(index).equals("|"))?index:-1;
        }//end else

    if(this.index==-1){
                    throw new IndexOutOfBoundsException("Invalid Index"  );
    }

    }

    public void setIndex(int index) {
        this.index = index;
    }

    public int getIndex() {
        return index;
    }

    public void setComplement(HalfDet complement) {
        this.complement = complement;
    }

    public HalfDet getComplement() {
        return complement;
    }
/**
 *
 * @param isleftHalf A boolean condition that is true if the object (whose complement we seek) itself is the opening
 * half of the determinant brace...i.e the one to the left. If false, then it is the one to the right...i.e the one to the right.
 * @param start The index of the HalfDet object whose complement we seek.
 * @param scan The ArrayList that this object resides in.
 * @return the index where the complement is to be found in the ArrayList object.
 */
    public static int getComplementIndex( boolean isleftHalf, int start, ArrayList<String>scan   ){

int index=0;
    if(isleftHalf){

    for(index=start+1;index<scan.size();index++){
if(scan.get(index).equals("|")){
     return index;
        }
    }//end for
    }//end if
    else if(!isleftHalf){
       
    for(index=start-1;index>=0;index--){
if(scan.get(index).equals("|")){
     return index;
        }
    }//end for
        
    }//end else if
return -1;


    }









}//end class
