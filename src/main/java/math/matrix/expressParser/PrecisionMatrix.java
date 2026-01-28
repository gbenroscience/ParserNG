/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Random;

/**
 * Deals with numbers of higher
 * precision than Matrix.
 * @JIBOYE OLUWAGBEMIRO OLAOLUWA
 */
public class PrecisionMatrix{




/**
 * The simple name used to
 * label this PrecisionMatrix object.
 *
 */
     private String name;
     /**
    * the data array
    * used to create this PrecisionMatrix object
    */
     private BigDecimal array[][];
/**
 * attribute used to compute the determinant
 * of the PrecisionMatrix object.
 */
     private static BigDecimal det=BigDecimal.ZERO;
/**
 *
 * @param rows The number of rows in the PrecisionMatrix.
 * @param cols The number of columns in the PrecisionMatrix.
 */
    public PrecisionMatrix( int rows, int cols) {
         this("NEW");
     array=new BigDecimal[rows][cols];
    }

/**
 * @param name the simple name used to identify used by the user to
 * label this PrecisionMatrix object.
 *
 * Assigns name unknown to the PrecisionMatrix object
 * and a 2D array that has just a row and a column
 *
 */
    public PrecisionMatrix(String name) {
        this.name = name;
        this.array=new BigDecimal[][]{{new BigDecimal("0.0")}};
    }//end constructor


/**
 *
 * @param array the data array
 * used to create this PreicionMatrix object
 */
     public PrecisionMatrix(BigDecimal[][] array){
         this("NEW");
     setArray(array);
     }//end constructor


/**
 *
 * @param matrix constructs a new PrecisionMatrix object having
 * similar properties to the one passed as argument,
 * but holding no reference to its data.
 */
     public PrecisionMatrix(PrecisionMatrix matrix){
      this("NEW");
    BigDecimal arr[][]=new BigDecimal[matrix.getRows()][matrix.getCols()];
/* Copy the array of the PrecisionMatrix object parameter into
 * a separate array object and use for the
 * PrecisionMatrix object about to be created.
 * This ensures that the new PrecisionMatrix object
 * has no connection to the one from which it is created
 * so that it is a true twin or duplicate of the other one.
 * The user is free to perform operations on this one without fear
 * that it will cause changes in the other one.
 *
 *
 */

   for(int row=0;row<matrix.getRows();row++){
    for(int col=0;col<matrix.getCols();col++){
        BigDecimal val=matrix.array[row][col];
      arr[row][col]=val;
    }//end inner loop
    }//end outer loop
         this.array=arr;
     }//end constructor




/**
 *
 * @return  the number of rows in this matrix object
 */
    public int getRows() {
        return array.length;
    }

/**
 *
 * @return  the number of columns in this matrix object
 */
    public int getCols() {
     return array[0].length;
    }



/**
 *
 * @param array sets the data of this matrix
 */
    public void setArray(BigDecimal[][] array) {

        if(array.length>0){
         this.array=new BigDecimal[array.length][array[0].length];
     for(int row=0;row<array.length;row++){
    for(int col=0;col<array[0].length;col++){

      this.array[row][col]=array[row][col];
    }//end inner loop
    }//end outer loop
        }
        else{
            this.array=new BigDecimal[][]{{new BigDecimal("0")},{new BigDecimal("0")}}; 
        }
    }//end method
/**
 *
 * @return the data of this matrix
 */
    public BigDecimal[][] getArray() {
        return this.array;
    }
/**
 *
 * @param det the determinant attribute of
 * objects of this class
 */
    private static void setDet(BigDecimal det) {
        PrecisionMatrix.det = det;
    }
/**
 *
 * @return the determinant
 */
    private static BigDecimal getDet() {
        return det; 
    }
/**
 *
 * @param name set's the simple name used to identify used by the user to
 * label this PrecisionMatrix object.
 */
    public void setName(String name) {
        this.name = name;
    }
/**
 *
 * @return the simple name used to identify used by the user to
 * label this PrecisionMatrix object.
 */
    public String getName() {
        return name;
    }
/**
 * Fills this matrix object with values
 */
    public void fill(){
        java.util.Scanner scanner = new java.util.Scanner(System.in);

for(int row=0;row < getRows();row++){

      for (int column=0;column<getCols();column++){
          array[row][column]=scanner.nextBigDecimal();
      }
}
    }//end method fill.
/**
 * Fills the matrix with randomly generated values between
 * 1 and 101.
 */
public void randomFill(){
Random ran=new Random();
    for(int row=0;row<getRows();row++){
    for(int col=0;col<getCols();col++){
   array[row][col]=new BigDecimal("1.0").add( new BigDecimal( String.valueOf(ran.nextInt(101)) ) );
    }//end inner for
    }//end outer for
    }//end method

/**
 * Fills the matrix with randomly generated values
 * between 1 and n
 * @param n the maximum possible size of the integer
 * numbers generated.
 */
public void randomFill(int n){
Random ran=new Random();
    for(int row=0;row<getRows();row++){
    for(int col=0;col<getCols();col++){
   array[row][col]= BigDecimal.valueOf( 1+ran.nextInt(n) );



    }//end inner for



    }//end outer for

}//end method randomFill


    /**
     *
     * @param matrice the matrix to be added to this one.
     * The operation is ( this + matrice )
     * @return an array containing the product matrix.
     */
    public PrecisionMatrix add(PrecisionMatrix matrice){

BigDecimal array1[][]=matrice.array;

        BigDecimal matrix[][]=new BigDecimal[getRows()][getCols()];

        if(getRows()==matrice.getCols()&&getCols()==matrice.getRows()){
       for(int row=0;row < getRows();row++){

      for (int column=0;column<getCols();column++){

matrix[row][column]=(array[row][column]).add(array1[row][column]);



      }//end inner for

}//end outer for

        }//end if
        else{
             util.Utils.logError("ERROR IN MATRIX INPUT!!");
        }

return new PrecisionMatrix(matrix);
    }//end method add



    /**
     *
     * @param matrice the matrix to be subtracted from this one.
     * The operation is ( this - matrice )
     * @return an array containing the product matrix.
     */
    public PrecisionMatrix subtract(PrecisionMatrix matrice){

BigDecimal array1[][]=matrice.array;

        BigDecimal matrix[][]=new BigDecimal[getRows()][getCols()];

        if(getRows()==matrice.getCols()&&getCols()==matrice.getRows()){
       for(int row=0;row < getRows();row++){

      for (int column=0;column<getCols();column++){

matrix[row][column]=(array[row][column]).subtract((array1[row][column]));



      }//end inner for

}//end outer for

        }//end ifghjjk
        else{
             util.Utils.logError("ERROR IN MATRIX INPUT!!");
        }//end else
return new PrecisionMatrix(matrix);
    }//end method subtract

    /**
     *
     * @param scalar the constant to be multiplied with this matrix
     * The operation is ( scalar X matrice )
     * @return an array containing the product matrix.
     */
    public PrecisionMatrix scalarMultiply(double scalar){

BigDecimal scale = new BigDecimal(String.valueOf(scalar));

        BigDecimal matrix[][]=new BigDecimal[getRows()][getCols()];
       for(int row=0;row < getRows();row++){
      for (int column=0;column<getCols();column++){
matrix[row][column]=(array[row][column]).multiply(scale,MathContext.DECIMAL128);
      }//end inner for
}//end outer for

return new PrecisionMatrix(matrix);
    }//end method scalarMultiply


    /**
     *
     * @param scalar the constant to be multiplied with this matrix
     * The operation is (  matrice/scalar )
     * @return an array containing the scaled matrix.
     */
    public PrecisionMatrix scalarDivide(double scalar){
BigDecimal scale = new BigDecimal(String.valueOf(scalar));
        BigDecimal matrix[][]=new BigDecimal[getRows()][getCols()];
       for(int row=0;row < getRows();row++){
      for (int column=0;column<getCols();column++){
matrix[row][column]=(array[row][column]).divide(scale,MathContext.DECIMAL128);
      }//end inner for
}//end outer for
return new PrecisionMatrix(matrix);
    }//end method scalarDivide

    /**
     *
     * The operation of matrix multiplication.
     * For this method to run, The pre-multiplying
     * matrix must have its number of columns equal
     * to the number of rows in the pre-multiplying one.
     *
     * The product matrix is one that has its number of columns equal
     * to the number of columns in the pre-multiplying matrix, and its rows
     * equal to that in the post-multiplying matrix.
     *
     *
     * So if the operation is A X B = C, and A is an m X n matrix while B is an r X p
     * matrix, then r = n is a necessary condition for the operation to occur.
     * Also, C is an m X p matrix.
     *
     * @param matrice1 the matrix to be pre-multiplying the other one.
     * The operation is ( matrice1 X matrice2 )
     * @param matrice2 the post-multiplying matrix
     * @return a new PrecisionMatrix object containing the product matrix
     * of the operation matrice1 X matrice2
     */
    public static PrecisionMatrix multiply(PrecisionMatrix matrice1,PrecisionMatrix matrice2){
 PrecisionMatrix m= new PrecisionMatrix(matrice1.getRows(),matrice2.getCols());
        if(matrice1.getCols()==matrice2.getRows()){
      for(int i=0;i<matrice1.getRows();i++){
       for(int row=0; row < matrice2.getCols(); row++){
       BigDecimal sum= new BigDecimal("0");
      for (int column=0;column<matrice1.getCols();column++){
sum = sum.add( matrice1.array[i][column].multiply(matrice2.array[column][row],MathContext.DECIMAL128) );
      }//end inner for
    m.array[i][row]=sum;
}//end outer for
        }//end outermost for
        }//end if
        else{
             util.Utils.logError("ERROR IN MATRIX INPUT!!");
        }
 return m;
    }//end method
    /**
     *
     * @param matrice the matrix to be multiplied by this one.
     * This operation modifies this matrix and changes its data array
     * into that of the product matrix
     * The operation is ( this X matrice )
     */
    public void multiply(PrecisionMatrix matrice){
setArray(   PrecisionMatrix.multiply(this,matrice).array );
    }//end method
/**
 *
 * @param mat the matrix to raise to a given power
 * @param pow the power to raise this matrix to
 * @return the matrix multiplied by itself..pow times
 */
 public static PrecisionMatrix pow(PrecisionMatrix mat,int pow){
       BigDecimal matrix[][]=new BigDecimal[mat.getRows()][mat.getCols()];
       PrecisionMatrix m = new PrecisionMatrix(mat.array);
       for(int i=0;i<pow-1;i++){
       m = PrecisionMatrix.multiply(m, mat);

       }
     return m;
 }//end method

/**
 *
 * @return a unit matrix of the same
 * dimension as this matrix object
 */
 public PrecisionMatrix unitMatrix(){
       BigDecimal matrix[][]=new BigDecimal[getRows()][getCols()];
for(int row=0;row < getRows();row++){
for (int column=0;column<getCols();column++){
    if(column==row){
matrix[row][column]=new BigDecimal("1");
    }//end if
    else{
matrix[row][column]=new BigDecimal("0");
    }//end else
      }//end inner for loop
        }//end outer for loop
       return new PrecisionMatrix(matrix);
 }//end method unitMatrix



/**
 *
 * @param rowSize the number of rows that the unit matrix will have
 * @param colSize the number of columns that the unit matrix will have
 * @return a unit matrix having number of rows = rowSize
 * and number of columns=colSize.
 */
 public static PrecisionMatrix unitMatrix(int rowSize,int colSize){
        BigDecimal matrix[][]=new BigDecimal[rowSize][colSize];

for(int row=0;row < rowSize;row++){
for (int column=0;column<colSize;column++){
    if(column==row){
matrix[row][column]=BigDecimal.ONE;
    }
    else{
matrix[row][column]=BigDecimal.ZERO;
    }


      }//end inner for loop
        }//end outer for loop
       return new PrecisionMatrix(matrix);
 }//end method unitMatrix

/**
 *
 * @param mat the PrecisionMatrix object that we wish to
 * construct a unit matrix of similar dimensions for.
 * @return a unit matrix of equal dimensions as this
 * unit matrix.
 */
 public static PrecisionMatrix unitMatrix(PrecisionMatrix mat){
     int rowSize=mat.getRows();
     int colSize=mat.getCols();
        BigDecimal matrix[][]=new BigDecimal[rowSize][colSize];

for(int row=0;row < rowSize;row++){
for (int column=0;column<colSize;column++){
    if(column==row){
matrix[row][column]=BigDecimal.ONE;
    }
    else{
matrix[row][column]=BigDecimal.ZERO;
    }
      }//end inner for loop
        }//end outer for loop
       return new PrecisionMatrix(matrix);
 }//end method unitMatrix

 


/**
 * Place the first PrecisionMatrix object side by side with
 * the second one passed as argument to this method.
 * The result is a new matrix where:
 *
 * if        3 4 5              7 5 9
 *    mat1 = 2 3 1  and mat2 =  4 2 6
 *           1 6 7              5 7 3
 * in a new matrix object (mat).
 *
 *
 * e.g    3 4 5 7 5 9
 *        2 3 1 4 2 6
 *        1 6 7 5 7 3
 * A necessary condition for this method to run
 * is that the 2 objects must have an equal number of
 * rows. IF THIS CONDITION IS NOT MET,
 * THE METHOD RETURNS A ZERO MATRIX.
 *
 *
 * @param mat1 the first PrecisionMatrix object
 * @param mat2 the second PrecisionMatrix object that we column join with this one
 * @return a new PrecisionMatrix object that contains this PrecisionMatrix
 * object placed side by side with the PrecisionMatrix object passed as argument.
 */
 public static PrecisionMatrix columnJoin(PrecisionMatrix mat1,PrecisionMatrix mat2){

     PrecisionMatrix join= new PrecisionMatrix(mat1.getRows(),mat1.getCols()+mat2.getCols());
     if(mat1.getRows()==mat2.getRows()){
int columnextender=0;
for(int row=0;row<mat1.getRows();row++){

    for(int col=0;col<join.getCols();col++){
        if(col<mat1.getCols()){
            columnextender=0;//reset to zero
join.array[row][col]=mat1.array[row][col];
        }
        else if(col>=mat1.getCols()){
join.array[row][col]=mat2.array[row][columnextender];
columnextender++;
        }
}//end inner for
}//end outer for



     }//end if
 return join;
 }
/**
 *
 * @param value The value to insert
 * @param row The row where the value is to be inserted.
 * @param column The column where the value is to be inserted.
 */
 public void update( BigDecimal value,int row , int column ){
     if(row<getRows()&&column<getCols()){
array[row][column]=value;
     }
 }


 /**
 * Place the first PrecisionMatrix object side by side with
 * the second one passed as argument to this method.
 * The result is a new PrecisionMatrix where:
 *
 * if        3 4 5              7 5 9
 *    mat1 = 2 3 1  and mat2 =  4 2 6
 *           1 6 7              5 7 3
 * in a new PrecisionMatrix object (mat).
 *
 *
 * e.g  3 4 5
  *     2 3 1
  *     1 6 7
  *     7 5 9
  *     4 2 6
  *     5 7 3
  *
 * A necessary condition for this method to run
 * is that the 2 objects must have an equal number of
 * columns. IF THIS CONDITION IS NOT MET,
 * THE METHOD RETURNS A ZERO PrecisionMatrix.
 *
 * @param mat1 the first PrecisionMatrix object
 * @param mat2 the second PrecisionMatrix object that we row join with this one
 * @return a new PrecisionMatrix object that contains the first PrecisionMatrix
 * object argument placed top to bottom with the second PrecisionMatrix object argument.
 */
 public static PrecisionMatrix rowJoin(PrecisionMatrix mat1,PrecisionMatrix mat2){

     PrecisionMatrix join= new PrecisionMatrix(mat1.getRows()+mat2.getRows(),mat1.getCols());
     if(mat1.getCols()==mat2.getCols()){
int rowextender=0;
for(int row=0;row<join.getRows();row++){
    for(int col=0;col<join.getCols();col++){
        if(row<mat1.getRows()){
 join.array[row][col]=mat1.array[row][col] ;
        }
        else if(row>=mat1.getRows()){
 join.array[row][col]=mat2.array[rowextender][col];
        }


}//end inner for
    if(row>=mat1.getRows()){
     rowextender++;
    }
}//end outer for



     }//end if
 return join;
 }//end method rowJoin








/**
 * Deletes all the specified number of columns
 * from the PrecisionMatrix object starting from the end of the PrecisionMatrix object
 * @param column the number of columns to
 * remove from the PrecisionMatrix object.
 * This method will take the object that
 * calls it and perform this operation on it.
 * So it modifies the PrecisionMatrix object that calls it.
 * Be careful, as data will be lost.
 *
 * e.g if     3 4 5 6 7 8 9
 *            2 1 8 1 4 7 0
 *       A =  3 3 2 1 5 7 1
 *
 *       then the call:
 *
 *      A.columnDeleteFromEnd(3)
 *
 *     will delete the last three columns in this object leaving:
 *
 *               3 4 5 6
 *          A =  2 1 8 1
 *               3 3 2 1
 *
 *
 */
 public void columnDeleteFromEnd(int column){
      if(column>=0&&column<=getCols()){
PrecisionMatrix matrix=new PrecisionMatrix(this.getRows(),this.getCols()-column);
     for(int row=0;row<getRows();row++){
     for(int col=0;col<matrix.getCols();col++){
     matrix.array[row][col]=this.array[row][col];
     }//end inner for
     }//end outer for
this.setArray(matrix.array);
      }
      else{
     util.Utils.logError("COLUMN VALUE SHOULD " +
            "RANGE FROM ZERO TO THE NUMBER OF COLUMNS IN THIS MATRIX.");
      }
 }//end method columnDeleteFromEnd


 /**
 * Deletes all the specified number of columns
 * from the PrecisionMatrix object from the beginning of the PrecisionMatrix object
 * @param column the number of columns to
 * remove from the PrecisionMatrix object's beginning.
 * This method will take the object that
 * calls it and perform this operation on it.
 * So it modifies the PrecisionMatrix object that calls it.
 * Be careful, as data will be lost.
 *
 * e.g if     3 4 5 6 7 8 9
 *            2 1 8 1 4 7 0
 *       A =  3 3 2 1 5 7 1
 *
 *       then the call:
 *
 *      A.columnDeleteFromStart(3)
 *
 *     will delete the last three columns in this object leaving:
 *
 *              6 7 8 9
 *          A=  1 4 7 0
 *              1 5 7 1
 *
 *
 */
 public void columnDeleteFromStart(int column){
     if(column>=0&&column<=getCols()){
PrecisionMatrix matrix=new PrecisionMatrix(this.getRows(),this.getCols()-column);
     for(int row=0;row<getRows();row++){
         int counter=0;
     for(int col=column;col<getCols();col++,counter++){
     matrix.array[row][counter]=this.array[row][col];
     }//end inner for
     }//end outer for


this.setArray(matrix.array);
     }
     else{
     util.Utils.logError("COLUMN VALUE SHOULD " +
            "RANGE FROM ZERO TO THE NUMBER OF COLUMNS IN THIS MATRIX.");
     }
     }//end method columnDeleteFromStart

/**
 * Deletes the specified number of rows
 * from the end of the PrecisionMatrix object
 * @param numOfRows the number of rows to
 * remove from the PrecisionMatrix object's beginning.
 * This method will take the object that
 * calls it and perform this operation on it.
 * So it modifies the PrecisionMatrix object that calls it.
 * Be careful, as data will be lost.
 *
 * e.g if     3 4 5 6
 *            2 1 8 1
 *       A =  3 3 2 1
 *            7 8 9 2
 *            4 7 0 5
 *            5 7 1 8
 *       then the call:
 *
 *      A.rowDeleteFromEnd(3)
 *
 *     will delete the last three rows in this object leaving:
 *
 *
 *            3 4 5 6
 *            2 1 8 1
 *       A =  3 3 2 1
 *
 *
 */
 public void rowDeleteFromEnd(int numOfRows){
      if(numOfRows>=0&&numOfRows<=getRows()){
  PrecisionMatrix matrix=new PrecisionMatrix(this.getRows()-numOfRows,this.getCols());
     for(int row=0;row<matrix.getRows();row++){
     for(int col=0;col<getCols();col++){
     matrix.array[row][col]=this.array[row][col];
     }//end inner for

     }//end outer for


this.setArray(matrix.array);
     }
     else{
     util.Utils.logError("NUMBER OF ROWS TO BE DELETED SHOULD " +
            "RANGE FROM ZERO TO (AND INCLUDING) THE NUMBER OF ROWS IN THIS MATRIX.");
     }
 }//end method rowDeleteFromEnd


 /**
 * Deletes the specified number of rows from
  * the beginning of the PrecisionMatrix object
 * @param numOfRows the number of rows to
 * remove from the PrecisionMatrix object's beginning.
 * This method will take the object that
 * calls it and perform this operation on it.
 * So it modifies the PrecisionMatrix object that calls it.
 * Be careful, as data will be lost.
 *
 * e.g if     3 4 5 6
 *            2 1 8 1
 *       A =  3 3 2 1
 *            7 8 9 2
 *            4 7 0 5
 *            5 7 1 8
 *       then the call:
 *
 *      A.rowDeleteFromStart(3)
 *
 *     will delete the last three rows in this object leaving:
 *
 *
 *       A =  7 8 9 2
 *            4 7 0 5
 *            5 7 1 8
 *
 *
 */
 public void rowDeleteFromStart(int numOfRows){
     if(numOfRows>=0&&numOfRows<=getRows()){
PrecisionMatrix matrix=new PrecisionMatrix(this.getRows()-numOfRows,this.getCols());
         int counter=0;
     for(int row=numOfRows;row<getRows();row++,counter++){

     for(int col=0;col<getCols();col++){
     matrix.array[counter][col]=this.array[row][col];
     }//end inner for

     }//end outer for


this.setArray(matrix.array);
     }
     else{
     util.Utils.logError("NUMBER OF ROWS TO BE DELETED SHOULD " +
            "RANGE FROM ZERO TO (AND INCLUDING) THE NUMBER OF ROWS IN THIS MATRIX.");
     }
     }//end method rowDeleteFromStart



/**
 *
 * @return an upper triangular matrix
 * obtained by row reduction.
 */
public PrecisionMatrix reduceToTriangularMatrix(){
    PrecisionMatrix mat= new PrecisionMatrix(this);
    //Now we row-reduce.
for(int j=0;j<mat.getRows();j++){
for(int row=0;row<mat.getRows();row++){
BigDecimal val = mat.array[row][row];
for(int col=0;col<mat.getCols();col++){
mat.array[row][col]=mat.array[row][col].divide(val,MathContext.DECIMAL128);
}//end inner for loop

for(int rowed=row+1;rowed<mat.getRows();rowed++){
    BigDecimal mul=mat.array[rowed][row];
for(int coled=row;coled<mat.getCols();coled++){
mat.array[rowed][coled]=mat.array[rowed][coled].subtract(mul.multiply(mat.array[row][coled],
        MathContext.UNLIMITED),  MathContext.UNLIMITED);
//mat.array[rowed][coled]=mat.array[rowed][coled]-mul*mat.array[row][coled];
}//end for
}//end for


}//end outer for loop
}//end outermost for loop.




return mat;
}




/**
 * Used to solve a system of simultaneous equations placed in this
 * PrecisionMatrix object.
 * @return a PrecisionMatrix object containing
 * the solution matrix.
 */
public PrecisionMatrix solveEquation(){
return PrecisionMatrix.solveEquation(this);
}//end method solnMatrix


/**
 * Used to solve a system of simultaneous equations placed in the
 * PrecisionMatrix object.
 * @param matrix The row X row+1 matrix, containing the system of linear equations
 * @return a PrecisionMatrix object containing
 * the solution matrix.
 */
public static PrecisionMatrix solveEquation(PrecisionMatrix matrix){
    PrecisionMatrix solnMatrix=new PrecisionMatrix(matrix.getRows(),1);

    PrecisionMatrix matrixLoader=matrix.reduceToTriangularMatrix();
  //   util.Utils.logError( matrixLoader  );


if(matrix.getRows()==matrix.getCols()-1){
    //Back-Substitution Algorithm.
    BigDecimal sum= BigDecimal.ZERO;//summation variable
    int counter=1;
    for(int row=matrixLoader.getRows()-1;row>=0;row--){
        for(int col=row+1;col<matrixLoader.getCols();col++){

        if(col<matrixLoader.getCols()-1){
sum = sum.add( matrixLoader.array[row][col].multiply(solnMatrix.array[col][0],
        MathContext.DECIMAL128) , MathContext.DECIMAL128);
//sum+=( matrixLoader.array[row][col]*solnMatrix.array[col][0] );//sum the product of each coefficient and its unknown's value in the solution matrix
            }
        else if(col==matrixLoader.getCols()-1){
 sum=matrixLoader.array[row][col].subtract(sum, MathContext.DECIMAL128);//end of summing.Now subtract the sum from the last entry on the row
            }
        }//end inner loop
    solnMatrix.array[matrixLoader.getRows()-counter][0]= sum.divide(matrixLoader.array[row][row], MathContext.UNLIMITED);
      counter++;// increment counter
      sum=BigDecimal.ZERO;//reset sum
    }//end outer loop

}//end if
else{
throw new IndexOutOfBoundsException("Invalid System Of Linear Equations");
}//end else

return solnMatrix;
}//end method solnMatrix






/**
 *
 * @return the transpose of this PrecisionMatrix object.
 * Does not modify this matrix object but generates the
 * transpose of this PrecisionMatrix object as another PrecisionMatrix object.
 */
 public PrecisionMatrix transpose(){
BigDecimal matrix[][]=new BigDecimal[getRows()][getCols()];

for(int row=0;row<getRows();row++){
    for(int col=0;col<getCols();col++){
matrix[row][col]=array[col][row];
    }//end inner for


}//end outer for

return new PrecisionMatrix(matrix);
 }//end method transpose





/**
 *
 * @param i the row on which the element whose minor is needed lies.
 * @param j the column on which the element whose minor is needed lies.
 * @return the minor of this matrix relative to this element.
 */
public PrecisionMatrix minor(int i,int j){

BigDecimal matrix[][]=new BigDecimal[getRows()-1][getCols()-1];

       for(int row=0;row < getRows();row++){

      for (int column=0;column<getCols();column++){
if(row<i&&column<j){
matrix[row][column]=array[row][column];
}
else if(row<i&&column>j){
matrix[row][column-1]=array[row][column];
}
else if(row>i&&column<j){
matrix[row-1][column]=array[row][column];
}
else if(row>i&&column>j){
matrix[row-1][column-1]=array[row][column];
}
}//end inner for


}//end outer for



return new PrecisionMatrix(matrix);
}//end method minor

/**
 *
 * @return true if this PrecisionMatrix object
 * is a square matrix.
 */
public boolean isSquareMatrix(){
    return getRows()==getCols();
}

/**
 *
 * @return true if this PrecisionMatrix object
 * can represent a system of equations
 * solvable by reduction to triangular
 * form and subsequent evaluation.
 */
public boolean isSystemOfEquations(){
    return getRows()==getCols()-1;
}


/**
 *
 * @param m a 2 X 2 matrix
 * @return the determinant of this matrix
 */

private static BigDecimal $2X2determinant(PrecisionMatrix m){
    return m.array[0][0].multiply(m.array[1][1], MathContext.DECIMAL128).subtract(  m.array[1][0].multiply(m.array[0][1], MathContext.DECIMAL128) , MathContext.DECIMAL128);
}
/**
 * @param m the matrix whose top row is to be multiplied by a scalar
 * @param scalar The scalar to employ in multiplying the top row.
 * Multiplies the top row of a matrix by a scalar.
 * This is an important operation during the evaluation of a determinant.
 */
private static PrecisionMatrix topRowScalarMultiply(PrecisionMatrix m,BigDecimal scalar){
     for(int col=0;col<m.getCols();col++){
       m.array[0][col] = m.array[0][col].multiply(scalar, MathContext.DECIMAL128);
     }
return new PrecisionMatrix(m.array);
}



public static void mdeterm(){





}


/**
 *
 * @param m the PrecisionMatrix object whose
 * determinant is desired.
 * @return the determinant of the matrix
 */
private static BigDecimal det(PrecisionMatrix m){
    //must be square matrix
    if(m.getRows()==m.getCols()){

if(m.getRows()==2){
    return $2X2determinant(m);
}//end else
else{
BigDecimal[] topRow=new BigDecimal[m.getRows()];
PrecisionMatrix[] matrix=new PrecisionMatrix[m.getRows()];
        for(int col=0;col<m.getCols();col++){
          topRow[col]=m.array[0][col].multiply(BigDecimal.valueOf(Math.pow(-1,col)), MathContext.DECIMAL128);
          PrecisionMatrix mat=topRowScalarMultiply(  m.minor(0,col),topRow[col] );
          matrix[col]=mat;


          if(matrix[col].getRows()>2){
         det(matrix[col]);
          }
          else{
            det = det.add($2X2determinant(matrix[col]),MathContext.DECIMAL128);
          }


        }//end for

return det;
}//end else

    }//end if
    else{
       return BigDecimal.valueOf( Double.POSITIVE_INFINITY );
    }//end else

}//end method determinant




/**
 *
 * @return the determinant of this matrix.
 */
public BigDecimal determinant(){
    BigDecimal determinant=det(this);
    setDet(BigDecimal.ZERO);//reset the determinant attribute.
     return determinant; 
}



/**
 *
 * @param matrixValue A string that is to be checked if
 * it conforms to valid syntax for representing a matrix in this software.
 * @return true if the command string is a valid matrix.e.g [2,1,4:5,3,-2:4,4,5]
 * value.
 */
public static boolean isMatrixValue(String matrixValue){
MatrixValueParser matrixValueParser = new MatrixValueParser(matrixValue);

boolean isValid = matrixValueParser.isValid();

return isValid;
}













/**
 *
 * @param row The row in this PrecisionMatrix object to be converted into a
 * new PrecisionMatrix object. This operation generates a new PrecisionMatrix
 * object which is a row matrix.
 */
public PrecisionMatrix getRowMatrix(int row){
    BigDecimal[][]arr = new BigDecimal[1][getCols()];
    for(int col = 0; col < getCols(); col++){
    arr[0][col] = array[row][col];
     }//end for
return new PrecisionMatrix(arr);
}

/**
 *
 * @param column The column to be converted into a
 * new PrecisionMatrix object. This operation generates a new PrecisionMatrix
 * object which is a column matrix.
 */
public PrecisionMatrix getColumnMatrix(int column){
    BigDecimal[][]arr = new BigDecimal[getRows()][1];
    for(int row = 0; row < getRows(); row++){
    arr[row][0] = array[row][column];
//System.out.println("Seee!!! = "+array[row][0]);
     }//end for
return new PrecisionMatrix(arr);
}



/**
 *
 * @return the inverse of the PrecisionMatrix
 * as another PrecisionMatrix object.
 */
public PrecisionMatrix inverse(     ){


PrecisionMatrix m = new PrecisionMatrix(this);
PrecisionMatrix unit = m.unitMatrix();
PrecisionMatrix inverse = new PrecisionMatrix(new BigDecimal[m.getRows()][m.getCols()]);
    if(m.isSquareMatrix()){
 for( int rows = 0; rows < m.getCols(); rows++){
PrecisionMatrix c = PrecisionMatrix.columnJoin( m , unit.getColumnMatrix(rows));
inverse = PrecisionMatrix.columnJoin(inverse, c.solveEquation());
 }//end for
}//end if
inverse.columnDeleteFromStart(m.getRows());
return inverse;
}









    /**
     *
     * @return a string representation of the matrix in rows and columns.
     */
    @Override
    public String toString(){
        String output ="\n";
        String appender="";
       for(int row=0;row < getRows();row++){

      for (int column=0;column<getCols();column++){

       if(column < getCols()){
       appender+= String.format("%7s%3s", array[row][column], ",");
       }
       if(column == getCols()-1){
           appender=appender.substring(0,appender.length()-1);
       appender+="\n";
       }
              output+=appender;
                     appender="";
      }
}

      return output;
    }//end method toString







public static void main(String arg[]){
/*
MatrixValueParser mat=new MatrixValueParser("[1,0,0:2,1,0:0,0,1:]");

PrecisionMatrix a = mat.getMatrix();

mat.setValues("[0,1,1,0:0,0,1,1:1,1,1,1:]");

PrecisionMatrix b = mat.getMatrix();
 util.Utils.logError(a+"\n"+b);

//
a.multiply(b);
 util.Utils.logError(a);
*/
PrecisionMatrix m = new PrecisionMatrix(20,21);

m.randomFill(220);

System.out.print(  m  );

System.out.print(  m.solveEquation()  );
//System.out.print( PrecisionMatrix.multiply( m , m.inverse() ) );

}

}//end class