/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package math.matrix.expressParser;

import parser.STRING;
import parser.Variable;
 


/**
 *
 * @author JIBOYE Oluwagbemiro Olaoluwa
 */
public class MatrixVariable{
/**
 * The name of the Matrix Variable.
 * A valid Matrix Variable Name must start with #
 * and the rest of its body must be a valid variable name.
 */
    private String name;

/**
 * The matrix object stored by this variable.
 */
private Matrix matrix;

/**
 * 
 * @param name The name of the Matrix object.
 * @param matrix The Matrix object used to set the Matrix value stored
 * by this MatrixVariable object.
 */
    public MatrixVariable(String name, Matrix matrix) {
        if(isMatrixVariableName(name)){
          this.name = name;
          this.matrix = matrix;
        }
        else{
            util.Utils.logError( "Error In Matrix Name Format.");
        }
    }
/**
 * Here an expression like A=[2,3,4:-2,3,4:5,1,20] is passed into
 * the constructor.
 * @param name The name of the matrix.
 * @param expression The compound expression containing
 * information useful in creating the MatrixVariable object..e.g [2,3,4:-2,3,4:5,1,20]
 */
    public MatrixVariable(String name,String expression) {
MatrixValueParser parser = new MatrixValueParser(expression);
        this.name = name;
        this.matrix = parser.getMatrix();
    }





    public void setMatrixVariable(String expression){
MatrixValueParser parser = new MatrixValueParser(expression);
        this.matrix = parser.getMatrix();
    }

    public void setMatrix(Matrix matrix) {
        this.matrix = matrix;
    }

    public Matrix getMatrix() {
        return matrix;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

/**
 *
 * @return the equivalent string expression that
 * creates the matrix variable and stores its present matrix value in it.
 */
    public String getCreatingExpression(){


double matArray[][]=matrix.getArray();
String stringvalue=name+" = [";

for(int rows=0;rows<matrix.getRows();rows++){
    for (int cols = 0; cols < matrix.getCols(); cols++) {
        if(cols<matrix.getCols()-1){
      stringvalue+=matArray[rows][cols]+",";
        }//end if
        else{
         stringvalue+=matArray[rows][cols]+":";
        }//end else

    }//end inner for


}//end outer for


return stringvalue+"]";
    }//end method

/**
 * 
 * A + B + 2*C + A.B - A.B^5 + 3*4*A^2*C+I
 * Matrix names must start with # in
 * this syntax.
 * @param name The string to be checked if or not it
 * represents a valid matrix name.
 * @return true if the name represents a valid
 * matrix name.
 */
public static boolean isMatrixVariableName(String name){
    String n=STRING.purifier(name);
try{
    if(n.substring(0,1).equals("#")&&n.substring(1).length()>0){
     return n.substring(0,1).equals("#")&&Variable.isVariableString(n.substring(1));
    }
    else{
        return false;
    }//end else.
}
catch(IndexOutOfBoundsException indexErr){
    return false;
}
}

/**
 *
 * @return the view of the matrix in the format...A =  num1,   num2,  num3 ....numi
 *                                                     numi+1,numi+2,numi+3....numj
 *                                                     numj+1,numj+2,numj+3....numk
 *                                                          e.t.c.
 */
    @Override
public String toString(){
    return name+" = "+matrix;
}

public static void main(String args[]){
MatrixVariable var = new MatrixVariable("#A","[2,3,7,9,3,4:89,2,-28,12,4,5:,1,1,0,1,0,1:]");
    System.out.println(var);
    MatrixVariable reduced = new MatrixVariable("#B", var.getMatrix().reduceToTriangularMatrix());
     System.out.println(reduced);


     System.out.println(var.getCreatingExpression()+"\n"+reduced.getCreatingExpression());

}



}//end class MatrixVariable
