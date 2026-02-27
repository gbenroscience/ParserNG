/*
 * Copyright 2026 GBEMIRO.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.gbenroscience.math.matrix.expressParser;

import java.util.HashMap;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Random;
import com.github.gbenroscience.parser.Scanner;
import com.github.gbenroscience.parser.Operator;

/**
 *
 * @author GBEMIRO
 */
public class Matrix {

    public static final String lambda = "n";

    /**
     * The simple name used to label this Matrix object.
     *
     */
    private String name;
    /**
     * the data array used to create this Matrix object
     */
    private double[] array;
    /**
     * Number of rows in the matrix.
     */
    private int rows;
    /**
     * Number of columns in the matrix.
     */
    private int cols;

    /**
     *
     * @param rows The number of rows in the Matrix.
     * @param cols The number of columns in the Matrix.
     */
    public Matrix(int rows, int cols) {
        this("NEW", rows, cols);
    }//end constructor

    /**
     * @param name the simple name used to identify used by the user to label
     * this Matrix object.
     *
     * Assigns name unknown to the Matrix object and a 2D array that has just a
     * row and a column
     *
     */
    public Matrix(String name) {
        this(name, 1, 1);
        array[0] = 0;
    }

    private Matrix(String name, int rows, int cols) {
        this.name = name;
        this.rows = rows;
        this.cols = cols;
        this.array = new double[rows * cols];
    }

    /**
     *
     * @param array the 2D data supplied as a 1D array used to create this
     * @param rows
     * @param columns Matrix object
     */
    public Matrix(double[] flatArray, int rows, int columns) {
        this("NEW");
        if (rows * columns == flatArray.length) {
            this.array = flatArray;
            this.rows = rows;
            this.cols = columns;
            return;
        }
        throw new IllegalArgumentException("the flat array representation should have rows*columns elements...flatArray size=" + flatArray.length + ", rowsXcolumns=" + rows * columns);
    }//end constructor

    /**
     *
     * @param array the data array used to create this Matrix object
     */
    public Matrix(double[][] array) {
        this("NEW");
        setArray(array);
    }//end constructor

    /**
     *
     * @param matrix constructs a new Matrix object having similar properties to
     * the one passed as argument, but holding no reference to its data.
     */
    public Matrix(Matrix matrix) {
        this("NEW", matrix.rows, matrix.cols);
        System.arraycopy(matrix.array, 0, this.array, 0, this.array.length);
    }//end constructor

    /**
     *
     * @return the number of rows in this matrix object
     */
    public int getRows() {
        return rows;
    }

    /**
     *
     * @return the number of columns in this matrix object
     */
    public int getCols() {
        return cols;
    }

    /**
     *
     * @param array sets the data of this matrix
     */
    public final void setArray(double[][] array) {
        if (array.length > 0 && array[0].length > 0) {
            this.rows = array.length;
            this.cols = array[0].length;
            this.array = new double[rows * cols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    this.array[row * cols + col] = array[row][col];
                }//end inner loop
            }//end outer loop
        } else {
            this.rows = 2;
            this.cols = 1;
            this.array = new double[rows * cols];
        }
    }//end method

    public final void setArray(double[] flatArray, int rows, int columns) {
         if (rows * columns == flatArray.length) {
            this.array = flatArray;
            this.rows = rows;
            this.cols = columns;
            return;
        }
        throw new IllegalArgumentException("the flat array representation should have rows*columns elements...flatArray size=" + flatArray.length + ", rowsXcolumns=" + rows * columns);
    }//end method

    public double[] getFlatArray() {
        return array;
    }

    /**
     *
     * @return the data of this matrix
     */
    public double[][] getArray() {
        double[][] arr = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                arr[row][col] = array[row * cols + col];
            }
        }
        return arr;
    }

    public double getElem(int row, int col) {
        return array[row * cols + col];
    }

    /**
     *
     * @param name set's the simple name used to identify used by the user to
     * label this Matrix object.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     *
     * @return the simple name used to identify used by the user to label this
     * Matrix object.
     */
    public String getName() {
        return name;
    }

    /**
     *
     * @param row1 The first row.
     * @param row2 The second row.
     *
     * <b color='red'> BE CAREFUL!!!!<br>
     * THIS METHOD ACTS ON THE MATRIX OBJECT ON WHICH IT IS CALLED AND MODIFIES
     * IT!
     * </b>
     *
     */
    public void swapRow(int row1, int row2) {
        for (int column = 0; column < cols; column++) {
            double v1 = array[row1 * cols + column];
            double v2 = array[row2 * cols + column];
            double v3 = v1;
            array[row1 * cols + column] = v2;
            array[row2 * cols + column] = v3;
        }//end for
    }//end method.

    /**
     *
     * @param col1 The first row.
     * @param col2 The second row.
     *
     * <b color='red'> BE CAREFUL!!!!<br>
     * THIS METHOD ACTS ON THE MATRIX OBJECT ON WHICH IT IS CALLED AND MODIFIES
     * IT!
     * </b>
     *
     */
    public void swapColumn(int col1, int col2) {
        for (int row = 0; row < rows; row++) {
            double v1 = array[row * cols + col1];
            double v2 = array[row * cols + col2];
            double v3 = v1;
            array[row * cols + col1] = v2;
            array[row * cols + col2] = v3;
        }//end for
    }//end method.

    /**
     * Fills this matrix object with values
     */
    public void fill() {
        java.util.Scanner scanner = new java.util.Scanner(System.in);

        for (int row = 0; row < rows; row++) {

            for (int column = 0; column < cols; column++) {
                array[row * cols + column] = scanner.nextDouble();
            }
        }
    }//end method fill.

    /**
     *
     * @param matrice the matrix to be added to this one. The operation is (
     * this + matrice )
     * @return a Matrix containing the product matrix.
     */
    public Matrix add(Matrix matrice) {

        if (rows != matrice.rows || cols != matrice.cols) {
            System.out.println("ERROR IN MATRIX INPUT!!");
            return new Matrix(1, 1);
        }

        Matrix matrix = new Matrix(rows, cols);

        for (int i = 0; i < array.length; i++) {
            matrix.array[i] = array[i] + matrice.array[i];
        }

        return matrix;
    }//end method add

    /**
     *
     * @param matrice the matrix to be subtracted from this one. The operation
     * is ( this - matrice )
     * @return a Matrix containing the product matrix.
     */
    public Matrix subtract(Matrix matrice) {

        if (rows != matrice.rows || cols != matrice.cols) {
            System.out.println("ERROR IN MATRIX INPUT!!");
            return new Matrix(1, 1);
        }

        Matrix matrix = new Matrix(rows, cols);

        for (int i = 0; i < array.length; i++) {
            matrix.array[i] = array[i] - matrice.array[i];
        }

        return matrix;
    }//end method subtract

    /**
     *
     * @param scalar the constant to be multiplied with this matrix The
     * operation is ( scalar X matrice )
     * @return an array containing the product matrix.
     */
    public Matrix scalarMultiply(double scalar) {
        Matrix matrix = new Matrix(rows, cols);
        for (int i = 0; i < array.length; i++) {
            matrix.array[i] = scalar * array[i];
        }
        return matrix;
    }//end method scalarMultiply

    /**
     *
     * @param scalar the constant to be multiplied with this matrix The
     * operation is ( matrice/scalar )
     * @return an array containing the scaled matrix.
     */
    public Matrix scalarDivide(double scalar) {
        Matrix matrix = new Matrix(rows, cols);
        for (int i = 0; i < array.length; i++) {
            matrix.array[i] = array[i] / scalar;
        }
        return matrix;
    }//end method scalarDivide

    /**
     *
     * The operation of matrix multiplication. For this method to run, The
     * pre-multiplying matrix must have its number of columns equal to the
     * number of rows in the pre-multiplying one.
     *
     * The product matrix is one that has its number of columns equal to the
     * number of columns in the pre-multiplying matrix, and its rows equal to
     * that in the post-multiplying matrix.
     *
     *
     * So if the operation is A X B = C, and A is an m X n matrix while B is an
     * r X p matrix, then r = n is a necessary condition for the operation to
     * occur. Also, C is an m X p matrix.
     *
     * @param matrice1 the matrix to be pre-multiplying the other one. The
     * operation is ( matrice1 X matrice2 )
     * @param matrice2 the post-multiplying matrix
     * @return a new Matrix object containing the product matrix of the
     * operation matrice1 X matrice2
     */
    public static Matrix multiply(Matrix matrice1, Matrix matrice2) {

        int r1 = matrice1.rows;
        int c1 = matrice1.cols;
        int r2 = matrice2.rows;
        int c2 = matrice2.cols;

        if (c1 != r2) {
            System.out.println("ERROR IN MATRIX INPUT!!");
            return new Matrix(1, 1);
        }

        Matrix m = new Matrix(r1, c2);

        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                double sum = 0;
                for (int k = 0; k < c1; k++) {
                    sum += matrice1.array[i * c1 + k] * matrice2.array[k * c2 + j];
                }//end inner for
                m.array[i * c2 + j] = sum;
            }//end outer for
        }//end outermost for

        return m;
    }

    /**
     *
     * @param matrice the matrix to be multiplied by this one. This operation
     * modifies this matrix and changes its data array into that of the product
     * matrix The operation is ( this X matrice )
     */
    public void multiply(Matrix matrice) {
        Matrix product = Matrix.multiply(this, matrice);
        this.rows = product.rows;
        this.cols = product.cols;
        this.array = product.array;
    }

    /**
     *
     * @param mat the matrix to raise to a given power
     * @param pow the power to raise this matrix to
     * @return the
     */
    public static Matrix pow(Matrix mat, int pow) {
        Matrix m = new Matrix(mat);
        for (int i = 0; i < pow - 1; i++) {
            m = Matrix.multiply(m, mat);

        }
        return m;
    }

    public static Matrix power(Matrix mat, int pow) {
        assert (pow >= 0);
        switch (pow) {
            case 0:
                return unitMatrix(mat.rows, mat.cols);
            case 1:
                return mat;
            default:
                /**
                 * n=3:
                 * mul(power(mat,2),mat)---mul(mul(power(mat,1),mat),mat)--mul(
                 * mul(mat,mat),mat )---mul(mat2,mat)-->mat3
                 */
                return multiply(power(mat, pow - 1), mat);

        }

    }

    /**
     *
     * @return a unit matrix of the same dimension as this matrix object
     */
    public Matrix unitMatrix() {
        return unitMatrix(rows, cols);
    }//end method unitMatrix

    /**
     *
     * @param rowSize the number of rows that the unit matrix will have
     * @param colSize the number of columns that the unit matrix will have
     * @return a unit matrix having number of rows = rowSize and number of
     * columns=colSize.
     */
    public static Matrix unitMatrix(int rowSize, int colSize) {
        Matrix matrix = new Matrix(rowSize, colSize);

        for (int row = 0; row < rowSize; row++) {
            for (int column = 0; column < colSize; column++) {
                if (column == row) {
                    matrix.array[row * colSize + column] = 1;
                } else {
                    matrix.array[row * colSize + column] = 0;
                }

            }//end inner for loop
        }//end outer for loop
        return matrix;
    }//end method unitMatrix

    /**
     *
     * @param mat the Matrix object that we wish to construct a unit matrix of
     * similar dimensions for.
     * @return a unit matrix of equal dimensions as this unit matrix.
     */
    public static Matrix unitMatrix(Matrix mat) {
        return unitMatrix(mat.rows, mat.cols);
    }//end method unitMatrix

    /**
     * Place the first Matrix object side by side with the second one passed as
     * argument to this method. The result is a new matrix where:
     *
     * if 3 4 5 7 5 9 mat1 = 2 3 1 and mat2 = 4 2 6 1 6 7 5 7 3 in a new matrix
     * object (mat).
     *
     *
     * e.g 3 4 5 7 5 9 2 3 1 4 2 6 1 6 7 5 7 3 A necessary condition for this
     * method to run is that the 2 objects must have an equal number of rows. IF
     * THIS CONDITION IS NOT MET, THE METHOD RETURNS A ZERO MATRIX.
     *
     *
     * @param mat1 the first Matrix object
     * @param mat2 the second Matrix object that we column join with this one
     * @return a new Matrix object that contains this Matrix object placed side
     * by side with the Matrix object passed as argument.
     */
    public static Matrix columnJoin(Matrix mat1, Matrix mat2) {

        if (mat1.rows != mat2.rows) {
            return new Matrix(1, 1);
        }

        int r = mat1.rows;
        int c1 = mat1.cols;
        int c2 = mat2.cols;

        Matrix join = new Matrix(r, c1 + c2);

        for (int row = 0; row < r; row++) {
            for (int col = 0; col < c1; col++) {
                join.array[row * (c1 + c2) + col] = mat1.array[row * c1 + col];
            }
            for (int col = 0; col < c2; col++) {
                join.array[row * (c1 + c2) + (c1 + col)] = mat2.array[row * c2 + col];
            }
        }//end outer for

        return join;
    }

    /**
     *
     * @param value The value to insert
     * @param row The row where the value is to be inserted.
     * @param column The column where the value is to be inserted.
     */
    public boolean update(double value, int row, int column) {
        if (row < rows && column < cols) {
            array[row * cols + column] = value;
            return true;
        }
        return false;
    }

    /**
     * Place the first Matrix object side by side with the second one passed as
     * argument to this method. The result is a new matrix where:
     *
     * if 3 4 5 7 5 9 mat1 = 2 3 1 and mat2 = 4 2 6 1 6 7 5 7 3 in a new matrix
     * object (mat).
     *
     *
     * e.g 3 4 5 2 3 1 1 6 7 7 5 9 4 2 6 5 7 3
     *
     * A necessary condition for this method to run is that the 2 objects must
     * have an equal number of columns. IF THIS CONDITION IS NOT MET, THE METHOD
     * RETURNS A ZERO MATRIX.
     *
     * @param mat1 the first Matrix object
     * @param mat2 the second Matrix object that we row join with this one
     * @return a new Matrix object that contains the first Matrix object
     * argument placed top to bottom with the second Matrix object argument.
     */
    public static Matrix rowJoin(Matrix mat1, Matrix mat2) {

        if (mat1.cols != mat2.cols) {
            return new Matrix(1, 1);
        }

        int r1 = mat1.rows;
        int r2 = mat2.rows;
        int c = mat1.cols;

        Matrix join = new Matrix(r1 + r2, c);

        for (int row = 0; row < r1; row++) {
            for (int col = 0; col < c; col++) {
                join.array[row * c + col] = mat1.array[row * c + col];
            }
        }

        for (int row = 0; row < r2; row++) {
            for (int col = 0; col < c; col++) {
                join.array[(r1 + row) * c + col] = mat2.array[row * c + col];
            }
        }

        return join;
    }//end method rowJoin

    /**
     * Deletes all the specified number of columns from the Matrix object
     * starting from the end of the Matrix object
     *
     * @param column the number of columns to remove from the Matrix object.
     * This method will take the object that calls it and perform this operation
     * on it. So it modifies the Matrix object that calls it. Be careful, as
     * data will be lost.
     *
     * e.g if 3 4 5 6 7 8 9 2 1 8 1 4 7 0 A = 3 3 2 1 5 7 1
     *
     * then the call:
     *
     * A.columnDeleteFromEnd(3)
     *
     * will delete the last three columns in this object leaving:
     *
     * 3 4 5 6
     * A = 2 1 8 1
     * 3 3 2 1
     *
     *
     */
    public void columnDeleteFromEnd(int column) {
        if (column >= 0 && column <= cols) {
            int newCols = cols - column;
            double[] newArray = new double[rows * newCols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < newCols; col++) {
                    newArray[row * newCols + col] = array[row * cols + col];
                }//end inner for
            }//end outer for
            this.array = newArray;
            this.cols = newCols;
        } else {
            System.out.println("COLUMN VALUE SHOULD "
                    + "RANGE FROM ZERO TO THE NUMBER OF COLUMNS IN THIS MATRIX.");
        }
    }//end method columnDeleteFromEnd

    /**
     * Deletes all the specified number of columns from the Matrix object from
     * the beginning of the Matrix object
     *
     * @param column the number of columns to remove from the Matrix object's
     * beginning. This method will take the object that calls it and perform
     * this operation on it. So it modifies the Matrix object that calls it. Be
     * careful, as data will be lost.
     *
     * e.g if 3 4 5 6 7 8 9 2 1 8 1 4 7 0 A = 3 3 2 1 5 7 1
     *
     * then the call:
     *
     * A.columnDeleteFromStart(3)
     *
     * will delete the last three columns in this object leaving:
     *
     * 6 7 8 9
     * A= 1 4 7 0 1 5 7 1
     *
     *
     */
    public void columnDeleteFromStart(int column) {
        if (column >= 0 && column <= cols) {
            int newCols = cols - column;
            double[] newArray = new double[rows * newCols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < newCols; col++) {
                    newArray[row * newCols + col] = array[row * cols + (column + col)];
                }//end inner for
            }//end outer for
            this.array = newArray;
            this.cols = newCols;
        } else {
            System.out.println("COLUMN VALUE SHOULD "
                    + "RANGE FROM ZERO TO THE NUMBER OF COLUMNS IN THIS MATRIX.");
        }
    }//end method columnDeleteFromStart

    /**
     * Deletes the specified number of rows from the end of the Matrix object
     *
     * @param numOfRows the number of rows to remove from the Matrix object's
     * beginning. This method will take the object that calls it and perform
     * this operation on it. So it modifies the Matrix object that calls it. Be
     * careful, as data will be lost.
     *
     * e.g if 3 4 5 6 2 1 8 1 A = 3 3 2 1 7 8 9 2 4 7 0 5 5 7 1 8 then the call:
     *
     * A.rowDeleteFromEnd(3)
     *
     * will delete the last three rows in this object leaving:
     *
     *
     * 3 4 5 6
     * 2 1 8 1
     * A = 3 3 2 1
     *
     *
     */
    public void rowDeleteFromEnd(int numOfRows) {
        if (numOfRows >= 0 && numOfRows <= rows) {
            int newRows = rows - numOfRows;
            double[] newArray = new double[newRows * cols];
            for (int row = 0; row < newRows; row++) {
                for (int col = 0; col < cols; col++) {
                    newArray[row * cols + col] = array[row * cols + col];
                }//end inner for

            }//end outer for

            this.array = newArray;
            this.rows = newRows;
        } else {
            System.out.println("NUMBER OF ROWS TO BE DELETED SHOULD "
                    + "RANGE FROM ZERO TO (AND INCLUDING) THE NUMBER OF ROWS IN THIS MATRIX.");
        }
    }//end method rowDeleteFromEnd

    /**
     * Deletes the specified number of rows from the beginning of the Matrix
     * object
     *
     * @param numOfRows the number of rows to remove from the Matrix object's
     * beginning. This method will take the object that calls it and perform
     * this operation on it. So it modifies the Matrix object that calls it. Be
     * careful, as data will be lost.
     *
     * e.g if 3 4 5 6 2 1 8 1 A = 3 3 2 1 7 8 9 2 4 7 0 5 5 7 1 8 then the call:
     *
     * A.rowDeleteFromStart(3)
     *
     * will delete the last three rows in this object leaving:
     *
     *
     * A = 7 8 9 2
     * 4 7 0 5
     * 5 7 1 8
     *
     *
     */
    public void rowDeleteFromStart(int numOfRows) {
        if (numOfRows >= 0 && numOfRows <= rows) {
            int newRows = rows - numOfRows;
            double[] newArray = new double[newRows * cols];
            for (int row = 0; row < newRows; row++) {
                for (int col = 0; col < cols; col++) {
                    newArray[row * cols + col] = array[(numOfRows + row) * cols + col];
                }//end inner for

            }//end outer for

            this.array = newArray;
            this.rows = newRows;
        } else {
            System.out.println("NUMBER OF ROWS TO BE DELETED SHOULD "
                    + "RANGE FROM ZERO TO (AND INCLUDING) THE NUMBER OF ROWS IN THIS MATRIX.");
        }
    }//end method rowDeleteFromStart

    /**
     *
     * @return an upper triangular matrix obtained by row reduction.
     */
    public Matrix reduceToTriangularMatrix() {

        Matrix mat = new Matrix(this);
        //Now we row-reduce.

        for (int row = 0; row < mat.rows; row++) {
            if (row >= this.cols) {
                break;
            }
            double val = mat.array[row * mat.cols + row];

            /**
             * The division coefficient must not be zero. If zero, search for a
             * lower row, and swap.
             */
            if (val == 0.0) {

                for (int rw = row; rw < mat.rows; rw++) {
                    val = mat.array[rw * mat.cols + row];

                    if (val != 0.0) {
                        mat.swapRow(row, rw);
                        break;
                    }//end if

                }//end for loop

                if (val == 0.0) {
                    throw new InputMismatchException("EQUATION CANNOT BE SOLVED");
                }
            }//end if

            for (int col = row; col < mat.cols; col++) {
                mat.array[row * mat.cols + col] /= val;
            }//end inner for loop

            for (int rowed = row + 1; rowed < mat.rows; rowed++) {
                double mul = mat.array[rowed * mat.cols + row];
                for (int coled = row; coled < mat.cols; coled++) {
                    mat.array[rowed * mat.cols + coled] = mat.array[rowed * mat.cols + coled] - mul * mat.array[row * mat.cols + coled];
                }//end for

            }//end for

        }//end outer for loop

        return mat;
    }//end method

    /**
     *
     * @return an upper triangular matrix obtained by row reduction.
     */
    public Matrix reduceToRowEchelonMatrix() {

        Matrix mat = new Matrix(this);
        //Now we row-reduce.

        for (int row = 0; row < mat.rows; row++) {
            if (row >= this.cols) {
                break;
            }
            double val = mat.array[row * mat.cols + row];

            /**
             * The division coefficient must not be zero. If zero, search for a
             * lower row, and swap.
             */
            if (val == 0.0) {

                for (int rw = row; rw < mat.rows; rw++) {
                    val = mat.array[rw * mat.cols + row];

                    if (val != 0.0) {
                        mat.swapRow(row, rw);
                        break;
                    }//end if

                }//end for loop

                if (val == 0.0) {
                    throw new InputMismatchException("EQUATION CANNOT BE SOLVED");
                }
            }//end if

            for (int rowed = row + 1; rowed < mat.rows; rowed++) {
                double mul = mat.array[rowed * mat.cols + row];
                for (int coled = row; coled < mat.cols; coled++) {
                    mat.array[rowed * mat.cols + coled] = val * mat.array[rowed * mat.cols + coled] - mul * mat.array[row * mat.cols + coled];
                }//end for

            }//end for

        }//end outer for loop

        return mat;
    }//end method

    /**
     * Used to solve a system of simultaneous equations placed in this Matrix
     * object.
     *
     * @return a Matrix object containing the solution matrix.
     */
    public Matrix solveEquation() {
        return Matrix.solveEquation(this);
    }//end method solnMatrix

    /**
     * Used to solve a system of simultaneous equations placed in the Matrix
     * object.
     *
     * @param matrix The row X row+1 matrix, containing the system of linear
     * equations
     * @return a Matrix object containing the solution matrix.
     */
    public static Matrix solveEquation(Matrix matrix) {
        Matrix solnMatrix = new Matrix(matrix.rows, 1);

        Matrix matrixLoader = matrix.reduceToTriangularMatrix();
        // System.out.println( matrixLoader  );

        if (matrix.rows == matrix.cols - 1) {
            //Back-Substitution Algorithm.
            double sum = 0;//summation variable
            int counter = 1;
            for (int row = matrixLoader.rows - 1; row >= 0; row--) {
                for (int col = row + 1; col < matrixLoader.cols; col++) {

                    if (col < matrixLoader.cols - 1) {
                        sum += (matrixLoader.array[row * matrixLoader.cols + col] * solnMatrix.array[col * solnMatrix.cols + 0]);//sum the product of each coefficient and its unknown's value in the solution matrix
                    } else if (col == matrixLoader.cols - 1) {
                        sum = matrixLoader.array[row * matrixLoader.cols + col] - sum;//end of summing.Now subtract the sum from the last entry on the row
                    }
                }//end inner loop
                solnMatrix.array[(matrixLoader.rows - counter) * solnMatrix.cols + 0] = sum / matrixLoader.array[row * matrixLoader.cols + row];
                counter++;// increment counter
                sum = 0;//reset sum
            }//end outer loop

        }//end if
        else {
            throw new IndexOutOfBoundsException("Invalid System Of Linear Equations");
        }//end else

        return solnMatrix;
    }//end method solnMatrix

    /**
     *
     * @return the transpose of this Matrix object. Does not modify this matrix
     * object but generates the transpose of this Matrix object as another
     * Matrix object.
     */
    public Matrix transpose() {
        Matrix matrix = new Matrix(cols, rows);

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                matrix.array[col * rows + row] = array[row * cols + col];
            }//end inner for

        }//end outer for

        return matrix;
    }//end method transpose

    public Matrix adjoint() {
        if (isSquareMatrix()) {
            Matrix matrix = new Matrix(rows, cols);

            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    matrix.array[row * cols + col] = getCofactor(row, col).determ();
                }//end inner for
            }//end outer for

            return matrix.transpose();
        }
        return null;
    }

    /**
     *
     * @return a matrix that contains the cofactors of the elements of this
     * Matrix.
     */
    public Matrix getCofactorMatrix() {

        if (isSquareMatrix()) {

            Matrix matrix = new Matrix(rows, cols);

            int count = 0;
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++, count++) {
                    matrix.array[row * cols + col] = (count % 2 == 0 ? 1 : -1) * getCofactor(row, col).determ();
                }//end inner for
            }//end outer for    

            return matrix;
        }

        return null;
    }

    /**
     *
     * @param i the row on which the element whose minor is needed lies.
     * @param j the column on which the element whose minor is needed lies.
     * @return the minor of this matrix relative to this element.
     */
    public Matrix minor(int i, int j) {

        Matrix matrix = new Matrix(rows - 1, cols - 1);

        int subRow = 0;
        for (int row = 0; row < rows; row++) {
            if (row == i) {
                continue;
            }
            int subCol = 0;
            for (int col = 0; col < cols; col++) {
                if (col == j) {
                    continue;
                }
                matrix.array[subRow * matrix.cols + subCol] = array[row * this.cols + col];
                subCol++;
            }//end inner for
            subRow++;
        }//end outer for

        return matrix;
    }//end method minor

    /**
     *
     * @return true if this Matrix object is a square matrix.
     */
    public boolean isSquareMatrix() {
        return rows == cols;
    }

    /**
     *
     * @return true if this Matrix object can represent a system of equations
     * solvable by reduction to triangular form and subsequent evaluation.
     */
    public boolean isSystemOfEquations() {
        return rows == cols - 1;
    }

    /**
     *
     * @param m a 2 X 2 matrix
     * @return the determinant of this matrix
     */
    private static double $2X2determinant(Matrix m) {
        return m.array[0 * m.cols + 0] * m.array[1 * m.cols + 1] - m.array[1 * m.cols + 0] * m.array[0 * m.cols + 1];
    }

    /**
     *
     * @param m a 2 X 2 matrix
     * @return the determinant of this matrix
     */
    private static String $2X2determinantForEigen(String[][] m) {
        /*
        {
        {2-n  ,   4}
        {3    , 5-n}
        
        
        }
        m[0][0].m[1][1]-m[1][0].m[0][1]
         */
        //looks like (2-n)(3-n)-(4)(2)

        String v1 = "", v2 = "";
        if (com.github.gbenroscience.parser.Number.validNumber(m[1][0]) && com.github.gbenroscience.parser.Number.validNumber(m[0][1])) {
            double val = Double.parseDouble(m[1][0]) * Double.parseDouble(m[0][1]);
            v1 = String.valueOf(val);
        }

        if (com.github.gbenroscience.parser.Number.validNumber(m[0][0]) && com.github.gbenroscience.parser.Number.validNumber(m[1][1])) {
            double val = Double.parseDouble(m[0][0]) * Double.parseDouble(m[1][1]);
            v2 = String.valueOf(val);
        }

        if (!v1.isEmpty() && !v2.isEmpty()) {
            return String.valueOf(Double.parseDouble(v2) - Double.parseDouble(v1));
        }

        String expr1 = v1.isEmpty() ? uniVariableExpressionExpander(lambda, m[1][0], m[0][1]) : v1;
        String expr2 = v2.isEmpty() ? uniVariableExpressionExpander(lambda, m[0][0], m[1][1]) : v2;

        String negExpr1 = uniVariableExpressionExpander(lambda, "-1", expr1);

        String expanded = uniVariableExpressionExpander(lambda, "1", expr2 + "+" + negExpr1);

        return expanded;

    }

    /**
     * @param m the matrix whose top row is to be multiplied by a scalar
     * Multiplies the top row of a matrix by a scalar. This is an important
     * operation during the evaluation of a determinant.
     */
    private static Matrix topRowScalarMultiply(Matrix m, double scalar) {
        for (int col = 0; col < m.cols; col++) {
            m.array[0 * m.cols + col] *= scalar;
        }
        return new Matrix(m);
    }

    /**
     * Sarus' rule for computing determinants. This technique is too slow and
     * memory intensive for large matrices..n>=10. Please use the determ()
     * instance method. It uses a O(cube_n) algorithm as against this method's
     * O(n!)
     *
     * @param m the Matrix object whose determinant is desired.
     * @return the determinant of the matrix
     */
    private static final double det(Matrix m, double acc) {
        //must be square matrix
        if (m.rows == m.cols) {
            if (m.rows == 2) {
                return $2X2determinant(m);
            }//end else
            else {
                for (int col = 0; col < m.cols; col++) {
                    double topRow = m.array[0 * m.cols + col] * Math.pow(-1, col);
                    Matrix mat = topRowScalarMultiply(m.minor(0, col), topRow);

                    if (mat.rows > 2) {
                        acc = det(mat, acc);
                    } else {
                        acc += $2X2determinant(mat);
                    }
                }//end for
                return acc;
            }//end else
        }//end if
        else {
            return Double.POSITIVE_INFINITY;
        }//end else
    }//end method determinant

    /**
     * Sarus' rule for computing determinants. This technique is too slow and
     * memory intensive for large matrices..n>=10. Please use the determ()
     * instance method. It uses a O(cube_n) algorithm as against this method's
     * O(n!)
     *
     * @param m the Matrix object whose determinant is desired.
     * @return the determinant of the matrix
     */
    private double sarusDet() {
        double acc = 1;
        return det(this, acc);
    }//end method

    /**
     *
     * @return the determinant of this matrix using a row reduction technique.
     */
    public double determinant() {
        return this.determ();
    }

    /**
     * Fills the matrix with randomly generated values between 1 and 101.
     */
    public void randomFill() {
        Random ran = new Random();
        for (int i = 0; i < array.length; i++) {
            array[i] = 1 + (double) ran.nextInt(101);
        }
    }//end method randomFill

    /**
     * Fills the matrix with randomly generated values between 1 and n
     *
     * @param n the maximum possible size of the integer numbers generated.
     */
    public void randomFill(int n) {
        Random ran = new Random();
        for (int i = 0; i < array.length; i++) {
            array[i] = 1 + ran.nextInt(n);
        }
    }//end method randomFill

    /**
     *
     * @param matrixValue A string that is to be checked if it conforms to valid
     * syntax for representing a matrix in this software.
     * @return true if the command string is a valid matrix.e.g
     * [2,1,4:5,3,-2:4,4,5] value.
     */
    public static boolean isMatrixValue(String matrixValue) {
        MatrixValueParser matrixValueParser = new MatrixValueParser(matrixValue);

        boolean isValid = matrixValueParser.isValid();

        return isValid;
    }

    /**
     *
     * @return a string representation of the matrix in rows and columns.
     */
    @Override
    public String toString() {
        String output = "\n";
        String appender = "";
        for (int row = 0; row < rows; row++) {

            for (int column = 0; column < cols; column++) {

                if (column < cols) {
                    appender += String.format("%7s%3s", array[row * cols + column], ",");
                }
                if (column == cols - 1) {
                    appender = appender.substring(0, appender.length() - 1);
                    appender += "          \n";
                }
                output += appender;
                appender = "";
            }
        }

        return output;
    }//end method toString

    /**
     * @param mat The string matrix
     */
    public static void printTextMatrix(String[][] mat) {
        String output = "\n";
        String appender = "";

        int rows = mat.length;
        if (mat.length == 0) {
            System.out.println("EMPTY");
        }
        int cols = mat[0].length;
        for (int row = 0; row < rows; row++) {

            for (int column = 0; column < cols; column++) {

                if (column < cols) {
                    appender += String.format("%7s%3s", mat[row][column], ",");
                }
                if (column == cols - 1) {
                    appender = appender.substring(0, appender.length() - 1);
                    appender += "          \n";
                }
                output += appender;
                appender = "";
            }
        }

        System.out.println("MATRIX:\n" + output);

    }//end method toString

    /**
     *
     * @param row The row in this Matrix object to be converted into a new
     * Matrix object. This operation generates a new Matrix object which is a
     * row matrix.
     */
    public Matrix getRowMatrix(int row) {
        Matrix arr = new Matrix(1, cols);
        for (int col = 0; col < cols; col++) {
            arr.array[col] = array[row * cols + col];
        }//end for
        return arr;
    }

    /**
     *
     * @param column The column to be converted into a new Matrix object. This
     * operation generates a new Matrix object which is a column matrix.
     */
    public Matrix getColumnMatrix(int column) {
        Matrix arr = new Matrix(rows, 1);
        for (int row = 0; row < rows; row++) {
            arr.array[row] = array[row * cols + column];
        }//end for
        return arr;
    }

    /**
     * Computes the inverse of this Matrix object. This technique should never
     * be used in practise as it is a mere proof of concept. It computes the
     * inverse by computing the roots of the equations and then reverse
     * engineering the form A.x = B to get the inverse matrix.
     *
     * @return the inverse of the Matrix as another Matrix object.
     */
    public Matrix oldInverse() {

        Matrix m = new Matrix(this);
        Matrix unit = m.unitMatrix();
        Matrix inverse = new Matrix(m.rows, m.cols);
        if (m.isSquareMatrix()) {
            for (int rws = 0; rws < m.cols; rws++) {
                Matrix c = Matrix.columnJoin(m, unit.getColumnMatrix(rws));
                inverse = Matrix.columnJoin(inverse, c.solveEquation());
            }//end for
        }//end if
        inverse.columnDeleteFromStart(m.rows);
        return inverse;
    }

    /**
     * Row reduction technique used to compute the inverse of the matrix. Always
     * use this technique.
     *
     * @return the inverse of the Matrix as another Matrix object.
     */
    public Matrix inverse() {

        Matrix m = new Matrix(this);
        if (m.isSquareMatrix()) {
            Matrix unit = m.unitMatrix();
            Matrix inverse = Matrix.columnJoin(this, unit);

            int rws = inverse.rows;
            int cls = inverse.cols;
            for (int row = 0; row < rws; row++) {

                double pivot = inverse.array[row * cls + row];

                /**
                 * The division coefficient must not be zero. If zero, search
                 * for a lower row, and swap.
                 */
                if (pivot == 0.0) {

                    for (int rw = row; rw < rws; rw++) {
                        pivot = inverse.array[rw * cls + row];

                        if (pivot != 0.0) {
                            inverse.swapRow(row, rw);
                            break;
                        }//end if

                    }//end for loop

                    if (pivot == 0.0) {
                        throw new InputMismatchException("INVERSE DOES NOT EXISTS!");
                    }
                }//end if   

                for (int col = row; col < cls; col++) {
                    inverse.array[row * cls + col] /= pivot;
                }//end inner for loop

                for (int rw = row + 1; rw < rws; rw++) {
                    double newRowMultiplier = -1 * inverse.array[rw * cls + row];
                    for (int col = row; col < cls; col++) {
                        inverse.array[rw * cls + col] = newRowMultiplier * inverse.array[row * cls + col] + inverse.array[rw * cls + col];
                    }
                }//end inner for loop

            }//end for
            //////////////Now reduce upwards from the right border of the main matrix...on the partition between the 2 matrices.

            for (int row = rws - 1; row >= 0; row--) {

                for (int rw = row - 1; rw >= 0; rw--) {

                    double newRowMultiplier = -1 * inverse.array[rw * cls + row];
                    /**
                     * The division coefficient must not be zero. If zero,
                     * search for a lower row, and swap.
                     */
                    if (newRowMultiplier == 0.0) {
                        continue;
                    }//end if  
                    for (int col = row; col < cls; col++) {
                        inverse.array[rw * cls + col] = newRowMultiplier * inverse.array[row * cls + col] + inverse.array[rw * cls + col];
                    }
                }

            }//end for

            inverse.columnDeleteFromStart(m.rows);
            return inverse;
        }//end if
        return null;
    }

    /**
     * Row reduction technique used to compute the determinant of this matrix.
     * The other method using recursion is not worth it above n = 10; The memory
     * consumed by the process and the time used to compute it is incomparable
     * to this method's performance.
     *
     * @return the inverse of the Matrix as another Matrix object.
     */
    public double determ() {

        double detMultiplier = 1;

        Matrix mat = new Matrix(this);
        //Now we row-reduce.
        int rws = mat.rows;
        int cls = mat.cols;

        if (rws == cls) {
            for (int row = 0; row < rws; row++) {
                double pivot = mat.array[row * cls + row];
                /**
                 * The division coefficient must not be zero. If zero, search
                 * for a lower row, and swap.
                 */
                if (pivot == 0.0) {
                    for (int rw = row; rw < rws; rw++) {
                        pivot = mat.array[rw * cls + row];
                        if (pivot != 0.0) {
                            mat.swapRow(row, rw);
                            detMultiplier *= -1;
                            break;
                        }//end if
                    }//end for loop
                    if (pivot == 0.0) {
                        throw new InputMismatchException("INVERSE DOES NOT EXISTS!");
                    }
                }//end if
                for (int col = row; col < cls; col++) {
                    mat.array[row * cls + col] /= pivot;
                }//end inner for loop
                detMultiplier *= pivot;
                for (int rw = row + 1; rw < rws; rw++) {
                    double newRowMultiplier = -1 * mat.array[rw * cls + row];
                    for (int col = row; col < cls; col++) {
                        mat.array[rw * cls + col] = newRowMultiplier * mat.array[row * cls + col] + mat.array[rw * cls + col];
                    }
                }//end inner for loop
            }//end for
            for (int row = 0; row < rws; row++) {
                detMultiplier *= mat.array[row * cls + row];
            }//end for
            return detMultiplier;
        }

        throw new InputMismatchException("The input to the determinant function be a square matrix!");
    }

    /**
     *
     * @param cofactors The matrix of cofactors.
     * @return the algebraic expression for the determinant of the matrix of
     * cofactors.
     */
    private static String findDetEigen(String[][] cofactors) {
        if (cofactors.length == 2 && cofactors[0].length == 2) {
            return $2X2determinantForEigen(cofactors);
        } else {
            StringBuilder eqnBuilder = new StringBuilder();
            int row = 0;
            for (int col = 0; col < cofactors[0].length; col++) {
                String[][] cfs = getCofactorTextArray(cofactors, row, col);
                String cofactorDet = findDetEigen(cfs);
                if (col == 0) {
                    String expr = uniVariableExpressionExpander(lambda, cofactors[row][col], cofactorDet);
                    if (!expr.isEmpty()) {
                        eqnBuilder.append("+").append(expr);
                    }
                } else {
                    if (col % 2 == 0) {
                        String expr = uniVariableExpressionExpander(lambda, cofactors[row][col], cofactorDet);
                        if (!expr.isEmpty()) {
                            eqnBuilder.append("+").append(expr);
                        }
                    } else {
                        String negExpr = uniVariableExpressionExpander(lambda, "-1", cofactors[row][col]);
                        if (!negExpr.isEmpty()) {
                            String expr = uniVariableExpressionExpander(lambda, negExpr, cofactorDet);
                            if (!expr.isEmpty()) {
                                eqnBuilder.append("+").append(expr);
                            }
                        }

                    }
                }
            }

            return eqnBuilder.toString();
        }
    }

    public final String getCharacteristicPolynomialForEigenVector() {
        return findCharacteristicPolynomialForEigenValues(this);
    }

    /**
     *
     * @param m The Matrix object
     * @return the characteristic polynomial
     */
    public static final String findCharacteristicPolynomialForEigenValues(Matrix m) {

        if (m.isSquareMatrix()) {

            int rws = m.rows;
            int cls = m.cols;

            String[][] mat = new String[rws][cls];

            //Create matrix array in string format
            for (int row = 0; row < rws; row++) {
                for (int col = 0; col < cls; col++) {
                    mat[row][col] = String.valueOf(m.array[row * cls + col]);
                }
            }

            //Create A-λ array in string format
            for (int row = 0, col = 0; row < rws; row++, col++) {
                String c = mat[row][col];
                mat[row][col] = c + "-" + lambda;
            }

            // printTextMatrix(mat);
            StringBuilder eqnBuilder = new StringBuilder();

            int row = 0;
            for (int col = 0; col < cls; col++) {
                String entry = mat[row][col];
                String cofactorDet;
                if (mat.length == 2) {
                    return findDetEigen(mat);
                }
                String[][] cofactors = getCofactorTextArray(mat, row, col);
                cofactorDet = findDetEigen(cofactors);

                if (col == 0) {
                    eqnBuilder.append(uniVariableExpressionExpander(lambda, entry, cofactorDet));
                } else {
                    if (col % 2 == 0) {
                        String expr = uniVariableExpressionExpander(lambda, entry, cofactorDet);
                        if (!expr.isEmpty()) {
                            eqnBuilder.append("+").append(expr);
                        }
                    } else {
                        String negExpr = uniVariableExpressionExpander(lambda, "-1", entry);
                        if (!negExpr.isEmpty()) {
                            String expr = uniVariableExpressionExpander(lambda, negExpr, cofactorDet);
                            if (!expr.isEmpty()) {
                                eqnBuilder.append("+").append(expr);
                            }
                        }
                    }
                }

            }

            String expr = uniVariableExpressionExpander(lambda, "1", eqnBuilder.toString());

            return expr;
        }

        return null;

    }

    /**
     *
     * @param rw The row of a given element
     * @param cl The position of that element
     * @return the cofactor sub-matrix used to calculate the cofactor element of
     * the specified position
     */
    private Matrix getCofactor(int rw, int cl) {
        if (rw >= 0 && cl >= 0) {

            Matrix mat = new Matrix(rows - 1, cols - 1);

            int subRow = 0;
            for (int row = 0; row < rows; row++) {
                if (row == rw) {
                    continue;
                }
                int subCol = 0;
                for (int col = 0; col < cols; col++) {
                    if (col == cl) {
                        continue;
                    }
                    mat.array[subRow * mat.cols + subCol] = array[row * this.cols + col];
                    subCol++;
                }
                subRow++;
            }
            return mat;
        }

        return null;
    }

    /**
     *
     * @param matrix A 2d matrix array
     * @param rw The row of a given element
     * @param cl The position of that element
     * @return the cofactor sub-matrix used to calculate the cofactor element of
     * the specified position
     */
    private static String[][] getCofactorTextArray(String[][] matrix, int rw, int cl) {
        if (rw >= 0 && cl >= 0) {

            int rws = matrix.length;
            int cls = matrix[0].length;

            String[][] mat = new String[rws - 1][cls - 1];

            int subRow = 0;
            for (int row = 0; row < rws; row++) {
                if (row == rw) {
                    continue;
                }
                int subCol = 0;
                for (int col = 0; col < cls; col++) {
                    if (col == cl) {
                        continue;
                    }
                    mat[subRow][subCol] = matrix[row][col];
                    subCol++;
                }
                subRow++;
            }
            return mat;
        }

        return null;
    }

    private void reduceToHessenberg(double[][] H, int n) {
        // We only need to go up to n-2 because the last 2x2 block is already Hessenberg
        for (int k = 0; k < n - 2; k++) {
            double[] v = new double[n];
            double sigma = 0;

            // 1. Calculate the norm of the column below the sub-diagonal
            for (int i = k + 1; i < n; i++) {
                sigma += H[i][k] * H[i][k];
            }

            if (sigma < 1e-15) {
                continue; // Already zeroed
            }
            double vk1 = H[k + 1][k];
            double mag = Math.sqrt(sigma);

            // Choose sign to avoid catastrophic cancellation
            double u1 = (vk1 >= 0) ? vk1 + mag : vk1 - mag;

            // 2. Create the Householder vector v
            v[k + 1] = 1.0;
            for (int i = k + 2; i < n; i++) {
                v[i] = H[i][k] / u1;
            }

            double beta = 2.0 / (1.0 + (sigma - vk1 * vk1) / (u1 * u1));

            // 3. Apply Householder Reflection from the left: H = (I - beta*v*vT) * H
            for (int j = k; j < n; j++) {
                double sum = 0;
                for (int i = k + 1; i < n; i++) {
                    sum += v[i] * H[i][j];
                }
                sum *= beta;
                for (int i = k + 1; i < n; i++) {
                    H[i][j] -= sum * v[i];
                }
            }

            // 4. Apply Householder Reflection from the right: H = H * (I - beta*v*vT)
            // This preserves the eigenvalues (Similarity Transformation)
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = k + 1; j < n; j++) {
                    sum += v[j] * H[i][j];
                }
                sum *= beta;
                for (int j = k + 1; j < n; j++) {
                    H[i][j] -= sum * v[j];
                }
            }
        }
    }

    private double calculateWilkinsonShift(double[][] H, int m) {
        // Indices for the bottom 2x2 submatrix
        double a = H[m - 2][m - 2];
        double b = H[m - 2][m - 1];
        double c = H[m - 1][m - 2];
        double d = H[m - 1][m - 1];

        // d is the current best guess for the eigenvalue
        // we calculate the shift based on the characteristic equation of the 2x2 block
        double delta = (a - d) / 2.0;

        // sign of delta to ensure we choose the eigenvalue closer to d
        double signDelta = (delta >= 0) ? 1.0 : -1.0;

        // Calculate the shift: mu = d - (sign(delta) * b * c) / (|delta| + sqrt(delta^2 + b*c))
        // This is a numerically stable version of the quadratic formula
        double denom = Math.abs(delta) + Math.sqrt(delta * delta + b * c);

        if (denom == 0) {
            return d; // Fallback to the last diagonal element
        }

        double shift = d - (signDelta * b * c) / denom;

        return shift;
    }

    private void qrStepHessenberg(double[][] H, int m) {
        double[] sin = new double[m - 1];
        double[] cos = new double[m - 1];

        // 1. QR Decomposition: Eliminate the sub-diagonal using Givens Rotations
        // This transforms H into R
        for (int i = 0; i < m - 1; i++) {
            double a = H[i][i];
            double b = H[i + 1][i];

            // Numerically stable calculation of Givens coefficients (c = cos, s = sin)
            if (b == 0) {
                cos[i] = 1;
                sin[i] = 0;
            } else {
                if (Math.abs(b) > Math.abs(a)) {
                    double tau = -a / b;
                    sin[i] = 1.0 / Math.sqrt(1 + tau * tau);
                    cos[i] = sin[i] * tau;
                } else {
                    double tau = -b / a;
                    cos[i] = 1.0 / Math.sqrt(1 + tau * tau);
                    sin[i] = cos[i] * tau;
                }
            }

            // Apply rotation to the columns of H (Left multiplication)
            for (int j = i; j < m; j++) {
                double h1 = H[i][j];
                double h2 = H[i + 1][j];
                H[i][j] = cos[i] * h1 - sin[i] * h2;
                H[i + 1][j] = sin[i] * h1 + cos[i] * h2;
            }
        }

        // 2. Complete the similarity transformation: H = R * Q
        // We apply the same rotations from the right (Right multiplication)
        for (int i = 0; i < m - 1; i++) {
            double c = cos[i];
            double s = sin[i];

            // Apply rotation to the rows of H
            for (int j = 0; j <= i + 1; j++) {
                double h1 = H[j][i];
                double h2 = H[j][i + 1];
                H[j][i] = c * h1 - s * h2;
                H[j][i + 1] = s * h1 + c * h2;
            }
        }
    }

    public double[] computeEigenValues() {
        int n = this.rows;
        double[][] H = this.getArray(); // Working copy

        // 1. Transform to Hessenberg Form (using Householder reflections)
        reduceToHessenberg(H, n);

        // 2. Iterative QR with shifts
        int maxIterations = 100 * n;
        int iter = 0;
        int m = n;

        while (m > 1 && iter < maxIterations) {
            // Look for convergence at the bottom of the matrix
            if (Math.abs(H[m - 1][m - 2]) < 1e-12) {
                m--; // Found an eigenvalue at H[m][m]
                continue;
            }

            // Wilkinson Shift calculation (using the bottom 2x2 submatrix)
            double mu = calculateWilkinsonShift(H, m);

            // Apply Shift: H = H - mu*I
            for (int i = 0; i < m; i++) {
                H[i][i] -= mu;
            }

            // QR Factorization (using Givens Rotations for O(n^2) efficiency)
            qrStepHessenberg(H, m);

            // Apply back: H = RQ + mu*I
            for (int i = 0; i < m; i++) {
                H[i][i] += mu;
            }

            iter++;
        }

        // 3. Extract eigenvalues from the diagonal
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = H[i][i];
        }
        return eigenvalues;
    }

    /**
     *
     * <code>A = VɅV-¹</code> Check: Is ||A*v - lambda*v|| close to zero?
     * double[] Av = matrixMultiply(A, v); double[] lv = scalarMultiply(lambda,
     * v); double residual = computeNorm(subtract(Av, lv));
     *
     * if (residual > 1e-9) { // This eigenvalue/vector pair might be inaccurate
     * }
     *
     * @param lambda
     * @return
     */
    public double[] computeEigenVector(double lambda) {
        Matrix A = new Matrix(this);
        int n = A.rows;
        // 1. Construct (A - λI)
        double[][] mat = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat[i][j] = A.getElem(i, j);
            }
            mat[i][i] -= lambda;
        }

        // 2. Gaussian Elimination with Partial Pivoting
        for (int i = 0; i < n - 1; i++) {
            // Pivot selection for numerical stability
            int pivot = i;
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(mat[j][i]) > Math.abs(mat[pivot][i])) {
                    pivot = j;
                }
            }
            double[] temp = mat[i];
            mat[i] = mat[pivot];
            mat[pivot] = temp;

            for (int j = i + 1; j < n; j++) {
                double factor = mat[j][i] / mat[i][i];
                for (int k = i; k < n; k++) {
                    mat[j][k] -= factor * mat[i][k];
                }
            }
        }

        // 3. Back-substitution
        // Since the matrix is singular, we expect the last row to be 0.
        // We set v[n-1] = 1.0 and solve upwards.
        double[] v = new double[n];
        v[n - 1] = 1.0;

        for (int i = n - 2; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += mat[i][j] * v[j];
            }
            // If mat[i][i] is nearly zero, this indicates a higher geometric multiplicity
            v[i] = (Math.abs(mat[i][i]) < 1e-12) ? 0 : -sum / mat[i][i];
        }

        // 4. Normalize the vector (World-Class Step)
        double norm = 0;
        for (double component : v) {
            norm += component * component;
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < n; i++) {
            v[i] /= norm;
        }

        return v;
    }

    /**
     * Generates the expression map of the expression... a map whose keys are
     * the powers of the variable of the expression and whose values are the
     * coefficients of the variables. e.g. 3x^2-2x+1 would produce:
     * [{2,3},{1,-2},{0,1}]
     *
     * @param variableName The name of the variable
     * @param scan The list to smoothen
     * @return the expression map
     */
    private static HashMap<Double, Double> generateExpressionMap(String variableName, List<String> scan) {
        if (scan.isEmpty()) {
            return new HashMap<>();
        }
        if (scan.get(0).equals(Operator.PLUS)) {
            scan.remove(0);
        }
        if (scan.isEmpty()) {
            return new HashMap<>();
        }

        if (scan.get(0).equals(Operator.MINUS)) {
            if (com.github.gbenroscience.parser.Number.isNumber(scan.get(1))) {
                scan.set(1, (-1 * Double.parseDouble(scan.get(1))) + "");
                scan.remove(0);
            }
        }

        /**
         * change kx^n to k*x^n
         */
        for (int i = 0; i < scan.size(); i++) {
            if (i > 0 && scan.get(i).equals(variableName)) {
                if (com.github.gbenroscience.parser.Number.isNumber(scan.get(i - 1))) {
                    scan.add(i, "*");
                    i += 1;
                }
            }
        }

        // apply coeffs to variables missing their coefficients
        for (int i = 0; i < scan.size(); i++) {
            if (scan.get(i).equals(variableName)) {

                if (i == 0) {
                    scan.add(0, "*");
                    scan.add(0, "1");
                    i += 2;
                } else {
                    if (Operator.isPlusOrMinus(scan.get(i - 1))) {
                        scan.add(i, "*");
                        scan.add(i, "1");
                        if (i == scan.size() - 1) {
                            break;
                        }
                        i += 2;
                    }
                }
            }
        }//end for loop

        //smoothing change constants; Change terms in x to: x^1 and constant terms to x^0
        for (int i = 0; i < scan.size(); i++) {

            if (i + 1 < scan.size()) {
                if (scan.get(i).equals(Operator.PLUS) && scan.get(i + 1).equals(Operator.PLUS)) {
                    scan.set(i, Operator.PLUS);
                    scan.remove(i + 1);
                }

                if (scan.get(i).equals(Operator.PLUS) && scan.get(i + 1).equals(Operator.MINUS)) {
                    scan.set(i, Operator.MINUS);
                    scan.remove(i + 1);
                }

                if (scan.get(i).equals(Operator.MINUS) && scan.get(i + 1).equals(Operator.PLUS)) {
                    scan.set(i, Operator.MINUS);
                    scan.remove(i + 1);
                }

                if (scan.get(i).equals(Operator.MINUS) && scan.get(i + 1).equals(Operator.MINUS)) {
                    scan.set(i, Operator.PLUS);
                    scan.remove(i + 1);
                }

            }
            //locate a variable
            if (scan.get(i).equals(variableName)) {
                //A free variable can never be at the beginning due to the previous for loop
                //if at the end
                if (i == scan.size() - 1) {//make its exponent 1
                    scan.add(Operator.POWER);
                    scan.add("1");
                    break;
                } else {//else if it is just before a ± operator (e.g. x + or x -), then make its exponent 1
                    if (Operator.isPlusOrMinus(scan.get(i + 1))) {
                        scan.add(i + 1, "1");
                        scan.add(i + 1, Operator.POWER);
                        i += 2;
                    }
                }

            } /*locate a number*/ else if (com.github.gbenroscience.parser.Number.isNumber(scan.get(i))) {

                if (i == 0) {  //if at start: make its x coefficient 0
                    if (i + 1 < scan.size()) {
                        if (Operator.isPlusOrMinus(scan.get(i + 1))) {
                            scan.add(1, "0");
                            scan.add(1, Operator.POWER);
                            scan.add(1, variableName);
                            scan.add(1, Operator.MULTIPLY);
                            i += 4;
                        }
                    } else {
                        scan.add(Operator.MULTIPLY);
                        scan.add(variableName);
                        scan.add(Operator.POWER);
                        scan.add("0");

                        break;

                    }

                } else if (i == scan.size() - 1) {//if at end, make its x coefficient 0
                    if (Operator.isPlusOrMinus(scan.get(i - 1))) {
                        scan.add(Operator.MULTIPLY);
                        scan.add(variableName);
                        scan.add(Operator.POWER);
                        scan.add("0");
                    }
                    break;
                } else {//if somewhere within the production, make its x coefficient 0
                    if (Operator.isPlusOrMinus(scan.get(i - 1)) && Operator.isPlusOrMinus(scan.get(i + 1))) {
                        scan.add(i + 1, Operator.MULTIPLY);
                        scan.add(i + 1, variableName);
                        scan.add(i + 1, Operator.POWER);
                        scan.add(i + 1, "0");
                        i += 4;
                    }
                }
            }

        }//end for loop

        HashMap<Double, Double> map = new HashMap<>();//key is the exponent(power of the variable), value is the coefficient
        //a.x^n
        for (int i = 0; i < scan.size(); i++) {
            if (scan.get(i).equals(Operator.POWER)) {
                double exp = Double.parseDouble(scan.get(i + 1));
                double coeff = Double.parseDouble(scan.get(i - 3));//3*x^2

                if (i - 4 >= 0 && scan.get(i - 4).equals(Operator.MINUS)) {
                    coeff *= -1;
                }
                double coef = map.getOrDefault(exp, 0.0);
                map.put(exp, coeff + coef);
            }
        }

        return map;
    }

    /**
     *
     * @param variableName The variable
     * @param exprs The different expressions of the variable to be multiplied
     * @return the expanded product of the expressions.
     */
    private static final String uniVariableExpressionExpander(String variableName, String... exprs) {

        String eqn = exprs[0];
        for (int i = 1; i < exprs.length; i++) {
            eqn = uniVariableExpressionExpander(variableName, eqn, exprs[i]);
        }
        return eqn;
    }

    /**
     *
     * @param expression Must be a math expression of the
     * form:(polynomial_1)(polynomial_2) e.g: (1*x^1-2)(3*x^2+5*x^1+8)
     * @return
     */
    private static final String uniVariableExpressionExpander(String variableName, String expr1, String expr2) {

        List<String> tokens1 = new Scanner(expr1, true, Operator.PLUS, Operator.MINUS, Operator.MULTIPLY, Operator.DIVIDE, Operator.POWER, variableName).scan();

        List<String> tokens2 = new Scanner(expr2, true, Operator.PLUS, Operator.MINUS, Operator.MULTIPLY, Operator.DIVIDE, Operator.POWER, variableName).scan();

        HashMap<Double, Double> map1 = generateExpressionMap(variableName, tokens1);
        HashMap<Double, Double> map2 = generateExpressionMap(variableName, tokens2);

        HashMap<Double, Double> product = new HashMap();

        for (HashMap.Entry<Double, Double> e : map1.entrySet()) {

            double exp = e.getKey();
            double coeff = e.getValue();

            for (HashMap.Entry<Double, Double> f : map2.entrySet()) {

                double ex = f.getKey();
                double coef = f.getValue();

                double coeffProd = coef * coeff;
                double expProd = ex + exp;//3x^2*5x^3

                double oldCoef = product.getOrDefault(expProd, 0.0);
                product.put(expProd, coeffProd + oldCoef);

            }

        }

        int i = 0;
        StringBuilder result = new StringBuilder();
        for (HashMap.Entry<Double, Double> e : product.entrySet()) {

            double exp = e.getKey();
            double coeff = e.getValue();
            if (coeff == 0) {
                continue;
            }

            if (i == 0) {
                result.append(coeff).append("*").append(variableName).append(Operator.POWER).append(exp);
                //result.append(appendLogic(coeff, variableName, exp));
            } else {
                if (coeff >= 0) {
                    result.append("+").append(coeff).append("*").append(variableName).append(Operator.POWER).append(exp);
                    //result.append("+").append(appendLogic(coeff, variableName, exp));
                } else {
                    result.append(coeff).append("*").append(variableName).append(Operator.POWER).append(exp);
                    //result.append("+").append(appendLogic(coeff, variableName, exp));
                }
            }

            i++;
        }
        String res = result.toString();
        /*
        res = res.replace("+-", "-");
        res = res.replace("-+", "-");
        res = res.replace("--", "+");
        res = res.replace("++", "+");
         */
        return res;
    }

    private static String appendLogic(double coeff, String variableName, double exp) {
        if (exp == 0.0) {
            return coeff + "";
        } else if (exp == 1) {
            return coeff + "*" + variableName;
        } else {
            return coeff + "*" + variableName + Operator.POWER + exp;
        }
    }

    public void print() {
        System.out.println(toString());
    }

    /**
     * (2-x)(3-x)(1-x)=(6-5x+x^2)(1-x)=6-11x+6x^2-x^3 {1, 2, 3, 4, 5} {6, 7, 8,
     * 9, 0} {1, 2, 3, 4, 5} {6, 7, 8, 9, 0} {1, 2, 3, 4, 5}
     *
     * @param args The command line args
     */
    public static void main(String... args) {

        String expanded = uniVariableExpressionExpander("x", "2-x", "-8-7*x+x^2");
        System.out.println("expanded: " + expanded);

        double array1[][] = {
            {1, 2, 3},
            {0, 4, 5},
            {1, 0, 6}
        };
        Matrix ma = new Matrix(array1);

        System.out.println("Matrix...");
        ma.print();

        Matrix cof = ma.getCofactorMatrix();

        System.out.println("Cofactor Matrix...");
        cof.print();

        String eqn = Matrix.findCharacteristicPolynomialForEigenValues(ma);

        System.out.println("eigen-value-rnd: " + eqn);

        double arr[][] = {
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 0},
            {12, -2, 8, 2, 7},
            {2, 9, -3, 5, 10},
            {21, 4, 13, 0, 15}
        };
        Matrix a = new Matrix(arr);

        System.out.println("eigen-eqn-a: " + findCharacteristicPolynomialForEigenValues(a));

        double ar[][] = {
            {2, 0, 0, 0},
            {1, 2, 0, 0},
            {0, 1, 3, 0},
            {0, 0, 1, 3}
        };
        a = new Matrix(ar);

        System.out.println("eigen-eqn-a: " + findCharacteristicPolynomialForEigenValues(a));

        double arr1[][] = {
            {2, 0, 0},
            {0, 4, 5},
            {0, 4, 3}
        };
        Matrix b = new Matrix(arr1);

        System.out.println("eigen-eqn-b: " + findCharacteristicPolynomialForEigenValues(b));

        double arr2[][] = {
            {2, 1},
            {1, 2}
        };
        Matrix c = new Matrix(arr2);

        System.out.println("eigen-eqn-c: " + findCharacteristicPolynomialForEigenValues(c));

        Matrix m1 = new Matrix(4, 4);
        double array[][] = {{1, -8, 2, 5}, {4, 8, 2, 4}, {6, 5, 2, 1}, {2, 1, 6, 8}};
        m1.setArray(array);
        System.out.printf("Matrix: %s\nDeterminant:\n %f\n", m1.toString(), det(m1, 0));
        Matrix triMatrix = m1.reduceToRowEchelonMatrix();

        System.out.printf("Echelon-matrix of above: %s\n", triMatrix.toString());

        int rows = 11;
        int cols = 11;
        Matrix m = new Matrix(rows, cols);
        m.randomFill(20);

        //System.out.println("Matrix: \n" + m);
        System.out.println("Processing begins.");

        double t0 = System.nanoTime();

        double det = m.sarusDet();// 

        double t1 = System.nanoTime() - t0;
        System.out.printf("Old method for determinant gives %4f in %4f %2s \n", det, (t1 * 1.0E-6), "ms");

        double t2 = System.nanoTime();

        double det_1 = m.determ();
        double t3 = System.nanoTime() - t2;

        System.out.printf("New method for determinant gives %4f in %4f %2s \n", det_1, (t3 * 1.0E-6), "ms");

        Matrix m11 = new Matrix(new double[][]{{5, -6, 8, 9}, {3, 1, 0, 6}, {2, 10, 4, 5}, {16, 12, 2, 4}});

        System.out.println("--------------------------Matrix:\n" + m11);
        System.out.println("Inverse: " + m11.inverse());

        Matrix m2 = new Matrix(2, 2);
        m2.randomFill(20);

        Matrix m3 = new Matrix(m2);
        System.out.println("OLD INVERSE METHOD ");

        System.out.println("NEW MATRIX: " + m2);

        Matrix m4 = m2.oldInverse();
        System.out.println("INVERSE: " + m4);

        System.out.println("Product using old method: M X 1/M: " + Matrix.multiply(m2, m4));

        System.out.println("NEW INVERSE METHOD ");

        System.out.println("NEW MATRIX: " + m3);

        Matrix m5 = m3.inverse();
        System.out.println("INVERSE: " + m5);

        System.out.println("Product using new method: M X 1/M: " + Matrix.multiply(m3, m5));
        /**
         * double t0 = System.nanoTime(); Matrix m = new Matrix(3024,3025);
         * double t1 = System.nanoTime() - t0; System.out.println( "Creating the
         * matrix in "+ (t1 * 1.0E-6)+" ms" ); t0 = System.nanoTime();
         * m.randomFill(22000); t1 = System.nanoTime() - t0;
         *
         * System.out.println( "Filling the matrix in "+ (t1 * 1.0E-6)+" ms" );
         * //System.out.print( m );
         *
         * t0 = System.nanoTime(); Matrix soln = m.solveEquation(); t1 =
         * System.nanoTime() - t0; System.out.print( "Solved the matrix in "+
         * (t1 * 1.0E-6)+" ms" ); //System.out.print( soln );
         *
         * //Matrix a = new
         * MatrixValueParser("[2,4,5:3,9.939,45.2:1,4,2:]").getMatrix();
         * //Matrix b = new
         * MatrixValueParser("[-2,-4,-5:-3,-9.939,-45.2:-1,-4,-2:]").getMatrix();
         *
         * //System.out.print( Matrix.pow(a, 5) );
         */
    }

}//end class Matrix
