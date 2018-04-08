import java.util.Arrays;

/**
 * Utility class for matrix operations
 */
public class Matrix {
    public static void main(String[] args) {
        // 2x4 matrix
        double[][] testMatrix1 = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, };
        // 4x2 matrix
        double[][] testMatrix2 = { { 9, 10, }, { 11, 12, }, { 13, 14, }, { 15, 16, }, };
        // 3x4 matrix, with duplicate rows
        double[][] testMatrix3 = { { 3, 4, 5, 6 }, { 1.5, 2, 2.5, 3 }, { 4, 5, 6, 7 }, };

        // 3x4 matrix, with empty column
        double[][] testMatrix4 = { { 3, 4, 0, 6 }, { 1.5, 2, 0, 3 }, { 4, 5, 0, 7 }, };

        double[][] testMatrix5 = { { 13, 4, 10 }, { 7, 8, 5 }, { 11, 12, 6 }, };

        double[][] testMatrix6 = { { 4, 5, 2, 14 }, { 3, 9, 6, 21 }, { 8, 10, 7, 28 }, { 1, 2, 9, 5 }, };

        double[][] identityMatrix3by3 = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, };

        double[][] identityMatrix4by4 = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 }, };

        System.out.println("Testing matrix multiplication 1");
        double[][] resultMultiply1 = matrixMultiply(testMatrix1, testMatrix2, false, false);
        prettyPrintMatrix(resultMultiply1);
        System.out.printf("Row X Col: %dx%d\n\n", resultMultiply1.length, resultMultiply1[0].length);

        System.out.println("Testing matrix multiplication 2");
        double[][] resultMultiply2 = matrixMultiply(testMatrix1, testMatrix2, true, true);
        prettyPrintMatrix(resultMultiply2);
        System.out.printf("Row X Col: %dx%d\n\n", resultMultiply1.length, resultMultiply1[0].length);

        System.out.println("Testing GJ elmination 1");
        prettyPrintMatrix(testMatrix3);
        gaussJordanElmination(testMatrix3);
        prettyPrintMatrix(testMatrix3);

        System.out.println("Testing GJ elmination 2");
        prettyPrintMatrix(testMatrix4);
        gaussJordanElmination(testMatrix4);
        prettyPrintMatrix(testMatrix4);

        System.out.println("Testing Solving Matrix 1");
        prettyPrintMatrix(solveMatrix(testMatrix5, identityMatrix3by3));

        System.out.println("Testing Solving Matrix 2");
        if (solveMatrix(testMatrix6, identityMatrix4by4) == null) {
            System.out.println("Matrix is inverse!");
        }
    }

    /**
     * Do gauss jordan elimnation to the matrix passed in, reducing it to its
     * reduced row echelon form
     */
    public static void gaussJordanElmination(double[][] augmentedMatrix) {
        int startColumn = 0;
        for (int row = 0; row < augmentedMatrix.length; row++) {
            // if the number at the start column is 0, try to find a non-zero row
            // for the start column, and swap with that row
            while (startColumn < augmentedMatrix[row].length && augmentedMatrix[row][startColumn] == 0.0) {
                boolean switched = false;
                int i = row;
                while (!switched && i < augmentedMatrix.length) {
                    if (augmentedMatrix[i][startColumn] != 0.0) {
                        swapRow(augmentedMatrix, row, i);
                        switched = true;
                    }
                    i++;
                }
                //If entire row has no non-zero rows column-wise, move to the next column
                if (augmentedMatrix[row][startColumn] == 0.0) {
                    startColumn++;
                }
            }
            // When the for loop at the top shifts the startColumn all the way to
            // the end of length of that row
            // (i.e. when we have a total zero row)
            if (startColumn >= augmentedMatrix[row].length) {
                return;
            }

            //if the number isn't one, reduce to one
            if (augmentedMatrix[row][startColumn] != 1.0) {
                double divisor = augmentedMatrix[row][startColumn];
                scalarRowMultiply(augmentedMatrix, row, 1 / divisor);
            }
            //make sure the number in the start column of all other rows is 0
            for (int i = 0; i < augmentedMatrix.length; i++) {
                if (i != row && augmentedMatrix[i][startColumn] != 0) {
                    double multiple = 0 - augmentedMatrix[i][startColumn];
                    rowAddition(augmentedMatrix, i, row, multiple);
                }
            }
            startColumn++;
        }
    }

    /**
     * Wrapper to solve AX = B using gauss jordan elimination
     * @return 	the solution vector/matrix if the resultant LHS is reduced to an
     * 			identity matrix, else returns null
     */
    public static double[][] solveMatrix(double[][] A, double[][] B) {
        double[][] augMat = getAugmentedMatrix(A, B);
        gaussJordanElmination(augMat);
        int numOfColsCoefficientMatrix = A[0].length;
        double[][] reducedA = getCoefficientMatrix(augMat, numOfColsCoefficientMatrix);
        return (isIdentity(reducedA)) ? getSolutionMatrix(augMat, numOfColsCoefficientMatrix) : null;
    }

    public static boolean isIdentity(double[][] matrix) {
        if (matrix.length != matrix[0].length) {
            return false;
        }
        boolean isIdent = true;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if ((i == j && matrix[i][j] != 1.0) || (i != j && matrix[i][j] != 0.0)) {
                    isIdent = false;
                    break;
                }
            }
            if (!isIdent) {
                break;
            }
        }
        return isIdent;
    }

    public static double[][] getIdentityMatrix(int dimension) {
        double[][] identityMat = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            identityMat[i][i] = 1;
        }
        return identityMat;
    }

    public static void prettyPrintMatrix(double[][] matrix) {
        System.out.println("=======================");
        for (int i = 0; i < matrix.length; i++) {
            System.out.println(Arrays.toString(matrix[i]));
        }
        System.out.println("=======================");
    }

    /**
     * ============================================
     * Row Operations: Mutates the row passed in for a given matrix
     * ============================================
     */

    private static void swapRow(double[][] matrix, int firstRowIndex, int secondRowIndex) {
        double[] temp = matrix[firstRowIndex];
        matrix[firstRowIndex] = matrix[secondRowIndex];
        matrix[secondRowIndex] = temp;
    }

    private static void scalarRowMultiply(double[][] matrix, int row, double scalar) {
        for (int j = 0; j < matrix[row].length; j++) {
            matrix[row][j] = matrix[row][j] * scalar;
        }
    }

    /**
     * Row addition, performs addition in the form of row1 = row1 + multiple * row2
     */
    private static void rowAddition(double[][] matrix, int row1, int row2, double multiple) {
        for (int j = 0; j < matrix[row1].length; j++) {
            matrix[row1][j] = matrix[row1][j] + matrix[row2][j] * multiple;
        }
    }

    /**
     * ============================================
     * Matrix Operations, Retrieves a new matrix after performing the operation
     * ============================================
     */

    public static double[][] getAugmentedMatrix(double[][] A, double[][] B) {
        // Increase the column size
        double[][] augMat = new double[A.length][A[0].length + B[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                augMat[i][j] = A[i][j];
            }
        }
        for (int i = 0; i < B.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                augMat[i][j + A[0].length] = B[i][j];
            }
        }
        return augMat;
    }

    public static double[][] getSolutionMatrix(double[][] augmentedMatrix, int numOfColsCoefficientMatrix) {
        int numRows = augmentedMatrix.length;
        // number of columns that solution matrix has is:
        // total number of columns - number of columns that coefficient matrix has
        int numCols = augmentedMatrix[0].length - numOfColsCoefficientMatrix;
        double[][] result = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            System.arraycopy(augmentedMatrix[i], numOfColsCoefficientMatrix, result[i], 0, numCols);
        }
        return result;
    }

    public static double[][] getCoefficientMatrix(double[][] augmentedMatrix, int numOfColsCoefficientMatrix) {
        int numRows = augmentedMatrix.length;
        // number of columns that solution matrix has is:
        // total number of columns - number of columns that coefficient matrix has
        int numCols = numOfColsCoefficientMatrix;
        double[][] result = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            System.arraycopy(augmentedMatrix[i], 0, result[i], 0, numCols);
        }
        return result;
    }

    /**
     * Multiply matrix1 (NxM) and matrix2 (MxK)
     * @param matrix1 	left matrix
     * @param matrix2	right matrix
     * @param transposeMatrix1	if True, transpose matrix1
     * @param transposeMatrix2	if True, transpose matrix2
     */
    public static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2, boolean transposeMatrix1,
            boolean transposeMatrix2) {
        //Transposing is just swapping col and row indexes
        int m1Row = (!transposeMatrix1) ? matrix1.length : matrix1[0].length;
        int m1Col = (!transposeMatrix1) ? matrix1[0].length : matrix1.length;
        int m2Row = (!transposeMatrix2) ? matrix2.length : matrix2[0].length;
        int m2Col = (!transposeMatrix2) ? matrix2[0].length : matrix2.length;

        double[][] resultMatrix = new double[m1Row][m2Col];
        int rmRow = resultMatrix.length;
        int rmCol = resultMatrix[0].length;
        double c1 = -1;
        double c2 = -1;

        if (m1Col != m2Row) {
            return null;
        } else if (rmRow != m1Row || rmCol != m2Col) {
            return null;
        }

        for (int i = 0; i < rmRow; i++) {
            for (int j = 0; j < rmCol; j++) {
                resultMatrix[i][j] = 0;
                for (int k = 0; k < m1Col; k++) {
                    c1 = (!transposeMatrix1) ? matrix1[i][k] : matrix1[k][i];
                    c2 = (!transposeMatrix2) ? matrix2[k][j] : matrix2[j][k];
                    resultMatrix[i][j] += c1 * c2;
                }
            }
        }
        return resultMatrix;
    }

    public static double[][] matrixSum(double[][] matrix1, double[][] matrix2, boolean transposeMatrix1,
            boolean transposeMatrix2) {
        //Transposing is just swapping col and row indexes
        int m1Row = (!transposeMatrix1) ? matrix1.length : matrix1[0].length;
        int m1Col = (!transposeMatrix1) ? matrix1[0].length : matrix1.length;
        int m2Row = (!transposeMatrix2) ? matrix2.length : matrix2[0].length;
        int m2Col = (!transposeMatrix2) ? matrix2[0].length : matrix2.length;

        double c1 = -1;
        double c2 = -1;

        if ((m1Row != m2Row) || (m1Col != m2Col)) {
            return null;
        }
        double[][] resultMatrix = new double[m1Row][m1Col];

        for (int i = 0; i < m1Row; i++) {
            for (int j = 0; j < m1Col; j++) {
                c1 = (!transposeMatrix1) ? matrix1[i][j] : matrix1[j][i];
                c2 = (!transposeMatrix2) ? matrix2[i][j] : matrix2[j][i];
                resultMatrix[i][j] = c1 + c2;
            }
        }
        return resultMatrix;
    }

    public static double[][] matrixScalarMultiply(double[][] matrix, double scalar) {
        return matrixScalarOperation(matrix, scalar, MULT);
    }

    public static double[][] matrixScalarAdd(double[][] matrix, double scalar) {
        return matrixScalarOperation(matrix, scalar, ADD);
    }

    private static final int ADD = 1, MULT = 2;

    private static double[][] matrixScalarOperation(double[][] matrix, double scalar, int op) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                switch (op) {
                case ADD:
                    result[i][j] = matrix[i][j] + scalar;
                    break;
                case MULT:
                    result[i][j] = matrix[i][j] * scalar;
                    break;
                default:
                    break;
                }
            }
        }
        return result;
    }
}
