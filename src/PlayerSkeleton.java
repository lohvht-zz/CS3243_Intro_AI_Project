import java.util.stream.IntStream;

import org.w3c.dom.css.Counter;

import java.util.Arrays;
import java.util.Random;

public class PlayerSkeleton {
	FeatureFunction f = new FeatureFunction();
	double[] weights = {
			1.982963276321388E-5, 1.1071800975126921E-7, 3.0996457506140976E-4, -2.52163738556298E-5, 4.450032757338146E-5, -6.323160558657387E-6, -8.81422807771626E-6, -6.781594693912656E-6, 5.899508205258744E-7,
	};
	NState nextState = new NState();

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		int bestMove = 0;
		double maxValue = -Double.MAX_VALUE;
		double currentValue = -Double.MAX_VALUE;
		for(int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			double[] featureValues = f.getFeatureValues(nextState);
			currentValue = f.calculateValue(featureValues, weights);
			// System.out.printf("state turn %d, next state turn %d, move %d, value %f\n", s.getTurnNumber(), nextState.getTurnNumber(), move, currentValue);
			if(currentValue > maxValue) {
				maxValue = currentValue;
				bestMove = move;
			}
		}
		return bestMove;
	}
	
	public static void main(String[] args) {
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
			s.draw();
			s.drawNext(0,0);
			try {
				// Thread.sleep(300); Uncomment to switch back to normal
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
	}
}

// Helper Class to calculate the value function or Q function, with a given state/action
class FeatureFunction {
	private static int countFeatures = 0;
	// Indexes of the feature array values
	public static final int INDEX_MAX_COL_HEIGHT = countFeatures++;
	public static final int INDEX_LANDING_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_ROWS_REMOVED = countFeatures++;
	public static final int INDEX_AV_DIFF_COL_HEIGHT = countFeatures++;
	public static final int INDEX_AV_COL_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_HOLES = countFeatures++;
	public static final int INDEX_COL_TRANSITION = countFeatures++;
	public static final int INDEX_ROW_TRANSITION = countFeatures++;
	public static final int INDEX_TOTAL_WELL_DEPTH = countFeatures++;

	public static final int NUM_FEATURES = countFeatures;

	/**
	 * Helper function that computes all the features and returns it as a vector
	 * @param nextState This is the next game state
	 * @return an array representing the vector of calculated feature values
	 */
	public double[] getFeatureValues(NState nextState) {
		double[] features = new double[NUM_FEATURES];
		// The rest of the feature vector
		features[INDEX_MAX_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_LANDING_HEIGHT] = getLandingHeight(nextState);
		features[INDEX_NUM_ROWS_REMOVED] = getRowsRemoved(nextState);
		features[INDEX_AV_DIFF_COL_HEIGHT] = getAverageDifferenceColumnHeight(nextState);
		features[INDEX_AV_COL_HEIGHT] = getAverageColumnHeight(nextState);
		double[] holesTransitions = getHolesTransitions(nextState);
		features[INDEX_NUM_HOLES] = holesTransitions[0];
		features[INDEX_COL_TRANSITION] = holesTransitions[1];
		features[INDEX_ROW_TRANSITION] = holesTransitions[2];
		features[INDEX_TOTAL_WELL_DEPTH] = getWellDepths(nextState);
		return features;
	}

	/**
	 * The maximum column height of the board
	 * @return      the highest row of the highest column is found, 0 if empty
	 */
	public double getMaxColHeight(NState state) {
		// TODO: Implement Me!
		int[] top = state.getTop();
		int max = top[0];
		for(int i=1;i < top.length;i++){
			if(top[i] > max){
	    		max = top[i];
	    	}
		} 
		return max;
	}

	/**
	 * Height where the piece is put (= the height of the column + (the height of
	 * the piece / 2))
	 */
	public double getLandingHeight(NState state) {
		int action = state.getCurrentAction();
		int piece = state.getNextPiece();
		int orient = state.legalMoves()[action][State.ORIENT];
		int slot = state.legalMoves()[action][State.SLOT];

		return state.getTop()[slot] + State.getpHeight()[piece][orient] / 2.0;
	}

	public double getRowsRemoved(NState nextState) {
		// Add extra 1 in there to avoid the chance a state where the feature returns 0
		return nextState.getRowsCleared() - nextState.getOState().getRowsCleared() + 1;
	}

	/**
	 * The average of all absolute differences of all column heights
	 */
	public double getAverageDifferenceColumnHeight(NState state) {
		// TODO: implement me!
		int[] top = state.getTop();
		double total = 0.0;
		for (int i=0; i<top.length-1; i++) {
			total += (double) Math.abs(top[i]-top[i+1]);
		}
		return total/((double)(state.COLS-1));
	}

	/**
	 * The average column height
	 */
	public double getAverageColumnHeight(NState state) {
		// TODO: implement me!
		return IntStream.of(state.getTop()).sum()/state.COLS;
	}

	/**
	 * For the following Features:
	 * Holes:
	 * 		number of empty cells that has at least one filled cell above
	 * 		it in the same column
	 * Row Transitions:
	 * 		number of filled gaps where the adjacent cell is empty along the same
	 * 		row, we count the side borders as a filled gap as well
	 * Column Transitions:
	 * 		number of filled gaps where the adjacent cell is empty along the same
	 * 		column
	 * @return	An array that has the calculated number of holes, row transitions,
	 * 			and column transitions of the form {HOLES, COL_TRANSITIONS, ROW_TRANSITIONS }
	 */
	public double[] getHolesTransitions(NState state) {
		int rowTransitions = 0;
		int colTransitions = 0;
		int holes = 0;

		int[][] field = state.getField();
		for(int i=0; i < State.ROWS - 1; i++) {
			// If cell next to the border on the right side is empty
			// we count that as a row transition
			if(field[i][0] == 0) { rowTransitions++; }
			// If cell next to the border on the left side is empty
			// we count that as a row transition
			if(field[i][State.COLS-1] == 0) { rowTransitions++; }
			for(int j=0; j<State.COLS; j++) {
				if(j > 0 && ((field[i][j] == 0) != (field[i][j-1] == 0))){ rowTransitions++; }
				if((field[i][j] != 0) != (field[i+1][j] != 0)) { colTransitions++; }
				if (field[i][j] <= 0 && field[i + 1][j] > 0) { holes++; }
		}
	}
		double[] result = { holes, colTransitions, rowTransitions };
		return result;
	}

	public double getWellDepths(NState state) {
		int[] top = state.getTop();

		double totalSum = 0;

		for(int i = 0; i < State.COLS; i++) {
			int left = i == 0 ? State.ROWS : top[i-1];
			int right = i == State.COLS -1 ? State.ROWS : top[i+1];
			// Take the shorter of
			int wellDepth = Math.min(left, right) - top[i];
			if(wellDepth > 0) {
				totalSum += (wellDepth * (wellDepth+1))/2;
			}
		}
		return totalSum;
	}

	public double calculateValue(double[] featureValues, double[] weight) {
		double values = 0;
		for (int i = 0; i < featureValues.length; i++) {
			values += featureValues[i]*weight[i];
		}
		return values;
	}
}

/**
 * State class extended to be more useful than the original State class
 */
class NState extends State {
	private static final int[][][] pBottom = State.getpBottom();
	private static final int[][] pHeight = State.getpHeight();
	private static final int[][][] pTop = State.getpTop();

	private State oState;
	private int turn = 0;
	private int cleared = 0;
	private int[] top = new int[COLS];
	private int[][] field = new int[ROWS][COLS];
	
	// Index of move made from the legalMoves array: Must be set!
	private int currentAction = -1; 

	/**
	 * Default Constructor
	 */
	public NState(){
		this.turn = 0;
		this.cleared = 0;
		this.field = new int[ROWS][COLS];
		this.top = new int[COLS];

		this.lost = false;
	};

	public void copy(State state) {
		// Preserve the original state
		this.setOState(state);
		// Copy all relevant private members to this new state
		this.setTurnNumber(this.oState.getTurnNumber());;
		this.setRowsCleared(this.oState.getRowsCleared());
		this.setField(this.oState.getField());
		this.setTop(this.oState.getTop());
		// replace relevant protected/public variables
		this.lost = this.oState.lost;
		this.setNextPiece(this.oState.getNextPiece());
		// currentAction set to -1 (not made a move yet)
		this.setCurrentAction(-1);
	}

	public State getOState() { return this.oState; }
	public void setOState(State _state) { this.oState = _state; }

	public int getTurnNumber() { return this.turn; }
	public void setTurnNumber(int _turn) { this.turn = _turn; }

	public int getRowsCleared() { return this.cleared; }
	public void setRowsCleared(int _cleared) { this.cleared = _cleared; }

	public int[] getTop() { return this.top; }
	public void setTop(int[] _top) {
		for (int i = 0; i < COLS; i++) {
			this.top[i] = _top[i];
		}
	}

	public int[][] getField(){ return this.field; }
	public void setField(int[][] _field) {
		for(int i=0; i<ROWS; i++) {
			for(int j=0; j<COLS; j++) {
				this.field[i][j] = _field[i][j];
			}
		}
	}

	public int getCurrentAction() { return this.currentAction; }
	public void setCurrentAction(int _currentAction) { this.currentAction = _currentAction; }

	public int getNextPiece() { return this.nextPiece; }
	public void setNextPiece(int _nextPiece) { this.nextPiece = _nextPiece; }

	public boolean hasLost() { return lost; }

	//make a move based on the move index - its order in the legalMoves list
	public void makeMove(int move) {
		this.setCurrentAction(move);
		makeMove(legalMoves[nextPiece][move]);
	}

	//returns false if you lose - true otherwise
	public boolean makeMove(int orient, int slot) {
		turn++;
		int height = top[slot] - pBottom[nextPiece][orient][0];
		//for each column beyond the first in the piece
		for (int c = 1; c < pWidth[nextPiece][orient]; c++) {
			height = Math.max(height, top[slot + c] - pBottom[nextPiece][orient][c]);
		}

		//check if game ended
		if (height + pHeight[nextPiece][orient] >= ROWS) {
			lost = true;
			return false;
		}

		//for each column in the piece - fill in the appropriate blocks
		for (int i = 0; i < pWidth[nextPiece][orient]; i++) {

			//from bottom to top of brick
			for (int h = height + pBottom[nextPiece][orient][i]; h < height + pTop[nextPiece][orient][i]; h++) {
				field[h][i + slot] = turn;
			}
		}
		//adjust top
		for (int c = 0; c < pWidth[nextPiece][orient]; c++) {
			top[slot + c] = height + pTop[nextPiece][orient][c];
		}
		//check for full rows - starting at the top
		for (int r = height + pHeight[nextPiece][orient] - 1; r >= height; r--) {
			//check all columns in the row
			boolean full = true;
			for (int c = 0; c < COLS; c++) {
				if (field[r][c] == 0) {
					full = false;
					break;
				}
			}
			//if the row was full - remove it and slide above stuff down
			if (full) {
				cleared++;
				//for each column
				for (int c = 0; c < COLS; c++) {

					//slide down all bricks
					for (int i = r; i < top[c]; i++) {
						field[i][c] = field[i + 1][c];
					}
					//lower the top
					top[c]--;
					while (top[c] >= 1 && field[top[c] - 1][c] == 0)
						top[c]--;
				}
			}
		}
		return true;
	}
}

class Player {
	FeatureFunction f = new FeatureFunction();
	double[] weights = new double[FeatureFunction.NUM_FEATURES];
	Random rand = new Random();
	// double[] weights = {
	// 	0.01312372243109472,
	// 	-0.990979431758106,
	// 	-0.02128292106473606,
	// 	-0.022732603091860426,
	// 	0.015024981557833383,
	// 	-0.08072804804837652,
	// 	0.023073229188750266,
	// 	-0.015957721520064583,
	// };
	NState nextState = new NState();
	/**
	 * current and successor features, in 1xk vectors (DO REMEMBER TO TRANSPOSE)
	 */
	public double[][] currentFeature = new double[1][FeatureFunction.NUM_FEATURES];
	public double[][] successorFeature = new double[1][FeatureFunction.NUM_FEATURES];
	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves, LearnerLSPI learner) {
		int bestMove = 0;
		double maxValue = -Double.MAX_VALUE;
		double currentValue = -Double.MAX_VALUE;

		for(int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			double[] featureValues = f.getFeatureValues(nextState);
			// System.out.println(Arrays.toString(featureValues));
			currentValue = f.calculateValue(featureValues, weights);
			if(currentValue > maxValue) {
				maxValue = currentValue;
				bestMove = move;
				// System.out.println("Old Val: " + Arrays.toString(successorFeature[0]));
				System.arraycopy(featureValues, 0, successorFeature[0], 0, featureValues.length);
				// System.out.println("New Val: "+Arrays.toString(successorFeature[0]));
				learner.LSTDQUpdate(currentFeature, successorFeature, nextState);
			}
			double[] tmp = successorFeature[0];
			successorFeature[0] = currentFeature[0];
			currentFeature[0] = tmp;
		}
		return bestMove;
	}

	Player() {
		this.rand = new Random();
		// Random Weights
		this.weights = new double[FeatureFunction.NUM_FEATURES];
		for(int i = 0; i < weights.length; i++) {
			this.weights[i] = this.rand.nextDouble();
		}
	}

	public static void playGame(LearnerLSPI learner) {
		State s = new State();
		Player p = new Player();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves(), learner));
		}
		// System.out.println("You have completed "+s.getRowsCleared()+" rows.");
	}
	
	public static void main(String[] args) {
		int count = 0;
		LearnerLSPI learner = new LearnerLSPI();
		int NUM_GAMES = 1000000;
		while(count < NUM_GAMES) {
			playGame(learner);
			count++;
			if(count%1000 == 0) {
				System.out.printf("Game %d of %d, %f through\n", count, NUM_GAMES, (double)count/NUM_GAMES*100);
			}
		}
		// LearnerLSPI.prettyPrintMatrix(learner.A);
		// LearnerLSPI.prettyPrintMatrix(learner.b);
		double[] weightUpdated = learner.getWeight();
		System.out.println("New weights: " + Arrays.toString(weightUpdated));
		// LearnerLSPI.prettyPrintMatrix(learner.A);
	}
}

class LearnerLSPI {
	/**
	 * Running using LSTDQ, where we can find the weight vector using
	 * w = inv(A) * b
	 * A is a kxk matrix of the following form, at each step:
	 * A := A + currentFeatures*transpose(currentFeatures - DISCOUNT*successorFeatures)
	 * 
	 * And b is a kx1 vector of the following form, at each step:
	 * b := b + currentFeatures*currentReward
	 * 
	 * where currentFeatures, successorFeatures are kx1 vectors representing the feature array
	 */
	public static double DISCOUNT = 0.96f;
	
	public double[][] A = new double[FeatureFunction.NUM_FEATURES][FeatureFunction.NUM_FEATURES];
	public double[][] stepA = new double[FeatureFunction.NUM_FEATURES][FeatureFunction.NUM_FEATURES];
	public double[][] b = new double[FeatureFunction.NUM_FEATURES][1];

	/**
	 * currentFeature and successFeature passed in are both 1xk vectors
	 */
	public void LSTDQUpdate(double[][] currentFeature, double[][] successorFeature, NState state) {
		double[][] featureDiff = matrixSum(currentFeature, scalarMultiply(successorFeature, -1*DISCOUNT), false, false);
		// calculate currentFeature*transpose(currentFeatures - DISCOUNT*successorFeatures)
		matrixMultiply(currentFeature, featureDiff, stepA, true, false);
		A = matrixSum(A, stepA, false, false);
		int reward = state.getRowsCleared() - state.getOState().getRowsCleared();
		// prettyPrintMatrix(b);
		b = matrixSum(b, scalarMultiply(currentFeature, (double)reward), false, true);	
	}

	public double[] getWeight() {
		double[][] weightsVector = new double[FeatureFunction.NUM_FEATURES][1];
		matrixMultiply(invert(A), b, weightsVector, false, false);
		double[] weights = new double[FeatureFunction.NUM_FEATURES];
		for(int i = 0; i < FeatureFunction.NUM_FEATURES; i++) {
			weights[i] = weightsVector[i][0];
		}
		return weights;
	}

	public static void prettyPrintMatrix(double[][] matrix) {
		System.out.println("=======================");
		for(int i = 0; i < matrix.length; i++) {
			System.out.println(Arrays.toString(matrix[i]));
		}
		System.out.println("=======================");
	}

	/**
	 * Multiply matrix1 (NxM) and matrix2 (MxK), where the result is stored in
	 * the provided resultMatrix of size (NxK)
	 * @param matrix1 	left matrix
	 * @param matrix2	right matrix
	 * @param resultMatrix	Resultant Matrix, will be returned
	 * @param transposeMatrix1	if True, transpose matrix1
	 * @param transposeMatrix2	if True, transpose matrix2
	 */
	public static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2, double[][] resultMatrix, boolean transposeMatrix1, boolean transposeMatrix2) {
		//Transposing is just swapping col and row indexes
		int m1Row = (!transposeMatrix1) ? matrix1.length : matrix1[0].length;
		int m1Col = (!transposeMatrix1) ? matrix1[0].length :matrix1.length;
		int m2Row = (!transposeMatrix2) ? matrix2.length : matrix2[0].length;
		int m2Col = (!transposeMatrix2) ? matrix2[0].length : matrix2.length;
		int rmRow = resultMatrix.length;
		int rmCol = resultMatrix[0].length;

		double c1 = -1;
		double c2 = -1;

		if(m1Col != m2Row) {
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

	public static double[][] matrixSum(double[][] matrix1, double[][] matrix2, boolean transposeMatrix1, boolean transposeMatrix2) {
		//Transposing is just swapping col and row indexes
		int m1Row = (!transposeMatrix1) ? matrix1.length : matrix1[0].length;
		int m1Col = (!transposeMatrix1) ? matrix1[0].length : matrix1.length;
		int m2Row = (!transposeMatrix2) ? matrix2.length : matrix2[0].length;
		int m2Col = (!transposeMatrix2) ? matrix2[0].length : matrix2.length;

		double c1 = -1;
		double c2 = -1;

		if((m1Row != m2Row) || (m1Col != m2Col)) {
			return null;
		}
		double[][] resultMatrix = new double[m1Row][m1Col];

		for(int i = 0; i < m1Row; i++) {
			for(int j = 0; j < m1Col; j++) {
				c1 = (!transposeMatrix1) ? matrix1[i][j] : matrix1[j][i];
				c2 = (!transposeMatrix2) ? matrix2[i][j] : matrix2[j][i];
				resultMatrix[i][j] = c1 + c2;
			}
		}
		return resultMatrix;
	}

	public static double[][] scalarMultiply(double[][] matrix, double scalar) {
		double[][] result = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				result[i][j] = matrix[i][j] * scalar;
			}
		}
		return result;
	}

	public static double[][] invert(double a[][]) {
		int n = a.length;
		double x[][] = new double[n][n];
		double b[][] = new double[n][n];
		int index[] = new int[n];
		for (int i = 0; i < n; ++i)
			b[i][i] = 1;

		// Transform the matrix into an upper triangle
		gaussian(a, index);

		// Update the matrix b[i][j] with the ratios stored
		for (int i = 0; i < n - 1; ++i)
			for (int j = i + 1; j < n; ++j)
				for (int k = 0; k < n; ++k)
					b[index[j]][k] -= a[index[j]][i] * b[index[i]][k];

		// Perform backward substitutions
		for (int i = 0; i < n; ++i) {
			x[n - 1][i] = b[index[n - 1]][i] / a[index[n - 1]][n - 1];
			for (int j = n - 2; j >= 0; --j) {
				x[j][i] = b[index[j]][i];
				for (int k = j + 1; k < n; ++k) {
					x[j][i] -= a[index[j]][k] * x[k][i];
				}
				x[j][i] /= a[index[j]][j];
			}
		}
		return x;
	}

	public static void gaussian(double a[][], int index[]) {
		int n = index.length;
		double c[] = new double[n];

		// Initialize the index
		for (int i = 0; i < n; ++i)
			index[i] = i;

		// Find the rescaling factors, one from each row
		for (int i = 0; i < n; ++i) {
			double c1 = 0;
			for (int j = 0; j < n; ++j) {
				double c0 = Math.abs(a[i][j]);
				if (c0 > c1)
					c1 = c0;
			}
			c[i] = c1;
		}

		// Search the pivoting element from each column
		int k = 0;
		for (int j = 0; j < n - 1; ++j) {
			double pi1 = 0;
			for (int i = j; i < n; ++i) {
				double pi0 = Math.abs(a[index[i]][j]);
				pi0 /= c[index[i]];
				if (pi0 > pi1) {
					pi1 = pi0;
					k = i;
				}
			}

			// Interchange rows according to the pivoting order
			int itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;
			for (int i = j + 1; i < n; ++i) {
				double pj = a[index[i]][j] / a[index[j]][j];

				// Record pivoting ratios below the diagonal
				a[index[i]][j] = pj;

				// Modify other elements accordingly
				for (int l = j + 1; l < n; ++l)
					a[index[i]][l] -= pj * a[index[j]][l];
			}
		}
	}
}
