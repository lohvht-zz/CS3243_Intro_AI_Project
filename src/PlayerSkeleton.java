import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;
public class PlayerSkeleton {
	FeatureFunction f = new FeatureFunction();


	double[] weights = {
		-4.500158825082766,
		4.500158825082766,
		-4.500158825082766,
		-3.4181268101392694,
		-2.52163738556298,
		8.450032757338146E-5,
		-7.899265427351652,
		-3.2178882868487753,
		-9.348695305445199,
		-9.348695305445199,
		-3.3855972247263626,
	};
	NState nextState = new NState();
	double[] featureValues = null;

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		int bestMove = 0;
		double maxValue = -Double.MAX_VALUE;
		double currentValue = -Double.MAX_VALUE;

		for(int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			featureValues = f.getFeatureValues(nextState);
			currentValue = f.calculateValue(featureValues, weights);
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
				// Thread.sleep(300); // Uncomment this to revert to normal sleep 300
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
		/*
		StateGenerator sg = new StateGenerator();
		sg.generateState(20);
		sg.generatePseudoBestState(20);
		*/
	}
	
}

// Helper Class to calculate the value function or Q function, with a given state/action
class FeatureFunction {
	private static int countFeatures = 0;
	// Indexes of the feature array values
	public static final int INDEX_MAX_HEIGHT = countFeatures++;
	public static final int INDEX_MIN_HEIGHT = countFeatures++;
	public static final int INDEX_AV_HEIGHT = countFeatures++;
	public static final int INDEX_AV_DIFF_HEIGHT = countFeatures++;
	public static final int INDEX_LANDING_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_ROWS_REMOVED = countFeatures++;
	public static final int INDEX_NUM_HOLES = countFeatures++;
	public static final int INDEX_COL_TRANSITION = countFeatures++;
	public static final int INDEX_ROW_TRANSITION = countFeatures++;
	public static final int INDEX_COVERED_GAPS = countFeatures++;
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
		double[] columnFeatures = getColumnFeatures(nextState);
		features[INDEX_MAX_HEIGHT] = columnFeatures[0];
		features[INDEX_MIN_HEIGHT] = columnFeatures[1];
		features[INDEX_AV_HEIGHT] = columnFeatures[2];
		features[INDEX_AV_DIFF_HEIGHT] = columnFeatures[3];
		features[INDEX_LANDING_HEIGHT] = getLandingHeight(nextState);
		features[INDEX_NUM_ROWS_REMOVED] = getRowsRemoved(nextState);
		double[] holesTransitions = getHolesTransitionsCoveredGaps(nextState);
		features[INDEX_NUM_HOLES] = holesTransitions[0];
		features[INDEX_COL_TRANSITION] = holesTransitions[1];
		features[INDEX_ROW_TRANSITION] = holesTransitions[2];
		features[INDEX_COVERED_GAPS] = holesTransitions[3];
		features[INDEX_TOTAL_WELL_DEPTH] = getWellDepths(nextState);
		return features;
	}

	/**
	* MAX HEIGHT:
	* 		Height of the tallest column
	* MIN HEIGHT:
	* 		Height of the shortest column
	* AV HEIGHT:
	* 		Average height of all columns
	* DIFF HEIGHT:
	* 		Average height difference of all adjacent columns
	*/
	public double[] getColumnFeatures(NState state) {
		int[] top = state.getTop();
		int maxHeight = 0;
		int minHeight = Integer.MAX_VALUE;
		int totalHeight = 0;
		int totalDiffHeight = 0;

		for (int i = 0; i < State.COLS; i++) {
			totalHeight += top[i];
			totalDiffHeight += (i > 0) ? Math.abs(top[i] - top[i - 1]) : 0;
			maxHeight = Math.max(maxHeight, top[i]);
			minHeight = Math.min(minHeight, top[i]);
		}
		double[] result = { maxHeight, minHeight, ((double) totalHeight) / State.COLS,
				((double) totalDiffHeight) / (State.COLS - 1) };
		return result;
	}

	/**
	 * Height where the piece is put (= the height of the column + (the height of
	 * the piece / 2))
	 */
	public double getLandingHeight(NState state) {
	    int nextPiece = state.getNextPiece();
	    
	    int move = state.getCurrentAction();
	    int[][] moves = state.legalMoves();
	    int orient = moves[move][State.ORIENT];
	    int slot = moves[move][State.SLOT];
	    
	    int[][] pWidth = State.getpWidth();
	    int pieceWidth = pWidth[nextPiece][orient];
	    int[][][] pTop = State.getpTop();
	    
	    int[] top = state.getTop();
	    int maxLandingHeight = -Integer.MAX_VALUE;
	    
	    for(int c = 0; c < pieceWidth; c++) {
	        int currentLandingHeight = top[slot+c]+pTop[nextPiece][orient][c] / 2;
	        if (currentLandingHeight > maxLandingHeight) {
	            maxLandingHeight = currentLandingHeight;
	        }
	    }
	    return maxLandingHeight;
	}

	public double getRowsRemoved(NState nextState) {
		return nextState.getRowsCleared() - nextState.getOState().getRowsCleared();
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
	* Covered Gaps:
	* 		number of empty holes that are covered up
	* @return	An array that has the calculated number of holes, row transitions,
	* 			and column transitions of the form {HOLES, COL_TRANSITIONS, ROW_TRANSITIONS }
	*/
	public double[] getHolesTransitionsCoveredGaps(NState state) {
		int rowTransitions = 0;
		int colTransitions = 0;
		int holes = 0;
		int coveredGaps = 0;

		int[][] field = state.getField();
		int[] top = state.getTop();
		for (int i = 0; i < State.ROWS - 1; i++) {
			// If cell next to the border on the right side is empty
			// we count that as a row transition
			if (field[i][0] == 0) {
				rowTransitions++;
			}
			// If cell next to the border on the left side is empty
			// we count that as a row transition
			if (field[i][State.COLS - 1] == 0) {
				rowTransitions++;
			}
			for (int j = 0; j < State.COLS; j++) {
				if (j > 0 && ((field[i][j] == 0) != (field[i][j - 1] == 0))) {
					rowTransitions++;
				}
				if ((field[i][j] != 0) != (field[i + 1][j] != 0)) {
					colTransitions++;
				}
				if (field[i][j] <= 0 && field[i + 1][j] > 0) {
					holes++;
				}
				if (field[i][j] <= 0 && i < top[j]) {
					coveredGaps++;
				}
			}
		}
		double[] result = { holes, colTransitions, rowTransitions, coveredGaps };
		return result;
	}

	public double getWellDepths(NState state) {
		int[] top = state.getTop();

		double totalSum = 0;

		for (int i = 0; i < State.COLS; i++) {
			int left = i == 0 ? State.ROWS : top[i - 1];
			int right = i == State.COLS - 1 ? State.ROWS : top[i + 1];
			// Take the shorter of
			int wellDepth = Math.min(left, right) - top[i];
			if (wellDepth > 0) {
				totalSum += (wellDepth * (wellDepth + 1)) / 2;
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
		System.arraycopy(_top, 0, this.top, 0, _top.length);
	}

	public int[][] getField(){ return this.field; }
	public void setField(int[][] _field) {
		for(int i=0; i<ROWS; i++) {
			System.arraycopy(_field[i], 0, this.field[i], 0, _field[i].length);
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

/**
 * Class to generate a random state that is valid
 * Things to ensure that state is valid:
 * 		field is set as int[][], with no TOTALLY EMPTY rows below the max height
 * 		top is set as int[], and follows with respective max heights for each
 * 		column as stated in field
 * 		nextPiece is set as int, and is between the numbers 0-7, where 7 is the
 * 		State.N_PIECE
 */
class StateGenerator {
	
	public static NState generatePseudoBestState(int moves) {
		FeatureFunction ff = new FeatureFunction();
		State state = new State();
		NState prevState = new NState();
		for (int i=0; i<moves; i++) {
			if (!state.hasLost()) {
				//save existing state
				prevState.copy(state);
				//set lowest height to postive inf
				double lowestHeight = Double.MAX_VALUE;
				//create a temp state to check moves, allows copying of prev state to check all moves
				NState tempState = new NState();
				//store all best moves here
				ArrayList<Integer> bestMoves = new ArrayList<Integer>();
				for (int j=0; j<prevState.legalMoves().length; j++) {
					tempState.copy(prevState);
					tempState.makeMove(j);
					//need to know how to use feature function
					if (ff.getColumnFeatures(tempState)[0]<lowestHeight) {
						bestMoves.clear();
						bestMoves.add(j);
						lowestHeight = ff.getColumnFeatures(tempState)[0];
					} else if (ff.getColumnFeatures(tempState)[0]==lowestHeight){
						bestMoves.add(j);
					}
				}
				Random r = new Random();
				state.makeMove(bestMoves.get(r.nextInt(bestMoves.size())));
			} else {
				System.out.println("Game Lost, printing Previous State:");
				printState(prevState);
				return prevState;
			}
		}
		prevState.copy(state);
		System.out.println("Moves Complete, printing Final State:");
		printState(prevState);
		return prevState;
	}
	
	public static NState generateState(int moves) {
		State state = new State();
		NState prevState = new NState();
		for (int i=0; i<moves; i++) {
			if (!state.hasLost()) {
				//System.out.println("Before move made.");
				//printState(state);
				prevState.copy(state);
				Random r = new Random();
				state.makeMove(r.nextInt(state.legalMoves().length));
				//System.out.println("\nAfter move made");
				//printState(state);
			} else {
				System.out.println("Game Lost, printing Previous State:");
				printState(prevState);
				return prevState;
			}
		}
		prevState.copy(state);
		System.out.println("Moves Complete, printing Final State:");
		printState(prevState);
		return prevState;
	}
	
	public static void printState(State s) {
		int[][] field = s.getField();
		for (int i=s.ROWS-1; i>=0;i--) {
			String row = "";
			for (int j=s.COLS-1; j>=0; j--) {
				if (field[i][j]!=0) {
					row += "[" + field[i][j] + "]";
				}else {
					row += "[ ]";
				}
			}
			System.out.println(row);
			row = "";
		}
	}
}

/**
 * Learner class, this class implements LSTDQ and LSPI such that we get a ending
 * weight vector that can play Tetris well
 */
class Learner {
	/**
	* Running LSTDQ to update, where we can find the weight vector using
	* w = inv(A) * b
	* A is a kxk matrix of the following form, at each step:
	* A := A + currentFeatures*transpose(currentFeatures - DISCOUNT*successorFeatures)
	* 
	* And b is a kx1 vector of the following form, at each step:
	* b := b + currentFeatures*currentReward
	* 
	* where currentFeatures, successorFeatures are kx1 vectors representing the feature array
	*/
	public static double DISCOUNT = 0.7;
	public static double LOST_REWARD = -100000;
	public static double ERROR = 0.0005;

	public double[][] A = new double[FeatureFunction.NUM_FEATURES][FeatureFunction.NUM_FEATURES];
	public double[][] b = new double[FeatureFunction.NUM_FEATURES][1];

	private double[] prevWeights;
	private double[] weights;
	
	public void LSTDQ(int sampleSize) {
		/**
		 * TODO: Implement me
		 * main Loop for LSTDQ, general pseudocode:
		 * 
		 * 	for each state s generated (up to a certain limit):
		 * 		make the best move that maximises the value function, using the weights,
		 * 		the resultant state will be s'
		 * 		Update the A and b by calling LSTDQUpdate()
		 * 		try to update the weights if A^-1 can be found using extractAndUpdateWeights()
		 */
	}

	public double[] LSPI() {
		/**
		 * TODO: Implement me
		 * 
		 * 	main Loop for LSPI, general pseudocode:
		 * 
		 * 	while (weights - prevWeights) >= ERROR, do:
		 * 		Run LSTDQ(LIMIT)
		 */
		return new double[1];
	}

	// Features coming in are of the form of 1xk vectors (i.e. row vectors)
	public void LSTDQUpdate(double[][] currentFeature, double[][] successorFeature, NState successorState) {
		double[][] difference = Matrix.matrixSum(
			currentFeature,
			Matrix.matrixScalarMultiply(successorFeature, -1*DISCOUNT), false, false);
		double[][] stepA = Matrix.matrixMultiply(currentFeature, difference, true, false);
		A = Matrix.matrixSum(A, stepA, false, false);
		double reward = getReward(successorState);
		b = Matrix.matrixSum(b, Matrix.matrixScalarMultiply(currentFeature, reward), false, true);
	}

	private static double getReward(NState s) {
		return (s.hasLost()) ? LOST_REWARD : s.getRowsCleared() - s.getOState().getRowsCleared();
	}

	/**
	 * Updates the weights, assuming if an inverse for A can be found
	 */
	private void extractAndUpdateWeights() {
		// this should be a Kx1 column vector, where K is the number of features
		double[][] weightVector = Matrix.solveMatrix(A, b);
		if(weightVector == null) {
			return;
		}
		this.prevWeights = this.weights;
		this.weights = colVectorToArray(weightVector);
	}

	// Column vectors are vectors of mX1
	private static double[] colVectorToArray(double[][] vector) {
		double[] newArray = new double[vector.length];
		for(int i = 0; i < vector.length; i++) {
			newArray[i] = vector[i][0];
		}
		return newArray;
	}
}

/**
 * Utility class for matrix operations
 */
class Matrix {
	public static void main(String[] args) {
		// 2x4 matrix
		double[][] testMatrix1 = {
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		};
		// 4x2 matrix
		double[][] testMatrix2 = {
			{9, 10,},
			{11, 12,},
			{13, 14,},
			{15, 16,},
		};
		// 3x4 matrix, with duplicate rows
		double[][] testMatrix3 = {
			{3, 4, 5, 6},
			{1.5, 2, 2.5, 3 },
			{4, 5, 6, 7},
		};

		// 3x4 matrix, with empty column
		double[][] testMatrix4 = {
			{3, 4, 0, 6},
			{1.5, 2, 0, 3 },
			{4, 5, 0, 7},
		};

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
			if(startColumn >= augmentedMatrix[row].length) {
				return;
			}

			//if the number isn't one, reduce to one
			if (augmentedMatrix[row][startColumn] != 1.0) {
				double divisor = augmentedMatrix[row][startColumn];
				scalarRowMultiply(augmentedMatrix, row, 1/divisor);
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
		if(matrix.length != matrix[0].length) {
			return false;
		}
		boolean isIdent = true;
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[0].length; j++) {
				if((i == j && matrix[i][j] != 1.0) || (i != j && matrix[i][j] != 0.0)) {
					isIdent = false;
					break;
				}
			}
			if(!isIdent) {
				break;
			}
		}
		return isIdent;
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
		for(int j = 0; j < matrix[row1].length; j++) {
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

	public static double[][] getSolutionMatrix(double[][] augmentedMatrix,
		int numOfColsCoefficientMatrix) {
		int numRows = augmentedMatrix.length;
		// number of columns that solution matrix has is:
		// total number of columns - number of columns that coefficient matrix has
		int numCols = augmentedMatrix[0].length - numOfColsCoefficientMatrix;
		double[][] result = new double[numRows][numCols];
		for(int i = 0; i < numRows; i++){
			System.arraycopy(augmentedMatrix[i], numOfColsCoefficientMatrix, result[i], 0, numCols);
		}
		return result;
	}

	public static double[][] getCoefficientMatrix(double[][] augmentedMatrix,
		int numOfColsCoefficientMatrix) {
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
	public static double[][] matrixMultiply(double[][] matrix1, double[][] matrix2,
			boolean transposeMatrix1, boolean transposeMatrix2) {
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

	public static double[][] matrixSum(double[][] matrix1, double[][] matrix2,
			boolean transposeMatrix1, boolean transposeMatrix2) {
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
				switch(op) {
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
