import java.util.Arrays;

public class PlayerSkeleton {
	FeatureFunction f = new FeatureFunction();
	double[] weights = {1, 1, 1, 1, 1, 1, 1, 1, 1}; // weight vector of length NUM_FEATURES + 1
	NState nextState = new NState();

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		int bestMove = 0;
		double maxValue = Double.NEGATIVE_INFINITY;
		double currentValue = Double.NEGATIVE_INFINITY;
		for(int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			currentValue = f.calculateValue(nextState, weights);
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
				Thread.sleep(300);
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
	public static final int INDEX_NUM_HOLES = countFeatures++;
	public static final int INDEX_LANDING_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_ROWS_REMOVED = countFeatures++;
	public static final int INDEX_AV_DIFF_COL_HEIGHT = countFeatures++;
	public static final int INDEX_AV_COL_HEIGHT = countFeatures++;
	public static final int INDEX_COL_TRANSITION = countFeatures++;
	public static final int INDEX_ROW_TRANSITION = countFeatures++;

	public static final int NUM_FEATURES = countFeatures;

	/**
	 * Helper function that computes all the features and returns it as a vector
	 * @param nextState This is the next game state (NOTE victor@24/02/18: I
	 * will implement a class that will extend State, which will add a helper
	 * method to get the action as well, for now just go on an assumption that we pass in the action)
	 * @return an array representing the vector of calculated feature values
	 */
	public double[] getFeatureValues(NState nextState) {
		double[] features = new double[NUM_FEATURES+1];
		// A Bias to the linear vector, may help in learning
		features[0] = 1;
		// The rest of the feature vector
		features[INDEX_MAX_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_NUM_HOLES] = getTotalNumberofHoles(nextState);
		features[INDEX_LANDING_HEIGHT] = getLandingHeight(nextState);
		features[INDEX_NUM_ROWS_REMOVED] = getRowsRemoved(nextState);
		features[INDEX_AV_DIFF_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_AV_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_COL_TRANSITION] = getRowTransitions(nextState);
		features[INDEX_ROW_TRANSITION] = getColumnTransitions(nextState);
		return features;
	}

	/**
	 * The maximum column height of the board
	 */
	public double getMaxColHeight(NState state) {
		// TODO: Implement Me!
		return state.COLS;
	}

	/**
	 * Total number of holes in the wall, the number of empty cells that has at
	 * least one filled cell above it in the same column
	 */
	public double getTotalNumberofHoles(NState state) {
		// TODO: Implement Me!
		int count = 0;
		int[][] field = state.getField();
		//-2 for the illegal and top legal row
		for (int i=0; i<(state.ROWS-2);i++) {
			for (int j=0; j<state.COLS; j++) {
				if (field[i][j]<=0 && field[i+1][j]>0) {
					count += 1;
				}
			}
		}
		return count;
	}

	/**
	 * Height where the piece is put (= the height of the column + (the height of
	 * the piece / 2))
	 */
	public double getLandingHeight(NState state) {
		// TODO: Implement Me!
		return -1;
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
		return -1;
	}

	/**
	 * The average column height
	 */
	public double getAverageColumnHeight(NState state) {
		// TODO: implement me!
		return -1;
	}

	/**
	 * The total number of row transitions. Row transitions are when an empty cell
	 * is adjacent to a filled cell on the same row.
	 */
	public double getRowTransitions(NState state) {
		// TODO: implement me!
		return -1;
	}

	public double getColumnTransitions(NState state) {
		// TODO: implement me!
		return -1;
	}

	public double calculateValue(NState state, double[] weight) {
		double values = 0;
		double[] featureValues = getFeatureValues(state);
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
	//private variables from State
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
		super();
	};

	public NState(State state) {
		this.copy(state);
	}

	public void copy(State state) {
		// Copy all relevant private members to this new state
		this.turn = state.getTurnNumber();
		this.cleared = state.getRowsCleared();
		this.setField(state.getField());
		this.setTop(state.getTop());
		// replace relevant protected/public variables
		this.lost = state.lost;
		this.nextPiece = state.getNextPiece();
		// currentAction set to -1 (not made a move yet)
		currentAction = -1;
		// Preserve the original state
		this.oState = state;
	}

	private void setField(int[][] field) {
		for(int i=0; i<ROWS; i++) {
			for(int j=0; j<COLS; j++) {
				this.field[i][j] = field[i][j];
			}
		}
	}

	private void setTop(int[] top) {
		for(int i=0; i<COLS; i++) {
			this.top[i] = top[i];
		}
	}

	public State getOState() {
		return this.oState;
	}

	public int getCurrentAction() {
		return this.currentAction;
	}

	public int getRowsCleared() {
		return this.cleared;
	}

	//make a move based on the move index - its order in the legalMoves list
	public void makeMove(int move) {
		currentAction = move;
		makeMove(legalMoves[nextPiece][move]);
	}

	//returns false if you lose - true otherwise
	public boolean makeMove(int orient, int slot) {
		turn++;
		//height if the first column makes contact
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
