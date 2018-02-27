import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.stream.IntStream;

public class PlayerSkeleton {
	FeatureFunction f = new FeatureFunction();


	double[] weights = {
		0.375436867970272,
		0.7925442710411554,
		0.40162002789633044,
		0.3312561769239417,
		0.8661011224368202,
		0.6498664140640098,
		0.8090459288058522,
		0.007873808594357268,
	};

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		int bestMove = 0;
		double maxValue = -Double.MAX_VALUE;
		double currentValue = -Double.MAX_VALUE;

		NState nextState = new NState();

		for(int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			currentValue = f.calculateValue(nextState, weights);
			// System.out.printf("state turn %d, next state turn %d, move %d, value %f\n", s.getTurnNumber(), nextState.getTurnNumber(), move, currentValue);
			if(currentValue > maxValue) {
				maxValue = currentValue;
				bestMove = move;
			}
		}
		System.out.printf("Best Move: %d\n", bestMove);
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
	public static final int INDEX_LANDING_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_ROWS_REMOVED = countFeatures++;
	public static final int INDEX_AV_DIFF_COL_HEIGHT = countFeatures++;
	public static final int INDEX_AV_COL_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_HOLES = countFeatures++;
	public static final int INDEX_COL_TRANSITION = countFeatures++;
	public static final int INDEX_ROW_TRANSITION = countFeatures++;

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
