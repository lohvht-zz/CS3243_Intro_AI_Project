
public class PlayerSkeleton {

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		
		return 0;
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

// Helper Class to calculate features, with a given state/action
class FeatureFunction {
	public static final int NUM_FEATURES = 8;
	// Indexes of the feature array values
	public static final int INDEX_MAX_COL_HEIGHT = 1;
	public static final int INDEX_NUM_HOLES = 2;
	public static final int INDEX_LANDING_HEIGHT = 3;
	public static final int INDEX_NUM_ROWS_REMOVED = 4;
	public static final int INDEX_AV_DIFF_COL_HEIGHT = 5;
	public static final int INDEX_AV_COL_HEIGHT = 6;
	public static final int INDEX_COL_TRANSITION = 7;
	public static final int INDEX_ROW_TRANSITION = 8;

	/**
	 * Helper function that computes all the features and returns it as a vector
	 * @param nextState This is the next game state (NOTE victor@24/02/18: I
	 * will implement a class that will extend State, which will add a helper
	 * method to get the action as well, for now just go on an assumption that we pass in the action)
	 * @return an array representing the vector of calculated feature values
	 */
	public double[] getFeatureValues(State nextState) {
		double[] features = new double[NUM_FEATURES+1];
		// A Bias to the linear vector, may help in learning
		features[0] = 1;
		// The rest of the feature vector
		features[INDEX_MAX_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_NUM_HOLES] = getTotalNumberofHoles(nextState);
		features[INDEX_LANDING_HEIGHT] = getLandingHeight(nextState);
		// Change function parameter when the extended State class is implemented
		features[INDEX_NUM_ROWS_REMOVED] = getRowsRemoved(nextState, nextState);
		features[INDEX_AV_DIFF_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_AV_COL_HEIGHT] = getMaxColHeight(nextState);
		features[INDEX_COL_TRANSITION] = getMaxColHeight(nextState);
		features[INDEX_ROW_TRANSITION] = getMaxColHeight(nextState);
		return features;
	}

	/**
	 * The maximum column height of the board
	 */
	public double getMaxColHeight(State state) {
		// TODO: Implement Me!
		return -1;
	}

	/**
	 * Total number of holes in the wall, the number of empty cells that has at
	 * least one filled cell above it in the same column
	 */
	public double getTotalNumberofHoles(State state) {
		// TODO: Implement Me!
		return -1;
	}

	/**
	 * Height where the piece is put (= the height of the column + (the height of
	 * the piece / 2))
	 */
	public double getLandingHeight(State state) {
		// TODO: Implement Me!
		return -1;
	}

	public double getRowsRemoved(State nextState, State prevState) {
		// TODO: implement me!
		return -1;
	}

	/**
	 * The average of all absolute differences of all column heights
	 */
	public double getAverageDifferenceColumnHeight(State state) {
		// TODO: implement me!
		return -1;
	}

	/**
	 * The average column height
	 */
	public double getAverageColumnHeight(State state) {
		// TODO: implement me!
		return -1;
	}

	/**
	 * The total number of row transitions. Row transitions are when an empty cell
	 * is adjacent to a filled cell on the same row.
	 */
	public double getRowTransitions(State state) {
		// TODO: implement me!
		return -1;
	}

	public double getColumnTransitions(State state) {
		// TODO: implement me!
		return -1;
	}
}

class NextState extends State {

}
