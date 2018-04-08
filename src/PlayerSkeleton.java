import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.List;
import java.util.Scanner;

public class PlayerSkeleton {
	FeatureFunction f = new FeatureFunction();
	// Out of 600 games (trained with board of max 9 rows: "Converged"
	// at generation 464)
	// 25th Percentile: 4,000,932.25
	// Median: 9,323,531.5
	// 75th Percentile: 18,056,433.5
	// Mean: 12,872,842.7867
	// Highest: 70,106,597
	// Lowest: 31,606
	// { 
	// 	0.27108678297658184, // INDEX_NUM_ROWS_REMOVED
	// 	0.052210127461541356, // INDEX_MAX_HEIGHT
	// 	-0.01175971153359219, // INDEX_AV_HEIGHT
	// 	-9.362791850485086E-4, // INDEX_AV_DIFF_HEIGHT
	// 	-0.22840826350913523, // INDEX_LANDING_HEIGHT
	// 	-0.3074418797607427, // INDEX_NUM_HOLES
	// 	-0.12518629866820255, // INDEX_COL_TRANSITION
	// 	-0.17175759824362818, // INDEX_ROW_TRANSITION
	// 	-0.7871385684504404, // INDEX_COVERED_GAPS
	// 	-0.15907062512458905, // INDEX_TOTAL_WELL_DEPTH
	// 	0.29307537483267665 // INDEX_HAS_LOST
	// };
	static final double[] defaultWeights =
		// NEW set of weights row 13 trainer (trained with board of max 13 rows:
		// Not converged yet at 132!:
		// Out of 600 games
		// 25th Percentile: 6,307,657.5
		// Median: 13,655,622
		// 75th Percentile: 25,716,898.5
		// Mean: 19,793,958.20
		// Highest: 216,319,742
		// Lowest: 5125
		{
			-0.10994115458466136,
			-0.1154697834187254,
			-0.04390525258236673,
			0.017912908135268947,
			-0.17018693844059846,
			-0.3044476707923254,
			-0.38617473506172584,
			-0.22806177833393343,
			-0.7696058904564755,
			-0.19377750577164388,
			0.13672271498097804
		};

	double[] weights;

	NState nextState = new NState();
	double[] featureValues = null;

	// Use the default weight
	PlayerSkeleton() {
		weights = new double[defaultWeights.length];
		System.arraycopy(defaultWeights, 0, weights, 0, defaultWeights.length);
	}

	PlayerSkeleton(double[] _weights) {
		weights = new double[_weights.length];
		System.arraycopy(_weights, 0, weights, 0, _weights.length);
	}

	/**
	 * Plays the game until the end with the given weight vector
	 * @param weights 	The weight vector passed in, if weight vector is null,
	 * 					use the hardedcoded default weight
	 */
	public static double playGame(double[] weights) {
		State s = new State();
		PlayerSkeleton p = (weights == null) ? new PlayerSkeleton() : new PlayerSkeleton(weights);
		while (!s.hasLost()) {
			s.makeMove(p.pickMove(s, s.legalMoves()));
		}
		return s.getRowsCleared();
	}

	/**
	 * Plays the game until max moves reached or has lost
	 * @param weights 	The weight vector passed in, if weight vector is null,
	 * 					use the hardedcoded default weight
	 * @param maxMoves	The maximum amount of moves allowed
	 */
	public static double playGame(double[] weights, int maxMoves) {
		State s = new State();
		PlayerSkeleton p = (weights == null) ? new PlayerSkeleton() : new PlayerSkeleton(weights);
		while (!s.hasLost() && s.getTurnNumber() < maxMoves) {
			s.makeMove(p.pickMove(s, s.legalMoves()));
		}
		return s.getRowsCleared();
	}

	public static double runGames(int numGames, double[] weights, int maxMoves) {
		ExecutorService executor = Executors.newFixedThreadPool(numGames);
		Callable<Double> runGame = (maxMoves <= 0)
		? () -> {
			return playGame(weights);
		}
		: () -> {
			return playGame(weights, maxMoves);
		};
		List<Callable<Double>> gamesToRun = new ArrayList<>();

		for (int i = 0; i < numGames; i++) {
			gamesToRun.add(runGame);
		}
		
		double sum = 0;
		double score = 0;
		double highScore = Double.MIN_VALUE;
		double lowScore = Double.MAX_VALUE;
		try {
			List<Future<Double>> results = executor.invokeAll(gamesToRun);
			for(Future<Double> result: results) {
				score = result.get();
				sum += score;
				System.out.println(score);
				highScore = Math.max(highScore, score);
				lowScore = Math.min(lowScore, score);
			}
		} catch (InterruptedException ie) {
			System.out.println("Interrupted games!");
			executor.shutdown();
		} catch (ExecutionException ex) {
			ex.printStackTrace();
		}
		executor.shutdown();
		double averageScore = sum / numGames;
		System.out.println("High score: "+highScore+" Low Score: "+lowScore);
		return averageScore;
	}

	public static double runGamesSequential(int numGames, double[] weights, int numMoves) {
		double sum = 0;
		double score = 0;
		double highScore = Double.MIN_VALUE;
		double lowScore = Double.MAX_VALUE;

		for (int i = 0; i < numGames; i++) {
			score = playGame(weights);
			sum += score;
			System.out.println(score);
			highScore = Math.max(highScore, score);
			lowScore = Math.min(lowScore, score);
		}
		double averageScore = sum / numGames;
		System.out.println("High score: " + highScore + " Low Score: " + lowScore);
		return averageScore;
	}

	/**
	 * Wrapper to play take input and play games
	 * reads from std.in
	 * argument1: numGames ==> number of games to play
	 * argument2: numMoves ==> max number of moves per game
	 * argument3: weightString ==> if "null" use default hardcoded weights,
	 * 				else if its a string of doubles, use that as weight array
	 */
	public static void takeInputsAndRunGames() {
		Scanner sc = new Scanner(System.in);
		int numGames = sc.nextInt();
		int numMoves = sc.nextInt();
		String weightString = sc.nextLine();
		double[] weights;
		if(!weightString.equals("null")) {
			String[] weightstringArray = weightString.split(",");
			weights = new double[FeatureFunction.NUM_FEATURES];
			for(int i = 0; i < weightstringArray.length; i++) {
				weights[i] = Double.parseDouble(weightstringArray[i].trim());
			}
		} else {
			weights = null;
		}
		if(weights != null) {
			System.out.println("Playing Games for this set of weights: ");
			System.out.println(Arrays.toString(weights));
		} else {
			System.out.println("Playing with hardcoded weights!");
		}

		long startTime = System.currentTimeMillis();
		double score = runGames(numGames, weights, numMoves);
		long totalTImeElapsed = (System.currentTimeMillis() - startTime)/1000;
		System.out.println("Total time taken: "+ totalTImeElapsed+ " Average Score was: "+ score);
		sc.close();
	}

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		int bestMove = 0;
		double maxValue = -Double.MAX_VALUE;
		double currentValue = -Double.MAX_VALUE;

		for (int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			featureValues = f.getFeatureValues(nextState);
			currentValue = f.calculateValue(featureValues, weights);
			if (currentValue > maxValue) {
				maxValue = currentValue;
				bestMove = move;
			}
		}
		return bestMove;
	}
	
	public static void main(String[] args) {
		// Uncomment this and comment the original code below to let scanner
		// take inputs
		// takeInputsAndRunGames();

		// Original Code provided
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while (!s.hasLost()) {
			s.makeMove(p.pickMove(s, s.legalMoves()));
			s.draw();
			s.drawNext(0,0);
			try {
				Thread.sleep(300);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed " + s.getRowsCleared() + " rows.");
	}
}

/**
 * Features used
 */
class FeatureFunction {
	private static int countFeatures = 0;
	// Indexes of the feature array values
	public static final int INDEX_NUM_ROWS_REMOVED = countFeatures++;
	public static final int INDEX_MAX_HEIGHT = countFeatures++;
	// public static final int INDEX_AV_HEIGHT = countFeatures++;
	// public static final int INDEX_AV_DIFF_HEIGHT = countFeatures++;
	public static final int INDEX_TOTAL_HEIGHT = countFeatures++;
	public static final int INDEX_TOTAL_DIFF_HEIGHT = countFeatures++;
	public static final int INDEX_LANDING_HEIGHT = countFeatures++;
	public static final int INDEX_NUM_HOLES = countFeatures++;
	public static final int INDEX_COL_TRANSITION = countFeatures++;
	public static final int INDEX_ROW_TRANSITION = countFeatures++;
	public static final int INDEX_COVERED_GAPS = countFeatures++;
	public static final int INDEX_TOTAL_WELL_DEPTH = countFeatures++;
	public static final int INDEX_HAS_LOST = countFeatures++;

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
		features[INDEX_NUM_ROWS_REMOVED] = getRowsRemoved(nextState);
		features[INDEX_MAX_HEIGHT] = columnFeatures[0];
		// features[INDEX_AV_HEIGHT] = columnFeatures[1];
		// features[INDEX_AV_DIFF_HEIGHT] = columnFeatures[2];
		features[INDEX_TOTAL_HEIGHT] = columnFeatures[1];
		features[INDEX_TOTAL_DIFF_HEIGHT] = columnFeatures[2];
		features[INDEX_LANDING_HEIGHT] = getLandingHeight(nextState);
		double[] holesTransitions = getHolesTransitionsCoveredGaps(nextState);
		features[INDEX_NUM_HOLES] = holesTransitions[0];
		features[INDEX_COL_TRANSITION] = holesTransitions[1];
		features[INDEX_ROW_TRANSITION] = holesTransitions[2];
		features[INDEX_COVERED_GAPS] = holesTransitions[3];
		features[INDEX_TOTAL_WELL_DEPTH] = getWellDepths(nextState);
		features[INDEX_HAS_LOST] = getHasLost(nextState);
		return features;
	}

	/**
	* MAX HEIGHT:
	* 		Height of the tallest column
	* TOTAL HEIGHT:
	* 		Total height of all columns
	* TOTAL DIFF HEIGHT:
	* 		Total height difference of all adjacent columns
	*/
	public double[] getColumnFeatures(NState state) {
		int[] top = state.getTop();
		int maxHeight = 0;
		int totalHeight = 0;
		int totalDiffHeight = 0;

		for (int i = 0; i < State.COLS; i++) {
			totalHeight += top[i];
			totalDiffHeight += (i > 0) ? Math.abs(top[i] - top[i - 1]) : 0;
			maxHeight = Math.max(maxHeight, top[i]);
		}
		double[] result = { maxHeight, (double) totalHeight, (double) totalDiffHeight };
		return result;
	}

	/**
	 * Height where the piece is put (= the height of the column + (the height of
	 * the piece / 2))
	 */
	public double getLandingHeight(NState state) {
		// Current action is actually the action taken to derive NState
		int move = state.getCurrentAction();

		State originalState = state.getOState();
		int piece = originalState.getNextPiece();
		int[][] moves = originalState.legalMoves();
		int orient = moves[move][State.ORIENT];
		int slot = moves[move][State.SLOT];
		int[] top = originalState.getTop();

		int pieceWidth = State.getpWidth()[piece][orient];
		int[][][] pTop = State.getpTop();

		double maxLandingHeight = -Double.MAX_VALUE;

		for (int c = 0; c < pieceWidth; c++) {
			double currentLandingHeight = top[slot + c] + (double)pTop[piece][orient][c] / 2.0;
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
				totalSum += (wellDepth * (double)(wellDepth + 1)) / 2.0;
			}
		}
		return totalSum;
	}

	public double getHasLost(NState state) {
		return (state.hasLost()) ? -10000 : 100;
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

	public int[][] legalMoves() {
		return legalMoves[this.nextPiece];
	}

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
