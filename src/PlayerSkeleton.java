import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.List;

public class PlayerSkeleton {
	FeatureFunction f = new FeatureFunction();
	double[] weights =
	// Average score over 100 games ==> 847094.92 rows cleared
	// Highest cleared ===> 3,727,291
	{
		0.00134246,	// INDEX_NUM_ROWS_REMOVED
		-0.01414993, // INDEX_MAX_HEIGHT
		-0.00659672, // INDEX_AV_HEIGHT
		0.00140868, // INDEX_AV_DIFF_HEIGHT
		-0.02396361, // INDEX_LANDING_HEIGHT
		-0.03055654, // INDEX_NUM_HOLES
		-0.06026152, // INDEX_COL_TRANSITION
		-0.02105507, // INDEX_ROW_TRANSITION
		-0.0340038, // INDEX_COVERED_GAPS
		-0.0117935, // INDEX_TOTAL_WELL_DEPTH
		1.00, // INDEX_HAS_LOST, after implementing this, score went up by a significant amount
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

	public static double playGame() {
		State s = new State();
		PlayerSkeleton p = new PlayerSkeleton();
		while (!s.hasLost()) {
			s.makeMove(p.pickMove(s, s.legalMoves()));
		}
		return s.getRowsCleared();
	}

	public static void runGames(int numGames) {
		ExecutorService executor = Executors.newFixedThreadPool(20);
		Callable<Double> runGame = () -> {
			return playGame();
		};
		List<Callable<Double>> gamesToRun = new ArrayList<>();

		for (int i = 0; i < numGames; i++) {
			gamesToRun.add(runGame);
		}
		
		double sum = 0;
		double score = 0;
		try {
			List<Future<Double>> results = executor.invokeAll(gamesToRun);
			for(Future<Double> result: results) {
				score = result.get();
				sum += score;
				System.out.println(score);
			}
		} catch (InterruptedException ie) {
			System.out.println("Interrupted games!");
		} catch (ExecutionException ex) {
			ex.printStackTrace();
		}
		double averageScore = sum / numGames;
		System.out.println("You have completed " + averageScore + " rows on the average.");
	}
	
	public static void main(String[] args) {
		// runGames(100);
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while (!s.hasLost()) {
			s.makeMove(p.pickMove(s, s.legalMoves()));
			s.draw();
			s.drawNext(0,0);
			try {
				// Thread.sleep(300); // Uncomment this to revert to normal sleep 300
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed " + s.getRowsCleared() + " rows.");
	}
}

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
	* 		Average height of all columns
	* TOTAL DIFF HEIGHT:
	* 		Average height difference of all adjacent columns
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
				// System.out.println("Game Lost, printing Previous State:");
				// printState(prevState);
				return prevState;
			}
		}
		prevState.copy(state);
		// System.out.println("Moves Complete, printing Final State:");
		// printState(prevState);
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
				// System.out.println("Game Lost, printing Previous State:");
				// printState(prevState);
				return prevState;
			}
		}
		prevState.copy(state);
		// System.out.println("Moves Complete, printing Final State:");
		// printState(prevState);
		return prevState;
	}
	
	public static void printState(State s) {
		int[][] field = s.getField();
		for (int i=State.ROWS-1; i>=0;i--) {
			String row = "";
			for (int j=State.COLS-1; j>=0; j--) {
				if (field[i][j]!=0) {
					row += "[" + field[i][j] + "]";
				}else {
					row += "[ ]";
				}
			}
			System.out.println(row);
			row = "";
		}
		System.out.println("Piece "+s.getNextPiece());
	}
}

/**
 * Learner class, this class implements LSTDQ and LSPI such that we get a ending
 * weight vector that can play Tetris well
 */
class Learner {
	/**
	* Running the optimised version of LSTDQ, LSTDQ-OPT to update, where we can
	* find the weight vector using
	* w = B*b
	* B is a kxk matrix of the following form, at each step:
	* B := B - B*currentFeatures*transpose(difference)*B/ (1 + transpose(difference)*B*currentFeatures)
	* 
	* difference is (currentFeatures - DISCOUNT*successorFeatures), which is a kx1 vector
	*
	* And b is a kx1 vector of the following form, at each step:
	* b := b + currentFeatures*currentReward
	* 
	* where currentFeatures, successorFeatures are kx1 vectors representing the feature array
	*/
	private static final double DISCOUNT = 0.9;
	private static final double STOPPING_CRITERION = 0.005;
	private static final double _P = 1.0/7.0;

	private double[][] B;
	private double[][] b;

	public double[] prevWeights;
	public double[] weights;
	private Random rand = new Random(System.nanoTime());

	private FeatureFunction f = new FeatureFunction();
	private NState _state = new NState();
	private NState _statePrime = new NState();
	

	// returns the weight vector resulting from the LSTDQ update
	private double[] LSTDQ(int sampleSize) {
		int sampleNumber = 0;
		B = new double[FeatureFunction.NUM_FEATURES][FeatureFunction.NUM_FEATURES];
		b = new double[FeatureFunction.NUM_FEATURES][1];
		for (int i = 0; i < FeatureFunction.NUM_FEATURES; i++) {
			B[i][i] = 0.001;
		}

		double percentageOfPseudoGoodStates = 0.8;
		while(sampleNumber < sampleSize) {
			int generatingMoves = rand.nextInt(50) + 1;
			NState state = (rand.nextDouble() <= percentageOfPseudoGoodStates)
				? StateGenerator.generatePseudoBestState(generatingMoves)
				: StateGenerator.generateState(generatingMoves);

			int[][] legalMoves = state.legalMoves();
			for(int a = 0; a < legalMoves.length; a++) {
				double[][] currentFeatures = new double[1][FeatureFunction.NUM_FEATURES];
				double[][] successorFeatures = new double[1][FeatureFunction.NUM_FEATURES];

				_state.copy(state);
				_state.makeMove(a);
				// If the state after making a move has lost, ignore and move to the next move
				if(_state.hasLost()) {
					continue;
				}
				currentFeatures[0] = f.getFeatureValues(_state);
				double[][] phiPrime = new double[1][FeatureFunction.NUM_FEATURES];;
				double reward = 0;
				for(int piece = 0; piece < State.N_PIECES; piece++) {
					_state.setNextPiece(piece);
					_statePrime.copy(_state);
					// This is the best move from the current policy
					int bestMove = pickBestMove(_statePrime, weights);
					_statePrime.makeMove(bestMove);
					phiPrime[0] = f.getFeatureValues(_statePrime);
					successorFeatures = Matrix.matrixSum(successorFeatures, phiPrime, false, false);
					reward += rewardFunction(_statePrime);
				}
				// multiply the probabilty inside the sum of rewards for s' and pi(s')
				// and also the successorFeatures
				reward *= _P;
				successorFeatures = Matrix.matrixScalarMultiply(successorFeatures, _P);
				LSTDQ_OPTUpdate(currentFeatures, successorFeatures, reward);
			}
			sampleNumber++;
		}
		return extractWeightVectorLSTDQ_OPT();
	}

	/**
	 * Plays a game while running the LSTDQ Algorithm
	 */
	private double[] LSTDQPlay() {
		B = new double[FeatureFunction.NUM_FEATURES][FeatureFunction.NUM_FEATURES];
		b = new double[FeatureFunction.NUM_FEATURES][1];
		for (int i = 0; i < FeatureFunction.NUM_FEATURES; i++) {
			B[i][i] = 0.001;
		}
		double[][] currentFeatures = new double[1][FeatureFunction.NUM_FEATURES];
		double[][] successorFeatures = new double[1][FeatureFunction.NUM_FEATURES];
		State s = new State();
		while (!s.hasLost()) {
			int bestMove = pickBestMove(s, weights);
			_statePrime.copy(s);
			_statePrime.makeMove(bestMove);
			successorFeatures[0] = f.getFeatureValues(_statePrime);
			double reward = rewardFunction(_statePrime);
			LSTDQ_OPTUpdate(currentFeatures, successorFeatures, reward);
			System.arraycopy(successorFeatures[0], 0, currentFeatures[0], 0, FeatureFunction.NUM_FEATURES);
			// Actually make the move to go to the next turn
			s.makeMove(bestMove);
		}
		System.out.print("Score is: "+s.getRowsCleared() + " ");
		return extractWeightVectorLSTDQ_OPT();
	}

	public int pickBestMove(State s, double[] wArray) {
		int[][] legalMoves = s.legalMoves();

		int bestMove = 0;
		NState nextState = new NState();
		double[] features;
		double maxValue = -Double.MAX_VALUE;
		double value;

		for(int move = 0; move < legalMoves.length; move++) {
			nextState.copy(s);
			nextState.makeMove(move);
			features = f.getFeatureValues(nextState);
			value = f.calculateValue(features, wArray);
			if(value > maxValue) {
				maxValue = value;
				bestMove = move;
			}
		}
		return bestMove;
	}

	double minDifference = Double.MAX_VALUE;
	public double[] LSPI(boolean isPlayLearning, int limit, int sampleSize, String filename, double[] _startWeights) {
		weights = new double[FeatureFunction.NUM_FEATURES];
		prevWeights = new double[FeatureFunction.NUM_FEATURES];

		if(_startWeights == null) {
			// Let weights start from 0
			// for (int i = 0; i < weights.length; i++) {
			// 	weights[i] = (rand.nextBoolean()) ? rand.nextDouble() : -1 * rand.nextDouble();
			// }
		} else {
			System.arraycopy(_startWeights, 0, weights, 0, _startWeights.length);
		}

		int count = 0;
		double difference;
		
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(String.format(filename)))) {
			do {
				System.arraycopy(weights, 0, prevWeights, 0, weights.length);
				if (isPlayLearning) {
					// plays a game, and updates
					weights = LSTDQPlay();
				} else {
					weights = LSTDQ(sampleSize);
				}
				difference = difference(prevWeights, weights);
				minDifference = Math.min(difference, minDifference);
				System.out.printf("The difference is: %f, of count %d.\n", difference, count);
				bw.write("Count: "+count+" ");
				bw.write(weightsToString());
				bw.newLine();
				bw.flush();
				count++;
				if (count >= limit) {
					break;
				}
			} while (difference >= STOPPING_CRITERION);

			bw.write("RESULT WEIGHTS AT: ");
			bw.write(weightsToString());
			bw.newLine();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return weights;
	}

	private String weightsToString() {
		StringBuilder weightsLine = new StringBuilder();
		weightsLine.append("{ ");
		for (int i = 0; i < weights.length; i++) {
			// weightsLine.append(df.format(weights[i]));
			weightsLine.append(weights[i]);
			if(i!=weights.length-1) {
				weightsLine.append(", ");
			}
		}
		weightsLine.append(" }");
		return weightsLine.toString();
	}

	// Features coming in are of the form of 1xk vectors (i.e. row vectors), do transpose them if using as column
	private void LSTDQ_OPTUpdate(double[][] currentFeature, double[][] successorFeature, double reward) {
		double[][] featureDifference = Matrix.matrixSum(
			currentFeature,
			Matrix.matrixScalarMultiply(successorFeature, -1.0*DISCOUNT),
			false, false);
		double[][] featureDifferenceMultB = Matrix.matrixMultiply(
			featureDifference,
			B,
			false, false);
		double[][] _BMultCurrentFeatures = Matrix.matrixMultiply(B, currentFeature, false, true);
		double[][] stepB = Matrix.matrixMultiply(_BMultCurrentFeatures, featureDifferenceMultB, false, false);
		double denominator = 1 + Matrix.matrixMultiply(
			featureDifferenceMultB,
			currentFeature,
			false, true)[0][0];
		stepB = Matrix.matrixScalarMultiply(stepB, -1/denominator);
		B = Matrix.matrixSum(B, stepB, false, false);
		b = Matrix.matrixSum(b, Matrix.matrixScalarMultiply(currentFeature, reward), false, true);
	}

	private double[] extractWeightVectorLSTDQ_OPT() {
		double[][] weightVector = Matrix.matrixMultiply(B, b, false, false);
		// Matrix.prettyPrintMatrix(B);
		// Matrix.prettyPrintMatrix(b);
		return colVectorToArray(weightVector);
	}

	private double rewardFunction(NState successorState) {
		// if(successorState.hasLost()) {
		// 	return -1000;
		// }
		return successorState.getRowsCleared() - successorState.getOState().getRowsCleared();
	}

	// Column vectors are vectors of mX1
	private static double[] colVectorToArray(double[][] vector) {
		double[] newArray = new double[vector.length];
		for(int i = 0; i < vector.length; i++) {
			newArray[i] = vector[i][0];
		}
		return newArray;
	}
	
	private static double difference(double[] prevWeights, double[] weights) {
		double squaredDifferences = 0;
		for (int i = 0; i < FeatureFunction.NUM_FEATURES; i++) {
			squaredDifferences += Math.pow(prevWeights[i] - weights[i], 2);
		}
		return Math.sqrt(squaredDifferences);
	}

	public static void main(String[] args) {
		Learner learner = new Learner();
		
		String filename = (args[0].length() != 0) ? args[0] : "weights_1.txt";
		boolean isPlayLearning = (args[1].equals("y")) ? true : false;
		int limit = Integer.MAX_VALUE;
		int sampleSize = 10000;
		try {
			limit = Integer.parseInt(args[2]);
		} catch(Exception e) {}
		try {
			sampleSize = Integer.parseInt(args[3]);
		} catch (Exception e) {}
		boolean usePredefinedWeights = false;
		try {
			usePredefinedWeights = args[4].equals("y") ? true : false;
		} catch (Exception e) {}


		double[] startingWeights =
		{
			0.00134246,	// INDEX_NUM_ROWS_REMOVED
			-0.01414993, // INDEX_MAX_HEIGHT
			-0.00659672, // INDEX_AV_HEIGHT
			0.00140868, // INDEX_AV_DIFF_HEIGHT
			-0.02396361, // INDEX_LANDING_HEIGHT
			-0.03055654, // INDEX_NUM_HOLES
			-0.06026152, // INDEX_COL_TRANSITION
			-0.02105507, // INDEX_ROW_TRANSITION
			-0.0340038, // INDEX_COVERED_GAPS
			-0.0117935, // INDEX_TOTAL_WELL_DEPTH
			1.00, // INDEX_HAS_LOST
		};
		if (!usePredefinedWeights) {
			startingWeights = null;
		}

		learner.LSPI(isPlayLearning, limit, sampleSize, filename, startingWeights);
		System.out.println("=================FINAL=====================");
		System.out.println(Arrays.toString(learner.prevWeights));
		System.out.println(Arrays.toString(learner.weights));
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

		double[][] testMatrix5 = {
			{13, 4, 10},
			{7, 8, 5},
			{11, 12, 6},
		};

		double[][] testMatrix6 = {
			{4, 5, 2, 14},
			{3, 9, 6, 21},
			{8, 10, 7, 28},
			{1, 2, 9, 5},
		};

		double[][] identityMatrix3by3 = {
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		};

		double[][] identityMatrix4by4 = {
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
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

		System.out.println("Testing Solving Matrix 1");
		prettyPrintMatrix(solveMatrix(testMatrix5, identityMatrix3by3));

		System.out.println("Testing Solving Matrix 2");
		if(solveMatrix(testMatrix6, identityMatrix4by4) == null) {
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

	public static double[][] getIdentityMatrix(int dimension){
		double[][] identityMat = new double[dimension][dimension];
		for(int i = 0; i < dimension; i++) {
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
