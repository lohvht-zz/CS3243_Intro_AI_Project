import java.util.ArrayList;
import java.util.Random;
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
        for (int i = 0; i < moves; i++) {
            if (!state.hasLost()) {
                //save existing state
                prevState.copy(state);
                //set lowest height to postive inf
                double lowestHeight = Double.MAX_VALUE;
                //create a temp state to check moves, allows copying of prev state to check all moves
                NState tempState = new NState();
                //store all best moves here
                ArrayList<Integer> bestMoves = new ArrayList<Integer>();
                for (int j = 0; j < prevState.legalMoves().length; j++) {
                    tempState.copy(prevState);
                    tempState.makeMove(j);
                    //need to know how to use feature function
                    if (ff.getColumnFeatures(tempState)[0] < lowestHeight) {
                        bestMoves.clear();
                        bestMoves.add(j);
                        lowestHeight = ff.getColumnFeatures(tempState)[0];
                    } else if (ff.getColumnFeatures(tempState)[0] == lowestHeight) {
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
        for (int i = 0; i < moves; i++) {
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
        for (int i = State.ROWS - 1; i >= 0; i--) {
            String row = "";
            for (int j = State.COLS - 1; j >= 0; j--) {
                if (field[i][j] != 0) {
                    row += "[" + field[i][j] + "]";
                } else {
                    row += "[ ]";
                }
            }
            System.out.println(row);
            row = "";
        }
        System.out.println("Piece " + s.getNextPiece());
    }
}