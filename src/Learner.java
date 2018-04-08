import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Random;
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
    private static final double _P = 1.0 / 7.0;

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
        while (sampleNumber < sampleSize) {
            int generatingMoves = rand.nextInt(50) + 1;
            NState state = (rand.nextDouble() <= percentageOfPseudoGoodStates)
                    ? StateGenerator.generatePseudoBestState(generatingMoves)
                    : StateGenerator.generateState(generatingMoves);

            int[][] legalMoves = state.legalMoves();
            for (int a = 0; a < legalMoves.length; a++) {
                double[][] currentFeatures = new double[1][FeatureFunction.NUM_FEATURES];
                double[][] successorFeatures = new double[1][FeatureFunction.NUM_FEATURES];

                _state.copy(state);
                _state.makeMove(a);
                // If the state after making a move has lost, ignore and move to the next move
                if (_state.hasLost()) {
                    continue;
                }
                currentFeatures[0] = f.getFeatureValues(_state);
                double[][] phiPrime = new double[1][FeatureFunction.NUM_FEATURES];
                ;
                double reward = 0;
                for (int piece = 0; piece < State.N_PIECES; piece++) {
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
        System.out.print("Score is: " + s.getRowsCleared() + " ");
        return extractWeightVectorLSTDQ_OPT();
    }

    public int pickBestMove(State s, double[] wArray) {
        int[][] legalMoves = s.legalMoves();

        int bestMove = 0;
        NState nextState = new NState();
        double[] features;
        double maxValue = -Double.MAX_VALUE;
        double value;

        for (int move = 0; move < legalMoves.length; move++) {
            nextState.copy(s);
            nextState.makeMove(move);
            features = f.getFeatureValues(nextState);
            value = f.calculateValue(features, wArray);
            if (value > maxValue) {
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

        if (_startWeights == null) {
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
                bw.write("Count: " + count + " ");
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
            if (i != weights.length - 1) {
                weightsLine.append(", ");
            }
        }
        weightsLine.append(" }");
        return weightsLine.toString();
    }

    // Features coming in are of the form of 1xk vectors (i.e. row vectors), do transpose them if using as column
    private void LSTDQ_OPTUpdate(double[][] currentFeature, double[][] successorFeature, double reward) {
        double[][] featureDifference = Matrix.matrixSum(currentFeature,
                Matrix.matrixScalarMultiply(successorFeature, -1.0 * DISCOUNT), false, false);
        double[][] featureDifferenceMultB = Matrix.matrixMultiply(featureDifference, B, false, false);
        double[][] _BMultCurrentFeatures = Matrix.matrixMultiply(B, currentFeature, false, true);
        double[][] stepB = Matrix.matrixMultiply(_BMultCurrentFeatures, featureDifferenceMultB, false, false);
        double denominator = 1 + Matrix.matrixMultiply(featureDifferenceMultB, currentFeature, false, true)[0][0];
        stepB = Matrix.matrixScalarMultiply(stepB, -1 / denominator);
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
        for (int i = 0; i < vector.length; i++) {
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
        } catch (Exception e) {
        }
        try {
            sampleSize = Integer.parseInt(args[3]);
        } catch (Exception e) {
        }
        boolean usePredefinedWeights = false;
        try {
            usePredefinedWeights = args[4].equals("y") ? true : false;
        } catch (Exception e) {
        }

        double[] startingWeights = { 0.00134246, // INDEX_NUM_ROWS_REMOVED
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
