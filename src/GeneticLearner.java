import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class GeneticLearner {
    public static void runGeneticAlgorithm() {
        double[] heuristicWeights =
        {
            0.00134246, // INDEX_NUM_ROWS_REMOVED
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

        int populationSize = 1000;
        double percentOfParents = 0.4;
        double percentOfElites = 0.005;
        double percentOfOffsprings = 0.4;
        double percentOfCrossover = 0.8;
        double percentOfMutation = 0.08;
        int convergenceCount = 50;
        String populationName = "row_13_trainers";

        // instantiate population
        Population population = new Population(
            populationSize,
            percentOfParents,
            // percentOfElites,
            percentOfOffsprings,
            percentOfCrossover,
            percentOfMutation,
            convergenceCount,
            // heuristicWeights,
            populationName
        );
        // Initial Report!
        Population.generateReport(population);
        while(!population.hasConverged()) {
            System.out.printf("Running Generation %d next, convergence: %d of %d",
                population.getGenerationCount()+1,
                population.getImprovementCounter(),
                convergenceCount
            );
            System.out.println();
            population.runGeneration();
            // Generate a report every generation
            Population.generateReport(population);
        }
    }

    public static void main(String[] args) {
        runGeneticAlgorithm();
    }
}

class Chromosome implements Comparable<Chromosome> {
    private final double mutationChance; 
    private Random rand;
    private double[] weights;
    private double fitness = 0;

    public Chromosome(int _numGenes, double _mutationChance) {
        mutationChance = _mutationChance;
        rand = new Random(System.nanoTime());
        weights = new double[_numGenes];
        // start with random weights
        for(int i = 0; i < _numGenes; i++) {
            weights[i] = (rand.nextBoolean()) ? rand.nextDouble() : -rand.nextDouble();
        }
    }

    public Chromosome(double[] _weights, double _mutationChance) {
        mutationChance = _mutationChance;
        rand = new Random(System.nanoTime());
        weights = new double[_weights.length];
        
        double squareSum = 0.001;
        // normalise weights
        for (int i = 0; i < _weights.length; i++) {
            squareSum += _weights[i]*_weights[i];
        }
        squareSum = Math.sqrt(squareSum);
        for(int i = 0; i < _weights.length; i++) {
            weights[i] = _weights[i] / squareSum;
        }
    }

    /**
     * returns an exact clone of the chromosome, with the previous fitness
     */
    public Chromosome copy() {
        double[] copyWeights = new double[this.weights.length];
        System.arraycopy(weights, 0, copyWeights, 0, weights.length);
        Chromosome clone = new Chromosome(copyWeights, mutationChance);
        clone.setFitness(this.getFitness());
        return clone;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double _fitness) {
        fitness = _fitness;
    }

    public void evaluateFitness(int games, int moves) {
        // Run 100 games to set the fitness, of max moves 500, OR till they lost
        double score = PlayerSkeleton.runGames(games, this.getWeights(), moves);
        this.setFitness(score);
    }

    public void forceMutate() {
        mutateBase();
    }

    public void mutate() {
        if(rand.nextDouble() <= mutationChance) {
            mutateBase();
        }
    }


    private void mutateBase() {
        int indexToMutate = rand.nextInt(weights.length);
        double mutationAmount = rand.nextGaussian() * 0.10;
        weights[indexToMutate] += mutationAmount;
    }

    // Whole arithmetic recombination crossover
    public static void crossOver(Chromosome c1, Chromosome c2) {
        double alpha = c1.rand.nextDouble();
        for(int i = 0; i < c1.weights.length; i++) {
            double x = c1.weights[i];
            double y = c2.weights[i];
            c1.weights[i] = alpha * x + (1 - alpha) * y;
            c2.weights[i] = alpha * y + (1 - alpha) * x;
        }
    }

    public int compareTo(Chromosome other) {
        return Double.compare(this.fitness, other.fitness);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(fitness);
        for(double weight: weights) {
            sb.append(", ");
            sb.append(weight);
        }
        return sb.toString();
    }
}

class Population {
    private int generationCount;
    // Counter to check if the fitness has improved (If not improved, 
    // increment it until it reaches convergence count)
    private int improvementCounter;
    private Random rand;

    /**
     * Properties that stays constant once set
     */
    private final int populationSize;
    // percentage of the population to be considered as candidates for parents to the next gene
    private final double percentageOfParents;
    // percentage of the population to be kept as the elites
    // private final double percentageOfElites;
    // percentage of offsprings per generation
    private final double percentageOfOffsprings;
    // chance of crossing over
    private final double percentageCrossOver;
    // chance of mutation
    private final double percentageMutation;
    private final int convergenceCount;
    private final String populationName;

    private ArrayList<Chromosome> populationPool;
    /**
     * Initialises a population and evaluates its fitness
     */
    public Population(
        int _populationSize,
        double _percentageOfParents,
        // double _percentageOfElites,
        double _percentageOfOffsprings,
        double _percentageCrossOver,
        double _percentageMutation,
        int _convergenceCount,
        // Inserts heuristic weights as elites, if the aren't null at the start
        // double[] _heuristicWeights,
        String _populationName
    ) {
        generationCount = 0;
        improvementCounter = 0;
        rand = new Random(System.nanoTime());

        populationSize = _populationSize;
        percentageOfParents = _percentageOfParents;
        // percentageOfElites = _percentageOfElites;
        percentageOfOffsprings = _percentageOfOffsprings;
        percentageCrossOver = _percentageCrossOver;
        percentageMutation = _percentageMutation;
        convergenceCount = _convergenceCount;
        populationName = _populationName;

        populationPool = new ArrayList<>(populationSize);
        // if(_heuristicWeights != null) {
        //     int numberOfElites = (int) (populationSize * percentageOfElites);
        //     for(int i = 0; i < numberOfElites; i++) {
        //         Chromosome elite = new Chromosome(_heuristicWeights, percentageMutation);
        //         elite.mutate();
        //         populationPool.add(elite);
        //     }
        // }

        while(populationPool.size() < populationSize) {
            populationPool.add(new Chromosome(FeatureFunction.NUM_FEATURES, percentageMutation));
        }
        for (Chromosome chromosome : populationPool) {
            chromosome.evaluateFitness(50, -1);
        }
        // Sort population in descending order, so that the fittest is in front
        Collections.sort(populationPool, Collections.reverseOrder());
    }

    public int getGenerationCount() {
        return generationCount;
    }

    public int getImprovementCounter() {
        return improvementCounter;
    }

    public boolean hasConverged() {
        /**
         * TODO: hasConverged is now kept to false
         * this running the genetic algorithm indefinitely, but checking up on
         * the counter is also useful in telling us how much improvement is needed
         */
        // return improvementCounter > convergenceCount;
        return false;
    }

    public String getPopulationName() {
        return populationName;
    }

    public ArrayList<Chromosome> getPopulationPool() {
        return populationPool;
    }

    public void runGeneration() {
        generationCount++;
        int numberOfOffsprings = (int) (populationSize * percentageOfOffsprings);
        ArrayList<Chromosome> offSpringPool = new ArrayList<>(numberOfOffsprings);

        int numberOfParents = (int)(populationSize * percentageOfParents);
        ArrayList<Chromosome> parentSelection = stochasticUniversalSampling(numberOfParents);

        double bestOffspringScore = Double.MIN_VALUE;
        // Fill the new population pool with crossed over children
        while(offSpringPool.size() < numberOfOffsprings) {
        // for(int i = 1; i < parentSelection.size(); i++) {
            int i = 0; int j = 0;
            do{
                i = rand.nextInt(parentSelection.size());
                j = rand.nextInt(parentSelection.size());
            } while (i != j);
            
            Chromosome c1 = parentSelection.get(i).copy();
            Chromosome c2 = parentSelection.get(j).copy();
            // has a % chance to be crossed over
            if(rand.nextDouble() <= percentageCrossOver) {
                Chromosome.crossOver(c1, c2);
            }
            c1.mutate();
            c2.mutate();
            c1.evaluateFitness(50, -1);
            c2.evaluateFitness(50, -1);
            bestOffspringScore = Math.max(bestOffspringScore, Math.max(c1.getFitness(), c2.getFitness()));
            offSpringPool.add(c1);
            offSpringPool.add(c2);
        }
        // Check BEFORE culling if the any of the offsprings are better than the
        // individuals inside the population
        // If YES, then reset the counter, IF NO, then add 1 to the counter
        checkIfImproved(bestOffspringScore);

        // Cull the bottom few and replace it with the child chromosomes
        int numberOfChromosomesToKeep = populationSize - offSpringPool.size();
        // CULLING STEP
        populationPool.subList(numberOfChromosomesToKeep, populationPool.size()).clear();

        // ADDING THE NEW OFFSPRINGS IN
        populationPool.addAll(offSpringPool);

        // Sort population in descending order, so that the fittest is in front
        Collections.sort(populationPool, Collections.reverseOrder());
    }

    private ArrayList<Chromosome> stochasticUniversalSampling(int numToSelect) {
        double totalFitness = 0;
        for(Chromosome chromosome: populationPool) {
            totalFitness += chromosome.getFitness();
        }
        double distanceBetweenPointers = totalFitness / numToSelect;
        // pointer, initially start at the 1st pointer, which is [0, totalFitness/numToSelect]
        double start = rand.nextDouble() * distanceBetweenPointers;
        ArrayList<Chromosome> picked = new ArrayList<>(numToSelect);
        int index = 0;
        double sum = populationPool.get(index).getFitness();
        for(int i = 0; i < numToSelect; i++) {
            double pointer = start + i * distanceBetweenPointers;
            while(sum < pointer) {
                index++;
                sum += populationPool.get(index).getFitness();
            }
            picked.add(populationPool.get(index).copy());
        }
        // shuffled the picked samples
        Collections.shuffle(picked, rand);
        return picked;
    }

    /**
     * Call this method BEFORE culling the old population and merging the
     * new chromosome pool inside the population
     */
    private void checkIfImproved(double bestOffspringScore) {
        boolean hasImproved = false;
        // Run it backwards, seeing the lousier ones in the current population
        // The population is always sorted in DESCENDING ORDER.
        for(int i = populationPool.size() - 1; i >= 0; i--) {
            // Naively checking such that IF the best offspring isnt good enough
            // for the worst/starting from the worst to the best in the population
            // then there is no improvement for this next generation
            if(bestOffspringScore > populationPool.get(i).getFitness()) {
                hasImproved = true;
                break;
            }
        }
        if(hasImproved) {
            improvementCounter = 0;
        } else {
            improvementCounter++;
        }
    }

    public static void generateReport(Population population) {
        String filename = population.getPopulationName() + "_" + population.getGenerationCount()+".csv";
        ArrayList<Chromosome> populationPool = population.getPopulationPool();
        String[] header = {
            "fitness",
            "rows_removed",
            "max_height",
            "average_height",
            "average_difference_height",
            "landing_height",
            "number_of_holes",
            "column_transitions",
            "row_transitions",
            "covered_gaps",
            "total_well_depth",
            "has_lost",
        };
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for(String headerValue: header) {
            if(!first) {
                sb.append(", ");
            }
            sb.append(headerValue);
            first = false;
        }
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(String.format(filename)))) {
            bw.write(sb.toString());
            bw.newLine();
            for (Chromosome chromosome : populationPool) {
                bw.write(chromosome.toString());
                bw.newLine();
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
