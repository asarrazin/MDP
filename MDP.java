import java.util.Arrays;
import java.lang.*;
import java.math.*;
import java.text.DecimalFormat;
import java.util.Random; 

/**
 * CSCI 3420: Optimization and Uncertainty
 * Assignment 1: Markov Decision Processes
 * Authors: Adrienne Miller and Anaïs Sarrazin
 * 22 April 2020
 * 
 * This class contains two different solution techniques, in Java programming, to solving a more complicated version of the gridworld problem:
 *      1) Value Iteration
 *      2) Policy Iteration
 *      
 *  This class also contains multiple helper methods: round, move, chooseAction, and createGrid. round rounds the double for
 *  utility value to the correct decimal place. move and chooseAction are helper methods used in q-learning to choose 
 *  next moves and nextStates based on q-values/softmax exploration and the transition function, respectively. createGrid
 *  is used in our main method to print out the final grid, with policy and utilities. 
 *  
 *  The user may specify the type of method that will be used to solve the grid problem, along with the other relavent 
 *  parameters to theses solution techniques. For more detailed instructions on how to run main method, consult ReadMe. 
 * 
 */ 
public class MDP
{

    private static double[] u;       // Array holding all the utility values
    private static int[] p;      // Array holding all the policy 
    private static long time;  
    private static int iterations; 
    private static int actions;  // for q-learning, number of actions taken 
    /**
     * Constructor for objects of class ValueIteration.
     *  
     */
    public MDP(){
        this.u = new double[65];
        this.p = new int[65];
        this.iterations = 0; 
        this.time = 0; 
        this.actions = 0; 
    }

    /**
     * Purpose: This rounds a double to a given number of places by using the BigDecimal object. (Note: retrieved help
     * for this method on the internet. Cited on README). 
     * Parameters: The given double number to be rounded and the certain amount of places desired.
     * Return Value: The number rounded to the desired number of places. 
     */
    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        // Reduce the number of fractional digits to the number of places and ensure it rounds up, with CEILING. 
        bd = bd.setScale(places, RoundingMode.CEILING);
        return bd.doubleValue();
    }

    /**
     * Purpose: This takes a state and an action to find a new state. We return the reward for this new state in the 
     * q-learning method.Then, this method selects a random double between 0 and 1, and depending on the range that 
     * this double falls into, selects a particular nextState.
     * Parameters: A designated state. A designated action. A 3D array holding the transition matrix created in the 
     * TransitionFunction class. An array holding the reward values collected from the RewardFunction class. 
     * Return Value: A possible next state. If there is no next state, that is, if the current state is a terminal state, 
     * the move function returns -1. 
     */
    public static int move(int state, int action, double[][][] transition, double[] rewards){ 
        Random rand = new Random(); 
        double randomNum = rand.nextDouble(); 
        double startIndex = 0;
        double endIndex = 0; 
        int nextState = -1; // Returns -1 if there is no nextState (if we have reached a terminal state) 

        // For loop works by 'allocating' certain ranges from 0-1 to the probability that you'll end up in .
        for (int i = 0;  i < 65; i++){ 
            if (transition[state][action][i] != 0){ 
                endIndex = startIndex + transition[state][action][i];
                if ((randomNum >= startIndex) && (randomNum < endIndex)){
                    nextState = i;
                    break; 
                }
                startIndex = endIndex; 
            }
        }

        return nextState; 
    } 

    /**
     * Purpose: This uses softmax exploration to pick the action the agent will take from its current state. In the softmax 
     * approach to exploration, actions are chosen probabilistically based on their Q-values. Then, this method selects 
     * a random double between 0 and 1, and depending on the range that this double falls into, selects a particular action.
     * Parameters: A given state. A 2D array holding the Q-values. 
     * Return Value: An action. 
     */
    public static int chooseAction(int state, double[][] Q){ 
        double[] probAction = new double[5]; // array to hold probability that any given action will occur
        double eSum = 0; 

        // Calulates denominator for softmax exploration, to determine probabilities of each action
        for (int j = 1; j < 5; j ++){ 
            eSum += java.lang.Math.exp(Q[state][j]);
        } 

        // Finds probability for each action given their Q values 
        for (int i = 1; i < 5; i++){
            probAction[i] = (java.lang.Math.exp(Q[state][i])) / eSum; 
        } 
        Random rand = new Random(); 
        double num = rand.nextDouble(); 
        double startIndex = 0; 
        double endIndex = 0; 
        int action = 1; 

        // This loops works similarly to the one in move, by determining the next action that should be taken based
        // On a random number, and the "range" of probability that one action covers 
        for (int j = 1; j < 5; j++){
            endIndex = startIndex + probAction[j]; 
            if ((num >= startIndex) && (num < endIndex)){
                action = j;
                break; 
            }
            startIndex = endIndex; 
        } 

        return action; 

    } 

    /**
     * Purpose: This solves this problem using Quantitative Learning, where you learn values on state-action pairs.
     * Parameters: A 3D array holding the transition matrix created in the TransitionFunction class. An array holding the 
     * reward values collected from the RewardFunction class. A discount factor. Number of trajectories to run. A learning rate.
     * A boolean specificying whether to show state-action-state transitions.
     * Return Value: An array holding the policy values for each state. 
     */
    public static int[] qLearning(double[][][] transition, double[] rewards, double discount, int trajectories, double learning, boolean show){ 
        long startTime = System.currentTimeMillis(); 
        iterations = 0; // Number of trajectories (for each state) that have been excecuted

        double[][] Q = new double[65][5];

        // Set q values for terminal states. 
        for (int i = 1; i < 5; i++){
            Q[28][i] = 1; 
            Q[29][i] = -0.04;
            Q[44][i] = -1; 
            Q[45][i] = -1; 
            Q[48][i] = -1; 
            Q[49][i] = -1; 
        } 

        while (iterations < trajectories){ 
            //  Loop through each state, starting a trajectory at each one.
            for (int i = 0; i < 65; i++){
                int currentState = i; 
                boolean notTerminal = true; 
                while (notTerminal){ 
                    int action = chooseAction(currentState, Q);  // Chooses the action to be taken 
                    p[currentState] = action; // Updates policy to be current action 
                    int nextState = move(currentState, action, transition, rewards); // Makes move
                    // Make sure if nextState is in terminal state. If statement stops inner while loop if reached terminal.
                    if (nextState == -1){ 
                        notTerminal = false; 
                        break; 
                    } 

                    double qMax = -100; 
                    // Find maximum q-value over all actions for the nextState.
                    for (int j = 1; j < 5; j++){
                        if (Q[nextState][j] > qMax){
                            qMax = Q[nextState][j];
                        }
                    } 
                    qMax = qMax * discount; // Apply discount factor to qMax

                    // Update q value for state action pair. 
                    double currentQVal = Q[currentState][action]; // Store temporary q value
                    Q[currentState][action] = currentQVal + (learning *(rewards[currentState] + qMax - currentQVal));

                    // Print out moves if user specifies.
                    if (show){
                        String actionString = "";
                        switch (action){
                            case 1: actionString = "N"; break;
                            case 2: actionString = "E"; break;
                            case 3: actionString = "S"; break;
                            case 4: actionString = "W"; break;
                        }  
                        System.out.println("("  + currentState + ", " + action + ", " + nextState + ")");
                    }
                    currentState = nextState;  // "Moves" agent to next state
                    actions++;  // Increase actions counter
                } 
            }
            iterations++; 

        }

        // Transfer q-values to u array, so that they will print in grid. 
        for (int i = 0; i < 65; i++){
            u[i] = Q[i][p[i]];

        } 

        long endTime = System.currentTimeMillis(); 
        time = endTime - startTime; 
        return p; 

    } 

    /**
     * Purpose: This solves this problem using Policy Iteration, where you iteratively improve the current policy. 
     * Parameters: A 3D array holding the transition matrix created in the TransitionFunction class. An array holding the reward 
     * values collected from the RewardFunction class. A discount factor provided by the user. 
     * Return Value: An array holding the policy values for each state. 
     */
    public static int[] policyIteration(double[][][] transition, double[] rewards, 
    double discount){

        long startTime = System.currentTimeMillis(); 
        double [][] ba = new double[65][1]; 

        // Initialize policy with arbitrary values and initializes rewards array. 
        for (int k = 0; k < 65; k++){ 
            p[k] = 1; // Assigns N as the default policy 
            ba[k][0] = rewards[k]; // Initialize the array that will become rewards matrix.
        }

        Matrix B = new Matrix(ba); 
        boolean unchanged = false; 

        while (!unchanged){ 

            // Next two for loops initializes transition matrix 

            double[][] ma = new double[65][65]; 

            for (int i = 0; i < 65; i++){ 
                ma[i][i] = 1; // Make ma identity matrix 
            } 

            for (int j = 0; j < 65; j++){ 
                for (int k = 0; k < 65; k++){ 
                    // Change values using discount and transition values
                    ma[j][k] = ma[j][k] - (discount * transition[j][p[j]][k]);
                }
            } 

            Matrix M = new Matrix(ma); 

            Matrix X = M.solve(B); // Solve for utility matrix

            // Put matrix solution (utility values) into array form. 
            for (int s = 0; s < 65; s++) {
                u[s] = X.get(s,0);
            }

            unchanged = true; // Asssumes that policy is unchanged, until it changes 
            // Do one round of Bellman updates
            for (int i = 0; i < 65; i++){
                int currentPolicy = p[i];  // Temp variable for current policy of state i 
                double max = -1000000000; 
                for (int j = 1; j < 5; j++){
                    double weightProb = 0; 
                    // Calculate weighted probabilities
                    for(int k = 0; k < 65; k++){
                        weightProb += (transition[i][j][k] * u[k]); 

                    }
                    if (weightProb > max){
                        max = weightProb; 
                        p[i] = j; // Changes policy to what Bellman update would suggest

                    } 

                }

                if (currentPolicy != p[i]){
                    // If the policy has changed since Bellman update, unchanged is false 
                    unchanged =  false;  
                }

            }
            iterations++; 
        } 
        long endTime = System.currentTimeMillis(); 
        time = endTime - startTime; 
        return p; 
    } 

    /**
     * Purpose: This solves this problem using synchronus, in-place Value Iteration, where you calculate new utilities for all 
     * the states on every iteration.
     * Parameters: A 3D array holding the transition model created in the TransitionFunction class. An array holding the reward 
     * values collected from the RewardFunction class. A discount factor provided by the user. A maximum allowable error allowed 
     * in the utility of any state in the iteration.
     * Return Value: An array holding the policy values for each state. 
     */
    public static int[] valueIteration(double[][][]transition, double [] rewards, double discount,
    double maxError){ 
        long startTime = System.currentTimeMillis(); 
        double maxChange = 1; 

        while (maxChange > (maxError * ((1 - discount) / discount))) { 
            maxChange = 0;
            iterations++; 

            // Perform a Bellman update
            for (int i = 0; i < 65; i++){ // For each state
                double temp = u[i]; 
                double max = -100; 
                for (int j = 1; j < 5; j++){  // For each action 
                    double weightedProb = 0; 

                    for (int k = 0; k < 65; k++){  // For each possible next state
                        // Calculate weighted probability  
                        weightedProb += (transition[i][j][k] * u[k]);

                    }

                    // Update maximum weighted probabilit 
                    if (weightedProb > max){ 
                        max = weightedProb;
                        p[i] = j; 
                    }
                }
                u[i] = rewards[i] + (discount * max); // Update new utility value 
                double change; 
                change = u[i] - temp; // Find change in utility

                if (change > maxChange){ 
                    maxChange = change;
                } 
            }
        }
        long endTime = System.currentTimeMillis(); 
        time = endTime - startTime; 
        return p; 
    } 

    /**
     * Purpose: This creates the grid holding solution diagram for the state values and the policy.
     * Parameters: N/A.
     * Return Value: N/A. 
     */
    public static void createGrid()
    {
        double utility = 0;
        String policy = "";
        double currentState = 0;
        String[] p2 = new String[65];  
        String even = "";  // put even states information in this line
        String odd = "";  // put odd states information in this line

        // Replace the policy numbers with the Actions for North, East, South, and West. 
        for (int q = 0; q < 65; q ++){ 
            int policyInt = p[q]; 
            switch ((int)policyInt){
                case 1: 
                p2[q] = "N"; 
                break; 
                case 2:  
                p2[q] = "E"; 
                break; 
                case 3: 
                p2[q] = "S"; 
                break; 
                case 4:     
                p2[q] = "W"; 
                break; 

            } 
        }

        // Print state values and policy for states 58 to 64. 
        for(int i = 58; i < 65; i++) {
            utility = u[i];
            policy = p2[i];
            currentState = i; 
            if (i % 2 == 0){ 
                even += ( "(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[i] + ")   "); 
            }else{
                odd += ("(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[i] + ")   ");
            }
        } 

        System.out.println(even); 
        System.out.println(odd); 
        even = "";
        odd = ""; 
        System.out.println(""); 

        // Print state values and policy for states 50 to 57. 
        for(int k = 50; k < 58; k++) {
            utility = u[k];
            policy = p2[k];
            currentState = k; 
            if (k % 2 == 0){ 
                even += ( "(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[k] + ")   "); 
            }else{
                odd +=("(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[k] + ")   ");

            }
        }
        System.out.println(even); 
        System.out.println(odd); 
        even = "";
        odd = ""; 
        System.out.println("");

        // Print state values and policy for states 40 to 49. 
        for(int j = 40 ; j < 50; j++) {
            utility = u[j];
            policy = p2[j];
            currentState = j; 
            if (j % 2 == 0){ 
                even += ( "(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[j] + ")   "); 
            }else{
                odd += ("(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[j] + ")   ");

            } 
        } 
        System.out.println(even); 
        System.out.println(odd); 
        even = "";
        odd = ""; 
        System.out.println("");

        // Print state values and policy for states 30 to 39. 
        for(int l = 30; l < 40; l++) {
            utility = u[l];
            policy = p2[l];
            currentState = l; 
            if (l % 2 == 0){ 
                even +=(  "(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[l] + ")   "); 
            }else{
                odd +=("(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[l] + ")   ");

            } 
        } 
        System.out.println(even); 
        System.out.println(odd); 
        even = "";
        odd = ""; 
        System.out.println("");

        // Print state values and policy for states 0 to 29. 
        for(int m = 0; m < 30; m++) {
            utility = u[m];
            policy = p2[m];
            currentState = m; 
            if (m % 2 == 0){ 
                even += ( "(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[m] + ") "); 
            }else{
                odd += ("(" + currentState + ") " + (Math.round(utility * 100.0) / 100.0) + " " + "("  + p2[m] + ")  ");

            } 
        } 
        System.out.println(even); 
        System.out.println(odd); 
        even = "";
        odd = ""; 
    }

    /**
     * Main method.
     */
    public static void main(String args[]){ 
        // Get input from command line
        double discountFactor = Double.parseDouble(args[0]); 
        double maxError = Double.parseDouble(args[1]);
        double keyLossProb = Double.parseDouble(args[2]);
        double positiveReward = Double.parseDouble(args[3]);
        double negativeReward = Double.parseDouble(args[4]);
        double stepCost = Double.parseDouble(args[5]);
        char method = args[6].charAt(0); 
        int trajectories = Integer.parseInt(args[7]); 
        char showTransition = args[8].charAt(0); 
        double learningRate = Double.parseDouble(args[9]);

        // char method = 'p'; 
        // boolean showTransition = false; 
        // double learningRate = 0.1;
        // double discountFactor = .999999;
        // double positiveReward = 1; 
        // double negativeReward = -1; 
        // double stepCost = -.04; 
        // double keyLossProb = 0.5; 
        // int trajectories = 10000; 
        //MDP pol = new MDP(); 

        // Set up grid world (transition function, reward function)
        MDP mdp = new MDP();
        TransitionFunction T = new TransitionFunction(keyLossProb); 
        double[][][] tGrid = T.getGrid(); // Retrieve 3d array representation of transition functoin 
        RewardFunction R = new RewardFunction(positiveReward, negativeReward, tGrid, stepCost); 
        double[] rGrid = R.getRewardArray();  // Retrieve reward array for 

        String solutionMethod = ""; 
        System.out.println("");  
        boolean show = false; 
        if (showTransition == 't'){
            show = true; 
        } 

        if (method == 'v'){ // Run value iteration
            solutionMethod = "Value Iteration"; 
            System.out.println("here i am"); 
            mdp.valueIteration(tGrid, rGrid, discountFactor, .000001); 
        } 
        else if (method == 'p'){ // Run policy iteration 
            solutionMethod = "Policy Iteration"; 
            mdp.policyIteration(tGrid, rGrid, discountFactor); 

        } 

        System.out.println("Solution Technique: " + solutionMethod + "\n"); 
        System.out.println("Number of iterations : " + iterations ); 
        System.out.println("Time elapsed: " + time + " milliseconds \n");
        if (method == 'q'){ 
            System.out.println("Number of actions: " + actions); 
        }

        System.out.println("Discount factor: " + discountFactor); 

        if (method == 'v'){ 
            System.out.println("Max Error in State Utilities: " + maxError); // Max error only relevant to VI technique 
        }
        System.out.println("Positive Reward: " + positiveReward); 
        System.out.println("Positive Reward: " + negativeReward); 
        System.out.println("Step Cost: " + stepCost); 
        System.out.println("Key Loss Probability: " + keyLossProb);
        if (method == 'q'){ 
            System.out.println(trajectories + " trajectories"); // Trajectories only relevant to q-learning
            System.out.println("Show Transition: " + showTransition); // showTransition only relevant to q-learning
            System.out.println("Learning Rate: " + learningRate); // Learning rate only relevant to q-learning 

        } 
        System.out.println(" ");
        mdp.createGrid();
    } 
}
