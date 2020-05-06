import java.util.Arrays;

/**
 * CSCI 3420: Optimization and Uncertainty
 * Assignment 1: Markov Decision Processes
 * Authors: Adrienne Miller and Anaïs Sarrazin
 * 22 April 2020
 * 
 * This class finds the given reward for each action-state pair based on the following rules defined:
 *   - Every reward is -0.04 (step cost) with the following exceptions:
 *      1) If the agent is already in the square marked -1, and takes an action that lands it back in that square, the 
 *      reward is 0.0.
 *      2) If the agent is not already in the sqaure marked -1, but takes an action that lands it in that square, the 
 *      reward is -1.0.
 *      3) If the agent is already in the square marked +1, and takes an action that lands it back in that square, the 
 *      reward is 0.0.
 *      4) If the agent is not already in the square marked +1, but takes an action that ladns it in that square, and the agent 
 *      has a key, the reward is +1.0. 
 * The main class uses this reward array created in the different solution techniques. 
 * 
 */
public class RewardFunction
{

    private static double[]rewardArray;
    private static double posSquare;
    private static double negSquare;

    // Constants for special locations
    private static int posLocationWithKey;
    private static int posLocationWithoutKey;
    private static int negLocation1WithKey;
    private static int negLocation1WithoutKey;
    private static int negLocation2WithKey;
    private static int negLocation2WithoutKey;
    private static int key; 
    private static int loseKeyWithKey; 
    private static int loseKeyWithoutKey; 

    private static double stepCost; 

    private static int currentState;
    private static int nextState;
    private static int action;

    private static double[][][] tGrid; 

    /**
     * Constructor for objects of class RewardFunction.
     */
    public RewardFunction(double posSquare, double negSquare, double[][][] tGrid, double stepCost)
    {
        this.rewardArray = new double [65];
        this.posSquare = 1;
        this.negSquare = -1;
        this.posLocationWithKey = 28;
        this.posLocationWithoutKey = 29;
        this.negLocation1WithKey = 44;
        this.negLocation1WithoutKey = 45;
        this.negLocation2WithKey = 48;
        this.negLocation2WithoutKey = 49;
        this.key = 64;
        this.loseKeyWithKey = 40; 
        this.loseKeyWithoutKey = 41; 
        this.tGrid = tGrid; 
        this.stepCost = stepCost; 
        updateArray(tGrid); 
    }
    
    /**
     * Purpose: This retrieves the reward array holding the reward values.
     * Parameters: N/A.
     * Return Value: A reward array.
     */
    public static double[] getRewardArray(){
        return rewardArray; 
    }

    /**
     * Purpose: This updates the values in the reward array.
     * Parameter: The 3D transition matrix.
     * Return Value: A reward array.
     */
    private static void updateArray(double [][][] transitionArray)
    {
        for(int i = 0; i < 65; i++) {
            int current = i; 
            for(int j = 1; j < 5; j++) {
                int action = j; 
                for (int k = 0; k < 65; k++) { 
                    int next = k; 
                    // Make sure you can get to the next state from the current state by checking that the probability isn't 0.
                    if  (transitionArray[current][action][next] != 0){ 
                        rewardArray[next] = reward(next); 
                    } 

                }
            }
        }
    }

    /**
     * Purpose: This computes the reward value for the next state based on the particular rules defined. 
     * Parameter: The next state. 
     * Return Value: The reward (or step cost) that will be taken, given this next state. 
     */
    public static double reward(int nextState){
        if(nextState == negLocation1WithoutKey || nextState == negLocation1WithKey || nextState == negLocation2WithKey || 
        nextState == negLocation2WithoutKey) {
            return negSquare; 

        } else if(nextState == posLocationWithKey) {
            return posSquare; 

        } else if(nextState == posLocationWithoutKey) {
            return stepCost;

        } else {
            return stepCost;
        }

    }
}
