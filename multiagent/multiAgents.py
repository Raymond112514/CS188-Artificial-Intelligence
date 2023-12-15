# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDist = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDist = min(foodDist) if len(foodDist) != 0 else 0
        maxFoodDist = max(foodDist) if len(foodDist) != 0 else 0

        minGhostDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        if len(foodDist) > 1:
            if minGhostDist >= 10:
                total_score = successorGameState.getScore() - minFoodDist
            else:
                total_score = successorGameState.getScore() + minGhostDist - (minFoodDist + maxFoodDist)
        else:
            total_score = successorGameState.getScore() + 0.5 * minGhostDist - minFoodDist - len(foodDist)
        return total_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agent, depth):
            ## Params
            # agent: 0 for a maximizer, minimizer otherwise
            # isMaxAgent: returns true if the agent is a maximizer
            isMaxAgent = True if agent == 0 else 0
            operator = max if isMaxAgent else min
            opt_val = -float("inf") if isMaxAgent else float("inf")
            next_depth = depth - 1 if isMaxAgent else depth
            next_agent = (agent + 1) % state.getNumAgents()
            opt_act = None
            if next_depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                action_val = operator(opt_val, minimax(successor, next_agent, next_depth)[0])
                if action_val != opt_val:
                    opt_val = action_val
                    opt_act = action
            return opt_val, opt_act
        return minimax(gameState, self.index, self.depth + 1)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(state, agent, depth, alpha, beta):
            isMaxAgent = True if agent == 0 else 0
            operator = max if isMaxAgent else min
            opt_val = -float("inf") if isMaxAgent else float("inf")
            next_depth = depth - 1 if isMaxAgent else depth
            next_agent = (agent + 1) % state.getNumAgents()
            opt_act = None
            if next_depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                action_val = operator(opt_val, alpha_beta(successor, next_agent, next_depth, alpha, beta)[0])
                if action_val != opt_val:
                    opt_val = action_val
                    opt_act = action
                if isMaxAgent:
                    if action_val > beta:
                        return opt_val, opt_act
                    alpha = max(alpha, opt_val)
                else:
                    if action_val < alpha:
                        return opt_val, opt_act
                    beta = min(beta, opt_val)
            return opt_val, opt_act
        return alpha_beta(gameState, self.index, self.depth + 1, -float("inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, agent, depth):
            isMaxAgent = True if agent == 0 else False
            opt_val = -float("inf")
            next_depth = depth - 1 if isMaxAgent else depth
            next_agent = (agent + 1) % state.getNumAgents()
            opt_act = None
            if next_depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            if not isMaxAgent:
                expected_val = 0.0
                for action in state.getLegalActions(agent):
                    successor = state.generateSuccessor(agent, action)
                    expected_val += expectimax(successor, next_agent, next_depth)[0]
                return expected_val / len(state.getLegalActions(agent)), None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                action_val = max(expectimax(successor, next_agent, next_depth)[0], opt_val)
                if action_val > opt_val:
                    opt_val = action_val
                    opt_act = action
            return opt_val, opt_act
        return expectimax(gameState, self.index, self.depth + 1)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: When there's much food left, the pacman is designed to be more risk-averse
    whereas when there's little food left, the pacman becomes more risk averse (such as decreasing
    the effect of minGhostDist and increasing the effect of numFoodRemain.

    When there's a lot of food left, if the ghost is far away (greater than 4 blocks, then the score is
    dominated by the effect of numFoodRemain. When the ghost becomes too close, the agent becomes
    nervous and the score becomes dominated by minGhostDist
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPosition = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = min([ghostState.scaredTimer for ghostState in ghostStates])
    score = currentGameState.getScore()

    numFoodRemain = len(foodPosition)
    foodDist = [manhattanDistance(pacmanPosition, food) for food in foodPosition]
    minGhostDist = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates])
    if len(foodDist) > 3:
        if minGhostDist >= 4 or scaredTimes > 0:
            totalScore = score - 2 * numFoodRemain
        else:
            totalScore = score + 2 * minGhostDist - numFoodRemain
    else:
        totalScore = 3 * score + 0.10 * minGhostDist - 5 * numFoodRemain
    return totalScore

# Abbreviation
better = betterEvaluationFunction
