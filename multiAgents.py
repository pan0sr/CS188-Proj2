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

        liveGhostPositions = [newGhostStates[i].getPosition() for i in range(len(newGhostStates)) if newScaredTimes[i] == 0]
        if (newPos in liveGhostPositions):
            return -500
        eval = 0
        if successorGameState.getNumFood() == 0:
            return 500

        man = lambda x, y: abs(x[0]-y[0]) + abs(x[1]-y[1])
        for i in range(len(liveGhostPositions)):
            eval -= 100/(man(newPos, liveGhostPositions[i]) ** 5)
        for n in newFood.asList():
            eval += 1/man(newPos, n)
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            eval += 20
        eval += random.gauss(0, 0.5) # break ties
        return eval
        return successorGameState.getScore()





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
        numAgents = gameState.getNumAgents()
        
        def helperBase(state, agentIndex):
            acts = state.getLegalActions(agentIndex)
            if len(acts) == 0:
                return 0, self.evaluationFunction(state)
            scores = [self.evaluationFunction(state.generateSuccessor(agentIndex, acts[i])) for i in range(len(acts))]
            if agentIndex == 0:
                score = max(scores)
                act = acts[scores.index(score)]
            else:
                score = min(scores) 
                act = acts[scores.index(score)]
            return act, score
            
        def helperRecursive(state, agentIndex, level):
            acts = state.getLegalActions(agentIndex)
            successors = [state.generateSuccessor(agentIndex, acts[i]) for i in range(len(acts))]
            if agentIndex == 0:
                fn = max
            else: 
                fn = min
            if len(acts) == 0:
                return 0, self.evaluationFunction(state)
            if level == 0:
                if agentIndex == numAgents - 1:
                    return helperBase(state, agentIndex)
                else:
                    scores = [helperRecursive(successors[i], agentIndex+1, level)[1] for i in range(len(acts))]
                    ind = scores.index(fn(scores))
                    return acts[ind], scores[ind]
            else:
                if agentIndex == numAgents - 1:
                    scores = [helperRecursive(successors[i], 0, level-1)[1] for i in range(len(acts))]
                    ind = scores.index(fn(scores))
                    return acts[ind], scores[ind]
                else:
                    scores = [helperRecursive(successors[i], agentIndex+1, level)[1] for i in range(len(acts))]
                    ind = scores.index(fn(scores))
                    return acts[ind], scores[ind]

        return helperRecursive(gameState, 0, self.depth-1)[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def maxAgent(state:GameState,alpha,beta,agentIndex,depth):
            best_move = None
            value = -999999999
            acts = state.getLegalActions(0)
            if (not acts):
                return self.evaluationFunction(state),0
            for move in acts:
                successorsState = state.generateSuccessor(agentIndex,move)
                new_value = max(value,minAgent(successorsState,alpha,beta,agentIndex+1,depth))
                if new_value > value:
                    value = new_value
                    best_move = move
                if value > beta:
                    return value, 0
                alpha = max(alpha,value)
            return value, best_move
        
        def minAgent(state : GameState,alpha,beta, agentIndex,depth):
            value = 999999999
            acts = state.getLegalActions(agentIndex)
            if (not acts):
                return self.evaluationFunction(state)
            if (agentIndex == state.getNumAgents() - 1):
                if depth == 0:
                    for move in acts:
                        successorState = state.generateSuccessor(agentIndex,move)
                        value = min(value,self.evaluationFunction(successorState))
                        if value < alpha: 
                            return value
                        else:
                            beta = min(beta,value)
                    return value
                else:
                    for move in acts:
                        successorState = state.generateSuccessor(agentIndex,move)
                        value = min(value,maxAgent(successorState,alpha,beta,0,depth-1)[0])
                        if value < alpha: 
                            return value
                        else:
                            beta = min(beta,value)
                    return value
            for move in acts:
                successorState = state.generateSuccessor(agentIndex,move)
                value = min(value,minAgent(successorState,alpha,beta,agentIndex+1,depth))
                if value < alpha: 
                    return value
                else:
                    beta = min(beta,value)
            return value
        return maxAgent(gameState,-99999999,9999999,0,self.depth-1)[1]


                    


    

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

