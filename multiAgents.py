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

class ReflexAgent(Agent):

    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        detectDistance=4
        result=0
        oldPos=currentGameState.getPacmanPosition()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        #print(util.manhattanDistance(oldPos,newPos))
        oldFood=currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()

        result += 20 * (len(oldFood) - len(newFood))
        for i in range(len(newFood)):
            if  util.manhattanDistance(newFood[i],newPos)<detectDistance:
                result+= detectDistance-util.manhattanDistance(newFood[i],newPos)
        # import time
        # time.sleep(1)

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions=currentGameState.getGhostPositions()
        for i in range(len(ghostPositions)):
            if util.manhattanDistance(ghostPositions[i],newPos)<detectDistance:
                result-=70/(util.manhattanDistance(ghostPositions[i],newPos)+1)

        "wonderful"
        return result

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        ghostNum = gameState.getNumAgents()-1
        #print(ghostNum)
        #print(self.depth)

        #print(newState.getPacmanPosition())
        #time.sleep(5)
        def stop(state,depth):
            return state.isWin() or state.isLose() or depth==self.depth


        def maxAgent(state,depth):
            if stop(state,depth):
                return self.evaluationFunction(state)

            actions=state.getLegalActions(0)
            newStates=[state.generateSuccessor(0,actions[i]) for i in range(len(actions))]

            #print(gameState.getPacmanPosition())
            value=-999999
            ghostIndex=1
            for newState in newStates:
                value=max(minAgent(newState,depth,ghostIndex),value)
            return value



        def minAgent(state, depth,ghostIndex):
            #print(state.getPacmanPosition())

            if stop(state,depth):
                return self.evaluationFunction(state)
            ghostActions=state.getLegalActions(ghostIndex)

            newStates=[state.generateSuccessor(ghostIndex,ghostActions[i]) for i in range(len(ghostActions))]


            value=999999
            for newState in newStates:
                if ghostIndex==ghostNum:
                    value =min(maxAgent(newState, depth+1), value)
                else:
                    value=min(minAgent(newState,depth,ghostIndex+1),value)
            #(value)
            return value
            # print(state.getGhostPosition(ghostIndex))
            # return 0
        #print((minAgent(gameState.generateSuccessor(0, 'East'), 0, 1)))
        result=[(minAgent(gameState.generateSuccessor(0, action), 0, 1),action) for action in
               gameState.getLegalActions(0)]
        result.sort()
        return result[-1][1]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ghostNum = gameState.getNumAgents() - 1
        largeNum=999999

        # print(ghostNum)
        # print(self.depth)

        # print(newState.getPacmanPosition())
        # time.sleep(5)
        def stop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def maxAgent(state, depth,alpha,beta):
            A=alpha
            B=beta
            if stop(state, depth):
                return self.evaluationFunction(state)

            actions = state.getLegalActions(0)
            #newStates = [state.generateSuccessor(0, actions[i]) for i in range(len(actions))]

            # print(gameState.getPacmanPosition())
            value = -largeNum
            ghostIndex = 1
            for i in range(len(actions)):
                newState=state.generateSuccessor(0, actions[i])
                value = max(minAgent(newState, depth, ghostIndex,A,B), value)
                if value>B:
                    return value
                A=max(A,value)
            return value

        def minAgent(state, depth, ghostIndex,alpha,beta):
            # print(state.getPacmanPosition())
            A = alpha
            B = beta

            if stop(state, depth):
                return self.evaluationFunction(state)
            ghostActions = state.getLegalActions(ghostIndex)

            #newStates = [state.generateSuccessor(ghostIndex, ghostActions[i]) for i in range(len(ghostActions))]

            value = largeNum
            for i in range(len(ghostActions)):
                newState=state.generateSuccessor(ghostIndex, ghostActions[i])
                if ghostIndex == ghostNum:
                    value = min(maxAgent(newState, depth + 1,A,B), value)
                else:
                    value = min(minAgent(newState, depth, ghostIndex + 1,A,B), value)

                #print(str(value)+":"+str(A))
                if value<A:
                    return value
                B=min(B,value)
            # (value)
            return value
            # print(state.getGhostPosition(ghostIndex))
            # return 0

        def entrance():
            A=-largeNum
            B=largeNum
            maxValue=-largeNum

            result=1
            actions = []
            for action in gameState.getLegalActions(0):
                actions.append(action)

            for i in range(len(actions)):
                temp=minAgent(gameState.generateSuccessor(0, actions[i]), 0, 1,A,B)
                if maxValue<temp:
                    result=actions[i]
                    maxValue=temp

                if maxValue > B:  # pruning
                    return maxValue
                A = max(A, temp)
            return result

        return entrance()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        ghostNum = gameState.getNumAgents()-1
        #print(ghostNum)
        #print(self.depth)

        #print(newState.getPacmanPosition())
        #time.sleep(5)
        def stop(state,depth):
            return state.isWin() or state.isLose() or depth==self.depth


        def maxAgent(state,depth):
            if stop(state,depth):
                return self.evaluationFunction(state)

            actions=state.getLegalActions(0)
            newStates=[state.generateSuccessor(0,actions[i]) for i in range(len(actions))]

            #print(gameState.getPacmanPosition())
            value=-999999
            ghostIndex=1
            for newState in newStates:
                value=max(minAgent(newState,depth,ghostIndex),value)
            return value

        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        def minAgent(state, depth,ghostIndex):
            #print(state.getPacmanPosition())

            if stop(state,depth):
                return self.evaluationFunction(state)
            ghostActions=state.getLegalActions(ghostIndex)

            newStates=[state.generateSuccessor(ghostIndex,ghostActions[i]) for i in range(len(ghostActions))]
            value=0
            p = 1 / len(state.getLegalActions(ghostIndex))
            for newState in newStates:
                if ghostIndex==GhostIndex[-1]:
                    value +=p*maxAgent(newState, depth+1)
                else:
                    value+=p*minAgent(newState,depth,ghostIndex+1)
            #(value)
            return value

        result=[(minAgent(gameState.generateSuccessor(0, action), 0, 1),action) for action in
               gameState.getLegalActions(0)]
        result.sort()

        return result[-1][1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isLose():
        return float('-inf')
    if currentGameState.isWin():
        return float('inf')
    pos = currentGameState.getPacmanPosition()
    ghostPos = [G.getPosition() for G in currentGameState.getGhostStates()]
    if pos in ghostPos:
        return float('-inf')
    foods = currentGameState.getFood().asList()

    distanceToFood=[util.manhattanDistance(pos,food) for food in foods]
    distanceToGhost=[util.manhattanDistance(pos,ghost) for ghost in ghostPos]
    mDF=distanceToFood[distanceToFood.index(min(distanceToFood))]
    mDG=distanceToGhost[distanceToGhost.index(min(distanceToGhost))]
    score=0
    detectDistance=3
    if mDG<=detectDistance:
        score-=20/(mDG-0.5)**2
    for i in range(5):
        if len(currentGameState.getCapsules()) < i:
            score += 40
    score += scoreEvaluationFunction(currentGameState) + 12 / mDF
    return score








# Abbreviation
better = betterEvaluationFunction
