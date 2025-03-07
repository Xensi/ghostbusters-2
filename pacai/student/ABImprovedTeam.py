import random

from pacai.agents.capture.capture import CaptureAgent
from pacai.util import util

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.custom.AttackAgent',
        second = 'pacai.agents.capture.custom.DefenseAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = OffensiveABAgent
    secondAgent = DefensiveABAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class ABPruningCaptureAgent(CaptureAgent):
    """
    Based off of ReflexCaptureAgent, this agent uses alpha-beta pruning
    to find the best move within a set horizon, taking into account
    enemy agent moves.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        """
        super().registerInitialState(gameState)

        # Initialize a list that can be accessed by the agent later
        # Consisting of itself and its two opponents
        # before, we could assume pacman was 0 and ghosts were 1,2
        # this is no longer true!
        self.agentAndEnemiesIndices = []
        self.agentAndEnemiesIndices.append(self.index)
        opponents = self.getOpponents(gameState)
        for opp in opponents:
            self.agentAndEnemiesIndices.append(opp)
        self.depth = 2  # seems like we cannot handle depth 3+
        self.startAlpha = float('-inf')
        self.startBeta = float('inf')
        self.totalTime = gameState.getTimeleft()

    def chooseAction(self, gameState):
        """
        Use AB pruning to find the best action.
        """
        legalMoves = gameState.getLegalActions(self.index)
        successors = [self.getSuccessor(gameState, action, self.index) for action in legalMoves]
        scores = []
        # for each successor, get the AB Pruning score
        # Simulate our agent acting by first looking at a successor
        # start ABPruning at 1 to simulate enemy agents acting after our agent "acts"
        for successor in successors:
            scores.append(self.ABPrune(successor, self.depth, 1, self.startAlpha, self.startBeta))
        # find the best score
        bestScore = max(scores)
        # explanation: a at start means don't change the value of the retrieved item;
        # and only retrieve items that have max value
        bestActions = [a for a, s in zip(legalMoves, scores) if s == bestScore]
        # return the best move or one of them at random if there's a tie
        return random.choice(bestActions)

    def ABPrune(self, gameState, depth, indexInAgentsList, a, b):
        # get the index of the agent we're currently simulating
        agentIndex = self.agentAndEnemiesIndices[indexInAgentsList]
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [self.getSuccessor(gameState, action, agentIndex) for action in legalMoves]
        # if state is terminal (no successors) or max depth reached
        if depth == 0 or len(successors) <= 0:
            return self.evaluate(gameState)  # return state utility using eval function
        if indexInAgentsList == 0:  # if agent is friendly (Maximizer)
            value = float('-inf')
            for successor in successors:
                value = max(value, self.ABPrune(successor, depth, indexInAgentsList + 1, a, b))
                if value >= b:
                    return value
                a = max(a, value)
            return value
        else:  # if agent is enemy (Minimizer)
            value = float('inf')
            nextAgentIndex = indexInAgentsList + 1
            nextDepth = depth
            if nextAgentIndex >= len(self.agentAndEnemiesIndices):
                nextAgentIndex = 0  # loop back to pacman
                nextDepth = depth - 1
            for successor in successors:
                value = min(value, self.ABPrune(successor, nextDepth, nextAgentIndex, a, b))
                if value <= a:
                    return value
                b = min(b, value)
            return value

    def getSuccessor(self, gameState, action, index):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(index, action)
        pos = successor.getAgentState(index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(index, action)
        else:
            return successor

    def evaluate(self, gameState):
        """
        Look ahead agents evaluate future states, not actions from the
        current state.
        Computes a linear combination of features and feature weights.
        """
        features = {
            'successorScore': self.getScore(gameState)
        }
        weights = {
            'successorScore': 1,
        }
        stateEval = sum(features[feature] * weights[feature] for feature in features)
        return stateEval


class OffensiveABAgent(ABPruningCaptureAgent):
    """
    An offensive version of the AB pruning agent.
    """
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        """
        super().registerInitialState(gameState)
        self.depth = 2
        self.stuckTime = 0

    def chooseAction(self, gameState):
        """
        Use AB pruning to find the best action.
        """
        legalMoves = gameState.getLegalActions(self.index)
        # check if we're in a stalemate
        myPos = gameState.getAgentState(self.index).getPosition()
        prevState = self.getPreviousObservation()
        if prevState is not None:
            prevPos = prevState.getAgentState(self.index).getPosition()
            if (myPos == prevPos):
                if self.stuckTime >= 10:
                    return random.choice(legalMoves)
                    self.stuckTime = 0
                else:
                    self.stuckTime += 1
        # end of stalemate checker
        # We only want to apply AB pruning with our agent and the most hostile enemy agent
        # i.e. the closest enemy
        # enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        opponents = self.getOpponents(gameState)
        closestEnemyIndex = opponents[0]
        minDist = float('inf')
        for oppIndex in opponents:
            enemy = gameState.getAgentState(oppIndex)
            dist = self.getMazeDistance(myPos, enemy.getPosition())
            if dist < minDist:
                minDist = dist
                closestEnemyIndex = oppIndex
        self.agentAndEnemiesIndices = [self.index, closestEnemyIndex]
        # End of updating closest enemy index

        successors = [self.getSuccessor(gameState, action, self.index) for action in legalMoves]
        scores = []
        # Simulate our agent acting by first looking at a successor
        # start ABPruning at 1 to simulate enemy agents acting after our agent "acts"
        for successor in successors:
            scores.append(self.ABPrune(successor, self.depth, 1, self.startAlpha, self.startBeta))
        bestScore = max(scores)
        # explanation: a at start means don't change the value of the retrieved item;
        # and only retrieve items that have max value
        bestActions = [a for a, s in zip(legalMoves, scores) if s == bestScore]
        return random.choice(bestActions)

    def evaluate(self, gameState):
        """
        Look ahead agents evaluate future states, not actions from the
        current state.
        Computes a linear combination of features and feature weights.
        """
        return self.offensiveEval(gameState)

    def offensiveEval(self, gameState):
        features = {
            'successorScore': self.getScore(gameState)
        }
        weights = {
            'successorScore': 100,
            'distanceToFood': 10,
            'numberOfGhosts': -1000,
            'distanceToCapsule': 20,
            'numCapsules': -100,
            'frozen': -10,
            'attacking': 1,
        }
        myState = gameState.getAgentState(self.index)
        myPos = gameState.getAgentState(self.index).getPosition()
        prevState = self.getPreviousObservation()
        if prevState is not None:
            prevPos = prevState.getAgentState(self.index).getPosition()
            if (myPos == prevPos):
                features['frozen'] = 1  # disincentivize freezing up

        features['attacking'] = 0
        if (myState.isPacman()):
            features['attacking'] = 1

        # allies = [gameState.getAgentState(i) for i in self.getTeam(gameState) if i != self.index]
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        # invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        # Compute distance to the nearest food.
        foodList = self.getFood(gameState).asList()
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = 1 / minDistance**2 + 1
        
        capsuleList = self.getCapsules(gameState)
        if (len(capsuleList) > 0):
            minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            features['distanceToCapsule'] = 1 / minDistance + 1  # try to go for capsules more
        features['numCapsules'] = len(capsuleList)  # incentivize eating capsules

        features['numberOfGhosts'] = len(ghosts)  # incentivizes us to reduce num ghosts

        stateEval = sum(features[feature] * weights[feature] for feature in features)
        return stateEval
        

class DefensiveABAgent(ABPruningCaptureAgent):
    """
    An defensive version of the AB pruning agent.
    """
    
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.
        """
        super().registerInitialState(gameState)
        self.depth = 2
        self.stuckTime = 0
        
    def chooseAction(self, gameState):
        """
        Use AB pruning to find the best action.
        """
        legalMoves = gameState.getLegalActions(self.index)
        myPos = gameState.getAgentState(self.index).getPosition()
        # Defender should only AB prune on the closest invader
        # enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        opponents = self.getOpponents(gameState)
        closestEnemyIndex = opponents[0]
        minDist = float('inf')
        for oppIndex in opponents:
            enemy = gameState.getAgentState(oppIndex)
            if enemy.isPacman() and enemy.getPosition is not None:
                dist = self.getMazeDistance(myPos, enemy.getPosition())
                if dist < minDist:
                    minDist = dist
                    closestEnemyIndex = oppIndex
        self.agentAndEnemiesIndices = [self.index, closestEnemyIndex]
        # End of updating closest enemy index

        successors = [self.getSuccessor(gameState, action, self.index) for action in legalMoves]
        scores = []
        # Simulate our agent acting by first looking at a successor
        # start ABPruning at 1 to simulate enemy agents acting after our agent "acts"
        for successor in successors:
            scores.append(self.ABPrune(successor, self.depth, 1, self.startAlpha, self.startBeta))
        bestScore = max(scores)
        # explanation: a at start means don't change the value of the retrieved item;
        # and only retrieve items that have max value
        bestActions = [a for a, s in zip(legalMoves, scores) if s == bestScore]
        return random.choice(bestActions)

    def evaluate(self, gameState):
        """
        Look ahead agents evaluate future states, not actions from the
        current state.
        Computes a linear combination of features and feature weights.
        """
        return self.defensiveEval(gameState)
    
    def defensiveEval(self, gameState):
        
        features = {
            'distanceToEnemies': 0
        }
        weights = {
            'distanceToEnemies': 1,
            'distanceToInvader': 100,
            'onDefense': 1000,
            'numInvaders': -1000,
            'frozen': -1
        }
        # idea: defensive should be able to hunt down nearby scared ghosts

        myState = gameState.getAgentState(self.index)
        myPos = gameState.getAgentState(self.index).getPosition()
        prevState = self.getPreviousObservation()
        if prevState is not None:
            prevPos = prevState.getAgentState(self.index).getPosition()
            if (myPos == prevPos):
                features['frozen'] = 1  # disincentivize freezing up

        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        # if there are no invaders, try to minimize distance between us and enemies
        features['distanceToEnemies'] = 0
        features['distanceToInvader'] = 0
        if (len(invaders) > 0):  # if there's an invader, they take priority
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['distanceToInvader'] = 1 / (min(dists) + 1)
            # incentivize agent to reduce the number of invaders by eating them
        else:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            for dist in dists:
                features['distanceToEnemies'] += 1 / (dist + 1)

        stateEval = sum(features[feature] * weights[feature] for feature in features)
        return stateEval
        
