from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.custom.AttackAgent',
        second = 'pacai.agents.capture.custom.DefenseAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = AttackAgent
    secondAgent = DefenseAgent
    # firstAgent = reflection.qualifiedImport(first)
    # secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
#####  DEFENSE EVALUATION FUNCTION ########
def defense_evaluation(self, currentGameState):
    # Get the current position of the agent
    myState = currentGameState(self.index)
    myPosition = myState.getPosition()

    #intialize the score
    score = 0
    #weights for each factor
    enemyweight = 10
    capsuleweight = 5
    foodweight = 1
    #First Factor: Number of Nearby Enemies based on their distance
    enemies = self.getOpponents(currentGameState)
    enemyProximity = 0
    for enemy in enemies:
        enemyPos = currentGameState.getAgentState(enemy).getPosition()
        if enemyPos is not None: #if there are enemies
            distance = self.getMazeDistance(myPos, enemyPos) #get the distance between enemy and me
            score += ( 1 / (distance ** 2 + 1) ) * enemyweight
            ##the following lines are another implementation that I think is worse
            #if distance <= 5: #how close should it be? #
            #    enemyProximity += 1
            ### Another thought would be to more heavily weigh closer, and less heavily further
            # if distance <= 2:
            #   enemyProximity += 1
            # if distance <= 5:
            #   enemyProxomity += 1
            # if distance <= 10:
            #   enemyProximity += 1

    #for each proximate enemy, the score increases by some amount of weight
    #weight = 10
    #score += enemyProximity * weight

    #Second Factor: How close we are to power pellets
    capsules = self.getCapsules()
    if capsules: #if there are available capsules
        nearestCapsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsules)])
        score += (1 / (nearestCapsuleDist ** 2 + 1)) * capsuleweight

    #Third Factor: Proximity to defended food
    DefendFood = self.getFoodYouAreDefending(currentGameState)
    if DefendFood: #if there is some food to defend
        foodPositions = DefendFood.asList()
        nearestFoodDist = min([self.getMazeDistance(myPos, food) for food in foodPositions])
        score += (1 / (nearestFoodDist ** 2 + 1)) * foodweight

    return score

class AttackAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        # Defines agent priorities
        # Other things offensive agent could care about:
        # Closest scared enemy ghost
        # Distance to power pellet
        # Bias towards clusters of food
        # If we're winning by a lot, defend?
        # If there's not much time left, attack?
        # Get num capsules from now to successor;
        # Reduced capsules is good
        features = {}
        
        # Stopping is bad
        if (action == Directions.STOP):
            features['stop'] = float('-inf')
            return features

        # Get the successor game state for taking an action
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        # getScore returns diff between your score
        # and their score; This number is negative if you are losing
        # Causes the agent to favor actions that increase the score
        features['successorScore'] = self.getScore(successor)

        # List of food we can eat
        foodList = self.getFood(successor).asList()
        
        # capsules = self.getCapsules(successor)
        
        myPos = successor.getAgentState(self.index).getPosition()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            # Get position in successor
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            # Higher number with lower distance
            features['distanceToFood'] = minDistance
        # Above causes agent to favor states that are closer to food
        # Without the successorScore feature, the agent will not actually eat the food
        # Theory: This is because the food will be gone in that state and so the
        # agent will technically be further away from food as a result.

        # On home side, agent is a ghost and should attack invaders
        # On enemy turf, agent is a pacman and should normally run from ghosts
        allies = [successor.getAgentState(i) for i in self.getTeam(successor) if i != self.index]
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        # features['numInvaders'] = len(invaders)
        
        # We should try to avoid our ally so we can spread out a bit
        if (len(allies) > 0):
            minDistance = min([self.getMazeDistance(myPos, ally.getPosition()) for ally in allies])
            features['distanceToAlly'] = minDistance
        
        closeGhost = False
        # If we're a pacman, fear ghosts
        if (len(ghosts) > 0 and myState.isPacman()):
            minDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
            fearThreshold = 5
            # We need to ignore the ghost distance unless it's too close
            # Otherwise, the agent will be paralyzed in fear!
            # The higher the distance, the better
            if minDistance <= fearThreshold:
                closeGhost = True
                features['distanceToGhost'] = minDistance
            else:
                features['distanceToGhost'] = minDistance * 0.25  # Have smaller fear when not close

        # If anyone is invading and we're a ghost, let's go get em
        invaderClose = False
        if (len(invaders) > 0):
            minDistance = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
            hungerThreshold = 5
            if minDistance <= hungerThreshold:
                invaderClose = True
            if (invaderClose or not myState.isPacman()):
                features['invaderDistance'] = minDistance
        
        # Reversing is bad because we make no progress that way
        # However, if there's a ghost very close to us, we merely discourage it
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            if closeGhost:
                features['reverse'] = -50
            else:
                features['reverse'] = -100

        # print(features)
        return features

    def getWeights(self, gameState, action):
        return {
            'stop': 1,
            'reverse': 1,
            'successorScore': 1000,
            'distanceToFood': -100,
            'distanceToGhost': 500,
            'invaderDistance': -1000,
            'distanceToAlly': 1,
        }

class DefenseAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2
        }
