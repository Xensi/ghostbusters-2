def defensiveEval(self, gameState):
    features = {
        'distanceToEnemies': 0
    }
    weights = {
        'distanceToEnemies': 1,
        'distanceToInvader': 100,
        'onDefense': 1,
        'numInvaders': -1000,
        'frozen': -10,  # Disincentivize freezing up
        'alive': 10,
        'distanceBetweenEnemyAndFood': 100,
        'highRiskFoodProximity': -50  # added defending at-risk food
    }

    myState = gameState.getAgentState(self.index)
    myPos = gameState.getAgentState(self.index).getPosition()
    prevState = self.getPreviousObservation()
    prevPos = None
    if prevState is not None:
        prevPos = prevState.getAgentState(self.index).getPosition()
        if myPos == prevPos:
            features['frozen'] = 1  # Penalize staying in place

    features['onDefense'] = 1 if not myState.isPacman() else 0

    if prevPos is not None:
        killed = prevPos != myPos and myPos == self.startPos
        features['alive'] = 0 if killed else 1

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    foodList = self.getFoodYouAreDefending(gameState).asList()  # Defended food

    # Keep enemies away from food
    features['distanceBetweenEnemyAndFood'] = sum(
        min(self.getMazeDistance(enemy.getPosition(), food) for food in foodList)
        for enemy in enemies if enemy.getPosition() is not None
    )

    # Find high-risk food (food closest to enemies)
    if foodList:
        highRiskFood = sorted(foodList, key=lambda food: 
            min(self.getMazeDistance(food, enemy.getPosition()) for enemy in enemies if enemy.getPosition() is not None)
        )

        if highRiskFood:
            nearestHighRiskFood = highRiskFood[0]
            features['highRiskFoodProximity'] = self.getMazeDistance(myPos, nearestHighRiskFood)

    # Invader tracking
    invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
    features['numInvaders'] = len(invaders)

    if invaders:  # If invaders exist, prioritize them
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['distanceToInvader'] = 1 / (min(dists) + 1)  # Move toward closest invader
    else:  # Otherwise, minimize distance between enemies
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies if a.getPosition() is not None]
        denom = sum(d**2 for d in dists) + 1 #stay positioned more evenly between multiple enemies instead of just one
        features['distanceToEnemies'] = 1 / denom

    return sum(features[feature] * weights[feature] for feature in features)

