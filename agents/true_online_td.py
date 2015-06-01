__author__ = 'alex'

from tiles.tiles import CollisionTable
from td_lambda import TDLambdaLearner
from tools.verifier import Verifier

class trueOnlineTD(TDLambdaLearner):
    """
        True online TD implementation
            * Has Dutch traces
    """
    def __init__(self, numTilings = 2, parameters = 2, rlAlpha = 0.5, rlLambda = 0.9, rlGamma = 0.9, cTableSize=0):
        self.numTilings = numTilings
        self.tileWidths = list()
        self.parameters = parameters
        self.rlAlpha = rlAlpha
        self.rlLambda = rlLambda
        self.rlGamma = rlGamma

        self.prediction = None
        self.lastS = None
        self.lastQ = None
        self.lastPrediction = None
        self.lastReward = None
        self.delta = None
        self.F = [0 for item in range(self.numTilings)]
        self.F2 = [0 for item in range(self.numTilings)]
        self.theta = [0 for item in range((self.numTilings**(self.parameters+1))+1)]
        self.cTable = CollisionTable(cTableSize, 'safe')
        self.update(None, None)
        self.e = [0 for item in range((self.numTilings**(self.parameters+1))+1)]
        self.verifier = Verifier(self.rlGamma)

    def update(self, features, target=None):
        if features != None:
            self.learn(features, target)
            return self.prediction
        else: return None

    def learn(self, state, reward):

        self.loadFeatures(state, self.F)
        self.currentq = 0
        for i in self.F: # create V(s)
            self.currentq += self.theta[i]

        if self.lastS != None:
            self.delta = reward + self.rlGamma * self.currentq - self.lastQ # create delta

            self.loadFeatures(self.lastS, self.F2)
            lastQ_2 = 0
            for i in self.F2:
                lastQ_2 += self.theta[i] # create new

            for i in range(len(self.e)):
                self.e[i] *= self.rlGamma*self.rlGamma
            ephi = 0
            for i in self.F2:
                ephi += self.e[i]

            for i in self.F2:
                self.e[i] += self.rlAlpha*(1-self.rlGamma*self.rlLambda*ephi)

            for i in self.F2:
                self.theta[i] += self.rlAlpha*(self.lastQ - lastQ_2)

            for i in range(len(self.theta)):
                self.theta[i] += self.delta*self.e[i]

        self.lastQ = self.currentq
        self.lastS = state
        self.prediction = self.currentq
        self.num_steps+=1
        self.verifier.updateReward(reward)
        self.verifier.updatePrediction(self.prediction)
