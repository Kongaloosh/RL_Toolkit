__author__ = 'alex'

from learner import Learner
from tools.traces import TraceHolder
from tools.verifier import Verifier
from tiles.tiles import CollisionTable, loadtiles

class TDLambdaLearner(Learner):
    """
    Note: the TileCoder is Rich's Python version, which is still in Alpha.
    See more at: http://webdocs.cs.ualberta.ca/~sutton/tiles2.html#Python%20Versions

        Collision Table notes:
            cTableSize is the size that the collision table will be instantiated to. The size must be  a power of two.
            In calls for get tiles, the collision table is used in stead of memory_size, as it already has it.

    """
    def __init__(self, numTilings = 1, parameters = 2, rlAlpha = 0.5, rlLambda = 0.9, rlGamma = 0.9, cTableSize=0):
        """ If you want to run an example of the code, simply just leave the parameters blank and it'll automatically set based on the parameters. """
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
        self.traceH = TraceHolder((self.numTilings**(self.parameters)+1), self.rlLambda, 1000)
        self.F = [0 for item in range(self.numTilings)] # the indices of the returned tiles will go in here
        self.theta = [0 for item in range((self.numTilings**(self.parameters+1))+1)] # weight vector.
        self.cTable = CollisionTable(cTableSize, 'safe') # look into this...
        self.verifier = Verifier(self.rlGamma)


    def update(self, features, target=None):
        if features != None:
            self.learn(features, target)
            return self.prediction
        else: return None


    def learn(self, state, reward):
        self.loadFeatures(state, self.F)
        currentq = self.computeQ()
        if self.lastS != None:
            self.delta = reward - self.lastQ
            self.delta += self.rlGamma * currentq
            amt = self.delta * (self.rlAlpha / self.numTilings)
            for i in self.traceH.getTraceIndices():
                self.theta[i] += amt * self.traceH.getTrace(i)
            self.traceH.decayTraces(self.rlGamma)
            self.traceH.replaceTraces(self.F)
        self.lastQ = currentq
        self.lastS = state
        self.prediction = currentq
        self.num_steps+=1
        self.verifier.updateReward(reward)
        self.verifier.updatePrediction(self.prediction)


    def computeQ(self):
        q = 0
        for i in self.F:
            q += self.theta[i]
        return q


    def loadFeatures(self, stateVars, featureVector):
        loadtiles(featureVector, 0, self.numTilings, self.numTilings**(self.parameters), stateVars)
        print "featureVector " + str(len(self.theta))
        """
        As provided in Rich's explanation
               tiles                   ; a provided array for the tile indices to go into
               starting-element        ; first element of "tiles" to be changed (typically 0)
               num-tilings             ; the number of tilings desired
               memory-size             ; the number of possible tile indices
               floats                  ; a list of real values making up the input vector
               ints)                   ; list of optional inputs to get different hashings
        """


    def loss(self, x, r, prev_state=None):
        """
        Returns the TD error assuming reward r given for
        transition from prev_state to x
        If prev_state is None will use leftmost element in exp_queue
        """
        if prev_state is None:
            if len(self.exp_queue) < self.horizon:
                return None
            else:
                prev_state = self.exp_queue[0][0]

        vp = r + self.gamma * self.value(x)
        v = self.value(prev_state)
        delta = vp - v
        return delta


    def predict (self,x):
        self.loadFeatures(x, self.F)
        return self.computeQ()

