__author__ = 'alex'
from learner import Learner
from tools.traces import TraceHolder
from tools.verifier import Verifier
from tiles.tiles import CollisionTable, loadtiles

class sarsa(Learner):


    def __init__(actions, self, numTilings = 1, parameters = 2,rlAlpha = 0.5, rlLambda = 0.9,
                 rlGamma = 0.9, rlEpsilon = 0.1, cTableSize=0, action_selection = 'softmax'):
        """ If you want to run an example of the code, simply just leave the parameters blank and it'll automatically set based on the parameters. """
        self.numTilings = numTilings
        self.tileWidths = list()
        self.parameters = parameters
        self.rlAlpha = rlAlpha
        self.rlLambda = rlLambda
        self.rlGamma = rlGamma
        self.rlEpsilon = rlEpsilon
        self.action_selection = action_selection

        self.lastS = None
        self.lastQ = None
        self.lastPrediction = None
        self.lastReward = None
        self.lastAction = None
        self.currentAction = None

        self.actions = actions # an array of actions which we can select from
        self.traceH = TraceHolder((self.numTilings**(self.parameters)+1), self.rlLambda, 1000)
        self.F = [[0 for item in range(self.numTilings)] for i in range(actions)] # the indices of the returned tiles will go in here
        self.q_vals = [0 for i in range(actions)]
        for action in actions:
            self.q.append(action,[0 for item in range((self.numTilings**(self.parameters+1))+1)]) # action and weight vec
        self.cTable = CollisionTable(cTableSize, 'safe') # look into this...
        self.verifier = Verifier(self.rlGamma)


    def chooseAction(self, features):
        for action in range(self.actions):
            self.loadFeatures(featureVector=self.F[action], stateVars=features)
            self.q_vals[action] = self.computeQ(action)
        return self.eGreedy()


    def eGreedy(self):
        if random.random() < self.rlEpsilon:
            return random.randrange(self.actions) # random action
        else:
            max_index, max_value = max(enumerate(self.q_vals), key=operator.itemgetter(1))
            return max_index # best action


    def update(self, features, target=None):
        # learning step
        if features != None:
            self.learn(features, target, self.currentAction)

        self.lastAction = self.currentAction
        self.currentAction = self.chooseAction(features)

        # action selection step
        return self.currentAction


    def learn(self, features, reward, action):
        self.loadFeatures(features, self.F)
        currentq = self.computeQ(action)

        if self.lastS != None and self.lastAction != None: # if we're past the first step
            delta = reward - self.lastQ
            delta += self.rlGamma * currentq
            amt = delta * (self.rlAlpha / self.numTilings)
            for i in self.traceH.getTraceIndices():
                self.theta[i] += amt * self.traceH.getTrace(i)
            self.traceH.decayTraces(self.rlGamma*self.rlLambda)
            self.traceH.replaceTraces(self.F[action])

        self.lastQ = currentq
        self.lastS = features
        self.num_steps+=1
        self.verifier.updateReward(reward)
        self.verifier.updatePrediction(self.prediction)


    def loadFeatures (self, stateVars, featureVector):
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


    def computeQ (self, a):
        "compute value of action for current F and theta"
        q = 0
        for i in self.F[a]:
            q += self.theta[i]
        return q

