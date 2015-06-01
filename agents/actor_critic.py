__author__ = 'alex'
from td_lambda import TDLambdaLearner
from learner import Learner
import random
import operator


class actor_critic(Learner):


    def __init__(self, actions, numTilings = 1, parameters = 2,rlAlpha = 0.5, rlLambda = 0.9,
                 rlGamma = 0.9, rlEpsilon = 0.1, cTableSize=0, action_selection = 'softmax'):
        self.td = TDLambdaLearner()
        self.beta = None
        self.actions = actions


    def update(self, features, target=None):
        self.td.update(features,target)
        action = self.chooseAction(features)
        criticism = self.beta*self.td.delta

        self.learn(features, action, criticism)


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


    def learn(self, action, criticism):
        for i in self.traceH.getTraceIndices():
            self.theta += criticism
        self.traceH.decayTraces(self.rlGamma*self.rlLambda)
        self.traceH.replaceTraces(self.F[action])
