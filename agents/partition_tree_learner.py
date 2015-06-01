__author__ = 'alex'
from learner import Learner
import numpy
from tools import utilities

class Partition_Tree_Learner(Learner):
    """
    Re-implementation of Anna's Partition Tree Learning

    Theoretical Properties:

    """

    def __init__(self, depth, learner_factory, weight_reset=True):
        self.learner_factory = learner_factory
        self.depth = depth
        self.prior = numpy.array([2 ** -min(i, self.depth) for i in range(self.depth+1, 0, -1)])
        self.log_prior = numpy.log(self.prior)
        self.weights = None
        self.weight_reset = weight_reset
        self.reset_all_nodes()

    def get_max_height(self):
        # Get the height of the maximum completed subtree during this step
        if self.num_steps > 0:
            return utilities.mscb(self.num_steps-1)
        else:
            return -1

    def predict(self, x):
    # the order matters---learners are updated in get_learner_weighting
        p = self.get_learner_predictions(x)
        self.w = self.get_learner_weighting()
        return self.w.dot(p)

    def update(self, features, target):
        """
        Update the PTL model given the input
        After the update, the nodes will have up to their full number of steps
        """
        # ANNOTATE
        max_height = self.get_max_height()

        if max_height > self.depth:
            raise NotImplementedError("Must set depth less than: "+ str(self.depth) + ". Currently is: " + str(max_height))

        self.num_steps += 1

        # go over all the nodes and update each one
        for number_parameters in self.nodes:
            number_parameters.update(features, target) # I presume target is a psuedo  reward

        # total error is defined as the error at the most complete node
        self.total_loss = self.nodes[self.depth].total_loss

    def reset_all_nodes(self):
        """
        This does a complete reset of the nodes
        """
        self.nodes = [Partition_Tree_Learner_Node(self.learner_factory, 0, weight_reset=self.weight_reset)]
        for i in range(self.depth):
            self.nodes.append(Partition_Tree_Learner_Node(self.learner_factory, i+1,
                                      child=self.nodes[i],
                                      weight_reset=self.weight_reset))
        self.reset_loss()

    def get_partial_totals(self):
        #returns a list of the total loss for each node
        return [number_parameters.total_loss for number_parameters in self.nodes[::-1]]

    def get_completed_totals(self):
        #returns a list of the completed subtree nodes
        return [number_parameters.prev_loss for number_parameters in self.nodes[::-1]]

    def get_learner_predictions(self, x):
        #returns a list of each learners predictions
        return [number_parameters.predict(x) for number_parameters in self.nodes]

    def get_learner_weighting(self, debug=False):
        """
        Return the normalized weights for each of the learners
        """
        if debug:
            print("Squash it!")
        wc = numpy.cumsum(self.get_completed_totals())
        wp = self.get_partial_totals()

        # back in the default order
        w = (wc+wp)[::-1]

        loss = w - self.log_prior
        norm_w = numpy.exp(-loss - utilities.log_sum_exp(-loss))
        return norm_w

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



class Partition_Tree_Learner_Node(object):
    """
    Used by PTL to keep track of a specific binary partition point
    """

    def __init__(self, learner_factory, height, child=None, weight_reset=True):
        self.height = height
        self.max_steps = 2**height
        self.learner_factory = learner_factory
        self.total_loss = 0.0 # the total loss for this period
        self.learner = None
        self.weight_reset = weight_reset
        self.child = child
        self.num_steps = 0
        self.reset()

    def reset_node(self):
        # reset the learner and store the completed loss
        self.prev_loss = self.total_loss

        if not self.learner: # if the learner is not instantiated
            self.learner = self.learner_factory()
        else:
            self.learner.reset_loss()
            if self.weight_reset: # some instances don't need to be reset
                self.learner.set_weights()

        self.total_loss = self.learner.total_loss # pull the loss up from the learner

    def reset_loss(self):
        self.total_loss = 0.0
        self.num_steps = 0

    def reset(self):
        # Reset all the things
        self.reset_node()
        self.reset_loss()
        self.prev_loss = 0

    def calculate_loss(self):
        # ANNOTATE : HOW DOES THIS WORK?
        if not self.child:
            return self.learner.total_loss
        else:
            nosplit = -self.learner.total_loss
            split = -self.child.prev_loss - self.child.total_loss
    # what is NP and log_sum_exp
            return numpy.log(2) - log_sum_exp([nosplit, split])

    def update(self, features, target):
        if self.check_partition_end():
            self.reset_node()
            if self.child:
                self.child.reset_completed()

        self.learner.update(features, target)
        self.total_loss = self.calculate_loss()
        self.num_steps = self.learner.num_steps

    def reset_completed(self): # WHAT IS THIS FOR?
        self.prev_loss = 0.0

    def check_partition_end(self):
        return self.num_steps >= self.max_steps

    def set_weights(self, weights):
        self.learner.set_weights(weights)

    def predict(self, features):
        return self.learner.predict(features)

