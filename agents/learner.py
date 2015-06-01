__author__ = 'alex'


class Learner(object):
    total_loss = 0
    num_steps = 0
    prediction = 0
    theta = list()

    def update(self, features, target=None):
        pass

    def set_weights(self):
        # update the weight given weight_init
        pass

    def predict(self, x):
        # calculate the prediction
        pass

    def reset_loss(self):
        # reset accumulated loss but keep learned weights
        self.total_loss = 0.0
        self.num_steps = 0
        try:
            self.alpha = self.alpha_init
        except:
            pass

    def reset(self):
        # reset everything
        self.reset_loss()
        self.set_weights()

    def set_alpha(self, alpha, alpha_decay = None):
        self.alpha = alpha
        if alpha_decay is None or alpha_decay < 0:
            self.alpha = alpha
            self.alpha_init = alpha
            self.alpha_decay = None
        else:
            raise Exception("Set_Alpha not INITIALIZED for alpha decay")

    def loss(self):
        pass

