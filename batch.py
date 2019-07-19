

class Batch():

    def __init__(self, actor, critic, batch_size = 64):
        self.actor = actor
        self.critic = critic
        self.states = []
        self.actions = []
        self.adv = []
        self.scales = []
        self.batch_size = batch_size



    def put(self, states, actions, adv, scales):
        self.states.extend(states)
        self.actions.extend(actions)
        self.adv.extend(adv)
        self.scales.extend(scales)
        if len(self.states) >= self.batch_size:
            self.fit()



    def fit(self):
        self.critic.fit(self.scales, self.states)
        self.actor.fit(self.adv, self.states, self.actions)
        self.states.clear()
        self.actions.clear()
        self.adv.clear()
        self.scales.clear()
