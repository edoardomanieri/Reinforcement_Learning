from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2
import gym
import numpy as np
import random
import keras.backend as K
from keras.layers.core import Lambda
import plotting
import collections, copy
import os


class Actor():
    def __init__(self,learning_rate, batch_size, input_shape, output_shape, entropy_beta):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.entropy_beta = entropy_beta

    def policy_loss(self, adv, states):

        def loss(y_true, y_pred):
            log_prob = K.log(y_pred)[:len(states),:]
            entropy = -K.mean(K.sum(y_pred * log_prob, axis = 1))
            entropy_loss = -self.entropy_beta * entropy
            argmax_flat = K.argmax(y_true, axis=1) + [self.output_shape * _ for _ in range(len(states))]
            log_prob = K.reshape(log_prob, (self.output_shape*len(states),))
            log_prob = K.gather(log_prob,argmax_flat)
            log_prob_action = K.cast(adv, K.floatx()) * log_prob
            return -K.mean(log_prob_action) + entropy_loss

        return loss

    #Building actor model
    def build_model(self):
        actor_inputs = Input(shape=(self.input_shape,))
        x = Dense(128, activation='relu')(actor_inputs)
        actor_outputs = Dense(self.output_shape, activation='softmax')(x)
        model = Model(actor_inputs, actor_outputs)
        return model


    def fit(self, model, adv, input_states, input_actions):
        actor_y = np_utils.to_categorical(input_actions, self.output_shape)
        model.compile(optimizer = Adam(lr=self.learning_rate), loss = self.policy_loss(adv, input_states))
        model.fit(x=np.array(input_states).reshape(-1,self.input_shape),\
                            y= np.array(actor_y).reshape(-1,self.output_shape), batch_size=self.batch_size, epochs=1, verbose = 0)


class Critic():

    def __init__(self,learning_rate, batch_size, input_shape, output_shape):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_model(self):
        critic_inputs = Input(shape=(self.input_shape,))
        x = Dense(128, activation='relu')(critic_inputs)
        critic_outputs = Dense(self.output_shape, activation='linear')(x)
        model = Model(critic_inputs, critic_outputs)
        model.compile(optimizer = Adam(lr=self.learning_rate), loss = "mse")
        return model

    def fit(self,model,rewards_np, input_states):
        critic_y = rewards_np.copy()
        model.fit(x=np.array(input_states).reshape(-1,self.input_shape), y= critic_y.reshape(-1,self.output_shape), \
                                                                batch_size=self.batch_size, epochs=50, verbose = 0)

class A2C():

    def __init__(self,env, gamma= 0.99,actor_learning_rate = 0.001, actor_batch_size = 64,critic_learning_rate = 0.01,entropy_beta = 0.01,\
                        critic_batch_size = 16,reward_steps = 50,save_rendering = False, solving_mean = 195, videopath = "."):
        self.env = gym.make(env).env
        self.gamma = gamma
        self.reward_steps = reward_steps
        self.save_rendering = save_rendering
        self.solving_mean = solving_mean
        self.videopath = videopath
        if self.env.observation_space.shape == ():
            input_shape = 1
        else:
            input_shape = self.env.observation_space.shape[0]
        self.actor = Actor(actor_learning_rate, actor_batch_size, input_shape, self.env.action_space.n, entropy_beta)
        self.critic = Critic(critic_learning_rate, critic_batch_size, input_shape, 1)
        self.initialize()

    def initialize(self):
        self.step_idx = 0
        self.done_episodes = 0
        self.states, self.actions,self.adv, self.not_done_idx, self.last_states = [], [], [], [], []
        self.discounted_rewards = []
        self.epochs = 0
        self.done = False
        self.show = False
        self.video = 0
        self.solved = False
        self.episodes = 0
        self.video_index = 0
        self.last_rewards = collections.deque()
        self.state = self.env.reset()
        self.stats = plotting.EpisodeStats(
                episode_lengths=np.zeros(10000),
                episode_rewards=np.zeros(10000))
        if self.save_rendering:
            self.rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(self.env, path=self.videopath + "/video1.mp4")
        self.actor_model = self.actor.build_model()
        self.critic_model = self.critic.build_model()




    def unroll_bellman(self,rewards):
        sum_r = 0.0
        while not len(rewards) == 0:
            sum_r *= self.gamma
            sum_r += rewards.popleft()
        return sum_r

    def check_if_solved(self):
        if self.episodes > 100 and self.episodes % 100 == 0:
            print("episode " + str(self.episodes) + " mean last 100: ", np.mean(self.stats.episode_rewards[self.episodes-100:self.episodes]))
            if np.mean(self.stats.episode_rewards[self.episodes-100:self.episodes]) >= self.solving_mean:
                self.solved = True
                print("solved")
            if self.episodes % 300 == 0 and self.save_rendering:
                self.video_index += 1
                if not os.path.isfile(self.videopath + "/video" + str(self.video_index) + ".mp4"):
                    os.mknod(self.videopath + "/video" + str(self.video_index) + ".mp4")
                    self.rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(self.env, path=self.videopath + "/video" + str(self.video_index) + ".mp4")
                    self.show = True



    def step(self, idx):
        #choosing action according to probability distribution
        if isinstance(self.state, int):
            self.state = np.array(self.state)
        action = np.random.choice([a for a in range(self.env.action_space.n)], \
                                    p=self.actor_model.predict(self.state.reshape(-1,self.actor.input_shape)).squeeze())
        next_state, reward, self.done, info = self.env.step(action)
        self.states.append(self.state)
        self.actions.append(action)
        self.last_rewards.append(reward)

        self.stats.episode_rewards[self.episodes] += reward
        self.stats.episode_lengths[self.episodes] = self.step_idx

        #if we have enough steps to unroll the bellman equation
        if self.step_idx >= self.reward_steps - 1:
            self.discounted_rewards.append(self.unroll_bellman(copy.copy(self.last_rewards)))
            self.last_rewards.popleft()

        #identifying states for which we need to add the value function
        if not self.done and idx not in [self.actor.batch_size - j for j in range(1,self.reward_steps)]:
            self.not_done_idx.append(idx)
            self.last_states.append(np.array(self.state, copy=False))

        self.step_idx += 1
        return next_state

    def ending_episode(self):
        if self.show:
            self.show = False
            self.env.close()
            self.rec.close()

        #if the episode has ended we need to remove the final states b\c they don't need the bellman approximation
        if len(self.not_done_idx) > self.reward_steps:
            self.not_done_idx = self.not_done_idx[:len(self.not_done_idx) - self.reward_steps]
            self.last_states = self.last_states[:len(self.last_states) - self.reward_steps]
        else:
            self.not_done_idx = []
            self.last_states = []

        #reset environment
        self.state = self.env.reset()
        self.step_idx = 0
        self.episodes += 1
        self.check_if_solved()

        #Unroll all the rewards since the episode has ended
        while not len(self.last_rewards) == 0:
            self.discounted_rewards.append(self.unroll_bellman(copy.copy(self.last_rewards)))
            self.last_rewards.popleft()

    def ending_batch(self):
        self.epochs += 1

        #keep only ACTOR_BATCH_SIZE rewards
        exceeding_rewards = self.discounted_rewards[self.actor.batch_size:]
        self.discounted_rewards = self.discounted_rewards[:self.actor.batch_size]
        rewards_np = np.array(self.discounted_rewards, dtype=np.float32)

        #Use the critic model to estimate value function
        if self.not_done_idx and self.step_idx > self.reward_steps:
            last_vals = self.critic_model.predict(np.array(self.last_states).reshape(-1,self.actor.input_shape)).squeeze()
            rewards_np[self.not_done_idx] += self.gamma ** self.reward_steps * last_vals
        return rewards_np, exceeding_rewards

    def compute_advantages(self,rewards_np,exceeding_rewards, input_states):

        #putting the exceeding_rewards in the new batch
        self.discounted_rewards = exceeding_rewards
        adv = rewards_np - self.critic_model.predict(np.array(input_states).reshape(-1,self.actor.input_shape)).squeeze()
        return adv



    def free_memory(self, *args):
        for arg in args:
            del arg

    def a2c_learning(self):
        while not self.solved:
            for idx in range(self.actor.batch_size):
                if self.show:
                    self.env.render(mode='rgb_array')
                    self.rec.capture_frame()

                tmp = self.step(idx)
                if self.done:
                    self.ending_episode()
                    if self.episodes == 10000:
                        break
                else:
                    #update the state
                    self.state = tmp

            rewards_np, exceeding_rewards = self.ending_batch()
            input_states = self.states[:len(self.discounted_rewards)]
            input_actions = self.actions[:len(self.discounted_rewards)]
            self.states = self.states[len(self.discounted_rewards):]
            self.actions = self.actions[len(self.discounted_rewards):]
            adv = self.compute_advantages(rewards_np, exceeding_rewards, input_states)
            self.critic.fit(self.critic_model,rewards_np, input_states)
            self.actor.fit(self.actor_model,adv, input_states, input_actions)

            #Free some memory in order to help long training sessions
            self.free_memory(input_states, input_actions, exceeding_rewards, adv, rewards_np)

cartPole = A2C("CartPole-v1")

cartPole.a2c_learning()
