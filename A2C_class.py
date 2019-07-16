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


class A2C():

    def __init__(gamma= 0.99, actor_learning_rate = 0.001,critic_learning_rate = 0.01,entropy_beta = 0.01,\
                        actor_batch_size = 64,critic_batch_size = 16,reward_steps = 50,save_rendering = False, solving_mean = 100, videopath = "."):
    self.gamma = gamma
    self.actor_learning_rate = actor_learning_rate
    self.critic_learning_rate = critic_learning_rate
    self.entropy_beta = entropy_beta
    self.actor_batch_size = actor_batch_size
    self.critic_batch_size = critic_batch_size
    self.reward_steps = reward_steps
    self.save_rendering = save_rendering
    self.solving_mean = solving_mean
    self.videopath = videopath
    self.initialize()

    def initialize(self):
        self.env = gym.make("CartPole-v1").env
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
        self.state = env.reset()
        self.stats = plotting.EpisodeStats(
                episode_lengths=np.zeros(10000),
                episode_rewards=np.zeros(10000))
        if self.save_rendering:
            self.rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(self.env, path=self.videopath + "/video1.mp4")
        self.build_actor_model()
        self.build_critic_model()

    def policy_loss(self, adv, states):

        def loss(y_true, y_pred):
            log_prob = K.log(y_pred)[:len(states),:]
            entropy = -K.mean(K.sum(y_pred * log_prob, axis = 1))
            entropy_loss = -ENTROPY_BETA * entropy
            argmax_flat = K.argmax(y_true, axis=1) + [env.action_space.n * _ for _ in range(len(states))]
            log_prob = K.reshape(log_prob, (env.action_space.n*len(states),))
            log_prob = K.gather(log_prob,argmax_flat)
            log_prob_action = K.cast(adv, K.floatx()) * log_prob
            return -K.mean(log_prob_action) + entropy_loss

        return loss


    def unroll_bellman(self,rewards):
        sum_r = 0.0
        while not len(rewards) == 0:
            sum_r *= GAMMA
            sum_r += rewards.popleft()
        return sum_r

    def check_if_solved(self):
        if self.episodes > 100 and self.episodes % 10 == 0:
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


    #Building actor model
    def build_actor_model(self):
        actor_inputs = Input(shape=(self.env.observation_space.shape[0],))
        x = Dense(128, activation='relu')(actor_inputs)
        actor_outputs = Dense(self.env.action_space.n, activation='softmax')(x)
        self.actor_model = Model(actor_inputs, actor_outputs)

    #Building critic model
    def build_critic_model(self):
        critic_inputs = Input(shape=(self.env.observation_space.shape[0],))
        x = Dense(128, activation='relu')(critic_inputs)
        critic_outputs = Dense(1, activation='linear')(x)
        self.critic_model = Model(critic_inputs, critic_outputs)
        self.critic_model.compile(optimizer = Adam(lr=self.critic_learning_rate), loss = "mse")

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
        self.state = env.reset()
        self.step_idx = 0
        self.episodes += 1
        self.check_if_solved()
        #Unroll all the rewards since the episode has ended
        while not len(self.last_rewards) == 0:
            self.discounted_rewards.append(self.unroll_bellman(copy.copy(self.last_rewards)))
            self.last_rewards.popleft()



    def step(self):
        #choosing action according to probability distribution
        action = np.random.choice([a for a in range(env.action_space.n)], \
                                    p=actor_model.predict(state.reshape(-1,env.observation_space.shape[0])).squeeze())
        next_state, reward, done, info = env.step(action)
        self.states.append(state)
        self.actions.append(action)
        self.last_rewards.append(reward)

        self.stats.episode_rewards[self.episodes] += reward
        self.stats.episode_lengths[self.episodes] = self.step_idx

        #if we have enough steps to unroll the bellman equation
        if self.step_idx >= self.reward_steps - 1:
            self.discounted_rewards.append(self.unroll_bellman(copy.copy(self.last_rewards)))
            self.last_rewards.popleft()

        self.step_idx += 1
        return next_state

    def ending_batch(self):
        self.epochs += 1
        #keep only ACTOR_BATCH_SIZE rewards
        exceeding_rewards = self.discounted_rewards[self.actor_batch_size:]
        self.discounted_rewards = self.discounted_rewards[:self.actor_batch_size]
        rewards_np = np.array(self.discounted_rewards, dtype=np.float32)
        #Use the critic model to estimate value function
        if self.not_done_idx and self.step_idx > self.reward_steps:
            last_vals = critic_model.predict(np.array(self.last_states).reshape(-1,self.env.observation_space.shape[0])).squeeze()
            rewards_np[self.not_done_idx] += self.gamma ** self.reward_steps * last_vals

    def a2c_learning(self):
        while not self.solved:
            for i in range(self.actor_batch_size):
                if self.show:
                    self.env.render(mode='rgb_array')
                    self.rec.capture_frame()



                #identifying states for which we need to add the value function
                if not self.done and i not in [self.actor_batch_size - i for i in range(1,self.reward_steps)]:
                    self.not_done_idx.append(i)
                    self.last_states.append(np.array(self.state, copy=False))
                elif self.done:
                    self.ending_episode()


                #update the state
                self.state = self.step()







            input_states = self.states[:len(self.discounted_rewards)]
            input_actions = self.actions[:len(self.discounted_rewards)]
            self.states = self.states[len(discounted_rewards):]
            self.actions = self.actions[len(self.discounted_rewards):]

            #putting the exceeding_rewards in the new batch
            self.discounted_rewards = exceeding_rewards

            critic_y = rewards_np.copy()
            adv = rewards_np - critic_model.predict(np.array(input_states).reshape(-1,env.observation_space.shape[0])).squeeze()
            critic_model.fit(x=np.array(input_states).reshape(-1,env.observation_space.shape[0]), y= critic_y.reshape(-1,1), \
                                                                batch_size=CRITIC_BATCH_SIZE, epochs=50, verbose = 0)



            actor_y = np_utils.to_categorical(input_actions, env.action_space.n)
            actor_model.compile(optimizer = Adam(lr=ACTOR_LEARNING_RATE), loss = policy_loss(adv, input_states))
            actor_model.fit(x=np.array(input_states).reshape(-1,env.observation_space.shape[0]),\
                                                y= np.array(actor_y).reshape(-1,env.action_space.n), batch_size=ACTOR_BATCH_SIZE, epochs=1, verbose = 0)


            #Free some memory in order to help long training sessions
            del input_states
            del input_actions
            del exceeding_rewards
            del adv
            del rewards_np
