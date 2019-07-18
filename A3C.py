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
from threading import Thread, Lock
from env_thread import Env_thread
from Actor import Actor
from Critic import Critic
from thread import training_thread
import tensorflow as tf
import time

class A3C():

    def __init__(self,env_name, num_threads, gamma= 0.99,actor_learning_rate = 0.001, actor_batch_size = 64,critic_learning_rate = 0.01,\
            entropy_beta = 0.01,critic_batch_size = 16, max_episodes_per_thread = 100, episode_to_train= 4):
        self.envs = [gym.make(env_name).env for _ in range(num_threads)]
        if self.envs[0].observation_space.shape == ():
            input_shape = 1
        else:
            input_shape = self.envs[0].observation_space.shape[0]

        self.actor = Actor(actor_learning_rate, actor_batch_size, input_shape, self.envs[0].action_space.n, entropy_beta)
        self.critic = Critic(critic_learning_rate, critic_batch_size, input_shape, 1)
        lock = Lock()
        self.threads = [Env_thread("thread" + str(i), lock, self.envs[i], self.actor, self.critic, gamma, max_episodes_per_thread, episode_to_train) for i in range(num_threads)]


    def train(self):
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()
            print("joined")


CartPole = A3C("CartPole-v1",4, max_episodes_per_thread =100, episode_to_train=5)
CartPole.train()
