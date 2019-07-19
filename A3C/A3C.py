import gym
import numpy as np
from threading import Thread, Lock
from env_thread import Env_thread
from actor import Actor
from critic import Critic
from batch import Batch

class A3C():

    def __init__(self,env_name, num_threads, gamma= 0.99,actor_learning_rate = 0.001, actor_batch_size = 64,critic_learning_rate = 0.01,\
            entropy_beta = 0.01,critic_batch_size = 16,critic_epochs = 100, max_episodes_per_thread = 100, episode_to_train= 4):
        self.envs = [gym.make(env_name).env for _ in range(num_threads)]
        if self.envs[0].observation_space.shape == ():
            input_shape = 1
        else:
            input_shape = self.envs[0].observation_space.shape[0]

        self.actor = Actor(actor_learning_rate, actor_batch_size, input_shape, self.envs[0].action_space.n, entropy_beta)
        self.critic = Critic(critic_learning_rate, critic_batch_size,critic_epochs, input_shape, 1)
        batch = Batch(self.actor, self.critic, batch_size = actor_batch_size)
        lock = Lock()
        self.threads = [Env_thread("thread" + str(i), lock, batch, self.envs[i], self.actor, self.critic, gamma, max_episodes_per_thread, episode_to_train) for i in range(num_threads)]


    def train(self):
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()
            print("joined")
