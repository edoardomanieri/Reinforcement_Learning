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
import tensorflow as tf


class Critic():

    def __init__(self,learning_rate, batch_size, epochs, input_shape, output_shape):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()


    def copy(self):
        return Critic(self.learning_rate, self.batch_size, self.input_shape, self.output_shape)

    def build_model(self):
        with self.graph.as_default():
            with self.session.as_default():
                critic_inputs = Input(shape=(self.input_shape,))
                x = Dense(128, activation='relu')(critic_inputs)
                critic_outputs = Dense(self.output_shape, activation='linear')(x)
                model = Model(critic_inputs, critic_outputs)
                model.compile(optimizer = Adam(lr=self.learning_rate), loss = "mse")
                return model

    def fit(self,rewards_np, input_states):
        with self.graph.as_default():
            with self.session.as_default():
                critic_y = np.array(rewards_np)
                self.model.fit(x=np.array(input_states).reshape(-1,self.input_shape), y= critic_y.reshape(-1,self.output_shape), \
                                                                batch_size=self.batch_size, epochs=self.epochs, verbose = 0)
    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y
