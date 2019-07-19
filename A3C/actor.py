from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
import gym
import numpy as np
import random
import keras.backend as K
import  plotting
import collections, copy
import tensorflow as tf

class Actor():

    def __init__(self,learning_rate, batch_size, input_shape, output_shape, entropy_beta):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.entropy_beta = entropy_beta
        self.model = self.build_model()

    def copy(self):
        return Actor(self.learning_rate, self.batch_size, self.input_shape, self.output_shape, self.entropy_beta)

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
        with self.graph.as_default():
            with self.session.as_default():
                actor_inputs = Input(shape=(self.input_shape,))
                x = Dense(128, activation='relu')(actor_inputs)
                actor_outputs = Dense(self.output_shape, activation='softmax')(x)
                model = Model(actor_inputs, actor_outputs)
                return model


    def fit(self, adv, input_states, input_actions):
        with self.graph.as_default():
            with self.session.as_default():
                actor_y = np_utils.to_categorical(input_actions, self.output_shape)
                self.model.compile(optimizer = Adam(lr=self.learning_rate), loss = self.policy_loss(adv, input_states))
                self.model.fit(x=np.array(input_states).reshape(-1,self.input_shape),\
                            y= np.array(actor_y).reshape(-1,self.output_shape), batch_size=len(input_states), epochs=1, verbose = 0)

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

    def get_weights(self):
        with self.graph.as_default():
            with self.session.as_default():
                weights = self.model.get_weights()
        return weights

    def set_weights(self, weights):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.set_weights(weights)
