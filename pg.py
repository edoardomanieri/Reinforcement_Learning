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

EPISODES_TO_TRAIN = 5
TOTAL_EPISODES = 20
ENTROPY_BETA = 0.001

env = gym.make("Taxi-v2").env

def custom_loss(batch_scales, batch_states):

    def loss(y_true, y_pred):
        log_prob = K.log(y_pred)[:len(batch_states),:]
        entropy = -K.mean(K.sum(y_pred * log_prob, axis = 1))
        entropy_loss = -ENTROPY_BETA * entropy
        argmax_flat = K.argmax(y_true, axis=1) + [env.action_space.n * _ for _ in range(len(batch_states))]
        log_prob = K.reshape(log_prob, (env.action_space.n*len(batch_states),))
        log_prob = K.gather(log_prob,argmax_flat)
        log_prob_action = K.cast(batch_scales, K.floatx()) * log_prob
        return -K.mean(log_prob_action) + entropy_loss

    return loss

inputs = Input(shape=(1,))
x = Dense(256, activation='relu', kernel_initializer='zero', kernel_regularizer=l2(0.001))(inputs)
predictions = Dense(env.action_space.n, activation='softmax', kernel_initializer='zero', kernel_regularizer=l2(0.001))(x)

model = Model(inputs, predictions)

total_rewards = []
step_idx = 0
done_episodes = 0

batch_episodes = 0
batch_states, batch_actions, batch_scales = [], [], []
cur_rewards = []
episodes = 20
gamma = 1

stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(episodes),
        episode_rewards=np.zeros(episodes))


###understand and add baseline!!
for i in range(1, episodes):

    state = env.reset()
    state = state / env.observation_space.n
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:

        action = np.random.choice([a for a in range(env.action_space.n)], p=model.predict([state]).squeeze())
        next_state, reward, done, info = env.step(action)
        next_state = next_state / env.observation_space.n
        batch_states.append(state)
        batch_actions.append(action)
        cur_rewards.append(reward)
        baseline = reward / i
        batch_scales.append(reward - baseline)

        stats.episode_rewards[i] += reward
        stats.episode_lengths[i] = epochs

        state = next_state
        epochs += 1

    print("episode", i)

    cur_rewards.clear()
    batch_episodes += 1

    if batch_episodes < EPISODES_TO_TRAIN:
        continue

    y = np_utils.to_categorical(batch_actions, env.action_space.n)
    model.compile(optimizer = Adam(lr=0.0001), loss = custom_loss(batch_scales, batch_states))
    model.fit(x=np.array(batch_states).reshape(-1,1), y= np.array(y).reshape(-1,6), batch_size=len(batch_states), epochs=100, verbose = 0)
    batch_episodes = 0
    batch_states.clear()
    batch_actions.clear()
    batch_scales.clear()

plotting.plot_episode_stats(stats)
