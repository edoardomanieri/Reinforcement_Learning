from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
import gym
import numpy as np
import random
import keras.backend as K
from keras.layers.core import Lambda
import plotting

EPISODES_TO_TRAIN = 20

env = gym.make("Taxi-v2").env

def custom_loss(batch_qvals, batch_states):

    def loss(y_true, y_pred):
        log_prob = K.log(y_pred)[:len(batch_states),:]
        argmax_flat = K.argmax(y_true, axis=1) + [env.action_space.n * _ for _ in range(len(batch_states))]
        log_prob = K.reshape(log_prob, (env.action_space.n*len(batch_states),))
        log_prob = K.gather(log_prob,argmax_flat)
        log_prob_action = K.cast(batch_qvals, K.floatx()) * log_prob
        return -K.mean(log_prob_action)

    return loss

def calc_qvals(rewards, gamma):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

inputs = Input(shape=(1,))
x = Dense(128, activation='relu', kernel_initializer='zero')(inputs)
x = Dense(128, activation='relu', kernel_initializer='zero')(x)
predictions = Dense(env.action_space.n, activation='softmax', kernel_initializer='zero')(x)

model = Model(inputs, predictions)

total_rewards = []
step_idx = 0
done_episodes = 0

batch_episodes = 0
batch_states, batch_actions, batch_qvals = [], [], []
cur_rewards = []
episodes = 100
gamma = 0.6

stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(episodes),
        episode_rewards=np.zeros(episodes))

for i in range(1, episodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False


    while not done:

        action = np.random.choice([a for a in range(env.action_space.n)], p=model.predict([int(state)]).squeeze())
        next_state, reward, done, info = env.step(action)
        batch_states.append(state)
        batch_actions.append(action)
        cur_rewards.append(reward)

        stats.episode_rewards[i] += reward
        stats.episode_lengths[i] = epochs

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    print("episode", i)
    batch_qvals.extend(calc_qvals(cur_rewards, gamma))
    cur_rewards.clear()
    batch_episodes += 1

    if batch_episodes < EPISODES_TO_TRAIN:
        continue

    y = np_utils.to_categorical(batch_actions, env.action_space.n)
    model.compile(optimizer = Adam(), loss = custom_loss(batch_qvals, batch_states))
    model.fit(x=np.array(batch_states).reshape(-1,1), y= np.array(y).reshape(-1,6), batch_size=len(batch_states), epochs=3)
    batch_episodes = 0
    batch_states.clear()
    batch_actions.clear()
    batch_qvals.clear()

plotting.plot_episode_stats(stats)
