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
TOTAL_EPISODES = 10000
ENTROPY_BETA = 0.01

env = gym.make("CartPole-v1").env

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

def calc_qvals(rewards, gamma, discounted_reward, step_idx):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    step_idx = (step_idx - len(res)) + 1
    for i in range(len(res)):
        discounted_reward += res[i]
        baseline = discounted_reward / step_idx
        step_idx += 1
        res[i] -= baseline
    return res, discounted_reward


inputs = Input(shape=(env.observation_space.shape[0],))
x = Dense(128, activation='relu')(inputs)
predictions = Dense(env.action_space.n, activation='softmax')(x)

model = Model(inputs, predictions)

total_rewards = []
step_idx = 0
done_episodes = 0

batch_episodes = 0
batch_states, batch_actions, batch_scales = [], [], []
cur_rewards = []
gamma = 0.99

stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(TOTAL_EPISODES),
        episode_rewards=np.zeros(TOTAL_EPISODES))

model.reset_states()
discounted_reward = 0
step_idx = 0
for i in range(1, TOTAL_EPISODES):

    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:

        step_idx += 1
        action = np.random.choice([a for a in range(env.action_space.n)], p=model.predict(state.reshape(-1,env.observation_space.shape[0])).squeeze())
        next_state, reward, done, info = env.step(action)
        batch_states.append(state)
        batch_actions.append(action)

        stats.episode_rewards[i] += reward
        stats.episode_lengths[i] = epochs


        cur_rewards.append(reward)
        state = next_state
        epochs += 1

    if i % 100 == 0:
        print("episode", i)

    res, discounted_reward = calc_qvals(cur_rewards, gamma, discounted_reward, step_idx)
    batch_scales.extend(res)
    cur_rewards.clear()
    batch_episodes += 1

    if i > 100 and i % 10 == 0:
        print("mean last 100: ", np.mean(stats.episode_rewards[i-100:i]))
        if np.mean(stats.episode_rewards[i-100:i]) >= 195:
            print("solved")
            break

    if batch_episodes < EPISODES_TO_TRAIN:
        continue

    y = np_utils.to_categorical(batch_actions, env.action_space.n)
    model.compile(optimizer = Adam(lr=0.001), loss = custom_loss(batch_scales, batch_states))
    model.fit(x=np.array(batch_states).reshape(-1,env.observation_space.shape[0]), y= np.array(y).reshape(-1,env.action_space.n), batch_size=len(batch_states), epochs=3, verbose = 0)
    batch_episodes = 0
    batch_states.clear()
    batch_actions.clear()
    batch_scales.clear()
