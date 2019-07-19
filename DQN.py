from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
import gym
import numpy as np
import random

env = gym.make("Taxi-v2").env

inputs = Input(shape=(1,))
x = Dense(32, activation='relu')(inputs)
predictions = Dense(env.action_space.n, activation='linear')(x)

model = Model(inputs, predictions)
model.compile(optimizer = Adam(), loss = 'mse')
#model.predict(np.array([10]))

#####All this can be avoided#####
#dictionary with states as keys and a tuple of as many values as actions, as values
q_table = {}

# Hyperparameters
alpha, alpha_in = 0.4, 0.4
gamma = 0.6
epsilon, epsilon_in = 0.3, 0.3
episodes = 50000

for i in range(1, episodes):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if state not in q_table.keys():
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action)

        #update dictionary
        if state in q_table.keys():
            old_value = q_table[state][action]
        else:
            old_value = 0

        if next_state in q_table.keys():
            next_max = np.max(q_table[next_state])
        else:
            next_max = 0

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        #update the dict
        if state in q_table.keys():
            old_list = list(q_table[state])
        else:
            old_list = [0 for _ in range(env.action_space.n)]

        old_list[action] = new_value
        q_table[state] = tuple(old_list)

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

model.fit(x=np.array(list(q_table.keys())), y = np.array(list(q_table.values())), epochs= 5)
########################################################



#####Algorithm with value-function approximation

# Hyperparameters
alpha, alpha_in = 0.4, 0.4
gamma = 0.6
epsilon, epsilon_in = 0.9, 0.9
episodes = 250

# For plotting metrics
all_epochs = []
all_penalties = []

#stats = plotting.EpisodeStats(
#        episode_lengths=np.zeros(episodes),
#        episode_rewards=np.zeros(episodes))

model.reset_states()
x_batch = np.array([])
y_batch = np.array([])
episodes_batch = []
for i in range(1, episodes):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    reward_per_episodio = 0

    while not done:
        old_values = model.predict([int(state)]).squeeze()
        #print(old_values)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(old_values) # Exploit learned values

        next_state, reward, done, info = env.step(action)
        reward_per_episodio += reward

        old_value = old_values[action]
        next_max = np.max(model.predict([int(next_state)]).squeeze())

        #stats.episode_rewards[i] += reward
        #stats.episode_lengths[i] = epochs

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        old_values[action] = new_value

        if len(x_batch) == 0:
            x_batch = np.array([int(state)])
            y_batch = old_values
        else:
            np.append(x_batch, [int(state)], axis=0)
            np.append(y_batch, old_values, axis=0)

        #successive calls to fit will incrementally train the model

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    episodes_batch.append([x_batch, y_batch, reward_per_episodio])
    x_batch = np.array([])
    y_batch = np.array([])

    if i % 10 == 0:
        rewards = list(map(lambda s: s[2], episodes_batch))
        x = list(map(lambda x: x[0], list(filter(lambda x: x[2] >= np.percentile(rewards, 50), episodes_batch))))
        y = list(map(lambda x: x[1], list(filter(lambda x: x[2] >= np.percentile(rewards, 50), episodes_batch))))
        model.fit(x = np.array(x).reshape(-1,1), y = np.array(y).reshape(-1,6), epochs = 1)
        epsilon *= epsilon_in

    if i % 1 == 0:
        #clear_output(wait=True)
        print(f"Episode: {i}")
        print(f"Alpha: {alpha}")
        print(f"epsilon: {epsilon}")
        print(f"gamma: {gamma}")

    if i % 1000 == 0:
        alpha *= alpha_in
        epsilon *= epsilon_in
        print(f"Alpha: {alpha}")
        print(f"epsilon: {epsilon}")
        print(f"gamma: {gamma}")

print("Training finished.\n")
