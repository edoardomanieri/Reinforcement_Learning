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

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
ACTOR_BATCH_SIZE = 64
CRITIC_BATCH_SIZE = 16
NUM_ENVS = 50
REWARD_STEPS = 50
SAVE_RENDERING = False

env = gym.make("CartPole-v1").env


def policy_loss(adv, states):

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


def unroll_bellman(rewards):
    sum_r = 0.0
    while not len(rewards) == 0:
        sum_r *= GAMMA
        sum_r += rewards.popleft()
    return sum_r

def check_if_solved(episodes, episodes_rewards, solving_mean, env, video):
    solved = False
    if episodes > 100 and episodes % 10 == 0:
        print("episode " + str(episodes) + " mean last 100: ", np.mean(episodes_rewards[episodes-100:episodes]))
        if np.mean(episodes_rewards[episodes-100:episodes]) >= solving_mean:
            solved = True
            print("solved")
        if episodes % 300 == 0 and SAVE_RENDERING:
            video += 1
            if not os.path.isfile("/home/edoardo/Desktop/Reinforcement_Learning/RL/videos/video" + str(video) + ".mp4"):
                os.mknod("/home/edoardo/Desktop/Reinforcement_Learning/RL/videos/video" + str(video) + ".mp4")
                rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, path="/home/edoardo/Desktop/Reinforcement_Learning/RL/videos/video" + str(video) + ".mp4")
                show = True
    return solved


#Building actor model
actor_inputs = Input(shape=(env.observation_space.shape[0],))
x = Dense(128, activation='relu')(actor_inputs)
actor_outputs = Dense(env.action_space.n, activation='softmax')(x)
actor_model = Model(actor_inputs, actor_outputs)

#Building critic model
critic_inputs = Input(shape=(env.observation_space.shape[0],))
x = Dense(128, activation='relu')(critic_inputs)
critic_outputs = Dense(1, activation='linear')(x)
critic_model = Model(critic_inputs, critic_outputs)
critic_model.compile(optimizer = Adam(lr=0.01), loss = "mse")



step_idx = 0
done_episodes = 0
states, actions, adv, not_done_idx, last_states = [], [], [], [], []
discounted_rewards = []
epochs, reward, = 0, 0
done = False
show = False
video = 0
solved = False
episodes = 0
last_rewards = collections.deque()

stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(10000),
        episode_rewards=np.zeros(10000))

critic_model.reset_states()
actor_model.reset_states()
state = env.reset()

if SAVE_RENDERING:
    rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, path="/home/edoardo/Desktop/Reinforcement_Learning/RL/videos/video1.mp4")

while not solved:

    for i in range(ACTOR_BATCH_SIZE):

        if show:
            env.render(mode='rgb_array')
            rec.capture_frame()

        #choosing action according to probability distribution
        action = np.random.choice([a for a in range(env.action_space.n)], \
                                    p=actor_model.predict(state.reshape(-1,env.observation_space.shape[0])).squeeze())
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        last_rewards.append(reward)

        stats.episode_rewards[episodes] += reward
        stats.episode_lengths[episodes] = step_idx

        #if we have enough steps to unroll the bellman equation
        if step_idx >= REWARD_STEPS - 1:
            discounted_rewards.append(unroll_bellman(copy.copy(last_rewards)))
            last_rewards.popleft()

        step_idx += 1

        #identifying states for which we need to add the value function
        if not done and i not in [ACTOR_BATCH_SIZE - i for i in range(1,REWARD_STEPS)]:
            not_done_idx.append(i)
            last_states.append(np.array(state, copy=False))
        elif done:
            #Recording if needed
            if show:
                show = False
                env.close()
                rec.close()
            #if the episode has ended we need to remove the final states b\c they don't need the bellman approximation
            if len(not_done_idx) > REWARD_STEPS:
                not_done_idx = not_done_idx[:len(not_done_idx) - REWARD_STEPS]
                last_states = last_states[:len(last_states) - REWARD_STEPS]
            else:
                not_done_idx = []
                last_states = []
            #reset environment
            state = env.reset()
            #print("n steps  " + str(step_idx) + " rewards :")
            step_idx = 0
            episodes += 1
            solved = check_if_solved(episodes, stats.episode_rewards, 195, env, video)
            #Unroll all the rewards since the episode has ended
            while not len(last_rewards) == 0:
                discounted_rewards.append(unroll_bellman(copy.copy(last_rewards)))
                last_rewards.popleft()
            #print(discounted_rewards)

        #update the state
        state = next_state

    epochs += 1
    #keep only ACTOR_BATCH_SIZE rewards
    exceeding_rewards = discounted_rewards[ACTOR_BATCH_SIZE:]
    discounted_rewards = discounted_rewards[:ACTOR_BATCH_SIZE]
    rewards_np = np.array(discounted_rewards, dtype=np.float32)
    #Use the critic model to estimate value function
    if not_done_idx and step_idx > REWARD_STEPS:
        last_vals = critic_model.predict(np.array(last_states).reshape(-1,env.observation_space.shape[0])).squeeze()
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals



    input_states = states[:len(discounted_rewards)]
    input_actions = actions[:len(discounted_rewards)]
    states = states[len(discounted_rewards):]
    actions = actions[len(discounted_rewards):]

    #putting the exceeding_rewards in the new batch
    discounted_rewards = exceeding_rewards

    critic_y = rewards_np.copy()
    adv = rewards_np - critic_model.predict(np.array(input_states).reshape(-1,env.observation_space.shape[0])).squeeze()
    critic_model.fit(x=np.array(input_states).reshape(-1,env.observation_space.shape[0]), y= critic_y.reshape(-1,1), \
                                                        batch_size=CRITIC_BATCH_SIZE, epochs=50, verbose = 0)


    #print("adv: ")
    #print(adv)
    actor_y = np_utils.to_categorical(input_actions, env.action_space.n)
    actor_model.compile(optimizer = Adam(lr=0.001), loss = policy_loss(adv, input_states))
    actor_model.fit(x=np.array(input_states).reshape(-1,env.observation_space.shape[0]),\
                                        y= np.array(actor_y).reshape(-1,env.action_space.n), batch_size=ACTOR_BATCH_SIZE, epochs=1, verbose = 0)


    #Free some memory in order to help long training sessions
    del input_states
    del input_actions
    del exceeding_rewards
    del adv
    del rewards_np
