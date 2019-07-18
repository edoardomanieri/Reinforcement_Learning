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
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.01
ENTROPY_BETA = 0.01
ACTOR_BATCH_SIZE = 64
CRITIC_BATCH_SIZE = 16
REWARD_STEPS = 50
SAVE_RENDERING = False
NUM_ENVS = 50

envs = [gym.make("CartPole-v1").env for i in range(NUM_ENVS)]

def policy_loss(adv, states):

    def loss(y_true, y_pred):
        log_prob = K.log(y_pred)[:len(states),:]
        entropy = -K.mean(K.sum(y_pred * log_prob, axis = 1))
        entropy_loss = -ENTROPY_BETA * entropy
        argmax_flat = K.argmax(y_true, axis=1) + [envs[0].action_space.n * _ for _ in range(len(states))]
        log_prob = K.reshape(log_prob, (envs[0].action_space.n*len(states),))
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


env_states = [env.reset() for env in envs]
step_idx = [0 for env in range(NUM_ENVS)]
done_idx_envs = [[] for env in range(NUM_ENVS)]
env_idx = 0


#Building actor model
actor_inputs = Input(shape=(envs[0].observation_space.shape[0],))
x = Dense(128, activation='relu')(actor_inputs)
actor_outputs = Dense(envs[0].action_space.n, activation='softmax')(x)
actor_model = Model(actor_inputs, actor_outputs)

#Building critic model
critic_inputs = Input(shape=(envs[0].observation_space.shape[0],))
x = Dense(128, activation='relu')(critic_inputs)
critic_outputs = Dense(1, activation='linear')(x)
critic_model = Model(critic_inputs, critic_outputs)
critic_model.compile(optimizer = Adam(lr=CRITIC_LEARNING_RATE), loss = "mse")




done_episodes = 0
states, actions, adv, not_done_idx, last_states = [], [], [], [], []
discounted_rewards = []
epochs, reward, = 0, 0
done = False
show = False
video = 0
solved = False
episodes = 0
last_rewards = [collections.deque() for i in range(NUM_ENVS)]

stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(10000),
        episode_rewards=np.zeros(10000))

critic_model.reset_states()
actor_model.reset_states()


if SAVE_RENDERING:
    rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(envs[0], path="/home/edoardo/Desktop/Reinforcement_Learning/RL/videos/video1.mp4")

while not solved:

    for i in range(ACTOR_BATCH_SIZE):
        env = envs[env_idx]
        state = env_states[env_idx]

        if show:
            envs[0].render(mode='rgb_array')
            rec.capture_frame()

        #choosing action according to probability distribution
        action = np.random.choice([a for a in range(envs[0].action_space.n)], \
                                    p=actor_model.predict(state.reshape(-1,envs[0].observation_space.shape[0])).squeeze())
        next_state, reward, done, info = envs[env_idx].step(action)
        states.append(state)
        actions.append(action)
        last_rewards[env_idx].append(reward)

        stats.episode_rewards[episodes] += reward
        stats.episode_lengths[episodes] = step_idx[env_idx]

        #if we have enough steps to unroll the bellman equation
        if step_idx[env_idx] > REWARD_STEPS:
            discounted_rewards.append(unroll_bellman(copy.copy(last_rewards[env_idx])))
            last_rewards[env_idx].popleft()

        step_idx[env_idx] += 1

        #identifying states for which we need to add the value function
        if not done and i not in [ACTOR_BATCH_SIZE - j for j in range(1,REWARD_STEPS)]:
            not_done_idx.append(i)
            last_states.append(np.array(state, copy=False))
            done_idx_envs[env_idx].append(i)
        elif done:
            #Recording if needed
            if show:
                show = False
                envs[0].close()
                rec.close()
            #if the episode has ended we need to remove the final states b\c they don't need the bellman approximation
            if len(done_idx_envs[env_idx]) > REWARD_STEPS:
                done_idx_envs[env_idx] = done_idx_envs[env_idx][len(done_idx_envs[env_idx]) - REWARD_STEPS:]
            not_done_idx = [idx for idx in not_done_idx if idx not in done_idx_envs[env_idx]]
            last_states = [state for idx,state in enumerate(last_states) if idx not in done_idx_envs[env_idx]]
            #reset environment
            state = envs[env_idx].reset()

            step_idx[env_idx] = 0
            episodes += 1
            solved = check_if_solved(episodes, stats.episode_rewards, 195, envs[env_idx], video)
            #Unroll all the rewards since the episode has ended
            while not len(last_rewards[env_idx]) == 0:
                discounted_rewards.append(unroll_bellman(copy.copy(last_rewards[env_idx])))
                last_rewards[env_idx].popleft()


        #update the state
        env_states[env_idx] = next_state

        #round_robin
        env_idx += 1
        env_idx %= NUM_ENVS

    epochs += 1
    #keep at most ACTOR_BATCH_SIZE rewards
    exceeding_rewards = discounted_rewards[ACTOR_BATCH_SIZE:]
    discounted_rewards = discounted_rewards[:ACTOR_BATCH_SIZE]
    rewards_np = np.array(discounted_rewards, dtype=np.float32)
    #Use the critic model to estimate value function
    if not_done_idx and step_idx[env_idx] > REWARD_STEPS:
        print(rewards_np.shape)
        print(len(last_states))
        print(len(not_done_idx))
        last_vals = critic_model.predict(np.array(last_states).reshape(-1,envs[0].observation_space.shape[0]))#.squeeze()
        print(len(last_vals))
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals
        print(last_vals.shape)



    input_states = states[:len(discounted_rewards)]
    input_actions = actions[:len(discounted_rewards)]
    states = states[len(discounted_rewards):]
    actions = actions[len(discounted_rewards):]

    #putting the exceeding_rewards in the new batch
    discounted_rewards = exceeding_rewards

    critic_y = rewards_np.copy()
    adv = rewards_np - critic_model.predict(np.array(input_states).reshape(-1,envs[0].observation_space.shape[0]))#.squeeze()
    critic_model.fit(x=np.array(input_states).reshape(-1,envs[0].observation_space.shape[0]), y= critic_y.reshape(-1,1), \
                                                        batch_size=CRITIC_BATCH_SIZE, epochs=50, verbose = 0)



    actor_y = np_utils.to_categorical(input_actions, envs[0].action_space.n)
    actor_model.compile(optimizer = Adam(lr=ACTOR_LEARNING_RATE), loss = policy_loss(adv, input_states))
    actor_model.fit(x=np.array(input_states).reshape(-1,envs[0].observation_space.shape[0]),\
                                        y= np.array(actor_y).reshape(-1,envs[0].action_space.n), batch_size=ACTOR_BATCH_SIZE, epochs=1, verbose = 0)


    #Free some memory in order to help long training sessions
    del input_states
    del input_actions
    del exceeding_rewards
    del adv
    del rewards_np
