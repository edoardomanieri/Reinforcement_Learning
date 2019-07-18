import threading
import numpy as np
import plotting

class Env_thread(threading.Thread):

    def __init__(self,name, lock, env, actor, critic,  gamma, max_episodes, episodes_to_train):
        threading.Thread.__init__(self)
        self.name = name
        self.env = env
        self.actor = actor
        self.critic = critic
        self.lock = lock
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.episodes_to_train = episodes_to_train
        self.stats = plotting.EpisodeStats(
                episode_lengths=np.zeros(10000),
                episode_rewards=np.zeros(10000))


    def calc_qvals(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.gamma
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))

    def compute_advantages(self,discounted_rewards, input_states):
        adv = discounted_rewards - self.critic.predict(np.array(input_states).reshape(-1,self.actor.input_shape)).squeeze()
        return adv

    def run(self):

        step_idx, batch_episodes, episodes = 0, 0, 0
        batch_states, batch_actions, batch_scales, cur_rewards = [], [], [], []

        for i in range(0, self.max_episodes):

            state = self.env.reset()
            done = False

            while not done:

                step_idx += 1
                print("start waiting..{}".format(self.name))
                self.lock.acquire()
                action = np.random.choice([a for a in range(self.actor.output_shape)], p=self.actor.predict(state.reshape(-1,self.actor.input_shape)).squeeze())
                self.lock.release()
                print("done {}".format(self.name))
                next_state, reward, done, info = self.env.step(action)
                batch_states.append(state)
                batch_actions.append(action)
                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = step_idx
                cur_rewards.append(reward)
                state = next_state


            episodes += 1
            if i > 10 and i % 10 == 0:
                print("mean last 10: ", np.mean(self.stats.episode_rewards[i-10:i]), flush = True)

            discounted_rewards = self.calc_qvals(cur_rewards)
            batch_scales.extend(discounted_rewards)
            cur_rewards.clear()
            batch_episodes += 1

            if batch_episodes < self.episodes_to_train:
                continue

            self.lock.acquire()
            print("training...{}".format(self.name))
            adv = batch_scales - self.critic.predict(np.array(batch_states).reshape(-1,self.actor.input_shape)).squeeze()
            self.critic.fit(np.array(batch_scales), batch_states)
            self.actor.fit(adv, batch_states, batch_actions)
            self.lock.release()

            batch_episodes = 0
            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()
