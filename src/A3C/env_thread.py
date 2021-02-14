import threading
import numpy as np
import plotting

class Env_thread(threading.Thread):

    def __init__(self,name, lock, batch, env, actor, critic,  gamma, max_episodes, episodes_to_train):
        threading.Thread.__init__(self)
        self.name = name
        self.env = env
        self.batch = batch
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
        tmp_actor = self.actor.copy()

        for i in range(0, self.max_episodes):

            state = self.env.reset()
            done = False


            while not done:

                step_idx += 1
                if isinstance(state, int):
                    state = np.array([state])
                action = np.random.choice([a for a in range(self.actor.output_shape)], p=tmp_actor.predict(state.reshape(-1,self.actor.input_shape)).squeeze())
                next_state, reward, done, info = self.env.step(action)
                batch_states.append(state)
                batch_actions.append(action)
                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = step_idx
                cur_rewards.append(reward)
                state = next_state


            episodes += 1
            if i > 100 and i % 10 == 0:
                print("mean last 100: {}, episodes: {}, thread: {}".format(np.mean(self.stats.episode_rewards[i-100:i]), episodes, self.name), flush = True)

            batch_scales.extend(self.calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

            if batch_episodes < self.episodes_to_train:
                continue

            self.lock.acquire()
            adv = batch_scales - self.critic.predict(np.array(batch_states).reshape(-1,self.actor.input_shape)).squeeze()
            self.batch.put(batch_states, batch_actions, adv, batch_scales)
            self.lock.release()

            #copying the model so that I can predict without acquiring the lock
            try:
                tmp_actor = self.actor.copy()
                tmp_actor.set_weights(self.actor.get_weights())
            except Exception():
                pass

            batch_episodes = 0
            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()
