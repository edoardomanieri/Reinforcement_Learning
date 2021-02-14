from A3C import A3C


if __name__ == "__main__":

    CartPole = A3C("LunarLander-v2",actor_batch_size = 128, num_threads=4, max_episodes_per_thread = 1000, episode_to_train=0)
    CartPole.train()
