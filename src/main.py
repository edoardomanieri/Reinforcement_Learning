from A3C import A3C


if __name__ == "__main__":

    CartPole = A3C("CartPole-v1",actor_batch_size = 64, num_threads=4, max_episodes_per_thread = 1000, episode_to_train=5)
    CartPole.train()
