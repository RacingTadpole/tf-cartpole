import collections
import tqdm
import numpy as np
import statistics
import tensorflow as tf

from typing import List, Tuple


from train_step import train_step
from env import env


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, _ = env.step(action)
    return (
        state.astype(np.float32),
        np.array(reward, np.int32),
        np.array(done, np.int32),
    )


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def train(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    max_episodes=10000,
    min_episodes_criterion=100,
    max_steps_per_episode=1000,
    gamma=0.99,  # Discount factor for future rewards
    reward_threshold=195,  # consider solved if average reward is >= 195 over 100 consecutive trials
):
    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(
        maxlen=min_episodes_criterion
    )
    running_reward = 0

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = int(
                train_step(
                    initial_state,
                    model,
                    optimizer,
                    gamma,
                    max_steps_per_episode,
                    tf_env_step,
                )
            )

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f"Episode {i}")
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break

    return i, running_reward
