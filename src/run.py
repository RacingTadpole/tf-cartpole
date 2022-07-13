from actor_critic import ActorCritic
import numpy as np
import tensorflow as tf

from env import env
from train import train


# Set seed for experiment reproducibility
seed = 42
env.reset(seed=seed)
tf.random.set_seed(seed)
np.random.seed(seed)

num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

print("Setup complete")

steps, reward = train(model, optimizer)

print(f"\nSolved at episode {steps}: average reward: {reward:.2f}!")
