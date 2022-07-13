# Render an episode and save as a GIF file
import gym
import numpy as np
import tensorflow as tf

# from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display

# from matplotlib import pyplot as plt

from env import env
from run import model  # This import will train the model

display = Display(visible=0, size=(400, 300))
display.start()


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
    screen = env.render(mode="rgb_array")
    im = Image.fromarray(screen)

    images = [im]

    state = tf.constant(env.reset(), dtype=tf.float32)
    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_probs, _ = model(state)
        action = np.argmax(np.squeeze(action_probs))

        state, _, done, _ = env.step(action)
        state = tf.constant(state, dtype=tf.float32)

        # Render screen every 10 steps
        if i % 10 == 0:
            screen = env.render(mode="rgb_array")
            images.append(Image.fromarray(screen))

        if done:
            break

    return images


# Save GIF image
images = render_episode(env, model, max_steps_per_episode=1000)
image_file = "cartpole-v0.gif"
# loop=0: loop forever, duration=1: play each frame for 1ms
images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
