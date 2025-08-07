import gym
import gym_md
import random
import matplotlib.pyplot as plt
import csv
import numpy as np
import cv2
from PIL import Image

env = gym.make("md-policy_8-v0")

obs = env.reset()
img =env.render(mode='human')
actions = [random.random() for _ in range(7)]
# action,obs, reward, done, info= env.step(actions)
img=env.generate()
print(img)
# plt.show(block=True)
img.save('./README/resources/md_stages_screenshots/md-policy_8-v0_step0.jpg')
