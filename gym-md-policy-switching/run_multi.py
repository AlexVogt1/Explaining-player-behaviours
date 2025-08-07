import gym
import gym_md
import random
import matplotlib.pyplot as plt
import csv
import numpy as np

#TODO try and see how to have random tile variables

def save_data(data, header,exp):
    with open("./data/data_policy_switching.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
    return data

# def save_numpy_data(data,header,exp)

def append_data(data, header,exp):
    with open("./data/data_policy_switching.csv", "a", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
    return data

play_styles = {
    "exit_safely_actions": [0, 4, 2, 3, -5, 1, 8],
    "monster_actions": [-1, -2, -2, -2, -2, -2, -2],
    "treasure_safely_actions": [0, 0, 2, 0, 0, 0, 1],
    "potion_safely_actions": [0, 0, 0, 0, 4, 0, 1],
}
lvl ='policy_0'
play_style ="potion_safely_actions"

header = ["experiment","level","run","step","x","y","grid","play_style","action","observation","done","reward","info","grid_rows","grid_columns"]
data =[]
exp = "single_play-potion"

env = gym.make(f'md-{lvl}-v0')

LOOP: int = 100
TRY_OUT: int = 30

for _ in range(TRY_OUT):
    observation = env.reset()
    reward_sum = 0
    for i in range(LOOP):
        # env.render(mode='human')
        # actions = [random.random() for _ in range(7)]
        actions = play_styles[play_style]
        action,observation, reward, done, info = env.step(actions)
        reward_sum += reward
        grid = np.array(env.grid.g)
        data.append([exp,lvl, _, i, env.agent.x, env.agent.y, np.array(env.grid.g), play_style, action, observation, done, reward, dict(info),np.array(env.grid.g).shape[0],np.array(env.grid.g).shape[1]])
        if done:
            # env.render()
            break

    # print(reward_sum)
# print(data)
append_data(data,header,exp)