import gym
import gym_md
import random
import matplotlib.pyplot as plt
import csv
import numpy as np

#TODO figure out how you want to store the data
# from gym_md.envs.agent.agent import Agent


def save_data(data, header):
    with open("./data/data.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
    return data

def np_save_data(data, header):
    with open("./data/data.csv", "w", newline="") as f:
        np.savetxt(fname='./data/data1.csv',X=data, header= header)
    return data

def save_visitaion_txt(x, y, lvl, play_style, ep):
    coords = []
    for i, j in zip(x, y):
        coords.append([i, j])

    file = open(f"/home/alex/test/data/{lvl}-{play_style}-{ep}.txt", "w")
    for items in coords:
        line = ",".join(str(x) for x in items)
        file.write(line + "\n")

    return coords

levels=["test", "holmgard_0", "holmgard_1", "holmgard_2", "holmgard_3", "holmgard_4", "holmgard_5", "holmgard_6", "holmgard_7", "holmgard_8", "holmgard_9", "holmgard_10","hard"]

play_styles = {
    "exit_safely_actions": [0, 4, 2, 3, -5, 1, 8],
    "monster_actions": [-1, -2, -2, -2, -2, -2, -2],
    "treasure_safely_actions": [0, 0, 2, 0, 0, 0, 1],
    "potion_safely_actions": [0, 0, 0, 0, 4, 0, 1],
}

# make env using level
play_style = "treasure_safely_actions"
lvl = "test"
env = gym.make(f"md-{lvl}-v0")

header = ["level","run","step","x","y","grid","play_style","action","observation","done","reward","info","grid_rows","grid_columns"]
header_np = "level,run,step,x,y,grid,play_style,action,observation,done,reward,info"
data = []  # [lvl]

LOOP: int = 250
TRY_OUT: int = 30
frame = []

for lvl in levels:
    for playstyle, policy in play_styles.items():
        for _ in range(TRY_OUT):
            observation = env.reset()
            reward_sum = 0
            frame = []
            for i in range(LOOP):
                # env.render(mode='human')
                # actions = [random.random() for _ in range(7)]
                
                actions = play_styles[playstyle]
                observation, reward, done, info = env.step(actions)
                # print(np.array(env.grid.g).shape)
                reward_sum += reward
                grid= np.array(env.grid.g)
                data.append([lvl, _, i, env.agent.x, env.agent.y, grid, play_style, info.action_taken, observation, done, reward, dict(info),grid.shape[0],grid.shape[1]])
                if done:
                    # env.render()
                    break

# print(data)
# data = np.asarray(data)
print(len(data), len(data[0]))
# print(data.shape)
save_data(data,header)