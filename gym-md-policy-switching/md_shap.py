import os
from typing import Final, List
import warnings
warnings.filterwarnings('ignore')
import gym
import gym_md
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, DQN
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
from PIL import Image
import imageio
from util import debug_env
import matplotlib.colors as c
import matplotlib.animation as animation
from numpy import float64
from torch import float32
from torch.autograd import Variable

# Global variables
obs_names = ['dist_monster', 'dist_treaure', 'safe_dist_teasure', 'dist_potion', 'safe_dist_potion', 'dist_exit', 'safe_dist_exit', 'HP']
obs_names = ['DM', 'DT', 'SDT', 'DP', 'SDP', 'DE', 'SDE', 'HP']

class_names = ['MONSTER','TREASURE','TREASURE_SAFELY','POTION','POTION_SAFELY','EXIT','EXIT_SAFELY']
LENGTH: Final[int] = 20

tiles_dir = "gym_md/envs/tiles"
tiles_names: Final[List[str]] = [
    "empty.png",
    "wall.png",
    "chest.png",
    "potion.png",
    "monster.png",
    "exit.png",
    "hero.png",
    "deadhero.png",
]
tiles_paths: Final[List[str]] = [os.path.join(tiles_dir, t) for t in tiles_names]
tiles_images = [Image.open(t).convert("RGBA") for t in tiles_paths]
split_images = [[x for x in img.split()] for img in tiles_images]

BASE_ACTION_MAP ={
    'KILLER': 1,
    'POTION': 4, 
    'RUNNER': 3, 
    'TREASURE': 2
}

legend_labels ={
    # "white":'Not Visited', 
    "green":'Killer', 
    "yellow":'Treasure', 
    "blue":'Runner',
    "red":'Potion', 
}
colors = {
    "white":0, 
    "green":1, 
    "yellow":2, 
    "blue":3,
    "red":4, 
    # "gray":5, 
    # "lightgreen":6, 
    # "lightblue":7,  
    # "lightcoral":8, 
    # "brown":9,
    # "violet":10, 
    # "blueviolet":11, 
    # "indigo":12, 
    # "khaki":13, 
    # "orange":14, 
    # "pink":15, 
    # "black":16
}
l_colors = sorted(colors, key=colors.get)
cMap = c.ListedColormap(l_colors)

def get_reward_scheme(type):
    rewards  ={}
    if type == "treasure_safe":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 1,
            "TREASURE":10,
            "POTION": 2,
            "DEAD": -20
        }
        return rewards
    elif type == "treasure_risky":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 1,
            "TREASURE":20,
            "POTION": 1,
            "DEAD": -10
        }
        return rewards
    elif type == "treasure":
        rewards = {
            "TURN": 2,  #0.2, #2
            "EXIT": 10, #-1, #-10
            "KILL": 0, #-1.5, # -25
            "TREASURE":50, #2.5, # 50
            "POTION": 0, #-1.5, # -25
            "DEAD": -250, #-10 # 250
        }
        return rewards
    elif type == "killer_safe":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 10,
            "TREASURE":1,
            "POTION": 9,
            "DEAD": -30
        }
        return rewards
    elif type == "killer_risky":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 20,
            "TREASURE":1,
            "POTION": 2,
            "DEAD": -10
        }
        return rewards
    elif type == "killer":
        rewards = {
            "TURN": 2, #0.2,
            "EXIT": 10, # -1,
            "KILL": 50, #2.5,
            "TREASURE": 0, #-1.5,
            "POTION": 0, #-1.5,
            "DEAD": -250 #-10
        }
        return rewards
    elif type == "runner_safe":
        rewards = {
            "TURN": 1,
            "EXIT": 15,
            "KILL": 1,
            "TREASURE":2,
            "POTION": 3,
            "DEAD": -10
        }
        return rewards
    elif type == "runner_risky":
        rewards = {
            "TURN": 1,
            "EXIT": 30,
            "KILL": 1,
            "TREASURE":1,
            "POTION": 1,
            "DEAD": -5
        }
        return rewards
    elif type == "runner":
        rewards = {
            "TURN": 2, #0.2,
            "EXIT": 50, #2.5,
            "KILL": 0, #-1.5,
            "TREASURE": 0, #-1.5,
            "POTION": 0, # -1.5,
            "DEAD": -250, #-10
        }
        return rewards
    elif type == "clearer_safe":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 15,
            "TREASURE":18,
            "POTION": 20,
            "DEAD": -15
        }
        return rewards
    elif type == "clearer_risky":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 18,
            "TREASURE":20,
            "POTION": 20,
            "DEAD": -5
        }
        return rewards
    elif type == "potion":
        rewards = {
            "TURN": 2, #0.2,
            "EXIT": 10, #-1,
            "KILL": 0, #-1.5,
            "TREASURE": 0, #-1.5,
            "POTION": 50, #2.5,
            "DEAD": -250, #-10
        }
        return rewards
    elif type == "switch":
        rewards = {
            "TURN": 1,
            "EXIT": 5,
            "KILL": 10,
            "TREASURE":10,
            "POTION": 10,
            "DEAD": -25
        }
        return rewards
    elif type == "hard":
        rewards= {
            "TURN": 1,
            "EXIT": 20,
            "KILL": 4,
            "TREASURE": 3,
            "POTION": 1,
            "DEAD": -20
        }
        return rewards
    else :
        return 

#------------------------------------------------Model Wrappers------------------------------------------#
class grid_sb3Wrapper(nn.Module):
    def __init__(self, model):
        super(grid_sb3Wrapper,self).__init__()
        self.extractor = model.policy.mlp_extractor
        self.policy_net = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net

    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.policy_net(x)
        x = self.action_net(x)
        return x

class sb3Wrapper(nn.Module):
    def __init__(self, model):
        super(sb3Wrapper,self).__init__()
        self.extractor = model.policy.mlp_extractor
        self.policy_net = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net

    def forward(self,x):
        x = self.policy_net(x)
        x = self.action_net(x)
        return x

class sb3_DQN_wrapper(nn.Module):
    def __init__(self, model):
        super(sb3_DQN_wrapper,self).__init__()
        self.q_net = model.policy.q_net.q_net
        self.q_net_target = model.q_net_target.q_net

    def forward(self,x):
        x = self.q_net(x)
        return x
    
    def forward_target(self,x):
        x = self.q_net_target(x)
        return x

#------------------------------------------------------------------Functions--------------------------------------------#

def get_reshape(row):
    # print(row)
    return np.fromstring(row['grid'].replace('\n','')
                .replace('[','')
                .replace(']','')
                .replace('  ',' '), sep=' ').reshape(row['grid_rows'],row['grid_columns'])


def gen_action_map(df, f):
    x_array = df['x'].to_numpy()
    y_array =df['y'].to_numpy()
    grid = np.zeros((df['grid_rows'][0],df['grid_columns'][0]))
    for index, row, in df.iterrows():
        x, y, action = row['x'], row['y'], row['action']
        if action in BASE_ACTION_MAP:
            grid[y, x] = BASE_ACTION_MAP[action]
    # print(grid)
    #TDOD Look into decrete colormaps for heatmap
    heat =sns.heatmap(grid,cmap=l_colors,vmin=0,vmax=len(colors)-1,alpha =0.4,zorder=2,cbar=False)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.keys()]
    # heat =
    my_image = cv2.imread(f)
    my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(my_image)
    plt.title(df['experiment'][0])
    # print(heat.get_aspect())
    # colorbar = heat.collections[0].colorbar
    # colorbar.set_ticks([1,2,3,4])
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in legend_labels.keys()]
    img = heat.imshow(my_image,extent= heat.get_xlim() + heat.get_ylim(),zorder=1)
    plt.legend(markers, legend_labels.values(), numpoints=1, loc= 1,bbox_to_anchor=(1.1, 2.2))

    img = convert_fig(plt)
    plt.clf()
    return img

def gen_action_data_img(configs):
    print('----------Generating Action Data----------')
    
    images =[]
    dfs =[]
    for config in configs:
        # plt.clear()
        print(config)
        if config['action_type']=='switch':
            data =gen_data(config)
            lvl = config['lvl']
            img_path=f'./README/resources/md_stages_screenshots/md-{lvl}-v0_step0.png'
            df = pd.DataFrame(data, columns=['experiment','level','run','step','x','y','hp','curr_grid','grid','play_style','action','curr_obs','observation','done','reward','info','grid_rows','grid_columns'])
            dfs.append(df)
            try:
                img =gen_action_map(df, img_path)
                images.append(img)
            except:
                img_path=f'./README/resources/md_stages_screenshots/md-{lvl}-v0_step0.jpg'
                img =gen_action_map(df, img_path)
                images.append(img)
        elif config['action_type']=='base':
            
            data =gen_data(config)
            lvl = config['lvl']
            img_path=f'./README/resources/md_stages_screenshots/md-{lvl}-v0_step0.png'
            df = pd.DataFrame(data, columns=['experiment','level','run','step','x','y','hp','curr_grid','grid','play_style','action','curr_obs','observation','done','reward','info','grid_rows','grid_columns'])
            dfs.append(df)
            try:
                img =gen_heatmap(df, img_path)
                images.append(img)
            except:
                img_path=f'./README/resources/md_stages_screenshots/md-{lvl}-v0_step0.jpg'
                img =gen_heatmap(df, img_path)
                images.append(img)
    
    return dfs, images

def gen_heatmap(df,f):
    df = df.reset_index()
    x_array = df['x'].to_numpy()
    y_array =df['y'].to_numpy()
    grid = np.zeros((df['grid_rows'][0],df['grid_columns'][0]))
    for i, j in zip(y_array,x_array):
        grid[i,j]+=1
    heat =sns.heatmap(grid,cmap='viridis',alpha =0.4,zorder=2, cbar = False)
    my_image = cv2.imread(f)
    my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(my_image)
    plt.title(df['experiment'][0])
    # print(heat.get_aspect())
    img = heat.imshow(my_image,extent= heat.get_xlim() + heat.get_ylim(),zorder=1)
    img = convert_fig(plt)
    plt.clf()
    return img

def gen_data(config):   
    print('----------Generating Data----------')
    
    # set variables
    data =[]
    action_type = config['action_type']
    exp_type =config['exp_type']
    lvl = config['lvl']
    play_style= config['play_style']
    reward_scheme=config['reward_scheme']
    algo= config['learning_algo']
    path = config['exp_path']

    if action_type == 'switch':
        exp = f"{action_type}-{lvl}_{play_style}_{reward_scheme}_{exp_type}_{algo}"
        env = gym.make(f"md-switch-{lvl}-v0",config=config)
    elif config['obs_type'] =='grid':
        exp = f"{action_type}-{lvl}_{play_style}_{reward_scheme}_{exp_type}_{algo}"
        env = gym.make(f"md-switch-{lvl}-v0",config=config)
    else:
        env = gym.make(f"md-{lvl}-v0")
        exp = f"{lvl}_{play_style}_{reward_scheme}_{exp_type}"


    env.setting.IS_ENEMY_POWER_RANDOM =False
    if play_style=='killer':
        env.setting.PLAYER_MAX_HP = 10000

    print(exp)
    if action_type =='base':
        model = PPO.load(f"{path}/{play_style}.zip")  # loading the model from ppo_cartpole.zip 
    elif action_type == 'path':
        model = PPO.load(f"logs/{path}/{exp}/best_model/best_model") 
    elif algo == 'DQN':
        model = DQN.load(f"logs/{path}/{exp}/{play_style}/best_model")  # loading the model from ppo_cartpole.zip 
    else:
        model = PPO.load(f"logs/{path}/{exp}/{play_style}/latest_model_500000_steps.zip")  # loading the model from ppo_cartpole.zip 
                                                                                     
    curr_obs = env.reset()  
    curr_grid = np.array(env.grid.g)
    # data.append([exp, lvl, 0, -1, env.agent.x, env.agent.y, env.agent.hp, start_grid, play_style, "NO_ACTION",curr_obs, curr_obs, False, 0, dict(env.info),start_grid.shape[0],start_grid.shape[1]])

    for ep in range(10):                                                                    
        for i in range(1000): 
            agent_x = env.agent.x
            agent_y = env.agent.y
            agent_hp = env.agent.hp                                                                  
            action, _state = model.predict(curr_obs, deterministic=True) 
            action = action.tolist()                  
            obs, reward, done, info = env.step(action) 
            grid= np.array(env.grid.g)  
            data.append([exp, lvl, ep,i, agent_x, agent_y, agent_hp, curr_grid,grid, play_style, info['action_taken'],curr_obs, obs, done, reward, dict(info),grid.shape[0],grid.shape[1]])
                                                
            # env.render(mode='human')                                                                        
            if done:                                                                            
                obs = env.reset()
                break

            curr_obs= obs
            curr_grid = grid
                
    return data

def gen_data_and_img(config):
    data =gen_data(config)
    lvl = config['lvl']
    img_path=f'./README/resources/md_stages_screenshots/md-{lvl}-v0_step0.png'
    df = pd.DataFrame(data, columns=['experiment','level','run','step','x','y','hp','curr_grid','grid','play_style','action','curr_obs','observation','done','reward','info','grid_rows','grid_columns'])
    try:
        action_img = gen_action_map
        img =gen_heatmap(df, img_path)
    except:
        img_path=f'./README/resources/md_stages_screenshots/md-{lvl}-v0_step0.jpg'
        img =gen_heatmap(df, img_path)
    return df,img

def convert_fig(plt):
    # Get the figure and its axes
    fig = plt.gcf()
    axes = plt.gca()

    # Draw the content
    fig.canvas.draw()

    # Get the RGB values
    rgb = fig.canvas.tostring_rgb()

    # Get the width and height of the figure
    width, height = fig.canvas.get_width_height()

    # Convert the RGB values to a PIL Image
    img = Image.frombytes('RGB', (width, height), rgb)
    img_array = np.array(img)
    return img_array


def shap_behaviour(config):
    action_type = config['action_type']
    exp_type =config['exp_type']
    lvl = config['lvl']
    play_style= config['play_style']
    reward_scheme=config['reward_scheme']
    algo= config['learning_algo']
    path = config['exp_path']
    #generate data
    data =gen_data(lvl=lvl, play_style=play_style, reward_scheme=reward_scheme, exp_type=exp_type, path=path)
    df = pd.DataFrame(data, columns=['experiment','level','run','x','y','grid','play_style','action','curr_obs','observation','done','reward','info','grid_rows','grid_columns'])
    print(df['action'].value_counts())
    df_to_analyse = df[df['action'].isin({f'{play_style.upper()}', f'{play_style.upper()}_SAFELY'})]
    # print(df_to_analyse[['action', 'observation']])
    print(df[['action','observation']])
    #load PPO model
    if action_type == "switch":
        exp = f"{action_type}-{lvl}_{play_style}_{reward_scheme}_{exp_type}_{algo}"
        model_path = f"logs/{path}/{lvl}_{play_style}_{reward_scheme}_{exp_type}/best_model/best_model"
    # else:

    # if algo =="PPO":
    #     model = PPO.load(model_path, device='cuda')
    # elif algo =='DQN':
        model = DQN.load(model_path, device='cuda')
    state_log = np.array(df['curr_obs'].values.tolist())
    data =torch.FloatTensor(state_log).to('cuda')
    model = sb3Wrapper(model)
    explainer = shap.DeepExplainer(model, data)
    # explainer=shap.KernelExplainer(model.policy.,state_log)
    shap_vals= explainer.shap_values(data)

    return explainer, shap_vals, data, df


def shappy(config,explainer_type ='deep'):
    print('----------Perfroming Shapley Analysis----------')
    action_type = config['action_type']
    exp_type =config['exp_type']
    lvl = config['lvl']
    play_style= config['play_style']
    reward_scheme=config['reward_scheme']
    algo= config['learning_algo']
    path = config['exp_path']
    #generate data
    data =gen_data(config)
    df = pd.DataFrame(data, columns=['experiment','level','run','step','x','y','hp','curr_grid','grid','play_style','action','curr_obs','observation','done','reward','info','grid_rows','grid_columns'])
    print(df['action'].value_counts())
    df_to_analyse = df[df['action'].isin({f'{play_style.upper()}', f'{play_style.upper()}_SAFELY'})]
    # print(df_to_analyse[['action', 'observation']])
    # print(df[['action','observation']])
    #load PPO model
    if action_type == "switch":
        exp = f"{action_type}-{lvl}_{play_style}_{reward_scheme}_{exp_type}_{algo}"
        model_path = f"logs/{path}/{exp}/{play_style}/best_model"
    else:
        exp = f"{lvl}_{play_style}_{reward_scheme}_{exp_type}"
        model_path = f"logs/{path}/{exp}/best_model/best_model"

    if algo =="PPO":
        model = PPO.load(model_path, device='cuda')
    elif algo =='DQN':
        model = DQN.load(model_path, device='cuda')
    state_log = np.array(df['curr_obs'].values.tolist())
    data =torch.FloatTensor(state_log).to('cuda')
    # print(model.policy)
    # print(model.policy.q_net.q_net)
    if config['obs_type']== 'grid':
        model = grid_sb3Wrapper(model)
    elif algo =='PPO':
        model = sb3Wrapper(model)
    elif algo == 'DQN':
        model = sb3_DQN_wrapper(model)
    

    if explainer_type == "kernel":
        f = lambda x: model.forward(Variable(torch.from_numpy(x),requires_grad=False).to(float32).cuda()).detach().cpu().numpy()
        explainer = shap.KernelExplainer(f, state_log)
        shap_vals= explainer.shap_values(state_log)
    else:
        explainer = shap.DeepExplainer(model, data)
        shap_vals= explainer.shap_values(data,check_additivity=True)

    # explainer=shap.Explainer(model.forward,data)
    # shap_vals= explainer.shap_values(state_log)
    # return
    return explainer, shap_vals, data, df


def shap_behaviour_gif(lvl,play_style,reward_scheme,exp_type,path,gif_type):
    #generate data
    data =gen_data(lvl=lvl, play_style=play_style, reward_scheme=reward_scheme, exp_type=exp_type, path=path)
    df = pd.DataFrame(data, columns=['experiment','level','run','step','x','y','grid','play_style','action','observation','done','reward','info','grid_rows','grid_columns'])
    print(df['action'].value_counts())
    df_to_analyse = df[df['action'].isin({f'{play_style.upper()}', f'{play_style.upper()}_SAFELY'})]
    # print(df_to_analyse[['action', 'observation']])
    print(df[['action','observation']])
    #load PPO model

    model_path = f"logs/{path}/{lvl}_{play_style}_{reward_scheme}_{exp_type}/best_model/best_model"
    exp_name = f"{lvl}_{play_style}_{reward_scheme}_{exp_type}"
    model = PPO.load(model_path, device='cuda')
    state_log = np.array(df['observation'].values.tolist())
    data =torch.FloatTensor(state_log).to('cuda')
    model = sb3Wrapper(model)

    images = []
    if gif_type == "state_inc":
        for i in tqdm(range(len(state_log))):
            # print(data[:i+1,:])
            # data = torch.FloatTensor(state_log).to('cuda')
            cur_data = data[:i+1,:]
            explainer = shap.DeepExplainer(model, cur_data)
            shap_vals= explainer.shap_values(cur_data)
            shap.summary_plot(shap_vals,cur_data, feature_names=obs_names,class_names= class_names,plot_type='bar', show =False)
            plt.title(f'{exp_name}\n states used: {i}')
            images.append(convert_fig(plt))
            plt.clf()
    elif gif_type== "play":
        for i in tqdm(range(len(state_log))):
            cur_data = data[i]
            explainer = shap.DeepExplainer(model, cur_data)
            shap_vals= explainer.shap_values(cur_data)
            action = df['action'].iloc[i]
            shap.force_plot(shap_vals,cur_data, feature_names=obs_names,class_names= class_names,plot_type='bar', show =False)
            plt.title(f'{exp_name}\n states used: {i}')
            images.append(convert_fig(plt))
            plt.clf()
        
    print(images)
    imageio.mimsave(f'./shap/shap_gifs/{exp_name}.gif',images,format='GIF')

    return explainer, shap_vals, data, df

#------------------------------------------------ Analysis Functions ----------------------------------------------------------#
policy_classes = ['killer','treasure','runner','potion']
EXPLAINER_TYPE = "deep"
CLASS_NAMES =['Killer','Treasure','Runner','Potion']


def shap_index(explainer,data,df, index):
    #TODO Look  into barplot and chorts
    #https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html
    i = index
    if EXPLAINER_TYPE == 'deep':
        index_data = data[i:i+1]
        shap_val_single = explainer.shap_values(X=index_data)
        # get action taken
        action = df['action'].iloc[i]
        # get action taken index
        action_idx = policy_classes.index(action.lower())
        exp = shap.Explanation(shap_val_single[action_idx][0], explainer.expected_value[action_idx], data = df['curr_obs'].values[i], feature_names=obs_names)
    elif EXPLAINER_TYPE == "kernel":
        shap_val_single = explainer.shap_values(X = df['curr_obs'].values[i])
        index_data = df['curr_obs'].values[i]
        # get action taken
        action = df['action'].iloc[i]
        action_idx = policy_classes.index(action.lower())
        exp = shap.Explanation(shap_val_single[action_idx], explainer.expected_value[action_idx], data = df['curr_obs'].values[i], feature_names=obs_names)

    try:
        imgs = shap.plots.waterfall(exp, show = False)
    except:
        imgs = shap.plots.waterfall(shap_val_single)
    plt.title(f"Action: {action}")
    plt.rcParams['figure.constrained_layout.use'] = True
    imgs = convert_fig(plt)

    return imgs


def shap_state(explainer,df,data, state):
    # get index of states
    state_idx = df['observation'].tolist()
    print(state_idx)
    for i in range( len(state_idx)):
        if state_idx[i].tolist() == state:
            state_idx = i
            break
    print(state_idx)
    action=df['action'].iloc[state_idx]
    action_idx = policy_classes.index(action.lower())
    shap_val_state = explainer.shap_values(X=data[state_idx:state_idx+1,:])
    
    shap.summary_plot(shap_val_state[action_idx],data, feature_names=obs_names,plot_type='bar')
    img = convert_fig(plt)
    return img
# print(shap_state(explainer,df,data, [11, 3, 3, 19, 19, 23, 23, 100]))

def action_shap(shap_vals,data, action, loc):
    action_idx = policy_classes.index(action.lower())
    shap.summary_plot(shap_vals[action_idx],data, feature_names=obs_names, class_names =['Killer','Treasure','Runner','Potion'],plot_type='bar', show=False)
    plt.title(f"Summary for {action.lower()} Policy")
    plt.savefig(f"{loc}{action.lower()}_policy_action.png")
    

def state_image(grid, agent_x, agent_y,agent_hp):
    # tiles_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, "gym_md/envs/tiles")
    H = grid.shape[0]
    W = grid.shape[1]

    img = Image.new("RGB", (W * LENGTH, H * LENGTH))
    for i in range(H):
        for j in range(W):
            img.paste(tiles_images[0], (LENGTH * j, i * LENGTH))
            e: int = grid[i, j]
            if i == agent_y and j == agent_x:
                e = 6 if agent_hp > 0 else 7
            img.paste(tiles_images[e], (LENGTH * j, i * LENGTH), split_images[e][3])
    # maybey convert to array
    # print(img)
    # img = convert_fig(plt)
    # print(img)
    return img
def make_state_gif(images,file_name_loc):

    duration = len(images)*100
    print(duration)
    images[0].save(file_name_loc, format = "GIF",save_all=True, append_images= images[1:], transparency=0, duration= 1000, disposal=2, loop=0)
    
def make_shap_gif(images,file_name_loc):
    imageio.mimsave(file_name_loc,images,format='GIF', duration=1000)

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def merge_gifs(file1:str,file2:str,output_loc:str):
    #Create reader object for the gif
    gif1 = imageio.get_reader(file1)
    gif2 = imageio.get_reader(file2)

    # Get the frame count for each GIF
    num_frames1 = gif1.get_length()
    num_frames2 = gif2.get_length()

    # Determine the width and height of the resulting GIF
    width = gif1.get_data(0).shape[1] + gif2.get_data(0).shape[1]
    height = max(gif1.get_data(0).shape[0], gif2.get_data(0).shape[0])
    print(width, height)
    # print(gif1.get_meta_data())

    # Create a writer object for the resulting GIF
    output_gif = imageio.get_writer(output_loc, duration=gif1.get_meta_data()['duration'])
    merged_image_list = []

    # Loop through frames and merge them side by side
    for frame1, frame2 in zip(gif1, gif2):
        merged_frame = get_concat_h_blank(Image.fromarray(frame1),Image.fromarray(frame2))
        output_gif.append_data(merged_frame)
        merged_image_list.append(merged_frame)

    merged_image_list[0].save(f"{output_loc}.pdf", save_all = True, append_images= merged_image_list[1:])
    # Close the output GIF writer
    gif1.close()
    gif2.close()  
    output_gif.close()


def play_thru_analysis(explainer, shap_vals,data, df, loc):
    #create figure space
    # fig, (ax1, ax2) = plt.subplots(2,1)
    # iterate every
    grid_array =[]
    shap_array =[]
    images = []
    for index, row in df.iterrows():
        if row['run'] ==0:
            # grid_img = ax1.imshow(state_image(row['curr_grid'], row['x'],row['y'],row['hp']))
            grid_array.append(state_image(row['curr_grid'], row['x'],row['y'],row['hp']))
            plt.clf()
            # print(shap_index(explainer,data,df, index))
            # shap_img = ax2.imshow(shap_index(explainer,data,df, index))
            shap_array.append(Image.fromarray(shap_index(explainer,data,df, index)))
            plt.clf()
            # images.append([grid_img, shap_img])
    shap_file_loc = f'./{loc}/shap_gif.gif'
    state_file_loc = f'./{loc}/state_gif.gif'
    output = f'./{loc}/play_through_gif.gif'
    print(shap_array)
    make_state_gif(grid_array,state_file_loc)
    make_state_gif(shap_array,shap_file_loc)

    merge_gifs(state_file_loc,shap_file_loc,output)

def gen_config(lvl:str, algo:str, obs_type:str, exp_type:str,switch_path:str, base_path:str ='play_style_models/base/'):
    config_switch ={
        'action_type': 'switch',
        'action_space_type': 'discrete',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'switch',
        'reward_scheme': 'switch',
        'exp_type': f'{exp_type}',
        'learning_algo': f'{algo}',
        'exp_path': f'{switch_path}',
        'base_path': f'{base_path}',
    }
    config_hard ={
        'action_type': 'switch',
        'action_space_type': 'discrete',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'hard',
        'reward_scheme': 'hard',
        'exp_type': f'{exp_type}',
        'learning_algo': f'{algo}',
        'exp_path': f'{switch_path}',
        'base_path': f'{base_path}',
    }
    config_treasure ={
        'action_type': 'switch',
        'action_space_type': 'discrete',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'treasure',
        'reward_scheme': 'treasure',
        'exp_type': f'{exp_type}',
        'learning_algo': f'{algo}',
        'exp_path': f'{switch_path}',
        'base_path': f'{base_path}',
    }
    config_killer ={
        'action_type': 'switch',
        'action_space_type': 'discrete',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'killer',
        'reward_scheme': 'killer',
        'exp_type': f'{exp_type}',
        'learning_algo': f'{algo}',
        'exp_path': f'{switch_path}',
        'base_path': f'{base_path}',
    }
    config_potion ={
        'action_type': 'switch',
        'action_space_type': 'discrete',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'potion',
        'reward_scheme': 'potion',
        'exp_type': f'{exp_type}',
        'learning_algo': f'{algo}',
        'exp_path': f'{switch_path}',
        'base_path': f'{base_path}',
    }
    config_runner ={
        'action_type': 'switch',
        'action_space_type': 'discrete',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'runner',
        'reward_scheme': 'runner',
        'exp_type': f'{exp_type}',
        'learning_algo': f'{algo}',
        'exp_path': f'{switch_path}',
        'base_path': f'{base_path}',
    }
    config_base_treasure={
        'action_type': 'base',
        'action_space_type': 'box',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'treasure',
        'reward_scheme': 'fiftytwoFifty',
        'exp_type': 'treasure',
        'learning_algo': f'{algo}',
        'exp_path': f'{base_path}',
        'base_path': f'{base_path}',
    }
    config_base_killer={
        'action_type': 'base',
        'action_space_type': 'box',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'killer',
        'reward_scheme': 'fiftytwoFifty',
        'exp_type': 'killer',
        'learning_algo':f'{algo}',
        'exp_path': f'{base_path}',
        'base_path': f'{base_path}',
    }
    config_base_potion={
        'action_type': 'base',
        'action_space_type': 'box',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'potion',
        'reward_scheme': 'fiftytwoFifty',
        'exp_type': 'potion',
        'learning_algo': f'{algo}',
        'exp_path': f'{base_path}',
        'base_path': f'{base_path}',
    }
    config_base_runner={
        'action_type': 'base',
        'action_space_type': 'box',
        'obs_type': f'{obs_type}',
        'lvl': f'{lvl}',
        'play_style': 'runner',
        'reward_scheme': 'fiftytwoFifty',
        'exp_type': 'runner',
        'learning_algo': f'{algo}',
        'exp_path': f'{base_path}',
        'base_path': f'{base_path}',
    }

    return [config_switch, config_hard,config_treasure,config_killer, config_potion, config_runner, config_base_treasure, config_base_killer, config_base_potion, config_base_runner]

def plot_frequency(ax, x, y, xlabel, ylabel, title):
    sns.barplot(x=x, y=y, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

def qauntitative_results(explainer, shap_vals, data, df, save_loc):
    print("----------------------------------------------------Generting quantitative results-------------------------------------------")
    # create shap value df
    df_shap = df[['curr_obs', 'action']]
    shap_valss = []
    policy_classes = ['killer','treasure','runner','potion']

    reshaped_shap_vals = np.transpose(np.array(shap_vals),(0,2,1))
    action_shap_vals=[]
    for i, row in df.iterrows():
        action = row['action']
        # get action taken index
        action_idx = policy_classes.index(action.lower())
        action_shap_vals.append(reshaped_shap_vals[i][action_idx])

    # Add shap_vals to the dataframe
    df_shap["shap_vals"] = pd.Series(action_shap_vals)
    df_shap[obs_names] = pd.DataFrame(df_shap.shap_vals.to_list(), index = df_shap.index)

    # get the top five features for each state
    df_shap= df_shap.join(pd.DataFrame(df_shap[obs_names].abs().apply(lambda x: x.nlargest(5).index.tolist(), axis=1).tolist(), columns=['1st','2nd','3rd','4th','5th']))

    # create rankings dataframe for the whole level by adding up the value counts
    df_rankings = df_shap[['1st','2nd','3rd','4th','5th']].apply(pd.Series.value_counts).sum(axis=1)
    df_rankings = pd.DataFrame(data=df_rankings,columns=['Importance_count']).sort_values(by='Importance_count')

    ax = sns.barplot(x=df_rankings.index, y=df_rankings.Importance_count)
    ax.set(xlabel='Features', ylabel='Times in Top 5 Most Important Features', title="Frequency of a feature being important \nin deciding the chosen action")
    plt.savefig(f'{save_loc}/Feature_Rankings.png')

    melted_df = pd.melt(df_shap[['action','1st','2nd','3rd','4th','5th']], id_vars=['action'], value_vars=['1st', '2nd', '3rd', '4th', '5th'])
    df_action_rankings = pd.crosstab(index=melted_df['action'], columns=melted_df['value'])

    fig, ax = plt.subplots(1, len(df_action_rankings), figsize=(4 * len(df_action_rankings), 4))

    if len(df_action_rankings)==1:
        ax = sns.barplot(x=df_action_rankings.columns,y=df_action_rankings.iloc[0])
        ax.set(xlabel='Features', ylabel='Times in Top 5 Most Important Features',
                    title=f"Frequency of a feature being important \nwhen choosing the {df_action_rankings.index[0]} action")
        fig.savefig(f'{save_loc}/Feature_Rankings_Per_action.png')
    else:
        for i in range(len(df_action_rankings)):
            plot_frequency(ax[i], df_action_rankings.columns, df_action_rankings.iloc[i],
                        xlabel='Features', ylabel='Times in Top 5 Most Important Features',
                        title=f"Frequency of a feature being important \nwhen choosing the {df_action_rankings.index[i]} action")
        fig.savefig(f'{save_loc}/Feature_Rankings_Per_action.png')
    plt.tight_layout()
    plt.show() 
    plt.clf()
    return df_shap, df_rankings, df_action_rankings

def gen_grid_shap(shap_vals,df, save_loc):
    policy_classes = ['killer','treasure','runner','potion']
    state_loc = f"{save_loc}state_shap_images/"
    os.makedirs(state_loc,exist_ok=True)  
    shap_vals = np.array(shap_vals)
    print(shap_vals.shape)
    print(shap_vals[0][0].shape)
    reshaped_shap_vals = np.transpose(np.array(shap_vals),(1,0,2,3))
    print(reshaped_shap_vals.shape)
    print(reshaped_shap_vals[0][0].shape)
    df_shap = df[['run','x','y','hp','curr_obs', 'action']]
    # print(sha)
    action_shap_vals=[]
    #get shap_vals for chosen action
    for i, row in df.iterrows():
        action = row['action']
        # get action taken index
        action_idx = policy_classes.index(action.lower())
        action_shap_vals.append(reshaped_shap_vals[i][action_idx])

    # Add shap_vals to the dataframe
    df_shap["shap_vals"] = pd.Series(action_shap_vals)
    images =[]
    temp =df_shap[(df_shap['run']==0)]
    temp
    for i, row in df_shap[(df_shap['run']==0)].iterrows():
        # print(row.shap_vals.shape)
        # newcmp = ListedColormap(GradientBlueRed_res, name='BlueRed')
        heat =sns.heatmap(row.shap_vals,cmap='bwr',alpha= 0.6,zorder=2, center=0.0)
        # gen state image
        state = state_image(row['curr_obs'],row['x'],row['y'],row['hp'])
        # state = state_image(df_shap['curr_obs'].iloc[0],df['x'].iloc[0],df['y'].iloc[0],df['hp'].iloc[0])
        # state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        img = heat.imshow(state,aspect=heat.get_aspect(),extent= heat.get_xlim() + heat.get_ylim(),zorder=1)
        plt.title(f"Action: {row['action']}")
        img = convert_fig(plt)
        images.append(img)
        Image.fromarray(img).save(f'{state_loc}state{i}.png')

        # cv2.imwrite(f'/gifs/state_images/state{i}.png',img)
        plt.clf()

    # images[0].save(f"{state_loc}.pdf", save_all = True, append_images= images[1:])

    # make_state_gif(images, './gifs/AAA.gif')
    # images[0].save("/gifs/array.gif", save_all=True, append_images=img[1:], duration=50, loop=0)
    imageio.mimsave(f'{state_loc}State_Shap_Gif.gif', images,duration=1000)
    imageio.mimsave(f'{state_loc}State_shap.pdf', images)

def gen_analysis(analysis_loc,lvl, obs_type, explainer_type,exp_type='switching_analysis', exp_path='switching_analysis', base_path='play_style_models/base/'):
    plt.close()
    #TODO check if analysis loc exists and create folder if not
    #TODO make this include hard analysis
    #params
    #gen config
    os.makedirs(analysis_loc,exist_ok=True)  
    configs = gen_config(lvl = lvl, algo='PPO', obs_type = obs_type,exp_type=exp_type, switch_path=exp_path, base_path=base_path)
    print(len(configs))
    #gen and save images
    df, images = gen_action_data_img(configs)
    f, axarr = plt.subplots(2,2, figsize=(16,9))
    axarr[0,0].imshow(images[-4])
    axarr[0,0].axis('off')
    axarr[0,0].set_aspect('equal')

    axarr[0,1].imshow(images[-3])
    axarr[0,1].axis('off')
    axarr[0,1].set_aspect('equal')
    axarr[1,0].imshow(images[-2])
    axarr[1,0].axis('off')
    axarr[1,0].set_aspect('equal')
    axarr[1,1].imshow(images[-1])
    axarr[1,1].axis('off')
    axarr[1,1].set_aspect('equal')
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(f"{analysis_loc}/base_play-styles.png")
    plt.show()
    plt.clf()
    axarr[1,1].axis('off')
    plt.imshow(images[0])
    plt.axis('off')
    plt.savefig(f"{analysis_loc}/switching.png")
    plt.show()
    plt.clf()
    plt.imshow(images[1])
    plt.axis('off')
    plt.savefig(f"{analysis_loc}/switching.png")
    plt.show()
    plt.clf()

    #TODO fix summary saving plots
    switch_analysis = ['switch','hard','treasure','killer','potion','runner']
    # iteratively create analysis for all the siwtching agents using 
    for i, switch_type in enumerate(switch_analysis):
        sub_path = f'{analysis_loc}{switch_type}/' 
        os.makedirs(sub_path,exist_ok=True)   

        plt.imshow(images[i])
        plt.axis('off')
        plt.savefig(f"{sub_path}switching.png")
        plt.show()
        plt.clf()
        #shap stuff
        #explain switcher on switching rewward
        explainer , shap_vals, data, df =shappy(configs[i], explainer_type=explainer_type)

        if obs_type=='distance':
            df_shap, df_rankings, df_action_rankings = qauntitative_results(explainer=explainer,shap_vals=shap_vals, data=data, df=df, save_loc=f"{sub_path}")
            plt.clf()
            shap.summary_plot(shap_vals,data, feature_names=obs_names, class_names =['Killer','Treasure','Runner','Potion'],plot_type='bar',show=False)
            plt.savefig(f"{sub_path}Summary.png")
            plt.clf()
            for policy in CLASS_NAMES:
                #TODO fix saving plots
                action_shap(shap_vals=shap_vals, data = data, action= policy, loc =f"{sub_path}")
                plt.clf()
            # gen and save gifs
            play_thru_analysis(explainer,shap_vals,data,df, f"{sub_path}")
            plt.clf()
        elif obs_type=='grid':
            gen_grid_shap(shap_vals=shap_vals, df=df, save_loc=f"{sub_path}")

