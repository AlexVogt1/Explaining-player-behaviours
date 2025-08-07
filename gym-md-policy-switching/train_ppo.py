from ast import Tuple
from logging import config
import os
import numpy as np                                                   
import gym
from sympy import true   
import gym_md 
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
import wandb
from pprint import pprint
from util import debug_env
from wandb.integration.sb3 import WandbCallback
import warnings
import torch as th
warnings.filterwarnings("ignore")
#TODO add resume training from last best model functionality and restart
#TODO find A way to save config into json
#TODO increase batch_size
#TODO improve model naming (make names unique and informative so can just drag and drop into model folder)
#fix naming scheme

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
            "EXIT": 3,#5,
            "KILL": 5,#5,
            "TREASURE":5,#5,
            "POTION": 5,#5,
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

def parse_args():
    parser = argparse.ArgumentParser("PPO for MiniDungeons (gym-md)")
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="hard", help="name of the game")
    # Play-style to train (i.e. which reward scheme to use)
    parser.add_argument("--play_style", type=str, default="optimal", help= "what type of player are you wanting to train")
    parser.add_argument("--reward_scheme", type=str, default="original", help= "what reward scheme are you using for training")
    parser.add_argument("--exp_type", type=str, default="test", help= "what type of experiment are you running")
    parser.add_argument("--action_type", type=str, default="path", help= "what type action space")
    parser.add_argument("--action_space_type", type=str, default="discrete", help= "what type action space")
    parser.add_argument("--obs_type", type=str, default="distance", help= "what type obs space (grid ir disances)")
    parser.add_argument("--base_path", type=str, default="./play_style_models/grid_base_12x12/", help= "path direct for the base play-style policies")
    parser.add_argument("--algo", type=str, default="PPO", help= "what algo are you using to learn")


    # parser.add_argument("--resume", type=str, default=, help= "what type of experiment are you running")

    return parser.parse_args()

def uniquify(path, x=0):
    while True:
        dir_name = (path + ('_' + str(x) if x is not 0 else '')).strip()
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(dir_name)
            exp_name= dir_name.split('/')
            return dir_name, exp_name[-1]
        else:
            x = x + 1


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.latest_save
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if len(x)>100:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if len(x) >100:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
            

        return True


def main(lvl, config, steps, log_dir):
# def main(lvl,, log_dir):

    
    #create paths
    best_model_path = os.path.join(log_dir, config['play_style'])
    
    if config['action_type'] == 'policy':
        env = gym.make(f"md-{lvl}-v0",config = config)       
    else:
        env = gym.make(f"md-{lvl}-v0",config =config)       
    # env = Monitor(env)  
    # env = DummyVecEnv(env)  

    #change reward if not orginal
    if config['play_style'] != 'optimal':
        rewards = get_reward_scheme(config['play_style'])   
        env.change_reward_values(rewards)
        print(config['play_style'])
        print(env.setting.REWARDS) 
    else:
        print(config['play_style'])
        print(env.setting.REWARDS)   
    
    #chaning HP
    # env.change_player_hp(10000)
    # env.setting.DISTANCE_INF= -1
    print(f"inf value: {env.setting.DISTANCE_INF}")
    # env.setting.IS_ENEMY_POWER_RANDOM
    if config['play_style'] == 'killer':
        env.setting.PLAYER_MAX_HP = 100

    print("---------------------------------------------------------------------------------")
    print(f"Experiment : {config['exp_type']} \nLevel : {config['lvl']} \nPlay_style : {config['play_style']} \nTrainging_Method : PPO \nTraing_Steps : 1e6 \nCallback : EvalCallback" )
    print(config['play_style'])
    print(env.setting.REWARDS) 
    print(env.setting.PLAYER_MAX_HP)
    print("---------------------------------------------------------------------------------")
    debug_env(env)
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")
    print(config)
    print("---------------------------------------------------------------------------------")
    

    # update config
    # wandb.config['player_max_hp'] = env.setting.PLAYER_MAX_HP
    wandb.config.update({'player_max_hp': env.setting.PLAYER_MAX_HP}, allow_val_change=True)

    if config['algorithm'] == 'DQN':
        model = DQN(policy='MlpPolicy',env= env, batch_size=2560,learning_starts= 5000,target_update_interval=1000,verbose=1, device='cuda',tensorboard_log=log_dir,)             
    else: 
        # policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=dict(pi=[32, 32], vf=[32, 32]))                                                             
        # model = PPO(policy = "MlpPolicy",env =  env,batch_size=4096,verbose=1, device="cuda", tensorboard_log=log_dir) 
        model = PPO(policy = "MlpPolicy",env =  env,batch_size=64,verbose=1, device="cuda", tensorboard_log=log_dir)     
        # model = PPO(policy = "MlpPolicy",env =  env, batch_size=1024,verbose=1, device="cuda", tensorboard_log=log_dir) 
        # model = PPO(policy = "MlpPolicy",env =  env, batch_size=256,verbose=1, device="cuda", tensorboard_log=log_dir) 
        # model = PPO(policy = "MlpPolicy",env =  env, batch_size=2048,verbose=1, device="cuda", tensorboard_log=log_dir,policy_kwargs=policy_kwargs)   

    # callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)    
    eval_callback = EvalCallback(env, best_model_save_path=best_model_path, log_path=log_dir, eval_freq=1000,deterministic=False,verbose=1,render=False) 
    checkpoint_callback= CheckpointCallback(save_freq=10000, save_path=best_model_path,name_prefix='latest_model',verbose=2)
    callback = CallbackList([checkpoint_callback,eval_callback])
    wandb_callback = WandbCallback(verbose=1, model_save_path=log_dir, model_save_freq=5000)                   
    model.learn(total_timesteps=steps,callback=callback)                                                      
                                                                                            
    # model.save("ppo_cartpole")  # saving the model to ppo_cartpole.zip                      
    # model = PPO.load("ppo_cartpole")  # loading the model from ppo_cartpole.zip             
                                                                                            
    # obs = env.reset()                                                                       
    # for i in range(1000):                                                                   
    #     action, _state = model.predict(obs, deterministic=False)                             
    #     obs, reward, done, info = env.step(action) 
    #     print(info)                                         
    #     # env.render(mode='human')                                                                        
    #     if done:                                                                            
    #         obs = env.reset()


if __name__ == '__main__':
    args = parse_args()
    # get reward scheme to log in config
    rewards = get_reward_scheme(args.play_style)
    config ={
        "lvl": args.env,
        "play_style": args.play_style,
        "reward_scheme": args.reward_scheme,
        "rewards": rewards,
        "player_max_hp": 0,
        "exp_type": args.exp_type,
        "action_type": args.action_type,
        "action_space_type": args.action_space_type,
        "obs_type": args.obs_type,
        "base_path": args.base_path,
        "algorithm": args.algo,
    }
    print(config)
    # return
    log_dir = f"./logs/groupShap_64_01"
    exp = f"{config['lvl']}_{config['play_style']}_{config['reward_scheme']}_{config['exp_type']}_{config['algorithm']}"
    log_dir = os.path.join(log_dir,exp)
    log_dir, name= uniquify(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project="gym-md_groupShap_64_01", sync_tensorboard=True, config=config, name=name)
    main(lvl= args.env, config =config,steps=int(1e6),log_dir=log_dir)
    wandb.finish()
    

