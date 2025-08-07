from email import policy
import os
from pyexpat import model
from typing import Final, List
import torch
import numpy as np
import gym
import gym_md

from stable_baselines3 import PPO
from gym_md.envs.setting import Setting
from gym_md.helper.util import list_files

# class ppoWrapper:
#TODO create dictionary for PolicyName-model pairs

# wrapper class 
    # - wraps all the models 

class PolicyWapper:
    def __init__(self, path = '../play_style_models/grid_base_12x12/'):
        self.path: Final[str]= path
        self.path_list: Final[List[str]] = list_files(path)
        self.play_style_list: Final[List[str]]= [os.path.splitext(x)[0] for x in self.path_list]
        self.model_paths: Final[List[str]] = [F"{self.path}/{s}"for s in self.play_style_list]
        self.n_models: Final[int] = len(self.model_paths)
        if torch.cuda.is_available:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.policy_names = self.gen_policy_names()
        self.wrapped_models = self.build_all_models(path)


    def get_model_paths(self):
        play_style_list = [os.path.splitext(x)[0] for x in self.path_list]
        model_paths = [F"{self.path}/{s}"for s in play_style_list]
        return model_paths
    
    def gen_policy_names(self):
        play_style_list = [os.path.splitext(x)[0] for x in self.path_list]
        policy_names = [x.upper() for x in play_style_list]
        return policy_names
    
    def build_model(self, model_path):
        return PPO.load(model_path,device=self.device)

    def build_all_models(self, path):
        #TODO change to dict
        models_list = []
        for model in self.model_paths:
            models_list.append(self.build_model(model))
        models = dict(zip(self.policy_names, models_list))
        return models
    