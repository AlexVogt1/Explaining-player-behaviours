from os import path
from random import Random
from typing import DefaultDict, Final, List, Tuple, Dict
from PIL import Image
import numpy
import gym
from collections import defaultdict

from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.policy_agent import PolicyAgent
from gym_md.envs.agent.agent import Agent
# from gym_md.envs.agent.companion_agent import CompanionAgent, DirectionalAgent
# from gym_md.envs.renderer.collab_renderer import CollabRenderer
from gym_md.envs.agent.actioner import Actions
from gym_md.envs.grid import Grid
from gym_md.envs.point import Point
from gym_md.envs.renderer.renderer import Renderer
from gym_md.envs.setting import Setting
from gym_md.envs.definition import POLICY_ACTIONS
from gym_md.envs.grid import Grid

# from gym_md.envs.renderer.collab_renderer import CollabRenderer

#TODO change Action Space
#TODO Create function to cread in policies from model.zip and use is as actions
#TODO create function that creates the policy
#TODO fix rendering (add rendering function)
#TODO input to dict
# config = {
#     "action_type": "policy",
#     "action_space_type": "discrete",
#     "obs_type": "grid",
#     "base_path": "./play_style_models/base/",
# }


class MdPolicyEnv(MdEnvBase):
    def __init__(self, stage_name: str, config:dict, action_type='path',space_type='disc', policy_path = "./play_style_models/grid_base_12x12/",debug =True):
        super().__init__(stage_name)
        print(config)
        self.random = Random()
        self.debug = debug
        self.config = config
        self.space_type = config["action_type"]
        self.action_type = config["action_type"]
        self.action_space_type = config["action_space_type"]
        self.observation_type = config["obs_type"]
        self.base_path = config["base_path"]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)

        self.grid: Grid = Grid(self.stage_name, self.setting)
        self.info: DefaultDict[str, int] = defaultdict(int)
        # self.renderer: Final[Renderer] = Renderer(self.grid, self.agent, self.setting)
        if self.observation_type == "grid":
            self.observation_space = gym.spaces.Box(low = 0 , high = 7,  shape = (self.grid.H, self.grid.W), dtype =numpy.int32)
        elif self.observation_type == "distance":
            self.observation_space = gym.spaces.Box(
                low=0, high=self.setting.DISTANCE_INF, shape=(8,), dtype=numpy.int32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=self.setting.DISTANCE_INF, shape=(8,), dtype=numpy.int32
            )
        self.action_type = config["action_type"]
        if self.action_type=='policy' or self.action_type == 'switch':
            self.agent: PolicyAgent =PolicyAgent(self.grid, self.setting, self.random, self.action_type, path=self.base_path)
            self.n_actions: Final[int]= len(self.agent.actions)
            #TODO make action space be length of policy_names wrapped i.e. length of policy actions
            if self.action_space_type=='discrete':
                self.action_space = gym.spaces.Discrete(n=self.n_actions, start=0)
            else:
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_actions,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)
            try: 
                self.setting.POLICY_ACTIONS = self.agent.actions
                self.setting.POLICY_ACTION_TO_NUM = Setting.list_to_dict(self.setting.POLICY_ACTIONS)
                self.setting.NUM_TO_POLICY_ACTION = Setting.swap_dict(self.setting.POLICY_ACTION_TO_NUM)
            except:
                print("in except in init")
                self.setting.POLICY_ACTIONS = POLICY_ACTIONS
                self.setting.POLICY_ACTION_TO_NUM = Setting.list_to_dict(self.setting.POLICY_ACTIONS)
                self.setting.NUM_TO_POLICY_ACTION = Setting.swap_dict(self.setting.POLICY_ACTION_TO_NUM)
                # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # 4 for now becuase we are doing basic determinsitic policies (maybe use function to determine action space based on)                
        else:
            # print('inside else')
            self.agent: Agent = Agent(self.grid, self.setting, self.random)
            if self.action_space_type == "discrete":
                # print("making discrete_actions")
                self.action_space = gym.spaces.Discrete(n =7 ,start=0)
            else:
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.renderer1: Final[Renderer] =Renderer(self.grid, self.agent, self.setting) 

    def reset(self) -> List[int]:
        super().reset()
        self.agent.reset()
        self._update_grid()
        if self.observation_type == "grid":
            return self._get_grid_observation()
        else:
            return self._get_observation()
    
    def step(self, actions):
        # print("Start step")
        # print(actions)
        # print(self.setting.NUM_TO_POLICY_ACTION)
        if self.observation_type == 'grid':
            self.agent.obs = self._get_grid_observation()
        else:
            self.agent.obs = self._get_observation()

        # print(f"\n\nCurrent observation: {self.agent.obs}")
        if self.action_space_type == 'discrete':
            # print("discrete Actions")
            action = self.setting.NUM_TO_POLICY_ACTION[actions]
        else:
            # print(" continuous action")
            action = self.agent.select_action(actions)
        # print("take action")
        self.agent.take_action(action)
        reward =self._get_reward()
        done = self._is_done()
        self.info = self._get_info(self.info, action)
        # print(self.observation_space)
        if self.observation_type == "grid":
            # print("grid obs")
            observation = self._get_grid_observation()
        else:
            observation: List[int] =self._get_observation()
        self._update_grid()
        return observation, reward, done, self.info

    def change_reward_values(self, rewards: Dict[str, int]) -> None:
        """報酬を変更する."""
        super().change_reward_values(rewards=rewards)

    def restore_reward_values(self) -> None:
        self.setting.restore_reward_values()

    def change_player_hp(self, previous_hp: int) -> None:
        """前回のステージのHPに更新する。"""
        self.agent.change_player_hp(previous_hp)

    def set_random_seed(self, seed: int) -> None:
        """Seed 値を更新する."""
        self.random.seed(seed)

    def render(self, mode="human") -> Image:
        """画像の描画を行う.

        Notes
        -----
        画像自体も取得できるため，保存も可能.

        Returns
        -------
        Image
        """
        return self.renderer1.render(mode=mode)
    
    def _get_grid_observation(self):
        self._update_grid()
        grid = self.grid.g
        return numpy.array(grid, dtype= numpy.int32)
        
    
    def _get_observation(self) -> List[int]:
        
        return super()._get_observation()
        
    def _is_done(self) -> bool:
        """ゲームが終了しているか.
           Returns a boolean indicating whether the game is over or not.

        Returns
        -------
        bool
        """
        return super()._is_done()
    
    def _update_grid(self) -> None:
        """グリッドの状態を更新する.
           Update the state of the grid.

        Notes
        -----
        メソッド内でグリッドの状態を**直接更新している**ことに注意
        Note that we are **directly updating** the state of the grid
        in the method.

        Returns
        -------
        None
        """
        super()._update_grid()


    def generate(self, mode="human") -> Image:
        """画像を生成する.
           Generate the world image.

        Notes
        -----
        画像の保存などの処理はgym外で行う.
        Processing such as image saving is performed
        outside the gym environment.

        Returns
        -------
        Image
        """
        return self.renderer1.generate(mode=mode)
