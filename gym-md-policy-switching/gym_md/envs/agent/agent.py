"""agent module."""
from typing import Final
from random import Random

from gym_md.envs.agent.actioner import Actioner, Actions
from gym_md.envs.agent.move_info import MoveInfo
from gym_md.envs.agent.pather import Pather
from gym_md.envs.grid import Grid
from gym_md.envs.point import Point
from gym_md.envs.setting import Setting


class Agent:
    """Agent class.

    エージェントクラス．
    Pather, Actionerを持つ．
    """

    def __init__(self, grid: Grid, setting: Setting, random: Random):
        self.grid: Final[Grid] = grid
        self.setting: Final[Setting] = setting
        self.random = random
        self.path: Final[Pather] = Pather(grid=grid, setting=setting)
        self.actioner: Final[Actioner] = Actioner(setting=setting, random=self.random)
        self.hp: int = setting.PLAYER_MAX_HP
        self.y: int = -1
        self.x: int = -1
        self.reset()

    def reset(self) -> None:
        """エージェントの初期化をする.
        Reset/Initialize the agent

        Notes
        -----
        リセット形式にしている．
        これは，一回一回Agentインスタンスを作ると，rendererなどの参照とずれてしまうため
        - in reset format
        - This is because if you create an Agent instance once, it will be misaligned with references such as renderer

        """
        self.hp = self.setting.PLAYER_MAX_HP
        init_pos: Final[Point] = self._init_player_pos()
        self.y: int = init_pos[0]
        self.x: int = init_pos[1]

    def select_action(self, actions: Actions) -> str:
        """行動を選択する.
        Choose an action
        Notes
        -----
        行動を選択したときに，その行動が実行できない可能性がある．
        （マスがない可能性など）
        - When an action is selected, there is a possibility that the action cannot be executed
        (possibility that there are no squares, etc.)

        そのため，行動列すべてを受け取りできるだけ値の大きい実行できるものを選択する．
        **選択する**であり何も影響を及ぼさないことに注意．
        -Therefore, we select the one that can be executed with a large value as long as it can receive all the action sequences.
        Note that it **selects** and has no effect.

        Parameters
        ----------
        actions: Actions
            行動列

        Returns
        -------
        str
            選択した行動IDを返す
            Returns the selected action id
        """
        safe_info: Final[MoveInfo] = self.path.get_moveinfo(
            y=self.y, x=self.x, safe=True
        )
        unsafe_info: Final[MoveInfo] = self.path.get_moveinfo(
            y=self.y, x=self.x, safe=False
        )
        selected_action: Final[str] = self.actioner.select_action(
            actions=actions, safe_info=safe_info, unsafe_info=unsafe_info
        )
        return selected_action

    def take_action(self, action: str) -> None:
        """選択された行動を実行する.
        Perform the selected action

        - 移動を行う
        - 体力の増減が実行される
        - make a move
        - Physical strength increase/decrease is executed (hp)

        Notes
        -----
        エージェントの自身の更新は行うが
        Gridなどの更新は行わないことに注意する．
        Although the agent updates itself
        Note that it does not update Grid etc.

        Parameters
        ----------
        action: str
            選択済み行動ID
            Selected action ID
        """
        safe_info: Final[MoveInfo] = self.path.get_moveinfo(
            y=self.y, x=self.x, safe=True
        )
        unsafe_info: Final[MoveInfo] = self.path.get_moveinfo(
            y=self.y, x=self.x, safe=False
        )

        if "SAFELY" in action:
            to = safe_info[action[0]]
        else:
            to = unsafe_info[action[0]]

        self.y, self.x = to
        self.be_influenced(y=self.y, x=self.x)

    def is_dead(self) -> bool:
        return self.hp <= 0

    def is_exited(self) -> bool:
        return self.grid[self.y, self.x] == self.setting.CHARACTER_TO_NUM["E"]

    def be_influenced(self, y: int, x: int) -> None:
        """移動したプレイヤーに影響を与える.
        Affect moved players

        体力の増減を行う
        Increase/decrease health

        Parameters
        ----------
        y:int
            移動後のy
        x:int
            移動後のx
        """
        if self.grid[y, x] == self.setting.CHARACTER_TO_NUM["M"]:
            attack = self.setting.ENEMY_POWER
            if self.setting.IS_ENEMY_POWER_RANDOM:
                attack = self.random.randint(
                    self.setting.ENEMY_POWER_MIN, self.setting.ENEMY_POWER_MAX
                )
            self.hp -= attack
        if self.grid[y, x] == self.setting.CHARACTER_TO_NUM["P"]:
            self.hp += self.setting.POTION_POWER
            if self.setting.IS_PLAYER_HP_LIMIT:
                self.hp = min(self.hp, self.setting.PLAYER_MAX_HP)

    def change_player_hp(self, previous_hp: int) -> None:
        """前回のステージのHPに更新する。"""
        self.hp = min(previous_hp, self.setting.PLAYER_MAX_HP)

    def _init_player_pos(self) -> Point:
        """プレイヤーの座標を初期化して座標を返す.
        Initialize the player's coordinates and return the coordinates.
        
        Notes
        -----
        初期座標を表すSを'.'にメソッド内で書き換えていることに注意する．
        Note that S representing the initial coordinates is rewritten to '.' in the method

        Returns
        -------
        Point
            初期座標を返す
            return the initial coordinates
        """
        for i in range(self.grid.H):
            for j in range(self.grid.W):
                if self.grid[i, j] == self.setting.CHARACTER_TO_NUM["S"]:
                    self.grid[i, j] = self.setting.CHARACTER_TO_NUM["."]
                    return i, j
