"""actioner module."""
from typing import List, Tuple
from random import Random

from gym_md.envs.agent.move_info import MoveInfo
from gym_md.envs.setting import Setting

Actions = List[float]


class Actioner:
    """Actioner class.

    アクションの配列を受け取り
    実行するアクションを決定する

    Receive an array of actions
    Decide what action to take
    """

    def __init__(self, setting: Setting, random: Random):
        self.setting: Setting = setting
        self.random = random

    def select_action(
        self, actions: Actions, safe_info: MoveInfo, unsafe_info: MoveInfo
    ) -> str:
        """実行するアクションのIDを決定する.
        Determine the ID of the action to perform

        Notes
        -----
        同じ選択希望値の場合は，ランダムに選択する．
        In the case of the same selection desired value, select randomly.

        Parameters
        ----------
        actions: Actions
            エージェントが出力した各アクションの希望値
            Desired value for each action output by the agent
        safe_info: MoveInfo
            各タイルに移動するための次の座標の辞書
            Dictionary of next coordinates to go to each tile
        unsafe_info
            各タイルに移動するための次の座標の辞書
            Dictionary of next coordinates to go to each tile

        Returns
        -------
        str
            実行するアクションのID
            ID of the action to perform
        """
        actions_idx: List[Tuple[float, int]] = [
            (actions[i], i) for i in range(len(actions))
        ]
        actions_idx.sort(key=lambda z: (-z[0], -z[1]))

        i, j, n = 0, 0, len(actions_idx)
        while i < n:
            while j < n and actions_idx[i][0] == actions_idx[j][0]:
                j += 1
            idxs = [k for k in range(i, j)]
            self.random.shuffle(idxs)

            for k in idxs:
                action_id = actions_idx[k][1]
                action_name = self.setting.NUM_TO_ACTION[action_id]
                if "SAFELY" in action_name:
                    to = safe_info[action_name[0]]
                else:
                    to = unsafe_info[action_name[0]]

                # action_nameを実行できない
                # unable to execute action_name
                if to == (-1, -1):
                    continue

                # 実行するアクション文字列
                # the action string to perform
                return self.setting.NUM_TO_ACTION[action_id]

            i = j

    def select_policy(
        self, actions: Actions, safe_info: MoveInfo, unsafe_info: MoveInfo
    ) -> str:
        """実行するアクションのIDを決定する.
        Determine the ID of the action to perform

        Notes
        -----
        同じ選択希望値の場合は，ランダムに選択する．
        In the case of the same selection desired value, select randomly.

        Parameters
        ----------
        actions: Actions
            エージェントが出力した各アクションの希望値
            Desired value for each action output by the agent
        safe_info: MoveInfo
            各タイルに移動するための次の座標の辞書
            Dictionary of next coordinates to go to each tile
        unsafe_info
            各タイルに移動するための次の座標の辞書
            Dictionary of next coordinates to go to each tile

        Returns
        -------
        str
            実行するアクションのID
            ID of the action to perform
        """
        actions_idx: List[Tuple[float, int]] = [
            (actions[i], i) for i in range(len(actions))
        ]
        actions_idx.sort(key=lambda z: (-z[0], -z[1]))

        i, j, n = 0, 0, len(actions_idx)
        while i < n:
            while j < n and actions_idx[i][0] == actions_idx[j][0]:
                j += 1
            idxs = [k for k in range(i, j)]
            self.random.shuffle(idxs)

            for k in idxs:
                action_id = actions_idx[k][1]
                action_name = self.setting.NUM_TO_ACTION[action_id]
                if "SAFELY" in action_name:
                    to = safe_info[action_name[0]]
                else:
                    to = unsafe_info[action_name[0]]

                # action_nameを実行できない
                # unable to execute action_name
                if to == (-1, -1):
                    continue

                # 実行するアクション文字列
                # the action string to perform
                return self.setting.NUM_TO_ACTION[action_id]

            i = j
