# -*- coding: utf-8 -*-
"""
じゃんけん対戦の環境クラスモジュール

(C) Tasuku Hori, 2020
"""
import numpy as np

import gym
from gym import spaces
from agents import RandomPlayer

class RPSEnv(gym.Env):
    """
    OpenAI Gym 準拠のじゃんけん対戦環境クラス
    """
    metadata = {'render.modes': ['console', 'ansi']}

    def __init__(self, player=None, max_steps=10):
        """
        じゃんけん対戦者を引数に指定して対戦環境を構築する。
        引数：
            player      対戦相手となるプレイヤー
            max_steps   あいこを何回くりかえすか
        戻り値：
            なし
        """
        super().__init__()
        # 対戦オブジェクト
        self.player = RandomPlayer() if player is None else player
        # あいこによる再ステップの上限
        self.max_steps = max_steps
        # 過去10回の
        # その他情報
        self.info = {
            'env_id':       'RockPaperScissors-v0',         # env id
            'play':         self.player.__class__.__name__, # 対戦オブジェクトクラス名
            'max_steps':    self.max_steps,                 # あいこによる再ステップの上限
            'episode_no':   -1,                             # 現在のエピソード
            'step_no':      0,                              # 現在のステップ
            'total_reward': 0.0,                            # エピソード内報酬合計
        }
        # 行動空間 0:グー、1:パー、2:チョキ
        self.action_space = spaces.Discrete(len(self.player.ALL_ACTION))
        # 観測データ 
        self.observation_space = spaces.Box(
            min(self.player.ALL_ACTION), max(self.player.ALL_ACTION),
            (len(self.player.observation),), dtype=np.float32)

    def reset(self):
        """
        次のエピソード開始状態にする。
        引数：
            なし
        戻り値：
            observation     観測データ（過去の相手の行動リスト）
        """
        self.info['episode_no'] = self.info['episode_no'] + 1
        self.info['step_no'] = 0
        self.info['total_reward'] = 0.0
        return self.player.observation

    def step(self, action):
        """
        選択された行動により環境クラス上の状態を最新にする。
        引数：
            action      実行を選択したW/Fレコードのindex
        戻り値：
            最新の状態
            報酬値
            エピソード完了フラグ
            その他情報
        """
        self.info['step_no'] = self.info.get('step_no', -1) + 1
        result, done, obs = self.player.pon(action)
        reward, done = self.get_reward(result, done)
        return obs, reward, done, self.info
    
    def render(self, mode='console'):
        """
        環境の状態を可視化する。
        引数：
            mode    console:標準出力、ansi:文字列返却
        戻り値：
            None: console指定時
            文字列:ansi指定時
        """
        msg = '+++ episode:{}, step:{}\n'.format(
            str(self.info['episode_no']), str(self.info['step_no']))
        last_obs = str(self.player.observation[-3]) + \
            ',' + str(self.player.observation[-2]) + \
            ',' + str(self.player.observation[-1]) 
        msg = msg + '\n' + last_obs + '\n'
        if mode == 'console':
            print(msg)
            return None
        elif mode == 'ansi':
            return msg
        else:
            raise ValueError(f'mode={mode}: no match argument')
    
    def get_reward(self, result, done):
        """
        結果およびエピソード完をもとに報酬値を決定する。
        引数：
            result      結果
            done        エピソード完
        戻り値：
            reward      報酬
            done        エピソード完
        """
        reward = result * 10.0
        if not done:
            done = (self.info['max_steps'] <= self.info['step_no'])
            if done:
                reward = -1.0
        return reward, done

def test_rps_env():
    """
    RPSEnvの疎通テスト
    """
    from game import ProbDistPlay
    enemy_player = ProbDistPlay(prob_list=[0.25, 0.5, 0.25])
    try:
        from stable_baselines3 import PPO
    except:
        from stable_baselines import PPO
    model = PPO.load('rps_mlp_ppo_random')
    from agents import PolicyPlayer
    my_player = PolicyPlayer(model=model)
    env = RPSEnv(player=enemy_player)
    env.reset()
    for _ in range(100):
        my_action = my_player.get_action()
        _, _, done, _ = env.step(my_action)
        env.render()
        if done:
            env.reset()

if __name__ == '__main__':
    test_rps_env()