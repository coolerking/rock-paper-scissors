# -*- coding: utf-8 -*-
"""
じゃんけん対戦するOpenAI Gym 環境クラスを提供するモジュール
"""
import random
import numpy as np
try:
    import gym
    from gym import spaces
    from agents import BasePlayer
except:
    raise

class Playground(gym.Env):
    """
    じゃんけん対戦場をあらわす環境クラス
    """
    metadata = {'render.modes': ['console', 'ansi', 'json']}    # render モード

    REWARD_WIN = 10.0   # 勝ち確定時報酬値
    REWARD_LOSE = -10.0 # 負け確定時報酬値
    REWARD_DRAW = 0.0   # あいこ時報酬値

    def __init__(self, enemy_player, max_steps=10, history_length=100):
        """
        対戦相手となるプレイヤー、あいこ上限回数を指定する。
        観測データ空間、行動空間を定義する。
        引数：
            enemy_player        対戦相手となるPlayerインスタンス
            max_steps           あいこ上限回数
            history_length      履歴件数
        戻り値：
            なし
        """
        super().__init__()
        # 対戦相手となるプレイヤー
        self.enemy_player = enemy_player
        # あいこによる再ステップの上限
        self.max_steps = max_steps
        # 出した手履歴数
        self.history_length = history_length

        # 観測データ空間
        # [ 自分の過去の行動リスト, 敵の過去の行動リスト ]
        self.observation_space = spaces.Box(
            min(BasePlayer.ALL_ACTION), max(BasePlayer.ALL_ACTION),
            (2, self.history_length,), dtype=np.float32)
        # 観測データ初期化
        self.observation = Playground.init_observation(self.history_length)

        # 行動空間
        # int型スカラー値（0:グー、1:パー、2:チョキ）
        self.action_space = spaces.Discrete(len(BasePlayer.ALL_ACTION))

        # その他情報
        self.info = {
            'env_id':       'RockPaperScissors-v0',                 # env id
            'enemy_player': self.enemy_player.__class__.__name__,   # 対戦オブジェクトクラス名
            'max_steps':    self.max_steps,                         # あいこによる再ステップの上限
            'episode_no':   0,                                      # 現在のエピソード
            'step_no':      0,                                      # 現在のステップ
            'total_reward': 0.0,                                    # エピソード内報酬合計
        }

    def reset(self):
        """
        エピソード開始時点にリセットする。
        エピソード番号を+1、ステップ番号を0にする。
        引数：
            なし
        戻り値：
            観測データ [ 自分の過去の行動リスト, 敵の過去の行動リスト ]
        """
        self.info['episode_no'] = self.info['episode_no'] + 1
        self.info['step_no'] = -1.0
        self.info['total_reward'] = 0.0
        return self.observation

    def step(self, action):
        """
        方策が選択した行動を受け取り、環境の状態（観測データ）を更新する。
        ステップ番号を+1加算する。
        引数：
            action      自分の行動
        戻り値：
            observation 観測データ
            reward      報酬値
            done        エピソード完（真：完了、偽：あいこ）
            info        その他情報
        """
        self.info['step_no'] = self.info.get('step_no', -1) + 1
        my_action = action
        enemy_action = self.enemy_player.next_action(
            Playground.reverse_observation(self.observation))
        reward = self.compute_reward(
            my_action, enemy_action)
        done = Playground.eval_done(
            my_action, enemy_action)
        self.info['total_reward'] = self.info['total_reward'] + reward 
        self.observation = Playground.update_observation(
            self.observation, my_action, enemy_action)
        return self.observation, reward, done, self.info

    def render(self, mode='console'):
        """
        環境の状態を可視化する。
        引数：
            mode    console:標準出力、ansi:文字列返却
        戻り値：
            None:   console指定時
            文字列:  ansi指定時
        """
        msg = '+++ episode:{}, step:{}, '.format(
            str(self.info['episode_no']), str(self.info['step_no']))
        last_obs = 'recent my action: ' + str(self.observation[0][-3]) + \
            ',' + str(self.observation[0][-2]) + \
            ',' + str(self.observation[0][-1]) 
        msg = msg + last_obs
        if mode == 'console':
            print(msg)
            return None
        elif mode == 'ansi':
            return msg
        elif mode == 'json':
            return {
                'episode_no':   self.info['episode_no'],
                'step_no':      self.info['step_no'],
                'my_action':    self.observation[0][-1],
                'enemy_action': self.observation[1][-1],
                'done':         self.eval_done(self.observation[0][-1], self.observation[1][-1]),
                'reward':       self.compute_reward(self.observation[0][-1], self.observation[1][-1]),
                'total_reward': self.info['total_reward'],
            }
        else:
            raise ValueError(f'mode={mode}: no match argument')

    def set_model(self, model):
        """
        モデルを差し替える。
        引数：
            model       モデルインスタンス
        戻り値：
            なし
        """
        self.model = model

    @staticmethod
    def compute_reward(my_action, enemy_action):
        """
        報酬関数。
        自分の手、相手の手から報酬値を算出する。
        引数：
            my_action       自分の行動
            enemy_action    敵の行動
        戻り値：
            報酬値(float)
        """
        if my_action == BasePlayer.ROCK:
            if enemy_action == BasePlayer.ROCK:
                return Playground.REWARD_DRAW
            elif enemy_action == BasePlayer.PAPER:
                return Playground.REWARD_LOSE
            else:
                return Playground.REWARD_WIN
        elif my_action == BasePlayer.PAPER:
            if enemy_action == BasePlayer.PAPER:
                return Playground.REWARD_DRAW
            elif enemy_action == BasePlayer.SCISSORS:
                return Playground.REWARD_LOSE
            else:
                return Playground.REWARD_WIN
        else:
            if enemy_action == BasePlayer.SCISSORS:
                return Playground.REWARD_DRAW
            elif enemy_action == BasePlayer.ROCK:
                return Playground.REWARD_LOSE
            else:
                return Playground.REWARD_WIN

    @staticmethod
    def eval_done(my_action, enemy_action):
        """
        自分の手と相手の手から勝敗が決定したかどうか
        を返却する（エピソード完の返却）。
        引数：
            my_action       自分の行動
            enemy_action    敵の行動
        戻り値：
            done            エピソード完（真：完了、偽：あいこ）
        """
        return False if my_action == enemy_action else True

    @staticmethod
    def init_observation(history_length=100):
        """
        観測データ初期値を作成する。
        自分、敵両方の過去の手はすべてランダムに設定する。
        引数：
            history_length      履歴件数
        戻り値：
            観測データ（初期化された状態）
        """
        observation = []
        for _ in range(2):
            history = []
            for _ in range(history_length):
                history.append(random.randrange(len(BasePlayer.ALL_ACTION)))
            observation.append(history)
        return observation

    @staticmethod
    def update_observation(observation, my_action, enemy_action):
        """
        観測データを更新する。
        最古の行動を削除し、引数で与えられた行動を加える。
        引数：
            observation     観測データ（更新前）
            my_action       最新の自分の行動
            enemy_action    最新の敵の行動
        戻り値：
            observation     観測データ（更新後）
        """
        for i in range(len(observation)):
            observation[i] = observation[i][-(len(observation[i]) - 1):]
        observation[0].append(my_action)
        observation[1].append(enemy_action)
        return observation

    @staticmethod
    def reverse_observation(observation):
        """
        相手側の観測データに変換する。
        引数：
            observation         観測データ
        戻り値：
            reverse_observation 相手側の観測データ
        """
        reverse_observation = []
        for history in reversed(observation):
            reverse_observation.append(history)
        return reverse_observation


def test_playground():
    history_length=10
    class TestPlayer:
        def next_action(self, observation):
            assert(len(observation) == 2)
            for history in observation:
                assert(len(history) == history_length)
            return random.randrange(len(BasePlayer.ALL_ACTION))
    enemy_player = TestPlayer()
    env = Playground(enemy_player=enemy_player, max_steps=3, history_length=history_length)
    observation = env.reset()
    assert(observation == env.observation)
    assert(observation == Playground.reverse_observation(
        Playground.reverse_observation(env.observation)))
    my_player = TestPlayer()
    for _ in range(100):
        my_action = my_player.next_action(observation)
        observation, reward, done, info = env.step(my_action)
        env.render()
        if done:
            print(f'*** episode:{info["episode_no"]}, step:{info["step_no"]}, reward:{reward}')
            assert(observation == env.reset())

if __name__ == '__main__':
    #for _ in range(100):
    #    print(random.randrange(len(BasePlayer.ALL_ACTION)))
    test_playground()
