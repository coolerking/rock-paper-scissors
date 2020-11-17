# -*- coding: utf-8 -*-
"""
じゃんけん対戦するOpenAI Gym 環境クラスを提供するモジュール
"""
import random
import numpy as np
import gym
from gym import spaces

class RockPaperScissorsEnv(gym.Env):
    """
    OpenAI Gym 準拠のじゃんけん対戦環境クラス。
    必要最小限の実装のみ。
    """
    def __init__(self, player):
        """
        方策側の相手となる環境側プレイヤーを
        インスタンス変数へ格納し、
        行動空間・観測空間の定義を行い、
        観測の初期化を行う。
        引数：
            player  環境側プレイヤーインスタンス
        戻り値：
            なし
        """
        super().__init__()
        self.player = player
        # 行動空間：0=グー、1=パー、2=チョキ
        self.action_space = spaces.Discrete(2)
        # 観測空間：過去100件分の[方策側行動, 環境側行動]
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(100, 2), dtype=np.int8)
        # 観測初期化
        self.observation = self.init_observation()

    def reset(self):
        """
        エピソード開始時の観測を取得する。
        インスタンス変数observationの値をそのまま返却する。
        引数：
            なし
        戻り値：
            観測 [ 自分の行動, 敵の行動 ]×100
        """
        return self.observation

    def step(self, action):
        """
        方策が選択した行動を受け取り、更新後の観測、報酬、
        エピソード完、その他情報(空の辞書)を返却する。
        引数：
            action      自分の行動
        戻り値：
            observation 更新後の観測
            reward      報酬値
            done        エピソード完（真：完了、偽：あいこ）
            info        その他情報（空の辞書）
        """
        policy_action = int(action)
        env_action = int(self.player.predict(self.observation))
        self.observation = self.update_observation(
            self.observation, policy_action, env_action)
        reward = self.calc_reward(policy_action, env_action)
        done = self.is_done(policy_action, env_action)
        return self.observation, reward, done, {}

    @staticmethod
    def init_observation():
        """
        エピソード開始時の観測を取得する。
        引数：
            なし
        戻り値：
            observation 観測（初期値）
        """
        # 観測初期化
        observation = []
        for _ in range(100):
            #　乱数で初期化
            observation.append([
                random.randrange(2), 
                random.randrange(2)])
        return observation

    @staticmethod
    def update_observation(observation, policy_action, env_action):
        """
        観測の先頭（最古の両者の手）を削除し、末尾に最新の両者の手を追加して
        返却する。
        引数：
            observation         更新対象となる観測
            policy_action       方策側の行動
            env_action          環境側の行動
        戻り値：
            observation         更新後の観測
        """
        observation.pop(0)
        observation.append([policy_action, env_action])
        return observation

    @staticmethod
    def is_done(policy_action, env_action):
        """
        エピソード完了かどうかを判別する。
        引数：
            policy_action   方策側の行動
            env_action      環境側の行動
        戻り値：
            done            エピソード完（真：完了、偽：あいこ）
        """
        return False if policy_action == env_action else True

    @staticmethod
    def calc_reward(policy_action, env_action):
        """
        報酬関数。
        方策側の行動、環境側の行動から報酬値を算出する。
        引数：
            policy_action   方策側の行動
            env_action      環境側の行動
        戻り値：
            方策側が受け取る報酬値(float)
        """
        if policy_action == 0:  # 方策側：グー
            if env_action == 1:     # 環境側：パー
                return -10
            elif env_action == 2:   # 環境側：チョキ
                return 10
            else:                   # 環境側：グー
                return -1
        elif policy_action == 1: # 方策側：パー
            if env_action == 1:     # 環境側：パー
                return -1
            elif env_action == 2:   # 環境側：チョキ
                return -10
            else:                   # 環境側：グー
                return 10
        else:                    #　方策側：チョキ
            if env_action == 1:     # 環境側：パー
                return 10
            elif env_action == 2:   # 環境側：チョキ
                return -1
            else:                   # 環境側：グー
                return -10

class EvalEnv(RockPaperScissorsEnv):
    """
    じゃんけんAI用方策の評価用環境クラス。
    """
    # render モード
    metadata = {'render.modes': ['console', 'ansi', 'json']}
    def __init__(self, player):
        """
        インスタンス変数infoを初期化する。
        引数：
            player  環境側プレイヤーインスタンス
        戻り値：
            なし
        """
        super().__init__(player)
        self.info = {
            'env_id':       'RockPaperScissors-v0',         # env id
            'enemy_player': self.player.__class__.__name__, # 対戦オブジェクトクラス名
            'episode_no':   0,                              # 現在のエピソード
            'step_no':      0,                              # 現在のステップ
            'total_reward': 0.0,                            # エピソード内報酬合計
        }

    def reset(self):
        """
        エピソード開始時の観測を取得する。
        インスタンス変数observationの値をそのまま返却する。
        引数：
            なし
        戻り値：
            観測 [ 自分の行動, 敵の行動 ]×100
        """
        self.info['episode_no'] = self.info['episode_no'] + 1
        self.info['step_no'] = -1.0
        self.info['total_reward'] = 0.0
        return super().reset()

    def step(self, action):
        """
        親クラスのstep()を実行し、infoとして実行中ステップ情報を
        返却する
        引数：
            action      自分の行動
        戻り値：
            observation 更新後の観測
            reward      報酬値
            done        エピソード完（真：完了、偽：あいこ）
            info        その他情報（ステップ情報）
        """

        observation, reward, done, _ = super().step(action)
        self.info['step_no'] = self.info.get('step_no', -1) + 1
        self.info['total_reward'] = self.info['total_reward'] + reward 
        return observation, reward, done, self.info

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
        last_obs = 'recent policy action: ' + str(self.observation[-3][0]) + \
            ',' + str(self.observation[-2][0]) + \
            ',' + str(self.observation[-1][0]) 
        msg = msg + last_obs
        if mode == 'console':
            print(msg)
            return None
        elif mode == 'ansi':
            return msg
        elif mode == 'json':
            return {
                'episode_no':       self.info['episode_no'],
                'step_no':          self.info['step_no'],
                'policy_action':    self.observation[-1][0],
                'env_action':       self.observation[-1][1],
                'done':             self.is_done(self.observation[-1][0], self.observation[-1][1]),
                'reward':           self.calc_reward(self.observation[-1][0], self.observation[-1][1]),
                'total_reward':     self.info['total_reward'],
            }
        else:
            raise ValueError(f'mode={mode}: no match argument')

class Player:
    """
    プレイヤー基底クラス。
    """
    def predict(self, observation):
        """
        引数observationをもとに次の行動を選択する。
        本実装ではobservationを一切使用せずに、ランダムに行動を選択する。
        引数：
            observation     観測（使用しない）
        戻り値：
            ランダムに選択された行動
        """
        return random.randrange(2)

class ProbPlayer(Player):
    """
    コンストラクタで渡された各手の確率に従ってランダムに手を出す
    プレイヤークラス。
    """
    def __init__(self, prob_list=[0.333, 0.333, 0.334]):
        """
        各手の確率リストをインスタンス変数へ格納する。
        引数：
            prob_list   各手の確率
        戻り値：
            なし
        """
        self.prob_list = [
            float(prob_list[0])/float(sum(prob_list)),
            float(prob_list[1])/float(sum(prob_list)),
            float(prob_list[2])/float(sum(prob_list)),
        ]
        
    def predict(self, observation):
        """
        引数observationをもとに次の行動を選択する。
        本実装ではobservationを一切使用せずに、
        各手の確率に従ってランダムに行動を選択する。
        引数：
            observation     観測（使用しない）
        戻り値：
            各手の確率に従ってランダムに選択された行動
        """
        value = random.uniform(0.0, 1.0)
        if value <= self.prob_list[0]:
            return 0    # グー
        elif value <= (self.prob_list[0] + self.prob_list[1]):
            return 1    # パー
        else:
            return 2    # チョキ

class EnemyPlayer(ProbPlayer):
    """
    1/3の確率でグー・パー・チョキを選択するプレイヤー。
    """
    def __init__(self):
        """
        prob_list の要素がすべて1/3として親クラスの
        コンストラクタを呼び出す。
        引数：
            なし
        戻り値：
            なし
        """
        super().__init__(prob_list=[1.0/3.0, 1.0/3.0, 1.0/3.0])

class JurinaPlayer(Player):
    """
    常に同じ手を出すプレイヤー。
    """
    def __init__(self, action=1):
        """
        常に出す手をインスタンス変数へ格納する。
        引数：
            action      常に出す手（デフォルト：パー）
        """
        self.action = action

    def predict(self, observstion):
        """
        引数observationをもとに次の行動を選択する。
        本実装ではobservationを一切使用せずに、
        常に同じ行動を選択する。
        引数：
            observation     観測（使用しない）
        戻り値：
            コンストラクタで指定された行動
        """
        return self.action

class AIPlayer(Player):
    """
    学習済みモデルを使って行動を決めるプレイヤー。
    学習済みモデルと対戦評価する際に使用する。
    """
    def __init__(self, model):
        """
        学習済みモデルをコンストラクタに指定する。
        引数：
            model       学習済みモデルクラスのインスタンス
        戻り値：
            なし
        """
        self.model = model
    
    def predict(self, observation):
        """
        引数observationをもとに次の行動を選択する。
        本実装では学習済みモデルクラスのpredictメソッドを使って
        行動を選択する。
        引数：
            observation     観測
        戻り値：
            学習済みモデルが選択した行動
        """
        return int(self.model.predict(observation)[0])

# テスト

def test_observation():
    env = RockPaperScissorsEnv(Player())
    assert(len(env.observation)==100)
    assert(len(env.observation[0])==2)
    for i in range(100):
        assert(env.observation[i][0] >= 0 and env.observation[i][0]<=2)
        assert(env.observation[i][1] >= 0 and env.observation[i][1]<=2)
    oldest_policy_action = env.observation[1][0]
    oldest_env_action = env.observation[1][1]
    newest_policy_action = 3
    newest_env_action = 4
    env.update_observation(env.observation, newest_policy_action, newest_env_action)
    assert(env.observation[0][0]==oldest_policy_action)
    assert(env.observation[0][1]==oldest_env_action)
    assert(env.observation[99][0]==newest_policy_action)
    assert(env.observation[99][1]==newest_env_action)

def test_is_done():
    env = RockPaperScissorsEnv(Player())
    assert(env.is_done(0, 0) == False)
    assert(env.is_done(0, 1) == True)
    assert(env.is_done(0, 2) == True)
    assert(env.is_done(1, 0) == True)
    assert(env.is_done(1, 1) == False)
    assert(env.is_done(1, 2) == True)
    assert(env.is_done(2, 0) == True)
    assert(env.is_done(2, 1) == True)
    assert(env.is_done(2, 2) == False)

def test_calc_reward():
    env = RockPaperScissorsEnv(Player())
    for i in range(3):
        assert(env.calc_reward(i, i)==-1)
    assert(env.calc_reward(0, 1)==-10)
    assert(env.calc_reward(0, 2)==10)
    assert(env.calc_reward(1, 0)==10)
    assert(env.calc_reward(1, 2)==-10)
    assert(env.calc_reward(2, 0)==-10)
    assert(env.calc_reward(2, 1)==10)

def test_player():
    player = Player()
    prob_player = ProbPlayer(prob_list=[1, 7, 2])
    prob_player_pa = ProbPlayer(prob_list=[0.0, 1.0, 0.0])
    jurina_player = JurinaPlayer(action=1)
    for _ in range(100):
        assert(player.predict(None) in [0, 1, 2])
        assert(prob_player.predict(None) in [0, 1, 2])
        assert(prob_player_pa.predict(None)==1)
        assert(jurina_player.predict(None)==1)

def test_reset():
    env = RockPaperScissorsEnv(ProbPlayer())
    for _ in range(100):
        assert(env.reset() == env.observation)

if __name__ == '__main__':
    test_observation()
    test_is_done()
    test_calc_reward()
    test_player()
    test_reset()
