# -*- coding: utf-8 -*-
import random

class RandomPlayer:
    """
    乱数で手を決めるエージェント
    """
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    ALL_ACTION = [ROCK, PAPER, SCISSORS]
    WIN = 1
    DRAW = 0
    LOSE = -1
    ALL_RESULT = [WIN, DRAW, LOSE]

    def __init__(self, observation_length=100):
        """
        観測データ長を設定し、観測データを乱数で初期化する。
        引数：
            observation_length  観測データ長
        """
        # 観測データ長
        self.observation_length = observation_length
        # 観測データ：過去の出した行動の履歴
        # 初期状態がないのでランダムで履歴を作成
        self.observation = []
        for _ in range(self.observation_length):
            self.observation.append(self.get_action())

    def eval(self, my_action, target_action):
        """
        対戦結果を評価し、結果を返す。
        引数：
            my_action       選択した行動
            target_action   ゲーム側が選択した行動
        戻り値：
            結果            -1:負け、0:あいこ、1:勝ち
        """
        # 自分の行動を履歴へ格納
        self.observation = self.observation[-(self.observation_length - 1):]
        self.observation.append(my_action)
        
        # じゃんけんの判定
        if my_action == self.ROCK:
            if target_action == self.ROCK:
                return self.DRAW
            elif target_action == self.PAPER:
                return self.LOSE
            else:
                return self.WIN
        elif my_action == self.PAPER:
            if target_action == self.PAPER:
                return self.DRAW
            elif target_action == self.SCISSORS:
                return self.LOSE
            else:
                return self.WIN
        else:
            if target_action == self.SCISSORS:
                return self.DRAW
            elif target_action == self.ROCK:
                return self.LOSE
            else:
                return self.WIN

    def get_action(self):
        """
        プレイヤーの行動を返却する。
        本メソッドはランダムに行動を選択するため、
        サブクラスでオーバライドして行動の傾向を変更することができる。
        引数：
            なし
        戻り値：
            行動    0:グー、2:ちょき、1:パー
        """
        return random.randrange(max(self.ALL_ACTION))
    
    def pon(self, target_action):
        """
        じゃんけんを１回実行し結果を返却する。
        引数：
            target_action   自分が出した手
        戻り値：
            result      0:あいこ、1:勝ち、-1:負け
            done        エピソード完
            observation 観測
        """
        result = self.eval(self.get_action(), target_action)
        done = (result != 0)
        return result, done, self.observation

class PolicyPlayer(RandomPlayer):
    """
    Stable Baselines モデルクラスを使って行動を選択する
    エージェント
    """
    def __init__(self, model, observation_length=100):
        """
        観測データ長を設定し、観測データを乱数で初期化する。
        方策モデルクラスをインスタンス変数へ格納する。
        引数：
            model               モデルインスタンス（必須）
            observation_length  観測データ長
        戻り値：
            なし
        """
        # 観測データ長
        self.observation_length = observation_length
        # 観測データ初期化のためにランダムに作成
        self.observation = []
        for _ in range(self.observation_length):
            self.observation.append(random.randrange(max(self.ALL_ACTION)))
        # モデルを配置
        self.model = model

    def get_action(self):
        """
        プレイヤーの行動を方策モデルを使って選択する。
        引数：
            なし
        戻り値：
            行動    0:グー、2:ちょき、1:パー
        """
        #print(self.observation)
        return int(self.model.predict(self.observation)[0])

class JurinaPlayer(RandomPlayer):
    """
    常にパーを出し続けるエージェント
    """
    def __init__(self, observation_length=100):
        """
        観測データ長を設定し、観測データを乱数で初期化する。
        引数：
            observation_length  観測データ長
        """
        super().__init__(observation_length=observation_length)

    def get_action(self):
        """
        常にパーを行動選択する。
        引数：
            なし
        戻り値：
            行動    1:パー
        """
        return self.PAPER

class ProbPlayer(RandomPlayer):
    """
    与えられた確率分布に従って次の手を出す
    エージェント
    """
    def __init__(self, prob_list=[0.33, 0.33, 0.34], observation_length=100):
        if len(prob_list) != len(self.ALL_ACTION):
            raise ValueError(f'prob_list:({prob_list}) is not length = {len(self.ALL_ACTION)}')
        # 各行動の確率を算出
        self.prob_list = [
            prob_list[0]/sum(prob_list),
            prob_list[1]/sum(prob_list),
            prob_list[2]/sum(prob_list),
        ]
        # 観測データを初期化
        super().__init__(observation_length)

    def get_action(self):
        """
        確率分布に従って次の行動（手）を決める。
        引数：
            なし
        戻り値：
            行動：  0:グー、2:ちょき、1:パー
        """
        value = random.uniform(0.0, 1.0)
        if value < self.prob_list[0]:
            return self.ROCK
        elif value < self.prob_list[0] + self.prob_list[1]:
            return self.PAPER
        else:
            return self.SCISSORS

def test_random_player():
    """
    RandomPlayerのテスト
    """
    player = RandomPlayer(observation_length=100)
    print(player.__class__.__name__)
    print(player.observation)
    assert(len(player.observation)==100)
    assert(player.eval(player.ROCK, player.ROCK) == player.DRAW)
    assert(player.observation[-1]==player.ROCK)
    assert(len(player.observation)==100)
    assert(player.eval(player.PAPER, player.PAPER) == player.DRAW)
    assert(player.observation[-1]==player.PAPER)
    assert(len(player.observation)==100)
    assert(player.eval(player.SCISSORS, player.SCISSORS) == player.DRAW)
    assert(player.observation[-1]==player.SCISSORS)
    assert(len(player.observation)==100)
    assert(player.eval(player.ROCK, player.PAPER) == player.LOSE)
    assert(player.eval(player.ROCK, player.SCISSORS) == player.WIN)
    assert(player.eval(player.PAPER, player.ROCK) == player.WIN)
    assert(player.eval(player.PAPER, player.SCISSORS) == player.LOSE)
    assert(player.eval(player.SCISSORS, player.ROCK) == player.LOSE)
    assert(player.eval(player.SCISSORS, player.PAPER) == player.WIN)
    assert(len(player.observation)==100)
    for _ in range(100):
        assert(player.get_action() in player.ALL_ACTION)
        result, done, obs = player.pon(player.ROCK)
        assert(result in player.ALL_RESULT)
        print(f'result:{result}, done:{done}, obs:{obs}')

def test_jurina_player():
    """
    JurinaPlayerのテスト
    """
    player = JurinaPlayer(observation_length=100)
    print(player.__class__.__name__)
    print(player.observation)
    for _ in range(100):
        assert(player.get_action() in player.ALL_ACTION)
    result, done, _ = player.pon(player.ROCK)
    assert(result == player.WIN)
    assert(done)
    print(f'result:{result}, done:{done}')
    result, done, _ = player.pon(player.PAPER)
    assert(result == player.DRAW)
    assert(not done)
    print(f'result:{result}, done:{done}')
    result, done, _ = player.pon(player.SCISSORS)
    assert(result == player.LOSE)
    assert(done)
    print(f'result:{result}, done:{done}')

def test_prob_player():
    """
    ProbPlayerのテスト
    """
    player = ProbPlayer(prob_list=[5.0, 80.0, 15.0], observation_length=100)
    print(player.__class__.__name__)
    print(player.observation)
    assert(len(player.observation)==100)
    assert(player.prob_list[0]== 0.05)
    assert(player.prob_list[1]== 0.80)
    assert(player.prob_list[2]== 0.15)
    for _ in range(100):
        result, done, _ = player.pon(player.ROCK)
        print(f'result:{result}, done:{done}')
        result, done, _ = player.pon(player.PAPER)
        print(f'result:{result}, done:{done}')
        result, done, _ = player.pon(player.SCISSORS)
        print(f'result:{result}, done:{done}')

def test_policy_player():
    """
    PolicyPlayerのテスト
    要rps_mlp_ppo.zip
    """
    try:
        from stable_baselines3 import PPO
    except:
        from stable_baselines import PPO
    model = PPO.load('rps_mlp_ppo_random')
    player = PolicyPlayer(model=model, observation_length=100)
    print(player.__class__.__name__)
    print(player.observation)
    assert(len(player.observation)==100)
    for _ in range(1000):
        result, done, obs = player.pon(player.SCISSORS)
        assert(result in player.ALL_RESULT)
        print(f'result:{result}, done:{done}, obs:{obs}')

if __name__ == '__main__':
    """
    テストメソッドをすべて実行
    """
    test_random_player()
    test_jurina_player()
    test_prob_player()
    test_policy_player()