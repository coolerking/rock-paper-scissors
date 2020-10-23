# -*- coding: utf-8 -*-
"""
エージェントモジュール

方策に従うプレイヤー以外は
じゃんけん相手として使用する。

(C) Tasuku Hori, 2020
"""
import random

class BasePlayer:
    """
    じゃんけんプレイヤー基底クラス
    """
    ROCK = 0        # グー
    PAPER = 1       # パー
    SCISSORS = 2    # チョキ
    ALL_ACTION = [ROCK, PAPER, SCISSORS]    # 行動値域

    def next_action(self, observation):
        """
        観測データを引数に次の行動を選択し返却する。
        本クラスには実装がなく必ず例外を発生させる。
        引数：
            observation     観測データ
        戻り値：
            選択された行動
        """
        raise NotImplementedError('need to implement that returns next action')

class RandomPlayer(BasePlayer):
    """
    乱数で手を決めるプレイヤー
    """
    def next_action(self, observation):
        """
        ランダムに次の手を返却する。
        引数の観測データは一切使用しない。
        引数：
            observation     観測データ
        戻り値：
            選択された行動
        """
        return random.randrange(len(self.ALL_ACTION))

class ProbPlayer(BasePlayer):
    """
    確率分布リストに従って次の手を決定するプレイヤー
    """
    def __init__(self, prob_list=[0.33, 0.33, 0.34]):
        """
        引数で渡された確率分布をもとにしきい値を計算し
        インスタンス変数に格納する。
        引数：
            prod_list   長さ3のリスト(float)
        戻り値：
            なし
        """
        super().__init__()
        self.threshold_rock = prob_list[0]/sum(prob_list)
        self.threshold_paper = self.threshold_rock + prob_list[0]/sum(prob_list)

    def next_action(self, observation):
        """
        確率分布に則って次の手を選択・返却する。
        引数の観測データは一切使用しない。
        引数：
            observation     観測データ
        戻り値：
            選択された行動
        """
        value = random.uniform(0.0, 1.0)
        if value < self.threshold_rock:
            return self.ROCK
        elif value < self.threshold_paper:
            return self.PAPER
        else:
            return self.SCISSORS

class JurinaPlayer(BasePlayer):
    """
    常にパーを出し続けるプレイヤー
    """
    def next_action(self, observation):
        """
        引数の観測データにかかわらず常にパーを選択・返却する。
        引数：
            observation     観測データ
        戻り値：
            選択された行動（パー）
        """
        return self.PAPER

class PolicyPlayer(BasePlayer):
    """
    Stable Baselines モデルクラスを使って行動を選択する
    エージェント
    """
    def __init__(self, model):
        """
        Stable Baselinesの提供する方策モデルクラスの
        インスタンスを格納する。
        引数：
            model               PPOモデルインスタンス（必須）
        戻り値：
            なし
        """
        super().__init__()
        self.model = model

    def next_action(self, observation):
        """
        方策モデルへ観測データを渡し、次の手を返却する。
        引数：
            observation     観測データ
        戻り値：
            選択された行動
        """
        return int(self.model.predict(observation)[0])

def test_base_player():
    player = BasePlayer()
    try:
        player.next_action([])
    except NotImplementedError:
        return
    assert(False)

def test_random_player():
    player = RandomPlayer()
    for _ in range(100):
        assert(player.next_action([]) in BasePlayer.ALL_ACTION)

def test_prob_player():
    player = ProbPlayer()
    for _ in range(100):
        assert(player.next_action([]) in BasePlayer.ALL_ACTION)
    player = ProbPlayer([3, 1, 10])
    for _ in range(100):
        assert(player.next_action([]) in BasePlayer.ALL_ACTION)

def test_jurina_player():
    player = JurinaPlayer()
    for _ in range(100):
        assert(player.next_action([]) == BasePlayer.PAPER)

def test_policy_player():
    PolicyPlayer(model=None)
    # need more test
    pass

if __name__ == '__main__':
    """
    テストメソッドをすべて実行
    """
    test_random_player()
    test_jurina_player()
    test_prob_player()
    test_policy_player()