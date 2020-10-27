# -*- coding: utf-8 -*-
"""
Stable Baselines3/Stable Baseline 方策ユーティリティモジュール。
モデルインスタンスを取得するためのユーティリティクラス群を提供する。

(C) Tasuku Hori, 2020
"""
sb3 = True  # Stable Baselines3 使用可能かどうか
try:
    from stable_baselines3 import PPO
except:
    sb3 = False # Stable Baselines3 使用不可
    try:
        from stable_baselines import PPO

        class MlpLstm:
            """
            MLP 特徴抽出でLSTMを使用して、Actor-Criticを実装した方策を使う
            PPOモデルインスタンスを扱うクラス。
            """
            POLICY = 'MlpLstmPolicy'
            LOGDIR = './logs_mlplstm'
            PATH = 'ppo_mlplstm'

            @staticmethod
            def new_model(env, verbose=1):
                """
                MlpLstmPolicyを使った新規PPOモデルインスタンスを取得する。
                引数：
                    env         OpenAI Gym準拠の環境インスタンス
                    verbose     verboseオプション
                戻り値：
                    PPOインスタンス（未学習）
                """
                return PPO(
                    MlpLstm.POLICY, 
                    env, 
                    verbose=verbose, 
                    tensorboard_log=MlpLstm.LOGDIR)

            @staticmethod
            def load_model(suffix=''):
                """
                モデルファイルをロードした学習済みモデルインスタンス (MlpLstmPolicy) を取得する。
                引数：
                    suffix      接尾文字列
                戻り値：
                    PPOインスタンス（学習済み）
                """
                return PPO.load(MlpLstm.PATH + suffix)

        class MlpLnLstm:
            """
            MLP特徴抽出でレイヤ正規化LSTMを使用して、Actor-Criticを実装した方策を使う
            PPOモデルインスタンスを扱うクラス。
            """
            POLICY = 'MlpLnLstmPolicy'
            LOGDIR = './logs_mlplnlstm'
            PATH = 'ppo_mlplstm'

            @staticmethod
            def new_model(env, verbose=1):
                """
                MlpLnLstmPolicyを使った新規PPOモデルインスタンスを取得する。
                引数：
                    env         OpenAI Gym準拠の環境インスタンス
                    verbose     verboseオプション
                戻り値：
                    PPOインスタンス（未学習）
                """
                return PPO(
                    MlpLnLstm.POLICY, 
                    env, 
                    verbose=verbose, 
                    tensorboard_log=MlpLnLstm.LOGDIR)

            @staticmethod
            def load_model(suffix=''):
                """
                モデルファイルをロードした学習済みモデルインスタンス (MlpLnLstmPolicy) を取得する。
                引数：
                    suffix      接尾文字列
                戻り値：
                    PPOインスタンス（学習済み）
                """
                return PPO.load(MlpLnLstm.PATH + suffix)
        class CnnLstm:
            """
            CNN特徴抽出でLSTMを使用して、Actor-Criticを実装した方策を使う
            PPOモデルインスタンスを扱うクラス。
            """
            POLICY = 'CnnLstmPolicy'
            LOGDIR = './logs_cnnlstm'
            PATH = 'ppo_cnnlstm'

            @staticmethod
            def new_model(env, verbose=1):
                """
                CnnLstmPolicyを使った新規PPOモデルインスタンスを取得する。
                引数：
                    env         OpenAI Gym準拠の環境インスタンス
                    verbose     verboseオプション
                戻り値：
                    PPOインスタンス（未学習）
                """
                return PPO(
                    CnnLstm.POLICY, 
                    env, 
                    verbose=verbose, 
                    tensorboard_log=CnnLstm.LOGDIR)

            @staticmethod
            def load_model(suffix=''):
                """
                モデルファイルをロードした学習済みモデルインスタンス (CnnLstmPolicy) を取得する。
                引数：
                    suffix      接尾文字列
                戻り値：
                    PPOインスタンス（学習済み）
                """
                return PPO.load(CnnLstm.PATH + suffix)

        class CnnLnLstm:
            """
            CNN 特徴抽出でレイヤ正規化LSTMを使用して、Actor-Criticを実装した方策を使う
            PPOモデルインスタンスを扱うクラス。
            """
            POLICY = 'CnnLnLstmPolicy'
            LOGDIR = './logs_cnnlnlstm'
            PATH = 'ppo_cnnlnlstm'

            @staticmethod
            def new_model(env, verbose=1):
                """
                CnnLnLstmPolicyを使った新規PPOモデルインスタンスを取得する。
                引数：
                    env         OpenAI Gym準拠の環境インスタンス
                    verbose     verboseオプション
                戻り値：
                    PPOインスタンス（未学習）
                """
                return PPO(
                    CnnLnLstm.POLICY, 
                    env, 
                    verbose=verbose, 
                    tensorboard_log=CnnLnLstm.LOGDIR)

            @staticmethod
            def load_model(suffix=''):
                """
                モデルファイルをロードした学習済みモデルインスタンス (CnnLnLstmPolicy) を取得する。
                引数：
                    suffix      接尾文字列
                戻り値：
                    PPOインスタンス（学習済み）
                """
                return PPO.load(CnnLnLstm.PATH + suffix)

        def test_new_model_for_sb():
            """
            Stable Baselines3 上で使用可能な方策を対象としたテスト
            """
            try:
                import gym
            except:
                raise
            env = gym.make('CartPole-v1')
            assert(type(MlpLstm.new_model(env)) is PPO)
            assert(type(MlpLnLstm.new_model(env)) is PPO)
            env = gym.make('BreakoutNoFrameskip-v0')
            assert(type(CnnLstm.new_model(env)) is PPO)
            assert(type(CnnLnLstm.new_model(env)) is PPO)

    except:
        raise

class Mlp:
    """
    MLP (64x2層) を使用して、Actor-Criticを実装した方策を使う
    PPOモデルインスタンスを扱うクラス。
    """
    POLICY = 'MlpPolicy'
    LOGDIR = './logs_mlp'
    PATH = 'ppo_mlp'

    @staticmethod
    def new_model(env, verbose=1):
        """
        MlpPolicyを使った新規PPOモデルインスタンスを取得する。
        引数：
            env         OpenAI Gym準拠の環境インスタンス
            verbose     verboseオプション
        戻り値：
            PPOインスタンス（未学習）
        """
        return PPO(
            Mlp.POLICY, 
            env, 
            verbose=verbose, 
            tensorboard_log=Mlp.LOGDIR)

    @staticmethod
    def load_model(suffix=''):
        """
        モデルファイルをロードした学習済みモデルインスタンス (MlpPolicy) を取得する。
        引数：
            suffix      接尾文字列
        戻り値：
            PPOインスタンス（学習済み）
        """
        return PPO.load(Mlp.PATH + suffix)

class Cnn:
    """
    CNN (NatureCNN：CNNの後に全結層をつないたもの) を使用して、Actor-Criticを実装
    """
    POLICY = 'CnnPolicy'
    LOGDIR = './logs_cnn'
    PATH = 'ppo_cnn'

    @staticmethod
    def new_model(env, verbose=1):
        """
        CnnPolicyを使った新規PPOモデルインスタンスを取得する。
        引数：
            env         OpenAI Gym準拠の環境インスタンス
            verbose     verboseオプション
        戻り値：
            PPOインスタンス（未学習）
        """
        return PPO(
            Cnn.POLICY, 
            env, 
            verbose=verbose, 
            tensorboard_log=Cnn.LOGDIR)

    @staticmethod
    def load_model(suffix=''):
        """
        モデルファイルをロードした学習済みモデルインスタンス (CnnPolicy) を取得する。
        引数：
            suffix      接尾文字列
        戻り値：
            PPOインスタンス（学習済み）
        """
        return PPO.load(Cnn.PATH + suffix)

def test_new_model():
    """
    Stable Baselines3 上で使用可能な方策を対象としたテスト
    """
    try:
        import gym
    except:
        raise
    env = gym.make('CartPole-v1')
    assert(type(Mlp.new_model(env)) is PPO)
    env = gym.make('BreakoutNoFrameskip-v0')
    assert(type(Cnn.new_model(env)) is PPO)


if __name__ == '__main__':
    test_new_model()
    if not sb3:
        test_new_model_for_sb()