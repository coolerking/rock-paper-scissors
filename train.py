# -*- coding: utf-8 -*-
"""
トレーニングモジュール

(C) Tasuku Hori, 2020
"""
import os
import time
try:
    import gym
except:
    raise
try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except:
    # Stable Baselines3(pytorch) がない場合、
    # Stable Baselines(tensorflow) で代用
    try:
        from stable_baselines.common.vec_env import DummyVecEnv
        from stable_baselines import PPO
        from stable_baselines.bench import Monitor
        from sim import WorkflowEnv
    except:
        # 両方ない場合は例外発生
        raise
try:
    from agents import RandomPlayer, JurinaPlayer, ProbPlayer
    from envs import Playground
    from ppo import Mlp
except:
    raise

def train_mlp(enemy_player, model_path):
    """
    方策としてMlpPolicyを指定してじゃんけん対戦環境下で
    トレーニングを実行する。
    引数：
        enemy_player        敵プレイヤー
        model_path          モデルファイルパス
    戻り値：
        なし
    """
    # ログディレクトリの作成
    os.makedirs(Mlp.LOGDIR, exist_ok=True)
    # 環境の構築
    env = Playground(enemy_player)
    env = Monitor(env, Mlp.LOGDIR, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # 未学習モデルの準備
    model = Mlp.new_model(env, verbose=1)

    # トレーニング実行
    model.learn(total_timesteps=100000)

    # 学習済みモデルの保存
    model.save(model_path)

    # 環境のクローズ
    env.close()

def fine_tuning_mlp(enemy_player, new_model_path, org_model_path):
    """
    学習済みモデルに別の環境をセットしてファインチューニングを
    実行する。
    引数：
        enemy_player        新たな敵プレイヤー
        new_model_path      新規モデルファイルパス
        org_model_path      学習済みモデルファイルパス
    戻り値：
        なし
    """
    # ログディレクトリの作成
    os.makedirs(Mlp.LOGDIR, exist_ok=True)
    # 環境の構築
    env = Playground(enemy_player)
    env = Monitor(env, Mlp.LOGDIR, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # 学習済みモデルのロード
    model = PPO.load(org_model_path)

    # 環境のセット
    model.set_env(env)

    # トレーニング実行
    model.learn(total_timesteps=100000)

    # 学習済みモデルの保存
    model.save(new_model_path)

    # 環境のクローズ
    env.close()

if __name__ == '__main__':
    """
    RandomPlayer, ProbPlayer, JurinaPlayer を敵として
    各々トレーニング実行する。
    """

    print('start training against random player')
    start_time = time.time()
    train_mlp(RandomPlayer(), Mlp.PATH + '_random')
    term = time.time() - start_time
    print(f'end training against random player: elapsed time = {term}sec.')

    prob_list = [0.33, 0.33, 0.34]
    print(f'start training against prob player({prob_list})')
    start_time = time.time()
    train_mlp(ProbPlayer(prob_list), Mlp.PATH + '_prob')
    term = time.time() - start_time
    print(f'end training against prob player: elapsed time = {term}sec.')

    print('start training against jurina player')
    start_time = time.time()
    train_mlp(JurinaPlayer(), Mlp.PATH + '_jurina')
    term = time.time() - start_time
    print(f'end training against jurina player: elapsed time = {term}sec.')

    prob_list = [0.1, 0.5, 0.4]
    print(f'start transfer training against prob player({prob_list})')
    start_time = time.time()
    fine_tuning_mlp(ProbPlayer(prob_list), Mlp.PATH + '_prob2', Mlp.PATH + '_random')
    term = time.time() - start_time
    print(f'end transfer training against prob player: elapsed time = {term}sec.')
