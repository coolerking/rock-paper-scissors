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
    from sim import RPSEnv
except:
    raise

def ppo_mlp_train(enemy_player=None, log_dir='./logs', model_path='rps_mlp_ppo'):
    """
    トレーニング処理
    引数：
        log_dir     ログディレクトリパス
        model_path  モデルファイルパス(.zipの前の文字列)
    """
    # ログ出力ディレクトリ作成
    os.makedirs(log_dir, exist_ok=True)
    # トレーニング用環境の作成
    env = RPSEnv(enemy_player, max_steps=10000)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # Stable Baselines PPO モデルを MlpPolicy指定で作成
    # 妥当性検査ON、TensorBoardによる可視化ON
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    # トレーニング環境を使ってPPOをトレーニング
    model.learn(total_timesteps=10000)

    # モデルファイルを保存
    model.save(model_path)

    # 環境を閉じる
    env.close()

if __name__ == '__main__':
    print('start training against random player')
    start_time = time.time()
    ppo_mlp_train(
        enemy_player=RandomPlayer(), 
        log_dir='random_logs', 
        model_path='rps_mlp_ppo_random')
    term = time.time() - start_time
    print(f'end training against random player: elapsed time = {term}sec.')

    print('start training against jurina player(paper action only)')
    start_time = time.time()
    ppo_mlp_train(
        enemy_player=JurinaPlayer(), 
        log_dir='jurina_logs', 
        model_path='rps_mlp_ppo_jurina')
    term = time.time() - start_time
    print(f'end training on jurina player(paper action only): elapsed time = {term}sec.')

    prob_list = [0.15, 0.4, 0.45]
    print(f'start training on prob({prob_list}) player')
    start_time = time.time()
    ppo_mlp_train(
        enemy_player=ProbPlayer(prob_list=prob_list), 
        log_dir='prob_logs', model_path='rps_mlp_ppo_prob')
    term = time.time() - start_time
    print(f'end training on julinaplay: elapsed time = {term}sec.')
