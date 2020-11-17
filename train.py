# -*- coding: utf-8 -*-
"""
トレーニングモジュール

(C) Tasuku Hori, 2020
"""
import os
import time
import gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs import RockPaperScissorsEnv, ProbPlayer, JurinaPlayer, AIPlayer

LOGDIR = './logs'
os.makedirs(LOGDIR, exist_ok=True)


def train_prob_ppo(path='prob_ppo'):
    """
    1/3の確率で出を出す環境での学習を行う。
    引数：
        path    学習済みモデルファイルパス
    戻り値：
        なし
    """
    print(f'train ppo with prob_player path={path}')
    # じゃんけん環境の構築
    env = RockPaperScissorsEnv(ProbPlayer())
    env = Monitor(env, LOGDIR, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # PPOモデルの初期化
    model = PPO('MlpPolicy', env, verbose=1)

    # トレーニング実行
    elapsed = time.time()
    model.learn(total_timesteps=1000000)
    print(f'elapse time: {time.time() - elapsed}msec')

    # 学習済みモデルの保存
    model.save(path)

    # じゃんけん環境のクローズ
    env.close()

def train_pa_ppo(path='pa_ppo'):
    """
    1/3の確率で出を出す環境での学習を行う。
    引数：
        path    学習済みモデルファイルパス
    戻り値：
        なし
    """
    print(f'train ppo with jurina_player path={path}')
    # じゃんけん環境の構築
    env = RockPaperScissorsEnv(JurinaPlayer())
    env = Monitor(env, LOGDIR, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # PPOモデルの初期化
    model = PPO('MlpPolicy', env, verbose=1)

    # トレーニング実行
    elapsed = time.time()
    model.learn(total_timesteps=1000000)
    print(f'elapse time: {time.time() - elapsed}msec')

    # 学習済みモデルの保存
    model.save(path)

    # じゃんけん環境のクローズ
    env.close()

def train_policy_ppo(path='policy_ppo', org_path='prob_ppo'):
    """
    学習済み方策をつかった環境を相手にトレーニングを行う
    引数：
        path        学習済みモデルファイルパス
        org_path    学習元となる方策がロードする学習済みモデルファイルパス
    """
    print(f'train ppo with prob_player path={path}, org_path={org_path}')
    # 学習済みモデルファイルのロード
    model = PPO.load(org_path)
    
    # じゃんけん環境の構築
    env = RockPaperScissorsEnv(AIPlayer(model))
    env = Monitor(env, LOGDIR, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # トレーニング実行
    elapsed = time.time()
    model.learn(total_timesteps=1000000)
    print(f'elapse time: {time.time() - elapsed}msec')

    # 学習済みモデルの保存
    model.save(path)

    # じゃんけん環境のクローズ
    env.close()

if __name__ == '__main__':
    """
    各々トレーニング実行する。
    """
    train_prob_ppo()
    train_pa_ppo()
    train_policy_ppo()
