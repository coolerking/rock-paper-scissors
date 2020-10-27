# -*- coding: utf-8 -*-
"""
学習済みモデルを評価するためのモジュール。
要学習済みモデルファイル。
python eval.py を実行すると、標準出力に平均報酬値が出力される。
"""
from time import time
from ppo import Mlp
from agents import PolicyPlayer, RandomPlayer, JurinaPlayer, ProbPlayer
from envs import Playground

def test_ppo_mlp_random(steps=100, debug=True):
    """
    RandomPlayerで学習した方策を使って対戦し平均報酬値を出力する。
    引数：
        steps       実行するステップ回数
        debug       各ステップ結果を表示するかどうか
    戻り値：
        なし
    """
    model = Mlp.load_model('_random')
    my_player = PolicyPlayer(model)
    enemy_player = RandomPlayer()
    env = Playground(enemy_player)
    
    observation = env.reset()
    revenue = []
    episodes = 0
    elapsed = time()
    for _ in range(steps):
        my_action = my_player.next_action(observation)
        observation, reward, done, info = env.step(my_action)
        if debug:
            env.render()
        if done:
            revenue.append(reward)
            episodes = info['episode_no']
            observation = env.reset()
    print('** ppo_mlp_random test')
    #print(f'   enemy player {enemy_player.__class__.__name__}')
    print(f'   ran {steps} steps, {time() - elapsed} sec')
    #print(f'   {episodes} episodes done')
    if len(revenue) <= 0:
        print(f'   no revenues')
    else:
        print(f'   revenue average: {sum(revenue)/len(revenue)} per episodes')

def test_ppo_mlp_jurina(steps=100, debug=True):
    """
    JurinaPlayerで学習した方策を使って対戦し平均報酬値を出力する。
    引数：
        steps       実行するステップ回数
        debug       各ステップ結果を表示するかどうか
    戻り値：
        なし
    """
    model = Mlp.load_model('_jurina')
    my_player = PolicyPlayer(model)
    enemy_player = JurinaPlayer()
    env = Playground(enemy_player)
    
    observation = env.reset()
    revenue = []
    episodes = 0
    elapsed = time()
    for _ in range(steps):
        my_action = my_player.next_action(observation)
        observation, reward, done, info = env.step(my_action)
        if debug:
            env.render()
        if done:
            revenue.append(reward)
            episodes = info['episode_no']
            observation = env.reset()
    print('** ppo_mlp_jurina test')
    #print(f'   enemy player {enemy_player.__class__.__name__}')
    print(f'   ran {steps} steps, {time() - elapsed} sec')
    #print(f'   {episodes} episodes done')
    if len(revenue) <= 0:
        print(f'   no revenues')
    else:
        print(f'   revenue average: {sum(revenue)/len(revenue)} per episodes')

def test_ppo_mlp_prob(steps=100, debug=True):
    """
    ProbPlayerで学習した方策を使って対戦し平均報酬値を出力する。
    引数：
        steps       実行するステップ回数
        debug       各ステップ結果を表示するかどうか
    戻り値：
        なし
    """
    model = Mlp.load_model('_prob')
    my_player = PolicyPlayer(model)
    enemy_player = ProbPlayer(prob_list=[0.2, 0.1, 0.7])
    env = Playground(enemy_player)
    
    observation = env.reset()
    revenue = []
    episodes = 0
    elapsed = time()
    for _ in range(steps):
        my_action = my_player.next_action(observation)
        observation, reward, done, info = env.step(my_action)
        if debug:
            env.render()
        if done:
            revenue.append(reward)
            episodes = info['episode_no']
            observation = env.reset()
    print('** ppo_mlp_prob test: prob_list=[0.2, 0.1, 0.7]')
    #print(f'   enemy player {enemy_player.__class__.__name__}')
    print(f'   ran {steps} steps, {time() - elapsed} sec')
    #print(f'   {episodes} episodes done')
    if len(revenue) <= 0:
        print(f'   no revenues')
    else:
        print(f'   revenue average: {sum(revenue)/len(revenue)} per episodes')

if __name__ == '__main__':
    """
    学習済みモデルの平均報酬値を出力する。
    """
    step_list = [10, 100, 1000, 10000]
    debug = False
    print('*****************************')
    for steps in step_list:
        test_ppo_mlp_random(steps=steps, debug=debug)
    print('*****************************')
    for steps in step_list:
        test_ppo_mlp_jurina(steps=steps, debug=debug)
    print('*****************************')
    for steps in step_list:
        test_ppo_mlp_prob(steps=steps, debug=debug)
    print('*****************************')