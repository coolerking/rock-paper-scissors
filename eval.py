# -*- coding: utf-8 -*-
"""
学習済みモデルを評価するためのモジュール。
要学習済みモデルファイル。
python eval.py を実行すると、標準出力に平均報酬値が出力される。
"""
from time import time
from stable_baselines3 import PPO
from envs import EvalEnv, Player, ProbPlayer, JurinaPlayer, AIPlayer

# 学習済みモデルファイルパス
PROP_PPO = 'prob_ppo'
PA_PPO = 'pa_ppo'
POLICY_PPO = 'policy_ppo'
PATHS = [PROP_PPO, PA_PPO, POLICY_PPO]

def eval_ppo(env_player, path=PROP_PPO, steps=100, debug=True):
    """
    学習済み方策PPOを100ステップ実行し、
    平均収益を表示する。
    引数：
        env_player      評価環境側のプレイヤーインスタンス
        path            ロードする方策側学習済みモデルファイルパス
        steps           ステップ実行回数
        debug           Trueの場合毎ステップ表示する
    戻り値：
        なし
    """
    # 評価用環境の生成
    env = EvalEnv(env_player)
    # 評価対象学習済み方策モデルの復元
    model = PPO.load(path)
    
    # エピソード開始時の観測を取得
    observation = env.reset()
    # 収益リスト初期化
    revenue = []
    # エピソード数
    episodes = 0
    # 処理時間計測
    elapsed = time()
    for _ in range(steps):
        
        policy_action = model.predict(observation)
        if isinstance(policy_action, tuple):
            policy_action = policy_action[0]
        policy_action = int(policy_action)
        observation, reward, done, info = env.step(policy_action)
        if debug:
            env.render()
        if done:
            revenue.append(reward)
            episodes = info['episode_no']
            observation = env.reset()
    print(f'** path:{path} test')
    if debug:
        print(f'   env player {env_player.__class__.__name__}')
    print(f'   ran {steps} steps, {time() - elapsed} sec')
    if debug:
        print(f'   {episodes} episodes done')
    if len(revenue) <= 0:
        print(f'   no revenues')
    else:
        print(f'   revenue average: {sum(revenue)/len(revenue)} per episodes')


if __name__ == '__main__':
    """
    学習済みモデルの平均報酬値を出力する。
    """
    # ステップ数リスト
    step_list = [10, 100, 1000, 10000]

    # 評価環境側プレイヤーリスト
    players = [Player(), ProbPlayer(), JurinaPlayer(), AIPlayer(PPO.load(PROP_PPO))]
    debug = False
    print('*****************************')
    for steps in step_list:
        for path in PATHS:
            for player in players:
                print(f'*** steps={steps}, path={path}, player={player.__class__.__name__}')
                eval_ppo(env_player=player, path=path, steps=steps, debug=debug)
                print('*****************************')
