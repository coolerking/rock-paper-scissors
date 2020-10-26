# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Webアプリケーション「AI対戦じゃんけん」エントリポイントモジュール。

Setup:
    pip install docopt flask stable-baselines3

Usage:
    endpoint.py [--debug] [--model_path=<target_model_path>]

Options:
    --debug                             set debug on flask
    --model_path=<target_model_path>    set target model path
"""
try:
    from docopt import docopt
except:
    raise

try:
    from flask import Flask, jsonify, request, render_template
except:
    raise

try:
    from stable_baselines3 import PPO
except:
    try:
        from stable_baselines import PPO
    except:
        raise

try:
    from ppo import Mlp
    from agents import PolicyPlayer
    from envs import Playground
except:
    raise

# 対戦相手および環境の準備
model_path = Mlp.PATH + '_random'
#model_path = Mlp.PATH + '_prb'
#model_path = Mlp.PATH + '_jurina'
env = Playground(PolicyPlayer(PPO.load(model_path)))
obs = env.reset()

# アプリケーションオブジェクト生成
app = Flask(__name__)

def predict(my_action):
    """
    選択した行動によるじゃんけん結果を辞書型で取得する。
    引数：
        my_action   選択した行動
    戻り値：
        JSON文字列  結果
    """
    global obs
    # 選択した行動の結果を取得
    obs, reward, done, info = env.step(0)
    if done:
        env.reset()
    return {
        'my_action':    0,
        'model_path':   model_path,
        'reward':       reward,
        'observation':  obs,
        'done':         done,
        'info':         info
    }

@app.route('/', methods=['GET'])
def show_index():
    """
    index.html を表示する。
    引数：
        なし
    戻り値：
        なし
    """
    # /template/index.html を表示
    return render_template('index.html')

@app.route('/obs', methods=['GET'])
def show_obs():
    """
    観測データを取得する。
    引数：
        なし
    戻り値：
        JSON文字列  観測データ
    """
    return jsonify(obs)

@app.route('/pon/goo', methods=['POST'])
def goo():
    """
    グーを出したときの結果を返却する。
    引数：
        なし
    戻り値：
        JSON文字列  結果
    """
    return jsonify(predict(0))

@app.route('/pon/choki', methods=['POST'])
def choki():
    """
    チョキを出したときの結果を返却する。
    引数：
        なし
    戻り値：
        JSON文字列  結果
    """
    return jsonify(predict(2))

@app.route('/pon/paa', methods=['POST'])
def paa(my_action=None):
    """
    パーを出したときの結果を返却する。
    引数：
        なし
    戻り値：
        JSON文字列  結果
    """
    return jsonify(predict(1))

@app.route('/reload', methods=['POST'])
def load_model():
    """
    モデルをリロードする
    引数：
        なし
    戻り値：
        JSON文字列  観測データ
    """
    env.set_model(PPO.load(Mlp.PATH + '_random'))
    global obs
    obs = env.reset()
    return jsonify(obs)

if __name__ == '__main__':
    """
    起動時のオプション処理を行い
    Webアプリケーションを開始する。
    引数：
        なし
    戻り値：
        なし
    """
    args = docopt(__doc__)
    debug = args['--debug']
    target_model_path = args['--model_path']
    if target_model_path is not None:
        model_path = target_model_path
        env = Playground(PolicyPlayer(PPO.load(model_path)))
    app.run(debug=debug)