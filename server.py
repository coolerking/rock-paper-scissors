# -*- coding: utf-8 -*-
"""
Webアプリケーション「AI対戦じゃんけん」エントリポイントモジュール。
起動後、http://127.0.0.1:5000/ を開くとじゃんけん画面が表示される。

Setup:
    pip install docopt flask stable-baselines3

Usage:
    server.py [--debug] [--model_path=<target_model_path>]

Options:
    --debug                             set debug on flask
    --model_path=<target_model_path>    set target model path
"""
from docopt import docopt
from flask import Flask, jsonify, render_template, session
from stable_baselines3 import PPO
from envs import RockPaperScissorsEnv as env

# 方策のロード
PATH = 'prob_ppo' # 1/3の確率で手を出す環境相手に学習
#PATH = 'pa_ppo' # つねにパーを出す環境相手に学習
#PATH = 'policy_ppo' # prob_ppoを相手に学習
model = PPO.load(PATH)


# アプリケーションオブジェクト生成
app = Flask(__name__)
# session 用シークレットキー
app.secret_key='rock-paper-scissors'

def predict(my_action, obs):
    """
    選択した行動によるじゃんけん結果を辞書型で取得する。
    引数：
        my_action   選択した行動
    戻り値：
        JSON文字列  結果
    """
    enemy_action = int(model.predict(obs)[0])
    obs = env.update_observation(obs, my_action, enemy_action)
    done = env.is_done(my_action, enemy_action)
    reward = env.calc_reward(my_action, enemy_action)
    return {
        'my_action':    my_action,
        'model':        model.__class__.__name__,
        'reward':       reward,
        'observation':  obs,
        'done':         done,
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
    # セッション上に初期化した観測データを格納
    if 'obs' not in session:
        # 観測初期化
        obs = env.init_observation()
        session['obs'] = obs
    # /template/index.html を表示
    return render_template('index.html')

@app.route('/pon/goo', methods=['POST'])
def goo():
    """
    グーを出したときの結果を返却する。
    引数：
        なし
    戻り値：
        JSON文字列  結果
    """
    # 観測データを取得
    if 'obs' not in session:
        session['obs'] = env.init_observation()
    obs = session['obs']
    return jsonify(predict(0, obs))

@app.route('/pon/choki', methods=['POST'])
def choki():
    """
    チョキを出したときの結果を返却する。
    引数：
        なし
    戻り値：
        JSON文字列  結果
    """
    # 観測データを取得
    if 'obs' not in session:
        session['obs'] = env.init_observation()
    obs = session['obs']
    return jsonify(predict(2, obs))

@app.route('/pon/paa', methods=['POST'])
def paa():
    """
    パーを出したときの結果を返却する。
    引数：
        なし
    戻り値：
        JSON文字列  結果
    """
    # 観測データを取得
    if 'obs' not in session:
        session['obs'] = env.init_observation()
    obs = session['obs']
    return jsonify(predict(1, obs))

@app.route('/reload', methods=['GET'])
def load_model():
    """
    モデルをリロードする
    引数：
        なし
    戻り値：
        JSON文字列  観測データ
    """
    global model
    old_model = model
    new_model = PPO.load(PATH)
    is_updated = (new_model != old_model)
    model = new_model
    return jsonify({
        'old_model':    old_model.__class__.__name__,
        'new_model':    new_model.__class__.__name__,
        'is_updated':   is_updated,
    })

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
        model = PPO.load(model_path)
        PATH = model_path
    app.run(debug=debug)