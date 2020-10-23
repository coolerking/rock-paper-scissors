# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Execute REST listener sample

Setup:
    pip install docopt flask

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
        from stable_baselines3 import PPO
    except:
        raise

try:
    from ppo import Mlp
except:
    raise

model_paths = [Mlp.PATH + '_random', Mlp.PATH + '_jurina', Mlp.PATH + '_prob']
model_path = model_paths[0]
model = PPO.load(model_path)

app = Flask(__name__)

@app.route('/', method=['GET'])
def show_index():
    return render_template('index.html', title='rock-paper-scissors', name=name)

@app.route('/status', method=['GET'])
def say_hello():
    """
    生死確認用レスポンス処理。
    RESTサーバ側が認識している全W/FのリストをJSON形式で返却。
    """
    msg = {
        'model_path':               model_path,
        'available_model_paths':    model_paths,
        'copy_right':               'Tasuku Hori, 2020',
    }
    return jsonify(msg)

@app.route('/reload/<target_model_path>', methods=['GET'])
def reload(target_model_path=None):
    """
    モデルの再ロード。
    """
    global model_path
    if target_model_path is None:
        target_model_path = model_path
    new_model = PPO.load(target_model_path)
    global model
    model = new_model
    global player
    observation = player.observation
    player = PolicyPlayer(model=model)
    player.observation = observation
    model_path = target_model_path
    return jsonify({
        'model_path':               model_path,
        'available_model_paths':    model_paths,
        'reloaded':                 (model==new_model),
    })

@app.route('/pon/<my_action>', methods=['GET'])
def predict(my_action=None):
    """
    AI関数を実行し、次のactionを返却。
    """
    if my_action is None:
        my_action = 0
    result, done, obs = player.pon(my_action)
    return jsonify({
        'model_path':               model_path,
        'result':                   result,
        'observation':              obs,
        'done':                     done
    })

if __name__ == '__main__':
    """
    起動時のオプション処理。
    """
    
    args = docopt(__doc__)
    debug = args['--debug']
    target_model_path = args['--model_path']
    if target_model_path is not None:
        model = PPO.load(target_model_path)
        model_path = target_model_path
    app.run(debug=debug)