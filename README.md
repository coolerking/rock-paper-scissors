# Rock-Paper-Scissors サンプル

じゃんけん対戦の強化学習サンプルコードです。

以下のライブラリを使用しています。

* OpenAI Gym
* Stable Baselines3 / PyTorch
* flask, docopt

## インストール

* Anaconda をインストール、`conda env` コマンドを使い実行用環境を構築、`conda activate`コマンドで環境に入る
* `conda install pytorch torchvision cpuonly -c pytorch`
* `pip install stable-baselines3[extra]`
* `git clone https://github.com/coolerking/rock-paper-scissors.git`

## 使い方

### トレーニング

* `cd rock-paper-scissors`
* `python train.py`

### トレーニングの可視化

* `tensorboard --logdir play_logs`
* ブラウザで `http://127.0.0.1:6006/` を開く

引数の`logdir`のパスを `prob_dist_logs`や`jurina_logs`に変更することでほかの学習モデルのトレーニング可視化が可能です。

停止はCtrl+C。

### 実行

* `cd rock-paper-scissors`
* `conda install flask`
* `pip install docopt`
* `python app.py`

停止はCtrl+C。

#### 疎通確認

* ブラウザで `http://127.0.0.1:5000/` を開く

#### モデルのリロード

* ブラウザで `http://127.0.0.1:5000/` を開く

## ライセンス

本サンプルコードはMITライセンス準拠です。
