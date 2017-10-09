#coding:utf-8

#-moduleのimport-#
import numpy as np
import random
import matplotlib.pyplot as plt
#keras系のモジュール
from keras.layers import Activation,Dense
from keras.models import Sequential
from keras.layers.recurrent import LSTM

#-初期値いろいろ-#
#乱数の初期シード
random.seed(0)
#ノイズの係数
random_factor = 0.50

#-ノイズ混じりのsin波を生成する-#
time = np.arange(0, 3000, 0.1)#0から3000まで,0.1刻みで時間tを生成
noise = np.random.randn(time.size)#使用する時間と同じ数だけのノイズを生成
noise = (noise / np.linalg.norm(noise)) * random_factor#ノイズを正規化した後に、係数を掛ける
sin_curve = np.sin(time) + noise#sin波にノイズを被せる（足し算）
"""
#sin波の形状を表示　デバック用
plt.plot(sin_curve)
plt.xlim([0,100])
plt.ylim([-1.1,1.1])
plt.show()
"""

#-データの作成-#
#入力データと出力データを作成する
#入力：100step分のsin波
#出力：101step目
#訓練データ数：10000
#テストデータ数：10000
X_train = []#訓練・テストの入力・正解データを格納する空のリストを定義
y_train = []
X_test = []
y_test= []
#訓練データを作成していく
for i in range(10000):
    tmp_X = sin_curve[i:i+100]#100step分を切り取る
    tmp_y = sin_curve[i+100]#正解データとして101step目
    X_train.append(tmp_X)#リストに追加していく
    y_train.append(tmp_y)
#テストデータを生成していく
for i in range(10001,20000):#訓練データと同様
    tmp_X = sin_curve[i:i+100]
    tmp_y = sin_curve[i+100]
    X_test.append(tmp_X)
    y_test.append(tmp_y)

#訓練データを変換
#入力：（データの総数,タイムステップ数,特徴量の次元数）の三次元
#出力：（データの総数,特徴量の次元数）の二次元
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
y_train = y_train.reshape(y_train.shape[0],1)
#テストデータも変換
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
y_test = y_test.reshape(y_test.shape[0],1)

#-kerasでネットワークの構造を定義-#
model = Sequential()
#入力バッヂサイズ
#データ数：None（任意の数）
#バッヂサイズ：100
#入力次元数：1
model.add(LSTM(20, batch_input_shape=(None, 100, 1)))
model.add(Dense(1))
model.add(Activation("linear"))
model.summary()
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, batch_size=1000, nb_epoch=20)

#データの予測を
pre = model.predict(X_test)
#正解データと予測データをプロット
plt.plot(y_test, label="test_data")
plt.plot(pre, label="predict")
plt.legend()
plt.xlim([0,100])
plt.ylim([-1.1,1.1])
plt.show()