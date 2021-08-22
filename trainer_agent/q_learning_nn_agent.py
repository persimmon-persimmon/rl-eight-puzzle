import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import json
import random
from collections import deque,defaultdict
import numpy as np
from square_puzzle import SquarePuzzle
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from agent_base import AgentBase
import pickle

class QLearningNnAgent(AgentBase):
    def __init__(self,epsilon,actions,learning_rate=0.1):
        super().__init__(epsilon,actions)
        self.model = None
        self.learning_rate = learning_rate

    def save(self,model_path):
        """
        モデルを保存する.
        """
        with open(model_path,"wb") as f:
            pickle.dump(self.model,f)

    def read(self,model_path):
        """
        モデルを読み込む.
        """
        with open(model_path,"rb") as f:
            self.model = pickle.load(f)

    def initialize(self, experiences):
        """
        エージェントを初期化する.モデルを構築.
        """

        # 数字をOneHoeエンコーディングする
        ary=[]
        for i in range(9):
            ary.append([(j+i)%9 for j in range(9)])
        enc=OneHotEncoder(sparse=False)
        enc.fit(ary)
        estimator = MLPRegressor(hidden_layer_sizes=(1024,512), max_iter=1)
        self.model = Pipeline([("preprocessing", enc), ("estimator", estimator)])
        """
        # OneHoeエンコーディングしない
        ary=[]
        for i in range(9):
            ary.append([(j+i)%9 for j in range(9)])
        scaler = StandardScaler()
        scaler.fit(ary)
        estimator = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=1)
        self.model = Pipeline([("preprocessing", scaler), ("estimator", estimator)])
        """
        
        self.update([experiences[0]], gamma=0)
        self.initialized = True

    def estimate(self, s):
        """
        self.policy内で使用する.
        """
        estimated = self.model.predict([s])[0]
        return estimated

    def _predict(self, states):# 各状態における、各行動の価値を予測し、返す。
        """
        学習時に使用する.
        """
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    def update(self, experiences, gamma):
        states = np.vstack([e.state for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])

        estimateds = self._predict(states)
        future = self._predict(next_states)

        for i, e in enumerate(experiences):# 経験を一つずつ処理
            reward = e.reward # 報酬を取得
            if not e.done:
                reward += gamma * np.max(future[i]) # n_sが終了状態でない場合、新しい状態の予測価値に割引率をかけたものを加算
            estimateds[i][e.action] = reward # これで予測値を上書き

        estimateds = np.array(estimateds)
        states = self.model.named_steps["preprocessing"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimateds) # 上書きした予測値で学習

if __name__ == "__main__":
    agent = QLearningNnAgent(epsilon=0.1,actions=[0,1,2,3])

