import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import json
from square_puzzle import SquarePuzzle
from collections import deque,defaultdict
from agent_base import AgentBase
import numpy as np
class QLearningAgent(AgentBase):
    def __init__(self,epsilon,actions,learning_rate=0.1):
        super().__init__(epsilon,actions)
        self.Q = None
        self.learning_rate = learning_rate

    def save(self,model_path):
        """
        モデルを保存する.
        """
        with open(model_path,"w") as f:
            f.write(json.dumps(dict(self.Q)))

    def read(self,model_path):
        """
        モデルを読み込む.
        """
        with open(model_path,"r") as f:
            Q_ = json.loads(f.read())
        self.Q = defaultdict(lambda:[0]*len(self.actions))
        for k in Q_:
            self.Q[k] = Q_[k]

    def initialize(self, experiences=None):
        """
        エージェントを初期化する.
        """
        self.Q = defaultdict(lambda:[0]*len(self.actions))
        self.initialized = True

    def estimate(self, state):
        """
        与えた状態における各行動の価値を返す. policyから呼び出す.
        """
        if state in self.Q and sum(self.Q[state]) != 0:
            return self.Q[state]
        else:
            return np.random.random(len(self.actions))

    def update(self, experiences, gamma):
        """
        与えた経験でモデルを学習する. 外部から呼び出す.
        """
        for e in experiences:
            gain = e.reward + gamma * max(self.Q[e.next_state])
            estimated = self.Q[e.state][e.action]
            self.Q[e.state][e.action] += self.learning_rate * (gain - estimated)

if __name__ == "__main__":
    agent = QLearningAgent(epsilon=0.1,actions=[0,1,2,3])

