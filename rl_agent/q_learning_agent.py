import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import json
from square_puzzle import SquarePuzzle
from collections import deque,defaultdict
import numpy as np
import pickle
class QLearningAgent:
    """
    Q-Learningのエージェント.
    """
    def __init__(self,epsilon=0.1):
        """
        コンストラクタ.
        epsilon:epsilon-greedy法のepsilon. ここで指定した割合だけ探索的行動をする.
        """
        self.epsilon = epsilon
        self.Q = {}

    def policy(self,state,actions):
        """
        epsilon-greedy法で決定した行動を返す.
        epsilonの割合だけランダムに行動を決める.
        それ以外は過去の経験から算出した価値の高い行動を取る.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if state in self.Q and sum(self.Q[state]) != 0:
                return np.argmax(self.Q[state])
            else:
                return np.random.randint(len(actions))

    def learn(self,env,episode_count=100,gamma=0.9,learning_rate=0.1,read_model=False):
        """
        学習する.
        簡単な状態から学習するため,シャッフル回数の初期値を3として,十分学習できたら状態の難易度を上げる.
        50回勝率が9割以上になった時にシャッフル回数をプラスする.
        """
        actions = list(range(len(env.actions)))
        self.Q = defaultdict(lambda: [0]*len(actions))
        if read_model:self.read_model()
        shuffle_count = 3
        win_ratio = 0
        reward_ary = deque(maxlen=50)
        self.log_ary = []
        for i in range(episode_count):
            if win_ratio > 0.9:
                reward_ary = deque(maxlen=50)
                shuffle_count+=1
            state = env.reset(shuffle_count)
            done = False
            while not done:
                action = self.policy(state, actions)
                next_state, reward, done, info = env.step(action)
                gain = reward + gamma * max(self.Q[next_state])
                estimated = self.Q[state][action]
                self.Q[state][action] += learning_rate * (gain - estimated)
                state = next_state
                if info["step_count"] >= shuffle_count * 2:break
            reward_ary.append(reward)
            win_ratio = sum(reward_ary)/50
            self.log_ary.append([i,reward,win_ratio,shuffle_count,info["step_count"],len(self.Q)])
        self.save_model()
        self.output_log()
    
    def output_log(self,):
        """
        log_aryを出力する.
        """
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir,"q_learning.json"),"w") as f:
            f.write(json.dumps(self.log_ary))

    def save_model(self,):
        """
        Qテーブルを保存する.
        """
        current_dir = os.path.dirname(__file__)
        pickle.dump(dict(self.Q), open(os.path.join(current_dir,"q_table.pkl"),"wb"))
        
    def read_model(self,):
        """
        Qテーブルを読み込む.
        """
        current_dir = os.path.dirname(__file__)
        Q_ = pickle.load(open(os.path.join(current_dir,"q_table.pkl"),"rb"))
        action_length = len(list(Q_.keys())[0])
        self.Q = defaultdict(lambda: [0]*action_length)
        for k in Q_:
            self.Q[k] = Q_[k]

def train():
    agent = QLearningAgent()
    env = SquarePuzzle()
    agent.learn(env, episode_count=1000,gamma=0.95)

if __name__ == "__main__":
    train()

