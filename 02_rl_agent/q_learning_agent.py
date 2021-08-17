import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import json
from square_puzzle import SquarePuzzle
from collections import deque,defaultdict
import numpy as np
class QLearningAgent:
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon
        self.Q = {}
        self.reward_log = []

    def policy(self,state,actions):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if state in self.Q and sum(self.Q[state]) != 0:
                return np.argmax(self.Q[state])
            else:
                return np.random.randint(len(actions))

    def learn(self,env,episode_count=100,gamma=0.9,learning_rate=0.1):
        """
        簡単な状態から学習するため,シャッフル回数の初期値を3として,十分学習できたら状態の難易度を上げる.
        50回勝率が9割以上になった時にシャッフル回数をプラスする.
        """
        actions = list(range(len(env.actions)))
        self.Q = defaultdict(lambda: [0]*len(actions))
        shuffle_count = 3
        win_ratio = 0
        reward_ary = deque(maxlen=50)
        for i in range(episode_count):
            if win_ratio>.8:
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
            self.log(reward)
            reward_ary.append(reward)
            win_ratio = sum(reward_ary)/50
            print(i,reward,shuffle_count,info["step_count"],len(self.Q))
            if shuffle_count>=5:return
        self.save_model()

    def log(self,reward):
        self.reward_log.append(reward)

    def save_model(self,):
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir,"q_table.json"),"w") as f:
            f.write(json.dumps(dict(self.Q)))

    def read_model(self,):
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir,"q_table.json"),"r") as f:
            Q_=json.loads(f.read())
        k=Q_.keys()[0]
        action_num = len(Q_[k])
        self.Q=defaultdict(lambda:[0]*action_num)
        for k in Q_:
            self.Q[k]=Q_[k]
        
def train():
    agent = QLearningAgent()
    env = SquarePuzzle()
    agent.learn(env, episode_count=1000,gamma=0.95)

if __name__ == "__main__":
    train()

