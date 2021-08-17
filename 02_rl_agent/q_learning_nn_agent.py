import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import json
import random
from collections import deque,defaultdict,namedtuple
import numpy as np
from square_puzzle import SquarePuzzle
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pickle
Experience = namedtuple("Experience",["state", "action", "reward", "next_state", "done"])
class QLearningNnAgent:
    def __init__(self,epsilon=0.1,buffer_size=1024,batch_size=64):
        self.epsilon = epsilon
        self.reward_log = []
        self.estimate_probs = False
        self.initialized = False
        self.model = None
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences = deque(maxlen=self.buffer_size)
        self.actions = []

    def initialize(self, experiences):
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

    def policy(self,state,actions):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(actions))
        else:
            estimates = self.estimate(state)
            if self.estimate_probs:
                action = np.random.choice(actions,size=1,p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def learn(self,env,episode_count=100,gamma=0.9,learning_rate=0.1,render=False,report_interval=50):
        """
        簡単な状態から学習するため,シャッフル回数の初期値を3として,十分学習できたら状態の難易度を上げる.
        50回勝率が9割以上になった時にシャッフル回数をプラスする.
        """
        actions = list(range(len(env.actions)))
        self.actions = actions
        shuffle_count = 3
        win_ratio = 0
        reward_ary = deque(maxlen=50)
        for e in range(episode_count):
            if win_ratio>.8:
                reward_ary = deque(maxlen=50)
                shuffle_count+=1
            state = env.reset(shuffle_count)
            done = False
            action_count = 0
            while not done:
                action = self.policy(state, actions)
                next_state, reward, done, info = env.step(action)
                action_count += 1
                if done:
                    reward = 1
                else:
                    reward = 0
                exp = Experience(state,action,reward,next_state,done)
                self.experiences.append(exp)

                if len(self.experiences) == self.batch_size:
                    self.initialize(self.experiences)

                if len(self.experiences) >= self.batch_size:
                    batch = random.sample(self.experiences, self.batch_size)
                    self.update(batch,gamma)

                state = next_state
                if action_count >= min(shuffle_count*2,100):break
            self.log(reward)
            reward_ary.append(reward)
            win_ratio = sum(reward_ary)/50
            print(e,reward,shuffle_count,round(win_ratio,3),action_count)
        self.save_model()

    def solve(self,env):
        self.read_model()
        actions = list(range(len(env.actions)))
        state = env.get_state()
        done = False
        action_count = 0
        route = []
        route.append(state.copy())
        while not done:
            action = self.policy(state, actions)
            next_state, done = env.step(action)
            action_count += 1
            state = next_state
            if action_count >= 300:break
        if done:
            return route
        else:
            print("not solved.")
            return []

    def log(self,reward):
        self.reward_log.append(reward)

    def save_model(self,):
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir,"model_q_nn.pkl"),"wb") as f:
            pickle.dump(self.model,f)
        pass

    def read_model(self,):
        pass

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
        """
        """
        states = np.vstack([e.state for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])

        estimateds = self._predict(states)
        future = self._predict(next_states)

        for i, e in enumerate(experiences):
            reward = e.reward # 報酬を取得
            if not e.done:
                reward += gamma * np.max(future[i]) # n_sが終了状態でない場合、新しい状態の予測価値に割引率をかけたものを加算
            estimateds[i][e.action] = reward # これで予測値を上書き

        estimateds = np.array(estimateds)
        states = self.model.named_steps["preprocessing"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimateds) # 上書きした予測値で学習

def train():
    agent = QLearningNnAgent()
    env = SquarePuzzle()
    agent.learn(env, episode_count=10000)

if __name__ == "__main__":
    train()

