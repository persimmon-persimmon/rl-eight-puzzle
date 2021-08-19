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
    """
    Q-Learningのエージェント.
    NNを使うためにExperience Replay機能を実装する.
    Fixed Target Q-Netwrok.
    """
    def __init__(self,epsilon=0.1,buffer_size=1024,batch_size=64):
        """
        コンストラクタ.
        epsilon:epsilon-greedy法のepsilon. ここで指定した割合だけ探索的行動をする.
        buffer_size:Experience Replayのため,バッファしておく経験の量.
        batch_size:一度の学習で使用する経験のサイズ.
        """
        self.epsilon = epsilon
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
        estimator = MLPRegressor(hidden_layer_sizes=(512,512), max_iter=1)
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
        """
        epsilon-greedy法で決定した行動を返す.
        epsilonの割合だけランダムに行動を決める.
        それ以外は過去の経験から算出した価値の高い行動を取る.
        """
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(actions))
        else:
            estimates = self.estimate(state)
            if self.estimate_probs:
                action = np.random.choice(actions,size=1,p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def estimate(self, s):
        """
        self.policy内で使用する.
        """
        estimated = self.model.predict([s])[0]
        return estimated

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
        self.log_ary = []
        for i in range(episode_count):
            if win_ratio > 0.9:
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
            reward_ary.append(reward)
            win_ratio = sum(reward_ary)/50
            self.log_ary.append([i,reward,win_ratio,shuffle_count,info["step_count"]])
            print(self.log_ary[-1])
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
        モデルを保存する.
        """
        current_dir = os.path.dirname(__file__)
        pickle.dump(self.model, open(os.path.join(current_dir,"model_q_nn.pkl"),"wb"))

    def read_model(self,):
        """
        モデルを読み込む.
        """
        current_dir = os.path.dirname(__file__)
        self.model = pickle.load(open(os.path.join(current_dir,"model_q_nn.pkl"),"rb"))

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
        学習する.
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

