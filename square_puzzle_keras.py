from random import randint
import gym.spaces
import copy
import numpy as np
import random
import rl.callbacks
import rl.core
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory

class SquarePuzzleEnv(gym.core.Env):
    """
    正方形パズルクラス
    一辺の長さ=3の場合,8パズルになる. 以下が最終状態. 空白マスは0で表す.
    1 2 3
    4 5 6
    7 8 0
    一辺の長さ=4の場合,15パズルになる.
    """
    def __init__(self,edge_length=3,board=None):
        """
        edge_length:一辺の長さ.
        board:初期状態を指定する場合に使う. マスの配置を一次元化したもの.
        """
        if board is not None:
            assert len(board)==edge_length**2,f"invalid square. edge_length={edge_length} and board={board}"
            self.space = [x for x in range(edge_length ** 2) if board[x]==0][0]
            board = list(board)
        else:
            board=[i+1 for i in range(edge_length**2)]
            board[-1]=0  
            self.space = edge_length ** 2 - 1
        self.edge_length = edge_length
        self.board = board
        self.actions = [[0,1],[0,-1],[1,0],[-1,0]]
        self.n_action = 4
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.edge_length,self.edge_length,self.edge_length**2))
        self.step_count = 0
        self.shuffle_count = 100

    def reset(self):
        """
        板を初期化する.
        最終状態からshuffle_count回シャッフルする.shuffle_countが増えるほどパズルの難易度が上がる.
        """
        self.board=[i + 1 for i in range(self.edge_length ** 2)]
        self.board[-1]=0  
        self.space = self.edge_length ** 2 - 1
        self.step_count = 0
        pre_space = -1
        for _ in range(self.shuffle_count):
            i, j = divmod(self.space, self.edge_length)
            di, dj = self.actions[randint(0, len(self.actions) - 1)]
            ni, nj = i + di,j + dj
            if 0 <= ni < self.edge_length and 0 <= nj < self.edge_length and ni * self.edge_length + nj != pre_space:
                self.board[self.space], self.board[ni * self.edge_length+nj] = self.board[ni * self.edge_length + nj], self.board[self.space]
                pre_space = self.space
                self.space = ni * self.edge_length + nj
        return self.get_state()

    def step(self,action):
        """
        行動の結果(状態,報酬,終了フラグ,info)を返す.
        指定の方向へ動かせない場合,状態を変えず返す.
        action:行動.空白マスを動かす方向を表す. {0:右,1:左,2:下,3:上}
        """
        self.step_count += 1
        i,j = divmod(self.space,self.edge_length)
        di,dj = self.actions[action]
        ni,nj = i+di,j+dj
        if 0 <= ni < self.edge_length and 0 <= nj < self.edge_length:
            self.board[self.space],self.board[ni*self.edge_length+nj] = self.board[ni*self.edge_length+nj],self.board[self.space]
            self.space = ni * self.edge_length+nj
        done = all(self.board[i] == (i + 1) % (self.edge_length ** 2) for i in range(self.edge_length ** 2))
        reward = 1 if done else 0
        info = {"step_count":self.step_count,"shuffle_count":self.shuffle_count}
        return self.get_state(), reward, done, info
    
    def get_state(self):
        """
        現在の状態を返す.
        """
        return tuple(self.board)
        
    def get_able_actions(self):
        """
        可能なアクションを返す.
        """
        ret = []
        i,j = divmod(self.space,self.edge_length)
        if j < self.edge_length - 1:ret.append(0) # 右
        if 0 < j:ret.append(1) # 左
        if i < self.edge_length - 1:ret.append(2) # 下
        if 0 < i:ret.append(3) # 上
        return ret

    def show(self):
        """
        現在の状態をコンソールに表示する.
        """
        for i in range(self.edge_length):
            print(self.board[i*self.edge_length:(i+1)*self.edge_length])

    def set_shuffle_count(self,count):
        self.shuffle_count = count

# 難易度を調整するCallback
class ControlCallbak(rl.callbacks.Callback):
    def __init__(self,start_shuffle_count = 3):
        self.reward_history = deque([0]*50,maxlen=50)
        self.shuffle_count = start_shuffle_count

    def on_episode_end(self, episode, logs):
        self.reward_history.append(logs["episode_reward"])
        wp = sum(self.reward_history)/len(self.reward_history)
        if wp > 0.9:
            self.shuffle_count += 1
            self.env.set_shuffle_count(self.shuffle_count)
            self.reward_history = deque([0]*50,maxlen=50)

    def on_train_begin(self,logs):
        self.env.set_shuffle_count(self.shuffle_count)

# エピソードを打ち切るProcessor
class TrainProcessor(rl.core.Processor):
    def process_step(self, observation, reward, done, info):
        if info["step_count"]>2*info["shuffle_count"]:
            done = 1
        observation = self.process_observation(observation)
        return observation, reward, done, info

    def process_observation(self, observation):
        # 盤面をOneHotエンコーディングしテンソルにする
        ret = []
        ary = [0]*len(observation)
        l = int(round(len(observation)**0.5))
        for i in observation:
            ary[i] = 1
            ret.extend(ary.copy())
            ary[i] = 0
        ret = np.array(ret)
        ret = ret.reshape((l,l,l**2))
        return ret


env = SquarePuzzleEnv()
model = Sequential()
model.add(Conv2D(16,(2,2),activation="relu",input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(32,(2,2),activation="relu"))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(env.n_action))
model.add(Activation('linear'))
print(model.summary())

# モデルのコンパイル
pc = TrainProcessor()
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(0.1)
dqn = DQNAgent(model=model, nb_actions=env.n_action, memory=memory, nb_steps_warmup=500, target_model_update=1e-2, policy=policy, processor=pc)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

cc = ControlCallbak()
history = dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, nb_max_episode_steps=300, callbacks=[cc])
