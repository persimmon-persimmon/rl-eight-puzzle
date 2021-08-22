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
from q_learning_agent import QLearningAgent
from q_learning_nn_agent import QLearningNnAgent
from trainer_base import TrainerBase
import pickle
Experience = namedtuple("Experience",["state", "action", "reward", "next_state", "done"])

class SquarePuzzleTrainer(TrainerBase):

    def train_loop(self,env,agent,episode=200):
        self.experiences = deque(maxlen=self.buffer_size)
        self.reward_log = []
        self.training = False
        shuffle_count = 3
        win_ratio = 0
        reward_ary = deque(maxlen=50)
        for i in range(episode):
            if win_ratio>.9:
                reward_ary = deque(maxlen=50)
                shuffle_count += 1
            state = env.reset(shuffle_count)
            done = False
            while not done:
                action = agent.policy(state)
                next_state, reward, done, info = env.step(action)
                e = Experience(state, action, reward, next_state, done)
                self.experiences.append(e)
                if not self.training and len(self.experiences) == self.buffer_size:
                    agent.initialize(list(self.experiences)[:1])
                    self.training = True

                self.step(agent)
                state = next_state
                if info["step_count"]>=shuffle_count*2:break
            reward_ary.append(reward)
            win_ratio = sum(reward_ary)/50
            print(i,reward,shuffle_count,info["step_count"],self.training)
            #if shuffle_count>=5:return
    
    def step(self,agent):
        if self.training :
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)



def train():
    trainer = SquarePuzzleTrainer(buffer_size=1,batch_size=1,gamma=0.95)
    env = SquarePuzzle()
    #agent = QLearningAgent(epsilon=0.1,actions=env.actions)
    agent = QLearningNnAgent(epsilon=0.1,actions=env.actions)
    trainer.train_loop(env,agent,episode=100000)


if __name__ == "__main__":
    train()

