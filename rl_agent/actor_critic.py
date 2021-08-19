import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import json
from square_puzzle import SquarePuzzle
from collections import deque,defaultdict
import numpy as np
class Actor:
    def __init__(self,env,epsilon=0.1):
        self.epsilon = epsilon
        self.actions = list(range(len(env.actions)))
        self.Q = defaultdict(lambda:np.random.uniform(0, 1, len(self.actions)))
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, state):
        action = np.random.choice(self.actions, 1, p=self.softmax(self.Q[state]))
        return action[0]

class Critic():
    def __init__(self, env):
        self.V = defaultdict(int)

class ActorCritic():
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env, episode_count=1000, gamma=0.9,learning_rate=0.1, render=False, report_interval=50):
        """
        学習する.
        簡単な状態から学習するため,シャッフル回数の初期値を3として,十分学習できたら状態の難易度を上げる.
        50回勝率が9割以上になった時にシャッフル回数をプラスする.
        """
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        shuffle_count = 3
        win_ratio = 0
        reward_ary = deque(maxlen=50)
        for i in range(episode_count):
            if win_ratio > 0.9:
                reward_ary = deque(maxlen=50)
                shuffle_count += 1
            state = env.reset(shuffle_count)
            done = False
            action_count = 0
            while not done:
                action = actor.policy(state)
                next_state, reward, done, info = env.step(action)
                action_count += 1                
                if done:
                    reward = 1 if action_count <= min(shuffle_count,31) else 1 - (action_count-min(shuffle_count,31))/300
                else:
                    reward = 0
                gain = reward + gamma * critic.V[next_state]
                estimated = critic.V[state]
                td = gain - estimated
                actor.Q[state][action] += learning_rate * td
                critic.V[state] += learning_rate * td
                state = next_state
                if info["step_count"] >= shuffle_count * 2:break
            reward_ary.append(reward)
            win_ratio = sum(reward_ary)/50
            print(i,reward,win_ratio,shuffle_count,info["step_count"],len(actor.Q),len(critic.V))
        return actor, critic

def train():
    agent = ActorCritic(Actor, Critic)
    env = SquarePuzzle()
    agent.train(env, episode_count=10000)

if __name__ == "__main__":
    train()

