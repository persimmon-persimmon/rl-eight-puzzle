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
        self.reward_log = []
        self.actions = list(range(len(env.actions)))
        self.Q = defaultdict(lambda:np.random.uniform(0,1,len(self.actions)))
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, state):
        action = np.random.choice(self.actions, 1,p=self.softmax(self.Q[state]))
        return action[0]

class Critic():
    def __init__(self, env):
        self.V = defaultdict(int)

class ActorCritic():
    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.reward_log = []

    def train(self, env, episode_count=1000, gamma=0.9,learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)
        for e in range(episode_count):
            shuffle_count = max(int(50 * e/episode_count),3)
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
                if action_count >= 300:break
            self.log(reward)
            print(e,reward,shuffle_count,action_count)
            #if e != 0 and e % report_interval == 0:actor.show_reward_log(episode=e)
        return actor, critic

    def log(self,reward):
        self.reward_log.append(reward)

    def save_Q(self,):
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir,"q_table.json"),"w") as f:
            f.write(json.dumps(dict(self.Q)))

    def read_Q(self,):
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir,"q_table.json"),"r") as f:
            Q_=json.loads(f.read())
        k=Q_.keys()[0]
        action_num = len(Q_[k])
        self.Q=defaultdict(lambda:[0]*action_num)
        for k in Q_:
            self.Q[k]=Q_[k]
        
def train():
    agent = ActorCritic(Actor, Critic)
    env = SquarePuzzle()
    agent.train(env, episode_count=10000)

if __name__ == "__main__":
    train()

