import sys
import numpy as np
class AgentBase():
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    def save(self, model_path):
        """
        モデルを保存する.
        """
        raise NotImplementedError(f"class={self.__class__.__name__}, method={sys._getframe().f_code.co_name}.")

    def read(self, model_path):
        """
        モデルを読み込む.
        """
        raise NotImplementedError(f"class={self.__class__.__name__}, method={sys._getframe().f_code.co_name}.")

    def initialize(self, experiences):
        """
        エージェントを初期化する.
        """
        raise NotImplementedError(f"class={self.__class__.__name__}, method={sys._getframe().f_code.co_name}.")

    def estimate(self, state):
        """
        状態おける各行動の価値を返す. policyから呼び出す.
        """
        raise NotImplementedError(f"class={self.__class__.__name__}, method={sys._getframe().f_code.co_name}.")

    def update(self, experiences, gamma):
        """
        与えられた経験でモデルを学習する. 外部から呼び出す.
        """
        raise NotImplementedError(f"class={self.__class__.__name__}, method={sys._getframe().f_code.co_name}.")

    def policy(self, state):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(state)
            if self.estimate_probs:
                action = np.random.choice(self.actions,size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def play(self, env, episode_count=5, render=True):
        for i in range(episode_count):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                action = self.policy(state)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state

if __name__=='__main__':
    agent = AgentBase(epsilon=0.1,actions=[0,1,2,3,])
