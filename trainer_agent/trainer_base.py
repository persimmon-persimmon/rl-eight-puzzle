from collections import deque,namedtuple

"""
経験クラス. 毎回の行動の[状態,行動,報酬,次の状態,終了フラグ]をクラス化する. これを蓄積し,モデルの学習を行う.
"""
Experience = namedtuple("Experience",["state", "action", "reward", "next_state", "done"])

class TrainerBase():

    def __init__(self,buffer_size=1024,batch_size=32,gamma=0.9):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.experiences = deque(maxlen=buffer_size)
        self.training = False

    def train_loop(self,env,agent,episode=200):
        self.experiences = deque(maxlen=self.buffer_size)
        self.reward_log = []
        self.training = False

        for _ in range(episode):
            state = env.reset()
            done = False
            step_count = 0
            while not done:
                action = agent.policy(state)
                next_state,reward,done,info = env.step(action)
                e = Experience(state, action, reward, next_state, done)
                self.experiences.append(e)
                if not self.training and len(self.experiences) == self.buffer_size:
                    agent.initialize(self.experiences[0])
                    self.training = True

                self.step(agent)
                state = next_state
                step_count += 1

    def step(self,agent):
        """
        毎回の行動ごとに呼び出す関数.子クラスで実装.
        """
        raise NotImplementedError(f"class={self.__class__.__name__}, method={sys._getframe().f_code.co_name}.")

if __name__ == "__main__":
    trainer = Trainer()

