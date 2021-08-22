import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import time
from square_puzzle import SquarePuzzle, Node, all_solvable_boards
from setup_logger import setup_logger

def ValueIteration(gamma=0.95, th=0.01):
    """
    value iterationで全状態の価値評価する.
    はじめ,すべての状態は価値0とする.
    以下の価値更処理をすべての状態に行う.何度も繰り返し,更新幅がth未満になったら終了.
    ・終了状態に遷移できたら,その状態の価値を1とする.
    ・遷移可能な状態のうち最大価値に割引率gammaをかけたものをその状態の価値とする.
    繰り返しごとに終了状態に近いものから順に価値が伝搬していくイメージ.
    return:状態価値テーブルV
    """
    actions = list(range(4))
    states = all_solvable_boards()
    V = {}
    for state in states:
        V[tuple(state)] = 0
    count = 0
    while True:
        delta = 0 # 更新幅
        for state in V:
            if state == (1,2,3,4,5,6,7,8,0):continue
            expected_rewards = []
            env_ = SquarePuzzle(edge_length=3, board=state)
            for action in actions:
                next_state, reward, done, info = env_.step(action,air=True)
                r = reward + gamma * V[next_state]
                expected_rewards.append(r)
            max_reward = max(expected_rewards)
            delta = max(delta, abs(max_reward - V[state]))
            V[state] = max_reward
        if count % 10==0:
            print(count,delta)
        count+=1
        if delta < th:
            break
    return V

if __name__=='__main__':
    V = ValueIteration()
    import pickle
    current_dir=os.path.dirname(__file__)
    with open(os.path.join(current_dir,"V.pkl"),"wb") as f:
        pickle.dump(V,f)
    cnt=0
    print(V[(1,2,3,4,5,6,7,8,0)])
    print(V[(1,2,3,4,5,6,7,0,8)])
    for k,v in V.items():
        print(k,v)
        cnt+=1
        if cnt>20:break
