import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import time
from square_puzzle import SquarePuzzle, Node, all_solvable_boards
from setup_logger import setup_logger
from random import random
import numpy as np

def PolicyIteration(gamma=0.95, th=0.01):
    """
    policy iterationで状態の価値評価する.
    ->return:状態価値テーブルV
    """
    actions = list(range(4))
    states = all_solvable_boards()
    policy = {}
    for state in states:
        policy[tuple(state)] = [random() for _ in actions]

    def estimate_by_policy(gamma,th,policy):
        print("estimate_by_policy start.")
        V = {}
        th = 5/100
        for state in states:
            V[tuple(state)] = 0
        while True:
            delta = 0
            for state in V:
                if state == (1,2,3,4,5,6,7,8,0):continue
                env_ = SquarePuzzle(edge_length=3, board=state)
                expected_rewards = []
                for action in actions:
                    action_prob = policy[state][action]
                    next_state, reward, done, info = env_.step(action,air=False)
                    r = action_prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[state]))
                V[state] = value
            if delta < th:
                break
        print("estimate_by_policy end.")
        return V
    
    count = 0
    while True:
        V = estimate_by_policy(gamma, th, policy)
        update_stable = True
        not_best_action_count = 0
        for state in V:
            env_ = SquarePuzzle(edge_length=3, board=state)
            policy_action = np.argmax(policy[state])
            action_rewards = {}
            for action in actions:
                next_state, reward, done, info = env_.step(action,air=False)
                r = reward + gamma * V[next_state]
                action_rewards[action] = r
            best_action = np.argmax(action_rewards)
            if policy_action != best_action:
                update_stable = False
                not_best_action_count += 1
            for action in actions:
                prob = 1 if action == best_action else 0
                policy[state][action] = prob
        count += 1
        if count % 5 == 1:
            print(count,not_best_action_count)
        if update_stable:
            break
    return V,policy

if __name__=='__main__':
    V,policy = PolicyIteration()
    import pickle
    current_dir=os.path.dirname(__file__)
    with open(os.path.join(current_dir,"V_policy.pkl"),"wb") as f:
        pickle.dump(V,f)
    with open(os.path.join(current_dir,"policy.pkl"),"wb") as f:
        pickle.dump(policy,f)
    cnt=0
    print(V[(1,2,3,4,5,6,7,8,0)])
    print(V[(1,2,3,4,5,6,7,0,8)])
    print(policy[(1,2,3,4,5,6,7,0,8)])
    print(policy[(1,2,3,4,5,0,7,8,6)])
    for k,v in V.items():
        print(k,v)
        cnt+=1
        if cnt>10:break

