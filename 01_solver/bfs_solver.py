import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
from square_puzzle import SquarePuzzle
from collections import deque
class Node:
    def __init__(self,board,pre=None):
        self.board = board
        self.pre = pre
        self.cost = pre.cost+1 if pre is not None else 0 # 初期位置からboardまでの実コスト

    # Node比較用関数
    def __le__(self,other):
        return self.cost <= other.cost
    def __ge__(self,other):
        return self.cost >= other.cost
    def __lt__(self,other):
        return self.cost < other.cost
    def __gt__(self,other):
        return self.cost > other.cost
    def __qe__(self,other):
        return self.cost == other.cost

def bfs_solver(env):
    """
    幅優先探索で解く.
    env:square puzzleクラス.
    return:route(最終状態までの遷移),search_count(探索した状態の総数)
    """
    board = env.get_state()
    node = Node(board,pre=None)
    dist = {}
    dist[board] = node
    q = deque()
    q.append(node)
    print("search start.")
    search_count = 0
    end_node = None
    while end_node is None and q:
        search_count += 1
        if search_count % 1000 == 0:
            print(f"..search states = {search_count}")
        node = q.popleft()
        env_ = SquarePuzzle(env.edge_length,node.board)
        for action in env_.get_able_actions():
            next_state,reward,done,info = env_.step(action,air=True)
            next_node = Node(next_state,node)
            if done:
                end_node = next_node
                break
            if next_state in dist:continue
            dist[next_state] = next_node
            q.append(next_node)

    print("search end.")
    if end_node is None:
        print("no answer.")
        return [],search_count
    node = end_node
    route = []
    while node.pre is not None:
        route.append(node)
        node = node.pre
    route.reverse()
    return route,search_count

if __name__=='__main__':
    env = SquarePuzzle()
    env.reset()
    env.show()
    route,search_count = bfs_solver(env)
    for node in route:
        print(*node.board[:3])
        print(*node.board[3:6])
        print(*node.board[6:])
        print("↓")
    print(f"hands={len(route)},search_count={search_count}")
