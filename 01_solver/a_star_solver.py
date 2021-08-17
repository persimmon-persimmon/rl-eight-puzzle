import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
from square_puzzle import SquarePuzzle
from heapq import heappop,heappush
class Node():
    def __init__(self,board,pre=None):
        self.board = board
        self.pre = pre
        self.cost = pre.cost + 1 if pre is not None else 0 # 初期位置からboardまでの実コスト
        self.heuristic = self._get_heuristic() # boardからゴールまでの推定コスト（ヒューリスティック値）
        self.score = self.heuristic + self.cost
    
    def _get_heuristic(self):
        """
        最終状態までの推定コストを返す.
        """
        ret=0
        for i in range(3):
            for j in range(3):
                t = self.board[i*3+j] - 1
            ti,tj = divmod(t,3)
            ret += abs(i-ti) + abs(j-tj)
        return ret

    # Node比較用関数
    def __le__(self,other):
        return self.score <= other.score
    def __ge__(self,other):
        return self.score >= other.score
    def __lt__(self,other):
        return self.score < other.score
    def __gt__(self,other):
        return self.score > other.score
    def __qe__(self,other):
        return self.score == other.score

def A_star_solver(env):
    """
    A starアルゴリズムで解く.
    env:square puzzleクラス.
    return:route(最終状態までの遷移),search_count(探索した状態の総数)
    """
    board = env.get_state()
    node = Node(board,pre=None)
    dist = {}
    dist[board] = node
    q = []
    heappush(q,node)
    print("search start.")
    search_count = 0
    end_node = None
    while end_node is None and q:
        search_count += 1
        if search_count % 1000 == 0:
            print(f"..search states = {search_count}")
        node = heappop(q)
        env_ = SquarePuzzle(env.edge_length,node.board)
        for action in env_.get_able_actions():
            next_state,reward,done,info = env_.step(action,air=True)
            next_node = Node(next_state,pre=node)
            if done:
                end_node = next_node
                break
            if next_state in dist and dist[next_state]<=next_node:continue
            dist[next_state] = next_node
            heappush(q,next_node)

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
    route,search_count = A_star_solver(env)
    for node in route:
        print(node.board[:3])
        print(node.board[3:6])
        print(node.board[6:])
        print("↓")
    print(f"hands={len(route)},search_count={search_count}")
