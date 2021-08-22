import random
from random import randint

class Node():
    """
    盤面をノード化する.
    ノード同士を比較できるよう比較演算用関数をつける.
    比較する値scoreは,スタートから自ノードまでの実コスト+自ノードからゴールまでの推定コスト.
    """
    def __init__(self,board,pre=None,action=None):
        """
        コンストラクタ.
        board:盤面.
        pre:親ノード.
        action:親ノードからこのノードへ遷移する時の行動.
        """
        self.board = board
        self.edge_length = int(len(board)**0.5)
        self.pre = pre
        self.action = action
        self.cost = pre.cost + 1 if pre is not None else 0 # 初期状態からこのノードまでの実コスト
        self.heuristic = self._get_heuristic() # boardからゴールまでの推定コスト（ヒューリスティック値）
        self.score = self.heuristic + self.cost
    
    def _get_heuristic(self):
        """
        最終状態までの推定コストを返す.
        """
        ret=0
        for i in range(self.edge_length):
            for j in range(self.edge_length):
                t = self.board[i * self.edge_length+j] - 1
            ti,tj = divmod(t, self.edge_length)
            ret += abs(i - ti) + abs(j - tj)
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

def all_solvable_boards():
    """
    すべての完成可能な盤面を返す. edge_length=3のみ.
    空白マスの完成状態からのマンハッタン距離(=pt0)と
    任意の2マスを置換できる場合の完成状態までの置換回数(=pt1)の偶奇が一致している場合、完成可能.
    """
    end_board = [1,2,3,4,5,6,7,8,0]
    all_pat = []
    todo = [[i] for i in range(9)]
    while todo:
        v=todo.pop()
        for ni in range(9):
            if ni not in v:
                if len(v)+1==9:
                    all_pat.append(v+[ni])
                else:
                    todo.append(v+[ni])
    ret = []
    for pat in all_pat:
        pt0 = -1
        for x in range(9):
            if pat[x]==0:
                i,j=divmod(x,3)
                pt0 = 2 - i + 2 - j
                break
        pt1 = 0
        g = [[] for _ in range(9)]
        for x,y in zip(pat,end_board):
            g[x].append(y)
            g[y].append(x)
        mi=set(range(9))
        while mi:
            v0=mi.pop()
            todo=[v0]
            count=1
            while todo:
                v=todo.pop()
                for nv in g[v]:
                    if nv in mi:
                        mi.discard(nv)
                        count += 1
                        todo.append(nv)
            pt1+=count-1
        if pt0%2==pt1%2:
            ret.append(pat)
    return ret

from random import randint
class SquarePuzzle:
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
        self.step_count = 0

    def reset(self,shuffle_count=100):
        """
        板を初期化する.
        最終状態からshuffle_count回シャッフルする.shuffle_countが増えるほどパズルの難易度が上がる.
        """
        self.board=[i + 1 for i in range(self.edge_length ** 2)]
        self.board[-1]=0  
        self.space = self.edge_length ** 2 - 1
        self.step_count = 0
        pre_space = -1
        for _ in range(shuffle_count):
            i, j = divmod(self.space, self.edge_length)
            di, dj = self.actions[randint(0, len(self.actions) - 1)]
            ni, nj = i + di,j + dj
            if 0 <= ni < self.edge_length and 0 <= nj < self.edge_length and ni * self.edge_length + nj != pre_space:
                self.board[self.space], self.board[ni * self.edge_length+nj] = self.board[ni * self.edge_length + nj], self.board[self.space]
                pre_space = self.space
                self.space = ni * self.edge_length + nj
        return tuple(self.board)

    def step(self,action,air=False):
        """
        行動の結果(状態,報酬,終了フラグ,info)を返す.
        指定の方向へ動かせない場合,状態を変えず返す.
        action:行動.空白マスを動かす方向を表す. {0:右,1:左,2:下,3:上}
        air:状態を変えずに行動の結果を取得したい場合Trueにする.
        """
        if not air:self.step_count += 1
        i,j = divmod(self.space,self.edge_length)
        di,dj = self.actions[action]
        ni,nj = i+di,j+dj
        if air:
            board_ = self.board.copy()
        else:
            board_ = self.board
        if 0 <= ni < self.edge_length and 0 <= nj < self.edge_length:
            board_[self.space],board_[ni*self.edge_length+nj] = board_[ni*self.edge_length+nj],board_[self.space]
            if not air:self.space = ni * self.edge_length+nj
        done = all(board_[i] == (i + 1) % (self.edge_length ** 2) for i in range(self.edge_length ** 2))
        reward = 1 if done else 0
        info = {"step_count":self.step_count}
        return tuple(board_), reward, done, info
    
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

if __name__=='__main__1':
    """
    8パズルを遊ぶ.
    """
    env = SquarePuzzle(3)
    env.reset()
    done = False
    board = env.get_state()
    while done==False:
        env.show()
        print("select space move direction.right=0, left=1, down=2, up=3.")
        try:
            action=int(input())
            assert 0<=action<4
            board,reward,done,info = env.step(action)
        except KeyboardInterrupt:
            print("recieve [ctrl + C]")
            break
        except Exception as e:
            print("invalid input. try again.",e)
    else:
        print("==== complete!! ===")
        env.show()

if __name__=='__main__':
    """
    各ソルバーを比較する.
    """
    import time
    board = [6, 0, 2, 7, 4, 1, 8, 3, 5]
    env = SquarePuzzle(edge_length=4)
    env.reset()
    board = env.board
    print(board)

    from solver.bfs_solver import bfs_solver
    from solver.a_star_solver import A_star_solver
    from solver.iddfs_solver import IDDFS_solver
    from solver.id_a_star_solver import ID_A_star_solver
    solvers = (bfs_solver,A_star_solver,IDDFS_solver,ID_A_star_solver)
    solvers = (A_star_solver,ID_A_star_solver)
    from memory_profiler import memory_usage
    import numpy as np
    for solver in solvers:
        env = SquarePuzzle(edge_length=4 ,board=board)
        start = time.time()
        #flg,actions,search_count = solver(env)
        mem = memory_usage((solver,(env,)))
        process_time = time.time() - start
        #print(f"{solver.__name__}:hands={len(actions)}, search_count={search_count}, process_time = {process_time}, memory = {np.mean(mem)}")
        print(f"{solver.__name__}:process_time = {process_time}, memory = {np.mean(mem)}")


