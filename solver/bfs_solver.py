import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import time
from square_puzzle import SquarePuzzle, Node
from setup_logger import setup_logger
from collections import deque

def bfs_solver(env,logger=None):
    """
    幅優先探索で解く.
    env:square puzzleインスタンス.
    logger:loggerインスタンス.使わない場合はNone.
    -> return:解けたかどうかのフラグ,初期状態から最終状態までのaction配列,探索回数.
    """
    def log(message):
        if logger is not None:
            logger.info(message)
    board = env.get_state()

    # はじめから最終状態のときの処理
    if all(board[i]==(i+1)%(len(board)) for i in range(len(board))):
        return True,[],0

    node = Node(board,pre=None,action=None)
    dist = {}
    dist[board] = node
    q = deque()
    q.append(node)
    log("search start.")
    search_count = 0
    end_node = None
    while end_node is None and q:
        search_count += 1
        if search_count % 1000 == 0:log(f"..search states = {search_count}")
        node = q.popleft()
        env_ = SquarePuzzle(env.edge_length,node.board)
        for action in env_.get_able_actions():
            next_state, reward, done, info = env_.step(action,air=True)
            next_node = Node(next_state,node,action)
            if done:
                end_node = next_node
                break
            if next_state in dist:continue
            dist[next_state] = next_node
            q.append(next_node)

    log("search end.")
    if end_node is None:
        log("no answer.")
        return False,[],search_count
    node = end_node
    actions = []
    while node.pre is not None:
        actions.append(node.action)
        node = node.pre
    actions.reverse()
    return True,actions,search_count

if __name__=='__main__':
    env = SquarePuzzle(edge_length=3)
    env.reset()
    logger = setup_logger(__file__)
    start = time.time()
    flg,actions,search_count = bfs_solver(env,logger)
    process_time = time.time() - start
    for action in actions:
        env.show()
        env.step(action)
        print("↓")
    env.show()
    print(f"hands={len(actions)}, search_count={search_count}, process_time = {process_time}")
