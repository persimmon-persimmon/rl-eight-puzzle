import sys
import os
current_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir,"../"))
import time
from square_puzzle import SquarePuzzle,Node
from setup_logger import setup_logger
from heapq import heappop,heappush

def A_star_solver(env,logger=None):
    """
    A starアルゴリズムで解く.
    env:square puzzleインスタンス.
    logger:loggerインスタンス.
    return:初期状態から最終状態までのaction配列.
    """
    def log(message):
        if logger is not None:
            logger.info(message)
    board = env.get_state()
    node = Node(board,pre=None,action=None)
    dist = {}
    dist[board] = node
    q = []
    heappush(q,node)
    log("search start.")
    search_count = 0
    end_node = None
    while end_node is None and q:
        search_count += 1
        if search_count % 1000 == 0:log(f"..search states = {search_count}")
        node = heappop(q)
        env_ = SquarePuzzle(env.edge_length,node.board)
        for action in env_.get_able_actions():
            next_state, reward, done, info = env_.step(action,air=True)
            next_node = Node(next_state,pre=node,action=action)
            if done:
                end_node = next_node
                break
            if next_state in dist and dist[next_state]<=next_node:continue
            dist[next_state] = next_node
            heappush(q,next_node)

    log("search end.")
    if end_node is None:
        log("no answer.")
        return [],search_count
    node = end_node
    actions = []
    while node.pre is not None:
        actions.append(node.action)
        node = node.pre
    actions.reverse()
    return actions,search_count

if __name__=='__main__':
    env = SquarePuzzle(edge_length=3)
    env.reset()
    logger = setup_logger(__file__)
    start = time.time()
    actions,search_count = A_star_solver(env,logger)
    process_time = time.time() - start
    for action in actions:
        env.show()
        env.step(action)
        print("↓")
    env.show()
    print(f"hands={len(actions)}, search_count={search_count}, process_time = {process_time}")