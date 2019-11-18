import numpy as np
from tetris.utils import print_board_to_string, print_tetromino
from numba import njit


@njit
def evaluate(env, agent, num_runs, max_cleared_lines=np.inf, verbose=False):
    np.random.seed(1)
    rewards = np.zeros(num_runs, dtype=np.int64)
    for i in range(num_runs):
        env.reset()
        while not env.game_over and env.cleared_lines < max_cleared_lines:
            if verbose:
                print(print_board_to_string(env.current_board))
                print(print_tetromino(env.current_tetromino.current_tetromino_index))
            after_state = agent.choose_action(env=env)

            if verbose and after_state.is_terminal_state:
                print("Game over")
                print(env.cleared_lines)
            env.step(after_state)

        rewards[i] = env.cleared_lines
    return rewards
