import numpy as np
from numba import njit


@njit
def evaluate(env, agent, num_runs, verbose=False):
    np.random.seed(1)
    rewards = np.zeros(num_runs, dtype=np.int64)
    for i in range(num_runs):
        if verbose:
            print(f"Game {i} out of {num_runs}.")
        env.reset()
        while not env.game_over:
            if verbose:
                env.print_current_board()
                env.print_current_tetromino()
            after_state = agent.choose_action(env=env)

            if verbose and after_state.is_terminal_state:
                print("Game over")
                print(env.cleared_lines)
            env.step(after_state)

        rewards[i] = env.cleared_lines
    return rewards
