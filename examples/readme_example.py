import tetris
import agents
import numpy as np

# Create a Tetris environment using a 10 x 10 board.
env = tetris.Tetris(num_columns=10, num_rows=10)

# Create an agent with a randomly chosen linear policy.
agent = agents.LinearAgent(policy_weights=np.random.normal(0, 1, 8))

env.reset()
game_over = False
cleared_lines = 0
while not game_over:
    env.print_current_board()
    env.print_current_tetromino()
    after_state = agent.choose_action(env)
    _, reward, game_over = env.step(after_state)
    cleared_lines += reward

print(f"The agent cleared {np.mean(cleared_lines)} lines.")

env.print_current_board()
