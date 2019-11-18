import tetris
from agents.linear_agent import LinearAgent
import numpy as np
import time
from examples.evaluate import evaluate

print("TEST pytetris: will play 5 games using a random linear policy.")
np.random.seed(1)
max_cleared_lines = np.inf
verbose = True
num_runs = 5
start = time.time()
env = tetris.Tetris(num_columns=10, num_rows=10)

print("Compile numba functions.")
agent = LinearAgent(policy_weights=np.random.normal(0, 1, 8))
random_rewards = evaluate(env, agent, num_runs, max_cleared_lines, verbose)

end = time.time()
print("The games took ", end - start, " seconds.")

print("Test successful")


