import tetris
import agents
# from agents.linear_agent import LinearAgent
import numpy as np
import time
from datetime import datetime
from examples.evaluate import evaluate

time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
np.random.seed(1)
max_cleared_lines = np.inf
verbose = False


num_runs = 50
start = time.time()
env = tetris.Tetris(num_columns=10, num_rows=10)

print("RANDOM policy")
agent = agents.LinearAgent(policy_weights=np.random.normal(0, 1, 8))
random_rewards = evaluate(env, agent, num_runs, max_cleared_lines=np.inf, verbose=True)

print("Directed equal weights policy")
agent = agents.LinearAgent(policy_weights=np.array([-1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64))
ew_rewards = evaluate(env, agent, num_runs, max_cleared_lines, verbose)

print("Canonical non-compensatory weighting (i.e., 1/2, 1/4, 1/8, 1/16, 1/32, ...)")
agent = agents.LinearAgent(policy_weights=0.5**np.arange(8) * np.array([-1, -1, -1, -1, -1, -1, 1, -1]))
ttb_rewards = evaluate(env, agent, num_runs, max_cleared_lines, verbose)

end = time.time()
print("All together took ", end - start, " seconds.")

with open("t" + time_id + ".txt", "w") as text_file:
    print("Started at: " + time_id, file=text_file)  # + " from file " + str(__file__)
    print("Time spent: " + str((end - start)) + "seconds.", file=text_file)
    print("Random Rewards are: " + str(random_rewards), file=text_file)
    print("Random Rewards mean is : " + str(np.mean(random_rewards)), file=text_file)
    print("EW rewards are: " + str(ew_rewards), file=text_file)
    print("EW rewards mean is: " + str(np.mean(ew_rewards)), file=text_file)
    print("TTB rewards are: " + str(ttb_rewards), file=text_file)
    print("TTB rewards mean is: " + str(np.mean(ttb_rewards)), file=text_file)

