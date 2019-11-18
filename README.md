# pytetris - Tetris for reinforcement learning applications in Python
 
This repository contains a [Tetris](https://en.wikipedia.org/wiki/Tetris) implementation. The implementation is tailored to be used within the context of reinforcement learning (RL) applications, similar (but different) to OpenAI's [gym](https://github.com/openai/gym) environments. Most existing successful Tetris RL algorithms learn to evaluate afterstates (e.g., [Sutton & Barto 2019, Chap. 6.8](http://incompleteideas.net/book/RLbook2018.pdf)) using a set of hand-crafted features. In this implementation, we use the BCTS ([Thiery & Scherrer 2009](https://content.iospress.com/articles/icga-journal/icg32102)) feature set.

This implementation is _not_ supposed to be human-playable; it does not contain a proper GUI. Simple functions to print the current Tetris board and Tetromino to the console are provided for debugging purposes. These printing functions can also be used during learning and evaluation of an RL agent, but slow down performance considerably. 

This implementation uses the `numba` package for JIT compilation which results in relatively fast performance (compared to existing pure-Python implementations). The use of `numba`, however, required some weird design choices which unfortunately resulted in suboptimal code readability. If you have questions about the source code, please open up an issue.

Please cite the following publication if you use this code in your research:
```
@inproceedings{lichtenberg2019regularization,
  title={Regularization in directable environments with application to Tetris},
  author={Lichtenberg, Jan Malte and {\c{S}}im{\c{s}}ek, {\"O}zg{\"u}r},
  booktitle={International Conference on Machine Learning},
  pages={3953--3962},
  year={2019}
}
```

### Installation
Clone the repository and install the dependencies (numpy and numba) using the following terminal commands.
```zsh
git clone https://github.com/janmaltel/pytetris.git
cd pytetris
pip install requirements.txt
```

### Examples

The main methods of the `Tetris` class are:
* `get_after_states()` returns a list of the afterstates that result from all possible placements of the currently falling Tetromino given the current board configuration.
* `step(after_state)` takes an afterstate as input and returns the new state of the environment, the number of cleared lines (the reward), and whether the game is over (`True`) or not (`False`).
* `reset()` empties the current board.

An agent should have a `choose_action()` method that takes as input a list of afterstates and outputs one of the given afterstates.

The `agent/linear_agent.py` file contains the `LinearAgent` class whose policy is defined by a linear evaluation function of the BCTS features of each afterstate. This agent class can be used to evaluate a set of given policy weights. Note that this agent does _not_ learn policy weights from scratch (learning agents will be provided in a different repo soon).

The performance of a policy can be evaluated using the following code.

```python
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
    after_state = agent.choose_action(env=env)
    _, reward, game_over = env.step(after_state)
    cleared_lines += reward

print(f"The agent cleared {np.mean(cleared_lines)} lines.")
```
For debugging purposes, you can also print to the console the currently falling Tetromino using `env.print_current_tetromino()` and the current board configuration using `env.print_current_board()`.

The `examples/evaluate.py` file contains the `evaluate(env, agent, num_runs, verbose)` function that tests an agent for `num_runs` games. The `verbose` boolean determines printing behaviour. The function returns an array containing the cleared lines for each test game.
```python 
rewards = evaluate(env, agent, num_runs=5, verbose=True)
print(f"Average number of cleared lines was {np.mean(rewards)}.")
```















