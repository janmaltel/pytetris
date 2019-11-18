import numpy as np
from tetris.board import generate_terminal_board
from numba import jitclass, float64, int64


spec_agent = [  # Spec needed for numba's @jitclass
    ('policy_weights', float64[:]),
    ('num_features', int64),
]


@jitclass(spec_agent)
class LinearAgent:
    def __init__(self, policy_weights):
        self.policy_weights = policy_weights
        self.num_features = len(self.policy_weights)

    def choose_action(self, env):
        """
        Chooses among the utility-maximising action(s).
        """
        after_states = env.get_after_states()  # , current_board=
        num_children = len(after_states)
        if num_children == 0:
            return generate_terminal_board()

        action_features = np.zeros((num_children, self.num_features))
        for ix, after_state in enumerate(after_states):
            action_features[ix] = after_state.get_features(False)  # addRBF=
        utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
        max_utility_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_utility_indices)
        return after_states[move_index]

    # def choose_action(self, current_board, current_tetromino_index):
    #     """
    #     Chooses among the utility-maximising action(s).
    #     """
    #     after_states = current_tetromino_index.get_after_states(current_board)  # , current_board=
    #     num_children = len(after_states)
    #     if num_children == 0:
    #         # Terminal state!!
    #         return TerminalBoard()
    #
    #     action_features = np.zeros((num_children, self.num_features))
    #     for ix, after_state in enumerate(after_states):
    #         action_features[ix] = after_state.get_features(False)  # addRBF=
    #     utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
    #     max_indices = np.where(utilities == np.max(utilities))[0]
    #     move_index = np.random.choice(max_indices)
    #     return after_states[move_index]
