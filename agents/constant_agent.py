import numpy as np
from tetris.board import Board, TerminalBoard
import numba
from numba import jitclass, float64, int64


spec_agent = [  # Spec needed for numba's @jitclass
    ('policy_weights', float64[:]),
    ('feature_type', numba.types.string),
    ('num_features', int64),
]


@jitclass(spec_agent)
class ConstantAgent:
    def __init__(self, policy_weights):
        self.policy_weights = policy_weights
        self.num_features = len(self.policy_weights)

    def choose_action(self, current_board, current_tetromino):
        """
        Chooses among the utility-maximising action(s).
        """
        after_states = current_tetromino.get_after_states(current_board)  # , current_board=
        num_children = len(after_states)
        if num_children == 0:
            # Terminal state!!
            return TerminalBoard()

        action_features = np.zeros((num_children, self.num_features))
        for ix, after_state in enumerate(after_states):
            action_features[ix] = after_state.get_features_and_direct(self.feature_directors, False)  # direct_by=self.feature_directors  , order_by=None  , addRBF=False
        utilities = action_features.dot(np.ascontiguousarray(self.policy_weights))
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        return after_states[move_index]
