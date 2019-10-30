import numpy as np
from tetris.board import generate_empty_board
from tetris import board, tetromino
import numba
from numba import jitclass, bool_, int64


# Needed for numba.@jitclass
specTetris = [
    ('num_columns', int64),
    ('num_rows', int64),
    ('feature_type', numba.types.string),
    ('num_features', int64),
    ('max_cleared_test_lines', int64),
    ('game_over', bool_),
    ('current_board', board.Board.class_type.instance_type),
    ('current_tetromino', tetromino.Tetromino.class_type.instance_type),
    ('cleared_lines', int64)
]


@jitclass(specTetris)
class Tetris:
    """
    Tetris for reinforcement learning applications.

    Tailored to use with a set of hand-crafted features such as "BCTS" (Thiery & Scherrer 2009)

    The BCTS feature names (and order) are
    ['rows_with_holes', 'column_transitions', 'holes', 'landing_height',
    'cumulative_wells', 'row_transitions', 'eroded', 'hole_depth']

    """
    def __init__(self,
                 num_columns,
                 num_rows,
                 feature_type="bcts",
                 ):
        if feature_type == "bcts":
            self.feature_type = feature_type
            self.num_features = 8
        else:
            raise AssertionError("Only the BCTS feature set (Thiery & Scherrer 2009) is currently implemented.")
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.current_tetromino = tetromino.Tetromino(self.feature_type, self.num_features, self.num_columns)
        self.current_board = generate_empty_board(self.num_rows, self.num_columns)
        self.cleared_lines = 0
        self.game_over = False

    def reset(self):
        self.current_board = generate_empty_board(self.num_rows, self.num_columns)
        self.current_board.calc_bcts_features()
        self.cleared_lines = 0
        self.game_over = False
        self.current_tetromino.next_tetromino()

    def make_step(self, after_state):
        self.game_over = after_state.terminal_state
        if not self.game_over:
            self.cleared_lines += after_state.n_cleared_lines
            self.current_board = after_state
            self.current_tetromino.next_tetromino()
        return [[self.current_board, self.current_tetromino], after_state.n_cleared_lines, self.game_over]

    def get_after_states(self):
        return self.current_tetromino.get_after_states(self.current_board)

