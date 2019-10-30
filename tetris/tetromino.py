import numpy as np
import numba
from numba import jitclass, int64
from tetris import board

specTetromino = [
    ('current_tetromino', int64),
    ('tetromino_names', int64[:]),
    ('feature_type', numba.types.string),
    ('num_features', int64),
    ('num_columns', int64)
]


@jitclass(specTetromino)
class Tetromino:
    def __init__(self, feature_type, num_features, num_columns):
        assert(feature_type == "bcts")
        self.feature_type = feature_type
        self.num_features = num_features
        self.num_columns = num_columns
        self.tetromino_names = np.array([0,   # "straight",
                                         1,   # "square
                                         2,   # "snaker"
                                         3,   # "snakel"
                                         4,   # "t"
                                         5,   # "rcorner"
                                         6])  # "lcorner"
        self.current_tetromino = 0
        self.next_tetromino()

    def next_tetromino(self):
        self.current_tetromino = np.random.choice(self.tetromino_names)

    def copy_with_same_current_tetromino(self):
        current_tetromino = self.current_tetromino
        new_tetromino_object = Tetromino(self.feature_type, self.num_features, self.num_columns)
        new_tetromino_object.current_tetromino = current_tetromino
        return new_tetromino_object

    def get_after_states(self, current_state):
        """
        This version of get_after_states() reintroduces the possibility that
        a tetromino is placed such that it is legal AFTER lines are cleared.
        I.e., when playing on a 10x10 board, the initial placement could occupy
        a row in cell 11. This move would only be legal if it cleared at least
        one line (such that the highest occupied cell is in row 10.
        """

        after_states = []

        if self.current_tetromino == 0:
            # STRAIGHT
            # Vertical placements
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows):
                anchor_row = free_pos
                if not anchor_row + 4 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] += 4
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row:(anchor_row + 4), col_ix] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 4, 1, np.int64),
                                            np.array([1, 1, 1, 1], dtype=np.int64),
                                            1.5,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:  # has overlapping fields
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] += 4
                    # Add four lines to allow for overlapping fields (which could possibly be removed by clear_lines(), making it a legal move)
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row:(anchor_row + 4), col_ix] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 4, 1, np.int64),
                                            np.array([1, 1, 1, 1], dtype=np.int64),
                                            1.5,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True  # =
                                            )
                    if not new_state.terminal_state:
                        after_states.append(new_state)


                # Horizontal placements
                if col_ix < self.num_columns - 3:
                    anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 4)])
                    if not anchor_row + 1 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix:(col_ix + 4)] = anchor_row + 1
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row, col_ix:(col_ix + 4)] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([4], dtype=np.int64),
                                                0,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix:(col_ix + 4)] = anchor_row + 1
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row, col_ix:(col_ix + 4)] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([4], dtype=np.int64),
                                                0,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)
        elif self.current_tetromino == 1:
            # SQUARE
            # Horizontal placements
            max_col_index = self.num_columns - 1
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
                anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
                if not anchor_row + 2 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row:(anchor_row + 2), col_ix:(col_ix + 2)] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([2, 2], dtype=np.int64),
                                            0.5,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row:(anchor_row + 2), col_ix:(col_ix + 2)] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([2, 2], dtype=np.int64),
                                            0.5,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)
        elif self.current_tetromino == 2:
            # SNAKER
            # Vertical placements
            max_col_index = self.num_columns - 1
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
                anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, current_state.lowest_free_rows[col_ix + 1])
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 3
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 2
                    new_representation = current_state.representation.copy()
                    new_representation[(anchor_row + 1):(anchor_row + 3), col_ix] = 1
                    new_representation[anchor_row:(anchor_row + 2), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            # np.arange(col_ix, col_ix + 2, 1, np.int64),
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 3
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 2
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[(anchor_row + 1):(anchor_row + 3), col_ix] = 1
                    new_representation[anchor_row:(anchor_row + 2), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            # np.arange(col_ix, col_ix + 2, 1, np.int64),
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                # Horizontal placements
                # max_col_index = self.num_columns - 2
                # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
                if col_ix < self.num_columns - 2:
                    anchor_row = np.maximum(np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)]),
                                            current_state.lowest_free_rows[col_ix + 2] - 1)
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix] = anchor_row + 1
                        new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 2
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row, col_ix:(col_ix + 2)] = 1
                        new_representation[anchor_row + 1, (col_ix + 1):(col_ix + 3)] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([2], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix] = anchor_row + 1
                        new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 2
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row, col_ix:(col_ix + 2)] = 1
                        new_representation[anchor_row + 1, (col_ix + 1):(col_ix + 3)] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([2], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)

        elif self.current_tetromino == 3:
            # SNAKEL
            # Vertical placements
            max_col_index = self.num_columns - 1
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
                anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 1)
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 2
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 3
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row:(anchor_row + 2), col_ix] = 1
                    new_representation[(anchor_row + 1):(anchor_row + 3), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 2
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 3
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row:(anchor_row + 2), col_ix] = 1
                    new_representation[(anchor_row + 1):(anchor_row + 3), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                ## Horizontal placements
                if col_ix < self.num_columns - 2:
                    anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1,
                                            np.max(current_state.lowest_free_rows[(col_ix + 1):(col_ix + 3)]))
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
                        new_lowest_free_rows[col_ix + 2] = anchor_row + 1
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row, (col_ix + 1):(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix:(col_ix + 2)] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([2], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
                        new_lowest_free_rows[col_ix + 2] = anchor_row + 1
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row, (col_ix + 1):(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix:(col_ix + 2)] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([2], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)
        elif self.current_tetromino == 4:
            # T

            # Vertical placements.
            max_col_index = self.num_columns - 1
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
                # Single cell on left
                anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, current_state.lowest_free_rows[col_ix + 1])
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 2
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 3
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row + 1, col_ix] = 1
                    new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 2
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 3
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row + 1, col_ix] = 1
                    new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                # Single cell on right
                anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 1)
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 3
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 2
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
                    new_representation[anchor_row + 1, col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 3
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 2
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
                    new_representation[anchor_row + 1, col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                            np.array([1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                if col_ix < self.num_columns - 2:
                    # upside-down T
                    anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix] = anchor_row + 1
                        new_lowest_free_rows[col_ix + 2] = anchor_row + 1
                        new_lowest_free_rows[col_ix + 1] = anchor_row + 2
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix + 1] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix] = anchor_row + 1
                        new_lowest_free_rows[col_ix + 2] = anchor_row + 1
                        new_lowest_free_rows[col_ix + 1] = anchor_row + 2
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix + 1] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)

                    # T
                    anchor_row = np.maximum(current_state.lowest_free_rows[col_ix + 1],
                                            np.maximum(current_state.lowest_free_rows[col_ix],
                                                       current_state.lowest_free_rows[col_ix + 2]) - 1)
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row, col_ix + 1] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                                np.array([1, 3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row, col_ix + 1] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                                np.array([1, 3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)
        elif self.current_tetromino == 5:
            # RCorner

            # Vertical placements.
            max_col_index = self.num_columns - 1
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
                # Top-right corner
                anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 2, current_state.lowest_free_rows[col_ix + 1])
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row + 2, col_ix] = 1
                    new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 3, 1, np.int64),
                                            np.array([1, 1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row + 2, col_ix] = 1
                    new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 3, 1, np.int64),
                                            np.array([1, 1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                # Bottom-left corner
                anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 3
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 1
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
                    new_representation[anchor_row, col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                            np.array([2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix] = anchor_row + 3
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 1
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
                    new_representation[anchor_row, col_ix + 1] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                            np.array([2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                if col_ix < self.num_columns - 2:
                    # Bottom-right corner
                    anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 1
                        new_lowest_free_rows[col_ix + 2] = anchor_row + 2
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix + 2] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 1
                        new_lowest_free_rows[col_ix + 2] = anchor_row + 2
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix + 2] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)

                    # Top-left corner
                    anchor_row = np.maximum(current_state.lowest_free_rows[col_ix],
                                            np.max(current_state.lowest_free_rows[(col_ix + 1):(col_ix + 3)]) - 1)
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row, col_ix] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                                np.array([1, 3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row, col_ix] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                                np.array([1, 3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)

        elif self.current_tetromino == 6:
            # LCorner
            # Vertical placements. 'height' becomes 'width' :)
            max_col_index = self.num_columns - 1
            for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
                # Top-left corner
                anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 2)
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row + 2, col_ix + 1] = 1
                    new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 3, 1, np.int64),
                                            np.array([1, 1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row + 2, col_ix + 1] = 1
                    new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 3, 1, np.int64),
                                            np.array([1, 1, 2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                # Bottom-right corner
                anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
                if not anchor_row + 3 > current_state.num_rows:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 3
                    new_lowest_free_rows[col_ix] = anchor_row + 1
                    new_representation = current_state.representation.copy()
                    new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
                    new_representation[anchor_row, col_ix] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                            np.array([2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            False)
                    after_states.append(new_state)
                else:
                    new_lowest_free_rows = current_state.lowest_free_rows.copy()
                    new_lowest_free_rows[col_ix + 1] = anchor_row + 3
                    new_lowest_free_rows[col_ix] = anchor_row + 1
                    new_representation = np.vstack((current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                    new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
                    new_representation[anchor_row, col_ix] = 1
                    new_state = board.Board(new_representation,
                                            new_lowest_free_rows,
                                            np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                            np.array([2], dtype=np.int64),
                                            1,
                                            self.num_features,
                                            self.feature_type,
                                            False,
                                            True)
                    if not new_state.terminal_state:
                        after_states.append(new_state)

                if col_ix < self.num_columns - 2:
                    # Bottom-left corner (= 'hole' in top-right corner)
                    anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix] = anchor_row + 2
                        new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 1
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix] = anchor_row + 2
                        new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 1
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row + 1, col_ix] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 1, 1, np.int64),
                                                np.array([3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)

                    # Top-right corner
                    anchor_row = np.maximum(np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)]) - 1,
                                            current_state.lowest_free_rows[col_ix + 2])
                    if not anchor_row + 2 > current_state.num_rows:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
                        new_representation = current_state.representation.copy()
                        new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row, col_ix + 2] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                                np.array([1, 3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                False)
                        after_states.append(new_state)
                    else:
                        new_lowest_free_rows = current_state.lowest_free_rows.copy()
                        new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
                        new_representation = np.vstack(
                            (current_state.representation.copy(), np.zeros((4, self.num_columns), dtype=np.bool_)))
                        new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
                        new_representation[anchor_row, col_ix + 2] = 1
                        new_state = board.Board(new_representation,
                                                new_lowest_free_rows,
                                                np.arange(anchor_row, anchor_row + 2, 1, np.int64),
                                                np.array([1, 3], dtype=np.int64),
                                                0.5,
                                                self.num_features,
                                                self.feature_type,
                                                False,
                                                True)
                        if not new_state.terminal_state:
                            after_states.append(new_state)
        else:
            raise ValueError("wrong current tetromino!")

        return after_states

    # def get_after_states(self, current_board):
    #     # after_states = List()
    #     after_states = []
    #
    #     if self.current_tetromino == 0:
    #         # STRAIGHT
    #         # Vertical placements
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows):
    #             anchor_row = free_pos
    #             if not anchor_row + 4 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix] += 4
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row:(anchor_row + 4), col_ix] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         np.arange(anchor_row, anchor_row + 4, 1, np.int64),
    #                                         np.array([1, 1, 1, 1], dtype=np.int64),
    #                                         1.5,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #                 after_states.append(new_state)
    #
    #             # Horizontal placements
    #             if col_ix < self.num_columns - 3:
    #             # max_col_index = self.num_columns - 3
    #             # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #                 anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 4)])
    #                 if not anchor_row + 1 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix:(col_ix + 4)] = anchor_row + 1
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row, col_ix:(col_ix + 4)] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 4, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                             np.array([4], dtype=np.int64),
    #                                             0,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #                                             # current_board.col_transitions_per_col,
    #                                             # current_board.row_transitions_per_col,
    #                                             # current_board.array_of_rows_with_holes,
    #                                             # current_board.holes_per_col,
    #                                             # current_board.hole_depths_per_col,
    #                                             # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #     elif self.current_tetromino == 1:
    #         # SQUARE
    #         # Horizontal placements
    #         max_col_index = self.num_columns - 1
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 2)])
    #             if not anchor_row + 2 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row:(anchor_row + 2), col_ix:(col_ix + 2)] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                         np.array([2, 2], dtype=np.int64),
    #                                         0.5,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col
    #                 # )
    #                 after_states.append(new_state)
    #     elif self.current_tetromino == 2:
    #         # SNAKER
    #         # Vertical placements
    #         max_col_index = self.num_columns - 1
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             anchor_row = np.maximum(current_board.lowest_free_rows[col_ix] - 1, current_board.lowest_free_rows[col_ix + 1])
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix] = anchor_row + 3
    #                 new_lowest_free_rows[col_ix + 1] = anchor_row + 2
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[(anchor_row + 1):(anchor_row + 3), col_ix] = 1
    #                 new_representation[anchor_row:(anchor_row + 2), col_ix + 1] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                         np.array([1, 2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             # Horizontal placements
    #             # max_col_index = self.num_columns - 2
    #             # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             if col_ix < self.num_columns - 2:
    #                 anchor_row = np.maximum(np.max(current_board.lowest_free_rows[col_ix:(col_ix + 2)]),
    #                                         current_board.lowest_free_rows[col_ix + 2] - 1)
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix] = anchor_row + 1
    #                     new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 2
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row, col_ix:(col_ix + 2)] = 1
    #                     new_representation[anchor_row + 1, (col_ix + 1):(col_ix + 3)] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                             np.array([2], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #
    #                     # current_board.col_transitions_per_col,
    #                     # current_board.row_transitions_per_col,
    #                     # current_board.array_of_rows_with_holes,
    #                     # current_board.holes_per_col,
    #                     # current_board.hole_depths_per_col,
    #                     # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #     elif self.current_tetromino == 3:
    #         # SNAKEL
    #         # Vertical placements
    #         max_col_index = self.num_columns - 1
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             anchor_row = np.maximum(current_board.lowest_free_rows[col_ix], current_board.lowest_free_rows[col_ix + 1] - 1)
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix] = anchor_row + 2
    #                 new_lowest_free_rows[col_ix + 1] = anchor_row + 3
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row:(anchor_row + 2), col_ix] = 1
    #                 new_representation[(anchor_row + 1):(anchor_row + 3), col_ix + 1] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                         np.array([1, 2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             ## Horizontal placements
    #             # max_col_index = self.num_columns - 2
    #             # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             if col_ix < self.num_columns - 2:
    #                 anchor_row = np.maximum(current_board.lowest_free_rows[col_ix] - 1,
    #                                         np.max(current_board.lowest_free_rows[(col_ix + 1):(col_ix + 3)]))
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
    #                     new_lowest_free_rows[col_ix + 2] = anchor_row + 1
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row, (col_ix + 1):(col_ix + 3)] = 1
    #                     new_representation[anchor_row + 1, col_ix:(col_ix + 2)] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                             np.array([2], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #
    #                     # current_board.col_transitions_per_col,
    #                     # current_board.row_transitions_per_col,
    #                     # current_board.array_of_rows_with_holes,
    #                     # current_board.holes_per_col,
    #                     # current_board.hole_depths_per_col,
    #                     # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #     elif self.current_tetromino == 4:
    #         # T
    #
    #         # Vertical placements.
    #         max_col_index = self.num_columns - 1
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             # Single cell on left
    #             anchor_row = np.maximum(current_board.lowest_free_rows[col_ix] - 1, current_board.lowest_free_rows[col_ix + 1])
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix] = anchor_row + 2
    #                 new_lowest_free_rows[col_ix + 1] = anchor_row + 3
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row + 1, col_ix] = 1
    #                 new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                         np.array([1, 2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             # Single cell on right
    #             anchor_row = np.maximum(current_board.lowest_free_rows[col_ix], current_board.lowest_free_rows[col_ix + 1] - 1)
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix] = anchor_row + 3
    #                 new_lowest_free_rows[col_ix + 1] = anchor_row + 2
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
    #                 new_representation[anchor_row + 1, col_ix + 1] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                         np.array([1, 2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             if col_ix < self.num_columns - 2:
    #                 # Horizontal placements
    #                 # max_col_index = self.num_columns - 2
    #                 # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #                 # upside-down T
    #                 anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 3)])
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     # new_lowest_free_rows[[col_ix, col_ix + 2]] = anchor_row + 1
    #                     new_lowest_free_rows[col_ix] = anchor_row + 1
    #                     new_lowest_free_rows[col_ix + 2] = anchor_row + 1
    #                     new_lowest_free_rows[col_ix + 1] = anchor_row + 2
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
    #                     new_representation[anchor_row + 1, col_ix + 1] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                             np.array([3], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #
    #                     # current_board.col_transitions_per_col,
    #                     # current_board.row_transitions_per_col,
    #                     # current_board.array_of_rows_with_holes,
    #                     # current_board.holes_per_col,
    #                     # current_board.hole_depths_per_col,
    #                     # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #
    #                 # T
    #                 # anchor_row = np.maximum(current_board.lowest_free_rows[col_ix + 1], np.max(current_board.lowest_free_rows[[col_ix, col_ix + 2]]) - 1)
    #                 anchor_row = np.maximum(current_board.lowest_free_rows[col_ix + 1],
    #                                         np.maximum(current_board.lowest_free_rows[col_ix],
    #                                                    current_board.lowest_free_rows[col_ix + 2]) - 1)
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
    #                     new_representation[anchor_row, col_ix + 1] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                             np.array([1, 3], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #
    #                     # current_board.col_transitions_per_col,
    #                     # current_board.row_transitions_per_col,
    #                     # current_board.array_of_rows_with_holes,
    #                     # current_board.holes_per_col,
    #                     # current_board.hole_depths_per_col,
    #                     # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #     elif self.current_tetromino == 5:
    #         # RCorner
    #
    #         # Vertical placements.
    #         max_col_index = self.num_columns - 1
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             # Top-right corner
    #             anchor_row = np.maximum(current_board.lowest_free_rows[col_ix] - 2, current_board.lowest_free_rows[col_ix + 1])
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row + 2, col_ix] = 1
    #                 new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 3, 1, np.int64),
    #                                         np.array([1, 1, 2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             # Bottom-left corner
    #             anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 2)])
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix] = anchor_row + 3
    #                 new_lowest_free_rows[col_ix + 1] = anchor_row + 1
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
    #                 new_representation[anchor_row, col_ix + 1] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                         np.array([2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             if col_ix < self.num_columns - 2:
    #                 # Horizontal placements
    #                 # max_col_index = self.num_columns - 2
    #                 # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #                 # Bottom-right corner
    #                 anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 3)])
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 1
    #                     new_lowest_free_rows[col_ix + 2] = anchor_row + 2
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
    #                     new_representation[anchor_row + 1, col_ix + 2] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                             np.array([3], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #                     after_states.append(new_state)
    #
    #                 # Top-left corner
    #                 anchor_row = np.maximum(current_board.lowest_free_rows[col_ix],
    #                                         np.max(current_board.lowest_free_rows[(col_ix + 1):(col_ix + 3)]) - 1)
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
    #                     new_representation[anchor_row, col_ix] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                             np.array([1, 3], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #                     after_states.append(new_state)
    #
    #     elif self.current_tetromino == 6:
    #         # LCorner
    #         # Vertical placements. 'height' becomes 'width' :)
    #         max_col_index = self.num_columns - 1
    #         for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #             # Top-left corner
    #             anchor_row = np.maximum(current_board.lowest_free_rows[col_ix], current_board.lowest_free_rows[col_ix + 1] - 2)
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row + 2, col_ix + 1] = 1
    #                 new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 3, 1, np.int64),
    #                                         np.array([1, 1, 2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             # Bottom-right corner
    #             anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 2)])
    #             if not anchor_row + 3 > current_board.num_rows:
    #                 new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                 new_lowest_free_rows[col_ix + 1] = anchor_row + 3
    #                 new_lowest_free_rows[col_ix] = anchor_row + 1
    #                 new_representation = current_board.representation.copy()
    #                 new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
    #                 new_representation[anchor_row, col_ix] = 1
    #                 new_state = state.Board(new_representation,
    #                                         new_lowest_free_rows,
    #                                         # np.arange(col_ix, col_ix + 2, 1, np.int64),
    #                                         np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                         np.array([2], dtype=np.int64),
    #                                         1,
    #                                         self.num_features,
    #                                         self.feature_type,
    #                                         False)
    #
    #                 # current_board.col_transitions_per_col,
    #                 # current_board.row_transitions_per_col,
    #                 # current_board.array_of_rows_with_holes,
    #                 # current_board.holes_per_col,
    #                 # current_board.hole_depths_per_col,
    #                 # current_board.cumulative_wells_per_col)
    #                 after_states.append(new_state)
    #
    #             if col_ix < self.num_columns - 2:
    #
    #                 # max_col_index = self.num_columns - 2
    #                 # for col_ix, free_pos in enumerate(current_board.lowest_free_rows[:max_col_index]):
    #                 # Bottom-left corner (= 'hole' in top-right corner)
    #                 anchor_row = np.max(current_board.lowest_free_rows[col_ix:(col_ix + 3)])
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix] = anchor_row + 2
    #                     new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 1
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
    #                     new_representation[anchor_row + 1, col_ix] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 1, 1, np.int64),
    #                                             np.array([3], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #                     # current_board.col_transitions_per_col,
    #                     # current_board.row_transitions_per_col,
    #                     # current_board.array_of_rows_with_holes,
    #                     # current_board.holes_per_col,
    #                     # current_board.hole_depths_per_col,
    #                     # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #
    #                 # Top-right corner
    #                 anchor_row = np.maximum(np.max(current_board.lowest_free_rows[col_ix:(col_ix + 2)]) - 1,
    #                                         current_board.lowest_free_rows[col_ix + 2])
    #                 if not anchor_row + 2 > current_board.num_rows:
    #                     new_lowest_free_rows = current_board.lowest_free_rows.copy()
    #                     new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
    #                     new_representation = current_board.representation.copy()
    #                     new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
    #                     new_representation[anchor_row, col_ix + 2] = 1
    #                     new_state = state.Board(new_representation,
    #                                             new_lowest_free_rows,
    #                                             # np.arange(col_ix, col_ix + 3, 1, np.int64),
    #                                             np.arange(anchor_row, anchor_row + 2, 1, np.int64),
    #                                             np.array([1, 3], dtype=np.int64),
    #                                             0.5,
    #                                             self.num_features,
    #                                             self.feature_type,
    #                                             False)
    #
    #                     # current_board.col_transitions_per_col,
    #                     # current_board.row_transitions_per_col,
    #                     # current_board.array_of_rows_with_holes,
    #                     # current_board.holes_per_col,
    #                     # current_board.hole_depths_per_col,
    #                     # current_board.cumulative_wells_per_col)
    #                     after_states.append(new_state)
    #     else:
    #         raise ValueError("wrong current tetromino!")
    #
    #     return after_states

