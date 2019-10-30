import numpy as np
import numba
from numba import njit, jitclass, float64, bool_, int64


spec = [
    ('representation', bool_[:, :]),
    ('lowest_free_rows', int64[:]),
    ('changed_lines', int64[:]),
    ('pieces_per_changed_row', int64[:]),
    ('landing_height_bonus', float64),
    ('num_features', int64),
    ('feature_type', numba.types.string),
    ('num_rows', int64),
    ('num_columns', int64),
    ('n_cleared_lines', int64),
    ('anchor_row', int64),
    ('cleared_rows_relative_to_anchor', bool_[:]),
    ('features_are_calculated', bool_),
    ('features', float64[:]),
    ('terminal_state', bool_)
]


@jitclass(spec)
class Board:
    def __init__(self,
                 representation,
                 lowest_free_rows,
                 changed_lines,  #=np.array([0], dtype=np.int64),
                 pieces_per_changed_row,  #=np.array([0], dtype=np.int64),
                 landing_height_bonus,  # =0.0,
                 num_features,  #=8,
                 feature_type,  #="bcts",
                 terminal_state,  # this is useful to generate a "terminal state"
                 has_overlapping_fields=False
                 ):
        self.terminal_state = terminal_state

        if not terminal_state:
            self.representation = representation
            self.lowest_free_rows = lowest_free_rows
            self.num_rows, self.num_columns = representation.shape
            if has_overlapping_fields:
                self.num_rows -= 4  # representation input in this case has e.g., 14 rows for a 10x10 board... => put it back to 10
            self.pieces_per_changed_row = pieces_per_changed_row
            self.landing_height_bonus = landing_height_bonus
            self.num_features = num_features
            self.feature_type = feature_type  # "bcts"
            self.n_cleared_lines = 0  # Gets updated in self.clear_lines()
            self.anchor_row = changed_lines[0]
            self.cleared_rows_relative_to_anchor = self.clear_lines(changed_lines)
            self.features_are_calculated = False
            self.features = np.zeros(self.num_features, dtype=np.float64)
            if has_overlapping_fields:
                self.terminal_state = check_terminal(self.representation, self.num_rows)
                self.representation = self.representation[:self.num_rows, ]

    def get_features_pure(self, addRBFandIntercept=False):  #, order_by=None, standardize_by=None, addRBF=False
        if not self.features_are_calculated:
            if self.feature_type == "bcts":
                self.calc_bcts_features()
                self.features_are_calculated = True
            else:
                raise ValueError("Feature type must be either bcts or standardized_bcts or simple or super_simple")
        features = self.features
        if addRBFandIntercept:
            features = np.concatenate((
                np.array([1.]),
                features,
                np.exp(-(np.mean(self.lowest_free_rows) - np.arange(5) * self.num_rows / 4) ** 2 / (2 * (self.num_rows / 5) ** 2))
                ))
        return features

    def get_features_order_and_direct(self, direct_by, order_by, addRBF=False):
        if not self.features_are_calculated:
            if self.feature_type == "bcts":
                self.calc_bcts_features()
                self.features_are_calculated = True
            else:
                raise ValueError("Feature type must be either bcts or standardized_bcts or simple or super_simple")
        out = self.features * direct_by  # .copy()
        out = out[order_by]
        # if standardize_by is not None:
        #     features = features / standardize_by
        if addRBF:
            out = np.concatenate((
                out,
                np.exp(-(np.mean(self.lowest_free_rows) - np.arange(5) * self.num_rows / 4) ** 2 / (2 * (self.num_rows / 5) ** 2))
                ))
        return out

    def get_features_and_direct(self, direct_by, addRBF=False):
        if not self.features_are_calculated:
            if self.feature_type == "bcts":
                self.calc_bcts_features()
                self.features_are_calculated = True
            else:
                raise ValueError("Feature type must be either bcts or standardized_bcts or simple or super_simple")
        out = self.features * direct_by  # .copy()
        if addRBF:
            out = np.concatenate((
                out,
                np.exp(-(np.mean(self.lowest_free_rows) - np.arange(5) * self.num_rows / 4) ** 2 / (2 * (self.num_rows / 5) ** 2))
                ))
        return out

    def clear_lines(self, changed_lines):
        num_columns = self.num_columns
        row_sums = np.sum(self.representation[changed_lines, :], axis=1)
        is_full = (row_sums == num_columns)
        full_lines = np.where(is_full)[0]
        n_cleared_lines = len(full_lines)
        if n_cleared_lines > 0:
            # print(self)
            representation = self.representation
            lowest_free_rows = self.lowest_free_rows
            lines_to_clear = changed_lines[full_lines].astype(np.int64)
            mask_keep = np.ones(len(representation), dtype=np.bool_)
            mask_keep[lines_to_clear] = False
            new_cols = np.zeros((n_cleared_lines, num_columns), dtype=np.bool_)
            representation = np.vstack((representation[mask_keep], new_cols))
            for col_ix in range(num_columns):  # col_ix = 0
                old_lowest_free_row = lowest_free_rows[col_ix]
                if old_lowest_free_row > lines_to_clear[-1] + 1:
                    lowest_free_rows[col_ix] -= n_cleared_lines
                else:
                    lowest = 0
                    for row_ix in range(old_lowest_free_row - n_cleared_lines - 1, -1, -1):
                        if representation[row_ix, col_ix] == 1:
                            lowest = row_ix + 1
                            break
                    lowest_free_rows[col_ix] = lowest
            self.lowest_free_rows = lowest_free_rows
            self.representation = representation

        self.n_cleared_lines = n_cleared_lines
        return is_full

    def calc_bcts_features(self):
        rows_with_holes_set = {1000}
        representation = self.representation
        num_rows, num_columns = representation.shape
        lowest_free_rows = self.lowest_free_rows
        col_transitions = 0
        row_transitions = 0
        holes = 0
        hole_depths = 0
        cumulative_wells = 0
        # row_transitions = 0
        for col_ix, lowest_free_row in enumerate(lowest_free_rows):
            # There is at least one column_transition from the highest full cell (or the bottom which is assumed to be full) to "the top".
            col_transitions += 1
            if col_ix == 0:
                local_well_streak = 0
                if lowest_free_row > 0:
                    col = representation[:lowest_free_row, col_ix]
                    cell_below = 1

                    # Needed for hole_depth
                    # TODO: Optimize... only count the first time when an actual hole is found
                    number_of_full_cells_above = numba_sum_int(col)

                    for row_ix, cell in enumerate(col):
                        if cell == 0:
                            # Holes
                            holes += 1
                            rows_with_holes_set.add(row_ix)
                            hole_depths += number_of_full_cells_above

                            # Column transitions
                            if cell_below:
                                col_transitions += 1

                            # Row transitions and wells
                            # Because col_ix == 0, all left_cells are 1
                            # row_transitions += 1
                            row_transitions += 1
                            if representation[row_ix, col_ix + 1]:  # if cell to the right is full
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0

                        else:  # cell is 1!
                            local_well_streak = 0

                            # Keep track of full cells above for hole_depth-feature
                            number_of_full_cells_above -= 1

                            # Column transitions
                            if not cell_below:
                                col_transitions += 1

                        # Define 'cell_below' for next (higher!) cell.
                        cell_below = cell

                # Check wells until lowest_free_row_right
                # Check transitions until lowest_free_row_left
                max_well_possibility = lowest_free_rows[col_ix + 1]
                if max_well_possibility > lowest_free_row:
                    for row_ix in range(lowest_free_row, max_well_possibility):
                        if representation[row_ix, col_ix + 1]:  # if cell to the right is full
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0
                # # Add row transitions for each empty cell above lowest_free_row
                row_transitions += (num_rows - lowest_free_row)

            elif col_ix == num_columns - 1:
                local_well_streak = 0
                if lowest_free_row > 0:
                    col = representation[:lowest_free_row, col_ix]
                    cell_below = 1

                    # Needed for hole_depth
                    number_of_full_cells_above = numba_sum_int(col)

                    for row_ix, cell in enumerate(col):
                        if cell == 0:
                            # Holes
                            holes += 1
                            rows_with_holes_set.add(row_ix)
                            hole_depths += number_of_full_cells_above

                            # Column transitions
                            if cell_below:
                                col_transitions += 1

                            # Wells and row transitions
                            # Because this is the last column (the right border is "full") and cell == 0:
                            row_transitions += 1
                            if representation[row_ix, col_ix - 1]:  # if cell to the left is full
                                row_transitions += 1
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0

                        else:  # cell is 1!
                            local_well_streak = 0

                            # Keep track of full cells above for hole_depth-feature
                            number_of_full_cells_above -= 1

                            # Column transitions
                            if not cell_below:
                                col_transitions += 1

                            # Row transitions
                            cell_left = representation[row_ix, col_ix - 1]
                            if not cell_left:
                                row_transitions += 1

                        # Define 'cell_below' for next (higher!) cell.
                        cell_below = cell

                # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
                # Check transitions until lowest_free_row_left
                max_well_possibility = lowest_free_rows[col_ix - 1]
                if max_well_possibility > lowest_free_row:
                    for row_ix in range(lowest_free_row, max_well_possibility):
                        if representation[row_ix, col_ix - 1]:  # if cell to the left is full
                            row_transitions += 1
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0
                # # Add row transitions from last column to border
                row_transitions += (num_rows - lowest_free_row)
            else:
                local_well_streak = 0
                if lowest_free_row > 0:
                    col = representation[:lowest_free_row, col_ix]
                    cell_below = 1

                    # Needed for hole_depth
                    number_of_full_cells_above = numba_sum_int(col)

                    for row_ix, cell in enumerate(col):
                        if cell == 0:
                            # Holes
                            holes += 1
                            rows_with_holes_set.add(row_ix)
                            hole_depths += number_of_full_cells_above

                            # Column transitions
                            if cell_below:
                                col_transitions += 1

                            # Wells and row transitions
                            cell_left = representation[row_ix, col_ix - 1]
                            if cell_left:
                                row_transitions += 1
                                cell_right = representation[row_ix, col_ix + 1]
                                if cell_right:
                                    local_well_streak += 1
                                    cumulative_wells += local_well_streak
                                else:
                                    local_well_streak = 0
                            else:
                                local_well_streak = 0

                        else:  # cell is 1!
                            local_well_streak = 0
                            # Keep track of full cells above for hole_depth-feature
                            number_of_full_cells_above -= 1

                            # Column transitions
                            if not cell_below:
                                col_transitions += 1

                            # Row transitions
                            cell_left = representation[row_ix, col_ix - 1]
                            if not cell_left:
                                row_transitions += 1

                        # Define 'cell_below' for next (higher!) cell.
                        cell_below = cell
                # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
                # Check transitions until lowest_free_row_left
                lowest_free_row_left = lowest_free_rows[col_ix - 1]
                lowest_free_row_right = lowest_free_rows[col_ix + 1]
                max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)

                # Weird case distinction because max_well_possibility always "includes" lowest_free_row_left
                #  but lowest_free_row_left can be higher than max_well_possibility. Don't want to double count.
                if max_well_possibility > lowest_free_row:
                    for row_ix in range(lowest_free_row, max_well_possibility):
                        cell_left = representation[row_ix, col_ix - 1]
                        if cell_left:
                            row_transitions += 1
                            cell_right = representation[row_ix, col_ix + 1]
                            if cell_right:
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0
                        else:
                            local_well_streak = 0
                    if lowest_free_row_left > max_well_possibility:
                        for row_ix in range(max_well_possibility, lowest_free_row_left):
                            cell_left = representation[row_ix, col_ix - 1]
                            if cell_left:
                                row_transitions += 1
                elif lowest_free_row_left > lowest_free_row:
                    for row_ix in range(lowest_free_row, lowest_free_row_left):
                        cell_left = representation[row_ix, col_ix - 1]
                        if cell_left:
                            row_transitions += 1

        rows_with_holes_set.remove(1000)
        rows_with_holes = len(rows_with_holes_set)
        eroded_pieces = numba_sum_int(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
        # n_cleared_lines = numba_sum_int(self.cleared_rows_relative_to_anchor)
        eroded_piece_cells = eroded_pieces * self.n_cleared_lines
        landing_height = self.anchor_row + self.landing_height_bonus
        self.features = np.array([rows_with_holes, col_transitions, holes, landing_height,
                                  cumulative_wells, row_transitions, eroded_piece_cells, hole_depths])


def generate_empty_board(num_rows, num_columns):
    return Board(np.zeros((num_rows, num_columns), dtype=np.bool_),  # representation=
                 np.zeros(num_columns, dtype=np.int64),  # lowest_free_rows=
                 np.array([0], dtype=np.int64),  # changed_lines=
                 np.array([0], dtype=np.int64),  # pieces_per_changed_row=
                 0.0,  # landing_height_bonus=
                 8,  # num_features=
                 "bcts",  # feature_type=
                 False,  # terminal_state=
                 False  # has_overlapping_fields=
                 )

specTerm = [
    ('terminal_state', bool_),
]


@jitclass(specTerm)
class TerminalBoard:
    def __init__(self):
        self.terminal_state = True


@njit(cache=False)
def check_terminal(representation, num_rows):
    is_terminal = False
    for ix in representation[num_rows]:
        if ix:
            is_terminal = True
            break
    return is_terminal


@njit(fastmath=True, cache=False)
def numba_sum_int(int_arr):
    acc = 0
    for i in int_arr:
        acc += i
    return acc


