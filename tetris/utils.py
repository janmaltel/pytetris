from numba import njit


@njit
def print_board_to_string(state):
    string = "\n"
    for row_ix in range(state.representation.shape[0]):
        # Start from top
        row_ix = state.num_rows - row_ix - 1
        string += "|"
        for col_ix in range(state.num_columns):
            if state.representation[row_ix, col_ix]:
                string += "██"
            else:
                string += "  "
        string += "|\n"
    return string


@njit
def print_tetromino(tetromino_index):
    if tetromino_index == 0:
        string = '''
██ ██ ██ ██'''
    elif tetromino_index == 1:
        string = '''
██ ██ 
██ ██'''
    elif tetromino_index == 2:
        string = '''
   ██ ██ 
██ ██'''
    elif tetromino_index == 3:
        string ='''
██ ██ 
   ██ ██'''
    elif tetromino_index == 4:
        string ='''
   ██
██ ██ ██'''
    elif tetromino_index == 5:
        string ='''
██ ██ ██
██'''
    elif tetromino_index == 6:
        string="""
██ ██ ██
      ██"""
    return string

