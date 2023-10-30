
def solve_sudoku(board):
    def solve_sudoku_16(board):
        def is_valid(board, row, col, num):
            for i in range(16):
                if board[row][i] == num or board[i][col] == num:  # Check row and column
                    return False
        
            start_row, start_col = 4 * (row // 4), 4 * (col // 4)
            for i in range(start_row, start_row + 4):
                for j in range(start_col, start_col + 4):  # Check 4x4 sub-grid
                    if board[i][j] == num:
                        return False
                    
            return True  # num has not been used yet

        for row in range(16):
            for col in range(16):
                if board[row][col] == 0:  # Find an empty spot
                    for num in range(1, 17):  # Try all numbers from 1 to 16
                        if is_valid(board, row, col, num):
                            board[row][col] = num  # Assign num
                            if solve_sudoku_16(board):  # Continue with next spot
                                return True  # Found a solution
                            board[row][col] = 0  # Undo & try again
                    return False  # No valid number can be placed at this spot
        return True  # Sudoku is solved


    size = len(board)
    if size == 16:
        return solve_sudoku_16(board)
    if size == 9:
        rows, cols = 'ABCDEFGHI', '123456789'
        digits = '123456789'
        box_size = 3
    elif size == 6:
        rows, cols = 'ABCDEF', '123456'
        digits = '123456'
        box_size = 2

    squares, units, peers = create_units(rows, cols, box_size)

    
    def parse_grid(grid, digits, squares, units, peers):
        values = dict((s, digits) for s in squares)
        for s, d in grid_values(grid, squares).items():
            if d in digits and not assign(values, s, d):
                return False
        return values

    def grid_values(grid, squares):
        chars = [c for c in grid if c in '123456789' or c in '0.']
        assert len(chars) == len(squares)
        return dict(zip(squares, chars))

    def assign(values, s, d):
        other_values = values[s].replace(d, '')
        if all(eliminate(values, s, d2) for d2 in other_values):
            return values
        else:
            return False

    def eliminate(values, s, d):
        if d not in values[s]:
            return values
        values[s] = values[s].replace(d, '')
        if len(values[s]) == 0:
            return False
        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(eliminate(values, s2, d2) for s2 in peers[s]):
                return False
        for u in units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False
            elif len(dplaces) == 1:
                if not assign(values, dplaces[0], d):
                    return False
        return values

    def search(values):
        if values is False:
            return False
        if all(len(values[s]) == 1 for s in squares):
            return values
        n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
        return some(search(assign(values.copy(), s, d)) for d in values[s])

    def some(seq):
        for e in seq:
            if e:
                return e
        return False

    board_str = ''.join(str(num) for row in board for num in row)
    board_str = board_str.replace('0', '.')
    values = search(parse_grid(board_str, digits, squares, units, peers))
    if values:
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                board[i][j] = int(values[row + col])
        return True
    else:
        return False

def create_units(rows, cols, box_size):
    def cross(A, B):
        return [a + b for a in A for b in B]

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    squares = cross(rows, cols)
    row_units = [cross(r, cols) for r in rows]
    col_units = [cross(rows, c) for c in cols]
    if len(rows) == 9:
        box_units = [cross(rs, cs) for rs in chunks(rows, box_size) for cs in chunks(cols, box_size)]
    else:  # for 6x6 sudoku
        box_units = [cross(rs, cs) for rs in chunks(rows, 2) for cs in chunks(cols, 3)]
    unitlist = row_units + col_units + box_units
    
    units = dict((s, [u for u in unitlist if s in u]) for s in squares)
    peers = dict((s, set(sum(units[s], []))-set([s])) for s in squares)
    return squares, units, peers

