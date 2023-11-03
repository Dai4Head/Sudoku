#还是按peter novig算法来，单独添加函数检测颜色矩阵color_matrix
def solve_sudoku_rainbow(board, color_matrix):
    rows, cols = 'ABCDEFGHI', '123456789'
    digits = '123456789'
    box_size = 3

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
            if is_valid_assignment(values, color_matrix):
                return values
            else:
                print("Invalid assignment")
                return False
        n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
        return some(search(assign(values.copy(), s, d)) for d in values[s])


    def some(seq):
        for e in seq:
            if e:
                return e
        return False
    
    #添加颜色约束规则，排除白色这个颜色
    def is_valid_assignment(values, color_matrix):
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                cell = row + col
                color = color_matrix[i][j]
                if color != 'white' and values[cell] != '.':
                    for peer in peers[cell]:
                        peer_i, peer_j = ord(peer[0]) - ord('A'), int(peer[1]) - 1
                        peer_color = color_matrix[peer_i][peer_j]
                        if peer_color == color and values[peer] == values[cell] and values[peer] != '.':
                            print(f"Conflict: {cell} ({color}, {values[cell]}) and {peer} ({peer_color}, {values[peer]})")
                            return False
        return True

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
    box_units = [cross(rs, cs) for rs in chunks(rows, box_size) for cs in chunks(cols, box_size)]
    unitlist = row_units + col_units + box_units
    
    units = dict((s, [u for u in unitlist if s in u]) for s in squares)
    peers = dict((s, set(sum(units[s], []))-set([s])) for s in squares)
    return squares, units, peers
