class DancingLinks:
    def __init__(self, matrix):
        self.headers = [Node(i) for i in range(len(matrix[0]))]
        self.root = Node("root")
        self.root.right = self.headers[0]
        for i in range(len(self.headers)):
            self.headers[i].left = self.headers[i - 1]
            self.headers[i].right = self.headers[(i + 1) % len(self.headers)]
        self.headers[0].left = self.root
        self.root.left = self.headers[-1]

        for row in matrix:
            last = None
            row_header = None
            for j, val in enumerate(row):
                if val == 1:
                    node = Node(j, self.headers[j])
                    if last is not None:
                        last.right = node
                        node.left = last
                    else:
                        row_header = node
                    last = node
                    self.headers[j].up.down = node
                    self.headers[j].up = node
                    node.down = self.headers[j]
                    node.up = self.headers[j].up
                    self.headers[j].size += 1
            if row_header is not None:
                row_header.left = last
                last.right = row_header

    def search(self, solution):
        if self.root.right == self.root:
            return solution.copy()

        c = self.choose_column()
        self.cover(c)

        for r in self.iterate(c.down, "down"):
            solution.append(r.row)
            for j in self.iterate(r.right, "right"):
                self.cover(j.column)

            result = self.search(solution)
            if result is not None:
                return result

            solution.pop()
            for j in self.iterate(r.left, "left"):
                self.uncover(j.column)

        self.uncover(c)
        return None

    def cover(self, c):
        c.right.left = c.left
        c.left.right = c.right
        for i in self.iterate(c.down, "down"):
            for j in self.iterate(i.right, "right"):
                j.down.up = j.up
                j.up.down = j.down
                j.column.size -= 1

    def uncover(self, c):
        for i in self.iterate(c.up, "up"):
            for j in self.iterate(i.left, "left"):
                j.column.size += 1
                j.down.up = j
                j.up.down = j
        c.right.left = c
        c.left.right = c

    def choose_column(self):
        min_size = float("inf")
        min_column = None
        for c in self.iterate(self.root.right, "right"):
            if c.size < min_size:
                min_size = c.size
                min_column = c
        return min_column

    def iterate(self, start, direction):
        current = getattr(start, direction)
        while current != start:
            yield current
            current = getattr(current, direction)

class Node:
    def __init__(self, name, column=None):
        self.name = name
        self.size = 0
        self.column = column if column is not None else self
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.row = None

def build_matrix(sudoku):
    size = 16
    matrix = []
    for i in range(size):
        for j in range(size):
            num = sudoku[i][j]
            row = [0] * (size * size * 4)
            row[i * size + j] = 1
            row[size * size + i * size + num - 1] = 1
            row[2 * size * size + j * size + num - 1] = 1
            row[3 * size * size + (i // 4 * 4 + j // 4) * size + num - 1] = 1
            matrix.append(row)
    return matrix

def print_sudoku(matrix):
    for row in matrix:
        print(" ".join(str(num) for num in row))

def solve_sudoku(sudoku_matrix):
    matrix = build_matrix(sudoku_matrix)
    dlx = DancingLinks(matrix)
    solution = dlx.search([])
    if solution is not None:
        for row_index in solution:
            i, j = divmod(row_index // 16, 16)
            num = row_index % 16 + 1
            sudoku_matrix[i][j] = num
        return True
    return False

sudoku_matrix = [
[0, 2, 0, 3, 5, 0, 0, 0, 13, 0, 0, 0, 12, 0, 0, 11]
[0, 6, 0, 0, 0, 2, 12, 0, 0, 5, 15, 3, 4, 0, 14, 0]
[0, 0, 12, 14, 0, 4, 5, 5, 0, 0, 16, 0, 0, 0, 0, 0]
[0, 0, 0, 11, 0, 0, 0, 0, 0, 2, 0, 10, 0, 0, 6, 16]
[0, 14, 8, 0, 0, 0, 4, 0, 12, 0, 13, 0, 0, 0, 0, 1]
[0, 0, 5, 0, 3, 15, 0, 0, 9, 0, 0, 1, 6, 14, 0, 12]
[15, 0, 0, 0, 12, 5, 5, 1, 0, 14, 0, 0, 10, 0, 0, 0]
[0, 0, 1, 9, 8, 0, 0, 0, 0, 0, 4, 0, 16, 15, 11, 0]
[0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15]
[2, 0, 11, 13, 5, 14, 0, 0, 3, 0, 7, 0, 0, 0, 5, 10]
[0, 0, 0, 0, 4, 8, 0, 5, 0, 0, 0, 0, 0, 6, 13, 0]
[3, 10, 15, 8, 1, 0, 0, 0, 5, 16, 9, 0, 2, 11, 0, 0]
[14, 0, 6, 15, 0, 12, 0, 5, 0, 0, 1, 0, 0, 10, 2, 0]
[0, 0, 16, 0, 0, 1, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 3, 13, 14, 14, 5, 0, 7, 15, 0, 0, 8]
[14, 4, 14, 0, 15, 0, 10, 5, 5, 12, 0, 13, 0, 1, 0, 5]
]

if solve_sudoku(sudoku_matrix):
    print("Solved Sudoku:")
    print_sudoku(sudoku_matrix)
else:
    print("No solution found")
