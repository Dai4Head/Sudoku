from trytry import Sudoku_Get as SGet
from trytry import Cell

import numpy as np

from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import cv2

# # 导入模型
# model_path = './models_first/sudoscan.h5'
# model = load_model(model_path)


# if __name__ == "__main__":
#     image_path = r'D:\NUS_1\PRS\project\example.jpg'
    
#     extractor = SGet(image_path)
#     warped_image = extractor.main(image_path)


# cells = Cell.extract_cells(warped_image)
# predictions = Cell.predict_cells(cells, model)
# sudoku_matrix = Cell.assemble_sudoku_matrix(predictions)

# 导入模型
model_path = './drecv2_model'
model = load_model(model_path)

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:  # Check row and column
            return False
    
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):  # Check 3x3 sub-grid
            if board[i][j] == num:
                return False
                
    return True  # num has not been used yet


def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:  # Find an empty spot
                for num in range(1, 10):  # Try all numbers
                    if is_valid(board, row, col, num):
                        board[row][col] = num  # Assign num
                        if solve_sudoku(board):  # Continue with next spot
                            return True  # Found a solution
                        board[row][col] = 0  # Undo & try again
                return False  # No valid number can be placed at this spot
    return True  # Sudoku is solved

if __name__ == "__main__":
    image_path = r'D:\NUS_1\PRS\project\example.jpg'
    
    extractor = SGet(image_path)
    warped_image, color_warped_image = extractor.main(image_path)


    cells = Cell.extract_cells(warped_image)
    predictions = Cell.predict_cells(cells, model)
    sudoku_matrix = Cell.assemble_sudoku_matrix(predictions)
    
if solve_sudoku(sudoku_matrix):
    for row in sudoku_matrix:
        print(row)
    # else:
    #     print("No soluti on exists.")
origin = np.array(predictions).reshape(9, 9)
solution = sudoku_matrix
extractor.overlay_solution(origin, solution, color_warped_image)    