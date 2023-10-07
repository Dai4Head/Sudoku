from trytry import Sudoku_Get as SGet
from trytry import Cell
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import base64

# Flask setup
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
model_path = './drecv2_model'
model = load_model(model_path)

# Sudoku functions
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
                
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def process_sudoku_from_image_path(image_path):
    extractor = SGet(image_path)
    warped_image, color_warped_image = extractor.main(image_path)

    cells = Cell.extract_cells(warped_image)
    predictions = Cell.predict_cells(cells, model)
    sudoku_matrix = Cell.assemble_sudoku_matrix(predictions)
    
    original_sudoku = [row.copy() for row in sudoku_matrix]

    if solve_sudoku(sudoku_matrix):
        extractor.overlay_solution(original_sudoku, sudoku_matrix, color_warped_image)
        
        _, buffer = cv2.imencode('.jpg', color_warped_image)
        io_buf = BytesIO(buffer)
        base64_str = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        
        return base64_str
    else:
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_sudoku_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        solution_base64 = process_sudoku_from_image_path(filepath)
        
        if solution_base64:
            return render_template('solution.html', solution_image=solution_base64)
        else:
            return "Unable to solve sudoku or process image", 500
    else:
        return "Invalid file type", 400

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)