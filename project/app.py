from Sudoku_normal import Sudoku_Get as NormalGet
from Sudoku_normal import Cell as NormalCell
from Sudoku_mini import Sudoku_Get as MiniSGet
from Sudoku_mini import Cell as MiniCell
from Sudoku_giant import Sudoku_Get as GiantGet
from Sudoku_giant import Cell as GiantCell
from sudoku_solver import solve_sudoku
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for, jsonify
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



def process_sudoku_from_image_path(image_path, sudoku_type):
    #两个模型 一个能识别数字和字母 一个识别数字
    if(sudoku_type == "giant"):
            current_directory = os.path.dirname(os.path.abspath(__file__))
            new_model_path = os.path.join(current_directory, 'printed_model', 'sudoku_model.h5')
            model = load_model(new_model_path)
    else:
        # Load model
        model_path = os.path.join('.', 'drecv2_model')
        model = load_model(model_path)
    try:
        if sudoku_type == "normal":
            extractor = NormalGet(image_path)
            warped_image, color_warped_image = extractor.main(image_path)
            cells = NormalCell.extract_cells(warped_image)
            predictions = NormalCell.predict_cells(cells, model)
            sudoku_matrix = NormalCell.assemble_sudoku_matrix(predictions)
        elif sudoku_type == "mini":
            extractor = MiniSGet(image_path)
            warped_image, color_warped_image = extractor.main(image_path)
            cells = MiniCell.extract_cells(warped_image)
            predictions = MiniCell.predict_cells(cells, model)
            sudoku_matrix = MiniCell.assemble_sudoku_matrix(predictions)
        elif sudoku_type == "giant":
            extractor = GiantGet(image_path)
            warped_image, color_warped_image = extractor.main(image_path)
            cells = GiantCell.extract_cells(warped_image)
            predictions = GiantCell.predict_cells(cells, model)
            sudoku_matrix = GiantCell.assemble_sudoku_matrix(predictions)
        else:
            raise ValueError("Invalid sudoku type")
            
        original_sudoku = [row.copy() for row in sudoku_matrix]

        print("Original Sudoku:")
        for row in original_sudoku:
            print(row)
        if solve_sudoku(sudoku_matrix):
            print("Solved Sudoku:")
            for row in sudoku_matrix:
                print(row)

            extractor.overlay_solution(original_sudoku, sudoku_matrix, color_warped_image)
            
            _, buffer = cv2.imencode('.jpg', color_warped_image)
            io_buf = BytesIO(buffer)
            base64_str = base64.b64encode(io_buf.getvalue()).decode('utf-8')
            
            return base64_str
        else:
            print("Unable to solve Sudoku.")
            return None
    except Exception as e:
        print("Error in process_sudoku_from_image_path:")
        print(f"Message: {e}")
        import traceback
        traceback.print_exc()
        return None
    

@app.route('/', methods=['GET'])
def index():
    sudoker_img_url = url_for('static', filename='Sudoker.jpg')
    holdplace_img_url = url_for('static', filename='holdplace.png')
    return render_template('mainpage.html', sudoker_img_url=sudoker_img_url, holdplace_img_url=holdplace_img_url)


@app.route('/upload', methods=['POST'])
def upload_sudoku_image():
    sudoku_type = request.form.get('sudokuType')
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        solution_base64 = process_sudoku_from_image_path(filepath, sudoku_type)
        
        if solution_base64:
            return render_template('solution.html', solution_image=solution_base64)
        else:
            return "Unable to solve sudoku or process image", 500
    else:
        return "Invalid file type", 400
    
@app.route('/rotate_image', methods=['POST'])
def rotate_image():
    try:
        sudoku_type = request.form.get('sudokuType') # 获取数独类型
        #print("type"+sudoku_type)
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            solution_base64 = process_sudoku_from_image_path(filepath, sudoku_type) # 传递数独类型
            
            if solution_base64:
                return jsonify({'rotated_image': solution_base64})
            else:
                return "Unable to solve sudoku or process image", 500
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    board = data["board"]
    

    if solve_sudoku(board):
        return jsonify({"result": "success", "board": board})
    else:
        return jsonify({"result": "failure", "message": "Unable to solve sudoku"})
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)


