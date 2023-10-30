import cv2
import numpy as np

# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 预处理部分
class Sudoku_Get:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # print(self.image.shape)
    def Grayscale_get(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
    def Filter_apply(self):# 滤波器选择
        # self.filtered1 =cv2.blur(self.gray,(3,3))
        # gaussian
        # self.filtered1 = cv2.GaussianBlur(self.gray, (5,5), 0)
        # # median
        # self.filtered = cv2.medianBlur(self.gray, 9)
        # bilateral:
        # self.filtered = cv2.bilateralFilter(self.gray, 7, 75, 75) 
        # self.filtered = self.gray
        
        # self.filtered1 = cv2.GaussianBlur(self.gray, (15,15), 0)
        # self.filtered = cv2.bilateralFilter(self.filtered1, -1, 11, 5) 
        self.filtered = self.gray
        # cv2.namedWindow("filtered Image",cv2.WINDOW_NORMAL)       
        # cv2.imshow("filtered Image", self.filtered)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.filtered

    def filtered_binarize(self): #二值化
        # _, self.binary = cv2.threshold(self.filtered,90,255,cv2.THRESH_BINARY_INV)#负片简单阈值
        # _, self.binary= cv2.threshold(self.filtered,80,255,cv2.THRESH_TRUNC)#简单阈值
        self.binary_1= cv2.adaptiveThreshold(self.filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 11, 4)#自适应阈值
        # _, self.binary = cv2.threshold(self.filtered,0,255,\
        #                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)#大津阈值
        # opening
        self.kernel_o = np.ones((11,11), np.uint8)
        self.opening = cv2.morphologyEx(self.binary_1, cv2.MORPH_OPEN, self.kernel_o)
        # closing
        self.kernel_c = np.ones((9,9), np.uint8)
        self.closing = cv2.morphologyEx(self.opening, cv2.MORPH_CLOSE, self.kernel_c)
        
        # 根据底色调整是否反相，如果是白底黑字则执行以下这句代码
        # self.binary = np.subtract(255, self.binary_1)
        self.binary = self.binary_1
        return self.binary
###########################################################
    def find_edges(self):
        self.edges = cv2.Canny(self.binary, 50, 150, apertureSize=3)
        return self.edges

    def find_largest_contour(self):
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        self.largest_contour = contours[0]
        return self.largest_contour

    def draw_contours(self):
        contour_img = self.image.copy()
        cv2.drawContours(contour_img, [self.largest_contour], 0, (0, 255, 0), 2)


    def find_corners(self):
        epsilon = 0.05 * cv2.arcLength(self.largest_contour, True)
        corners = cv2.approxPolyDP(self.largest_contour, epsilon, True)
        self.corners = np.array([corner[0] for corner in corners], dtype='float32')
        return self.corners

    def warp_perspective(self):
        side = max([
            self.image.shape[1],
            self.image.shape[0]
        ])
        dst_points = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(self.corners, dst_points)
        self.warped = cv2.warpPerspective(self.image, matrix, (side, side))
        return self.warped


    def find_intersection_points(self):
        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, 200)
        if lines is not None:
            lines = [l[0] for l in lines]

        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                rho1, theta1 = line1
                rho2, theta2 = line2
                A = np.array([
                    [np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]
                ])
                b = np.array([[rho1], [rho2]])
                x0, y0 = np.linalg.solve(A, b)
                intersections.append([int(x0), int(y0)])

        intersections = sorted(intersections, key=lambda x: (x[1], x[0]))
        return intersections

    def transform_and_combine(self, intersections):
        square_side = 50
        combined_image = np.zeros((16 * square_side, 16 * square_side), dtype=np.uint8)
        for i in range(15):  # Change to 31
            for j in range(15):  # Change to 31
                src = np.array([
                    intersections[i*16 + j],
                    intersections[i*16 + j + 1],
                    intersections[(i+1)*16 + j + 1],
                    intersections[(i+1)*16 + j]
                ], dtype='float32')

                dst = np.array([
                    [j * square_side, i * square_side],
                    [(j+1) * square_side, i * square_side],
                    [(j+1) * square_side, (i+1) * square_side],
                    [j * square_side, (i+1) * square_side]
                ], dtype='float32')

                matrix = cv2.getPerspectiveTransform(src, dst)
                warp = cv2.warpPerspective(self.image, matrix, (16 * square_side, 16 * square_side))
                combined_image[i*square_side:(i+1)*square_side, j*square_side:(j+1)*square_side] = \
                    warp[i*square_side:(i+1)*square_side, j*square_side:(j+1)*square_side]
        return combined_image


    def process_sudoku_image(self):
        self.Grayscale_get()
        self.Filter_apply()
        self.filtered_binarize()
        self.find_edges()
        self.find_largest_contour()
        self.find_corners()
        self.warp_perspective()

####################################################

    def find_contours(self):# 找轮廓
        contours, _ = cv2.findContours(self.closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True) # 根据面积从大到小排序
        self.polygon = contours[0] # 第一大的轮廓通常是数独网格
        
        # 这里可能需要加一个检测过程，仅仅找第二大的方框不是很严谨
        return self.polygon

    def perspective_transform(self):
        # 获取轮廓的四个顶点
        corners = self._four_corners(self.polygon)
    
        # 对角点分类，防止图片上下左右颠倒
        top_left = min(corners, key=lambda pt: (pt[1], pt[0]))
        top_right = max(corners, key=lambda pt: pt[0] - pt[1])
        bottom_right = max(corners, key=lambda pt: (pt[1], pt[0]))
        bottom_left = min(corners, key=lambda pt: pt[0] - pt[1])

        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        side = max([
            self._distance(bottom_right, top_right),
            self._distance(top_left, bottom_left),
            self._distance(bottom_right, bottom_left),
            self._distance(top_left, top_right)
          ])
 
        dst = np.array([
            [0, 0],
            [side - 1, 0],
            [side - 1, side - 1],
            [0, side - 1]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(src, dst)
        self.warped = cv2.warpPerspective(self.binary, matrix, (int(side), int(side)))
        return self.warped

    def _four_corners(self, contour):
        # 获取轮廓的四个顶点
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=lambda x: x[1])
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=lambda x: x[1])
        bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=lambda x: x[1])
        top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=lambda x: x[1])
        return contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]

    def _distance(self, pt1, pt2):
        # 计算两点之间的距离
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    #对RGB图同样进行变换
    def perspective_transform_color(self, img):# 这个是对彩图做相同截取，以便输出图像
        # 获取轮廓的四个顶点
        corners = self._four_corners(self.polygon)
    
        top_left = min(corners, key=lambda pt: (pt[1], pt[0]))
        top_right = max(corners, key=lambda pt: pt[0] - pt[1])
        bottom_right = max(corners, key=lambda pt: (pt[1], pt[0]))
        bottom_left = min(corners, key=lambda pt: pt[0] - pt[1])
        
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        
        side = max([
            self._distance(bottom_right, top_right),
            self._distance(top_left, bottom_left),
            self._distance(bottom_right, bottom_left),
            self._distance(top_left, top_right)
        ])
        
        dst = np.array([
            [0, 0],
            [side - 1, 0],
            [side - 1, side - 1],
            [0, side - 1]
        ], dtype='float32')
        
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_color = cv2.warpPerspective(img, matrix, (int(side), int(side)))
        return warped_color


    # def main(self, image_path):
    #     extractor = Sudoku_Get(image_path)
    #     extractor.Grayscale_get()
    #     extractor.Filter_apply()
    #     extractor.filtered_binarize()

    #     extractor.find_contours()
        
    #     warped_image = extractor.perspective_transform()
    
    #     cv2.namedWindow("Warped Sudoku Grid",cv2.WINDOW_NORMAL)
    #     cv2.imshow("Warped Sudoku Grid", warped_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return warped_image
    def overlay_solution(self, original_sudoku, solution, original_img):
        cell_width = original_img.shape[1] // 16
        cell_height = original_img.shape[0] // 16
        
        for y in range(16):
            for x in range(16):
                if original_sudoku[y][x] == 0 and solution[y][x] != 0:
                    number = solution[y, x]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (0, 255, 0)  # Green
                    
                    # Get size of the text
                    if 1 <= number <= 9:
                        text = str(number)
                    else:  # 10-16 to A-G
                        text = chr(65 + number - 10)
                        
                    (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
                    
                    # Calculate font scale based on cell size and text size
                    font_scale = min(cell_width / text_width, cell_height / text_height) * 0.8
                    
                    # Update text size with new font scale
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)
                    
                    position = ((x * cell_width + (cell_width - text_width) // 2), 
                                (y * cell_height + (cell_height + text_height) // 2))
                    
                    cv2.putText(original_img, text, position, font, font_scale, color, 2, cv2.LINE_AA)
        

        return original_img

        
    def main(self, image_path):
        extractor = Sudoku_Get(image_path)
        extractor.Grayscale_get()
        extractor.Filter_apply()
        extractor.filtered_binarize()
        extractor.find_contours()
        warped_image = extractor.perspective_transform()
        
        # 也变换彩色图像
        color_warped_image = extractor.perspective_transform_color(extractor.image)
        
        return warped_image, color_warped_image
    
#获取单元格

class Cell:
    def __init__(self):
        pass
    
    def preprocess_cell(cell_img):
        # 对于16x16，可能需要考虑保留更多的图像信息，因此可能需要重新调整大小。
        resized = cv2.resize(cell_img, (32, 32))  # 修改为32x32，但这取决于您的模型输入大小。
        return resized.flatten()

    def extract_cells(sudoku_img):
        stepX = sudoku_img.shape[1] // 16
        stepY = sudoku_img.shape[0] // 16
        cells = []

        for y in range(0, 16):
            for x in range(0, 16):
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                cells.append(sudoku_img[startY:endY, startX:endX])
    

        return cells

    def predict_cells(cells, model):
        predictions = []
        for index, cell in enumerate(cells):
            marginX = int(cell.shape[1] * 0.15)
            marginY = int(cell.shape[0] * 0.1)
            trimmed_cell = cell[marginY:-marginY, marginX:-marginX]

            # code used for testing
            # if index in [0, 18, 53, 73]:
            #     cv2.imshow(f"Trimmed Cell {index}", trimmed_cell)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            
            white_area = (trimmed_cell == 255).sum()
            total_area = trimmed_cell.shape[0] * trimmed_cell.shape[1]
            
            if white_area < 0.05 * total_area:
                predictions.append(0)
            else:
                cell = cv2.resize(trimmed_cell, (28, 28))  # 修改为32x32，但这取决于您的模型输入大小。
                cell = img_to_array(cell)
                cell = np.expand_dims(cell, axis=0)
                cell = cell / 255.0  
                prediction = model.predict(cell)
                predictions.append(np.argmax(prediction)) 
        
        return predictions
    
    def assemble_sudoku_matrix(predictions):
        print(np.array(predictions).reshape(16, 16))
        return np.array(predictions).reshape(16, 16)

      
        
   