import cv2
import numpy as np

# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 预处理部分
class Sudoku_Get:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    def determine_image_type(self):  # 判断数独图片是实际拍摄还是电子版的
        # 计算对比度
        contrast = np.std(self.gray)
        # 设置阈值 (此阈值可能需要进行调整)
        threshold = 60  
        if contrast > threshold:
            print("elec")
            return "electronic"
        else:
            print("physical")
            return "physical"
        
    def Grayscale_get(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
    def Filter_apply(self):# 滤波器选择
        img_type = self.determine_image_type()
        
        if img_type == "electronic":
            self.filtered = self.gray
        else:
            self.filtered1 = cv2.GaussianBlur(self.gray, (5,5), 0)
            self.filtered = cv2.bilateralFilter(self.filtered1, -1, 11, 5)

        return self.filtered

    def filtered_binarize(self): #二值化
        img_type = self.determine_image_type()
        
        if img_type == "electronic":
            _, self.binary_1 = cv2.threshold(self.filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            self.binary_1 = cv2.adaptiveThreshold(self.filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY, 25, 4)
        self.kernel_o = np.ones((11,11), np.uint8)
        self.opening = cv2.morphologyEx(self.binary_1, cv2.MORPH_OPEN, self.kernel_o)
        # closing
        self.kernel_c = np.ones((9,9), np.uint8)
        self.closing = cv2.morphologyEx(self.opening, cv2.MORPH_CLOSE, self.kernel_c)
        # 根据底色调整是否反相，如果是白底黑字则执行以下这句代码
        self.binary = np.subtract(255, self.binary_1)
   
        return self.binary

    def find_contours(self):# 找轮廓
        contours, _ = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        self.polygon = None

        img_type = self.determine_image_type()
        if img_type == "electronic":
            # 对于电子版图片，可能只需要考虑面积最大的轮廓
            self.polygon = max(contours, key=cv2.contourArea)
        else:
            for contour in contours:
            # 逼近轮廓以获得具有较少顶点的多边形
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果多边形有4个顶点，则可能是我们的数独网格
                if len(approx) == 4:
                    self.polygon = approx
                    break
        
        if self.polygon is None:
            raise Exception("未找到数独网格")
        
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


   
    def overlay_solution(self, original_sudoku, solution, original_img):
        cell_width = original_img.shape[1] // 9
        cell_height = original_img.shape[0] // 9

        for y in range(9):
            for x in range(9):
                if original_sudoku[y][x] == 0 and solution[y][x] != 0:
                    number = solution[y, x]
                    # 获取文本的大小
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # 计算参考尺寸
                    (ref_width, ref_height), _ = cv2.getTextSize("5", font, 1, 2)
                    # 根据单元格大小计算字体比例
                    font_scale = min(cell_width / ref_width, cell_height / ref_height) * 0.8  # 这里乘以0.8是为了留出一些边距
                    # 计算文本的尺寸
                    (width, height), _ = cv2.getTextSize(str(number), font, font_scale, 2)
                    # 计算数字应该放置的位置
                    center_x = x * cell_width + cell_width // 2
                    center_y = y * cell_height + cell_height // 2
                    # 计算文本的左下角位置
                    position = (int(center_x - width / 2), int(center_y + height / 2))
                    color = (0, 255, 0)  # 绿色
                    cv2.putText(original_img, str(number), position, font, font_scale, color, 2, cv2.LINE_AA)

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
        resized = cv2.resize(cell_img, (28, 28))
        return resized.flatten()

    def extract_cells(sudoku_img):
        stepX = sudoku_img.shape[1] // 9
        stepY = sudoku_img.shape[0] // 9
        cells = []
        for y in range(0, 9):
            for x in range(0, 9):
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
            white_area = (trimmed_cell == 255).sum()
            total_area = trimmed_cell.shape[0] * trimmed_cell.shape[1]
            if white_area < 0.05 * total_area:
                # print(white_area / total_area)
                predictions.append(0)
            else:
                # 转换图像为模型接受的形状并归一化
                cell = cv2.resize(trimmed_cell, (28, 28)) # 调整到模型期望的大小
                cell = img_to_array(cell)
                cell = np.expand_dims(cell, axis=0)
                cell = cell / 255.0  # 假设模型期望的输入范围是[0,1]
                # 进行预测
                prediction = model.predict(cell)
                predictions.append(np.argmax(prediction))  # 将one-hot编码转换为单一标签
        return predictions
    
    def assemble_sudoku_matrix(predictions):
        print(np.array(predictions).reshape(9, 9))
        return np.array(predictions).reshape(9, 9)        