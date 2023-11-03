import cv2
import numpy as np

#这个类只需要处理颜色识别，因此图像预处理操作简化
class RainbowCell:
    @staticmethod
    def extract_cells_rainbow(image_path):

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("图片未找到，请检查路径")

        # 进行边框检测和透视变换
        processed_image = RainbowCell.preprocess_image(image)

        # 提取每个单元格并识别颜色
        color_cells = RainbowCell.extract_cells(processed_image)
        return color_cells

    @staticmethod
    def detect_cells_color(color_cells):
        color_matrix = []
        for row in color_cells:
            color_row = []
            for cell in row:
                color = RainbowCell.detect_cell_color(cell)
                color_row.append(color)
            color_matrix.append(color_row)
        return color_matrix

    @staticmethod
    def preprocess_image(image):
        # 使用高斯模糊，帮助减少图像噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 使用Canny边缘检测
        edged = cv2.Canny(hsv, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按轮廓面积降序排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 初始化数独网格的轮廓
        sudoku_contour = None
        
        # 遍历轮廓
        for contour in contours:
            # 逼近轮廓以获得具有较少顶点的多边形
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果多边形有4个顶点，则可能是我们的数独网格
            if len(approx) == 4:
                sudoku_contour = approx
                break
        
        if sudoku_contour is None:
            raise Exception("未找到数独网格")
        
        # 进行透视变换，获取正视图
        processed_image = RainbowCell.four_point_transform(image, sudoku_contour.reshape(4, 2))
        
        return processed_image

    @staticmethod
    def extract_cells(image):
        # 获取图像的尺寸
        height, width = image.shape[:2]
        
        # 计算每个单元格的尺寸
        cell_size = height // 9
        
        # 初始化颜色单元格列表
        color_cells = []
        
        # 设置裁剪的边缘大小，因为边缘的格子因为边框太厚容易识别错误颜色
        crop_size = 0
        
        # 遍历每个单元格
        for i in range(9):
            row = []
            for j in range(9):
                # 计算单元格的边界
                x = j * cell_size
                y = i * cell_size
                cell = image[y + crop_size:y + cell_size - crop_size, x + crop_size:x + cell_size - crop_size]
                
                # 将单元格添加到列表中
                row.append(cell)
            color_cells.append(row)
        return color_cells
    
    #检测颜色用的K-Means聚类算法，K值=支持检测的颜色数
    @staticmethod
    def detect_cell_color(cell):
        # 将图片数据转换为浮点数
        cell = np.float32(cell)

        # 定义停止条件、聚类数目和应用K均值聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 7
        _, labels, centers = cv2.kmeans(cell.reshape((-1, 3)), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 将数据转换回8位整数
        centers = np.uint8(centers)

        # 计算每个聚类中心的像素数目
        counts = np.bincount(labels.flatten(), minlength=k)

        # 找到最大的聚类中心
        dominant_color_index = np.argmax(counts)
        dominant_color = centers[dominant_color_index]

        # 将BGR颜色转换为HSV颜色
        dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # 判断是否为白色或接近白色
        if dominant_color_hsv[2] > 200 and dominant_color_hsv[1] < 50:
            return 'white'

        # 根据HSV颜色的H值判断颜色
        h, s, v= dominant_color_hsv[0], dominant_color_hsv[1], dominant_color_hsv[2]
        if 0 <= h < 10 or 160 <= h <= 180:
            return 'red'
        elif 10 <= h < 30:
            return 'orange'
        elif 30 <= h < 40:
            return 'yellow'
        elif 40 <= h < 90:
            return 'green'
        elif 90 <= h < 150:
            return 'blue'
        elif 150 <= h < 160:
            return 'purple'
        else:
            return 'unknown'


    @staticmethod
    def order_points(pts):
        # 初始化一个坐标点的列表，表示矩形框
        rect = np.zeros((4, 2), dtype="float32")

        # 按顺序找到对应坐标点：左上，右上，右下，左下
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @staticmethod
    def four_point_transform(image, pts):
        # 获取输入坐标点
        rect = RainbowCell.order_points(pts)
        (tl, tr, br, bl) = rect

        # 计算输入的w和h值
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # 设置目标点的坐标，依次为左上角，右上角，右下角，左下角
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        # 计算透视变换矩阵并进行透视变换
        M = cv2.getPerspectiveTransform(rect, dst)
        transformed_image = cv2.warpPerspective(image, M, (max_width, max_height))

        return transformed_image

