import cv2
import numpy as np
import json
import os


class AffineTransformer:
    """
    仿射变换器类，用于处理图像的对齐和融合
    主要功能：
    1. 加载仿射变换矩阵
    2. 对图像进行仿射变换
    3. 图像融合
    """
    def __init__(self, config_path):
        """
        初始化仿射变换器
        Args:
            config_path: JSON配置文件路径，包含仿射变换矩阵
        """
        self.config_path = config_path
        self.affine_matrix = None
        self.load_config()

    def load_config(self):
        """加载配置文件中的仿射变换矩阵"""
        try:
            with open(self.config_path, 'r') as file:
                m_data = json.load(file)
                if m_data is not None and "M" in m_data:
                    # 获取并处理仿射变换矩阵
                    affine_matrix_3x3 = np.array(m_data["M"]).reshape(3, 3)
                    self.affine_matrix = affine_matrix_3x3[:2, :]
                    
                    # 提取变换参数
                    tx, ty = self.affine_matrix[0:2, 2]  # 平移参数
                    sx = np.sqrt(self.affine_matrix[0,0]**2 + self.affine_matrix[0,1]**2)  # x方向缩放
                    sy = np.sqrt(self.affine_matrix[1,0]**2 + self.affine_matrix[1,1]**2)  # y方向缩放
                    theta = np.degrees(np.arctan2(self.affine_matrix[1,0], self.affine_matrix[0,0]))  # 旋转角度
                    
                    # 计算剪切参数
                    shear_x = np.arctan2(self.affine_matrix[0,1], self.affine_matrix[0,0])  # x方向剪切
                    shear_y = np.arctan2(self.affine_matrix[1,0], self.affine_matrix[1,1])  # y方向剪切
                    
                    # 重新构建变换矩阵
                    scale_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
                    rotation_matrix = np.array([
                        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
                        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
                        [0, 0, 1]
                    ])
                    shear_matrix = np.array([
                        [1, np.tan(shear_x), 0],
                        [np.tan(shear_y), 1, 0],
                        [0, 0, 1]
                    ])
                    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
                    
                    # 组合变换矩阵：先缩放，再旋转，再剪切，最后平移
                    self.affine_matrix = (translation_matrix @ shear_matrix @ rotation_matrix @ scale_matrix)[:2, :]
                else:
                    raise ValueError("Invalid configuration file format")
        except Exception as e:
            raise Exception(f"Failed to load configuration: {str(e)}")

    def adjust_matrix_for_scale(self, scale_factor):
        """
        根据缩放比例调整仿射变换矩阵
        Args:
            scale_factor: 缩放比例
        Returns:
            调整后的仿射变换矩阵
        """
        if self.affine_matrix is None:
            raise ValueError("Affine matrix not loaded")
            
        # 从原始矩阵中提取参数
        # 缩放、旋转和剪切部分
        a = self.affine_matrix[0, 0]
        b = self.affine_matrix[0, 1]
        c = self.affine_matrix[1, 0]
        d = self.affine_matrix[1, 1]
        # 平移部分
        tx = self.affine_matrix[0, 2]
        ty = self.affine_matrix[1, 2]
        
        # 创建调整后的矩阵，保持剪切效果
        adjusted_matrix = np.array([
            [a, b, tx * scale_factor],  # 平移参数需要乘以缩放比例
            [c, d, ty * scale_factor]   # 平移参数需要乘以缩放比例
        ], dtype=np.float32)
        
        return adjusted_matrix

    def transform_image(self, image, hd_image):
        """
        对图像进行仿射变换，以高清图像的左上角为坐标系原点
        Args:
            image: 输入图像（原始尺寸）
            hd_image: 高清图像（原始尺寸）
        Returns:
            变换后的图像
        Raises:
            ValueError: 当仿射矩阵未加载或输入图像无效时
        """
        if self.affine_matrix is None:
            raise ValueError("Affine matrix not loaded")
            
        if image is None or hd_image is None:
            raise ValueError("Input images cannot be None")
            
        if image.size == 0 or hd_image.size == 0:
            raise ValueError("Input images cannot be empty")

        try:
            # 计算变换后图像的边界框
            h, w = image.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            transformed_corners = cv2.transform(corners.reshape(-1, 1, 2), self.affine_matrix.reshape(2, 3)).reshape(-1, 2)
            
            # 计算变换后图像的最小外接矩形
            min_x, min_y = np.min(transformed_corners, axis=0)
            max_x, max_y = np.max(transformed_corners, axis=0)
            
            # 从原始仿射矩阵中提取平移参数
            tx, ty = self.affine_matrix[0:2, 2]
            
            # 计算新的尺寸，确保能容纳变换后的图像（考虑平移参数）
            new_w = int(np.ceil(max_x - min_x + tx))
            new_h = int(np.ceil(max_y - min_y + ty))
            
            if new_w <= 0 or new_h <= 0:
                raise ValueError(f"Invalid transformed image size: {new_w}x{new_h}")
            
            # 计算偏移量，使变换后的图像相对于高清图像左上角定位
            offset_x, offset_y = tx - min_x, ty - min_y
            
            # 创建平移矩阵并组合变换
            translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
            final_matrix = np.vstack([translation_matrix @ np.vstack([self.affine_matrix, [0, 0, 1]]), [0, 0, 1]])[:2, :]

            # 执行仿射变换
            transformed_image = cv2.warpAffine(image, final_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
            
            if transformed_image is None:
                raise ValueError("Failed to perform affine transformation")
            
            # 获取高清图像和变换后图像的尺寸
            hd_height, hd_width = hd_image.shape[:2]
            if_height, if_width = transformed_image.shape[:2]
            
            # 创建新画布并放置变换后的图像
            new_height, new_width = max(hd_height, if_height), max(hd_width, if_width)
            new_transformed_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            new_transformed_image[:if_height, :if_width] = transformed_image
            
            # 按高清图像尺寸裁剪
            cropped_image = new_transformed_image[:hd_height, :hd_width]    
            return cropped_image
            
        except Exception as e:
            raise Exception(f"Error during image transformation: {str(e)}")

    def merge_images(self, if_image, hd_image, roi=None, alpha=0.5):
        """
        融合高清图像和红外图像
        Args:
            if_image: 红外图像（原始尺寸）
            hd_image: 高清图像（原始尺寸）
            roi: 感兴趣区域 [x1, y1, x2, y2]，如果为None则处理全图
            alpha: 融合权重
        Returns:
            融合后的图像
        Raises:
            ValueError: 当输入图像无效或ROI区域无效时
        """
        if if_image is None or hd_image is None:
            raise ValueError("Input images cannot be None")
            
        if if_image.size == 0 or hd_image.size == 0:
            raise ValueError("Input images cannot be empty")
            
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        try:
            # 对红外图像进行仿射变换
            if_restored_image = self.transform_image(if_image, hd_image)
            
            # 获取高清图像尺寸
            hd_height, hd_width = hd_image.shape[:2]
            
            # ROI处理：未指定则使用全图，否则检查并限制在图像范围内
            if roi is None:
                roi = [0, 0, hd_width, hd_height]
            else:
                if len(roi) != 4:
                    raise ValueError(f"ROI must be a list of 4 values [x1, y1, x2, y2], got {roi}")
                    
                x1, y1, x2, y2 = roi
                # 检查并限制ROI边界
                if x1 < 0 or y1 < 0 or x2 > hd_width or y2 > hd_height:
                    print(f"警告：ROI区域 [{x1}, {y1}, {x2}, {y2}] 超出图像边界 [{0}, {0}, {hd_width}, {hd_height}]")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(hd_width, x2), min(hd_height, y2)
                    roi = [x1, y1, x2, y2]
                    print(f"已调整ROI区域为: {roi}")
                
                # 检查ROI有效性
                if x1 >= x2 or y1 >= y2:
                    raise ValueError(f"无效的ROI区域: [{x1}, {y1}, {x2}, {y2}]，左上角坐标必须小于右下角坐标")
            
            # 提取并融合ROI区域
            x1, y1, x2, y2 = roi
            hd_roi = hd_image[y1:y2, x1:x2]
            if_roi = if_restored_image[y1:y2, x1:x2]
            
            if hd_roi.size == 0 or if_roi.size == 0:
                raise ValueError("ROI region is empty after extraction")
            
            # 检查ROI区域的有效性
            if not np.all(np.isfinite(hd_roi)) or not np.all(np.isfinite(if_roi)):
                raise ValueError("ROI region contains invalid values")
            
            # 在ROI区域内进行图像融合
            blended_roi = cv2.addWeighted(hd_roi, 1-alpha, if_roi, alpha, 0)
            
            # 将融合结果替换到原图中
            result_image = hd_image.copy()
            result_image[y1:y2, x1:x2] = blended_roi
            
            return result_image
            
        except Exception as e:
            raise Exception(f"Error during image merging: {str(e)}")

    @staticmethod
    def load_image(image_path):
        """
        加载图像文件
        Args:
            image_path: 图像文件路径
        Returns:
            加载的图像
        Raises:
            FileNotFoundError: 当图像文件不存在时
            ValueError: 当图像加载失败时
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            # 验证图像格式
            if len(image.shape) != 3:
                raise ValueError(f"Image must be color image (3 channels): {image_path}")
            if image.shape[2] != 3:
                raise ValueError(f"Image must be in BGR format: {image_path}")
                
            # 验证图像尺寸
            height, width = image.shape[:2]
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")
            if width > 10000 or height > 10000:  # 最大尺寸限制
                raise ValueError(f"Image dimensions too large: {width}x{height}")
                
            # 验证图像数据
            if not np.all(np.isfinite(image)):
                raise ValueError(f"Image contains invalid values: {image_path}")
                
            return image
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 感兴趣区域|融合权重|仿射变换矩阵文件路径
    roi = [250, 54, 1761, 1077]         # 若不设置则为高清图像全区域
    # roi = None
    alpha = 0.5
    json_file = "./results/20250603_200201_affine_matrix.json"
    hd_image_path = './data/hd_right_f1_00016.jpg'
    if_image_path = './data/if_right_f1_00007.jpg'  

    # 仿射变换矩阵的加载|图像的加载|图像的融合
    transformer = AffineTransformer(json_file)          
    hd_image = transformer.load_image(hd_image_path)
    if_image = transformer.load_image(if_image_path)
    merge_result_image = transformer.merge_images(if_image, hd_image, roi, 0.5)

    # 显示结果
    cv2.imshow("Result Image", merge_result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



