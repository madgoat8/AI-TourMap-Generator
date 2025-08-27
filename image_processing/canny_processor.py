"""
Canny边缘检测处理器
"""
import cv2
import numpy as np
from PIL import Image
import logging
from config import ModelConfig

logger = logging.getLogger(__name__)

class CannyProcessor:
    """Canny边缘检测处理器"""
    
    def __init__(self, low_threshold: int = ModelConfig.CANNY_LOW_THRESHOLD, 
                 high_threshold: int = ModelConfig.CANNY_HIGH_THRESHOLD):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def process_image(self, image: Image.Image, save_path: str = None) -> Image.Image:
        """
        对图像进行Canny边缘检测
        
        Args:
            image: 输入的PIL图像
            save_path: 可选的保存路径
            
        Returns:
            处理后的边缘图像
        """
        logger.info("开始Canny边缘检测处理...")
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 转换为灰度图
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array
        
        # 应用Canny边缘检测
        canny_edges = cv2.Canny(gray_image, self.low_threshold, self.high_threshold)
        
        # 转换回PIL图像（RGB格式）
        canny_image = Image.fromarray(canny_edges).convert("RGB")
        
        # 保存图像（如果指定了路径）
        if save_path:
            canny_image.save(save_path)
            logger.info(f"Canny边缘图保存到 {save_path}")
        
        logger.info("Canny边缘检测处理完成")
        return canny_image
    
    def process_semantic_map(self, semantic_image: Image.Image, save_path: str = None) -> Image.Image:
        """
        专门处理语义地图的Canny边缘检测
        
        Args:
            semantic_image: 语义地图图像
            save_path: 可选的保存路径
            
        Returns:
            边缘检测结果图像
        """
        return self.process_image(semantic_image, save_path)