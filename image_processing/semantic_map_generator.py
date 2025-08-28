"""
语义地图生成器
"""
import logging
from PIL import Image, ImageDraw
from typing import Dict, Any
from config import IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_MAP, ModelConfig
from utils import scale_coordinates

logger = logging.getLogger(__name__)

class SemanticMapGenerator:
    """语义地图生成器"""
    
    def __init__(self, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT, color_map: Dict[str, str] = COLOR_MAP):
        self.width = width
        self.height = height
        self.color_map = color_map
    
    def create_semantic_map(self, osm_data: Dict[str, Any], bbox_details: Dict[str, float], save_path: str) -> Image.Image:
        """
        从OSM数据创建语义地图
        
        Args:
            osm_data: OSM数据字典
            bbox_details: 边界框详情
            save_path: 保存路径
            
        Returns:
            生成的语义地图图像
            
        Raises:
            ValueError: 当区域太小或数据无效时
        """
        # 创建黑色背景图像
        img = Image.new("RGB", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)
        
        # 解析OSM数据
        nodes = {node['id']: node for node in osm_data['elements'] if node['type'] == 'node'}
        ways = [elem for elem in osm_data['elements'] if elem['type'] == 'way']
        
        # 检查边界框有效性
        lat_range = bbox_details["max_lat"] - bbox_details["min_lat"]
        lng_range = bbox_details["max_lng"] - bbox_details["min_lng"]
        if lat_range == 0 or lng_range == 0:
            raise ValueError("边界框范围无效，区域可能太小")
        
        # 绘制每个way
        for way in ways:
            self._draw_way(draw, way, nodes, bbox_details)
        
        # 保存图像
        img.save(save_path)
        logger.info(f"语义地图保存到 {save_path}")
        
        # 检查图像是否为空
        if img.getbbox() is None:
            raise ValueError("生成的语义地图为空白")
        
        return img
    
    def _draw_way(self, draw: ImageDraw.Draw, way: Dict[str, Any], nodes: Dict[int, Dict], bbox_details: Dict[str, float]):
        """
        绘制单个way
        
        Args:
            draw: PIL绘图对象
            way: way数据
            nodes: 节点数据字典
            bbox_details: 边界框详情
        """
        # 获取way的所有点
        points = [nodes.get(node_id) for node_id in way['nodes']]
        points = [p for p in points if p]  # 过滤掉None值
        
        if len(points) < 2:
            return  # 点数不足，跳过
        
        # 转换为像素坐标
        pixel_coords = [
            scale_coordinates(p['lon'], p['lat'], bbox_details, self.width, self.height) 
            for p in points
        ]
        
        # 确定颜色
        color = self._get_way_color(way)
        line_width = ModelConfig.LINE_WIDTH
        
        # 绘制
        if way['nodes'][0] == way['nodes'][-1] and len(points) > 2:
            # 闭合多边形
            draw.polygon(pixel_coords, fill=color)
            draw.line(pixel_coords + [pixel_coords[0]], fill=color, width=line_width)
        else:
            # 开放线条
            draw.line(pixel_coords, fill=color, width=line_width)
    
    def _get_way_color(self, way: Dict[str, Any]) -> str:
        """
        根据way的标签确定颜色 - 简化版本，只处理道路和水面
        
        Args:
            way: way数据
            
        Returns:
            颜色十六进制字符串
        """
        tags = way.get('tags', {})
        
        # 优先检查道路
        if 'highway' in tags:
            return self.color_map.get("highway", self.color_map["default"])
        
        # 检查水面
        if 'natural' in tags and tags['natural'] == 'water':
            return self.color_map.get("natural=water", self.color_map["default"])
        
        # 检查水道
        if 'waterway' in tags:
            return self.color_map.get("waterway", self.color_map["default"])
        
        # 默认背景色（对于不匹配的元素）
        return self.color_map["default"]