"""
坐标处理工具函数
"""
from typing import List
from pydantic import BaseModel

class GeoCoordinate(BaseModel):
    lat: float
    lng: float

def get_bounding_box_details(coordinates: List[GeoCoordinate]) -> dict:
    """
    计算坐标列表的边界框详情
    
    Args:
        coordinates: 坐标点列表
        
    Returns:
        包含边界框信息的字典
    """
    min_lat = min(c.lat for c in coordinates)
    max_lat = max(c.lat for c in coordinates)
    min_lng = min(c.lng for c in coordinates)
    max_lng = max(c.lng for c in coordinates)
    bbox_str = f"{min_lat},{min_lng},{max_lat},{max_lng}"
    
    return {
        "min_lat": min_lat, 
        "max_lat": max_lat, 
        "min_lng": min_lng, 
        "max_lng": max_lng, 
        "bbox_str": bbox_str
    }

def scale_coordinates(lon: float, lat: float, bbox_details: dict, image_width: int, image_height: int) -> tuple:
    """
    将地理坐标转换为图像像素坐标
    
    Args:
        lon: 经度
        lat: 纬度
        bbox_details: 边界框信息
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        (x, y) 像素坐标
    """
    lat_range = bbox_details["max_lat"] - bbox_details["min_lat"]
    lng_range = bbox_details["max_lng"] - bbox_details["min_lng"]
    
    x = ((lon - bbox_details["min_lng"]) / lng_range) * image_width
    y = ((bbox_details["max_lat"] - lat) / lat_range) * image_height
    
    return (x, y)